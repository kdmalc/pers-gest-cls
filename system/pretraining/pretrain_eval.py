"""
pretrain_eval.py
================
Evaluation suite for pretrained EMG models, following Meta/Bengio et al. Fig. 4 methodology.
 
Three evaluation modes:
  1. LinearProbe (in-distribution)   — probe on a train user
  2. LinearProbe (out-of-distribution) — probe on a val/test user (the one that matters)
  3. Representation visualization  — PCA/tSNE per-layer, colored by gesture and by user_id
 
The linear probe is the most informative metric for MAML pretraining quality:
  - If the frozen backbone gives >chance accuracy on a NEW user with a linear classifier,
    it means the features are user-invariant and gesture-discriminative — exactly what
    MAML needs as an initialization.
  - Rule of thumb for 10-way classification: random = 10%. Good pretrain ≥ 50% on NEW users.

MoE routing analysis (when config["use_moe"] = True):
  - run_full_eval() will call run_moe_routing_eval() on both the train and val dataloaders
    after the standard probe/viz pipeline is complete.
  - The analysis uses RoutingCollector + RoutingAnalyzer (same as pretrain_trainer.py and
    mamlpp.py) to check for expert collapse, load imbalance, and gesture/user routing
    specialisation at evaluation time — on the best checkpoint, not mid-training.
  - Saves plots to {save_dir}/moe_routing/{split}/ and returns reports in the results dict.
  - viz_from_checkpoint() also runs routing analysis when use_moe=True, so you can inspect
    any saved checkpoint without re-running the full eval pipeline.

Usage:
    from pretrain_eval import run_full_eval
    results = run_full_eval(model, tensor_dict, config)
"""
 
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MOE.MOE_analysis import RoutingCollector, RoutingAnalyzer, save_routing_record


def ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
    """Helper from eval_knn_proto.py to ensure (N, C, T) shape."""
    if x is None or x.dim() != 3:
        return x
    # If the last dimension matches known channel counts, swap it
    if x.shape[-1] in [16, 72]:
        # Using transpose(1, 2) is equivalent to permute(0, 2, 1) here
        return x.transpose(1, 2).contiguous()
    return x
 
# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def extract_features(model, tensor_dict, pids, target_reps, device, use_imu=False):
    """
    Extract backbone features for all (pid, gesture_class, trial) combinations,
    filtering to only the repetition indices listed in target_reps.
 
    Args:
        target_reps: list of 1-based repetition indices to include (e.g. [1,2,3])
 
    Returns:
        feats_final:  np.ndarray (N, feat_dim) — final layer features
        feats_layers: list of np.ndarray (N, feat_dim) — one per LSTM/block layer
        labels:       np.ndarray (N,) — gesture class index
        user_ids:     np.ndarray (N,) — pid index (int)
        pid_list:     list of unique pids (maps int index → pid)
    """
    model.eval()
    model.to(device)
 
    # All PIDs share the same gesture classes — one lookup is sufficient.
    first_valid_pid = next((p for p in pids if p in tensor_dict), None)
    if first_valid_pid is None:
        raise ValueError("No valid PIDs found in tensor_dict.")
    sorted_gesture_classes = sorted(tensor_dict[first_valid_pid].keys())
    label_map = {g: i for i, g in enumerate(sorted_gesture_classes)}
 
    all_final, all_layers, all_labels, all_users = [], [], [], []
    pid_list   = sorted(pids)
    pid_to_idx = {p: i for i, p in enumerate(pid_list)}
 
    for pid in pid_list:
        if pid not in tensor_dict:
            continue
        for gesture_class in sorted_gesture_classes:
            if gesture_class not in tensor_dict[pid]:
                continue
            slot    = tensor_dict[pid][gesture_class]
            emg_all = slot['emg']   # (n_trials, T, C) or (n_trials, C, T)
            imu_all = slot.get('imu', None)
 
            emg_data = ensure_channel_first(emg_all)
 
            # Convert 1-based rep numbers to 0-based indices, clamp to available trials.
            valid_idxs = [rep - 1 for rep in target_reps if 0 <= rep - 1 < emg_data.shape[0]]
            if not valid_idxs:
                continue
 
            emg_data = emg_data[valid_idxs].float().to(device)
            imu_input = None
            if use_imu and imu_all is not None:
                imu_data  = ensure_channel_first(imu_all)
                imu_input = imu_data[valid_idxs].float().to(device)
 
            feat_final, layer_feats = model.backbone(emg_data, imu_input)
 
            all_final.append(feat_final.cpu().numpy())
            if not all_layers:
                all_layers = [[] for _ in layer_feats]
            for li, lf in enumerate(layer_feats):
                all_layers[li].append(lf.cpu().numpy())
 
            n_trials = emg_data.shape[0]
            all_labels.extend([label_map[gesture_class]] * n_trials)
            all_users.extend([pid_to_idx[pid]] * n_trials)
 
    if not all_final:
        raise ValueError(
            f"extract_features: no data collected for pids={pids}, target_reps={target_reps}. "
            "Check that rep indices are 1-based and exist in tensor_dict."
        )
 
    feats_final  = np.concatenate(all_final,  axis=0)
    feats_layers = [np.concatenate(l, axis=0) for l in all_layers]
    labels       = np.array(all_labels)
    user_ids     = np.array(all_users)
 
    return feats_final, feats_layers, labels, user_ids, pid_list
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────
 
def linear_probe(
    model,
    tensor_dict: dict,
    probe_pids: list,
    all_pids_train: list,
    target_reps: list,          # renamed from 'gestures' — these are rep indices, not class names
    device,
    use_imu: bool = False,
    n_way: int = 10,
    label: str = "probe",
):
    """
    Fit a logistic regression on train users' features, evaluate on probe_pids.
    For "in-distribution" probe:  probe_pids ⊂ all_pids_train.
    For "out-of-distribution" probe: probe_pids ∩ all_pids_train = ∅.
 
    Args:
        target_reps: 1-based repetition indices to extract features from (e.g. [1,2,3,4,5])
 
    Returns dict with accuracy and per-class breakdown.
    """
    print(f"[LinearProbe:{label}] Extracting train features ({len(all_pids_train)} users)...")
    feats_tr, _, labels_tr, _, _ = extract_features(
        model, tensor_dict, all_pids_train, target_reps, device, use_imu
    )
 
    print(f"[LinearProbe:{label}] Extracting probe features ({len(probe_pids)} users)...")
    feats_pr, _, labels_pr, _, _ = extract_features(
        model, tensor_dict, probe_pids, target_reps, device, use_imu
    )
 
    probe_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
        )),
    ])
    probe_clf.fit(feats_tr, labels_tr)
 
    acc    = probe_clf.score(feats_pr, labels_pr)
    chance = 1.0 / n_way
 
    print(f"[LinearProbe:{label}] Accuracy = {acc*100:.1f}% (chance = {chance*100:.1f}%)")
 
    return {
        'label':    label,
        'accuracy': acc,
        'chance':   chance,
        'n_train':  len(labels_tr),
        'n_probe':  len(labels_pr),
        'clf':      probe_clf,
    }
 
# ─────────────────────────────────────────────────────────────────────────────
# PCA / tSNE visualization (Fig. 4f-h style)
# ─────────────────────────────────────────────────────────────────────────────
 
def visualize_representations(
    model,
    tensor_dict: dict,
    pids: list,
    gestures: list,
    device,
    use_imu: bool = False,
    method: str = 'pca',       # 'pca' or 'tsne'
    max_samples: int = 500,    # subsample for speed (500 matches Meta paper)
    gesture_names: dict = None,
    save_path: str = None,
    model_name: str = "Model",
    split_label: str = None,   # 'train' or 'val' — prefixes filename and appears in title
):
    """
    Replicates Meta paper Fig. 4f-h:
      Each row = one LSTM/Transformer layer
      Column 1 = colored by gesture category
      Column 2 = colored by user identity
 
    Unlike the Meta paper we don't have band placement or RMS columns (different data),
    so we do 2 columns: gesture and user.
 
    Args:
        method:  'pca' (fast, deterministic) or 'tsne' (slower, often nicer)
        max_samples: subsample total for speed; set to None for all
    """
    # Inject split_label prefix into save_path if provided.
    # e.g. "outputs/TST_representation_viz.png"
    #   -> "outputs/trainusers_TST_representation_viz.png"
    if save_path is not None and split_label is not None:
        import os as _os
        _parent = _os.path.dirname(save_path)
        _fname  = _os.path.basename(save_path)
        save_path = _os.path.join(_parent, f"{split_label}users_{_fname}") if _parent else f"{split_label}users_{_fname}"
 
    split_tag = f" [{split_label} users]" if split_label else ""
    print(f"\n[Visualization{split_tag}] Extracting features from {len(pids)} users...")
    feats_final, feats_layers, labels, user_ids, pid_list = extract_features(
        model, tensor_dict, pids, gestures, device, use_imu
    )
 
    n_layers = len(feats_layers)
    all_layer_feats = feats_layers   # list of (N, dim)
 
    # Subsample for speed
    N = labels.shape[0]
    if max_samples is not None and N > max_samples:
        rng  = np.random.default_rng(42)
        idxs = rng.choice(N, size=max_samples, replace=False)
        all_layer_feats = [f[idxs] for f in all_layer_feats]
        labels          = labels[idxs]
        user_ids        = user_ids[idxs]
 
    n_gestures = len(set(labels.tolist()))
    n_users    = len(set(user_ids.tolist()))
 
    # Colormaps
    gesture_cmap = cm.get_cmap('tab10', n_gestures)
    user_cmap    = cm.get_cmap('tab20', max(n_users, 2))
 
    gesture_colors = [gesture_cmap(int(l)) for l in labels]
    user_colors    = [user_cmap(int(u)) for u in user_ids]
 
    split_title = f" — {split_label.capitalize()} Users" if split_label else ""
    fig, axes = plt.subplots(
        n_layers, 2,
        figsize=(10, 4 * n_layers),
        squeeze=False,
    )
    fig.suptitle(f"{model_name}{split_title} — {method.upper()} Layer Representations\n"
                 f"(N={len(labels)}, {n_layers} layers)", fontsize=13, y=1.01)
 
    for layer_idx, layer_feats in enumerate(all_layer_feats):
        layer_name = f"Layer {layer_idx + 1}"
 
        # Reduce to 2D
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(layer_feats) // 4))
            coords  = reducer.fit_transform(layer_feats)
        else:
            pca    = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(StandardScaler().fit_transform(layer_feats))
 
        # Column 0: by gesture
        ax0 = axes[layer_idx][0]
        ax0.scatter(coords[:, 0], coords[:, 1], c=gesture_colors, s=18, alpha=0.7, linewidths=0)
        ax0.set_title(f"{layer_name} — By Gesture", fontsize=10)
        ax0.set_xticks([]); ax0.set_yticks([])
        # Legend for gestures
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=gesture_cmap(i),
                       markersize=7, label=gesture_names.get(i, f"G{i}") if gesture_names else f"Gesture {i}")
            for i in range(n_gestures)
        ]
        ax0.legend(handles=legend_handles, fontsize=6, loc='lower right', ncol=2)
 
        # Column 1: by user
        ax1 = axes[layer_idx][1]
        ax1.scatter(coords[:, 0], coords[:, 1], c=user_colors, s=18, alpha=0.7, linewidths=0)
        ax1.set_title(f"{layer_name} — By User ID", fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])
        legend_handles_u = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=user_cmap(i),
                       markersize=7, label=f"User {pid_list[i]}")
            for i in range(n_users)
        ]
        ax1.legend(handles=legend_handles_u, fontsize=6, loc='lower right', ncol=2)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved to {save_path}")
    plt.show()
    return fig
 
 
def visualize_conv_filters(model, save_path: str = None, model_name: str = "Model"):
    """
    Visualize learned 1D convolutional filter weights (Meta paper Fig. 4b).
    Shows each filter as a heatmap over (out_channels × time).
    Also plots frequency response (FFT magnitude) of each filter.
 
    Only works for MetaCNNLSTM and DeepCNNLSTM which have a first Conv1d layer.
    """
    # Find the first Conv1d
    conv_layer = None
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            conv_layer = m
            break
 
    if conv_layer is None:
        print("[visualize_conv_filters] No Conv1d found in model.")
        return
 
    W = conv_layer.weight.detach().cpu().numpy()  # (out_ch, in_ch, kernel_size)
    n_filters, n_in_ch, k = W.shape
 
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
 
    # Panel 1: Filter weights heatmap (out_ch × kernel_size, averaged over in_ch)
    W_mean = W.mean(axis=1)   # (out_ch, k)
    im = axes[0].imshow(W_mean, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title(f"{model_name} — Conv Filter Weights (mean over input channels)", fontsize=11)
    axes[0].set_xlabel("Kernel position (time)")
    axes[0].set_ylabel("Filter index")
    plt.colorbar(im, ax=axes[0])
 
    # Panel 2: Frequency response (FFT magnitude, zero-padded)
    fft_len = 256
    for fi in range(n_filters):
        filt = W_mean[fi]
        freq_resp = np.abs(np.fft.rfft(filt, n=fft_len))
        freq_resp = freq_resp / (freq_resp.max() + 1e-8)
        freqs = np.fft.rfftfreq(fft_len)
        axes[1].plot(freqs, freq_resp, alpha=0.4, color='steelblue', linewidth=0.8)
 
    # Median frequency response
    all_resps = []
    for fi in range(n_filters):
        filt = W_mean[fi]
        fr = np.abs(np.fft.rfft(filt, n=fft_len))
        fr = fr / (fr.max() + 1e-8)
        all_resps.append(fr)
    median_resp = np.median(np.stack(all_resps, axis=0), axis=0)
    axes[1].plot(freqs, median_resp, color='navy', linewidth=2.0, label='Median')
    axes[1].set_title("Normalized Frequency Response of Conv Filters", fontsize=11)
    axes[1].set_xlabel("Normalized Frequency")
    axes[1].set_ylabel("Magnitude (normalized)")
    axes[1].legend()
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[ConvViz] Saved to {save_path}")
    plt.show()
    return fig
 
 
def compute_variance_explained(feats_layers, labels, user_ids, n_components=10):
    """
    Compute proportion of variance explained by gesture category vs user identity
    at each layer, following Meta paper Fig. 4i methodology.
 
    Uses R² from a linear regression of the first PC scores onto one-hot variables.
    This is a simplified version; the full Meta method uses η² from ANOVA per PC.
 
    Returns:
        gesture_var: list (n_layers,) — proportion of variance explained by gesture
        user_var:    list (n_layers,) — proportion of variance explained by user
    """
    from sklearn.linear_model import LinearRegression
 
    gesture_var, user_var = [], []
 
    for feats in feats_layers:
        pca   = PCA(n_components=min(n_components, feats.shape[1]))
        scores = pca.fit_transform(StandardScaler().fit_transform(feats))
 
        # One-hot encode gesture and user
        def one_hot(y, n):
            oh = np.zeros((len(y), n))
            oh[np.arange(len(y)), y] = 1.0
            return oh
 
        n_gest = len(set(labels.tolist()))
        n_user = len(set(user_ids.tolist()))
        X_gest = one_hot(labels, n_gest)
        X_user = one_hot(user_ids, n_user)
 
        # R² of gesture → PC scores
        reg_g = LinearRegression().fit(X_gest, scores)
        r2_g  = reg_g.score(X_gest, scores)
 
        # R² of user → PC scores
        reg_u = LinearRegression().fit(X_user, scores)
        r2_u  = reg_u.score(X_user, scores)
 
        gesture_var.append(r2_g)
        user_var.append(r2_u)
 
    return gesture_var, user_var
 
 
def plot_variance_explained(gesture_var, user_var, model_name="Model", save_path=None):
    """Replicates Meta paper Fig. 4i."""
    n_layers = len(gesture_var)
    layer_names = [f"Layer {i+1}" for i in range(n_layers)]
 
    x = np.arange(n_layers)
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, gesture_var, width, label='Gesture category', color='steelblue')
    ax.bar(x + width/2, user_var,    width, label='User identity',     color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel("Proportion of variance explained (R²)")
    ax.set_title(f"{model_name} — Variance Explained per Layer")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MoE routing analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_moe_routing_eval(
    model,
    dataloader,
    config: dict,
    split_label: str = "val",
    save_dir: str = None,
) -> dict:
    """
    Run a full-pass MoE routing analysis on a standard (non-episodic) pretrain dataloader.

    This is the post-training counterpart to the epoch-level analysis already inside
    pretrain_trainer.py. Call it after training on the *best* checkpoint to get the
    cleanest picture of routing behaviour — not a mid-training snapshot.

    The dataloader must yield dicts with at minimum:
        "emg"    : (B, C, T)
        "labels" : (B,)
    Optionally:
        "imu"          : (B, C, T)
        "pid" / "pids" / "user_id" : list[str|int] length B
        "demographics" : (B, D)

    Args:
        model       : trained model (must support forward(emg, imu, return_routing=True))
        dataloader  : standard DataLoader built by get_pretrain_dataloaders()
        config      : full config dict (needs use_moe, num_experts, use_imu, device,
                      model_type, demo_dim_labels)
        split_label : "train" or "val" — used for print tags and filenames
        save_dir    : if not None, routing plots are saved here and the RoutingRecord
                      is saved as a .pt file for later offline inspection

    Returns:
        dict with keys:
            "report"  : full_report() dict from RoutingAnalyzer
            "figures" : dict of matplotlib Figure objects from plot_all()
            "record_path" : path to saved RoutingRecord .pt (or None)

    What to look for:
        - entropy['mean_entropy_normalised'] near 1.0 → all experts equally used
          (probably no specialisation, or no collapse yet)
        - entropy['mean_entropy_normalised'] near 0.0 → hard routing / collapse
        - load_balance['hard_imbalance_ratio'] >> 1 → one expert dominates
        - routing_by_gesture shows same dominant expert for every gesture → gesture
          specialisation has collapsed (or never developed)
        - routing_by_pid shows same expert for every user → user-agnostic routing
    """
    device      = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    num_experts = int(config.get('num_experts', 4))
    use_imu     = config.get('use_imu', False)
    model_name  = config.get('model_type', 'Model')
    demo_labels = config.get('demo_dim_labels', None)

    collector = RoutingCollector(
        num_experts=num_experts,
        model_name=f"{model_name}_{split_label}",
    )

    model.eval()
    model.to(device)

    n_batches_collected = 0
    with torch.no_grad():
        for batch in dataloader:
            emg    = batch['emg'].to(device)
            labels = batch['labels'].cpu()
            imu    = batch.get('imu')
            if use_imu and imu is not None:
                imu = imu.to(device)
            else:
                imu = None

            # Support both key conventions used across the codebase
            pids = batch.get('user_id',
                   batch.get('pid',
                   batch.get('pids', [f'user_{i}' for i in range(emg.size(0))])))
            if isinstance(pids, str):
                pids = [pids] * emg.size(0)
            # Handle tensor pid (integer user indices from some dataloaders)
            if isinstance(pids, torch.Tensor):
                pids = [str(p.item()) for p in pids]

            demo = batch.get('demographics')

            # Forward with routing info requested
            try:
                out = model(emg, imu, return_routing=True)
            except TypeError:
                # Model forward doesn't accept imu as positional — try keyword
                out = model(emg, return_routing=True)

            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                _logits, routing_info = out
                gate_w = routing_info.get('gate_weights')
                if gate_w is None:
                    print(f"[MoE eval:{split_label}] Warning: routing_info missing "
                          "'gate_weights' key. Skipping batch.")
                    continue
                collector.add(
                    gate_weights   = gate_w.detach().cpu(),
                    gesture_labels = labels,
                    pids           = pids,
                    demographics   = demo.cpu() if demo is not None else None,
                )
                n_batches_collected += 1
            else:
                print(f"[MoE eval:{split_label}] Warning: model did not return "
                      "(logits, routing_info) tuple. Is use_moe=True and does the "
                      "model support return_routing=True? Aborting routing analysis.")
                return {}

    if n_batches_collected == 0:
        print(f"[MoE eval:{split_label}] No routing data collected — returning empty results.")
        return {}

    print(f"\n[MoE eval:{split_label}] Collected {n_batches_collected} batches. "
          f"Running analysis...")

    record   = collector.finalize()
    analyzer = RoutingAnalyzer(record)
    report   = analyzer.full_report(print_report=True, demo_dim_labels=demo_labels)
    report['split'] = split_label

    # ── Plots ─────────────────────────────────────────────────────────────────
    figures = {}
    record_path = None
    if save_dir is not None:
        routing_plot_dir = os.path.join(save_dir, "moe_routing", split_label)
        os.makedirs(routing_plot_dir, exist_ok=True)
        figures = analyzer.plot_all(save_dir=routing_plot_dir)
        print(f"[MoE eval:{split_label}] Routing plots saved → {routing_plot_dir}")

        # Save the raw RoutingRecord so the user can reload it offline
        record_path = os.path.join(routing_plot_dir, "routing_record.pt")
        save_routing_record(record, record_path)
        print(f"[MoE eval:{split_label}] RoutingRecord saved → {record_path}")

    return {
        "report":      report,
        "figures":     figures,
        "record_path": record_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader + standalone viz
# ─────────────────────────────────────────────────────────────────────────────
 
def viz_from_checkpoint(
    checkpoint_path: str,
    tensor_dict: dict,
    config: dict,
    save_dir: str = ".",
    method: str = 'pca',
    split: str = 'train',   # 'train', 'val', or 'both'
):
    """
    Load a pretrained model from a .pt checkpoint and produce latent-space
    visualisation plot(s) without running the full evaluation suite.
 
    Typical use case: inspect a model you trained earlier, or compare multiple
    checkpoints side-by-side without re-running linear probes.
 
    Args:
        checkpoint_path : path to a .pt file saved by pretrain_trainer.pretrain()
                          (must contain the full model state_dict under key
                          'model_state_dict', OR be a bare state_dict).
        tensor_dict     : the same tensor_dict used at train time.
        config          : config dict used at train time (needs train_PIDs,
                          val_PIDs, train_reps, val_reps, model_type, device,
                          use_imu).
        save_dir        : directory to write the PNG(s) into.
        method          : 'pca' or 'tsne'.
        split           : which users to visualise — 'train', 'val', or 'both'.
 
    Returns:
        dict with keys 'train' and/or 'val' mapping to matplotlib Figure objects.
        If use_moe=True, also keys 'moe_train' and/or 'moe_val' with routing results.

    Example:
        from pretrain_eval import viz_from_checkpoint
        from pretrain_configs import PRETRAIN_CONFIG, MODEL_CONFIGS
        import pickle, torch
 
        with open('data/tensor_dict.pkl', 'rb') as f:
            td = pickle.load(f)['data']
 
        config = {
            **PRETRAIN_CONFIG,
            **MODEL_CONFIGS['TST'],
            'model_type': 'TST',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        figs = viz_from_checkpoint(
            'pretrain_outputs/checkpoints/TST_best.pt',
            td, config, save_dir='pretrain_outputs/eval/TST',
        )
    """
    from pretrain_models import build_model
 
    os.makedirs(save_dir, exist_ok=True)
 
    device     = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    train_reps = config['train_reps']
    val_reps   = config['val_reps']
    use_imu    = config.get('use_imu', False)
    mname      = config.get('model_type', 'Model')
    use_moe    = config.get('use_moe', False)
 
    train_pids = config['train_PIDs']
    val_pids   = config['val_PIDs']
 
    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"[viz_from_checkpoint] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
 
    model = build_model(config)
    # Support both wrapped {'model_state_dict': ...} and bare state_dict formats.
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[viz_from_checkpoint] Model loaded ({mname}).")
 
    results = {}
    base_save = f"{save_dir}/{mname}_representation_viz.png"
 
    if split in ('train', 'both'):
        fig_train = visualize_representations(
            model, tensor_dict, train_pids, train_reps, device,
            use_imu     = use_imu,
            method      = method,
            max_samples = 500,
            save_path   = base_save,
            model_name  = mname,
            split_label = "train",
        )
        results['train'] = fig_train
 
    if split in ('val', 'both'):
        fig_val = visualize_representations(
            model, tensor_dict, val_pids, val_reps, device,
            use_imu     = use_imu,
            method      = method,
            max_samples = 500,
            save_path   = base_save,
            model_name  = mname,
            split_label = "val",
        )
        results['val'] = fig_val

    # ── MoE routing analysis (if applicable) ─────────────────────────────────
    # We rebuild dataloaders here so viz_from_checkpoint is self-contained and
    # doesn't require the caller to hold onto dataloader objects.
    if use_moe:
        print(f"\n[viz_from_checkpoint] use_moe=True — running post-hoc routing analysis...")
        try:
            from pretrain_data_pipeline import get_pretrain_dataloaders
            train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict)

            if split in ('train', 'both'):
                results['moe_train'] = run_moe_routing_eval(
                    model, train_dl, config,
                    split_label = "train",
                    save_dir    = save_dir,
                )

            if split in ('val', 'both'):
                results['moe_val'] = run_moe_routing_eval(
                    model, val_dl, config,
                    split_label = "val",
                    save_dir    = save_dir,
                )
        except Exception as e:
            print(f"[viz_from_checkpoint] MoE routing analysis failed: {e}")
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience runner
# ─────────────────────────────────────────────────────────────────────────────
 
def run_full_eval(
    model,
    tensor_dict: dict,
    config: dict,
    save_dir: str = ".",
    method: str = 'pca',
    train_dl=None,
    val_dl=None,
):
    """
    Run all evaluation modes:
      1. Linear probe on an in-distribution train user
      2. Linear probe on out-of-distribution val users
      3. PCA/tSNE visualization of all val users
      4. [MoE only] Routing analysis on train + val dataloaders

    config keys:
      train_PIDs, val_PIDs, train_reps, val_reps,
      use_imu, n_way, model_type, device
      use_moe        : bool  — activates MoE routing eval (step 4)
      num_experts    : int   — required when use_moe=True
      demo_dim_labels: list  — optional labels for demographics dimensions

    Args:
        train_dl, val_dl : optional pre-built DataLoaders. If not provided and
                           use_moe=True, they will be constructed internally via
                           get_pretrain_dataloaders(). Pass them in from run_pretrain.py
                           to avoid loading/batching data twice.

    Note on train_reps vs val_reps:
      These are 1-based repetition indices, not gesture class names.
      The gesture class set is inferred from tensor_dict keys inside extract_features.
      train_reps: reps used to fit the probe classifier (e.g. [1,2,3,4,5])
      val_reps:   reps used to evaluate / visualize (e.g. [6,7,8])
    """
    os.makedirs(save_dir, exist_ok=True)
 
    device     = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    train_reps = config.get('train_reps')   # 1-based rep indices for training the probe
    val_reps   = config.get('val_reps')     # 1-based rep indices for evaluating / visualizing
    use_imu    = config.get('use_imu', False)
    n_way      = config.get('n_way', 10)
    mname      = config.get('model_type', 'Model')
    use_moe    = config.get('use_moe', False)
 
    train_pids = config['train_PIDs']
    val_pids   = config['val_PIDs']
 
    results = {}
 
    # ── 1. In-distribution linear probe ─────────────────────────────────────
    # Hold out the first train user as probe; fit on the rest.
    probe_in_pid  = [train_pids[0]]
    train_without = [p for p in train_pids if p not in probe_in_pid]
 
    results['probe_in_dist'] = linear_probe(
        model, tensor_dict,
        probe_pids     = probe_in_pid,
        all_pids_train = train_without,
        target_reps    = train_reps,    # fit and probe both use train reps
        device         = device,
        use_imu        = use_imu,
        n_way          = n_way,
        label          = f"in-distribution (user {probe_in_pid[0]})",
    )
 
    # ── 2. Out-of-distribution linear probe ──────────────────────────────────
    # TODO: confirm whether val_pids are truly new users (cross-subject OOD) or
    # the same users evaluated on held-out reps (intra-subject split). The probe
    # fit uses train_reps on train_pids; evaluation uses val_reps on val_pids.
    results['probe_out_dist'] = linear_probe(
        model, tensor_dict,
        probe_pids     = val_pids,
        all_pids_train = train_pids,
        target_reps    = val_reps,      # val users evaluated on their val reps
        device         = device,
        use_imu        = use_imu,
        n_way          = n_way,
        label          = f"out-of-distribution ({len(val_pids)} val users)",
    )
 
    # ── 3. Representation visualization — TRAIN users ────────────────────────
    # Shows what the backbone learned to separate on subjects it was trained on.
    # Saved as: trainusers_{mname}_representation_viz.png
    fig_repr_train = visualize_representations(
        model, tensor_dict, train_pids, train_reps, device,
        use_imu     = use_imu,
        method      = method,
        max_samples = 500,
        save_path   = f"{save_dir}/{mname}_representation_viz.png",
        model_name  = mname,
        split_label = "train",
    )
    results['repr_fig_train'] = fig_repr_train
 
    # ── 3b. Representation visualization — VAL users ──────────────────────────
    # The more diagnostic plot: can the backbone generalise to new users?
    # Saved as: valusers_{mname}_representation_viz.png
    fig_repr_val = visualize_representations(
        model, tensor_dict, val_pids, val_reps, device,
        use_imu     = use_imu,
        method      = method,
        max_samples = 500,
        save_path   = f"{save_dir}/{mname}_representation_viz.png",
        model_name  = mname,
        split_label = "val",
    )
    results['repr_fig'] = fig_repr_val   # keep legacy key for downstream code
 
    # ── 3b. Conv filter visualization (CNN models only) ──────────────────────
    if mname in ('MetaCNNLSTM', 'DeepCNNLSTM'):
        fig_filt = visualize_conv_filters(
            model,
            save_path  = f"{save_dir}/{mname}_conv_filters.png",
            model_name = mname,
        )
        results['filter_fig'] = fig_filt
 
    # ── 3c. Variance explained plot ───────────────────────────────────────────
    # BUG FIX: was using train_reps for val_pids — now consistently uses val_reps.
    print(f"\n[VarExplained] Computing variance explained for {len(val_pids)} val users...")
    feats_final, feats_layers, labels, user_ids, _ = extract_features(
        model, tensor_dict, val_pids, val_reps, device, use_imu
    )
    g_var, u_var = compute_variance_explained(feats_layers, labels, user_ids)
    fig_var = plot_variance_explained(
        g_var, u_var,
        model_name = mname,
        save_path  = f"{save_dir}/{mname}_variance_explained.png",
    )
    results['variance_explained'] = {'gesture': g_var, 'user': u_var}

    # ── 4. MoE routing analysis ───────────────────────────────────────────────
    # Only runs when use_moe=True.  We need DataLoaders (not just tensor_dict)
    # because the routing collector must receive batched (emg, labels, pid) tuples.
    # Dataloaders passed in from run_pretrain.py are preferred; if not provided
    # we reconstruct them here — slightly redundant but keeps eval self-contained.
    if use_moe:
        print(f"\n{'='*60}")
        print(f"[MoE Routing Eval] Running post-training routing analysis for {mname}...")
        print(f"{'='*60}")

        _train_dl = train_dl
        _val_dl   = val_dl

        if _train_dl is None or _val_dl is None:
            try:
                from pretrain_data_pipeline import get_pretrain_dataloaders
                _train_dl, _val_dl, _ = get_pretrain_dataloaders(config, tensor_dict)
                print("[MoE Routing Eval] DataLoaders reconstructed from tensor_dict.")
            except Exception as e:
                print(f"[MoE Routing Eval] Could not build DataLoaders: {e}. "
                      "Pass train_dl and val_dl to run_full_eval() to avoid this.")
                _train_dl = _val_dl = None

        moe_save_dir = os.path.join(save_dir, "moe_routing")

        if _train_dl is not None:
            print(f"\n[MoE Routing Eval] --- TRAIN split ---")
            results['moe_train'] = run_moe_routing_eval(
                model, _train_dl, config,
                split_label = "train",
                save_dir    = save_dir,
            )

        if _val_dl is not None:
            print(f"\n[MoE Routing Eval] --- VAL split ---")
            results['moe_val'] = run_moe_routing_eval(
                model, _val_dl, config,
                split_label = "val",
                save_dir    = save_dir,
            )

        # ── Collapse summary ──────────────────────────────────────────────────
        # Print a concise MoE health summary so collapse is obvious at a glance.
        _print_moe_health_summary(results, mname)

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY — {mname}")
    print(f"{'='*60}")
    print(f"  In-dist  linear probe: {results['probe_in_dist']['accuracy']*100:.1f}%")
    print(f"  OOD      linear probe: {results['probe_out_dist']['accuracy']*100:.1f}%  "
          f"(chance = {100/n_way:.1f}%)")
    print(f"  Variance explained by gesture (per layer): "
          + " → ".join([f"{v*100:.0f}%" for v in g_var]))
    print(f"  Variance explained by user    (per layer): "
          + " → ".join([f"{v*100:.0f}%" for v in u_var]))
    print(f"{'='*60}\n")
 
    return results


def _print_moe_health_summary(results: dict, mname: str) -> None:
    """
    Print a concise MoE routing health summary to stdout after run_full_eval().

    Interprets the val routing report (falls back to train if val is absent) and
    flags potential collapse / load-imbalance in plain language.

    Collapse indicators surfaced here:
      - Normalised entropy < 0.1 → near-hard routing, high specialisation or collapse
      - Normalised entropy > 0.9 → near-uniform routing, experts not differentiating
      - Hard imbalance ratio > 5  → one expert handling most samples
      - All gestures route to same dominant expert → gesture specialisation absent
    """
    moe_result = results.get('moe_val') or results.get('moe_train')
    if not moe_result or 'report' not in moe_result:
        return

    report = moe_result['report']
    split  = moe_result['report'].get('split', '?')

    ent  = report.get('entropy', {})
    lb   = report.get('load_balance', {})
    rg   = report.get('routing_by_gesture', {})

    norm_ent   = ent.get('mean_entropy_normalised', float('nan'))
    imbal_hard = lb.get('hard_imbalance_ratio', float('nan'))

    print(f"\n{'─'*60}")
    print(f"  MoE HEALTH SUMMARY — {mname} ({split} split)")
    print(f"{'─'*60}")
    print(f"  Normalised routing entropy : {norm_ent:.3f}  "
          f"(0=hard/collapsed, 1=uniform/soft)")
    print(f"  Load imbalance (hard)      : {imbal_hard:.1f}x  (ideal = 1.0x)")

    # Expert load fractions
    fracs = lb.get('expert_hard_fraction', [])
    if fracs:
        frac_strs = "  |  ".join([f"E{i}: {f:.2f}" for i, f in enumerate(fracs)])
        print(f"  Expert dom-freq fractions  : {frac_strs}")

    # Gesture routing collapse check
    dom_freq_mat = rg.get('dominant_freq_matrix')
    if dom_freq_mat is not None:
        import numpy as np
        mat = np.array(dom_freq_mat)
        dominant_experts = mat.argmax(axis=1).tolist()
        all_same = len(set(dominant_experts)) == 1
        if all_same:
            print(f"  ⚠ COLLAPSE DETECTED: all gestures route to Expert {dominant_experts[0]}")
        else:
            print(f"  ✓ Gestures route to different experts: {dominant_experts}")

    # Plain-language verdict
    if norm_ent < 0.1:
        print("  → Very sharp routing. Check for collapse (imbalance ratio, gesture routing).")
    elif norm_ent > 0.9:
        print("  → Near-uniform routing. Experts may not be specialising.")
    else:
        print("  → Routing entropy looks reasonable.")

    print(f"{'─'*60}\n")