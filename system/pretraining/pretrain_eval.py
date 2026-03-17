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

Usage:
    from pretrain_eval import run_full_eval
    results = run_full_eval(model, tensor_dict, config)
"""

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
from collections import defaultdict

from pretrain_data_pipeline import ensure_channel_first

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, tensor_dict, pids, target_reps, device, use_imu=False):
    """
    Extract backbone features for all (pid, gesture, trial) combinations.

    Returns:
        feats_final:  np.ndarray (N, feat_dim) — final layer features
        feats_layers: list of np.ndarray (N, feat_dim) — one per LSTM/block layer
        labels:       np.ndarray (N,) — gesture index
        user_ids:     np.ndarray (N,) — pid index (int)
        pid_list:     list of unique pids (maps int index → pid)
    """
    model.eval()
    model.to(device)

    all_gestures = set()
    for pid in pids:
        if pid in tensor_dict:
            all_gestures.update(tensor_dict[pid].keys())
    sorted_gestures = sorted(list(all_gestures))
    label_map = {g: i for i, g in enumerate(sorted_gestures)}

    all_final, all_layers, all_labels, all_users = [], [], [], []
    pid_list = sorted(pids)
    pid_to_idx = {p: i for i, p in enumerate(pid_list)}

    for pid in pid_list:
        if pid not in tensor_dict:
            continue
        for gest in sorted_gestures:
            if gest not in tensor_dict[pid]:
                continue
            slot     = tensor_dict[pid][gest]
            emg_all = slot['emg']   # (n_trials, T, C) or (n_trials, C, T)
            imu_all = slot.get('imu', None)

            emg_data = ensure_channel_first(emg_all)

            # Slice specific validation repetitions! 
            valid_idxs = [rep - 1 for rep in target_reps if 0 <= rep - 1 < emg_data.shape[0]]
            if not valid_idxs: continue

            emg_data = emg_data[valid_idxs].float().to(device)  # Will shape to (n_valid_trials, ...)
            imu_input = None
            if use_imu and imu_all is not None:
                imu_data = ensure_channel_first(imu_all)
                imu_input = imu_data[valid_idxs].float().to(device)

            feat_final, layer_feats = model.backbone(emg_data, imu_input)

            all_final.append(feat_final.cpu().numpy())
            # layer_feats is a list of (N, dim) tensors
            if not all_layers:
                all_layers = [[] for _ in layer_feats]
            for li, lf in enumerate(layer_feats):
                all_layers[li].append(lf.cpu().numpy())

            n_trials = emg_data.shape[0]
            all_labels.extend([label_map[gest]] * n_trials)
            all_users.extend([pid_to_idx[pid]] * n_trials)

    feats_final  = np.concatenate(all_final, axis=0)
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
    gestures: list,
    device,
    use_imu: bool = False,
    n_way: int = 10,
    label: str = "probe",
):
    """
    Fit a logistic regression on train users' features, evaluate on probe_pids.
    For "in-distribution" probe: probe_pids ⊂ all_pids_train.
    For "out-of-distribution" probe: probe_pids ∩ all_pids_train = ∅.

    Returns dict with accuracy and per-class breakdown.
    """
    # Extract train features
    print(f"[LinearProbe:{label}] Extracting train features ({len(all_pids_train)} users)...")
    feats_tr, _, labels_tr, _, _ = extract_features(
        model, tensor_dict, all_pids_train, gestures, device, use_imu
    )

    # Extract probe features
    print(f"[LinearProbe:{label}] Extracting probe features ({len(probe_pids)} users)...")
    feats_pr, _, labels_pr, _, _ = extract_features(
        model, tensor_dict, probe_pids, gestures, device, use_imu
    )

    # Fit logistic regression (L2, liblinear solver — fast for small data)
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

    acc = probe_clf.score(feats_pr, labels_pr)
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
    print(f"\n[Visualization] Extracting features from {len(pids)} users...")
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

    fig, axes = plt.subplots(
        n_layers, 2,
        figsize=(10, 4 * n_layers),
        squeeze=False,
    )
    fig.suptitle(f"{model_name} — {method.upper()} Layer Representations\n"
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
# Top-level convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eval(
    model,
    tensor_dict: dict,
    config: dict,
    save_dir: str = ".",
    method: str = 'pca',
):
    """
    Run all three evaluation modes:
      1. Linear probe on an in-distribution train user
      2. Linear probe on out-of-distribution val users
      3. PCA/tSNE visualization of all val users

    config keys:
      train_PIDs, val_PIDs, train_reps, val_reps,
      use_imu, n_way, model_type, device
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    device   = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    gestures = config.get('train_reps')
    use_imu  = config.get('use_imu', False)
    n_way    = config.get('n_way', 10)
    mname    = config.get('model_type', 'Model')

    results = {}

    # ── 1. In-distribution linear probe ─────────────────────────────────────
    # Use a subset of train users as "probe" but fit on the REST of train users.
    train_pids    = config['train_PIDs']
    val_pids      = config['val_PIDs']
    # Pick one train user as the in-distribution probe
    probe_in_pid  = [train_pids[0]]
    train_without = [p for p in train_pids if p not in probe_in_pid]

    results['probe_in_dist'] = linear_probe(
        model, tensor_dict,
        probe_pids       = probe_in_pid,
        all_pids_train   = train_without,
        gestures         = gestures,
        device           = device,
        use_imu          = use_imu,
        n_way            = n_way,
        label            = f"in-distribution (user {probe_in_pid[0]})",
    )

    # TODO: I think the OOD isn't really OOD anymore, it is testing the intra-subject split gestures, not new users...
    # ── 2. Out-of-distribution linear probe ──────────────────────────────────
    results['probe_out_dist'] = linear_probe(
        model, tensor_dict,
        probe_pids       = val_pids,
        all_pids_train   = train_pids,
        gestures         = config.get('val_reps'),
        device           = device,
        use_imu          = use_imu,
        n_way            = n_way,
        label            = f"out-of-distribution ({len(val_pids)} val users)",
    )

    # ── 3. Representation visualization ──────────────────────────────────────
    # Visualize test/val users (held-out) — exactly what Meta does
    fig_repr = visualize_representations(
        model, tensor_dict, val_pids, gestures, device,
        use_imu    = use_imu,
        method     = method,
        max_samples= 500,
        save_path  = f"{save_dir}/{mname}_representation_viz.png",
        model_name = mname,
    )
    results['repr_fig'] = fig_repr

    # ── 3b. Conv filter visualization (CNN models only) ──────────────────────
    if mname in ('MetaCNNLSTM', 'DeepCNNLSTM'):
        fig_filt = visualize_conv_filters(
            model,
            save_path  = f"{save_dir}/{mname}_conv_filters.png",
            model_name = mname,
        )
        results['filter_fig'] = fig_filt

    # ── 3c. Variance explained plot ───────────────────────────────────────────
    print(f"\n[VarExplained] Computing variance explained for {len(val_pids)} val users...")
    feats_final, feats_layers, labels, user_ids, _ = extract_features(
        model, tensor_dict, val_pids, gestures, device, use_imu
    )
    g_var, u_var = compute_variance_explained(feats_layers, labels, user_ids)
    fig_var = plot_variance_explained(
        g_var, u_var,
        model_name = mname,
        save_path  = f"{save_dir}/{mname}_variance_explained.png",
    )
    results['variance_explained'] = {'gesture': g_var, 'user': u_var}

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
