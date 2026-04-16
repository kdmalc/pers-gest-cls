"""
A10_A11_A12_meta_pretrained.py
================================
Ablations A10, A11, A12: Meta Pretrained EMG Foundation Model Comparison

⚠️  DIFFERENT DATA: All three ablations use 2000 Hz EMG-only data (no IMU).
    Results are reported in a SEPARATE figure with explicit callout that the
    data format differs from all other ablations (M0, A1–A9).

A10: Meta's pretrained discrete_gestures model, prototypical zero-shot.
     No parameter updates. Features extracted from the frozen backbone
     are used to compute class prototypes; queries are classified by
     nearest prototype (Euclidean distance). This is the correct zero-shot
     protocol when the pretrained head outputs 9 gestures that are DIFFERENT
     from our target gesture set — we cannot re-use their classification head.

A11: Meta's pretrained model, 1-shot fine-tuning.
     Head replaced with a fresh n_way-class linear layer. Two fine-tuning
     modes: head_only (backbone frozen) and full (all params updated).

A12: Our full MAML + MoE model trained on the same 2kHz EMG-only data.
     Apples-to-apples control: same data format as A10/A11, but our method.
     Any gap between A12 and A10/A11 reflects method differences, not
     data format differences.

---------------------------------------------------------------------
KEY DESIGN NOTES — READ BEFORE MODIFYING

Meta's DiscreteGesturesArchitecture is a STREAMING CLASSIFIER.
  forward(x) input shape:  (B, 16, T)       [channel-first, 2kHz raw EMG]
  forward(x) output shape: (B, 9, T_out)    [frame-level logits over time]
  where T_out = floor((T - 21) / 10) + 1    [due to Conv1d kernel=21, stride=10]

  Minimum valid input: T >= 21 samples.
  Recommended: T = 1000 (500 ms at 2kHz, matching the paper's eval windows).

  To get a single per-window classification logit vector we MEAN-POOL over
  the time dimension: logits = output.mean(dim=-1)  → (B, 9)
  Mean-pooling is equivalent to majority-vote but differentiable.

Their checkpoint keys are prefixed with "network." because Lightning wraps
the nn.Module under self.network. We strip that prefix when loading.

Their final layer is self.projection (not self.head). MetaEMGWrapper
exposes a .head attribute (pointing to self.network.projection) so that
replace_head_for_eval() from ablation_config.py works without modification.

Preprocessing:
  Meta's model applies Reinhard compression internally as the first op,
  so we do NOT apply it externally. Their training data was high-pass
  filtered at 20 Hz and noise-floor normalised (std=1). Apply the same
  to your raw 2kHz data before passing it in.
---------------------------------------------------------------------

Usage:
    python A10_A11_A12_meta_pretrained.py --ablation A12
    python A10_A11_A12_meta_pretrained.py --ablation A10
    python A10_A11_A12_meta_pretrained.py --ablation A11
"""

import os, sys, copy, json, argparse
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

# Path to the cloned Meta repo so we can import their architecture class
NEUROMOTOR_REPO = Path(os.environ.get(
    "NEUROMOTOR_REPO",
    "/projects/my13/div-emg/generic-neuromotor-interface"
)).resolve()
sys.path.insert(0, str(NEUROMOTOR_REPO))

from ablation_config import (
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS,
    run_episodic_test_eval, run_supervised_test_eval,
    replace_head_for_eval,
    save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import mamlpp_pretrain

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================

# Path to Meta's pretrained checkpoint
META_CHECKPOINT_PATH = Path(
    "/rhf/allocations/my13/emg_models/discrete_gestures/model_checkpoint.ckpt"
)

# Path to your 2kHz EMG-only tensor_dict pkl (the "pre-downsampling" version).
# Same format as your standard tensor_dict but with seq_len matching 2kHz windows.
EMG_2KHZ_PKL_PATH = "/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/segfilt_2khz_emg_tensor_dict.pkl"

# Input dimensions for your 2kHz data.
# At 2kHz, 500 ms = 1000 samples.
EMG_2KHZ_IN_CH  = 16   
EMG_2KHZ_SEQ_LEN = 4300  # Check extraction NB or the data itself if you want to confirm

# Fine-tuning settings for A11.
# Per spec: tune ft_lr in [1e-5, 1e-2] and ft_steps in {10, 25, 50, 100}.
# These are reasonable defaults; add an Optuna sweep if the paper requires HPO.
A11_FT_LR    = 1e-4
A11_FT_STEPS = 50


# =============================================================================
# ── MetaEMGWrapper ────────────────────────────────────────────────────────────
# =============================================================================

class MetaEMGWrapper(nn.Module):
    """
    Wraps Meta's DiscreteGesturesArchitecture so it slots into our ablation
    eval pipeline (pretrain_finetune.py + ablation_config.py).

    What this wrapper does:
      1. Loads the pretrained weights from the Lightning checkpoint (stripping
         the "network." key prefix added by Lightning's LightningModule).
      2. Exposes a .head attribute (aliased to network.projection) so that
         replace_head_for_eval() works without any changes.
      3. Adapts the forward signature to (x_emg, x_imu=None, demographics=None)
         to match the interface expected by finetune_and_eval_user().
      4. Pools the frame-level output (B, n_classes, T_out) → (B, n_classes)
         via mean pooling over time, giving one logit vector per window.

    After construction the backbone is frozen. For A11 (fine-tuning),
    finetune_and_eval_user() handles unfreezing via the mode argument.

    Args:
        checkpoint_path : path to the Lightning .ckpt file
        freeze_backbone : if True (default), freeze all params except .head
                          Set to False for full fine-tuning (A11 full mode).
                          NOTE: finetune_and_eval_user manages freezing itself,
                          so you can leave this True here.
    """

    def __init__(self, checkpoint_path: Path, freeze_backbone: bool = True):
        super().__init__()

        # ── Import Meta's architecture class ─────────────────────────────────
        # We import here (not at module level) so the script can be imported
        # even if the neuromotor repo isn't on the path yet.
        try:
            # NOTE: I installed their repo as a package on the cluster, so this should work? We'll see
            from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture
        except ImportError as e:
            raise ImportError(
                f"Could not import DiscreteGesturesArchitecture from the Meta repo. "
                f"Check that NEUROMOTOR_REPO is set correctly and the repo is installed.\n"
                f"Original error: {e}"
            )

        # ── Instantiate the architecture (default params match the checkpoint) ─
        self.network = DiscreteGesturesArchitecture(
            input_channels=16,
            conv_output_channels=512,
            kernel_width=21,
            stride=10,
            lstm_hidden_size=512,
            lstm_num_layers=3,
            output_channels=9,   # original 9-class head; replaced at eval time
        )

        # ── Load pretrained weights ───────────────────────────────────────────
        self._load_checkpoint(checkpoint_path)

        # ── Expose .head so replace_head_for_eval() works ─────────────────────
        # replace_head_for_eval() does: model.head = nn.Linear(in_features, n_way)
        # We make .head a property-like alias by storing a reference here.
        # After replace_head_for_eval runs, self.head and self.network.projection
        # will point to different objects — that is intentional and correct.
        # forward() uses self.head (not self.network.projection) after replacement.
        self.head = self.network.projection  # Linear(512, 9) initially

        # ── Optionally freeze backbone ────────────────────────────────────────
        if freeze_backbone:
            self._freeze_backbone()

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_checkpoint(self, checkpoint_path: Path):
        """
        Load a Lightning checkpoint, strip the "network." prefix from keys,
        and load into self.network.
        """
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), (
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run: python -m generic_neuromotor_interface.scripts.download_models "
            f"--task discrete_gestures --output-dir ~/emg_models"
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert "state_dict" in ckpt, (
            f"Expected 'state_dict' key in checkpoint. Got keys: {list(ckpt.keys())}"
        )

        raw_sd = ckpt["state_dict"]

        # Lightning prefixes everything with "network." — strip it.
        PREFIX = "network."
        stripped_sd = {}
        for k, v in raw_sd.items():
            assert k.startswith(PREFIX), (
                f"Unexpected key in checkpoint (does not start with '{PREFIX}'): {k!r}\n"
                f"If Meta changed their Lightning module structure, update PREFIX."
            )
            stripped_sd[k[len(PREFIX):]] = v

        missing, unexpected = self.network.load_state_dict(stripped_sd, strict=True)
        # strict=True: any mismatch is a hard error — we want to know immediately.
        assert len(missing) == 0, f"Missing keys after load: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys after load: {unexpected}"

        print(f"[MetaEMGWrapper] Loaded pretrained weights from {checkpoint_path}")

    # ── Backbone freezing ─────────────────────────────────────────────────────

    def _freeze_backbone(self):
        """Freeze everything in self.network except self.head (projection layer)."""
        for name, param in self.network.named_parameters():
            # Freeze all params. finetune_and_eval_user() will selectively
            # unfreeze via mode='head_only' or mode='full'.
            param.requires_grad_(False)
        print("[MetaEMGWrapper] Backbone frozen.")

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x_emg: torch.Tensor,
        x_imu=None,          # ignored — Meta's model is EMG-only
        demographics=None,   # ignored
    ) -> torch.Tensor:
        """
        Args:
            x_emg : (B, C, T) — channel-first, 2kHz raw EMG, C=16
                     Must satisfy T >= 21 (Conv1d kernel width).

        Returns:
            logits : (B, n_classes) — mean-pooled over time dimension.
                     n_classes = 9 initially; changes to n_way after
                     replace_head_for_eval() swaps self.head.
        """
        # ── Backbone forward (everything up to but not including the head) ────
        # We call backbone() rather than self.network() so we can intercept
        # before self.head (which may have been replaced by replace_head_for_eval).
        features = self._backbone_forward(x_emg)  # (B, T_out, 512)

        # ── Head (may be the original projection or the replaced n_way head) ──
        # self.head is a Linear(512, n_classes).
        # features: (B, T_out, 512) → logits_per_frame: (B, T_out, n_classes)
        logits_per_frame = self.head(features)     # (B, T_out, n_classes)

        # ── Aggregate: mean-pool over time → single logit vector per window ───
        logits = logits_per_frame.mean(dim=1)      # (B, n_classes)

        return logits

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run everything in self.network EXCEPT the final projection layer,
        returning the post-LSTM-LayerNorm feature tensor of shape (B, T_out, 512).

        This is called by forward() and also directly by
        _extract_features_for_prototypical() for zero-shot eval.
        """
        net = self.network

        # Reinhard compression
        x = net.compression(x)          # (B, 16, T)

        # Conv1d + ReLU + Dropout
        x = net.conv_layer(x)           # (B, 512, T_out)
        x = net.relu(x)
        x = net.dropout(x)

        # LayerNorm expects (B, T_out, 512)
        x = x.transpose(1, 2)           # (B, T_out, 512)
        x = net.post_conv_layer_norm(x)

        # Stacked LSTM
        x, _ = net.lstm(x)              # (B, T_out, 512)

        # Post-LSTM LayerNorm
        x = net.post_lstm_layer_norm(x) # (B, T_out, 512)

        return x  # (B, T_out, 512)

    def get_pooled_features(self, x_emg: torch.Tensor) -> torch.Tensor:
        """
        Extract mean-pooled backbone features for prototypical zero-shot eval.

        Args:
            x_emg : (B, C, T)

        Returns:
            features : (B, 512) — mean-pooled over time, L2-normalised.
                        L2 normalisation is standard for prototypical networks
                        and cosine-distance nearest-neighbour classifiers.
        """
        with torch.no_grad():
            feats = self._backbone_forward(x_emg)  # (B, T_out, 512)
            feats = feats.mean(dim=1)               # (B, 512)
            feats = nn.functional.normalize(feats, dim=-1)  # L2 normalise
        return feats


# =============================================================================
# ── Prototypical zero-shot eval (A10) ────────────────────────────────────────
# =============================================================================

def prototypical_zeroshot_eval(
    model: MetaEMGWrapper,
    config: dict,
    tensor_dict_path: str,
    test_pids: list,
    num_episodes: int,
) -> dict:
    """
    Zero-shot prototypical evaluation for A10.

    Protocol (identical episode structure to MAML eval):
      For each episode:
        1. Sample n_way classes and k_shot support + q_query query per class.
        2. Extract backbone features for all support examples.
        3. Compute per-class prototype = mean of support features.
        4. Classify each query by nearest prototype (Euclidean distance in
           L2-normalised feature space, i.e. cosine similarity maximisation).
        5. Accuracy = fraction of correctly classified queries.

    No gradient updates. No head replacement. Backbone weights are frozen.

    Args:
        model            : MetaEMGWrapper with frozen backbone
        config           : ablation config dict (n_way, k_shot, q_query, etc.)
        tensor_dict_path : path to your 2kHz tensor_dict .pkl
        test_pids        : list of test participant IDs
        num_episodes     : number of episodes per test user

    Returns:
        dict with per_user_acc, mean_acc, std_acc, num_episodes
    """
    import pickle
    from collections import defaultdict
    from torch.utils.data import DataLoader
    from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate

    device = config["device"]

    from MAML.maml_data_pipeline import reorient_tensor_dict
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    test_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = test_pids,
        target_gesture_classes = config["maml_gesture_classes"],
        target_trial_indices   = config["target_trial_indices"],
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],
        q_query                = config.get("q_query", None),
        num_eval_episodes      = num_episodes,
        is_train               = False,
        seed                   = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=maml_mm_collate,
    )

    model.eval()
    model.to(device)

    user_metrics: dict = defaultdict(list)

    for batch in test_dl:
        uid     = batch["user_id"]
        support = batch["support"]
        query   = batch["query"]

        sup_emg    = support["emg"].to(device)    # (1, k*n_way, C, T)
        sup_labels = support["labels"].to(device)  # (1, k*n_way)
        qry_emg    = query["emg"].to(device)       # (1, q*n_way, C, T)
        qry_labels = query["labels"].to(device)    # (1, q*n_way)

        # Remove the batch-of-1 outer dim added by the dataloader
        sup_emg    = sup_emg.squeeze(0)    # (k*n_way, C, T)
        sup_labels = sup_labels.squeeze(0) # (k*n_way,)
        qry_emg    = qry_emg.squeeze(0)    # (q*n_way, C, T)
        qry_labels = qry_labels.squeeze(0) # (q*n_way,)

        n_way = config["n_way"]

        # ── Extract features ────────────────────────────────────────────────
        # get_pooled_features already applies no_grad + L2 normalisation
        sup_feats = model.get_pooled_features(sup_emg)  # (k*n_way, 512)
        qry_feats = model.get_pooled_features(qry_emg)  # (q*n_way, 512)

        # ── Compute class prototypes ─────────────────────────────────────────
        # sup_labels are remapped 0..n_way-1 by MetaGestureDataset
        prototypes = torch.zeros(n_way, sup_feats.shape[-1], device=device)
        for c in range(n_way):
            mask = (sup_labels == c)
            assert mask.sum() > 0, (
                f"Class {c} has no support examples. "
                f"Check MetaGestureDataset label remapping."
            )
            prototypes[c] = sup_feats[mask].mean(dim=0)

        # L2-normalise prototypes (support feats are already normalised;
        # their mean may not be unit norm, so re-normalise)
        prototypes = nn.functional.normalize(prototypes, dim=-1)  # (n_way, 512)

        # ── Nearest-prototype classification ─────────────────────────────────
        # Euclidean distance in L2-normalised space == cosine distance (monotone)
        # dists[i, c] = ||qry_feats[i] - prototypes[c]||^2
        dists = torch.cdist(qry_feats, prototypes)  # (q*n_way, n_way)
        preds = dists.argmin(dim=-1)                 # (q*n_way,)

        acc = (preds == qry_labels).float().mean().item()
        user_metrics[uid].append(acc)

    per_user = {uid: float(np.mean(accs)) for uid, accs in user_metrics.items()}
    vals = list(per_user.values())
    return {
        "per_user_acc": per_user,
        "mean_acc":     float(np.mean(vals)),
        "std_acc":      float(np.std(vals)),
        "num_episodes": num_episodes,
    }


# =============================================================================
# ── Shared config builder for A10 / A11 ──────────────────────────────────────
# =============================================================================

def build_config_meta(ablation_id: str) -> dict:
    """
    Config for A10 and A11 (Meta model, 2kHz EMG-only).

    Identical to A12 config except ablation_id.
    Both A10/A11 use the same data as A12 so results are directly comparable.
    """
    _check_2khz_data_configured()

    config = make_base_config(ablation_id=ablation_id)

    config["use_imu"]         = False
    config["emg_in_ch"]       = EMG_2KHZ_IN_CH
    config["sequence_length"] = EMG_2KHZ_SEQ_LEN
    config["dfs_load_path"]   = str(Path(EMG_2KHZ_PKL_PATH).parent) + "/"

    # Fine-tuning settings (used by A11 via run_supervised_test_eval)
    config["ft_lr"]           = A11_FT_LR
    config["ft_steps"]        = A11_FT_STEPS
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]
    config["ft_label_smooth"] = 0.0

    return config


# =============================================================================
# ── A10: prototypical zero-shot ───────────────────────────────────────────────
# =============================================================================

def run_a10():
    config = build_config_meta(ablation_id="A10")
    print("\nA10 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    tensor_dict_path = str(EMG_2KHZ_PKL_PATH)

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[A10] Seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"      Prototypical zero-shot — no parameter updates")
        print(f"{'='*70}")
        set_seeds(actual_seed)

        model = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
        model.to(config["device"])
        n_params = count_parameters(model)
        print(f"[A10 | seed={actual_seed}] Parameters (frozen backbone): {n_params:,}")

        # Zero-shot: backbone is identical across seeds, but episode sampling
        # uses the seed — run multiple seeds so variance reflects sampling noise.
        config_seed = copy.deepcopy(config)
        config_seed["seed"] = actual_seed

        test_results = prototypical_zeroshot_eval(
            model, config_seed, tensor_dict_path, config["test_PIDs"],
            num_episodes=500,  # NUM_TEST_EPISODES
        )

        print(f"[A10 | seed={actual_seed}] Zero-shot: "
              f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")

        all_seed_results.append({
            "seed":         actual_seed,
            "test_results": test_results,
            "n_params":     n_params,
        })

    test_accs = [r["test_results"]["mean_acc"] for r in all_seed_results]
    summary = {
        "ablation_id":    "A10",
        "description":    "Meta pretrained — prototypical zero-shot (no fine-tuning)",
        "data_note":      "2kHz EMG-only — compare only against A11/A12, not M0–A9",
        "eval_protocol":  "prototypical nearest-centroid in L2-normalised feature space",
        "n_params":       all_seed_results[0]["n_params"],
        "seed_results":   all_seed_results,
        "mean_test_acc":  float(np.mean(test_accs)),
        "std_test_acc":   float(np.std(test_accs)),
        "num_seeds":      NUM_FINAL_SEEDS,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[A10] FINAL zero-shot: "
          f"{summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
    print(f"      ⚠  2kHz EMG-only — compare only against A11/A12")
    print(f"{'='*70}")


# =============================================================================
# ── A11: Meta pretrained + 1-shot fine-tuning ─────────────────────────────────
# =============================================================================

def run_a11():
    config = build_config_meta(ablation_id="A11")
    print("\nA11 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    tensor_dict_path = str(EMG_2KHZ_PKL_PATH)

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[A11] Seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"      ft_lr={A11_FT_LR}, ft_steps={A11_FT_STEPS}")
        print(f"{'='*70}")
        set_seeds(actual_seed)

        # Load a fresh copy of the pretrained model for each seed.
        # The backbone is frozen here; finetune_and_eval_user() handles
        # unfreezing depending on ft_mode ('head_only' vs 'full').
        model = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
        model.to(config["device"])
        n_params = count_parameters(model)
        print(f"[A11 | seed={actual_seed}] Parameters: {n_params:,}")

        config_seed = copy.deepcopy(config)
        config_seed["seed"] = actual_seed

        # run_supervised_test_eval handles the episodic loop and calls
        # finetune_and_eval_user() which in turn calls replace_head_for_eval().
        # MetaEMGWrapper exposes .head so replace_head_for_eval works as-is.
        head_results = run_supervised_test_eval(
            model, config_seed, tensor_dict_path, config["test_PIDs"],
            ft_mode="head_only",
        )
        full_results = run_supervised_test_eval(
            model, config_seed, tensor_dict_path, config["test_PIDs"],
            ft_mode="full",
        )

        print(f"[A11 | seed={actual_seed}] head-only: "
              f"{head_results['mean_acc']*100:.2f}% ± {head_results['std_acc']*100:.2f}%")
        print(f"[A11 | seed={actual_seed}] full-ft:   "
              f"{full_results['mean_acc']*100:.2f}% ± {full_results['std_acc']*100:.2f}%")

        all_seed_results.append({
            "seed":           actual_seed,
            "test_head_only": head_results,
            "test_full_ft":   full_results,
            "n_params":       n_params,
        })

    head_accs = [r["test_head_only"]["mean_acc"] for r in all_seed_results]
    full_accs = [r["test_full_ft"]["mean_acc"]   for r in all_seed_results]

    summary = {
        "ablation_id":         "A11",
        "description":         "Meta pretrained — 1-shot head-only and full fine-tuning",
        "data_note":           "2kHz EMG-only — compare only against A10/A12, not M0–A9",
        "ft_lr":               A11_FT_LR,
        "ft_steps":            A11_FT_STEPS,
        "n_params":            all_seed_results[0]["n_params"],
        "seed_results":        all_seed_results,
        "mean_test_head_only": float(np.mean(head_accs)),
        "std_test_head_only":  float(np.std(head_accs)),
        "mean_test_full_ft":   float(np.mean(full_accs)),
        "std_test_full_ft":    float(np.std(full_accs)),
        "num_seeds":           NUM_FINAL_SEEDS,
        "config_snapshot":     {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[A11] FINAL head-only: "
          f"{summary['mean_test_head_only']*100:.2f}% ± {summary['std_test_head_only']*100:.2f}%")
    print(f"[A11] FINAL full-ft:   "
          f"{summary['mean_test_full_ft']*100:.2f}% ± {summary['std_test_full_ft']*100:.2f}%")
    print(f"      ⚠  2kHz EMG-only — compare only against A10/A12")
    print(f"{'='*70}")


# =============================================================================
# ── A12: MAML + MoE on 2kHz EMG-only (apples-to-apples control) ──────────────
# =============================================================================

def build_config_a12() -> dict:
    """
    A12 = M0 with use_imu=False and 2kHz EMG-only data.
    Uses M0's best hyperparameters as a starting point.
    Per spec: ideally run a separate HPO sweep; flag the caveat if skipped.
    """
    _check_2khz_data_configured()

    config = make_base_config(ablation_id="A12")

    config["use_imu"]         = False
    config["emg_in_ch"]       = EMG_2KHZ_IN_CH
    config["sequence_length"] = EMG_2KHZ_SEQ_LEN
    config["dfs_load_path"]   = str(Path(EMG_2KHZ_PKL_PATH).parent) + "/"

    return config


def run_a12_one_seed(seed: int, config: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[A12 | seed={seed}] Parameters: {n_params:,}")
    print(f"[A12 | seed={seed}] Input: EMG-only {config['emg_in_ch']}ch @ 2kHz, "
          f"seq_len={config['sequence_length']}")

    tensor_dict_path = str(EMG_2KHZ_PKL_PATH)
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[A12 | seed={seed}] Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "seed":             seed,
            "model_state_dict": train_history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
        },
        config,
        tag=f"seed{seed}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])
    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[A12 | seed={seed}] Test: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")

    return {
        "seed":         seed,
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


def run_a12():
    config = build_config_a12()
    print("\nA12 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[A12] Seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_a12_one_seed(actual_seed, config)
        all_seed_results.append(result)

    test_accs = [r["test_results"]["mean_acc"] for r in all_seed_results]
    summary = {
        "ablation_id":    "A12",
        "description":    "MAML + MoE on 2kHz EMG-only (apples-to-apples control for A10/A11)",
        "data_note":      "2kHz EMG-only — compare only against A10/A11, not M0–A9",
        "n_params":       all_seed_results[0]["n_params"],
        "seed_results":   all_seed_results,
        "mean_test_acc":  float(np.mean(test_accs)),
        "std_test_acc":   float(np.std(test_accs)),
        "num_seeds":      NUM_FINAL_SEEDS,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[A12] FINAL: {summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
    print(f"      ⚠  2kHz EMG-only data — compare only against A10/A11, not M0")
    print(f"{'='*70}")


# =============================================================================
# ── Guard: check that 2kHz data paths are configured before running ───────────
# =============================================================================

def _check_2khz_data_configured():
    if EMG_2KHZ_PKL_PATH is None:
        raise ValueError(
            "EMG_2KHZ_PKL_PATH is not set.\n"
            "Edit the TODO block near the top of this file and set it to the path "
            "of your 2kHz EMG-only tensor_dict .pkl file."
        )
    if not Path(EMG_2KHZ_PKL_PATH).exists():
        raise FileNotFoundError(
            f"EMG_2KHZ_PKL_PATH does not exist: {EMG_2KHZ_PKL_PATH}\n"
            "Check the path is correct."
        )


# =============================================================================
# ── Entry point ───────────────────────────────────────────────────────────────
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=["A10", "A11", "A12"],
        required=True,
        help="Which ablation to run.",
    )
    args = parser.parse_args()

    if args.ablation == "A10":
        run_a10()
    elif args.ablation == "A11":
        run_a11()
    elif args.ablation == "A12":
        run_a12()


if __name__ == "__main__":
    main()