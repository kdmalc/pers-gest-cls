# A10_A11_A12_meta_pretrained.py
"""
A10_A11_A12_meta_pretrained.py
================================
Ablations A10, A11, A12: Meta Pretrained EMG Foundation Model Comparison

⚠️  DIFFERENT DATA: All three ablations use 2000 Hz EMG-only data (no IMU).
    Results are reported in a SEPARATE figure with explicit callout that the
    data format differs from all other ablations (M0, A1–A9).

A10: Meta's pretrained discrete_gestures model, prototypical zero-shot.
A11: Meta's pretrained model, 1-shot fine-tuning (head_only and full).
A12: Our full MAML + MoE model trained on the same 2kHz EMG-only data.
     Apples-to-apples control: same data format as A10/A11, but our method.

test_procedure:
  'hpo_test_split' : Fixed split, single run at FIXED_SEED.
  'L2SO'           : Leave-2-Subjects-Out over all_PIDs (same subject list
                     as all other ablations — only the data frequency differs).
                     test=subjects[i], val=subjects[(i+1)%N], train=rest.
                     A10 has no training phase so L2SO only affects which
                     subjects appear as test subjects.

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

import sys
import subprocess

print("=== ENVIRONMENT DIAGNOSTICS ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")

result = subprocess.run(
    [sys.executable, "-m", "pip", "show", "omegaconf"],
    capture_output=True, text=True
)
print(f"omegaconf pip show:\n{result.stdout or result.stderr}")
print("================================")

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

NEUROMOTOR_REPO = Path(os.environ.get(
    "NEUROMOTOR_REPO",
    "/projects/my13/div-emg/generic-neuromotor-interface"
)).resolve()
sys.path.insert(0, str(NEUROMOTOR_REPO))

from ablation_config import (
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED,
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

META_CHECKPOINT_PATH = Path(
    "/rhf/allocations/my13/emg_models/discrete_gestures/model_checkpoint.ckpt"
)

EMG_2KHZ_PKL_PATH  = "/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/segfilt_2khz_emg_tensor_dict.pkl"
EMG_2KHZ_IN_CH     = 16
EMG_2KHZ_SEQ_LEN   = 4300

A11_FT_LR    = 0.01
A11_FT_STEPS = 150


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
    """

    def __init__(self, checkpoint_path: Path, freeze_backbone: bool = True):
        super().__init__()

        try:
            from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture
        except ImportError as e:
            raise ImportError(
                f"Could not import DiscreteGesturesArchitecture from the Meta repo. "
                f"Check that NEUROMOTOR_REPO is set correctly and the repo is installed.\n"
                f"Original error: {e}"
            )

        self.network = DiscreteGesturesArchitecture(
            input_channels=16,
            conv_output_channels=512,
            kernel_width=21,
            stride=10,
            lstm_hidden_size=512,
            lstm_num_layers=3,
            output_channels=9,
        )

        self._load_checkpoint(checkpoint_path)
        self.head = self.network.projection  # Linear(512, 9) initially

        if freeze_backbone:
            self._freeze_backbone()

    def _load_checkpoint(self, checkpoint_path: Path):
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
        PREFIX = "network."
        stripped_sd = {}
        for k, v in raw_sd.items():
            assert k.startswith(PREFIX), (
                f"Unexpected key in checkpoint (does not start with '{PREFIX}'): {k!r}\n"
                f"If Meta changed their Lightning module structure, update PREFIX."
            )
            stripped_sd[k[len(PREFIX):]] = v

        missing, unexpected = self.network.load_state_dict(stripped_sd, strict=True)
        assert len(missing) == 0,    f"Missing keys after load: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys after load: {unexpected}"
        print(f"[MetaEMGWrapper] Loaded pretrained weights from {checkpoint_path}")

    def _freeze_backbone(self):
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)
        print("[MetaEMGWrapper] Backbone frozen.")

    def forward(self, x_emg, x_imu=None, demographics=None):
        features         = self._backbone_forward(x_emg)  # (B, T_out, 512)
        logits_per_frame = self.head(features)             # (B, T_out, n_classes)
        logits           = logits_per_frame.mean(dim=1)    # (B, n_classes)
        return logits

    def _backbone_forward(self, x):
        net = self.network
        x = net.compression(x)
        x = net.conv_layer(x)
        x = net.relu(x)
        x = net.dropout(x)
        x = x.transpose(1, 2)
        x = net.post_conv_layer_norm(x)
        x, _ = net.lstm(x)
        x = net.post_lstm_layer_norm(x)
        return x  # (B, T_out, 512)

    def get_pooled_features(self, x_emg):
        with torch.no_grad():
            feats = self._backbone_forward(x_emg)
            feats = feats.mean(dim=1)
            feats = nn.functional.normalize(feats, dim=-1)
        return feats


# =============================================================================
# ── Prototypical zero-shot eval (A10) ────────────────────────────────────────
# =============================================================================

def prototypical_zeroshot_eval(model, config, tensor_dict_path, test_pids, num_episodes):
    import pickle
    from collections import defaultdict
    from torch.utils.data import DataLoader
    from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate, reorient_tensor_dict

    device = config["device"]

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
        q_query                = config["q_query"],
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

    user_metrics = defaultdict(list)

    for batch in test_dl:
        uid        = batch["user_id"]
        sup_emg    = batch["support"]["emg"].to(device).squeeze(0)
        sup_labels = batch["support"]["labels"].to(device).squeeze(0)
        qry_emg    = batch["query"]["emg"].to(device).squeeze(0)
        qry_labels = batch["query"]["labels"].to(device).squeeze(0)

        n_way     = config["n_way"]
        sup_feats = model.get_pooled_features(sup_emg)
        qry_feats = model.get_pooled_features(qry_emg)

        prototypes = torch.zeros(n_way, sup_feats.shape[-1], device=device)
        for c in range(n_way):
            mask = (sup_labels == c)
            assert mask.sum() > 0, (
                f"Class {c} has no support examples. "
                f"Check MetaGestureDataset label remapping."
            )
            prototypes[c] = sup_feats[mask].mean(dim=0)

        prototypes = nn.functional.normalize(prototypes, dim=-1)
        dists      = torch.cdist(qry_feats, prototypes)
        preds      = dists.argmin(dim=-1)
        acc        = (preds == qry_labels).float().mean().item()
        user_metrics[uid].append(acc)

    per_user = {uid: float(np.mean(accs)) for uid, accs in user_metrics.items()}
    vals     = list(per_user.values())
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
    _check_2khz_data_configured()
    config = make_base_config(ablation_id=ablation_id)
    config["use_imu"]         = False
    config["emg_in_ch"]       = EMG_2KHZ_IN_CH
    config["sequence_length"] = EMG_2KHZ_SEQ_LEN
    config["dfs_load_path"]   = str(Path(EMG_2KHZ_PKL_PATH).parent) + "/"
    config["ft_lr"]           = A11_FT_LR
    config["ft_steps"]        = A11_FT_STEPS
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]
    config["ft_label_smooth"] = 0.0
    return config


# =============================================================================
# ── L2SO fold builder (shared across A10, A11, A12) ──────────────────────────
# =============================================================================

def build_l2so_folds(all_pids: list) -> list:
    """
    Same round-robin scheme as all other ablations.
    test=all_pids[i], val=all_pids[(i+1)%N], train=rest.
    """
    n = len(all_pids)
    folds = []
    for i in range(n):
        test_pid   = all_pids[i]
        val_pid    = all_pids[(i + 1) % n]
        train_pids = [p for p in all_pids if p != test_pid and p != val_pid]
        folds.append({
            "fold_idx":   i,
            "test_pid":   test_pid,
            "val_pid":    val_pid,
            "train_pids": train_pids,
        })
    return folds


# =============================================================================
# ── A10: prototypical zero-shot ───────────────────────────────────────────────
# =============================================================================

def run_a10():
    config         = build_config_meta(ablation_id="A10")
    test_procedure = config["test_procedure"]

    print("\nA10 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"\nTest procedure: {test_procedure}")

    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'."
    )

    tensor_dict_path = str(EMG_2KHZ_PKL_PATH)

    if test_procedure == "hpo_test_split":
        result  = _run_a10_hpo_test_split(config, tensor_dict_path)
        summary = {
            "ablation_id":     "A10",
            "description":     "Meta pretrained — prototypical zero-shot (no fine-tuning)",
            "data_note":       "2kHz EMG-only — compare only against A11/A12, not M0–A9",
            "eval_protocol":   "prototypical nearest-centroid in L2-normalised feature space",
            "test_procedure":  "hpo_test_split",
            "seed":            FIXED_SEED,
            "n_params":        result["n_params"],
            "result":          result,
            "test_acc":        result["test_results"]["mean_acc"],
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }
    else:  # L2SO
        # A10 has no training phase. L2SO here means: for each fold, evaluate
        # the frozen pretrained model on that fold's test subject only.
        # val_pid and train_pids are unused by A10 but recorded for consistency.
        all_results = _run_a10_l2so(config, tensor_dict_path)
        test_accs   = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":     "A10",
            "description":     "Meta pretrained — prototypical zero-shot (no fine-tuning)",
            "data_note":       "2kHz EMG-only — compare only against A11/A12, not M0–A9",
            "eval_protocol":   "prototypical nearest-centroid in L2-normalised feature space",
            "test_procedure":  "L2SO",
            "seed":            FIXED_SEED,
            "n_params":        all_results[0]["n_params"],
            "fold_results":    all_results,
            "mean_test_acc":   float(np.mean(test_accs)),
            "std_test_acc":    float(np.std(test_accs)),
            "num_folds":       len(all_results),
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }

    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    if test_procedure == "hpo_test_split":
        print(f"[A10] FINAL zero-shot (hpo_test_split): "
              f"{summary['test_acc']*100:.2f}%  single run, seed={FIXED_SEED}")
    else:
        print(f"[A10] FINAL zero-shot (L2SO): "
              f"{summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
        print(f"     over {summary['num_folds']} L2SO folds (one per test subject)")
    print(f"      ⚠  2kHz EMG-only — compare only against A11/A12")
    print(f"{'='*70}")


def _run_a10_hpo_test_split(config, tensor_dict_path):
    print(f"\n{'='*70}")
    print(f"[A10] hpo_test_split: single run (seed={FIXED_SEED})")
    print(f"{'='*70}")
    set_seeds(FIXED_SEED)
    config_run        = copy.deepcopy(config)
    config_run["seed"] = FIXED_SEED
    model    = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
    model.to(config["device"])
    n_params     = count_parameters(model)
    test_results = prototypical_zeroshot_eval(
        model, config_run, tensor_dict_path, config["test_PIDs"], num_episodes=500,
    )
    print(f"[A10 | seed={FIXED_SEED}] Zero-shot: "
          f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")
    return {"seed": FIXED_SEED, "test_results": test_results, "n_params": n_params}


def _run_a10_l2so(config, tensor_dict_path):
    all_pids = config["all_PIDs"]
    assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."
    folds    = build_l2so_folds(all_pids)
    all_results = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        print(f"\n{'='*70}")
        print(f"[A10] L2SO fold {fold_idx+1}/{len(folds)}  "
              f"test={fold['test_pid']}  val={fold['val_pid']}  (no training phase)")
        print(f"{'='*70}")

        # A10 is zero-shot: backbone never changes, so we just evaluate on
        # this fold's test subject. val_pid / train_pids are unused.
        fold_config               = copy.deepcopy(config)
        fold_config["test_PIDs"]  = [fold["test_pid"]]
        fold_config["val_PIDs"]   = [fold["val_pid"]]
        fold_config["train_PIDs"] = fold["train_pids"]
        fold_config["seed"]       = FIXED_SEED

        set_seeds(FIXED_SEED)
        model = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
        model.to(config["device"])
        n_params = count_parameters(model)

        test_results = prototypical_zeroshot_eval(
            model, fold_config, tensor_dict_path,
            test_pids=[fold["test_pid"]], num_episodes=500,
        )
        print(f"[A10 | fold {fold_idx+1}] test={fold['test_pid']} "
              f"Zero-shot: {test_results['mean_acc']*100:.2f}%")
        all_results.append({
            "fold_id":    f"l2so_fold{fold_idx:02d}_test{fold['test_pid']}",
            "test_PID":   fold["test_pid"],
            "val_PID":    fold["val_pid"],
            "test_results": test_results,
            "n_params":   n_params,
        })

    return all_results


# =============================================================================
# ── A11: Meta pretrained + 1-shot fine-tuning ─────────────────────────────────
# =============================================================================

def run_a11():
    config         = build_config_meta(ablation_id="A11")
    test_procedure = config["test_procedure"]

    print("\nA11 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"\nTest procedure: {test_procedure}")

    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'."
    )

    tensor_dict_path = str(EMG_2KHZ_PKL_PATH)

    if test_procedure == "hpo_test_split":
        result    = _run_a11_hpo_test_split(config, tensor_dict_path)
        summary = {
            "ablation_id":         "A11",
            "description":         "Meta pretrained — 1-shot head-only and full fine-tuning",
            "data_note":           "2kHz EMG-only — compare only against A10/A12, not M0–A9",
            "test_procedure":      "hpo_test_split",
            "seed":                FIXED_SEED,
            "ft_lr":               A11_FT_LR,
            "ft_steps":            A11_FT_STEPS,
            "n_params":            result["n_params"],
            "result":              result,
            "test_head_only_acc":  result["test_head_only"]["mean_acc"],
            "test_full_ft_acc":    result["test_full_ft"]["mean_acc"],
            "config_snapshot":     {k: str(v) for k, v in config.items()},
        }
    else:  # L2SO
        all_results = _run_a11_l2so(config, tensor_dict_path)
        head_accs   = [r["test_head_only"]["mean_acc"] for r in all_results]
        full_accs   = [r["test_full_ft"]["mean_acc"]   for r in all_results]
        summary = {
            "ablation_id":         "A11",
            "description":         "Meta pretrained — 1-shot head-only and full fine-tuning",
            "data_note":           "2kHz EMG-only — compare only against A10/A12, not M0–A9",
            "test_procedure":      "L2SO",
            "seed":                FIXED_SEED,
            "ft_lr":               A11_FT_LR,
            "ft_steps":            A11_FT_STEPS,
            "n_params":            all_results[0]["n_params"],
            "fold_results":        all_results,
            "mean_test_head_only": float(np.mean(head_accs)),
            "std_test_head_only":  float(np.std(head_accs)),
            "mean_test_full_ft":   float(np.mean(full_accs)),
            "std_test_full_ft":    float(np.std(full_accs)),
            "num_folds":           len(all_results),
            "config_snapshot":     {k: str(v) for k, v in config.items()},
        }

    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    if test_procedure == "hpo_test_split":
        print(f"[A11] FINAL head-only (hpo_test_split): "
              f"{summary['test_head_only_acc']*100:.2f}%  single run, seed={FIXED_SEED}")
        print(f"[A11] FINAL full-ft   (hpo_test_split): "
              f"{summary['test_full_ft_acc']*100:.2f}%  single run, seed={FIXED_SEED}")
    else:
        print(f"[A11] FINAL head-only (L2SO): "
              f"{summary['mean_test_head_only']*100:.2f}% ± {summary['std_test_head_only']*100:.2f}%")
        print(f"[A11] FINAL full-ft   (L2SO): "
              f"{summary['mean_test_full_ft']*100:.2f}% ± {summary['std_test_full_ft']*100:.2f}%")
        print(f"     over {summary['num_folds']} L2SO folds (one per test subject)")
    print(f"      ⚠  2kHz EMG-only — compare only against A10/A12")
    print(f"{'='*70}")


def _run_a11_hpo_test_split(config, tensor_dict_path):
    print(f"\n{'='*70}")
    print(f"[A11] hpo_test_split: single run (seed={FIXED_SEED})")
    print(f"      ft_lr={A11_FT_LR}, ft_steps={A11_FT_STEPS}")
    print(f"{'='*70}")
    set_seeds(FIXED_SEED)
    config_run         = copy.deepcopy(config)
    config_run["seed"] = FIXED_SEED
    model = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
    model.to(config["device"])
    n_params     = count_parameters(model)
    head_results = run_supervised_test_eval(
        model, config_run, tensor_dict_path, config["test_PIDs"], ft_mode="head_only",
    )
    full_results = run_supervised_test_eval(
        model, config_run, tensor_dict_path, config["test_PIDs"], ft_mode="full",
    )
    print(f"[A11 | seed={FIXED_SEED}] head-only: "
          f"{head_results['mean_acc']*100:.2f}% ± {head_results['std_acc']*100:.2f}%")
    print(f"[A11 | seed={FIXED_SEED}] full-ft:   "
          f"{full_results['mean_acc']*100:.2f}% ± {full_results['std_acc']*100:.2f}%")
    return {
        "seed":           FIXED_SEED,
        "test_head_only": head_results,
        "test_full_ft":   full_results,
        "n_params":       n_params,
    }


def _run_a11_l2so(config, tensor_dict_path):
    all_pids = config["all_PIDs"]
    assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."
    folds       = build_l2so_folds(all_pids)
    all_results = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        print(f"\n{'='*70}")
        print(f"[A11] L2SO fold {fold_idx+1}/{len(folds)}  "
              f"test={fold['test_pid']}  val={fold['val_pid']}")
        print(f"      ft_lr={A11_FT_LR}, ft_steps={A11_FT_STEPS}")
        print(f"{'='*70}")

        # A11 has no cross-subject training phase (pretrained backbone is fixed).
        # val_pid is recorded for bookkeeping but not used for any training decision.
        fold_config               = copy.deepcopy(config)
        fold_config["test_PIDs"]  = [fold["test_pid"]]
        fold_config["val_PIDs"]   = [fold["val_pid"]]
        fold_config["train_PIDs"] = fold["train_pids"]
        fold_config["seed"]       = FIXED_SEED

        set_seeds(FIXED_SEED)
        model = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
        model.to(config["device"])
        n_params = count_parameters(model)

        head_results = run_supervised_test_eval(
            model, fold_config, tensor_dict_path,
            [fold["test_pid"]], ft_mode="head_only",
        )
        full_results = run_supervised_test_eval(
            model, fold_config, tensor_dict_path,
            [fold["test_pid"]], ft_mode="full",
        )
        print(f"[A11 | fold {fold_idx+1}] test={fold['test_pid']} "
              f"head-only: {head_results['mean_acc']*100:.2f}%  "
              f"full-ft: {full_results['mean_acc']*100:.2f}%")
        all_results.append({
            "fold_id":        f"l2so_fold{fold_idx:02d}_test{fold['test_pid']}",
            "test_PID":       fold["test_pid"],
            "val_PID":        fold["val_pid"],
            "test_head_only": head_results,
            "test_full_ft":   full_results,
            "n_params":       n_params,
        })

    return all_results


# =============================================================================
# ── A12: MAML + MoE on 2kHz EMG-only ─────────────────────────────────────────
# =============================================================================

def build_config_a12() -> dict:
    _check_2khz_data_configured()
    config = make_base_config(ablation_id="A12")
    config["use_imu"]          = False
    config["multimodal"]       = False
    config["use_demographics"] = False
    config["emg_in_ch"]        = EMG_2KHZ_IN_CH
    config["sequence_length"]  = EMG_2KHZ_SEQ_LEN
    config["dfs_load_path"]    = str(Path(EMG_2KHZ_PKL_PATH).parent) + "/"
    config["front_end_stride"] = 20
    return config


def run_a12():
    config         = build_config_a12()
    test_procedure = config["test_procedure"]

    print("\nA12 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"\nTest procedure: {test_procedure}")

    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'."
    )

    tensor_dict_path = str(EMG_2KHZ_PKL_PATH)

    if test_procedure == "hpo_test_split":
        result  = _run_a12_hpo_test_split(config, tensor_dict_path)
        summary = {
            "ablation_id":     "A12",
            "description":     "MAML + MoE on 2kHz EMG-only (apples-to-apples control)",
            "data_note":       "2kHz EMG-only — compare only against A10/A11, not M0–A9",
            "test_procedure":  "hpo_test_split",
            "seed":            FIXED_SEED,
            "n_params":        result["n_params"],
            "result":          result,
            "test_acc":        result["test_results"]["mean_acc"],
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }
    else:  # L2SO
        all_results = _run_a12_l2so(config, tensor_dict_path)
        test_accs   = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":     "A12",
            "description":     "MAML + MoE on 2kHz EMG-only (apples-to-apples control)",
            "data_note":       "2kHz EMG-only — compare only against A10/A11, not M0–A9",
            "test_procedure":  "L2SO",
            "seed":            FIXED_SEED,
            "n_params":        all_results[0]["n_params"],
            "fold_results":    all_results,
            "mean_test_acc":   float(np.mean(test_accs)),
            "std_test_acc":    float(np.std(test_accs)),
            "num_folds":       len(all_results),
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }

    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    if test_procedure == "hpo_test_split":
        print(f"[A12] FINAL (hpo_test_split): "
              f"{summary['test_acc']*100:.2f}%  single run, seed={FIXED_SEED}")
    else:
        print(f"[A12] FINAL (L2SO): "
              f"{summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
        print(f"     over {summary['num_folds']} L2SO folds (one per test subject)")
    print(f"      ⚠  2kHz EMG-only data — compare only against A10/A11, not M0")
    print(f"{'='*70}")


def _run_a12_hpo_test_split(config, tensor_dict_path):
    print(f"\n{'='*70}")
    print(f"[A12] hpo_test_split: single run (seed={FIXED_SEED})")
    print(f"{'='*70}")
    set_seeds(FIXED_SEED)
    config_run         = copy.deepcopy(config)
    config_run["seed"] = FIXED_SEED
    model    = build_maml_moe_model(config_run)
    n_params = count_parameters(model)
    print(f"[A12 | seed={FIXED_SEED}] Parameters: {n_params:,}  "
          f"EMG-only {config['emg_in_ch']}ch @ 2kHz, seq_len={config['sequence_length']}")
    train_dl, val_dl = get_maml_dataloaders(config_run, tensor_dict_path=tensor_dict_path)
    trained_model, train_history = mamlpp_pretrain(
        model, config_run, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[A12 | seed={FIXED_SEED}] Best val acc = {best_val_acc:.4f}")
    save_model_checkpoint(
        {
            "seed":             FIXED_SEED,
            "model_state_dict": train_history["best_state"],
            "config":           config_run,
            "best_val_acc":     best_val_acc,
        },
        config_run,
        tag=f"fixed_seed{FIXED_SEED}_best",
    )
    trained_model.load_state_dict(train_history["best_state"])
    test_results = run_episodic_test_eval(
        trained_model, config_run, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[A12 | seed={FIXED_SEED}] Test: "
          f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")
    return {
        "seed": FIXED_SEED, "best_val_acc": float(best_val_acc),
        "test_results": test_results, "n_params": n_params,
    }


def _run_a12_l2so(config, tensor_dict_path):
    all_pids = config["all_PIDs"]
    assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."
    folds       = build_l2so_folds(all_pids)
    all_results = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        print(f"\n{'='*70}")
        print(f"[A12] L2SO fold {fold_idx+1}/{len(folds)}  "
              f"test={fold['test_pid']}  val={fold['val_pid']}")
        print(f"{'='*70}")

        fold_config               = copy.deepcopy(config)
        fold_config["train_PIDs"] = fold["train_pids"]
        fold_config["val_PIDs"]   = [fold["val_pid"]]
        fold_config["test_PIDs"]  = [fold["test_pid"]]
        fold_config["seed"]       = FIXED_SEED

        set_seeds(FIXED_SEED)
        model    = build_maml_moe_model(fold_config)
        n_params = count_parameters(model)
        print(f"[A12 | fold {fold_idx+1}] Parameters: {n_params:,}")

        train_dl, val_dl = get_maml_dataloaders(fold_config, tensor_dict_path=tensor_dict_path)
        trained_model, train_history = mamlpp_pretrain(
            model, fold_config, train_dl, episodic_val_loader=val_dl,
        )
        best_val_acc = train_history["best_val_acc"]
        print(f"[A12 | fold {fold_idx+1}] Training complete. Best val acc = {best_val_acc:.4f}")

        save_model_checkpoint(
            {
                "fold_id":          f"l2so_fold{fold_idx:02d}_test{fold['test_pid']}",
                "model_state_dict": train_history["best_state"],
                "config":           fold_config,
                "best_val_acc":     best_val_acc,
            },
            fold_config,
            tag=f"l2so_fold{fold_idx:02d}_test{fold['test_pid']}_best",
        )

        trained_model.load_state_dict(train_history["best_state"])
        test_results = run_episodic_test_eval(
            trained_model, fold_config, tensor_dict_path, [fold["test_pid"]]
        )
        print(f"[A12 | fold {fold_idx+1}] test={fold['test_pid']} "
              f"Test: {test_results['mean_acc']*100:.2f}%")
        all_results.append({
            "fold_id":      f"l2so_fold{fold_idx:02d}_test{fold['test_pid']}",
            "test_PID":     fold["test_pid"],
            "val_PID":      fold["val_pid"],
            "best_val_acc": float(best_val_acc),
            "test_results": test_results,
            "n_params":     n_params,
        })

    return all_results


# =============================================================================
# ── Guard ─────────────────────────────────────────────────────────────────────
# =============================================================================

def _check_2khz_data_configured():
    if EMG_2KHZ_PKL_PATH is None:
        raise ValueError(
            "EMG_2KHZ_PKL_PATH is not set. Edit the path near the top of this file."
        )
    if not Path(EMG_2KHZ_PKL_PATH).exists():
        raise FileNotFoundError(
            f"EMG_2KHZ_PKL_PATH does not exist: {EMG_2KHZ_PKL_PATH}"
        )


# =============================================================================
# ── Entry point ───────────────────────────────────────────────────────────────
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation", choices=["A10", "A11", "A12"], required=True,
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