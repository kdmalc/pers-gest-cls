"""
num_eval_steps_sweep.py
=========================
Post-hoc sweep over the number of test-time adaptation steps for trained
checkpoints. No training (except A2, which is trained inline). No HPO.
No Optuna. Pure grid sweep.

Purpose
-------
Answer: "How many adaptation steps does each model need before it plateaus?"

For M0 (MAML) this is run AFTER training. It loads a frozen checkpoint
and sweeps the number of test-time gradient steps over a fixed grid. The
model weights are NEVER modified — each episode starts from the same
checkpoint init.

For A2, training takes ~2 min on the cluster. The model is trained inline
(single seed, FIXED_SEED), then the trained weights are deepcopied and used
for the eval sweep exactly as with A7. The checkpoint is also saved to disk.

Supported model types
---------------------
  M0 (MAML + MoE):
    Sweeps `maml_inner_steps_eval`.
    Also sweeps `maml_alpha_init_eval` jointly if --sweep-alpha is set
    (2D grid: steps x alpha). Use --sweep-alpha to find the best alpha
    first, then fix it with --alpha for the paper-curve run.

  A2 (CNN-LSTM, no MoE — trained inline, no checkpoint needed):
    Trains A2 from scratch at FIXED_SEED using the SAME config and
    param-matched architecture as the canonical A2 ablation. Saves
    checkpoint, then sweeps adaptation steps.
    ft_mode: 'full' by default; override with --ft-mode head_only.
    ft_lr fixed at maml_alpha_init_eval (mirroring the canonical A2).

  A7 (CNN-LSTM, no MoE — loads from checkpoint):
    Sweeps `num_ft_steps` in finetune_and_eval_user().
    --checkpoint required.
    ft_mode: 'full' by default; override with --ft-mode head_only.
    ft_lr mirrors maml_alpha_init_eval (same as canonical A7 config).

  A11 (Meta pretrained — no checkpoint needed):
    Uses MetaEMGWrapper + 2kHz EMG data.
    Config is built via build_config_meta("A11") from
    A10_A11_A12_meta_pretrained.py — exactly the same as the canonical
    A11 ablation. No --checkpoint needed (MetaEMGWrapper loads its own weights).
    ft_lr mirrors maml_alpha_init_eval (same as canonical A11).

Configs / architectures
-----------------------
  A2  : uses build_a2_config() which calls make_base_config + param-matching
        via compute_matched_filters_for_ablation — IDENTICAL to A2_no_maml_no_moe.py.
  A11 : uses build_a11_base_config() which calls build_config_meta("A11") from
        A10_A11_A12_meta_pretrained.py — IDENTICAL to the canonical A11 ablation.
  M0  : loaded from checkpoint; config comes from the checkpoint itself.
  A7  : loaded from checkpoint; config comes from the checkpoint itself.

Eval subjects
-------------
  By default, evaluates over VAL_PIDS + TEST_PIDS from ablation_config (fold 0).
  This is a diagnostic figure (adaptation plateau), NOT a held-out model
  selection decision, so combining val+test gives a more reliable plateau
  estimate with 8 subjects instead of 4. Override with --eval-pids if needed.

Workflow
--------
  # M0 — find best (steps, alpha) jointly
  python num_eval_steps_sweep.py \\
      --model-type M0 \\
      --checkpoint /path/to/M0_best.pt \\
      --ablation-id M0 \\
      --sweep-alpha \\
      --out-dir /scratch/.../steps_sweep/M0

  # M0 — paper curve at fixed best alpha
  python num_eval_steps_sweep.py \\
      --model-type M0 \\
      --checkpoint /path/to/M0_best.pt \\
      --ablation-id M0 \\
      --alpha <best_alpha_from_sweep> \\
      --out-dir /scratch/.../steps_sweep/M0

  # A7 — paper curve (ft_lr defaults to maml_alpha_init_eval from config)
  python num_eval_steps_sweep.py \\
      --model-type A7 \\
      --checkpoint /path/to/A7_best.pt \\
      --ablation-id A7 \\
      --out-dir /scratch/.../steps_sweep/A7

  # A2 — train inline, then paper curve
  python num_eval_steps_sweep.py \\
      --model-type A2 \\
      --ablation-id A2 \\
      --out-dir /scratch/.../steps_sweep/A2

  # A11 — paper curve (no checkpoint needed)
  python num_eval_steps_sweep.py \\
      --model-type A11 \\
      --ablation-id A11 \\
      --out-dir /scratch/.../steps_sweep/A11

Output
------
  <out_dir>/steps_sweep_<ablation_id>_<mode>_<timestamp>.json
  Partial results are written after every evaluated configuration —
  preemption-safe.

SLURM
-----
  Single-GPU job, no array needed. Estimated wall times:
    M0 2D sweep   (8 steps x 9 alphas x ~2 min):  ~2-3h
    M0 paper curve (10 steps x ~2.5 min):          ~25 min
    A2/A7 paper curve (10 steps x ~5 min):        ~50 min + 2 min training for A2
    A11 similar to A2 (no-MoE backbone).

Notes on comparability
----------------------
  MAML and transfer learning are NOT doing the same thing per step —
  MAML has LSLR per-parameter rates learned to be fast-adapting, while
  the supervised model uses a fixed LR over a backbone trained on all
  classes. "Equal steps" is NOT the right comparability criterion.

  The correct approach (implemented here) is: find each method's plateau
  independently, then compare best-case performance. If MAML plateaus
  faster, that IS a contribution (sample efficiency in adaptation) and
  should be highlighted, not hidden.
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# Paths — set from env vars (SLURM launcher) or fall through to cwd
# =============================================================================

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()

for _p in [
    CODE_DIR,
    CODE_DIR / "system",
    CODE_DIR / "system" / "MAML",
    CODE_DIR / "system" / "MOE",
    CODE_DIR / "system" / "pretraining",
]:
    sys.path.insert(0, str(_p))

# A11 uses Meta's neuromotor repo — only needed for --model-type A11
NEUROMOTOR_REPO = Path(os.environ.get(
    "NEUROMOTOR_REPO",
    "/projects/my13/div-emg/generic-neuromotor-interface",
)).resolve()
sys.path.insert(0, str(NEUROMOTOR_REPO))

print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"CODE_DIR       : {CODE_DIR}")
print(f"RUN_DIR        : {RUN_DIR}")

# =============================================================================
# A11 constants — only used when --model-type A11
# =============================================================================

META_CHECKPOINT_PATH = Path(
    "/rhf/allocations/my13/emg_models/discrete_gestures/model_checkpoint.ckpt"
)
EMG_2KHZ_PKL_PATH = (
    "/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/"
    "meta-learning-sup-que-ds/segfilt_2khz_emg_tensor_dict.pkl"
)
EMG_2KHZ_IN_CH   = 16
EMG_2KHZ_SEQ_LEN = 4300

# =============================================================================
# Step / LR / alpha grids
# =============================================================================

# M0 (MAML): paper figure trajectory — extend if plateau not yet reached.
MAML_PAPER_STEPS_GRID = [0, 1, 3, 5, 10, 20, 30, 50]

# M0 (MAML): 2D sweep grid for finding (steps, alpha) jointly.
MAML_HPO_STEPS_GRID   = [0, 1, 3, 5, 10]
MAML_ALPHA_GRID       = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030]

# Supervised (A2, A7, A11): paper figure trajectory.
SUP_PAPER_STEPS_GRID  = [0, 1, 3, 5, 10, 25, 50, 100, 150, 200]

# Supervised: 2D sweep grid for finding (ft_steps, ft_lr) jointly.
SUP_HPO_STEPS_GRID    = [0, 1, 3, 5, 10, 25, 50, 100, 150, 200]
SUP_LR_GRID           = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0]

NUM_VAL_EPISODES      = 200
FIXED_SEED            = 42

# =============================================================================
# MetaEMGWrapper (A11 only)
# =============================================================================

class MetaEMGWrapper(nn.Module):
    """
    Wraps Meta's DiscreteGesturesArchitecture for our ablation eval pipeline.
    Identical to the implementation in A10_A11_A12_meta_pretrained.py.
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
        self.head = self.network.projection

        if freeze_backbone:
            self._freeze_backbone()

    def _load_checkpoint(self, checkpoint_path: Path):
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), (
            f"Checkpoint not found: {checkpoint_path}"
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert "state_dict" in ckpt, (
            f"Expected 'state_dict' key in checkpoint. Got: {list(ckpt.keys())}"
        )

        raw_sd   = ckpt["state_dict"]
        PREFIX   = "network."
        stripped = {}
        for k, v in raw_sd.items():
            assert k.startswith(PREFIX), (
                f"Unexpected key in checkpoint (no '{PREFIX}' prefix): {k!r}"
            )
            stripped[k[len(PREFIX):]] = v

        missing, unexpected = self.network.load_state_dict(stripped, strict=True)
        assert len(missing) == 0,    f"Missing keys after load: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys after load: {unexpected}"
        print(f"[MetaEMGWrapper] Loaded weights from {checkpoint_path}")

    def _freeze_backbone(self):
        for param in self.network.parameters():
            param.requires_grad_(False)
        print("[MetaEMGWrapper] Backbone frozen.")

    def forward(self, x_emg: torch.Tensor, x_imu=None, demographics=None) -> torch.Tensor:
        features         = self._backbone_forward(x_emg)    # (B, T_out, 512)
        logits_per_frame = self.head(features)               # (B, T_out, n_classes)
        return logits_per_frame.mean(dim=1)                  # (B, n_classes)

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.network
        x   = net.compression(x)
        x   = net.conv_layer(x)
        x   = net.relu(x)
        x   = net.dropout(x)
        x   = x.transpose(1, 2)
        x   = net.post_conv_layer_norm(x)
        x, _= net.lstm(x)
        x   = net.post_lstm_layer_norm(x)
        return x


# =============================================================================
# Checkpoint loaders (M0/MAML and supervised/A7)
# =============================================================================

def load_maml_checkpoint(checkpoint_path: Path) -> tuple:
    """
    Load a trained MAML checkpoint (M0). Returns (model, config).

    Expected checkpoint keys:
        checkpoint["model_state_dict"]
        checkpoint["config"]
    """
    print(f"Loading MAML checkpoint: {checkpoint_path}")
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    use_moe = config["use_MOE"]
    if use_moe:
        from MOE.MOE_encoder import build_MOE_model
        model = build_MOE_model(config)
    else:
        from pretraining.pretrain_models import build_model
        model = build_model(config)

    # Recreate LSLR (monkey-patched at train time) so load_state_dict accepts its keys
    if config.get("maml_use_lslr", False):
        from MAML.mamlpp import PerParamPerStepLSLR, named_param_dict
        temp_params = named_param_dict(model, require_grad_only=True)
        model._lslr = PerParamPerStepLSLR(
            named_params = temp_params.items(),
            inner_steps  = config["maml_inner_steps"],
            init_lr      = config["maml_alpha_init"],
            learnable    = True,
            device       = device,
        ).to(device)

    # ── Key remapping: ctx_proj -> router.projector ───────────────────────────
    raw_sd = ckpt["model_state_dict"]
    remapped_sd = {}
    n_remapped = 0
    for k, v in raw_sd.items():
        new_k = k.replace("ctx_proj.", "router.projector.")
        new_k = new_k.replace("ctx_proj-", "router-projector-")
        if new_k != k:
            n_remapped += 1
        remapped_sd[new_k] = v
    if n_remapped > 0:
        print(f"  [ckpt remap] Remapped {n_remapped} keys: ctx_proj -> router.projector")

    model.load_state_dict(remapped_sd)
    model.to(device)
    model.eval()

    print(f"  Best val acc    : {ckpt.get('best_val_acc', 'N/A')}")
    print(f"  Trained steps   : {config.get('maml_inner_steps')}")
    print(f"  Trained alpha_e : {config.get('maml_alpha_init_eval')}")
    return model, config


def load_supervised_checkpoint(checkpoint_path: Path) -> tuple:
    """
    Load a trained supervised (non-MAML) checkpoint (A7). Returns (model, config).

    Expected checkpoint keys:
        checkpoint["model_state_dict"]
        checkpoint["config"]
    """
    print(f"Loading supervised checkpoint: {checkpoint_path}")
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    use_moe = config["use_MOE"]
    if use_moe:
        from MOE.MOE_encoder import build_MOE_model
        model = build_MOE_model(config)
    else:
        from pretraining.pretrain_models import build_model
        model = build_model(config)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


# =============================================================================
# A2 config + inline training
# =============================================================================

def build_a2_config() -> dict:
    """
    Build the A2 config for the steps sweep.

    This is IDENTICAL to the canonical A2_no_maml_no_moe.py config:
      - Calls make_base_config (inheriting all M0 Trial 89 hyperparameters)
      - Disables meta_learning and use_MOE
      - Runs compute_matched_filters_for_ablation so cnn_base_filters is
        param-matched to M0's full expert pool (same as the ablation table A2)
      - Sets ft_* to mirror M0's MAML inner-loop eval (same as canonical A2)

    The sweep overrides ft_steps per grid point — ft_lr is fixed at the
    canonical value (maml_alpha_init_eval) and NOT swept, since the paper
    curve should use the same LR as the canonical A2 ablation result.
    """
    from ablation_config import (
        make_base_config, compute_matched_filters_for_ablation,
    )
    config = make_base_config(ablation_id="A2")
    config["meta_learning"] = False
    config["use_MOE"]       = False
    config["batch_size"]    = 64
    config["train_reps"]    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["val_reps"]      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["augment"]       = False

    # ── Param matching: single encoder ≈ ALL M0 experts combined ─────────────
    # This is the same call as in A2_no_maml_no_moe.py. Without it, the A2
    # being swept here would be a different (smaller) model than the one in the
    # ablation table — making the curve meaningless for comparison.
    match_info = compute_matched_filters_for_ablation(
        ablation_id="A2",
        ablation_config=config,
        match_target="all_experts",
    )
    config["cnn_base_filters"] = match_info["matched_filters"]

    # Stash for auditing (mirrors A2_no_maml_no_moe.py)
    config["_param_match_target"]          = "all_experts_cnn"
    config["_m0_total_params"]             = match_info["m0_total_params"]
    config["_m0_all_expert_params"]        = match_info["m0_all_expert_params"]
    config["_m0_one_expert_params"]        = match_info["m0_one_expert_params"]
    config["_a2_matched_cnn_params"]       = match_info["matched_cnn_params"]
    config["_a2_total_params_after_match"] = match_info["matched_total_params"]
    config["_a2_param_ratio"]              = match_info["param_ratio"]

    # ── Eval-time adaptation: mirror M0's MAML inner-loop eval exactly ────────
    # Same as canonical A2_no_maml_no_moe.py: ft_steps is overridden per grid
    # point during the sweep; ft_lr, ft_optimizer, ft_weight_decay are fixed.
    config["ft_steps"]        = config["maml_inner_steps_eval"]  # = 10; overridden in sweep
    config["ft_lr"]           = config["maml_alpha_init_eval"]   # = 5.066e-3
    config["ft_optimizer"]    = "sgd"    # mirrors MAML inner-loop update rule
    config["ft_weight_decay"] = 0.0     # MAML inner loop has no weight decay

    return config


def train_a2_model(out_dir: Path) -> tuple:
    """
    Train A2 from scratch at FIXED_SEED using the param-matched config.
    Saves checkpoint to out_dir. Returns (trained_model, config, tensor_dict_path).

    The config and architecture are IDENTICAL to the canonical A2_no_maml_no_moe.py
    run — the only difference is we don't do a final test eval here, since the
    sweep handles that.
    """
    from ablation_config import (
        build_supervised_no_moe_model, set_seeds, save_model_checkpoint,
        count_parameters,
    )
    from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
    from pretraining.pretrain_trainer import pretrain
    from MAML.maml_data_pipeline import reorient_tensor_dict

    config = build_a2_config()
    config["seed"] = FIXED_SEED
    set_seeds(FIXED_SEED)

    print(f"\n{'='*70}")
    print(f"[A2] Training inline at seed={FIXED_SEED}")
    print(f"  train_PIDs         : {config['train_PIDs']}")
    print(f"  val_PIDs           : {config['val_PIDs']}")
    print(f"  cnn_base_filters   : {config['cnn_base_filters']}  (param-matched to M0 all experts)")
    print(f"  _a2_param_ratio    : {config['_a2_param_ratio']:.4f}  (should be ~1.0)")
    print(f"{'='*70}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    model = build_supervised_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Parameters : {n_params:,}")

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert n_classes == config["pretrain_num_classes"], (
        f"Expected {config['pretrain_num_classes']} classes, got {n_classes}."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else float("nan")
    print(f"[A2] Training complete. Best val acc = {best_val_acc:.4f}")

    ckpt_state = {
        "fold_id":                     f"steps_sweep_seed{FIXED_SEED}",
        "seed":                        FIXED_SEED,
        "model_state_dict":            trained_model.state_dict(),
        "config":                      config,
        "best_val_acc":                best_val_acc,
        "train_loss_log":              history["train_loss"],
        "val_acc_log":                 history["val_acc"],
        "cnn_base_filters":            config["cnn_base_filters"],
        "_a2_param_ratio":             config["_a2_param_ratio"],
        "_m0_all_expert_params":       config["_m0_all_expert_params"],
        "_a2_matched_cnn_params":      config["_a2_matched_cnn_params"],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_path = out_dir / f"A2_steps_sweep_trained_{timestamp}.pt"
    torch.save(ckpt_state, ckpt_path)
    print(f"[A2] Checkpoint saved to: {ckpt_path}")

    return trained_model, config, tensor_dict_path


# =============================================================================
# A11 base config builder
# =============================================================================

def build_a11_base_config() -> dict:
    """
    Build the A11 config for the steps sweep by calling build_config_meta("A11")
    from A10_A11_A12_meta_pretrained.py.

    This is IDENTICAL to the canonical A11 ablation config — same ft_lr,
    ft_optimizer, ft_weight_decay, ft_label_smooth. The sweep overrides
    ft_steps and ft_lr per grid point, but the base config ensures everything
    else (optimizer, weight decay, data format) matches the ablation table.

    NOTE: build_config_meta calls make_base_config internally, which calls
    _check_2khz_data_configured() — EMG_2KHZ_PKL_PATH must exist on the node.
    """
    # Import from the canonical A11 file to guarantee config identity.
    # We can't just import build_config_meta directly because A10_A11_A12_meta_pretrained.py
    # is in the ablations directory, not the system directory.
    _a11_module_path = str(
        CODE_DIR / "system" / "NOTS" / "paper" / "ablations" / "test_eval_files"
    )
    if _a11_module_path not in sys.path:
        sys.path.insert(0, _a11_module_path)

    from A10_A11_A12_meta_pretrained import build_config_meta
    config = build_config_meta(ablation_id="A11")
    return config


# =============================================================================
# Single-config evaluators
# =============================================================================

def eval_maml_one_config(
    model:            torch.nn.Module,
    base_config:      dict,
    inner_steps_eval: int,
    alpha_eval:       float,
    eval_pids:        list,
    tensor_dict_path: str,
    num_val_episodes: int,
) -> dict:
    """
    Episodic val eval for one (inner_steps_eval, alpha_eval) pair.
    Model weights are never modified.
    use_lslr_at_eval is forced False — the LSLR was trained for a specific
    step count; using it at a different count is invalid.
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from MAML.mamlpp import mamlpp_adapt_and_eval
    from torch.utils.data import DataLoader

    eval_config = copy.deepcopy(base_config)
    eval_config["maml_inner_steps_eval"] = inner_steps_eval
    eval_config["maml_alpha_init_eval"]  = alpha_eval
    eval_config["use_lslr_at_eval"]      = False  # see docstring
    eval_config["val_PIDs"]              = eval_pids

    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = eval_pids,
        target_gesture_classes  = eval_config["maml_gesture_classes"],
        target_trial_reps       = eval_config["target_trial_reps"],
        n_way                   = eval_config["n_way"],
        k_shot                  = eval_config["k_shot"],
        q_query                 = eval_config["q_query"],
        num_eval_episodes       = num_val_episodes,
        is_train                = False,
        seed                    = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=maml_mm_collate)

    model.eval()
    user_accs   = defaultdict(list)
    user_losses = defaultdict(list)

    for batch in val_dl:
        uid     = batch["user_id"]
        metrics = mamlpp_adapt_and_eval(
            model, eval_config, batch["support"], batch["query"]
        )
        user_accs[uid].append(float(metrics["acc"]))
        if "loss" in metrics:
            user_losses[uid].append(float(metrics["loss"]))

    per_user_acc  = {uid: float(np.mean(accs)) for uid, accs in user_accs.items()}
    per_user_loss = {uid: float(np.mean(ls))   for uid, ls   in user_losses.items() if ls}

    all_accs   = list(per_user_acc.values())
    all_losses = list(per_user_loss.values())

    return {
        "inner_steps_eval": inner_steps_eval,
        "alpha_eval":       alpha_eval,
        "mean_acc":         float(np.mean(all_accs)),
        "std_acc":          float(np.std(all_accs)),
        "mean_loss":        float(np.mean(all_losses)) if all_losses else None,
        "std_loss":         float(np.std(all_losses))  if all_losses else None,
        "per_user_acc":     per_user_acc,
        "per_user_loss":    per_user_loss,
    }


def eval_supervised_one_config(
    model:            torch.nn.Module,
    base_config:      dict,
    ft_steps:         int,
    ft_lr:            float,
    ft_mode:          str,
    eval_pids:        list,
    tensor_dict_path: str,
    num_val_episodes: int,
) -> dict:
    """
    Episodic finetuning eval for one (ft_steps, ft_lr) pair (A2 and A7).
    A fresh copy of the model head is used per episode (finetune_and_eval_user
    handles this internally via deepcopy). Backbone weights from the checkpoint
    are never modified.

    ft_optimizer is taken from base_config (set to "sgd" to mirror MAML inner-loop).
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from pretraining.pretrain_finetune import finetune_and_eval_user
    from torch.utils.data import DataLoader

    eval_config = copy.deepcopy(base_config)
    eval_config["ft_steps"]        = ft_steps
    eval_config["num_ft_steps"]    = ft_steps
    eval_config["ft_lr"]           = ft_lr
    eval_config["ft_label_smooth"] = 0.0
    eval_config["ft_weight_decay"] = 0.0
    # Use the optimizer from base_config ("sgd" for A2/A7, matching canonical ablations).
    # Do NOT hardcode "adam" here — that would diverge from the canonical A2/A7 setup.
    eval_config["val_PIDs"]        = eval_pids

    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = eval_pids,
        target_gesture_classes  = eval_config["maml_gesture_classes"],
        target_trial_reps       = eval_config["target_trial_reps"],
        n_way                   = eval_config["n_way"],
        k_shot                  = eval_config["k_shot"],
        q_query                 = eval_config["q_query"],
        num_eval_episodes       = num_val_episodes,
        is_train                = False,
        seed                    = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=maml_mm_collate)

    model.eval()
    user_accs = defaultdict(list)

    for batch in val_dl:
        uid     = batch["user_id"]
        support = batch["support"]
        query   = batch["query"]
        metrics = finetune_and_eval_user(
            model, eval_config,
            support_emg    = support["emg"],
            support_imu    = support.get("imu"),
            support_labels = support["labels"],
            query_emg      = query["emg"],
            query_imu      = query.get("imu"),
            query_labels   = query["labels"],
            mode           = ft_mode,
        )
        user_accs[uid].append(float(metrics["acc"]))

    per_user_acc = {uid: float(np.mean(accs)) for uid, accs in user_accs.items()}
    all_accs     = list(per_user_acc.values())

    return {
        "ft_steps":    ft_steps,
        "ft_lr":       ft_lr,
        "ft_mode":     ft_mode,
        "mean_acc":    float(np.mean(all_accs)),
        "std_acc":     float(np.std(all_accs)),
        "per_user_acc": per_user_acc,
    }


def eval_a11_one_config(
    ft_steps:         int,
    ft_lr:            float,
    ft_mode:          str,
    eval_pids:        list,
    num_val_episodes: int,
    a11_base_config:  dict,
) -> dict:
    """
    Episodic finetuning eval for A11 (Meta pretrained) for one (ft_steps, ft_lr) pair.
    The MetaEMGWrapper is reconstructed fresh per config call — stateless by design.

    ft_optimizer is taken from a11_base_config (set to "sgd" to mirror the
    canonical A11 ablation, NOT hardcoded to "adam").
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from pretraining.pretrain_finetune import finetune_and_eval_user
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
    model.to(device)

    eval_config = copy.deepcopy(a11_base_config)
    eval_config["ft_steps"]           = ft_steps
    eval_config["num_ft_steps"]       = ft_steps
    eval_config["ft_lr"]              = ft_lr
    eval_config["ft_label_smooth"]    = 0.0
    eval_config["ft_weight_decay"]    = 0.0
    # ft_optimizer comes from a11_base_config (built via build_config_meta,
    # which sets it to "sgd" — matching the canonical A11 ablation).
    eval_config["val_PIDs"]           = eval_pids
    eval_config["num_eval_episodes"]  = num_val_episodes
    eval_config["device"]             = device

    with open(EMG_2KHZ_PKL_PATH, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = eval_pids,
        target_gesture_classes  = eval_config["maml_gesture_classes"],
        target_trial_reps       = eval_config["target_trial_reps"],
        n_way                   = eval_config["n_way"],
        k_shot                  = eval_config["k_shot"],
        q_query                 = eval_config["q_query"],
        num_eval_episodes       = num_val_episodes,
        is_train                = False,
        seed                    = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=maml_mm_collate)

    user_accs = defaultdict(list)
    for batch in val_dl:
        uid     = batch["user_id"]
        support = batch["support"]
        query   = batch["query"]
        metrics = finetune_and_eval_user(
            model, eval_config,
            support_emg    = support["emg"],
            support_imu    = None,            # Meta model is EMG-only
            support_labels = support["labels"],
            query_emg      = query["emg"],
            query_imu      = None,
            query_labels   = query["labels"],
            mode           = ft_mode,
        )
        user_accs[uid].append(float(metrics["acc"]))

    per_user_acc = {uid: float(np.mean(accs)) for uid, accs in user_accs.items()}
    all_accs     = list(per_user_acc.values())

    return {
        "ft_steps":     ft_steps,
        "ft_lr":        ft_lr,
        "ft_mode":      ft_mode,
        "mean_acc":     float(np.mean(all_accs)),
        "std_acc":      float(np.std(all_accs)),
        "per_user_acc": per_user_acc,
    }


# =============================================================================
# Top-level sweep runners
# =============================================================================

def run_maml_sweep(
    checkpoint_path:  Path,
    ablation_id:      str,
    out_dir:          Path,
    eval_pids:        list,
    num_val_episodes: int,
    sweep_alpha:      bool,
    fixed_alpha:      float | None,
) -> None:
    """
    M0 (MAML) adaptation steps sweep.

    If sweep_alpha=True  → 2D grid: MAML_HPO_STEPS_GRID x MAML_ALPHA_GRID.
    If sweep_alpha=False → paper curve: MAML_PAPER_STEPS_GRID at fixed_alpha.
    """
    if not sweep_alpha:
        assert fixed_alpha is not None, \
            "--alpha must be provided when not using --sweep-alpha."

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag  = "2d_sweep" if sweep_alpha else "paper_curve"

    model, base_config = load_maml_checkpoint(checkpoint_path)
    if base_config.get("target_trial_reps", None) is None:
        base_config["target_trial_reps"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tensor_dict_path = os.path.join(
        base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
    )

    trained_steps = int(base_config.get("maml_inner_steps", 10))
    trained_alpha = float(base_config.get("maml_alpha_init_eval", 0.0))

    if sweep_alpha:
        steps_grid = MAML_HPO_STEPS_GRID
        alpha_grid = MAML_ALPHA_GRID
    else:
        steps_grid = MAML_PAPER_STEPS_GRID
        alpha_grid = [fixed_alpha]

    grid      = [(s, a) for s in steps_grid for a in alpha_grid]
    n_configs = len(grid)

    print(f"\n{'='*70}")
    print(f"M0 (MAML) Adaptation Steps Sweep — {ablation_id} ({mode_tag})")
    print(f"  Checkpoint     : {checkpoint_path}")
    print(f"  Eval PIDs      : {eval_pids}  ({len(eval_pids)} subjects)")
    print(f"  Val episodes   : {num_val_episodes}")
    print(f"  Trained steps  : {trained_steps}")
    print(f"  Trained alpha  : {trained_alpha:.6f}")
    print(f"  Steps grid     : {steps_grid}")
    print(f"  Alpha grid     : {alpha_grid}")
    print(f"  Total configs  : {n_configs}")
    print(f"  Output dir     : {out_dir}")
    print(f"{'='*70}\n")

    results          = []
    best_acc         = -1.0
    best_steps       = None
    best_alpha_found = None
    sweep_start      = time.time()
    out_path         = out_dir / f"steps_sweep_{ablation_id}_{mode_tag}_{timestamp}.json"

    for i, (steps, alpha) in enumerate(grid):
        t0 = time.time()
        print(f"[{i+1:>3}/{n_configs}] steps={steps:>4}, alpha={alpha:.5f} ...",
              end="", flush=True)

        result = eval_maml_one_config(
            model            = model,
            base_config      = base_config,
            inner_steps_eval = steps,
            alpha_eval       = alpha,
            eval_pids        = eval_pids,
            tensor_dict_path = tensor_dict_path,
            num_val_episodes = num_val_episodes,
        )
        elapsed = time.time() - t0
        print(f"  acc={result['mean_acc']*100:.2f}%  ({elapsed:.1f}s)")
        results.append(result)

        if result["mean_acc"] > best_acc:
            best_acc         = result["mean_acc"]
            best_steps       = steps
            best_alpha_found = alpha

        partial_output = {
            "ablation_id":          ablation_id,
            "model_type":           "M0",
            "mode":                 mode_tag,
            "checkpoint":           str(checkpoint_path),
            "eval_pids":            eval_pids,
            "num_eval_pids":        len(eval_pids),
            "num_val_episodes":     num_val_episodes,
            "trained_steps":        trained_steps,
            "trained_alpha":        trained_alpha,
            "steps_grid":           steps_grid,
            "alpha_grid":           alpha_grid,
            "fixed_seed":           FIXED_SEED,
            "n_configs_total":      n_configs,
            "n_configs_done":       i + 1,
            "best_steps_so_far":    best_steps,
            "best_alpha_so_far":    best_alpha_found,
            "best_mean_acc_so_far": best_acc,
            "results":              results,
        }
        with open(out_path, "w") as f:
            json.dump(partial_output, f, indent=2)

    total_elapsed = time.time() - sweep_start
    print(f"\n{'='*70}")
    print(f"[{ablation_id}] {mode_tag} complete in {total_elapsed/60:.1f} min")
    print(f"  Best steps : {best_steps}")
    print(f"  Best alpha : {best_alpha_found:.5f}")
    print(f"  Best acc   : {best_acc*100:.2f}%")
    print(f"  Results    : {out_path}")
    print(f"{'='*70}")

    if sweep_alpha:
        print(f"\nNext step — generate paper curve with best alpha:")
        print(f"  python num_eval_steps_sweep.py \\")
        print(f"      --model-type M0 \\")
        print(f"      --checkpoint {checkpoint_path} \\")
        print(f"      --ablation-id {ablation_id} \\")
        print(f"      --alpha {best_alpha_found:.6f} \\")
        print(f"      --out-dir {out_dir}")


def run_supervised_sweep(
    checkpoint_path:  Path | None,
    ablation_id:      str,
    out_dir:          Path,
    eval_pids:        list,
    num_val_episodes: int,
    sweep_lr:         bool,
    fixed_lr:         float,
    ft_mode:          str,
    model_type:       str,          # "A2", "A7", or "A11"
    a11_base_config:  dict | None,
    trained_model:    torch.nn.Module | None,
    trained_config:   dict | None,
    trained_tensor_dict_path: str | None,
) -> None:
    """
    Supervised (non-MAML) adaptation steps sweep for A7, A2, and A11.

    If sweep_lr=True  → 2D grid: SUP_HPO_STEPS_GRID x SUP_LR_GRID.
    If sweep_lr=False → paper curve: SUP_PAPER_STEPS_GRID at fixed_lr.

    For A2: model, config, and tensor_dict_path come from train_a2_model()
            and are passed in directly.
    For A7: loaded from checkpoint_path.
    For A11: loaded fresh per config call inside eval_a11_one_config().
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag  = "2d_sweep" if sweep_lr else "paper_curve"

    if model_type == "A7":
        assert checkpoint_path is not None, \
            "--checkpoint is required for --model-type A7."
        model, base_config = load_supervised_checkpoint(checkpoint_path)
        tensor_dict_path   = os.path.join(
            base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
        )
    elif model_type == "A2":
        assert trained_model is not None, "trained_model must be provided for A2."
        assert trained_config is not None, "trained_config must be provided for A2."
        assert trained_tensor_dict_path is not None, \
            "trained_tensor_dict_path must be provided for A2."
        model            = trained_model
        base_config      = trained_config
        tensor_dict_path = trained_tensor_dict_path
    else:  # A11
        assert a11_base_config is not None
        model            = None
        base_config      = a11_base_config
        tensor_dict_path = EMG_2KHZ_PKL_PATH

    if sweep_lr:
        steps_grid = SUP_HPO_STEPS_GRID
        lr_grid    = SUP_LR_GRID
    else:
        steps_grid = SUP_PAPER_STEPS_GRID
        lr_grid    = [fixed_lr]

    grid      = [(s, lr) for s in steps_grid for lr in lr_grid]
    n_configs = len(grid)

    print(f"\n{'='*70}")
    print(f"Supervised Adaptation Steps Sweep — {ablation_id} ({mode_tag})")
    if checkpoint_path:
        print(f"  Checkpoint     : {checkpoint_path}")
    print(f"  Model type     : {model_type}")
    print(f"  FT mode        : {ft_mode}")
    print(f"  FT LR          : {fixed_lr if not sweep_lr else 'sweep'}")
    print(f"  FT optimizer   : {base_config.get('ft_optimizer', 'N/A (A11 resolved below)')}")
    print(f"  Eval PIDs      : {eval_pids}  ({len(eval_pids)} subjects)")
    print(f"  Val episodes   : {num_val_episodes}")
    print(f"  Steps grid     : {steps_grid}")
    print(f"  LR grid        : {lr_grid}")
    print(f"  Total configs  : {n_configs}")
    print(f"  Output dir     : {out_dir}")
    print(f"{'='*70}\n")

    results       = []
    best_acc      = -1.0
    best_steps    = None
    best_lr_found = None
    sweep_start   = time.time()
    out_path      = out_dir / f"steps_sweep_{ablation_id}_{mode_tag}_{timestamp}.json"

    for i, (steps, lr) in enumerate(grid):
        t0 = time.time()
        print(f"[{i+1:>3}/{n_configs}] ft_steps={steps:>4}, ft_lr={lr:.4f} ...",
              end="", flush=True)

        if model_type in ("A2", "A7"):
            result = eval_supervised_one_config(
                model            = model,
                base_config      = base_config,
                ft_steps         = steps,
                ft_lr            = lr,
                ft_mode          = ft_mode,
                eval_pids        = eval_pids,
                tensor_dict_path = tensor_dict_path,
                num_val_episodes = num_val_episodes,
            )
        else:   # A11
            result = eval_a11_one_config(
                ft_steps         = steps,
                ft_lr            = lr,
                ft_mode          = ft_mode,
                eval_pids        = eval_pids,
                num_val_episodes = num_val_episodes,
                a11_base_config  = base_config,
            )

        elapsed = time.time() - t0
        print(f"  acc={result['mean_acc']*100:.2f}%  ({elapsed:.1f}s)")
        results.append(result)

        if result["mean_acc"] > best_acc:
            best_acc      = result["mean_acc"]
            best_steps    = steps
            best_lr_found = lr

        partial_output = {
            "ablation_id":          ablation_id,
            "model_type":           model_type,
            "ft_mode":              ft_mode,
            "mode":                 mode_tag,
            "checkpoint":           str(checkpoint_path) if checkpoint_path else (
                                        "trained_inline" if model_type == "A2"
                                        else "MetaEMGWrapper"
                                    ),
            "eval_pids":            eval_pids,
            "num_eval_pids":        len(eval_pids),
            "num_val_episodes":     num_val_episodes,
            "steps_grid":           steps_grid,
            "lr_grid":              lr_grid,
            "fixed_seed":           FIXED_SEED,
            "n_configs_total":      n_configs,
            "n_configs_done":       i + 1,
            "best_steps_so_far":    best_steps,
            "best_lr_so_far":       best_lr_found,
            "best_mean_acc_so_far": best_acc,
            "results":              results,
        }
        with open(out_path, "w") as f:
            json.dump(partial_output, f, indent=2)

    total_elapsed = time.time() - sweep_start
    print(f"\n{'='*70}")
    print(f"[{ablation_id}] {mode_tag} complete in {total_elapsed/60:.1f} min")
    print(f"  Best steps : {best_steps}")
    print(f"  Best LR    : {best_lr_found:.4f}")
    print(f"  Best acc   : {best_acc*100:.2f}%")
    print(f"  Results    : {out_path}")
    print(f"{'='*70}")

    if sweep_lr:
        print(f"\nNext step — generate paper curve with best LR:")
        print(f"  python num_eval_steps_sweep.py \\")
        print(f"      --model-type {model_type} \\")
        if checkpoint_path:
            print(f"      --checkpoint {checkpoint_path} \\")
        print(f"      --ablation-id {ablation_id} \\")
        print(f"      --ft-lr {best_lr_found:.6f} \\")
        print(f"      --ft-mode {ft_mode} \\")
        print(f"      --out-dir {out_dir}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc adaptation steps sweep for M0 (MAML), A2 (trained inline), "
            "A7 (supervised checkpoint), and A11 (Meta pretrained)."
        )
    )

    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=["M0", "A2", "A7", "A11"],
        dest="model_type",
        help=(
            "M0  : MAML + MoE checkpoint. Requires --checkpoint. "
            "A2  : Vanilla CNN-LSTM (no MoE), trained inline. No --checkpoint needed. "
            "A7  : Supervised CNN-LSTM checkpoint. Requires --checkpoint. "
            "A11 : Meta pretrained EMG model. No --checkpoint needed."
        ),
    )
    parser.add_argument(
        "--ablation-id", type=str, required=True, dest="ablation_id",
        help="e.g. M0, A2, A7, A11. Used for output file naming only.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, dest="checkpoint",
        help=(
            "Path to .pt checkpoint. Required for --model-type M0 and A7. "
            "Ignored for A2 and A11."
        ),
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, dest="out_dir",
        help="Output directory. Defaults to RUN_DIR env var.",
    )
    parser.add_argument(
        "--eval-pids", type=str, nargs="+", default=None, dest="eval_pids",
        help=(
            "PIDs to evaluate on. Defaults to VAL_PIDS + TEST_PIDS from ablation_config "
            "(fold-0 val + test, 8 subjects total). This is a diagnostic figure — using "
            "more subjects gives a more reliable plateau estimate."
        ),
    )
    parser.add_argument(
        "--num-val-episodes", type=int, default=NUM_VAL_EPISODES, dest="num_val_episodes",
        help=f"Number of eval episodes per config. Default: {NUM_VAL_EPISODES}.",
    )
    parser.add_argument(
        "--ft-mode", type=str, default="full",
        choices=["head_only", "full"], dest="ft_mode",
        help="Finetuning mode for A2 / A7 / A11. Default: full.",
    )

    # ── M0-specific ───────────────────────────────────────────────────────────
    maml_group = parser.add_argument_group("M0 (MAML) options")
    maml_mode  = maml_group.add_mutually_exclusive_group()
    maml_mode.add_argument(
        "--sweep-alpha", action="store_true", dest="sweep_alpha",
        help=(
            "M0 2D sweep: MAML_HPO_STEPS_GRID x MAML_ALPHA_GRID. "
            "Run this first to find the best alpha."
        ),
    )
    maml_group.add_argument(
        "--alpha", type=float, default=None, dest="alpha",
        help=(
            "Fixed maml_alpha_init_eval for the M0 paper curve. "
            "Take best_alpha_so_far from the 2D sweep JSON."
        ),
    )

    # ── A2 / A7 / A11 specific ───────────────────────────────────────────────
    sup_group = parser.add_argument_group("A2 / A7 / A11 options")
    sup_mode  = sup_group.add_mutually_exclusive_group()
    sup_mode.add_argument(
        "--sweep-lr", action="store_true", dest="sweep_lr",
        help=(
            "Supervised 2D sweep: SUP_HPO_STEPS_GRID x SUP_LR_GRID. "
            "Run this first to find the best ft_lr."
        ),
    )
    sup_group.add_argument(
        "--ft-lr", type=float, default=None, dest="ft_lr",
        help=(
            "Fixed ft_lr for the A2 / A7 / A11 paper curve. "
            "Defaults to maml_alpha_init_eval from ablation_config (mirrors canonical ablation). "
            "Override here if desired."
        ),
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if args.model_type == "M0":
        assert args.checkpoint is not None, "--checkpoint is required for --model-type M0."
        assert not (args.sweep_lr or args.ft_lr is not None), (
            "--sweep-lr and --ft-lr are for A2/A7/A11, not M0. "
            "Use --sweep-alpha / --alpha instead."
        )
        if not args.sweep_alpha:
            assert args.alpha is not None, (
                "For --model-type M0 without --sweep-alpha, "
                "provide --alpha <best_alpha_init_eval>."
            )

    if args.model_type in ("A2", "A7", "A11"):
        assert not (args.sweep_alpha or args.alpha is not None), (
            "--sweep-alpha and --alpha are for M0, not A2/A7/A11. "
            "Use --sweep-lr / --ft-lr instead."
        )

    if args.model_type == "A2":
        assert args.checkpoint is None, (
            "--checkpoint is not used for --model-type A2 (model is trained inline). "
            "Remove --checkpoint from your command."
        )

    if args.model_type == "A7":
        assert args.checkpoint is not None, "--checkpoint is required for --model-type A7."

    # ── Resolve eval PIDs: CLI > VAL_PIDS + TEST_PIDS default ────────────────
    if args.eval_pids:
        eval_pids = args.eval_pids
    else:
        from ablation_config import VAL_PIDS, TEST_PIDS
        eval_pids = VAL_PIDS + TEST_PIDS
        print(f"[eval_pids] Defaulting to VAL_PIDS + TEST_PIDS ({len(eval_pids)} subjects): {eval_pids}")

    # ── Resolve ft_lr default from ablation_config ────────────────────────────
    # Default to maml_alpha_init_eval so the sweep's fixed-LR paper curve uses
    # the same LR as the canonical A2/A7/A11 ablation configs. This ensures
    # the adaptation curve is measuring the same thing as the table entry.
    if args.model_type in ("A2", "A7", "A11") and not args.sweep_lr:
        if args.ft_lr is not None:
            ft_lr = args.ft_lr
        else:
            from ablation_config import make_base_config as _make_base_config
            _tmp = _make_base_config("_lr_resolve")
            ft_lr = _tmp["maml_alpha_init_eval"]
            print(f"[ft_lr] Defaulting to maml_alpha_init_eval from ablation_config: {ft_lr:.6e}")
    else:
        ft_lr = args.ft_lr  # may be None for sweep_lr=True (unused)

    # ── Resolve checkpoint paths ──────────────────────────────────────────────
    if args.model_type in ("M0", "A7"):
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        checkpoint_path = None

    # ── Resolve output dir ────────────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve() if args.out_dir else RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.model_type == "M0":
        run_maml_sweep(
            checkpoint_path  = checkpoint_path,
            ablation_id      = args.ablation_id,
            out_dir          = out_dir,
            eval_pids        = eval_pids,
            num_val_episodes = args.num_val_episodes,
            sweep_alpha      = args.sweep_alpha,
            fixed_alpha      = args.alpha,
        )

    elif args.model_type == "A2":
        trained_model, trained_config, trained_tensor_dict_path = train_a2_model(out_dir)

        run_supervised_sweep(
            checkpoint_path          = None,
            ablation_id              = args.ablation_id,
            out_dir                  = out_dir,
            eval_pids                = eval_pids,
            num_val_episodes         = args.num_val_episodes,
            sweep_lr                 = args.sweep_lr,
            fixed_lr                 = ft_lr,
            ft_mode                  = args.ft_mode,
            model_type               = "A2",
            a11_base_config          = None,
            trained_model            = trained_model,
            trained_config           = trained_config,
            trained_tensor_dict_path = trained_tensor_dict_path,
        )

    elif args.model_type == "A7":
        run_supervised_sweep(
            checkpoint_path          = checkpoint_path,
            ablation_id              = args.ablation_id,
            out_dir                  = out_dir,
            eval_pids                = eval_pids,
            num_val_episodes         = args.num_val_episodes,
            sweep_lr                 = args.sweep_lr,
            fixed_lr                 = ft_lr,
            ft_mode                  = args.ft_mode,
            model_type               = "A7",
            a11_base_config          = None,
            trained_model            = None,
            trained_config           = None,
            trained_tensor_dict_path = None,
        )

    else:  # A11
        a11_base_config = build_a11_base_config()

        run_supervised_sweep(
            checkpoint_path          = None,
            ablation_id              = args.ablation_id,
            out_dir                  = out_dir,
            eval_pids                = eval_pids,
            num_val_episodes         = args.num_val_episodes,
            sweep_lr                 = args.sweep_lr,
            fixed_lr                 = ft_lr,
            ft_mode                  = args.ft_mode,
            model_type               = "A11",
            a11_base_config          = a11_base_config,
            trained_model            = None,
            trained_config           = None,
            trained_tensor_dict_path = None,
        )