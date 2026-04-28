"""
num_eval_steps_sweep.py
=========================
Post-hoc sweep over the number of test-time adaptation steps for trained
checkpoints. No training. No HPO. No Optuna. Pure grid sweep.

Purpose
-------
Answer: "How many adaptation steps does each model need before it plateaus?"

This is run AFTER training. It loads a frozen checkpoint and sweeps the
number of test-time gradient steps over a fixed grid. The model weights are
NEVER modified — each episode starts from the same checkpoint init.

Supported model types
---------------------
  MAML (M0, A3, A4, A5, A8, A12):
    Sweeps `maml_inner_steps_eval`.
    Also sweeps `maml_alpha_init_eval` jointly if --sweep-alpha is set
    (2D grid: steps x alpha). Use --sweep-alpha to find the best alpha
    first, then fix it with --alpha for the paper-curve run.

  Non-MAML supervised (A7 = CNN-LSTM, no MoE):
    Sweeps `num_ft_steps` in finetune_and_eval_user().
    Requires --ft-lr (fixed finetuning LR — take the best from A11 HPO
    or run this with --sweep-lr to find the plateau jointly).
    ft_mode: 'head_only' by default; override with --ft-mode full.

  Non-MAML Meta pretrained (A11):
    Same as A7 but uses MetaEMGWrapper + 2kHz EMG data.
    The MetaEMGWrapper class is defined inline below (copied from
    A11_eval_hpo_extended.py so this script is self-contained).

Workflow
--------
  # Step 1: MAML — find best (steps, alpha) jointly on val set
  python num_eval_steps_sweep.py \\
      --model-type maml \\
      --checkpoint /path/to/M0_best.pt \\
      --ablation-id M0 \\
      --sweep-alpha \\
      --out-dir /scratch/.../steps_sweep/M0

  # Step 2: MAML — paper curve at fixed best alpha
  python num_eval_steps_sweep.py \\
      --model-type maml \\
      --checkpoint /path/to/M0_best.pt \\
      --ablation-id M0 \\
      --alpha <best_alpha_from_step1> \\
      --out-dir /scratch/.../steps_sweep/M0

  # Step 3: Non-MAML A7 — find plateau (sweep LR + steps jointly)
  python num_eval_steps_sweep.py \\
      --model-type supervised \\
      --checkpoint /path/to/A7_best.pt \\
      --ablation-id A7 \\
      --sweep-lr \\
      --out-dir /scratch/.../steps_sweep/A7

  # Step 4: Non-MAML A7 — paper curve at fixed best LR
  python num_eval_steps_sweep.py \\
      --model-type supervised \\
      --checkpoint /path/to/A7_best.pt \\
      --ablation-id A7 \\
      --ft-lr <best_lr_from_step3> \\
      --out-dir /scratch/.../steps_sweep/A7

  # A11 (Meta pretrained) — identical to A7 steps above but --model-type a11
  # No --checkpoint needed for A11 (uses hardcoded Meta ckpt path).
  python num_eval_steps_sweep.py \\
      --model-type a11 \\
      --ablation-id A11 \\
      --sweep-lr \\
      --out-dir /scratch/.../steps_sweep/A11

Output
------
  <out_dir>/steps_sweep_<ablation_id>_<mode>_<timestamp>.json
  Partial results are written after every evaluated configuration —
  preemption-safe.

SLURM
-----
  Single-GPU job, no array needed. Estimated wall times:
    MAML 2D sweep  (8 steps x 9 alphas x ~2 min):  ~2-3h
    MAML paper curve (10 steps x ~2.5 min):          ~25 min
    Supervised 2D sweep (5 steps x 7 LRs x ~5 min): ~3h
    Supervised paper curve (10 steps x ~5 min):      ~50 min
    A11 similar to supervised (no-MoE backbone).

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

# A11 uses Meta's neuromotor repo — only needed for --model-type a11
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
# A11 constants — only used when --model-type a11
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

# MAML: paper figure trajectory — includes low step counts to show
# adaptation speed advantage relative to the supervised baseline.
MAML_PAPER_STEPS_GRID = [1, 5, 10, 25, 50, 100, 150, 200]

# MAML: 2D sweep grid for finding (steps, alpha) jointly.
# Starts at 50 because you already know < 50 is suboptimal for M0.
MAML_HPO_STEPS_GRID   = [25, 50, 75, 100, 125, 150, 175, 200, 250]
MAML_ALPHA_GRID       = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030]

# Supervised / A11: paper figure trajectory — same range as MAML for a
# fair visual comparison on the same x-axis.
SUP_PAPER_STEPS_GRID  = [1, 5, 10, 25, 50, 100, 150, 200]

# Supervised / A11: 2D sweep grid for finding (ft_steps, ft_lr) jointly.
SUP_HPO_STEPS_GRID    = [25, 50, 100, 150, 200]
# LR grid — centered around A11 v1 best (0.01) and expanded upward since
# both ft_lr and ft_steps hit the upper bound in the original HPO.
SUP_LR_GRID           = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0]

NUM_VAL_EPISODES      = 200
FIXED_SEED            = 42

# =============================================================================
# MetaEMGWrapper (A11 only — copied from A11_eval_hpo_extended.py)
# =============================================================================

class MetaEMGWrapper(nn.Module):
    """
    Wraps Meta's DiscreteGesturesArchitecture for our ablation eval pipeline.
    See A11_eval_hpo_extended.py for full documentation.
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
# Checkpoint loader (MAML and supervised)
# =============================================================================

def load_maml_checkpoint(checkpoint_path: Path) -> tuple:
    """
    Load a trained MAML checkpoint. Returns (model, config).

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
            inner_steps  = config["maml_inner_steps"],   # must match train-time value
            init_lr      = config["maml_alpha_init"],
            learnable    = True,
            device       = device,
        ).to(device)

    # ── Key remapping: ctx_proj -> router.projector ───────────────────────────
    # Checkpoints saved before the router/projector rename used "ctx_proj" as
    # the attribute name. Remap on the fly so we never have to touch the ckpt
    # file or revert model code.
    #
    # Two naming schemes must be patched:
    #   dot-separated  (model params):  ctx_proj.X      -> router.projector.X
    #   hyphen-separated (LSLR keys):   ctx_proj-X      -> router-projector-X
    #   (LSLR prefix is "_lslr._lrs.", so the pattern is "ctx_proj-" mid-key)
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
    # ── End remapping ─────────────────────────────────────────────────────────

    model.load_state_dict(remapped_sd)
    model.to(device)
    model.eval()

    print(f"  Best val acc    : {ckpt.get('best_val_acc', 'N/A')}")
    print(f"  Trained steps   : {config.get('maml_inner_steps')}")
    print(f"  Trained alpha_e : {config.get('maml_alpha_init_eval')}")
    return model, config


def load_supervised_checkpoint(checkpoint_path: Path) -> tuple:
    """
    Load a trained supervised (non-MAML) checkpoint. Returns (model, config).

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

    print(f"  Best val acc : {ckpt.get('best_val_acc', 'N/A')}")
    return model, config


# =============================================================================
# Single-config evaluators
# =============================================================================

def eval_maml_one_config(
    model:            torch.nn.Module,
    base_config:      dict,
    inner_steps_eval: int,
    alpha_eval:       float,
    val_pids:         list,
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
    eval_config["val_PIDs"]              = val_pids

    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = val_pids,
        target_gesture_classes  = eval_config["maml_gesture_classes"],
        target_trial_indices    = eval_config["target_trial_indices"],
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
    val_pids:         list,
    tensor_dict_path: str,
    num_val_episodes: int,
) -> dict:
    """
    Episodic finetuning eval for one (ft_steps, ft_lr) pair.
    A fresh copy of the model head is used per episode (finetune_and_eval_user
    handles this internally via deepcopy). Backbone weights from the checkpoint
    are never modified.
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from pretraining.pretrain_finetune import finetune_and_eval_user
    from torch.utils.data import DataLoader

    eval_config = copy.deepcopy(base_config)
    eval_config["ft_steps"]       = ft_steps
    eval_config["num_ft_steps"]   = ft_steps   # cover both key names defensively
    eval_config["ft_lr"]          = ft_lr
    eval_config["ft_label_smooth"] = 0.0
    eval_config["ft_weight_decay"] = 0.0
    eval_config["ft_optimizer"]   = "adam"
    eval_config["val_PIDs"]       = val_pids

    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = val_pids,
        target_gesture_classes  = eval_config["maml_gesture_classes"],
        target_trial_indices    = eval_config["target_trial_indices"],
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
    val_pids:         list,
    num_val_episodes: int,
    a11_base_config:  dict,
) -> dict:
    """
    Episodic finetuning eval for A11 (Meta pretrained) for one (ft_steps, ft_lr) pair.
    The MetaEMGWrapper is reconstructed fresh per config call — this is intentional:
    the model is stateless between calls since finetune_and_eval_user deepcopies
    the model internally, but to be safe and consistent with _eval_a11() in
    A11_eval_hpo_extended.py, we rebuild here.

    If runtime is a concern, you can hoist model construction outside this function
    and pass it in, mirroring the MAML and supervised eval patterns above.
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
    eval_config["ft_steps"]        = ft_steps
    eval_config["num_ft_steps"]    = ft_steps
    eval_config["ft_lr"]           = ft_lr
    eval_config["ft_label_smooth"] = 0.0
    eval_config["ft_weight_decay"] = 0.0
    eval_config["ft_optimizer"]    = "adam"
    eval_config["val_PIDs"]        = val_pids
    eval_config["num_eval_episodes"] = num_val_episodes
    eval_config["device"]          = device

    with open(EMG_2KHZ_PKL_PATH, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = val_pids,
        target_gesture_classes  = eval_config["maml_gesture_classes"],
        target_trial_indices    = eval_config["target_trial_indices"],
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
    val_pids:         list,
    num_val_episodes: int,
    sweep_alpha:      bool,
    fixed_alpha:      float | None,
) -> None:
    """
    MAML adaptation steps sweep.

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
    tensor_dict_path   = os.path.join(
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
    print(f"MAML Adaptation Steps Sweep — {ablation_id} ({mode_tag})")
    print(f"  Checkpoint     : {checkpoint_path}")
    print(f"  Val PIDs       : {val_pids}")
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

        result  = eval_maml_one_config(
            model            = model,
            base_config      = base_config,
            inner_steps_eval = steps,
            alpha_eval       = alpha,
            val_pids         = val_pids,
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
            "model_type":           "maml",
            "mode":                 mode_tag,
            "checkpoint":           str(checkpoint_path),
            "val_pids":             val_pids,
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
        print(f"  python adaptation_steps_sweep.py \\")
        print(f"      --model-type maml \\")
        print(f"      --checkpoint {checkpoint_path} \\")
        print(f"      --ablation-id {ablation_id} \\")
        print(f"      --alpha {best_alpha_found:.6f} \\")
        print(f"      --out-dir {out_dir}")


def run_supervised_sweep(
    checkpoint_path:  Path | None,
    ablation_id:      str,
    out_dir:          Path,
    val_pids:         list,
    num_val_episodes: int,
    sweep_lr:         bool,
    fixed_lr:         float | None,
    ft_mode:          str,
    model_type:       str,       # "supervised" or "a11"
    a11_base_config:  dict | None,
) -> None:
    """
    Supervised (non-MAML) adaptation steps sweep for A7 and A11.

    If sweep_lr=True  → 2D grid: SUP_HPO_STEPS_GRID x SUP_LR_GRID.
    If sweep_lr=False → paper curve: SUP_PAPER_STEPS_GRID at fixed_lr.
    """
    if not sweep_lr:
        assert fixed_lr is not None, \
            "--ft-lr must be provided when not using --sweep-lr."

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag  = "2d_sweep" if sweep_lr else "paper_curve"

    if model_type == "supervised":
        assert checkpoint_path is not None, \
            "--checkpoint is required for --model-type supervised."
        model, base_config = load_supervised_checkpoint(checkpoint_path)
        tensor_dict_path   = os.path.join(
            base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
        )
    else:
        # A11: no checkpoint arg needed; MetaEMGWrapper loads its own weights.
        # base_config and tensor_dict_path are handled inside eval_a11_one_config.
        assert a11_base_config is not None
        model         = None
        base_config   = a11_base_config
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
    print(f"  Val PIDs       : {val_pids}")
    print(f"  Val episodes   : {num_val_episodes}")
    print(f"  Steps grid     : {steps_grid}")
    print(f"  LR grid        : {lr_grid}")
    print(f"  Total configs  : {n_configs}")
    print(f"  Output dir     : {out_dir}")
    print(f"{'='*70}\n")

    results      = []
    best_acc     = -1.0
    best_steps   = None
    best_lr_found = None
    sweep_start  = time.time()
    out_path     = out_dir / f"steps_sweep_{ablation_id}_{mode_tag}_{timestamp}.json"

    for i, (steps, lr) in enumerate(grid):
        t0 = time.time()
        print(f"[{i+1:>3}/{n_configs}] ft_steps={steps:>4}, ft_lr={lr:.4f} ...",
              end="", flush=True)

        if model_type == "supervised":
            result = eval_supervised_one_config(
                model            = model,
                base_config      = base_config,
                ft_steps         = steps,
                ft_lr            = lr,
                ft_mode          = ft_mode,
                val_pids         = val_pids,
                tensor_dict_path = tensor_dict_path,
                num_val_episodes = num_val_episodes,
            )
        else:   # a11
            result = eval_a11_one_config(
                ft_steps         = steps,
                ft_lr            = lr,
                ft_mode          = ft_mode,
                val_pids         = val_pids,
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
            "checkpoint":           str(checkpoint_path) if checkpoint_path else "MetaEMGWrapper",
            "val_pids":             val_pids,
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
        print(f"  python adaptation_steps_sweep.py \\")
        print(f"      --model-type {model_type} \\")
        if checkpoint_path:
            print(f"      --checkpoint {checkpoint_path} \\")
        print(f"      --ablation-id {ablation_id} \\")
        print(f"      --ft-lr {best_lr_found:.6f} \\")
        print(f"      --ft-mode {ft_mode} \\")
        print(f"      --out-dir {out_dir}")


# =============================================================================
# A11 base config builder (no training — just the eval boilerplate)
# =============================================================================

def build_a11_base_config() -> dict:
    """
    Minimal config for A11 eval. Mirrors _build_a11_config() in
    A11_eval_hpo_extended.py but without ft_lr/ft_steps (those are swept).
    """
    import json as _json
    user_split_json = (
        CODE_DIR / "system" / "fixed_user_splits" / "hpo_strat_kapanji_split.json"
    )
    with open(user_split_json, "r") as f:
        all_splits = _json.load(f)
    split = all_splits[0]   # fold 0 — consistent with all other ablations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "n_way":                  3,
        "k_shot":                 1,
        "q_query":                9,
        "num_eval_episodes":      NUM_VAL_EPISODES,
        "maml_gesture_classes":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target_trial_indices":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "train_reps":             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "val_reps":               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "use_imu":                False,
        "device":                 device,
        "emg_in_ch":              EMG_2KHZ_IN_CH,
        "sequence_length":        EMG_2KHZ_SEQ_LEN,
        "train_PIDs":             split["train"],
        "val_PIDs":               split["val"],
        "test_PIDs":              split["test"],
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc adaptation steps sweep for MAML and supervised checkpoints. "
            "No training. No HPO. Pure grid sweep."
        )
    )

    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=["maml", "supervised", "a11"],
        dest="model_type",
        help=(
            "maml       : MAML checkpoint (M0, A3, A4, A5, A8, A12). "
            "supervised : Supervised CNN-LSTM checkpoint (A7). "
            "a11        : Meta pretrained EMG model (no --checkpoint needed)."
        ),
    )
    parser.add_argument(
        "--ablation-id", type=str, required=True, dest="ablation_id",
        help="e.g. M0, A7, A11. Used for output file naming only.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, dest="checkpoint",
        help="Path to .pt checkpoint. Required for maml and supervised; ignored for a11.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, dest="out_dir",
        help="Output directory. Defaults to RUN_DIR env var.",
    )
    parser.add_argument(
        "--val-pids", type=str, nargs="+", default=None, dest="val_pids",
        help="Val PIDs to evaluate on. Defaults to fold-0 val split from ablation_config.",
    )
    parser.add_argument(
        "--num-val-episodes", type=int, default=NUM_VAL_EPISODES, dest="num_val_episodes",
        help=f"Number of val episodes per config. Default: {NUM_VAL_EPISODES}.",
    )
    parser.add_argument(
        "--ft-mode", type=str, default="head_only",
        choices=["head_only", "full"], dest="ft_mode",
        help="Finetuning mode for supervised / a11. Default: head_only.",
    )

    # ── MAML-specific ─────────────────────────────────────────────────────────
    maml_group = parser.add_argument_group("MAML options")
    maml_mode  = maml_group.add_mutually_exclusive_group()
    maml_mode.add_argument(
        "--sweep-alpha", action="store_true", dest="sweep_alpha",
        help=(
            "MAML 2D sweep: MAML_HPO_STEPS_GRID x MAML_ALPHA_GRID. "
            "Run this first to find the best alpha."
        ),
    )
    maml_group.add_argument(
        "--alpha", type=float, default=None, dest="alpha",
        help=(
            "Fixed maml_alpha_init_eval for the MAML paper curve. "
            "Take best_alpha_so_far from the 2D sweep JSON."
        ),
    )

    # ── Supervised / A11 specific ─────────────────────────────────────────────
    sup_group = parser.add_argument_group("Supervised / A11 options")
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
            "Fixed ft_lr for the supervised / a11 paper curve. "
            "Take best_lr_so_far from the 2D sweep JSON."
        ),
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if args.model_type == "maml":
        assert args.checkpoint is not None, "--checkpoint is required for --model-type maml."
        assert not (args.sweep_lr or args.ft_lr is not None), (
            "--sweep-lr and --ft-lr are for supervised/a11, not maml. "
            "Use --sweep-alpha / --alpha instead."
        )
        if not args.sweep_alpha:
            assert args.alpha is not None, (
                "For --model-type maml without --sweep-alpha, "
                "provide --alpha <best_alpha_init_eval>."
            )

    if args.model_type in ("supervised", "a11"):
        assert not (args.sweep_alpha or args.alpha is not None), (
            "--sweep-alpha and --alpha are for maml, not supervised/a11. "
            "Use --sweep-lr / --ft-lr instead."
        )
        if args.model_type == "supervised":
            assert args.checkpoint is not None, \
                "--checkpoint is required for --model-type supervised."
        if not args.sweep_lr:
            assert args.ft_lr is not None, (
                "For --model-type supervised/a11 without --sweep-lr, "
                "provide --ft-lr <best_ft_lr>."
            )

    # ── Resolve paths / defaults ──────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve() if args.out_dir else RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None

    # Resolve val PIDs: CLI > fold-0 default from ablation_config
    if args.val_pids:
        val_pids = args.val_pids
    else:
        # Import lazily so the module-level ablation_config prints happen here
        from ablation_config import VAL_PIDS
        val_pids = VAL_PIDS

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.model_type == "maml":
        run_maml_sweep(
            checkpoint_path  = checkpoint_path,
            ablation_id      = args.ablation_id,
            out_dir          = out_dir,
            val_pids         = val_pids,
            num_val_episodes = args.num_val_episodes,
            sweep_alpha      = args.sweep_alpha,
            fixed_alpha      = args.alpha,
        )

    else:   # supervised or a11
        a11_base_config = build_a11_base_config() if args.model_type == "a11" else None

        run_supervised_sweep(
            checkpoint_path  = checkpoint_path,
            ablation_id      = args.ablation_id,
            out_dir          = out_dir,
            val_pids         = val_pids,
            num_val_episodes = args.num_val_episodes,
            sweep_lr         = args.sweep_lr,
            fixed_lr         = args.ft_lr,
            ft_mode          = args.ft_mode,
            model_type       = args.model_type,
            a11_base_config  = a11_base_config,
        )