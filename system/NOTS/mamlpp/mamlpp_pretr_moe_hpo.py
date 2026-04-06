"""
mamlpp_pretrained_hpo.py
========================
Two-phase script:

  PHASE 1 — Supervised Pretraining
  ---------------------------------
  Runs a single supervised pretraining pass with fixed HPs synthesised from
  the standalone pretraining HPO study.  The best checkpoint is saved to
  PRETRAIN_CKPT_PATH (printed clearly at runtime).  If the MoE collapses
  during pretraining the script prints a loud banner and exits immediately.

  PHASE 2 — MAML++ HPO  (15 trials)
  -----------------------------------
  Optuna sweeps outer_lr × maml_inner_steps (the two HPs most likely to
  shift when starting from a warm pretrained init).  Every other HP is fixed
  to the best value found in the standalone MAML HPO study.  Each trial
  loads the checkpoint written in Phase 1 as the MAML initialiser.

Usage
-----
  python mamlpp_pretrained_hpo.py --model_type DeepCNNLSTM

  # Skip Phase 1 and reuse an existing checkpoint:
  python mamlpp_pretrained_hpo.py --skip_pretraining \\
      --pretrained_ckpt /path/to/DeepCNNLSTM_pretrained_YYYYMMDD_HHMM_best.pt

Environment variables (same as all other scripts)
---------------------------------------------------
  CODE_DIR   root of the repo       (default: ./)
  DATA_DIR   data root              (default: ./data)
  RUN_DIR    where outputs are saved (default: ./)
"""

# ── Trial count: 15 = 5 outer_lr values × 3 inner_steps values ───────────────
N_TRIALS   = 15
FIXED_SEED = 42

import os, sys, copy, json, time, random, warnings, pickle
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()

print(f"CODE_DIR : {CODE_DIR}")
print(f"DATA_DIR : {DATA_DIR}")
print(f"RUN_DIR  : {RUN_DIR}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")

RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── Tensor dict path (pretraining data) ───────────────────────────────────────
TENSOR_DICT_PATH = (
    CODE_DIR / "dataset" / "meta-learning-sup-que-ds" / "segfilt_rts_tensor_dict.pkl"
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.append(str(CODE_DIR))
MAML_DIR = CODE_DIR / "system" / "MAML"
MOE_DIR  = CODE_DIR / "system" / "MOE"
sys.path.extend([str(MAML_DIR), str(MOE_DIR)])

from system.MAML.mamlpp import mamlpp_pretrain, mamlpp_adapt_and_eval
from system.MAML.maml_data_pipeline import get_maml_dataloaders
from system.pretraining.pretrain_models import build_model as build_baseline_model
from system.pretraining.contrastive_net.contrastive_encoder import ContrastiveGestureEncoder
from MOE.MOE_encoder import build_MOE_model, load_pretrained_into_MOE

# User-split JSON
USER_SPLIT_JSON = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"
with open(USER_SPLIT_JSON, "r") as _f:
    ALL_SPLITS = json.load(_f)

NUM_FOLDS = 1   # single fold for this experiment

#
# All HPs are fixed from prior HPO studies.  Rationale for each value is given
# inline.  Architecture params marked ★ MUST MATCH THE PRETRAINED CHECKPOINT ★
# and cannot be changed without rerunning Phase 1.
#
# ── Architecture  (★ MUST MATCH CHECKPOINT ★) ────────────────────────────────
ARCH = dict(
    model_type        = "DeepCNNLSTM",  # DeepCNNLSTM backbone + MoE encoder layer
    sequence_length   = 64,
    emg_in_ch         = 16,
    imu_in_ch         = 72,
    demo_in_dim       = 12,
    cnn_base_filters  = 128,    # pretraining HPO: 128 best
    cnn_layers        = 3,
    cnn_kernel        = 5,
    groupnorm_num_groups = 4,   # MAML HPO: 4 used more; 8 is fine too
    lstm_hidden       = 128,    # pretraining & MAML HPO: 128 best; 64/512 bad
    lstm_layers       = 3,
    bidirectional     = True,
    head_type         = "mlp",  # pretraining HPO: MLP a little better
)

# ── MoE architecture  (★ MUST MATCH CHECKPOINT ★) ────────────────────────────
MOE_ARCH = dict(
    use_MOE           = True,
    MOE_placement     = "encoder",  # only placement tested
    num_experts       = 32,         # pretraining HPO: peak at 32; MAML HPO: 20-40
    MOE_top_k         = 8,          # both studies: 8-10 best; 1-2 bad
    MOE_ctx_out_dim   = 128,        # pretraining HPO: 128 chosen
    MOE_ctx_hidden_dim= 32,         # pretraining HPO: 64 okay; MAML: 32 best
                                    # ← NOTE: if pretraining ckpt used 32, change this to 32
    MOE_expert_expand = 1.0,        # fixed throughout both HPO studies
    MOE_mlp_hidden_mult=1.0,        # fixed throughout both HPO studies
)

# ── Pretraining HPs (Phase 1) ─────────────────────────────────────────────────
# These are used ONLY for the supervised pretraining pass and are NOT tuned
# in the MAML HPO (Phase 2).
PRETRAIN_HPS = dict(
    # Optimiser
    pretrain_lr           = 0.0001,  # pretraining HPO: lower better; 0.0001 was lowest tested
    pretrain_weight_decay = 0.001,   # pretraining HPO: best trial; most <0.0001
    pretrain_optimizer    = "adam",  # pretraining HPO: Adam a little better
    pretrain_dropout      = 0.15,    # pretraining HPO: no real trend; 0.05-0.25 all fine
    pretrain_label_smooth = 0.12,    # pretraining HPO: best 4 trials 0.09-0.15
    pretrain_batch_size   = 128,     # pretraining HPO: 128 best
    # MoE training HPs for the pretraining phase specifically
    pretrain_MOE_gate_temperature = 4.0,    # pretraining HPO: 2-6
    pretrain_MOE_aux_coeff        = 0.5,    # pretraining HPO: 0.2-1.0
    pretrain_MOE_dropout          = 0.14,   # pretraining HPO: 0.12-0.17
    # Schedule
    pretrain_num_epochs   = 100,
    pretrain_es_patience  = 10,
    pretrain_es_min_delta = 0.001,
)

# ── MAML HPO sweep axes (Phase 2) ─────────────────────────────────────────────
# These are the two HPs most likely to shift when starting from a warm
# pretrained init. Everything else is fixed below.
OUTER_LR_SWEEP    = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]   # 5 values
INNER_STEPS_SWEEP = [10, 15, 20]                        # 3 values  → 15 trials total

# ── Fixed MAML HPs (Phase 2) ──────────────────────────────────────────────────
MAML_FIXED = dict(
    # Optimiser
    weight_decay          = 0.0001,  # MAML HPO: >0.0001
    maml_alpha_init       = 0.003,   # MAML HPO: <0.005
    maml_alpha_init_eval  = 0.02,    # MAML HPO: 0.0075-0.075; geometric middle
    maml_inner_steps_eval = 100,     # MAML HPO: 100 most common; 50 also fine
    maml_use_lslr         = True,    # MAML HPO: strong signal
    use_lslr_at_eval      = False,   # MAML HPO: False dominant; True also fine
    meta_batchsize        = 24,      # MAML HPO: clear winner
    # Loss / regularisation
    label_smooth          = 0.15,    # MAML HPO: 0.15-0.2 slightly better
    use_maml_msl          = False,   # MAML HPO: False wins; hybrid had higher variance
    episodes_per_epoch    = 200,     # MAML HPO: 200-400 fine; drop-off above 500
    # MoE training HPs for MAML phase (can differ from pretraining)
    MOE_gate_temperature  = 1.0,     # MAML HPO strongly preferred <1.5 (lower=sharper)
    MOE_aux_coeff         = 0.02,    # MAML HPO: lower better; ~0.02
    MOE_aux_loss_plcmt    = "both",  # MAML HPO: Optuna favoured both; inner > outer alone
    MOE_dropout           = 0.05,    # MAML HPO: no strong trend; Optuna drifted ~0.05
    # MAML++ flags
    maml_opt_order        = "first",
    maml_first_order_to_second_order_epoch = 1_000_000,
    enable_inner_loop_optimizable_bn_params= False,
    use_cosine_outer_lr   = False,
    lr_scheduler_factor   = 0.1,
    lr_scheduler_patience = 6,
    # Training schedule
    num_epochs            = 50,
    use_earlystopping     = True,
    earlystopping_patience= 8,
    earlystopping_min_delta=0.005,
    # Misc
    optimizer             = "adam",
    use_batch_norm        = False,
    dropout               = 0.1,
    use_GlobalAvgPooling  = False,   # MAML HPO: False had tighter spread; more common
    num_workers           = 8,
    gradient_clip_max_norm= 10.0,
    num_eval_episodes     = 10,
    verbose               = False,
    use_label_shuf_meta_aug = True,
    track_gradient_alignment= False,
)
# =============================================================================
# END HP DICTIONARY
# =============================================================================


# ── Collapse detection ────────────────────────────────────────────────────────
COLLAPSE_MAX_LOAD_THRESHOLD = 0.80

def _check_moe_collapse(logs: dict, num_experts: int) -> float | None:
    """Returns the max expert load fraction (0-1), or None if not available."""
    reports = logs.get("routing_reports", [])
    if not reports:
        return None
    last = reports[-1]
    for key in ("max_expert_load", "dominant_fraction", "max_load"):
        if key in last:
            return float(last[key])
    lb    = last.get("load_balance", {})
    fracs = lb.get("expert_hard_fraction")
    if fracs:
        return float(max(fracs))
    imbal = lb.get("hard_imbalance_ratio")
    if imbal is not None:
        return float(imbal / (num_experts + imbal - 1))
    return None

def _die_on_collapse(max_load: float | None, num_experts: int, phase: str):
    """Prints a loud banner and sys.exit(1) if MoE has collapsed."""
    if max_load is not None and max_load > COLLAPSE_MAX_LOAD_THRESHOLD:
        print()
        print("!" * 70)
        print(f"  MoE COLLAPSE DETECTED during {phase}")
        print(f"  max_expert_load = {max_load:.3f}  (threshold = {COLLAPSE_MAX_LOAD_THRESHOLD})")
        print(f"  num_experts     = {num_experts}")
        print(f"  Aborting — adjust MOE_aux_coeff / MOE_gate_temperature and retry.")
        print("!" * 70)
        print()
        sys.exit(1)


# ── Shared config/path helpers ────────────────────────────────────────────────
def _make_base_config() -> dict:
    """Returns a config dict with all path-related keys and static task params."""
    cfg = {}
    cfg["NOTS"]                    = True
    cfg["user_split_json_filepath"]= str(USER_SPLIT_JSON)
    cfg["results_save_dir"]        = RUN_DIR
    cfg["models_save_dir"]         = RUN_DIR
    cfg["emg_imu_pkl_full_path"]   = str(CODE_DIR / "dataset" / "filtered_datasets" / "metadata_IMU_EMG_allgestures_allusers.pkl")
    cfg["pwmd_xlsx_filepath"]      = str(CODE_DIR / "dataset" / "Biosignal gesture questionnaire for participants with disabilities.xlsx")
    cfg["pwoutmd_xlsx_filepath"]   = str(CODE_DIR / "dataset" / "Biosignal gesture questionnaire for participants without disabilities.xlsx")
    cfg["dfs_save_path"]           = str(CODE_DIR / "dataset" / "")
    cfg["dfs_load_path"]           = str(CODE_DIR / "dataset" / "meta-learning-sup-que-ds" / "")
    cfg["pretrain_dir"]            = str(CODE_DIR / "pretrain_outputs" / "checkpoints" / "")
    cfg["device"]                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Static task params
    cfg["n_way"]                   = 3
    cfg["k_shot"]                  = 1
    cfg["q_query"]                 = 9
    cfg["num_classes"]             = 10
    cfg["feature_engr"]            = "None"
    cfg["multimodal"]              = True
    cfg["use_imu"]                 = True
    cfg["use_demographics"]        = False
    cfg["use_film_x_demo"]         = False
    cfg["FILM_on_context_or_demo"] = "context"
    cfg["emg_stride"]              = 1
    cfg["imu_stride"]              = 1
    cfg["padding"]                 = 0
    cfg["sequence_length"]         = ARCH["sequence_length"]
    cfg["emg_in_ch"]               = ARCH["emg_in_ch"]
    cfg["imu_in_ch"]               = ARCH["imu_in_ch"]
    cfg["demo_in_dim"]             = ARCH["demo_in_dim"]
    cfg["maml_gesture_classes"]    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cfg["target_trial_indices"]    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cfg["debug_one_user_only"]     = False
    cfg["debug_one_episode"]       = False
    cfg["debug_five_episodes"]     = False
    return cfg


def _inject_arch(cfg: dict) -> dict:
    """Writes architecture params into cfg (★ must match pretrained ckpt ★)."""
    cfg.update(ARCH)
    cfg.update(MOE_ARCH)
    cfg["top_k"]               = MOE_ARCH["MOE_top_k"]   # some modules read bare 'top_k'
    cfg["gate_type"]           = "context_feature_demo"  # legacy compatibility
    cfg["expert_architecture"] = "MLP"                   # legacy compatibility
    cfg["MOE_log_every"]       = 5
    cfg["MOE_plot_dir"]        = None
    return cfg


# =============================================================================
# PHASE 1 — Supervised Pretraining
# =============================================================================

def run_pretraining(model_type: str) -> Path:
    """
    Runs one supervised pretraining pass with the fixed PRETRAIN_HPS.
    Saves best and last checkpoints, then returns the path to the best one.

    A small JSON sidecar ({stem}_paths.json) is also written so Phase 2 can
    find the checkpoint even if run separately.

    Returns
    -------
    Path  — absolute path to the saved *_best.pt checkpoint.
    """
    print()
    print("=" * 70)
    print("  PHASE 1 — SUPERVISED PRETRAINING")
    print("=" * 70)

    cfg = _make_base_config()
    _inject_arch(cfg)

    # NOTE: Otherwise missing?
    cfg['train_reps']             = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # INTRA: [1, 2, 3, 4, 5, 6, 7, 8],
    cfg['val_reps']               = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # INTRA: [9, 10],
    cfg['num_val_episodes']       = 10  # This is MAML only I think

    # Pretraining-specific HPs
    cfg["learning_rate"]           = PRETRAIN_HPS["pretrain_lr"]
    cfg["weight_decay"]            = PRETRAIN_HPS["pretrain_weight_decay"]
    cfg["optimizer"]               = PRETRAIN_HPS["pretrain_optimizer"]
    cfg["dropout"]                 = PRETRAIN_HPS["pretrain_dropout"]
    cfg["label_smooth"]            = PRETRAIN_HPS["pretrain_label_smooth"]
    cfg["batch_size"]              = PRETRAIN_HPS["pretrain_batch_size"]
    cfg["num_epochs"]              = PRETRAIN_HPS["pretrain_num_epochs"]
    cfg["use_earlystopping"]       = True
    cfg["earlystopping_patience"]  = PRETRAIN_HPS["pretrain_es_patience"]
    cfg["earlystopping_min_delta"] = PRETRAIN_HPS["pretrain_es_min_delta"]
    cfg["MOE_gate_temperature"]    = PRETRAIN_HPS["pretrain_MOE_gate_temperature"]
    cfg["MOE_aux_coeff"]           = PRETRAIN_HPS["pretrain_MOE_aux_coeff"]
    cfg["MOE_dropout"]             = PRETRAIN_HPS["pretrain_MOE_dropout"]
    # Pretraining is supervised, not meta-learning
    cfg["meta_learning"]           = False
    cfg["pretrain_approach"]       = "None"   # random init — this IS the pretraining

    # Use fold-0 split
    split = ALL_SPLITS[0]
    cfg["train_PIDs"] = split["train"]
    cfg["val_PIDs"]   = split["val"]
    cfg["test_PIDs"]  = split["test"]

    print(f"\n[Phase 1] Building {model_type} + MoE model (random init for pretraining)...")
    model = build_MOE_model(cfg)
    model.to(cfg["device"])

    print(f"\n[Phase 1] Starting supervised pretraining...")
    print(f"[Phase 1] Checkpoints will be saved to: {RUN_DIR}")
    print()

    # ── Run pretraining loop ──────────────────────────────────────────────────
    # pretrain_trainer.pretrain() expects: (model, train_dl, val_dl, config, save_path).
    # get_pretrain_dataloaders() expects the config dict to contain the split PIDs
    # (train_PIDs / val_PIDs) and standard training keys (batch_size, num_workers, …).
    #
    # Early-stopping keys used by pretrain_trainer.py:
    #   use_early_stopping  (bool)
    #   es_patience         (int)
    #   es_min_delta        (float)
    # The PRETRAIN_HPS dict uses different names, so we map them here explicitly
    # rather than touching PRETRAIN_HPS (which would change the HPO interface).
    cfg["use_early_stopping"] = cfg.pop("use_earlystopping",       True)
    cfg["es_patience"]        = cfg.pop("earlystopping_patience",  PRETRAIN_HPS["pretrain_es_patience"])
    cfg["es_min_delta"]       = cfg.pop("earlystopping_min_delta", PRETRAIN_HPS["pretrain_es_min_delta"])

    # pretrain_trainer also reads 'grad_clip', 'use_scheduler', 'warmup_epochs'.
    # Provide sensible defaults if not already present.
    cfg.setdefault("grad_clip",      5.0)
    cfg.setdefault("use_scheduler",  True)
    cfg.setdefault("warmup_epochs",  5)
    cfg.setdefault("use_amp",        False)

    from system.pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
    from system.pretraining.pretrain_trainer import pretrain

    print(f"[Phase 1] Loading tensor dict from: {TENSOR_DICT_PATH}")
    with open(TENSOR_DICT_PATH, "rb") as _f:
        _raw = pickle.load(_f)
    tensor_dict = _raw["data"] if "data" in _raw else _raw
    print(f"[Phase 1] Tensor dict loaded — {len(tensor_dict)} subjects.")

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(cfg, tensor_dict=tensor_dict)
    print(f"[Phase 1] DataLoaders ready | n_classes={n_classes}")

    # pretrain() returns (model_with_best_weights_loaded, history_dict).
    # 'history' contains: train_loss, val_loss, val_acc, best_val_loss,
    #                      best_val_acc, best_epoch, routing_reports.
    # The model is already restored to its best state before pretrain() returns,
    # so best_state == model.state_dict() at this point.
    model, history = pretrain(model, train_dl, val_dl, cfg, save_path=None)

    best_state   = copy.deepcopy(model.state_dict())
    best_val_acc = float(history.get("best_val_acc", 0.0))

    # Expose the same keys the collapse-checker and checkpoint code expect
    pretrain_res = history          # history IS the logs dict for _check_moe_collapse

    # ── MoE collapse check (exits immediately if collapsed) ───────────────────
    max_load = _check_moe_collapse(pretrain_res, num_experts=cfg["num_experts"])
    print(f"\n[Phase 1] MoE max_expert_load = {max_load}")
    _die_on_collapse(max_load, cfg["num_experts"], phase="PRETRAINING (Phase 1)")

    # ── Save checkpoints ──────────────────────────────────────────────────────
    ckpt_stem = f"{model_type}_pretrained_{timestamp}"
    ckpt_best = RUN_DIR / f"{ckpt_stem}_best.pt"
    ckpt_last = RUN_DIR / f"{ckpt_stem}_last.pt"

    torch.save({
        "model_type":       model_type,
        "model_state_dict": best_state,
        "config":           cfg,
        "best_val_acc":     best_val_acc,
        "train_loss_log":   pretrain_res.get("train_loss", pretrain_res.get("train_loss_log", [])),
        "train_acc_log":   pretrain_res.get("train_acc", pretrain_res.get("train_acc_log", [])),
        "val_loss_log":     pretrain_res.get("val_loss",   pretrain_res.get("val_loss_log",   [])),
        "val_acc_log":      pretrain_res.get("val_acc",    pretrain_res.get("val_acc_log",    [])),
        "routing_reports":  pretrain_res.get("routing_reports", []),
    }, ckpt_best)

    last_state = pretrain_res.get("last_state", best_state)
    torch.save({
        "model_type":       model_type,
        "model_state_dict": last_state,
        "config":           cfg,
    }, ckpt_last)

    # Sidecar JSON so Phase 2 can find the checkpoint deterministically
    sidecar = RUN_DIR / f"{ckpt_stem}_paths.json"
    with open(sidecar, "w") as _f:
        json.dump({"best": str(ckpt_best), "last": str(ckpt_last)}, _f, indent=2)

    print()
    print("─" * 70)
    print(f"  [Phase 1] Pretraining COMPLETE")
    print(f"  [Phase 1] Best val acc  : {best_val_acc*100:.2f}%")
    print(f"  [Phase 1] Best ckpt     : {ckpt_best}")
    print(f"  [Phase 1] Last ckpt     : {ckpt_last}")
    print(f"  [Phase 1] Path sidecar  : {sidecar}")
    print("─" * 70)
    print()

    return ckpt_best


# =============================================================================
# PHASE 2 — MAML++ HPO
# =============================================================================

def build_maml_config_from_trial(trial, pretrained_ckpt_path: Path) -> dict:
    """
    Constructs a complete MAML config for one Optuna trial.
    The two sweep axes are outer_lr and maml_inner_steps.
    Everything else comes from MAML_FIXED / MOE_ARCH / ARCH.
    """
    cfg = _make_base_config()
    _inject_arch(cfg)

    # ── Sweep axes ────────────────────────────────────────────────────────────
    cfg["learning_rate"]    = trial.suggest_categorical("outer_lr",         OUTER_LR_SWEEP)
    cfg["maml_inner_steps"] = trial.suggest_categorical("maml_inner_steps", INNER_STEPS_SWEEP)

    # ── Fixed MAML HPs ────────────────────────────────────────────────────────
    cfg["weight_decay"]            = MAML_FIXED["weight_decay"]
    cfg["maml_alpha_init"]         = MAML_FIXED["maml_alpha_init"]
    cfg["maml_alpha_init_eval"]    = MAML_FIXED["maml_alpha_init_eval"]
    cfg["maml_inner_steps_eval"]   = MAML_FIXED["maml_inner_steps_eval"]
    cfg["maml_use_lslr"]           = MAML_FIXED["maml_use_lslr"]
    cfg["use_lslr_at_eval"]        = MAML_FIXED["use_lslr_at_eval"]
    cfg["meta_batchsize"]          = MAML_FIXED["meta_batchsize"]
    cfg["label_smooth"]            = MAML_FIXED["label_smooth"]
    cfg["use_maml_msl"]            = MAML_FIXED["use_maml_msl"]
    cfg["maml_msl_num_epochs"]     = 0   # consistent with use_maml_msl=False
    cfg["episodes_per_epoch_train"]= MAML_FIXED["episodes_per_epoch"]
    cfg["MOE_gate_temperature"]    = MAML_FIXED["MOE_gate_temperature"]
    cfg["MOE_aux_coeff"]           = MAML_FIXED["MOE_aux_coeff"]
    cfg["apply_MOE_aux_loss_inner_outer"] = MAML_FIXED["MOE_aux_loss_plcmt"]
    cfg["MOE_dropout"]             = MAML_FIXED["MOE_dropout"]
    cfg["maml_opt_order"]          = MAML_FIXED["maml_opt_order"]
    cfg["maml_first_order_to_second_order_epoch"] = MAML_FIXED["maml_first_order_to_second_order_epoch"]
    cfg["enable_inner_loop_optimizable_bn_params"]= MAML_FIXED["enable_inner_loop_optimizable_bn_params"]
    cfg["use_cosine_outer_lr"]     = MAML_FIXED["use_cosine_outer_lr"]
    cfg["lr_scheduler_factor"]     = MAML_FIXED["lr_scheduler_factor"]
    cfg["lr_scheduler_patience"]   = MAML_FIXED["lr_scheduler_patience"]
    cfg["num_epochs"]              = MAML_FIXED["num_epochs"]
    cfg["use_earlystopping"]       = MAML_FIXED["use_earlystopping"]
    cfg["earlystopping_patience"]  = MAML_FIXED["earlystopping_patience"]
    cfg["earlystopping_min_delta"] = MAML_FIXED["earlystopping_min_delta"]
    cfg["optimizer"]               = MAML_FIXED["optimizer"]
    cfg["use_batch_norm"]          = MAML_FIXED["use_batch_norm"]
    cfg["dropout"]                 = MAML_FIXED["dropout"]
    cfg["use_GlobalAvgPooling"]    = MAML_FIXED["use_GlobalAvgPooling"]
    cfg["num_workers"]             = MAML_FIXED["num_workers"]
    cfg["gradient_clip_max_norm"]  = MAML_FIXED["gradient_clip_max_norm"]
    cfg["num_eval_episodes"]       = MAML_FIXED["num_eval_episodes"]
    cfg["verbose"]                 = MAML_FIXED["verbose"]
    cfg["use_label_shuf_meta_aug"] = MAML_FIXED["use_label_shuf_meta_aug"]
    cfg["track_gradient_alignment"]= MAML_FIXED["track_gradient_alignment"]
    cfg["meta_learning"]           = True

    # ── Point at the Phase 1 checkpoint ───────────────────────────────────────
    # pretrained_model_filename is set to the full absolute path of the checkpoint.
    # build_maml_model() reads this and calls load_pretrained_into_MOE directly,
    # bypassing the default-stem lookup in get_pretrain_path().
    cfg["pretrain_approach"]         = "full_best"
    cfg["pretrained_model_filename"] = str(pretrained_ckpt_path)

    return cfg


def build_maml_model(cfg: dict) -> nn.Module:
    """
    Builds a DeepCNNLSTM+MoE model and loads Phase 1 pretrained weights.

    Uses load_pretrained_into_MOE (same as mpp_run.py's build_model for the
    MOE+pretrained case) so expert CNNs are seeded from the backbone weights.

    The checkpoint path comes from cfg["pretrained_model_filename"].
    Raises FileNotFoundError (with a clear message) if the checkpoint is missing.
    """
    model = build_MOE_model(cfg)

    ckpt_path = cfg["pretrained_model_filename"]

    print()
    print("─" * 70)
    print(f"  [Phase 2] Loading pretrained checkpoint:")
    print(f"            {ckpt_path}")

    if not Path(ckpt_path).exists():
        print()
        print("  *** ERROR: Checkpoint file not found! ***")
        print(f"  Expected path : {ckpt_path}")
        print("  Did Phase 1 complete successfully?")
        print("  Check RUN_DIR and the _paths.json sidecar.")
        print("─" * 70)
        raise FileNotFoundError(f"Pretrained checkpoint missing: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=cfg["device"], weights_only=False)
    state_dict = (checkpoint.get("model_state")
                  or checkpoint.get("model_state_dict")
                  or checkpoint)

    model = load_pretrained_into_MOE(
        MOE_model             = model,
        pretrained_state_dict = state_dict,
        placement             = cfg["MOE_placement"],
        seed_experts          = True,
        verbose               = True,
    )
    print(f"  [Phase 2] Pretrained weights loaded successfully.")
    print("─" * 70)
    print()

    model.to(cfg["device"])
    return model


def objective(trial, pretrained_ckpt_path: Path) -> float:
    """Optuna objective for one MAML++ trial."""
    fold_mean_accs     = []
    all_fold_user_accs = []
    maml_val_accs      = []

    for fold_idx in range(NUM_FOLDS):
        fold_start = time.time()

        print("=" * 70)
        print(f"  [Trial {trial.number}] Fold {fold_idx+1}/{NUM_FOLDS}")
        print(f"  outer_lr      = {trial.params.get('outer_lr', '(sampling...)')}")
        print(f"  maml_inner_steps = {trial.params.get('maml_inner_steps', '(sampling...)')}")
        print("=" * 70)

        cfg = build_maml_config_from_trial(trial, pretrained_ckpt_path)

        # Apply fold split
        split = ALL_SPLITS[fold_idx]
        cfg["train_PIDs"] = split["train"]
        cfg["val_PIDs"]   = split["val"]
        cfg["test_PIDs"]  = split["test"]

        # Build model and load pretrained weights
        model = build_maml_model(cfg)

        if cfg["device"].type == "cpu":
            print("  WARNING: Running on CPU — this will be slow!")

        # Data
        tensor_dict_path = str(Path(cfg["dfs_load_path"]) / "segfilt_rts_tensor_dict.pkl")
        episodic_train_loader, episodic_val_loader = get_maml_dataloaders(
            cfg, tensor_dict_path=tensor_dict_path,
        )

        # MAML++ meta-training
        trained_model, maml_res = mamlpp_pretrain(
            model, cfg, episodic_train_loader, episodic_val_loader=episodic_val_loader,
        )
        best_val_acc = maml_res["best_val_acc"]
        best_state   = maml_res["best_state"]
        maml_val_accs.append(float(best_val_acc))

        print(f"\n  [Trial {trial.number} | Fold {fold_idx}] "
              f"MAML meta-training done. Best val acc = {best_val_acc*100:.2f}%")

        # ── MoE collapse check (exits immediately if collapsed) ───────────────
        max_load = _check_moe_collapse(maml_res, num_experts=cfg["num_experts"])
        trial.set_user_attr("final_max_expert_load",
                            max_load if max_load is not None else -1.0)
        _die_on_collapse(max_load, cfg["num_experts"],
                         phase=f"MAML Phase 2 — Trial {trial.number} | Fold {fold_idx}")

        # ── Save trial checkpoint ─────────────────────────────────────────────
        ckpt_name = f"maml_pretrained_trial{trial.number:03d}_fold{fold_idx}_{timestamp}_best.pt"
        ckpt_path = RUN_DIR / ckpt_name
        torch.save({
            "trial_num":        trial.number,
            "fold_idx":         fold_idx,
            "model_state_dict": best_state,
            "config":           cfg,
            "best_val_acc":     best_val_acc,
            "train_loss_log":   maml_res["train_loss_log"],
            "val_loss_log":     maml_res["val_loss_log"],
            "val_acc_log":      maml_res["val_acc_log"],
        }, ckpt_path)
        print(f"  [Trial {trial.number} | Fold {fold_idx}] Checkpoint → {ckpt_path}")

        # ── Per-user adaptation + evaluation ─────────────────────────────────
        model.load_state_dict(best_state)
        user_metrics = defaultdict(list)

        for batch in episodic_val_loader:
            user_id     = batch["user_id"]
            support_set = batch["support"]
            query_set   = batch["query"]
            val_metrics = mamlpp_adapt_and_eval(model, cfg, support_set, query_set)
            user_metrics[user_id].append(val_metrics["acc"])

        all_user_means = []
        for uid, accs in user_metrics.items():
            m = np.mean(accs)
            all_user_means.append(float(m))
            print(f"    User {uid} | {m*100:.2f}%  ({len(accs)} episodes)")

        mean_acc = float(np.mean(all_user_means))
        std_acc  = float(np.std(all_user_means))
        pcts     = [round(a * 100, 2) for a in all_user_means]
        elapsed  = time.time() - fold_start

        print(f"\n  [Trial {trial.number} | Fold {fold_idx}] "
              f"User accs (%): {pcts}")
        print(f"  [Trial {trial.number} | Fold {fold_idx}] "
              f"Mean = {mean_acc*100:.2f}%  ±  {std_acc*100:.2f}%  "
              f"| elapsed = {elapsed:.0f}s")

        fold_mean_accs.append(mean_acc)
        all_fold_user_accs.append(all_user_means)

    overall = float(np.nanmean(fold_mean_accs))
    trial.set_user_attr("fold_mean_accs",    fold_mean_accs)
    trial.set_user_attr("fold_user_accs",    all_fold_user_accs)
    trial.set_user_attr("mean_maml_val_acc", float(np.nanmean(maml_val_accs)))
    return overall


def run_maml_hpo(pretrained_ckpt_path: Path, model_type: str):
    """Sets up and runs the 15-trial Optuna study (Phase 2)."""
    print()
    print("=" * 70)
    print("  PHASE 2 — MAML++ HPO")
    print(f"  Sweep: outer_lr {OUTER_LR_SWEEP}")
    print(f"         maml_inner_steps {INNER_STEPS_SWEEP}")
    print(f"  Total trials : {N_TRIALS}")
    print(f"  Pretrained checkpoint:")
    print(f"    {pretrained_ckpt_path}")
    print("=" * 70)
    print()

    db_dir = Path("/scratch/my13/kai/meta-pers-gest/optuna_dbs")
    db_dir.mkdir(parents=True, exist_ok=True)

    study_name   = f"mamlpp_pretrained_{model_type}_{timestamp}"
    journal_path = str(db_dir / f"{study_name}.log")

    print(f"  Optuna study  : {study_name}")
    print(f"  Journal path  : {journal_path}")
    print()

    sleep_t = random.uniform(0, 10)
    print(f"  Staggering start by {sleep_t:.1f}s to avoid journal-lock collisions...")
    time.sleep(sleep_t)

    storage = JournalStorage(JournalFileBackend(journal_path))
    time.sleep(random.uniform(0, 5))

    study = optuna.create_study(
        study_name    = study_name,
        direction     = "maximize",
        storage       = storage,
        load_if_exists= True,
    )
    study.optimize(
        lambda trial: objective(trial, pretrained_ckpt_path),
        n_trials      = N_TRIALS,
        gc_after_trial= True,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    best = study.best_trial
    print()
    print("=" * 70)
    print("  PHASE 2 COMPLETE — HPO Summary")
    print("=" * 70)
    print(f"  Best trial    : #{best.number}")
    print(f"  Best value    : {best.value*100:.2f}%")
    print(f"  Best params   :")
    for k, v in best.params.items():
        print(f"    {k} = {v}")
    print("=" * 70)

    summary_path = RUN_DIR / f"{study_name}_hpo_summary.json"
    with open(summary_path, "w") as _f:
        json.dump({
            "study_name":      study_name,
            "model_type":      model_type,
            "pretrained_ckpt": str(pretrained_ckpt_path),
            "n_trials":        N_TRIALS,
            "best_trial":      best.number,
            "best_value":      best.value,
            "best_params":     best.params,
            "all_trials": [
                {
                    "number":     t.number,
                    "value":      t.value,
                    "params":     t.params,
                    "user_attrs": t.user_attrs,
                }
                for t in study.trials
            ],
        }, _f, indent=2, default=str)
    print(f"  HPO summary   : {summary_path}")
    return study


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: Supervised pretraining with fixed HPs.\n"
            "Phase 2: MAML++ HPO sweeping outer_lr × maml_inner_steps (15 trials).\n\n"
            "Model name is 'DeepCNNLSTM' throughout — the MoE layer is embedded inside it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_type", default="DeepCNNLSTM",
        choices=["MetaCNNLSTM", "DeepCNNLSTM", "TST", "ContrastiveNet"],
        help="Model architecture. DeepCNNLSTM = DeepCNNLSTM backbone with MoE encoder layer.",
    )
    parser.add_argument(
        "--skip_pretraining", action="store_true",
        help="Skip Phase 1 and use an existing checkpoint for Phase 2. "
             "Requires --pretrained_ckpt.",
    )
    parser.add_argument(
        "--pretrained_ckpt", type=str, default=None,
        help="Absolute path to an existing pretrained *_best.pt checkpoint. "
             "Only used when --skip_pretraining is set.",
    )
    args = parser.parse_args()

    # ── Seed ─────────────────────────────────────────────────────────────────
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.skip_pretraining:
        if args.pretrained_ckpt is None:
            parser.error("--skip_pretraining requires --pretrained_ckpt <path>")
        pretrained_ckpt_path = Path(args.pretrained_ckpt)
        if not pretrained_ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {pretrained_ckpt_path}\n"
                "Run without --skip_pretraining to produce one."
            )
        print()
        print(f"[Phase 1] SKIPPED — using existing checkpoint:")
        print(f"          {pretrained_ckpt_path}")
        print()
    else:
        pretrained_ckpt_path = run_pretraining(args.model_type)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    run_maml_hpo(pretrained_ckpt_path, args.model_type)


if __name__ == "__main__":
    main()