# =============================================================================
# HPO v3  —  MAML+MoE  |  1-shot 3-way  |  50 inner-steps @ eval (FIXED)
# =============================================================================
# What changed from v2:
#   - maml_inner_steps_eval is FIXED to 50 (no longer HPO'd).
#     Rationale: 100-step eval caused meta-overfitting (epoch-0 was always best).
#     50 steps is now the canonical eval budget, matching ablation_config.py.
#   - Warm-start params are the top-10 from the 50-step HPO run you already have.
#   - Search spaces updated to reflect trends visible in the new warm-start data:
#       * cnn_base_filters / lstm_hidden re-opened for search (new top-10 show
#         more diversity here than the v2 "128/128 fixed" assumption).
#       * maml_use_lslr re-opened (new top-10 show mixed signal).
#       * use_maml_msl re-opened with 'hybrid' and False (new top-10 favour hybrid).
#       * maml_inner_steps range narrowed to [5,7,9,10] (9/10 appear in new data).
#       * outer_lr / wd lower bounds loosened slightly (new top-10 go a bit lower).
#       * MOE_ctx_out_dim / MOE_ctx_hidden_dim re-opened (new top-10 show 16-128 spread).
#       * MOE_dropout re-opened (new top-10 show 0.001-0.17 spread).
#       * label_smooth fixed to 0.05 (every single new top-10 trial uses it).
#       * episodes_per_epoch_train range updated to [100, 200, 250, 500].
#   - V3_SUGGEST_KEYS updated accordingly.
#   - reorient_tensor_dict added to data loading (matches ablation_config.py).
# =============================================================================

# =============================================================================
# WARM-START CONFIGURATION
# =============================================================================
# Top-10 param dicts from the 50-step-eval HPO run.
# Keys not actively suggest_*'d in v3 are dropped before enqueuing.
WARM_START_PARAMS: list[dict] = [
    {'cnn_base_filters': 64, 'lstm_hidden': 64, 'maml_inner_steps': 7, 'maml_alpha_init': 0.0016865261840566302, 'maml_alpha_init_eval': 0.03241222861959444, 'outer_lr': 0.00011753148144028081, 'wd': 0.0006958201039866241, 'groupnorm_num_groups': 8, 'use_GlobalAvgPooling': True, 'num_experts': 25, 'MOE_top_k': 6, 'MOE_gate_temperature': 0.5007953923754159, 'MOE_aux_coeff': 0.17418352079333946, 'MOE_ctx_out_dim': 32, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.10372039932801176, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 200, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 36, 'maml_use_lslr': True, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 64, 'lstm_hidden': 128, 'maml_inner_steps': 7, 'maml_alpha_init': 0.003183796421686842, 'maml_alpha_init_eval': 0.02561505243517187, 'outer_lr': 0.00017730562335905792, 'wd': 0.0009623784816096662, 'groupnorm_num_groups': 8, 'use_GlobalAvgPooling': True, 'num_experts': 23, 'MOE_top_k': 4, 'MOE_gate_temperature': 0.5894698227360172, 'MOE_aux_coeff': 0.13261101041518134, 'MOE_ctx_out_dim': 64, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.1742031578572473, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 250, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 23, 'maml_use_lslr': False, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 64, 'lstm_hidden': 64, 'maml_inner_steps': 9, 'maml_alpha_init': 0.002708370774699923, 'maml_alpha_init_eval': 0.0021108128739609285, 'outer_lr': 0.0001951781386901919, 'wd': 6.85027628401306e-05, 'groupnorm_num_groups': 8, 'use_GlobalAvgPooling': True, 'num_experts': 27, 'MOE_top_k': 3, 'MOE_gate_temperature': 0.9458878022205542, 'MOE_aux_coeff': 0.16641221051826932, 'MOE_ctx_out_dim': 16, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.08151144309939662, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 100, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 17, 'maml_use_lslr': True, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 64, 'lstm_hidden': 256, 'maml_inner_steps': 7, 'maml_alpha_init': 0.0013373621508281483, 'maml_alpha_init_eval': 0.06693266709636093, 'outer_lr': 0.00023241537465524889, 'wd': 0.0009827548412804656, 'groupnorm_num_groups': 8, 'use_GlobalAvgPooling': True, 'num_experts': 24, 'MOE_top_k': 7, 'MOE_gate_temperature': 1.0459149632012676, 'MOE_aux_coeff': 0.11694630533248768, 'MOE_ctx_out_dim': 128, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.024707209736742966, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 500, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 26, 'maml_use_lslr': False, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 64, 'lstm_hidden': 64, 'maml_inner_steps': 7, 'maml_alpha_init': 0.001282949149676896, 'maml_alpha_init_eval': 0.006278649366666367, 'outer_lr': 0.00014911507147964058, 'wd': 0.0003105413582509981, 'groupnorm_num_groups': 8, 'use_GlobalAvgPooling': True, 'num_experts': 26, 'MOE_top_k': 5, 'MOE_gate_temperature': 0.827815891577418, 'MOE_aux_coeff': 0.09490439617134293, 'MOE_ctx_out_dim': 16, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.09440229157373216, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 100, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 30, 'maml_use_lslr': True, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 96, 'lstm_hidden': 64, 'maml_inner_steps': 7, 'maml_alpha_init': 0.0036027098514744647, 'maml_alpha_init_eval': 0.028273543982439742, 'outer_lr': 0.00019567289773981725, 'wd': 0.0006080214276443864, 'groupnorm_num_groups': 4, 'use_GlobalAvgPooling': True, 'num_experts': 22, 'MOE_top_k': 6, 'MOE_gate_temperature': 0.5166154000372265, 'MOE_aux_coeff': 0.11646849831998997, 'MOE_ctx_out_dim': 32, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.07000153139552596, 'MOE_aux_loss_plcmt': 'both', 'episodes_per_epoch_train': 200, 'label_smooth': 0.05, 'use_maml_msl': False, 'maml_use_lslr': True, 'use_lslr_at_eval': False},
    {'cnn_base_filters': 64, 'lstm_hidden': 128, 'maml_inner_steps': 7, 'maml_alpha_init': 0.023934012189321143, 'maml_alpha_init_eval': 0.0010860796928206904, 'outer_lr': 0.00010689736539096814, 'wd': 2.586297450657217e-05, 'groupnorm_num_groups': 4, 'use_GlobalAvgPooling': True, 'num_experts': 28, 'MOE_top_k': 3, 'MOE_gate_temperature': 2.431471657932191, 'MOE_aux_coeff': 0.10087273019596443, 'MOE_ctx_out_dim': 16, 'MOE_ctx_hidden_dim': 32, 'MOE_dropout': 0.10564140269584042, 'MOE_aux_loss_plcmt': 'both', 'episodes_per_epoch_train': 100, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 1, 'maml_use_lslr': True, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 96, 'lstm_hidden': 256, 'maml_inner_steps': 7, 'maml_alpha_init': 0.001427354132007504, 'maml_alpha_init_eval': 0.05871583020447932, 'outer_lr': 0.00021890580109728611, 'wd': 0.00015067531388534113, 'groupnorm_num_groups': 4, 'use_GlobalAvgPooling': True, 'num_experts': 24, 'MOE_top_k': 8, 'MOE_gate_temperature': 1.037361717564086, 'MOE_aux_coeff': 0.06945893667706689, 'MOE_ctx_out_dim': 128, 'MOE_ctx_hidden_dim': 128, 'MOE_dropout': 0.00014882237603539017, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 500, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 19, 'maml_use_lslr': False, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 64, 'lstm_hidden': 256, 'maml_inner_steps': 7, 'maml_alpha_init': 0.0022409316363444297, 'maml_alpha_init_eval': 0.07980284866764323, 'outer_lr': 0.00027288242632654975, 'wd': 0.0006233783942107558, 'groupnorm_num_groups': 8, 'use_GlobalAvgPooling': True, 'num_experts': 27, 'MOE_top_k': 6, 'MOE_gate_temperature': 1.9588295665204578, 'MOE_aux_coeff': 0.2274257229744778, 'MOE_ctx_out_dim': 128, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.06693040779691198, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 500, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 27, 'maml_use_lslr': False, 'use_lslr_at_eval': True},
    {'cnn_base_filters': 64, 'lstm_hidden': 64, 'maml_inner_steps': 7, 'maml_alpha_init': 0.0010018043002410384, 'maml_alpha_init_eval': 0.006165059688105781, 'outer_lr': 0.00015557309153358813, 'wd': 0.0003484429564449149, 'groupnorm_num_groups': 8, 'use_GlobalAvgPooling': True, 'num_experts': 29, 'MOE_top_k': 5, 'MOE_gate_temperature': 0.7058908557205739, 'MOE_aux_coeff': 0.0971830179235051, 'MOE_ctx_out_dim': 16, 'MOE_ctx_hidden_dim': 64, 'MOE_dropout': 0.09089303028735389, 'MOE_aux_loss_plcmt': 'inner', 'episodes_per_epoch_train': 100, 'label_smooth': 0.05, 'use_maml_msl': 'hybrid', 'maml_msl_num_epochs': 33, 'maml_use_lslr': True, 'use_lslr_at_eval': True},
]

import os
N_TRIALS = int(os.environ.get("N_TRIALS", 1))
FIXED_SEED = 42
import argparse
import copy, json, time
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend
import random
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# env -> Path objects
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()
print(f"CODE_DIR: {CODE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RUN_DIR:  {RUN_DIR}")

# === SAVING (to SCRATCH) ===
results_save_dir = RUN_DIR
models_save_dir  = RUN_DIR
results_save_dir.mkdir(parents=True, exist_ok=True)
models_save_dir.mkdir(parents=True, exist_ok=True)

# === LOADING ===
user_split_json_filepath = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"

def apply_fold_to_config(config, all_splits, fold_idx):
    """Mutates config in-place to set train/val/test PIDs for the given fold."""
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]

from system.MAML.mamlpp import *
from system.MAML.maml_data_pipeline import get_maml_dataloaders, reorient_tensor_dict
from system.MAML.shared_maml import *

from system.pretraining.pretrain_models import build_model

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# ── Collapse detection constants (MoE only) ──────────────────────────────────
COLLAPSE_MAX_LOAD_THRESHOLD = 0.80
COLLAPSE_PENALTY            = 0.0

#############################################################

def inject_model_config(config: dict, model_type: str,
                        cnn_base_filters: int = None,
                        lstm_hidden: int = None):
    """
    Injects the exact architecture parameters used during pretraining.
    cnn_base_filters / lstm_hidden:
        When pretrain_approach == 'None', these are HPO'd and passed in.
        When loading pretrained weights they MUST match the checkpoint.
    """
    config["model_type"] = model_type
    config["sequence_length"] = 64
    config['emg_in_ch'] = 16
    config['imu_in_ch'] = 72
    config['demo_in_dim'] = 12

    if model_type == "DeepCNNLSTM":
        _cnn_base_filters = cnn_base_filters if cnn_base_filters is not None else 64
        _lstm_hidden      = lstm_hidden      if lstm_hidden      is not None else 64
        config.update({
            "cnn_base_filters": _cnn_base_filters, "cnn_layers": 3,
            "cnn_kernel": 5, "groupnorm_num_groups": config.get("groupnorm_num_groups", 8),
            "lstm_hidden": _lstm_hidden, "lstm_layers": 3, "bidirectional": True,
            "head_type": "mlp",
        })
    else:
        raise ValueError(f"inject_model_config: unsupported model_type='{model_type}'. "
                         "Only 'DeepCNNLSTM' is supported in this HPO script.")

    return config


def _check_moe_collapse(history_or_logs: dict, num_experts: int) -> float | None:
    """
    Read max_expert_load from the last routing_report in a history/logs dict.
    Returns the max_load (0-1) if found, else None.
    """
    reports = history_or_logs.get("routing_reports", [])
    if not reports:
        return None
    last = reports[-1]
    for key in ("max_expert_load", "dominant_fraction", "max_load"):
        if key in last:
            return float(last[key])
    lb = last.get("load_balance", {})
    fracs = lb.get("expert_hard_fraction")
    if fracs:
        return float(max(fracs))
    imbal = lb.get("hard_imbalance_ratio")
    if imbal is not None:
        return float(imbal / (num_experts + imbal - 1))
    return None


# ===================== OPTUNA TUNING SCRIPT =====================
def build_model_from_trial(trial, model_type, base_config=None):
    config = copy.deepcopy(base_config) if base_config else {}

    # === MOE: always on ===
    config["use_MOE"] = True

    # === No pretrained weights — free HPO over arch ===
    config["pretrain_approach"] = "None"
    config["pretrained_model_filename"] = None

    # ── Architecture: re-open cnn/lstm search based on new warm-start trends ──
    # New top-10 show: cnn_base_filters ∈ {64, 96}, lstm_hidden ∈ {64, 128, 256}.
    # v2 had these fixed at 128/128 (old study trend); new data clearly disagrees.
    _cnn_base_filters = trial.suggest_categorical("cnn_base_filters", [64, 96, 128])
    _lstm_hidden      = trial.suggest_categorical("lstm_hidden",      [64, 128, 256])
    config = inject_model_config(config, model_type,
                                 cnn_base_filters=_cnn_base_filters,
                                 lstm_hidden=_lstm_hidden)

    # === Task Setup: 1-shot 3-way (fixed for this study) ===
    config["n_way"]       = 3
    config["k_shot"]      = 1
    config["q_query"]     = 9
    config["num_classes"] = 10
    config["pretrain_num_classes"] = 10

    config["feature_engr"] = "None"

    config["NOTS"] = True
    config["user_split_json_filepath"] = user_split_json_filepath
    config["results_save_dir"]         = results_save_dir
    config["models_save_dir"]          = models_save_dir
    config["emg_imu_pkl_full_path"]    = str(CODE_DIR / "dataset" / "filtered_datasets"
                                             / "metadata_IMU_EMG_allgestures_allusers.pkl")
    config["dfs_save_path"]   = str(CODE_DIR / "dataset") + "/"
    config["dfs_load_path"]   = str(CODE_DIR / "dataset" / "meta-learning-sup-que-ds") + "/"
    config["pretrain_dir"]    = str(CODE_DIR / "pretrain_outputs" / "checkpoints") + "/"

    # DEBUG
    config["track_gradient_alignment"] = False
    config["debug_verbose"]            = False
    config["debug_one_user_only"]      = False
    config["debug_one_episode"]        = False
    config["debug_five_episodes"]      = False
    config["gradient_clip_max_norm"]   = 10.0
    config["num_eval_episodes"]        = 100
    config["meta_batchsize"]           = 24   # v2 trend: 24 dominant → fixed

    # === MAML Core Hyperparameters ===
    # New warm-start: inner_steps mostly 7, with 9 present → keep range [5,7,9,10].
    config["maml_inner_steps"] = trial.suggest_categorical("maml_inner_steps", [5, 7, 9, 10])

    # FIXED to 50. This is the whole point of v3 — no more HPO over eval steps.
    # 100 steps caused meta-overfitting (epoch-0 val always best due to easy fitting
    # of a random-init model to val user). 50 steps is the ablation-spec canonical value.
    config["maml_inner_steps_eval"] = 50

    # New warm-start: alpha_init spans 0.001-0.024 but cluster is 0.001-0.004.
    # Trial 7 (index 6) has an outlier at 0.024 — keep upper bound at 0.025 to
    # allow TPE to explore but not over-commit to that outlier.
    config["maml_alpha_init"] = trial.suggest_float("maml_alpha_init", 5e-4, 0.025, log=True)

    # New warm-start: alpha_init_eval spans 0.001-0.08 → open range.
    config["maml_alpha_init_eval"] = trial.suggest_float("maml_alpha_init_eval", 0.001, 0.08, log=True)

    # New warm-start: outer_lr spans ~1e-4 to ~3e-4; extend lower end slightly.
    config["learning_rate"] = trial.suggest_float("outer_lr", 5e-5, 5e-4, log=True)

    # New warm-start: wd spans 2.6e-5 to ~1e-3 → widen the range.
    config["weight_decay"] = trial.suggest_float("wd", 1e-5, 1e-3, log=True)

    config["device"]          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["use_batch_norm"]  = False
    # New warm-start: 8 dominant (8 of 10 trials) → keep both but TPE will figure it out.
    config["groupnorm_num_groups"] = trial.suggest_categorical("groupnorm_num_groups", [4, 8])
    config["dropout"]         = 0.1
    config["emg_stride"]      = 1
    config["imu_stride"]      = 1
    config["padding"]         = 0

    # New warm-start: True in all 10 trials → still sweep to be safe, but prior is strong.
    config["use_GlobalAvgPooling"] = True  #trial.suggest_categorical("use_GlobalAvgPooling", [True, False])

    # === Multimodal ===
    config["multimodal"]               = True
    config["use_imu"]                  = True
    config["use_demographics"]         = False
    config["use_film_x_demo"]          = False
    config["FILM_on_context_or_demo"]  = "context"

    # === MOE (Mixture of Experts) ===
    # New warm-start: num_experts in {22..29} → tighten around that cluster.
    # Keep some headroom above 30 in case the old v2 {30-40} range was actually fine
    # but just hadn't converged yet with the bad eval setup.
    config["num_experts"] = trial.suggest_categorical(
        "num_experts", [20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 36, 40, 44])

    # New warm-start: top_k ∈ {3,4,5,6,7,8} → match this range.
    config["MOE_top_k"]   = trial.suggest_categorical("MOE_top_k", [3, 4, 5, 6, 7, 8, 9, 10])
    config["top_k"]       = config["MOE_top_k"]

    config["MOE_placement"] = "encoder"  # fixed per all studies

    # New warm-start: gate_temperature spans 0.5-2.4 → widen upper bound.
    config["MOE_gate_temperature"] = trial.suggest_float("MOE_gate_temperature", 0.3, 2.5, log=True)

    # New warm-start: aux_coeff spans 0.07-0.23 → widen range vs v2.
    config["MOE_aux_coeff"] = trial.suggest_float("MOE_aux_coeff", 0.02, 0.25, log=True)

    # New warm-start: ctx_out_dim ∈ {16, 32, 64, 128} → re-open.
    config["MOE_ctx_out_dim"] = trial.suggest_categorical("MOE_ctx_out_dim", [16, 32, 64, 128])

    # New warm-start: ctx_hidden_dim ∈ {32, 64, 128} → re-open.
    config["MOE_ctx_hidden_dim"] = trial.suggest_categorical("MOE_ctx_hidden_dim", [32, 64, 128])

    # New warm-start: dropout spans 0.00015-0.174 → re-open as continuous.
    config["MOE_dropout"] = trial.suggest_float("MOE_dropout", 1e-4, 0.2, log=True)

    config["MOE_expert_expand"]  = 1.0
    config["MOE_mlp_hidden_mult"] = 1.0
    config["MOE_log_every"]      = 5
    config["MOE_plot_dir"]       = None
    config["gate_type"]          = "context_feature_demo"
    config["expert_architecture"] = "MLP"

    # New warm-start: 'inner' in 8/10 trials → strong prior, but keep 'both' available.
    config["apply_MOE_aux_loss_inner_outer"] = trial.suggest_categorical(
        "MOE_aux_loss_plcmt", ["inner", "both", "outer"])

    config["use_label_shuf_meta_aug"] = True
    config["num_epochs"] = 50

    # New warm-start: episodes_per_epoch ∈ {100, 200, 250, 500} → match exactly.
    config["episodes_per_epoch_train"] = trial.suggest_categorical(
        "episodes_per_epoch_train", [100, 200, 250, 500])

    # New warm-start: label_smooth = 0.05 in ALL 10 trials → fixed.
    config["label_smooth"] = 0.05

    config["maml_gesture_classes"]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["target_trial_indices"]  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["train_reps"]            = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["val_reps"]              = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["augment"]               = False
    config["ft_label_smooth"]       = 0.0

    # Pretraining optim
    config["optimizer"]               = "adam"
    config["use_earlystopping"]       = True
    config["earlystopping_patience"]  = 8
    config["earlystopping_min_delta"] = 0.005

    # MAML misc
    config["meta_learning"] = True
    config["num_workers"]   = 8
    config["batch_size"]    = 64

    config["seed"] = FIXED_SEED
    config["timestamp"] = TIMESTAMP
    config["available_gesture_classes"] = config["maml_gesture_classes"]

    # MULTI-STEP LOSS
    # New warm-start: 'hybrid' in 8/10 trials, False in 2 → sweep both.
    _use_maml_msl = trial.suggest_categorical("use_maml_msl", ["hybrid", False])
    config["use_maml_msl"] = _use_maml_msl
    if _use_maml_msl == "hybrid":
        # New warm-start: maml_msl_num_epochs spans 1-36 → keep full range.
        config["maml_msl_num_epochs"] = trial.suggest_int("maml_msl_num_epochs", 1, 40)
    else:
        config["maml_msl_num_epochs"] = 0

    # OPTIMIZATION ORDER: first-order only (speed + stability).
    config["maml_opt_order"] = "first"
    config["maml_first_order_to_second_order_epoch"] = 1_000_000

    # LSLR
    # New warm-start: maml_use_lslr is True in 7/10 trials (mixed signal) → sweep.
    config["maml_use_lslr"] = trial.suggest_categorical("maml_use_lslr", [True, False])

    config["enable_inner_loop_optimizable_bn_params"] = False

    # use_lslr_at_eval
    # New warm-start: True in 9/10 trials → strong prior but keep sweep cheap.
    config["use_lslr_at_eval"] = trial.suggest_categorical("use_lslr_at_eval", [True, False])

    config["use_cosine_outer_lr"]   = False
    config["lr_scheduler_factor"]   = 0.1
    config["lr_scheduler_patience"] = 6

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    from MOE.MOE_encoder import build_MOE_model
    model = build_MOE_model(config)

    print(f"--> pretrain_approach='None' — using random initialisation for {model_type}.")
    model.to(config["device"])
    return model, config


#############################################################
# ---------- Load splits once ----------
with open(user_split_json_filepath, "r") as f:
    ALL_SPLITS = json.load(f)
NUM_FOLDS = 1

BASE_CONFIG = {}

def objective(trial, model_type):
    """Optuna objective wrapped to accept model_type."""
    fold_mean_accs    = []
    all_fold_user_accs = []
    pretrain_val_accs  = []

    for fold_idx in range(NUM_FOLDS):
        fold_start_time = time.time()

        print("=" * 80)
        print(f"[Trial {trial.number}] Starting fold {fold_idx + 1}/{NUM_FOLDS} for model {model_type}")
        print("=" * 80)

        # ---- Build model + config for this trial/fold ----
        model, config = build_model_from_trial(trial, model_type, base_config=BASE_CONFIG)

        print("\nCONFIG:")
        print(config)
        print("\n")

        if config["device"].type == "cpu":
            print("HPO is happening on the CPU! Probably ought to switch to GPU!")

        apply_fold_to_config(config, ALL_SPLITS, fold_idx)

        # ---- Data Loading ----
        # reorient_tensor_dict is called inside get_maml_dataloaders, but we also
        # need it for the per-user adapt-and-eval loop below. Load the raw dict
        # once here and pass the path; the dataloader handles its own reorientation.
        tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
        episodic_train_loader, episodic_val_loader = get_maml_dataloaders(
            config,
            tensor_dict_path=tensor_dict_path,
        )

        # ---- MAML Meta-Training ----
        pretrained_model, pretrain_res_dict = mamlpp_pretrain(
            model,
            config,
            episodic_train_loader,
            episodic_val_loader=episodic_val_loader,
        )
        best_val_acc = pretrain_res_dict["best_val_acc"]
        best_state   = pretrain_res_dict["best_state"]
        pretrain_val_accs.append(float(best_val_acc))

        print(f"[Trial {trial.number} | Fold {fold_idx}] Meta-training done. Best val acc = {best_val_acc:.4f}")

        # ---- MoE collapse detection ----
        if config.get("use_MOE", False):
            max_load = _check_moe_collapse(pretrain_res_dict, num_experts=config["num_experts"])
            trial.set_user_attr("final_max_expert_load",
                                max_load if max_load is not None else -1.0)
            if max_load is not None and max_load > COLLAPSE_MAX_LOAD_THRESHOLD:
                print(f"  [Trial {trial.number} | Fold {fold_idx}] MoE COLLAPSED "
                      f"(max_load={max_load:.2f}). Penalising.")
                return COLLAPSE_PENALTY

        # ---- Save checkpoint ----
        model_filename = f"trial_{trial.number}_fold_{fold_idx}_best.pt"
        save_path = os.path.join(models_save_dir, model_filename)
        torch.save({
            "trial_num":       trial.number,
            "fold_idx":        fold_idx,
            "model_state_dict": best_state,
            "config":          config,
            "best_val_acc":    best_val_acc,
            "train_loss_log":  pretrain_res_dict["train_loss_log"],
            "train_acc_log":   pretrain_res_dict["train_acc_log"],
            "val_loss_log":    pretrain_res_dict["val_loss_log"],
            "val_acc_log":     pretrain_res_dict["val_acc_log"],
        }, save_path)
        print(f"Model saved to {save_path}")

        # ---- Per-user adapt-and-eval (HPO objective signal) ----
        model.load_state_dict(best_state)
        user_metrics = defaultdict(list)

        for batch in episodic_val_loader:
            user_id     = batch["user_id"]
            support_set = batch["support"]
            query_set   = batch["query"]
            val_metrics = mamlpp_adapt_and_eval(model, config, support_set, query_set)
            user_metrics[user_id].append(val_metrics["acc"])

        all_user_means = []
        for user_id, accs in user_metrics.items():
            m_acc = np.mean(accs)
            all_user_means.append(float(m_acc))
            print(f"  User {user_id} | Acc: {m_acc*100:.2f}% (over {len(accs)} episodes)")

        mean_acc_ratio  = np.mean(all_user_means)
        std_acc_ratio   = np.std(all_user_means)
        user_acc_pcts   = [round(a * 100, 2) for a in all_user_means]
        fold_duration   = time.time() - fold_start_time

        print(f"[Trial {trial.number} | Fold {fold_idx}] User accs (%): {user_acc_pcts}")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Mean acc: {mean_acc_ratio*100:.2f}% ± {std_acc_ratio*100:.2f}%")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Finished in {fold_duration:.2f} seconds.")

        fold_mean_accs.append(mean_acc_ratio)
        all_fold_user_accs.append(all_user_means)

    clean_fold_accs  = [float(f) for f in fold_mean_accs]
    overall_mean_acc = float(np.nanmean(clean_fold_accs))

    trial.set_user_attr("fold_mean_accs",        fold_mean_accs)
    trial.set_user_attr("fold_user_accs",         all_fold_user_accs)
    trial.set_user_attr("mean_pretrain_val_acc",  float(np.nanmean(pretrain_val_accs)))

    return overall_mean_acc


def _build_warm_start_params(raw_params: dict, trial_suggest_keys: set) -> dict:
    """
    Filter a raw param dict from the old study so it only contains keys that are
    actively suggest_*'d in the new study. Keys that are no longer suggest_*'d
    (e.g. maml_inner_steps_eval, which is now fixed) are dropped so Optuna does
    not raise an UnexpectedParameter error when enqueuing.
    """
    return {k: v for k, v in raw_params.items() if k in trial_suggest_keys}


# The set of HP keys actively suggest_*'d in build_model_from_trial v3.
# NOTE: maml_inner_steps_eval is intentionally absent — it is fixed to 50.
# NOTE: label_smooth is intentionally absent — it is fixed to 0.05.
# NOTE: maml_msl_num_epochs uses suggest_int but is only meaningful when
#       use_maml_msl == 'hybrid'; it is still in the suggest-key set so that
#       warm-start trials with hybrid can carry their value across.
V3_SUGGEST_KEYS = {
    "cnn_base_filters",
    "lstm_hidden",
    "maml_inner_steps",
    "maml_alpha_init",
    "maml_alpha_init_eval",
    "outer_lr",
    "wd",
    "groupnorm_num_groups",
    #"use_GlobalAvgPooling",  # Commented out since it appears that use_GAP is a unanimous True
    "num_experts",
    "MOE_top_k",
    "MOE_gate_temperature",
    "MOE_aux_coeff",
    "MOE_ctx_out_dim",
    "MOE_ctx_hidden_dim",
    "MOE_dropout",
    "MOE_aux_loss_plcmt",
    "episodes_per_epoch_train",
    "use_maml_msl",
    "maml_msl_num_epochs",
    "maml_use_lslr",
    "use_lslr_at_eval",
}


def run_study(study_name, storage_path, model_type, n_trials=1):
    sleep_time = random.uniform(0, 10)
    print(f"Staggering start: sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)

    use_journal = int(os.environ["HPO_USE_JOURNAL"])
    if use_journal:
        lock_obj = JournalFileBackend(storage_path)
        storage  = JournalStorage(lock_obj)
        print(f"Journal storage enabled: {storage_path}")
    else:
        storage = optuna.storages.InMemoryStorage()
        print("Journal storage DISABLED (debug mode) — using InMemoryStorage.")

    time.sleep(random.uniform(0, 10))

    # ── TPE Sampler ──────────────────────────────────────────────────────────
    # n_startup_trials: at least as many as warm-start trials so TPE sees the full
    # prior before it starts modelling. Rule of thumb: ~10-20% of total budget.
    n_startup = max(20, len(WARM_START_PARAMS))
    sampler = optuna.samplers.TPESampler(
        seed=FIXED_SEED,
        n_startup_trials=n_startup,
        n_ei_candidates=24,
        multivariate=True,    # model joint HP correlations (important for MoE/MAML)
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    # ── Warm-Start: enqueue top-N trials from the 50-step HPO run ────────────
    if WARM_START_PARAMS and len(study.trials) == 0:
        print(f"Warm-starting with {len(WARM_START_PARAMS)} trials from prior study...")
        for i, raw_params in enumerate(WARM_START_PARAMS):
            filtered = _build_warm_start_params(raw_params, V3_SUGGEST_KEYS)
            study.enqueue_trial(filtered)
            print(f"  Enqueued warm-start trial {i}: {filtered}")
    elif WARM_START_PARAMS and len(study.trials) > 0:
        print(f"Study already has {len(study.trials)} trials — skipping warm-start enqueue "
              f"(warm-start only applies to a fresh study).")

    study.optimize(lambda trial: objective(trial, model_type), n_trials=n_trials, gc_after_trial=True)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPO v3: MAML+MoE, 1-shot 3-way, maml_inner_steps_eval=50 (fixed).")
    parser.add_argument("--model_type", type=str, default="DeepCNNLSTM",
                        choices=["DeepCNNLSTM"],
                        help="Model architecture to optimize. Only DeepCNNLSTM is supported.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir",  type=str)
    args = parser.parse_args()

    db_dir = "/scratch/my13/kai/meta-pers-gest/optuna_dbs"
    os.makedirs(db_dir, exist_ok=True)

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    study_name   = f"mamlpp_MOE_PAPER_{args.model_type}_1fcv_50step_hpo"
    journal_path = os.path.join(db_dir, f"{study_name}.log")

    print(f"Starting HPO Study: {study_name}")
    print(f"Journal Path: {journal_path}")

    run_study(
        study_name=study_name,
        storage_path=journal_path,
        model_type=args.model_type,
        n_trials=N_TRIALS,
    )