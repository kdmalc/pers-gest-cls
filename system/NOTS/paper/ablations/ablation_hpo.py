# =============================================================================
# ablation_hpo.py
# ===============
# Unified HPO driver for ALL ablations in the MAML+MoE ablation study.
#
# Design rationale
# ----------------
# Every ablation shares the same DeepCNNLSTM backbone, so the search space is
# partitioned into four groups that are included/excluded per ablation:
#
#   SHARED    — LR, weight-decay. These are always tuned.
#               cnn_base_filters, lstm_hidden, and groupnorm_num_groups are
#               FIXED (not HPO'd) for all ablations. This ensures every
#               ablation uses an identical backbone architecture, making the
#               ablation table unambiguous: any accuracy difference is due to
#               MAML / MoE, not model size. Fixed values are derived from the
#               mode of the v3 warm-start top-10 trials (see FIXED_ARCH_* and
#               FIXED_GROUPNORM below). A4 is the sole exception: it uses
#               A4_CNN_BASE_FILTERS / A4_LSTM_HIDDEN (param-matched to the
#               full M0 expert bank) instead of the global fixed values.
#               label_smooth is in SHARED for supervised (non-MAML) ablations
#               only; it is fixed to 0.05 for MAML ablations (unanimous in
#               warm-start data).
#
#   MAML      — inner_steps, alpha_init, alpha_init_eval, LSLR, MSL,
#               episodes_per_epoch.
#               Included only when the ablation uses meta-learning.
#
#   MOE       — num_experts, top_k, gate_temperature, aux_coeff, ctx dims,
#               dropout, aux_loss placement.
#               Included only when the ablation uses MoE.
#
#   SUPERVISED — ft_lr, label_smooth (fine-tuning / supervised training HPs).
#               Included only for non-MAML ablations (A1, A2, A7).
#
#   A11_ONLY  — ft_lr, ft_steps only. A11 uses a pretrained Meta model so
#               architecture / training HPs are irrelevant. Its objective
#               branch loads the pretrained weights directly and only searches
#               over fine-tuning settings.
#
# Ablation → profile mapping:
#   M0           → (True,  True )  full model
#   A1           → (False, True )  supervised + MoE
#   A2, A7       → (False, False)  supervised, no MoE
#   A3           → (True,  False)  MAML, no MoE  (original width)
#   A4           → (True,  False)  MAML, no MoE  (param-matched; arch dims FIXED)
#   A5           → (True,  True )  expert-count sweep (same space as M0)
#   A8           → (True,  True )  subject-specific MAML+MoE (same space as M0)
#   A11          → special         Meta pretrained; ft_lr + ft_steps only
#   A12          → (True,  True )  Our model on 2kHz data (same space as M0)
#
# A10 is excluded: zero-shot protocol, no learnable HPs to tune.
#
# Usage (one trial per job — N_TRIALS=1 is set by the SLURM array script):
#   python ablation_hpo.py --ablation M0
#   python ablation_hpo.py --ablation A1
#   python ablation_hpo.py --ablation A3
#   ... etc.
# =============================================================================

import os
import sys
import copy
import json
import time
import random
import pickle
import argparse
import warnings
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# Environment / paths
# =============================================================================

FIXED_SEED = 42
N_TRIALS   = int(os.environ.get("N_TRIALS", 1))

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()

print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"CODE_DIR       : {CODE_DIR}")
print(f"DATA_DIR       : {DATA_DIR}")
print(f"RUN_DIR        : {RUN_DIR}")

RUN_DIR.mkdir(parents=True, exist_ok=True)

for _p in [CODE_DIR, CODE_DIR / "system", CODE_DIR / "system" / "MAML",
           CODE_DIR / "system" / "MOE", CODE_DIR / "system" / "pretraining"]:
    sys.path.insert(0, str(_p))

HPO_DB_DIR = Path(os.environ.get(
    "HPO_DB_DIR",
    "/scratch/my13/kai/meta-pers-gest/optuna_dbs"
))
HPO_DB_DIR.mkdir(parents=True, exist_ok=True)

# MoE collapse detection (same as v3 HPO script)
COLLAPSE_MAX_LOAD_THRESHOLD = 0.80
COLLAPSE_PENALTY            = 0.0

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# =============================================================================
# Fixed architecture constants  (shared by ALL ablations except A4)
# =============================================================================
# Derived from the mode of the v3 warm-start top-10 trials:
#   cnn_base_filters : 64  (8/10 trials)
#   lstm_hidden      : 64  (4/10 trials; 256 appears 3×, 128 appears 2× —
#                           distribution is noisy but 64 is the mode)
#   groupnorm_num_groups : 8  (8/10 trials)
#
# If you want to change the canonical architecture, edit these three values.
# Every ablation (except A4) will automatically use them.
FIXED_CNN_BASE_FILTERS:    int = 64
FIXED_LSTM_HIDDEN:         int = 64
FIXED_GROUPNORM_NUM_GROUPS: int = 8

# =============================================================================
# A4: Param-matched encoder dims
# =============================================================================
# A4 is the ONLY ablation that overrides the fixed arch above.
# Rule: encoder_params(A4) ≈ num_experts_M0 × params(one expert in M0).
# Scale cnn_base_filters and lstm_hidden proportionally; do NOT add layers
# (depth changes confound the comparison).
#
# How to compute these values:
#   1. After running M0 HPO, instantiate a single DeepCNNLSTM with
#      FIXED_CNN_BASE_FILTERS / FIXED_LSTM_HIDDEN and count its parameters:
#        one_expert_params = count_parameters(single_expert_model)
#   2. Multiply by the best num_experts from M0 HPO:
#        target = num_experts_M0_best * one_expert_params
#   3. Grid-search (cnn_base_filters, lstm_hidden) to find the combo whose
#      DeepCNNLSTM param count is within ~5% of target.  Example snippet:
#
#        for f in [128, 192, 256, 320, 384, 512]:
#            for h in [128, 192, 256, 384, 512, 640]:
#                cfg["cnn_base_filters"] = f; cfg["lstm_hidden"] = h
#                p = count_parameters(build_model(cfg))
#                if abs(p - target) / target < 0.05:
#                    print(f"cnn={f}, lstm={h}: {p:,}")
#
#   4. Fill the winning values in below, then run A4 HPO.
A4_CNN_BASE_FILTERS: int = 128   # TODO: set after step 3 above
A4_LSTM_HIDDEN: int      = 128   # TODO: set after step 3 above

# =============================================================================
# A11 / A12: 2kHz EMG-only data settings
# =============================================================================
# Must match the values in A10_A11_A12_meta_pretrained.py exactly.
EMG_2KHZ_PKL_PATH  = "/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/segfilt_2khz_emg_tensor_dict.pkl"
EMG_2KHZ_IN_CH     = 16
EMG_2KHZ_SEQ_LEN   = 4300
META_CHECKPOINT_PATH = Path("/rhf/allocations/my13/emg_models/discrete_gestures/model_checkpoint.ckpt")

# =============================================================================
# Ablation profile table
# =============================================================================

ABLATION_PROFILES: dict[str, dict] = {
    "M0": dict(
        use_maml=True,  use_moe=True,  use_sup_ft=False, is_a11=False, fix_arch=False,
        eval_mode="episodic",
        script="M0_full_model.py",
        note="Full model: MAML + MoE",
    ),
    "A1": dict(
        use_maml=False, use_moe=True,  use_sup_ft=True,  is_a11=False, fix_arch=False,
        eval_mode="supervised",
        script="A1_no_maml_moe.py",
        note="Supervised MoE (no MAML)",
    ),
    "A2": dict(
        use_maml=False, use_moe=False, use_sup_ft=True,  is_a11=False, fix_arch=False,
        eval_mode="supervised",
        script="A2_no_maml_no_moe.py",
        note="Vanilla supervised CNN-LSTM (no MAML, no MoE)",
    ),
    "A3": dict(
        use_maml=True,  use_moe=False, use_sup_ft=False, is_a11=False, fix_arch=False,
        eval_mode="episodic",
        script="A3_A4_maml_no_moe.py",
        note="MAML, no MoE, original width",
    ),
    "A4": dict(
        use_maml=True,  use_moe=False, use_sup_ft=False, is_a11=False, fix_arch=True,
        eval_mode="episodic",
        script="A3_A4_maml_no_moe.py",
        note="MAML, no MoE, param-matched encoder (arch dims FIXED — not HPO'd)",
    ),
    "A5": dict(
        use_maml=True,  use_moe=True,  use_sup_ft=False, is_a11=False, fix_arch=False,
        eval_mode="episodic",
        script="A5_expert_count_sweep.py",
        note="Expert-count sweep (same HP space as M0; num_experts matched to sweep values)",
    ),
    "A7": dict(
        use_maml=False, use_moe=False, use_sup_ft=True,  is_a11=False, fix_arch=False,
        eval_mode="supervised",
        script="A7_A8_subject_specific.py",
        note="Subject-specific supervised baseline",
    ),
    "A8": dict(
        use_maml=True,  use_moe=True,  use_sup_ft=False, is_a11=False, fix_arch=False,
        eval_mode="episodic",
        script="A7_A8_subject_specific.py",
        note="Subject-specific MAML+MoE",
    ),
    "A11": dict(
        use_maml=False, use_moe=False, use_sup_ft=False, is_a11=True,  fix_arch=False,
        eval_mode="a11",
        script="A10_A11_A12_meta_pretrained.py",
        note="Meta pretrained, 1-shot fine-tune (ft_lr + ft_steps only)",
    ),
    "A12": dict(
        use_maml=True,  use_moe=True,  use_sup_ft=False, is_a11=False, fix_arch=False,
        eval_mode="episodic",
        script="A10_A11_A12_meta_pretrained.py",
        note="Our MAML+MoE on 2kHz EMG-only data",
    ),
    # A10: zero-shot, no learnable HPs — intentionally omitted.
}

# =============================================================================
# Warm-start params  (M0 only — top-10 from the existing v3 50-step HPO run)
# =============================================================================

M0_WARM_START_PARAMS: list[dict] = [
    {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.0016865261840566302, "maml_alpha_init_eval": 0.03241222861959444, "outer_lr": 0.00011753148144028081, "wd": 0.0006958201039866241, "groupnorm_num_groups": 8,  "num_experts": 25, "MOE_top_k": 6,  "MOE_gate_temperature": 0.5007953923754159, "MOE_aux_coeff": 0.17418352079333946, "MOE_ctx_out_dim": 32,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.10372039932801176, "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 200, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 36, "maml_use_lslr": True,  "use_lslr_at_eval": True},
    {"cnn_base_filters": 64,  "lstm_hidden": 128, "maml_inner_steps": 7,  "maml_alpha_init": 0.003183796421686842,  "maml_alpha_init_eval": 0.02561505243517187,  "outer_lr": 0.00017730562335905792, "wd": 0.0009623784816096662, "groupnorm_num_groups": 8,  "num_experts": 23, "MOE_top_k": 4,  "MOE_gate_temperature": 0.5894698227360172, "MOE_aux_coeff": 0.13261101041518134, "MOE_ctx_out_dim": 64,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.1742031578572473,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 250, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 23, "maml_use_lslr": False, "use_lslr_at_eval": True},
    {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 9,  "maml_alpha_init": 0.002708370774699923,  "maml_alpha_init_eval": 0.0021108128739609285, "outer_lr": 0.0001951781386901919,  "wd": 6.85027628401306e-05,  "groupnorm_num_groups": 8,  "num_experts": 27, "MOE_top_k": 3,  "MOE_gate_temperature": 0.9458878022205542, "MOE_aux_coeff": 0.16641221051826932, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.08151144309939662,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 17, "maml_use_lslr": True,  "use_lslr_at_eval": True},
    {"cnn_base_filters": 64,  "lstm_hidden": 256, "maml_inner_steps": 7,  "maml_alpha_init": 0.0013373621508281483, "maml_alpha_init_eval": 0.06693266709636093,  "outer_lr": 0.00023241537465524889, "wd": 0.0009827548412804656, "groupnorm_num_groups": 8,  "num_experts": 24, "MOE_top_k": 7,  "MOE_gate_temperature": 1.0459149632012676, "MOE_aux_coeff": 0.11694630533248768, "MOE_ctx_out_dim": 128, "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.024707209736742966, "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 500, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 26, "maml_use_lslr": False, "use_lslr_at_eval": True},
    {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.001282949149676896,  "maml_alpha_init_eval": 0.006278649366666367,  "outer_lr": 0.00014911507147964058, "wd": 0.0003105413582509981, "groupnorm_num_groups": 8,  "num_experts": 26, "MOE_top_k": 5,  "MOE_gate_temperature": 0.827815891577418,  "MOE_aux_coeff": 0.09490439617134293, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.09440229157373216,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 30, "maml_use_lslr": True,  "use_lslr_at_eval": True},
    {"cnn_base_filters": 96,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.0036027098514744647, "maml_alpha_init_eval": 0.028273543982439742,  "outer_lr": 0.00019567289773981725, "wd": 0.0006080214276443864, "groupnorm_num_groups": 4,  "num_experts": 22, "MOE_top_k": 6,  "MOE_gate_temperature": 0.5166154000372265, "MOE_aux_coeff": 0.11646849831998997, "MOE_ctx_out_dim": 32,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.07000153139552596,  "MOE_aux_loss_plcmt": "both",  "episodes_per_epoch_train": 200, "use_maml_msl": False,      "maml_use_lslr": True,  "use_lslr_at_eval": False},
    {"cnn_base_filters": 64,  "lstm_hidden": 128, "maml_inner_steps": 7,  "maml_alpha_init": 0.023934012189321143,  "maml_alpha_init_eval": 0.0010860796928206904, "outer_lr": 0.00010689736539096814, "wd": 2.586297450657217e-05,  "groupnorm_num_groups": 4,  "num_experts": 28, "MOE_top_k": 3,  "MOE_gate_temperature": 2.431471657932191,  "MOE_aux_coeff": 0.10087273019596443, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 32,  "MOE_dropout": 0.10564140269584042,  "MOE_aux_loss_plcmt": "both",  "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 1,  "maml_use_lslr": True,  "use_lslr_at_eval": True},
    {"cnn_base_filters": 96,  "lstm_hidden": 256, "maml_inner_steps": 7,  "maml_alpha_init": 0.001427354132007504,  "maml_alpha_init_eval": 0.05871583020447932,  "outer_lr": 0.00021890580109728611, "wd": 0.00015067531388534113, "groupnorm_num_groups": 4,  "num_experts": 24, "MOE_top_k": 8,  "MOE_gate_temperature": 1.037361717564086,  "MOE_aux_coeff": 0.06945893667706689, "MOE_ctx_out_dim": 128, "MOE_ctx_hidden_dim": 128, "MOE_dropout": 0.00014882237603539017, "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 500, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 19, "maml_use_lslr": False, "use_lslr_at_eval": True},
    {"cnn_base_filters": 64,  "lstm_hidden": 256, "maml_inner_steps": 7,  "maml_alpha_init": 0.0022409316363444297, "maml_alpha_init_eval": 0.07980284866764323,  "outer_lr": 0.00027288242632654975, "wd": 0.0006233783942107558, "groupnorm_num_groups": 8,  "num_experts": 27, "MOE_top_k": 6,  "MOE_gate_temperature": 1.9588295665204578, "MOE_aux_coeff": 0.2274257229744778,  "MOE_ctx_out_dim": 128, "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.06693040779691198,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 500, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 27, "maml_use_lslr": False, "use_lslr_at_eval": True},
    {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.0010018043002410384, "maml_alpha_init_eval": 0.006165059688105781,  "outer_lr": 0.00015557309153358813, "wd": 0.0003484429564449149, "groupnorm_num_groups": 8,  "num_experts": 29, "MOE_top_k": 5,  "MOE_gate_temperature": 0.7058908557205739, "MOE_aux_coeff": 0.0971830179235051,  "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.09089303028735389,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 33, "maml_use_lslr": True,  "use_lslr_at_eval": True},
]

# Ablations that receive the M0 warm-start (identical HP space to M0's v3 study)
ABLATIONS_WITH_M0_WARMSTART = {"M0", "A5", "A8", "A12"}

# =============================================================================
# HP suggestion helpers
# =============================================================================

def _suggest_shared_hps(trial: optuna.Trial, config: dict, fix_arch: bool) -> dict:
    """
    Suggest HPs that are active for every ablation regardless of MAML/MoE.

    Architecture dims (cnn_base_filters, lstm_hidden, groupnorm_num_groups)
    are FIXED for all ablations — they are not HPO'd. This ensures every
    ablation uses an identical backbone so accuracy differences in the ablation
    table are unambiguously attributable to MAML / MoE, not model size.

    fix_arch=True (A4 only): use A4_CNN_BASE_FILTERS / A4_LSTM_HIDDEN instead
    of the global FIXED_* values. All other ablations use the global values.
    groupnorm is always FIXED_GROUPNORM_NUM_GROUPS regardless of fix_arch.
    """
    if fix_arch:
        # A4: param-matched encoder — different width than the global fixed arch
        config["cnn_base_filters"] = A4_CNN_BASE_FILTERS
        config["lstm_hidden"]      = A4_LSTM_HIDDEN
        print(f"  [A4] Arch dims FIXED (param-matched): "
              f"cnn_base_filters={A4_CNN_BASE_FILTERS}, lstm_hidden={A4_LSTM_HIDDEN}")
    else:
        config["cnn_base_filters"] = FIXED_CNN_BASE_FILTERS
        config["lstm_hidden"]      = FIXED_LSTM_HIDDEN

    config["groupnorm_num_groups"] = FIXED_GROUPNORM_NUM_GROUPS

    config["learning_rate"] = trial.suggest_float("outer_lr", 5e-5, 5e-4, log=True)
    config["weight_decay"]  = trial.suggest_float("wd",       1e-5, 1e-3, log=True)
    return config


def _suggest_label_smooth_supervised(trial: optuna.Trial, config: dict) -> dict:
    """
    Tune label_smooth for supervised (non-MAML) ablations.
    MAML warm-start was done entirely with 0.05, but supervised training
    is a different regime — open the range to [0.0, 0.5].
    """
    config["label_smooth"] = trial.suggest_float("label_smooth", 0.0, 0.5)
    return config


def _suggest_maml_hps(trial: optuna.Trial, config: dict) -> dict:
    """
    Suggest MAML-specific HPs. Only called when use_maml=True.
    Ranges mirror the v3 HPO script (1s3w_maml_moe_hpo.py) exactly.
    label_smooth is fixed to 0.05 for MAML ablations (unanimous in warm-start).
    """
    config["maml_inner_steps"]      = trial.suggest_categorical("maml_inner_steps", [5, 7, 9, 10])
    config["maml_inner_steps_eval"] = 50   # FIXED — see v3 rationale
    config["maml_alpha_init"]       = trial.suggest_float("maml_alpha_init",      5e-4, 0.025, log=True)
    config["maml_alpha_init_eval"]  = trial.suggest_float("maml_alpha_init_eval", 0.001, 0.08, log=True)
    config["maml_use_lslr"]         = trial.suggest_categorical("maml_use_lslr",   [True, False])
    config["use_lslr_at_eval"]      = trial.suggest_categorical("use_lslr_at_eval", [True, False])

    _use_maml_msl = trial.suggest_categorical("use_maml_msl", ["hybrid", False])
    config["use_maml_msl"] = _use_maml_msl
    if _use_maml_msl == "hybrid":
        config["maml_msl_num_epochs"] = trial.suggest_int("maml_msl_num_epochs", 1, 40)
    else:
        config["maml_msl_num_epochs"] = 0

    # Fixed MAML settings
    config["meta_batchsize"]         = 24
    config["maml_opt_order"]         = "first"
    config["maml_first_order_to_second_order_epoch"] = 1_000_000
    config["enable_inner_loop_optimizable_bn_params"] = False
    config["meta_learning"]          = True

    config["episodes_per_epoch_train"] = trial.suggest_categorical(
        "episodes_per_epoch_train", [100, 200, 250, 500])

    # label_smooth: fixed for MAML ablations (all v3 warm-start trials used 0.05)
    config["label_smooth"] = 0.05

    return config


def _suggest_moe_hps(trial: optuna.Trial, config: dict, ablation_id: str) -> dict:
    """
    Suggest MoE-specific HPs. Only called when use_moe=True.

    For A5 (expert-count sweep), num_experts is drawn from the same discrete
    set used by the sweep script so HPO explores the full mountain-curve range.
    For all other ablations, num_experts is drawn from the HPO-friendly
    coarser grid.
    """
    if ablation_id == "A5":
        # Match the sweep values exactly — HPO finds the best overall config
        # within this range; the sweep script then fixes num_experts to each
        # value and reuses the rest of the best config.
        config["num_experts"] = trial.suggest_categorical(
            "num_experts", [4, 8, 12, 16, 20, 24, 32, 40])
    else:
        config["num_experts"] = trial.suggest_categorical(
            "num_experts", [20, 22, 24, 25, 26, 27, 28, 30, 32, 36, 40, 44])

    config["MOE_top_k"]     = trial.suggest_categorical("MOE_top_k", [4, 5, 6, 7, 8, 9, 10])
    config["top_k"]         = config["MOE_top_k"]
    config["MOE_gate_temperature"] = trial.suggest_float(
        "MOE_gate_temperature", 0.3, 2.5, log=True)
    config["MOE_aux_coeff"]       = trial.suggest_float("MOE_aux_coeff", 0.02, 0.25, log=True)
    config["MOE_ctx_out_dim"]     = trial.suggest_categorical("MOE_ctx_out_dim",    [16, 32, 64, 128])
    config["MOE_ctx_hidden_dim"]  = trial.suggest_categorical("MOE_ctx_hidden_dim", [32, 64, 128])
    config["MOE_dropout"]         = trial.suggest_float("MOE_dropout", 1e-4, 0.2, log=True)
    config["apply_MOE_aux_loss_inner_outer"] = trial.suggest_categorical(
        "MOE_aux_loss_plcmt", ["inner", "both", "outer"])

    # Fixed MoE settings
    config["use_MOE"]              = True
    config["MOE_placement"]        = "encoder"
    config["MOE_expert_expand"]    = 1.0
    config["MOE_mlp_hidden_mult"]  = 1.0
    config["MOE_log_every"]        = 5
    config["MOE_plot_dir"]         = None
    config["gate_type"]            = "context_feature_demo"
    config["expert_architecture"]  = "MLP"

    return config


def _suggest_supervised_ft_hps(trial: optuna.Trial, config: dict) -> dict:
    """
    Suggest supervised fine-tuning HPs. Only called when use_sup_ft=True
    (A1, A2, A7).

    ft_lr: bracketed around MAML eval alpha range and slightly above.
    ft_steps: fixed at 50 to match maml_inner_steps_eval — identical
              adaptation budget across MAML and non-MAML ablations.
    """
    config["ft_lr"]           = trial.suggest_float("ft_lr", 5e-4, 0.1, log=True)
    config["ft_steps"]        = 50   # FIXED — mirrors maml_inner_steps_eval
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]
    return config


def _suggest_no_moe_fixed(config: dict) -> dict:
    """Apply fixed non-MoE config keys when use_moe=False."""
    config["use_MOE"] = False
    config["top_k"]   = 1   # unused but keep valid
    return config


def _suggest_no_maml_fixed(config: dict) -> dict:
    """Apply fixed non-MAML config keys when use_maml=False."""
    config["meta_learning"]          = False
    config["maml_inner_steps"]       = 0
    config["maml_inner_steps_eval"]  = 0
    config["maml_alpha_init"]        = 0.0
    config["maml_alpha_init_eval"]   = 0.0
    config["maml_use_lslr"]          = False
    config["use_lslr_at_eval"]       = False
    config["use_maml_msl"]           = False
    config["maml_msl_num_epochs"]    = 0
    config["meta_batchsize"]         = 0
    config["maml_opt_order"]         = "first"
    config["enable_inner_loop_optimizable_bn_params"] = False
    config["episodes_per_epoch_train"] = 200   # sentinel; flat DL ignores this
    return config


# =============================================================================
# Full config builder (called once per Optuna trial)
# =============================================================================

def build_config_from_trial(
    trial: optuna.Trial,
    ablation_id: str,
    profile: dict,
) -> dict:
    """
    Construct a full training config for one Optuna trial of a given ablation.

    Steps:
      1. Start from ablation_config.make_base_config (single source of truth).
      2. Overwrite every suggested HP — never leave base-config defaults for
         HPs that are being searched. Avoids silent stale-value bugs.
      3. Apply A12-specific overrides (2kHz data, no IMU, front_end_stride).
      4. Apply fixed values for HP groups that are NOT being searched.
    """
    from ablation_config import make_base_config

    config = make_base_config(ablation_id=ablation_id)

    config["results_save_dir"] = str(RUN_DIR)
    config["models_save_dir"]  = str(RUN_DIR)

    # ── Fixed architecture / task setup ──────────────────────────────────────
    config["n_way"]                 = 3
    config["k_shot"]                = 1
    config["q_query"]               = 9
    config["num_classes"]           = 10
    config["pretrain_num_classes"]  = 10
    config["num_eval_episodes"]     = 100   # HPO uses 100 val episodes (speed)
    config["num_epochs"]            = 50
    config["num_workers"]           = 8
    config["batch_size"]            = 64
    config["use_GlobalAvgPooling"]  = True
    config["dropout"]               = 0.1
    config["padding"]               = 0
    config["use_batch_norm"]        = False
    config["multimodal"]            = True
    config["use_imu"]               = True
    config["use_demographics"]      = False
    config["use_film_x_demo"]       = False
    config["FILM_on_context_or_demo"] = "context"
    config["optimizer"]             = "adam"
    config["use_cosine_outer_lr"]   = False
    config["lr_scheduler_factor"]   = 0.1
    config["lr_scheduler_patience"] = 6
    config["use_earlystopping"]     = True
    config["earlystopping_patience"]  = 8
    config["earlystopping_min_delta"] = 0.005
    config["use_label_shuf_meta_aug"] = True
    config["gradient_clip_max_norm"]  = 10.0
    config["ft_label_smooth"]         = 0.0
    config["track_gradient_alignment"] = False
    config["debug_verbose"]            = False
    config["debug_one_user_only"]      = False
    config["debug_one_episode"]        = False
    config["debug_five_episodes"]      = False
    config["NOTS"] = True
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["seed"]      = FIXED_SEED
    config["timestamp"] = TIMESTAMP
    config["cnn_layers"]    = 3
    config["cnn_kernel"]    = 5
    config["lstm_layers"]   = 3
    config["bidirectional"] = True
    config["head_type"]     = "mlp"
    config["model_type"]    = "DeepCNNLSTM"

    # ── A12: override data + architecture for 2kHz EMG-only ──────────────────
    # Must happen BEFORE suggest_shared_hps so that any architecture that reads
    # emg_in_ch / sequence_length sees the correct values.
    if ablation_id == "A12":
        config["use_imu"]          = False
        config["multimodal"]       = False
        config["use_demographics"] = False
        config["emg_in_ch"]        = EMG_2KHZ_IN_CH
        config["sequence_length"]  = EMG_2KHZ_SEQ_LEN
        config["dfs_load_path"]    = str(Path(EMG_2KHZ_PKL_PATH).parent) + "/"
        config["front_end_stride"] = 20

    # ── Suggest HPs by group ─────────────────────────────────────────────────
    _suggest_shared_hps(trial, config, fix_arch=profile["fix_arch"])

    if profile["use_maml"]:
        _suggest_maml_hps(trial, config)
    else:
        _suggest_no_maml_fixed(config)

    if profile["use_moe"]:
        _suggest_moe_hps(trial, config, ablation_id=ablation_id)
    else:
        _suggest_no_moe_fixed(config)

    if profile["use_sup_ft"]:
        _suggest_supervised_ft_hps(trial, config)
        _suggest_label_smooth_supervised(trial, config)

    return config


# =============================================================================
# Model builder
# =============================================================================

def build_model(config: dict):
    """Build model matching ablation profile. Mirrors ablation_config.py builders."""
    use_moe  = config.get("use_MOE",      False)

    if use_moe:
        from MOE.MOE_encoder import build_MOE_model
        model = build_MOE_model(config)
    else:
        from pretraining.pretrain_models import build_model as _build
        model = _build(config)

    model.to(config["device"])
    return model


# =============================================================================
# MoE collapse helper
# =============================================================================

def _check_moe_collapse(history_or_logs: dict, num_experts: int) -> float | None:
    reports = history_or_logs.get("routing_reports", [])
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


# =============================================================================
# Data split loading
# =============================================================================

USER_SPLIT_JSON = (CODE_DIR / "system" / "fixed_user_splits"
                   / "4kfcv_splits_shared_test.json")

with open(USER_SPLIT_JSON, "r") as f:
    ALL_SPLITS = json.load(f)

FOLD_IDX = 0


def apply_fold_to_config(config: dict) -> dict:
    split = ALL_SPLITS[FOLD_IDX]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    return config


# =============================================================================
# Objective functions
# =============================================================================

def _objective_episodic(trial: optuna.Trial, config: dict) -> float:
    """
    Objective for MAML ablations (M0, A3, A4, A5, A8, A12).
    Meta-train → adapt-and-eval on val users → return mean val acc.
    """
    from MAML.maml_data_pipeline import get_maml_dataloaders
    from MAML.mamlpp import mamlpp_pretrain, mamlpp_adapt_and_eval

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

    # A12 uses its own pkl path (set in build_config_from_trial)
    if config.get("ablation_id") == "A12":
        tensor_dict_path = EMG_2KHZ_PKL_PATH

    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    model = build_model(config)
    trained_model, history = mamlpp_pretrain(model, config, train_dl,
                                             episodic_val_loader=val_dl)
    best_val_acc = history["best_val_acc"]
    best_state   = history["best_state"]

    # MoE collapse check
    if config.get("use_MOE", False):
        max_load = _check_moe_collapse(history, num_experts=config["num_experts"])
        trial.set_user_attr("final_max_expert_load",
                            max_load if max_load is not None else -1.0)
        if max_load is not None and max_load > COLLAPSE_MAX_LOAD_THRESHOLD:
            print(f"  [Trial {trial.number}] MoE COLLAPSED (max_load={max_load:.2f}). Penalising.")
            return COLLAPSE_PENALTY

    # Save checkpoint
    model_filename = f"trial_{trial.number}_fold{FOLD_IDX}_best.pt"
    torch.save({
        "trial_num":        trial.number,
        "fold_idx":         FOLD_IDX,
        "model_state_dict": best_state,
        "config":           config,
        "best_val_acc":     best_val_acc,
        "train_loss_log":   history.get("train_loss_log", []),
        "val_acc_log":      history.get("val_acc_log", []),
    }, RUN_DIR / model_filename)
    print(f"  [Trial {trial.number}] Checkpoint saved: {model_filename}")

    # Per-user adapt-and-eval (the HPO signal)
    trained_model.load_state_dict(best_state)
    user_metrics: dict = defaultdict(list)
    for batch in val_dl:
        uid     = batch["user_id"]
        metrics = mamlpp_adapt_and_eval(
            trained_model, config, batch["support"], batch["query"])
        user_metrics[uid].append(metrics["acc"])

    per_user_means = [float(np.mean(accs)) for accs in user_metrics.values()]
    mean_acc = float(np.nanmean(per_user_means))

    trial.set_user_attr("mean_pretrain_val_acc", float(best_val_acc))
    trial.set_user_attr("per_user_val_accs",     per_user_means)
    print(f"  [Trial {trial.number}] Val acc: {mean_acc*100:.2f}% "
          f"(pretrain best: {best_val_acc*100:.2f}%)")
    return mean_acc


def _objective_supervised(trial: optuna.Trial, config: dict) -> float:
    """
    Objective for non-MAML ablations (A1, A2, A7).

    Protocol:
      1. Load tensor_dict once; pass to both the flat dataloader and the
         episodic val sampler — avoids the double-disk-read that the original
         version had.
      2. Flat supervised pretraining on train_PIDs.
      3. For each val episode: fine-tune (head-only) on support set, eval on
         query. finetune_and_eval_user() deepcopies the model internally, so
         each episode starts from clean pretrained weights.
      4. Return mean accuracy across val users/episodes.

    We use head_only fine-tuning as the HPO signal. Full fine-tuning is more
    expensive and rarely reranks trials differently at HPO time. The final
    ablation scripts report both modes.
    """
    from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
    from pretraining.pretrain_trainer import pretrain
    from pretraining.pretrain_finetune import finetune_and_eval_user
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict)
    from torch.utils.data import DataLoader
    from ablation_config import replace_head_for_eval

    # ── Load tensor_dict once — reuse for both the flat DL and episodic eval ─
    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    # ── Flat supervised training ──────────────────────────────────────────────
    train_dl, val_dl_flat, n_classes = get_pretrain_dataloaders(config, tensor_dict=tensor_dict)
    assert n_classes == config["pretrain_num_classes"], (
        f"Flat dataloader returned {n_classes} classes, expected "
        f"{config['pretrain_num_classes']}."
    )

    model = build_model(config)
    trained_model, history = pretrain(model, train_dl, val_dl_flat, config)
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else float("nan")

    # Save checkpoint
    model_filename = f"trial_{trial.number}_fold{FOLD_IDX}_best.pt"
    torch.save({
        "trial_num":        trial.number,
        "fold_idx":         FOLD_IDX,
        "model_state_dict": trained_model.state_dict(),
        "config":           config,
        "best_val_acc":     best_val_acc,
        "train_loss_log":   history.get("train_loss", []),
        "val_acc_log":      history.get("val_acc", []),
    }, RUN_DIR / model_filename)

    # ── Episodic val eval with supervised fine-tuning ─────────────────────────
    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = config["val_PIDs"],
        target_gesture_classes = config["maml_gesture_classes"],
        target_trial_indices   = config["target_trial_indices"],
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],
        q_query                = config.get("q_query", None),
        num_eval_episodes      = config["num_eval_episodes"],
        is_train               = False,
        seed                   = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    val_dl_epi = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=4, collate_fn=maml_mm_collate)

    # finetune_and_eval_user deepcopies the model internally for each call, so
    # every episode gets a fresh copy of the pretrained weights. We do NOT need
    # to deepcopy here — passing trained_model directly is correct and safe.
    user_metrics: dict = defaultdict(list)
    for batch in val_dl_epi:
        uid     = batch["user_id"]
        support = batch["support"]
        query   = batch["query"]
        metrics = finetune_and_eval_user(
            trained_model, config,
            support_emg=support["emg"],  support_imu=support.get("imu"),
            support_labels=support["labels"],
            query_emg=query["emg"],      query_imu=query.get("imu"),
            query_labels=query["labels"],
            mode="head_only",
        )
        user_metrics[uid].append(metrics["acc"])

    per_user_means = [float(np.mean(accs)) for accs in user_metrics.values()]
    mean_acc = float(np.nanmean(per_user_means))

    trial.set_user_attr("mean_pretrain_val_acc", float(best_val_acc))
    trial.set_user_attr("per_user_val_accs",     per_user_means)
    print(f"  [Trial {trial.number}] Val acc (head_only): {mean_acc*100:.2f}% "
          f"(pretrain best: {best_val_acc*100:.2f}%)")
    return mean_acc


def _objective_a11(trial: optuna.Trial) -> float:
    """
    Objective for A11: Meta pretrained model, 1-shot fine-tuning.

    The backbone is FIXED (pretrained Meta weights). The only HPs that matter
    are ft_lr and ft_steps. We do NOT train from scratch — we load the pretrained
    weights, run the episodic val loop with finetune_and_eval_user(), and return
    mean head-only val accuracy.

    Search space (per spec):
      ft_lr   : log-uniform [1e-5, 1e-2]
      ft_steps: categorical {10, 25, 50, 100}
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict)
    from pretraining.pretrain_finetune import finetune_and_eval_user
    from torch.utils.data import DataLoader

    # ── Suggest the only two HPs that matter for A11 ─────────────────────────
    ft_lr    = trial.suggest_float("ft_lr",    1e-5, 1e-2, log=True)
    ft_steps = trial.suggest_categorical("ft_steps", [10, 25, 50, 100])

    # Minimal config — only what finetune_and_eval_user needs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "n_way":                  3,
        "k_shot":                 1,
        "q_query":                9,
        "num_eval_episodes":      100,
        "maml_gesture_classes":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target_trial_indices":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "train_reps":             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "val_reps":               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "use_imu":                False,   # Meta model is EMG-only
        "device":                 device,
        "ft_lr":                  ft_lr,
        "ft_steps":               ft_steps,
        "ft_optimizer":           "adam",
        "ft_weight_decay":        0.0,
        "ft_label_smooth":        0.0,
        "emg_in_ch":              EMG_2KHZ_IN_CH,
        "sequence_length":        EMG_2KHZ_SEQ_LEN,
    }
    apply_fold_to_config(config)

    # ── Load pretrained Meta model ────────────────────────────────────────────
    # Import here to avoid hard dependency when the Meta repo isn't available
    # for ablations that don't need it.
    sys.path.insert(0, str(Path(os.environ.get(
        "NEUROMOTOR_REPO",
        "/projects/my13/div-emg/generic-neuromotor-interface"
    )).resolve()))
    from A10_A11_A12_meta_pretrained import MetaEMGWrapper
    model = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
    model.to(device)

    # ── Load tensor_dict ──────────────────────────────────────────────────────
    with open(EMG_2KHZ_PKL_PATH, "rb") as f:
        full_dict = pickle.load(f)
    from MAML.maml_data_pipeline import reorient_tensor_dict
    tensor_dict = reorient_tensor_dict(full_dict, config)

    # ── Episodic val eval ─────────────────────────────────────────────────────
    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = config["val_PIDs"],
        target_gesture_classes = config["maml_gesture_classes"],
        target_trial_indices   = config["target_trial_indices"],
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],
        q_query                = config.get("q_query", None),
        num_eval_episodes      = config["num_eval_episodes"],
        is_train               = False,
        seed                   = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    val_dl_epi = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=4, collate_fn=maml_mm_collate)

    # finetune_and_eval_user deepcopies model internally per call — safe to
    # pass the same model object across all episodes.
    user_metrics: dict = defaultdict(list)
    for batch in val_dl_epi:
        uid     = batch["user_id"]
        support = batch["support"]
        query   = batch["query"]
        metrics = finetune_and_eval_user(
            model, config,
            support_emg=support["emg"],  support_imu=None,
            support_labels=support["labels"],
            query_emg=query["emg"],      query_imu=None,
            query_labels=query["labels"],
            mode="head_only",
        )
        user_metrics[uid].append(metrics["acc"])

    per_user_means = [float(np.mean(accs)) for accs in user_metrics.values()]
    mean_acc = float(np.nanmean(per_user_means))

    trial.set_user_attr("ft_lr",    ft_lr)
    trial.set_user_attr("ft_steps", ft_steps)
    trial.set_user_attr("per_user_val_accs", per_user_means)
    print(f"  [A11 | Trial {trial.number}] ft_lr={ft_lr:.2e}, ft_steps={ft_steps} "
          f"→ val acc={mean_acc*100:.2f}%")
    return mean_acc


# =============================================================================
# Optuna objective wrapper
# =============================================================================

def objective(trial: optuna.Trial, ablation_id: str, profile: dict) -> float:
    trial_start = time.time()
    print("=" * 80)
    print(f"[{ablation_id} | Trial {trial.number}] Starting")
    print("=" * 80)

    if profile["is_a11"]:
        # A11 has its own fully self-contained objective — bypass the config
        # builder entirely since it would produce meaningless training HPs.
        result = _objective_a11(trial)
    else:
        config = build_config_from_trial(trial, ablation_id, profile)
        apply_fold_to_config(config)

        print(f"\n[{ablation_id} | Trial {trial.number}] Config snapshot:")
        for k, v in sorted(config.items()):
            if k not in ("train_PIDs", "val_PIDs", "test_PIDs", "device"):
                print(f"  {k}: {v}")
        print()

        if profile["eval_mode"] == "episodic":
            result = _objective_episodic(trial, config)
        elif profile["eval_mode"] == "supervised":
            result = _objective_supervised(trial, config)
        else:
            raise ValueError(f"Unknown eval_mode: {profile['eval_mode']!r}")

    elapsed = time.time() - trial_start
    print(f"[{ablation_id} | Trial {trial.number}] Done in {elapsed:.1f}s → val_acc={result*100:.2f}%")
    return result


# =============================================================================
# Warm-start filtering
# =============================================================================

def _get_suggest_keys_for_profile(ablation_id: str, profile: dict) -> set[str]:
    """
    Return the set of HP keys that will be suggest_*'d for this ablation profile.
    Used to filter warm-start params so Optuna doesn't raise UnexpectedParameter.

    Note: cnn_base_filters, lstm_hidden, and groupnorm_num_groups are NOT in
    this set — they are fixed constants, never passed to trial.suggest_*.
    Warm-start dicts that contain these keys will have them silently dropped
    by _filter_warm_start_params, which is correct behaviour.
    """
    if profile["is_a11"]:
        return {"ft_lr", "ft_steps"}

    keys: set[str] = {"outer_lr", "wd"}

    if profile["use_maml"]:
        keys |= {
            "maml_inner_steps", "maml_alpha_init", "maml_alpha_init_eval",
            "maml_use_lslr", "use_lslr_at_eval",
            "use_maml_msl", "maml_msl_num_epochs",
            "episodes_per_epoch_train",
        }
    if profile["use_moe"]:
        keys |= {
            "num_experts", "MOE_top_k", "MOE_gate_temperature",
            "MOE_aux_coeff", "MOE_ctx_out_dim", "MOE_ctx_hidden_dim",
            "MOE_dropout", "MOE_aux_loss_plcmt",
        }
    if profile["use_sup_ft"]:
        keys |= {"ft_lr", "label_smooth"}

    return keys


def _filter_warm_start_params(raw_params: dict, suggest_keys: set[str]) -> dict:
    return {k: v for k, v in raw_params.items() if k in suggest_keys}


# =============================================================================
# Study runner
# =============================================================================

def run_study(ablation_id: str, profile: dict, n_trials: int = 1) -> optuna.Study:
    sleep_time = random.uniform(0, 10)
    print(f"Staggering start: sleeping {sleep_time:.2f}s …")
    time.sleep(sleep_time)

    use_journal = int(os.environ.get("HPO_USE_JOURNAL", "1"))
    study_name  = f"ablation_{ablation_id}_1s3w_hpo_v1"
    journal_path = HPO_DB_DIR / f"{study_name}.log"

    if use_journal:
        lock_obj = JournalFileBackend(str(journal_path))
        storage  = JournalStorage(lock_obj)
        print(f"Journal storage: {journal_path}")
    else:
        storage = optuna.storages.InMemoryStorage()
        print("InMemoryStorage (debug mode — results not persisted).")

    time.sleep(random.uniform(0, 10))

    suggest_keys = _get_suggest_keys_for_profile(ablation_id, profile)

    warm_start = M0_WARM_START_PARAMS if ablation_id in ABLATIONS_WITH_M0_WARMSTART else []
    n_startup  = max(20, len(warm_start))

    sampler = optuna.samplers.TPESampler(
        seed=FIXED_SEED,
        n_startup_trials=n_startup,
        n_ei_candidates=24,
        multivariate=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    if warm_start and len(study.trials) == 0:
        print(f"Warm-starting study with {len(warm_start)} trials …")
        for i, raw in enumerate(warm_start):
            filtered = _filter_warm_start_params(raw, suggest_keys)
            study.enqueue_trial(filtered)
            print(f"  Enqueued warm-start trial {i}: {filtered}")
    elif warm_start and len(study.trials) > 0:
        print(f"Study already has {len(study.trials)} trials — skipping warm-start.")

    study.optimize(
        lambda trial: objective(trial, ablation_id, profile),
        n_trials=n_trials,
        gc_after_trial=True,
    )
    return study


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified ablation HPO driver (one job = one Optuna trial).")
    parser.add_argument(
        "--ablation", type=str, required=True,
        choices=sorted(ABLATION_PROFILES.keys()),
        help="Ablation ID to HPO (e.g. M0, A1, A3).",
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override DATA_DIR env var.")
    parser.add_argument("--out_dir",  type=str, default=None,
                        help="Override RUN_DIR env var (per-trial checkpoint dir).")
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = Path(args.data_dir).resolve()
    if args.out_dir:
        RUN_DIR = Path(args.out_dir).resolve()
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    ablation_id = args.ablation
    profile     = ABLATION_PROFILES[ablation_id]

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    print(f"\n{'='*70}")
    print(f"  Ablation HPO: {ablation_id}")
    print(f"  Profile     : {profile}")
    print(f"  N_TRIALS    : {N_TRIALS}")
    print(f"  HPO_DB_DIR  : {HPO_DB_DIR}")
    print(f"  RUN_DIR     : {RUN_DIR}")
    print(f"{'='*70}\n")

    run_study(ablation_id=ablation_id, profile=profile, n_trials=N_TRIALS)