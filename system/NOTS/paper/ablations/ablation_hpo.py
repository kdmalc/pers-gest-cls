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
# Set only by --replay-trial (top-3 re-evaluation): training seed, and dev episodes per user.
SEED_OVERRIDE = None
VAL_EPISODES_OVERRIDE = None

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

HPO_DB_DIR = Path(os.environ["HPO_DB_DIR"])   # set by the launcher (sub2/hpo/optuna_dbs); no default
HPO_DB_DIR.mkdir(parents=True, exist_ok=True)

# MoE collapse detection (same as v3 HPO script)
# (collapse threshold: relative to the uniform selection rate, set in mamlpp_pretrain; see _objective_episodic)
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
# Architecture-exploration variants (docs/SPEC_architecture_exploration.md)
# =============================================================================
# Each variant is its OWN Optuna study with the same budget (never a categorical arm inside one
# study). "base" = the profile whose HP groups it reuses (M0: MAML+MoE, A4: MAML no MoE at the
# param-matched widths, A2: supervised); "family" = model builder (legacy = pretrain_models /
# MOE_encoder, flexnet = arch_variants.flexnet, metric = arch_variants.metric_learners);
# "set" = fixed config overrides; "spaces" = extra HP groups (see _apply_variant).
ARCH_VARIANTS: dict[str, dict] = {
    # ── MoE structure (same expert bank as M0; one change each) ──────────────
    "X2":  dict(base="M0", family="legacy", set={"MOE_gate_mode": "uniform"}, spaces=("moe_uniform",),
                note="M0 with a fixed uniform gate (1/E, no router): routing or just ensembling?"),
    "X3":  dict(base="M0", family="legacy", set={"MOE_skip_unselected_experts": True}, spaces=("moe_sparse",),
                note="Sparse top-1/2 over a small bank + load balance + router z-loss (ST-MoE recipe)"),
    "X4":  dict(base="M0", family="legacy", set={"MOE_routing_signal": "context_proj_task"}, spaces=(),
                note="Task-conditioned routing: [sample context, support-set mean context]"),
    "X5":  dict(base="M0", family="legacy", set={"MOE_use_shared_expert": True}, spaces=(),
                note="Shared always-on expert + routed experts (DeepSeek-MoE style)"),
    "X6":  dict(base="M0", family="legacy", set={"MOE_routing_signal": "xattn"}, spaces=("xattn_router",),
                note="Router input = EMG tokens cross-attending to IMU tokens"),
    # ── EMG-IMU fusion, screened on the no-MoE MAML backbone (reference: A4 = early concat) ─
    "F1":  dict(base="A4", family="flexnet", set={"fusion_mode": "mid"}, spaces=("fusion",),
                note="Mid fusion: per-modality CNNs, features concatenated before the LSTM"),
    "F2":  dict(base="A4", family="flexnet", set={"fusion_mode": "film"}, spaces=("fusion",),
                note="FiLM: IMU CNN produces per-channel, per-step gamma/beta for the EMG CNN"),
    "F3":  dict(base="A4", family="flexnet", set={"fusion_mode": "xattn"}, spaces=("fusion", "fusion_xattn"),
                note="Cross-attention fusion: EMG tokens <-> IMU tokens"),
    # ── Transformers ─────────────────────────────────────────────────────────
    "T0":  dict(base="A4", family="flexnet", set={"fusion_mode": "none", "temporal_model": "transformer"},
                spaces=("tf",), note="Single transformer on the 64 time steps (no CNN, no MoE), MAML++"),
    "T2":  dict(base="M0", family="legacy", set={"temporal_model": "transformer"}, spaces=("tf",),
                note="M0's CNN expert bank, LSTM replaced by a transformer"),
    "T4":  dict(base="M0", family="flexnet",
                set={"fusion_mode": "none", "temporal_model": "transformer", "tf_moe_every": 2,
                     "MOE_skip_unselected_experts": True},
                spaces=("tf", "moe_sparse"), note="Transformer with sparse MoE feed-forward layers (Switch-style)"),
    # ── Alternatives to MAML (episodic metric learners, no inner loop) ──────────
    "P1":       dict(base="A4", family="metric", set={"metric_kind": "proto", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="ProtoNet trained episodically (A4 encoder)"),
    "P1_moe":   dict(base="A1", family="metric", set={"metric_kind": "proto", "metric_backbone": "moe"},
                     spaces=("metric",), note="ProtoNet trained episodically on M0's expert-bank encoder"),
    "P1_match": dict(base="A4", family="metric", set={"metric_kind": "matching", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="Matching Network (no FCE)"),
    "P1_rel":   dict(base="A4", family="metric", set={"metric_kind": "relation", "metric_backbone": "cnnlstm"},
                     spaces=("metric", "metric_relation"), note="Relation Network (CE on relation scores)"),
    "P1_ridge": dict(base="A4", family="metric", set={"metric_kind": "ridge", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="R2-D2 closed-form ridge head"),
    "P1_feat":  dict(base="A4", family="metric", set={"metric_kind": "feat", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="FEAT: set-to-set adapted prototypes"),
    # ── Other learning approaches ──────────────────────────────────────────
    "P3":       dict(base="A2", family="legacy", set={"adann_disc_hidden": 128}, spaces=("adann",),
                     note="ADANN-style: supervised + gradient-reversal user head, then fine-tune"),
    "A4_ANIL":  dict(base="A4", family="legacy", set={"maml_inner_param_include": ["head"]}, spaces=(),
                     note="ANIL: inner loop adapts the head only (A4 backbone)"),
    "M0_ANIL":  dict(base="M0", family="legacy", set={"maml_inner_param_include": ["head"]}, spaces=(),
                     note="ANIL on M0 (MoEMeta Port B is close to this)"),
}

for _vid, _v in ARCH_VARIANTS.items():
    _p = dict(ABLATION_PROFILES[_v["base"]])
    _p.update(variant=_v, note=_v["note"], script=None)
    if _v["family"] == "metric":
        _p.update(use_maml=False, use_sup_ft=False, eval_mode="metric")
    if _vid == "T4":
        _p["fix_arch"] = False
    ABLATION_PROFILES[_vid] = _p

# =============================================================================
# Warm-start params  (M0 only — top-10 from the existing v3 50-step HPO run)
# =============================================================================

# =============================================================================
# Warm-start params — DISABLED (v4)
# =============================================================================
# These were derived from HPO runs that used maml_inner_steps_eval=50 or 100
# (mismatched from train steps). With eval steps now matched to train steps,
# the optimal maml_alpha_init_eval changes substantially, making these priors
# misleading rather than helpful. Starting cold gives TPE a clean slate.
# Uncomment and re-populate once a clean matched-step HPO run is complete.
#
# M0_WARM_START_PARAMS: list[dict] = [
#     {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.0016865261840566302, "maml_alpha_init_eval": 0.03241222861959444, "outer_lr": 0.00011753148144028081, "wd": 0.0006958201039866241, "groupnorm_num_groups": 8,  "num_experts": 25, "MOE_top_k": 6,  "MOE_gate_temperature": 0.5007953923754159, "MOE_aux_coeff": 0.17418352079333946, "MOE_ctx_out_dim": 32,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.10372039932801176, "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 200, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 36, "maml_use_lslr": True,  "use_lslr_at_eval": True},
#     {"cnn_base_filters": 64,  "lstm_hidden": 128, "maml_inner_steps": 7,  "maml_alpha_init": 0.003183796421686842,  "maml_alpha_init_eval": 0.02561505243517187,  "outer_lr": 0.00017730562335905792, "wd": 0.0009623784816096662, "groupnorm_num_groups": 8,  "num_experts": 23, "MOE_top_k": 4,  "MOE_gate_temperature": 0.5894698227360172, "MOE_aux_coeff": 0.13261101041518134, "MOE_ctx_out_dim": 64,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.1742031578572473,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 250, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 23, "maml_use_lslr": False, "use_lslr_at_eval": True},
#     {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 9,  "maml_alpha_init": 0.002708370774699923,  "maml_alpha_init_eval": 0.0021108128739609285, "outer_lr": 0.0001951781386901919,  "wd": 6.85027628401306e-05,  "groupnorm_num_groups": 8,  "num_experts": 27, "MOE_top_k": 3,  "MOE_gate_temperature": 0.9458878022205542, "MOE_aux_coeff": 0.16641221051826932, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.08151144309939662,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 17, "maml_use_lslr": True,  "use_lslr_at_eval": True},
#     {"cnn_base_filters": 64,  "lstm_hidden": 256, "maml_inner_steps": 7,  "maml_alpha_init": 0.0013373621508281483, "maml_alpha_init_eval": 0.06693266709636093,  "outer_lr": 0.00023241537465524889, "wd": 0.0009827548412804656, "groupnorm_num_groups": 8,  "num_experts": 24, "MOE_top_k": 7,  "MOE_gate_temperature": 1.0459149632012676, "MOE_aux_coeff": 0.11694630533248768, "MOE_ctx_out_dim": 128, "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.024707209736742966, "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 500, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 26, "maml_use_lslr": False, "use_lslr_at_eval": True},
#     {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.001282949149676896,  "maml_alpha_init_eval": 0.006278649366666367,  "outer_lr": 0.00014911507147964058, "wd": 0.0003105413582509981, "groupnorm_num_groups": 8,  "num_experts": 26, "MOE_top_k": 5,  "MOE_gate_temperature": 0.827815891577418,  "MOE_aux_coeff": 0.09490439617134293, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.09440229157373216,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 30, "maml_use_lslr": True,  "use_lslr_at_eval": True},
#     {"cnn_base_filters": 96,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.0036027098514744647, "maml_alpha_init_eval": 0.028273543982439742,  "outer_lr": 0.00019567289773981725, "wd": 0.0006080214276443864, "groupnorm_num_groups": 4,  "num_experts": 22, "MOE_top_k": 6,  "MOE_gate_temperature": 0.5166154000372265, "MOE_aux_coeff": 0.11646849831998997, "MOE_ctx_out_dim": 32,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.07000153139552596,  "MOE_aux_loss_plcmt": "both",  "episodes_per_epoch_train": 200, "use_maml_msl": False,      "maml_use_lslr": True,  "use_lslr_at_eval": False},
#     {"cnn_base_filters": 64,  "lstm_hidden": 128, "maml_inner_steps": 7,  "maml_alpha_init": 0.023934012189321143,  "maml_alpha_init_eval": 0.0010860796928206904, "outer_lr": 0.00010689736539096814, "wd": 2.586297450657217e-05,  "groupnorm_num_groups": 4,  "num_experts": 28, "MOE_top_k": 3,  "MOE_gate_temperature": 2.431471657932191,  "MOE_aux_coeff": 0.10087273019596443, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 32,  "MOE_dropout": 0.10564140269584042,  "MOE_aux_loss_plcmt": "both",  "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 1,  "maml_use_lslr": True,  "use_lslr_at_eval": True},
#     {"cnn_base_filters": 96,  "lstm_hidden": 256, "maml_inner_steps": 7,  "maml_alpha_init": 0.001427354132007504,  "maml_alpha_init_eval": 0.05871583020447932,  "outer_lr": 0.00021890580109728611, "wd": 0.00015067531388534113, "groupnorm_num_groups": 4,  "num_experts": 24, "MOE_top_k": 8,  "MOE_gate_temperature": 1.037361717564086,  "MOE_aux_coeff": 0.06945893667706689, "MOE_ctx_out_dim": 128, "MOE_ctx_hidden_dim": 128, "MOE_dropout": 0.00014882237603539017, "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 500, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 19, "maml_use_lslr": False, "use_lslr_at_eval": True},
#     {"cnn_base_filters": 64,  "lstm_hidden": 256, "maml_inner_steps": 7,  "maml_alpha_init": 0.0022409316363444297, "maml_alpha_init_eval": 0.07980284866764323,  "outer_lr": 0.00027288242632654975, "wd": 0.0006233783942107558, "groupnorm_num_groups": 8,  "num_experts": 27, "MOE_top_k": 6,  "MOE_gate_temperature": 1.9588295665204578, "MOE_aux_coeff": 0.2274257229744778,  "MOE_ctx_out_dim": 128, "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.06693040779691198,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 500, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 27, "maml_use_lslr": False, "use_lslr_at_eval": True},
#     {"cnn_base_filters": 64,  "lstm_hidden": 64,  "maml_inner_steps": 7,  "maml_alpha_init": 0.0010018043002410384, "maml_alpha_init_eval": 0.006165059688105781,  "outer_lr": 0.00015557309153358813, "wd": 0.0003484429564449149, "groupnorm_num_groups": 8,  "num_experts": 29, "MOE_top_k": 5,  "MOE_gate_temperature": 0.7058908557205739, "MOE_aux_coeff": 0.0971830179235051,  "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 64,  "MOE_dropout": 0.09089303028735389,  "MOE_aux_loss_plcmt": "inner", "episodes_per_epoch_train": 100, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 33, "maml_use_lslr": True,  "use_lslr_at_eval": True},
# ]
# Top-10 trials from the first clean matched-step HPO run (v4, maml_inner_steps_eval
# = maml_inner_steps). These are valid priors — not contaminated by the old
# 50/100-step eval mismatch. All 10 have maml_inner_steps=10, which is a strong
# signal that TPE should explore 10 first; the other HPs vary meaningfully.
# Note: maml_msl_num_epochs is absent from trials where use_maml_msl=False —
# _filter_warm_start_params handles this correctly (missing keys are ignored).
M0_WARM_START_PARAMS: list[dict] = [
    {"outer_lr": 0.00010093304999603776, "wd": 0.0008325426470137959, "maml_inner_steps": 10, "maml_alpha_init": 0.0009888781900907544, "maml_alpha_init_eval": 0.006717556958813548, "maml_use_lslr": True,  "use_lslr_at_eval": False, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 39, "episodes_per_epoch_train": 250, "num_experts": 24, "MOE_top_k": 7, "MOE_gate_temperature": 1.1879664247660187, "MOE_aux_coeff": 0.08672942143224953, "MOE_ctx_out_dim": 64,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.022501513050004283,  "MOE_aux_loss_plcmt": "outer"},
    {"outer_lr": 0.00017135506539036396, "wd": 0.0007191575466293857, "maml_inner_steps": 10, "maml_alpha_init": 0.002234186564248213,  "maml_alpha_init_eval": 0.00874438550866414,  "maml_use_lslr": True,  "use_lslr_at_eval": True,  "use_maml_msl": "hybrid", "maml_msl_num_epochs": 36, "episodes_per_epoch_train": 250, "num_experts": 24, "MOE_top_k": 9, "MOE_gate_temperature": 0.6909206767846876, "MOE_aux_coeff": 0.16382789345409485, "MOE_ctx_out_dim": 64,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.0009733575106648108, "MOE_aux_loss_plcmt": "both"},
    {"outer_lr": 0.00021994946057480355, "wd": 0.0005215426795904579, "maml_inner_steps": 10, "maml_alpha_init": 0.0005880651085620183,  "maml_alpha_init_eval": 0.00829671572377371,  "maml_use_lslr": True,  "use_lslr_at_eval": False, "use_maml_msl": False,                                  "episodes_per_epoch_train": 250, "num_experts": 36, "MOE_top_k": 7, "MOE_gate_temperature": 0.8873092963668456, "MOE_aux_coeff": 0.10530573719633897, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.01349711067656777,   "MOE_aux_loss_plcmt": "both"},
    {"outer_lr": 0.0003129502130691339,  "wd": 0.000566622424313277,  "maml_inner_steps": 10, "maml_alpha_init": 0.00062284162048662,    "maml_alpha_init_eval": 0.007208099016837116, "maml_use_lslr": True,  "use_lslr_at_eval": False, "use_maml_msl": False,                                  "episodes_per_epoch_train": 250, "num_experts": 36, "MOE_top_k": 4, "MOE_gate_temperature": 0.8829499397705474, "MOE_aux_coeff": 0.0633848330361091,  "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.020923944663650867,  "MOE_aux_loss_plcmt": "both"},
    {"outer_lr": 0.00015159006650537128, "wd": 0.0006326809984503553, "maml_inner_steps": 10, "maml_alpha_init": 0.0005221987619876876,  "maml_alpha_init_eval": 0.02290821144458453,  "maml_use_lslr": True,  "use_lslr_at_eval": True,  "use_maml_msl": "hybrid", "maml_msl_num_epochs": 29, "episodes_per_epoch_train": 250, "num_experts": 24, "MOE_top_k": 7, "MOE_gate_temperature": 0.6699581483015503, "MOE_aux_coeff": 0.20333779607170907, "MOE_ctx_out_dim": 64,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.0006350844285118065, "MOE_aux_loss_plcmt": "outer"},
    {"outer_lr": 9.081232239899435e-05,  "wd": 0.000598810693758293,  "maml_inner_steps": 10, "maml_alpha_init": 0.007684897507390702,   "maml_alpha_init_eval": 0.011275020201139066, "maml_use_lslr": False, "use_lslr_at_eval": False, "use_maml_msl": False,                                  "episodes_per_epoch_train": 250, "num_experts": 36, "MOE_top_k": 4, "MOE_gate_temperature": 0.7428752741091442, "MOE_aux_coeff": 0.07275053149515735, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.07423475742353443,  "MOE_aux_loss_plcmt": "both"},
    {"outer_lr": 5.2401901969035514e-05, "wd": 0.0006349931134657805, "maml_inner_steps": 10, "maml_alpha_init": 0.012181889668758774,   "maml_alpha_init_eval": 0.009865480491952098, "maml_use_lslr": False, "use_lslr_at_eval": False, "use_maml_msl": False,                                  "episodes_per_epoch_train": 250, "num_experts": 36, "MOE_top_k": 4, "MOE_gate_temperature": 0.7757263694819606, "MOE_aux_coeff": 0.1010586963942683,  "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.08307471046993736,  "MOE_aux_loss_plcmt": "both"},
    {"outer_lr": 0.00019259251729666585, "wd": 0.000658782978177928,  "maml_inner_steps": 10, "maml_alpha_init": 0.0012414887370832365,  "maml_alpha_init_eval": 0.004905374028039118, "maml_use_lslr": True,  "use_lslr_at_eval": False, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 33, "episodes_per_epoch_train": 250, "num_experts": 24, "MOE_top_k": 7, "MOE_gate_temperature": 0.7082721548137876, "MOE_aux_coeff": 0.13263549328844307, "MOE_ctx_out_dim": 64,  "MOE_ctx_hidden_dim": 64, "MOE_dropout": 0.001715561171357145,  "MOE_aux_loss_plcmt": "outer"},
    {"outer_lr": 0.00029009258072038404, "wd": 0.0006248440745542929, "maml_inner_steps": 10, "maml_alpha_init": 0.0007385074738619472,  "maml_alpha_init_eval": 0.009986812917974252, "maml_use_lslr": True,  "use_lslr_at_eval": False, "use_maml_msl": False,                                  "episodes_per_epoch_train": 250, "num_experts": 32, "MOE_top_k": 4, "MOE_gate_temperature": 0.9607888023002135, "MOE_aux_coeff": 0.08675368517371769, "MOE_ctx_out_dim": 16,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.016488864331757783,  "MOE_aux_loss_plcmt": "both"},
    {"outer_lr": 0.00016110242483204413, "wd": 0.0007362654428714762, "maml_inner_steps": 10, "maml_alpha_init": 0.0006150553191724497,  "maml_alpha_init_eval": 0.005317262590552643, "maml_use_lslr": True,  "use_lslr_at_eval": False, "use_maml_msl": "hybrid", "maml_msl_num_epochs": 8,  "episodes_per_epoch_train": 250, "num_experts": 24, "MOE_top_k": 7, "MOE_gate_temperature": 0.948431969634207,  "MOE_aux_coeff": 0.15415159207685789, "MOE_ctx_out_dim": 64,  "MOE_ctx_hidden_dim": 32, "MOE_dropout": 0.003427874209772072,  "MOE_aux_loss_plcmt": "outer"},
]

# Ablations that receive the M0 warm-start (identical HP space to M0's v3 study)
ABLATIONS_WITH_M0_WARMSTART = {} #{"M0", "A5", "A8", "A12"}

# =============================================================================
# HP suggestion helpers
# =============================================================================

def _load_fixed_params() -> dict:
    """HPO_FIXED_PARAMS_JSON (launcher --fixed-params): {param name: value} frozen in every trial."""
    p = os.environ.get("HPO_FIXED_PARAMS_JSON", "")
    if not p:
        return {}
    fixed = json.loads(Path(p).read_text())
    if not isinstance(fixed, dict) or not fixed:
        raise SystemExit(f"HPO_FIXED_PARAMS_JSON={p} must hold a non-empty {{name: value}} object")
    return fixed


class _FixedParamTrial:
    """
    Wraps an Optuna trial so suggest_*(name, ...) returns the frozen value for names in `fixed`
    and samples everything else. Used by the architecture screen: the shared training HPs (outer
    LR, MAML step sizes, ...) are frozen at one tuned config, so each variant's study only searches
    its own few architecture HPs and a small budget suffices. A frozen value outside the search
    range raises (no silent clipping). Every name in `fixed` must actually be suggested by the
    profile; objective() checks that after the config is built.
    """

    def __init__(self, trial, fixed: dict):
        self._trial, self._fixed, self.used = trial, fixed, set()

    def __getattr__(self, name):
        return getattr(self._trial, name)

    def _get(self, name, check):
        v = self._fixed[name]
        if not check(v):
            raise ValueError(f"fixed param {name}={v!r} is outside this profile's search range")
        self.used.add(name)
        return v

    def suggest_float(self, name, low, high, **kw):
        if name in self._fixed:
            return float(self._get(name, lambda v: low <= float(v) <= high))
        return self._trial.suggest_float(name, low, high, **kw)

    def suggest_int(self, name, low, high, **kw):
        if name in self._fixed:
            return int(self._get(name, lambda v: low <= int(v) <= high))
        return self._trial.suggest_int(name, low, high, **kw)

    def suggest_categorical(self, name, choices):
        if name in self._fixed:
            return self._get(name, lambda v: v in choices)
        return self._trial.suggest_categorical(name, choices)


def _with_fixed_params(trial):
    fixed = _load_fixed_params()
    if not fixed or isinstance(trial, _FixedParamTrial):
        return trial
    return _FixedParamTrial(trial, fixed)


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
    is a different regime, so it is tuned over [0.0, 0.2].
    """
    # capped at 0.2: the 10/29 debug trial drew 0.42, which flattens 10-way targets towards uniform
    # (docs/10_29_26_debug_run_results.md, issue 4)
    config["label_smooth"] = trial.suggest_float("label_smooth", 0.0, 0.2)
    return config


def _suggest_maml_hps(trial: optuna.Trial, config: dict) -> dict:
    """
    Suggest MAML-specific HPs. Only called when use_maml=True.

    v4 changes vs v3:
      - maml_inner_steps_eval is now MATCHED to maml_inner_steps (no longer fixed
        at 50). Mismatching train/eval steps was corrupting the HPO signal: the
        objective was measuring quality-of-init + easy adaptation rather than
        quality of meta-learning. With matched steps the meta-gradient and the
        val signal are optimising the same thing.
      - maml_inner_steps range extended to include 15. The old v2 study (before
        the 50/100-step eval corruption was understood) showed 15 was competitive.
        With a clean signal we should let TPE revisit this.
      - maml_alpha_init_eval range widened to [5e-4, 0.05]. The optimal eval LR
        at matched steps (5-15) is different from the optimal at 50+ steps; the
        old range [0.001, 0.08] was tuned for the high-step regime.
      - label_smooth fixed to 0.05 (unanimous across all prior completed trials).

    Test-time eval steps: after HPO completes, run a post-hoc sweep over
    eval_inner_steps in [5, 10, 15, 20, 30, 50] on the val set to determine
    the best test-time budget for the final model. Do NOT make that decision
    during HPO — it conflates model quality with adaptation budget.
    """
    config["maml_inner_steps"]      = trial.suggest_categorical("maml_inner_steps", [5, 7, 10, 15])
    # Match eval steps to train steps — clean, unbiased HPO signal.
    config["maml_inner_steps_eval"] = config["maml_inner_steps"]
    config["maml_alpha_init"]       = trial.suggest_float("maml_alpha_init",      5e-4, 0.025, log=True)
    config["maml_alpha_init_eval"]  = trial.suggest_float("maml_alpha_init_eval", 5e-4, 0.05,  log=True)
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


def _suggest_moe_hps(trial: optuna.Trial, config: dict, ablation_id: str,
                     moe_space: str = "m0") -> dict:
    """
    Suggest MoE-specific HPs. Only called when use_moe=True.

    For A5 (expert-count sweep), num_experts is drawn from the same discrete
    set used by the sweep script so HPO explores the full mountain-curve range.
    For all other ablations, num_experts is drawn from the HPO-friendly
    coarser grid.

    moe_space (architecture variants):
      "m0"      the M0 space below
      "uniform" X2: fixed uniform gate. Only the bank size and expert dropout are searched; the
                router HPs do not exist (top_k = E, no aux loss).
      "sparse"  X3 / T4: small bank, top-1/2, Switch load balance + router z-loss, temperature 1
                (the ST-MoE recipe; the z-loss replaces temperature as the logit regulariser).
    """
    if moe_space == "uniform":
        E = trial.suggest_categorical("num_experts", [4, 8, 16, 22, 28, 36])
        config.update(num_experts=E, MOE_top_k=E, top_k=E, MOE_gate_temperature=1.0, MOE_aux_coeff=0.0,
                      MOE_ctx_out_dim=32, MOE_ctx_hidden_dim=64, apply_MOE_aux_loss_inner_outer="outer",
                      MOE_router_z_coeff=0.0)
        config["MOE_dropout"] = trial.suggest_float("MOE_dropout", 1e-4, 0.2, log=True)
        return _moe_fixed(config)
    if moe_space == "sparse":
        E = trial.suggest_categorical("num_experts", [4, 8, 16])
        k = trial.suggest_categorical("MOE_top_k", [1, 2])
        config.update(num_experts=E, MOE_top_k=k, top_k=k, MOE_gate_temperature=1.0)
        config["MOE_aux_coeff"]      = trial.suggest_float("MOE_aux_coeff", 1e-3, 0.1, log=True)
        config["MOE_router_z_coeff"] = trial.suggest_float("MOE_router_z_coeff", 1e-4, 1e-2, log=True)
        config["MOE_ctx_out_dim"]    = trial.suggest_categorical("MOE_ctx_out_dim", [16, 32, 64])
        config["MOE_ctx_hidden_dim"] = 64
        config["MOE_dropout"]        = trial.suggest_float("MOE_dropout", 1e-4, 0.2, log=True)
        config["apply_MOE_aux_loss_inner_outer"] = "outer"
        return _moe_fixed(config)
    if moe_space != "m0":
        raise ValueError(f"unknown moe_space {moe_space!r}")
    config["MOE_router_z_coeff"] = 0.0

    if ablation_id == "A5":
        # Match the sweep values exactly — HPO finds the best overall config
        # within this range; the sweep script then fixes num_experts to each
        # value and reuses the rest of the best config.
        config["num_experts"] = trial.suggest_categorical(
            "num_experts", [4, 8, 12, 16, 20, 24, 32, 40])
    else:
        # widened (SPEC_hpo.md): the old space started at 20 and could not find small expert banks
        config["num_experts"] = trial.suggest_categorical(
            "num_experts", [4, 8, 16, 22, 28, 36])

    # top_k is clipped to the bank size (Optuna categorical sets cannot depend on num_experts)
    config["MOE_top_k"]     = min(trial.suggest_categorical("MOE_top_k", [1, 2, 4, 6, 9, 12]),
                                  config["num_experts"])
    config["top_k"]         = config["MOE_top_k"]
    config["MOE_gate_temperature"] = trial.suggest_float(
        "MOE_gate_temperature", 0.3, 2.5, log=True)
    config["MOE_aux_coeff"]       = trial.suggest_float("MOE_aux_coeff", 0.02, 0.25, log=True)
    config["MOE_ctx_out_dim"]     = trial.suggest_categorical("MOE_ctx_out_dim",    [16, 32, 64, 128])
    config["MOE_ctx_hidden_dim"]  = trial.suggest_categorical("MOE_ctx_hidden_dim", [32, 64, 128])
    config["MOE_dropout"]         = trial.suggest_float("MOE_dropout", 1e-4, 0.2, log=True)
    config["apply_MOE_aux_loss_inner_outer"] = trial.suggest_categorical(
        "MOE_aux_loss_plcmt", ["inner", "both", "outer"])
    return _moe_fixed(config)


def _moe_fixed(config: dict) -> dict:
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
    # eval-time fine-tune budget is tuned on the dev users (SPEC_hpo.md); the MAML-mirrored budget
    # (10 sgd steps at maml_alpha_init_eval) is reported separately by the eval scripts.
    config["ft_steps"]        = trial.suggest_categorical("ft_steps", [10, 25, 50, 100, 200])
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]
    return config


def _arch_fixed_defaults(config: dict) -> dict:
    """
    Explicit values for every architecture-exploration key, written into every trial's config (and so
    into its printed snapshot and checkpoint). These reproduce M0 / A4 exactly; variants overwrite
    them in _apply_variant. Set here rather than read with .get() defaults in the models, so that
    no model silently falls back to a value nobody chose.
    """
    config.update(
        arch_family="legacy",
        MOE_gate_mode="learned", MOE_routing_signal="context_proj", MOE_use_shared_expert=False,
        MOE_router_z_coeff=0.0, MOE_skip_unselected_experts=False,
        MOE_xattn_d_model=64, MOE_xattn_heads=4,
        temporal_model="lstm",
        fusion_mode="early", fusion_branch_width=0.71, fusion_d_model=64, fusion_heads=4,
        tf_d_model=128, tf_heads=4, tf_blocks=3, tf_ff_mult=2, tf_moe_every=0,
        maml_inner_param_include=None,
    )
    return config


def _apply_variant(trial: optuna.Trial, config: dict, v: dict) -> dict:
    """Fixed overrides + the extra HP groups of an architecture variant (ARCH_VARIANTS)."""
    config["arch_family"] = v["family"]
    config.update(copy.deepcopy(v["set"]))
    spaces = v["spaces"]
    if "fusion" in spaces:
        # 0.71 ~ 1/sqrt(2): two branches at this width have about the conv parameters of one early CNN
        config["fusion_branch_width"] = trial.suggest_categorical("fusion_branch_width", [0.5, 0.71, 1.0])
    if "fusion_xattn" in spaces:
        config["fusion_d_model"] = trial.suggest_categorical("fusion_d_model", [32, 64, 128])
    if "xattn_router" in spaces:
        config["MOE_xattn_d_model"] = trial.suggest_categorical("MOE_xattn_d_model", [32, 64])
    if "tf" in spaces:
        # transformers get their own capacity / regularisation range (SPEC section 2C)
        config["tf_d_model"] = trial.suggest_categorical("tf_d_model", [64, 128, 192])
        config["tf_heads"]   = trial.suggest_categorical("tf_heads", [4, 8])
        config["tf_blocks"]  = trial.suggest_categorical("tf_blocks", [2, 3, 4])
        config["tf_ff_mult"] = trial.suggest_categorical("tf_ff_mult", [2, 4])
        config["dropout"]    = trial.suggest_float("tf_dropout", 0.0, 0.3)
    if "metric" in spaces:
        # episodic training without an inner loop: no MAML HPs; meta-batch as in MAML
        config.update(meta_learning=True, meta_batchsize=24, label_smooth=0.0,
                      maml_inner_steps=0, maml_inner_steps_eval=0)
        config["episodes_per_epoch_train"] = trial.suggest_categorical(
            "episodes_per_epoch_train", [100, 200, 250, 500])
    if "metric_relation" in spaces:
        config["metric_relation_hidden"] = trial.suggest_categorical("metric_relation_hidden", [64, 128, 256])
    if "adann" in spaces:
        config["adann_lambda"] = trial.suggest_float("adann_lambda", 0.01, 1.0, log=True)
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
    sys.path.append(os.path.join(os.path.dirname(__file__), 'test_eval_files'))
    from ablation_config import make_base_config, replace_head_for_eval

    trial = _with_fixed_params(trial)
    config = make_base_config(ablation_id=ablation_id)

    config["results_save_dir"] = str(RUN_DIR)
    config["models_save_dir"]  = str(RUN_DIR)

    # ── Fixed architecture / task setup ──────────────────────────────────────
    config["n_way"]                 = 10    # HPO at the primary task: 10-way 1-shot (Kai, 2026-10-21)
    config["k_shot"]                = 1
    config["q_query"]               = 9
    config["num_classes"]           = 10
    config["pretrain_num_classes"]  = 10
    config["num_eval_episodes"]     = 100   # only used by the "sampled" val path (unused here)
    # Dev episodes per user per epoch (cached file has 100/user). A 10-way eval episode is ~3x the
    # cost of a 3-way one, so the per-epoch HPO signal uses the first 25 per user (100 episodes in
    # total); the final top-3 re-evaluation uses all 100.
    config["val_episodes_per_user"] = 25
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
    config["gradient_clip_max_norm"]  = 10.0
    config["ft_label_smooth"]         = 0.0
    config["track_gradient_alignment"] = False
    config["debug_verbose"]            = False
    config["debug_one_user_only"]      = False
    config["debug_one_episode"]        = False
    config["debug_five_episodes"]      = False
    config["NOTS"] = True
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["seed"]      = FIXED_SEED if SEED_OVERRIDE is None else SEED_OVERRIDE
    if VAL_EPISODES_OVERRIDE is not None:
        config["val_episodes_per_user"] = VAL_EPISODES_OVERRIDE
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
    _arch_fixed_defaults(config)
    variant = profile.get("variant")
    _suggest_shared_hps(trial, config, fix_arch=profile["fix_arch"])

    if profile["use_maml"]:
        _suggest_maml_hps(trial, config)
    else:
        _suggest_no_maml_fixed(config)

    if profile["use_moe"]:
        moe_space = "m0"
        if variant is not None:
            moe_space = ("uniform" if "moe_uniform" in variant["spaces"]
                         else "sparse" if "moe_sparse" in variant["spaces"] else "m0")
        _suggest_moe_hps(trial, config, ablation_id=ablation_id, moe_space=moe_space)
    else:
        _suggest_no_moe_fixed(config)

    if profile["use_sup_ft"]:
        _suggest_supervised_ft_hps(trial, config)
        _suggest_label_smooth_supervised(trial, config)

    if variant is not None:
        _apply_variant(trial, config, variant)

    if isinstance(trial, _FixedParamTrial):
        unused = set(trial._fixed) - trial.used
        if unused:
            raise SystemExit(f"fixed params {sorted(unused)} are not searched by {ablation_id} (or are "
                             f"conditionally inactive); remove them from HPO_FIXED_PARAMS_JSON")
        config["hpo_fixed_params"] = dict(trial._fixed)
        if not isinstance(trial._trial, optuna.trial.FixedTrial):
            trial._trial.set_user_attr("fixed_params", dict(trial._fixed))
    return config


# =============================================================================
# Model builder
# =============================================================================

LAST_MODEL_N_PARAMS = None   # set by build_model; read into the trial's user attrs by objective()


def build_model(config: dict):
    """Build model matching ablation profile. Mirrors ablation_config.py builders."""
    global LAST_MODEL_N_PARAMS
    family = config["arch_family"]
    if family == "flexnet":
        from arch_variants.flexnet import build_flexnet
        model = build_flexnet(config)
    elif family == "metric":
        from arch_variants.metric_learners import build_metric_learner
        model = build_metric_learner(config)
    elif family != "legacy":
        raise ValueError(f"unknown arch_family {family!r}")
    elif config.get("use_MOE", False):
        from MOE.MOE_encoder import build_MOE_model
        model = build_MOE_model(config)
    else:
        from pretraining.pretrain_models import build_model as _build
        model = _build(config)

    LAST_MODEL_N_PARAMS = int(sum(p.numel() for p in model.parameters()))
    print(f"  [build_model] {family}: {LAST_MODEL_N_PARAMS:,} parameters")
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

# The split comes from ablation_config.get_hpo_split() (kfold7 fold-0 train users, 4 HPO-val
# users). The old Kapanji JSON is deliberately NOT loaded here.
FOLD_IDX = 0


def apply_fold_to_config(config: dict) -> dict:
    """HPO split (kfold7 protocol): 20 train users, validated on the 4 dev users only."""
    from ablation_config import get_hpo_split
    split = get_hpo_split()
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    config["val_episode_source"] = "dev"   # cached dev episodes (SPEC_episode_cache.md)
    return config


# =============================================================================
# Objective functions
# =============================================================================

def _objective_episodic(trial: optuna.Trial, config: dict) -> float:
    """
    Objective for MAML ablations (M0, A3, A4, A5, A8, A12).
    Meta-train → adapt-and-eval on val users → return mean val acc.

    v4 objective: best val acc AFTER a burn-in period, not the global peak.
    -------------------------------------------------------------------------
    With matched train/eval inner steps, epoch-0 val acc is no longer inflated
    by excessive adaptation steps, so raw epoch-0 peaking is less likely.
    However, as a belt-and-suspenders guard we still exclude the first
    HPO_BURNIN_EPOCHS epochs from the best-val-acc computation. This means:
      - A model that peaks at epoch 2 and then degrades is not rewarded for
        that transient peak.
      - A model that genuinely improves and plateaus at epoch 10-30 gets
        its true best value reported.
      - Early stopping still operates on the full val_acc trajectory (so it
        can still cut bad trials early), but the Optuna return value is
        max(val_acc_log[HPO_BURNIN_EPOCHS:]).
      - If a trial early-stops before HPO_BURNIN_EPOCHS epochs complete (i.e.
        it was bad enough to stop within the first 5 epochs), we report the
        best post-epoch-0 value we have. This is conservative but correct —
        a trial that can't survive 5 epochs deserves a low score.

    HPO_BURNIN_EPOCHS = 5: gives MAML ~750-1250 meta-gradient steps
    (5 epochs × 150-250 episodes) to move the init meaningfully before we
    start caring about val accuracy.
    """
    from MAML.maml_data_pipeline import get_maml_dataloaders
    from MAML.mamlpp import mamlpp_pretrain, mamlpp_adapt_and_eval

    HPO_BURNIN_EPOCHS = 5   # ignore val acc from epochs 0..HPO_BURNIN_EPOCHS-1

    from ablation_config import tensor_dict_file as _tdp
    tensor_dict_path = _tdp(config)

    # A12 uses its own pkl path (set in build_config_from_trial)
    if config.get("ablation_id") == "A12":
        tensor_dict_path = EMG_2KHZ_PKL_PATH

    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    model = build_model(config)

    def _prune_cb(epoch, val_acc):
        # report the per-user-mean dev accuracy each epoch; MedianPruner cuts weak trials early
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trained_model, history = mamlpp_pretrain(model, config, train_dl,
                                             episodic_val_loader=val_dl,
                                             epoch_callback=_prune_cb)
    val_acc_log  = history.get("val_acc_log", [])   # one entry per epoch (including epoch 0)
    global_best_val_acc = history["best_val_acc"]
    best_state          = history["best_state"]

    # Compute the burn-in-gated best val acc.
    # val_acc_log[0] is the epoch-0 baseline (before any training), so
    # post-burn-in entries start at index HPO_BURNIN_EPOCHS.
    post_burnin_accs = val_acc_log[HPO_BURNIN_EPOCHS:]
    if post_burnin_accs:
        burnin_best_val_acc = float(max(post_burnin_accs))
        burnin_best_epoch   = int(np.argmax(val_acc_log[HPO_BURNIN_EPOCHS:])) + HPO_BURNIN_EPOCHS
    else:
        # Trial early-stopped within the burn-in window. Use best of whatever
        # we have after epoch 0 — conservative score, trial will rank low.
        post_ep0_accs = val_acc_log[1:] if len(val_acc_log) > 1 else val_acc_log
        burnin_best_val_acc = float(max(post_ep0_accs)) if post_ep0_accs else 0.0
        burnin_best_epoch   = int(np.argmax(post_ep0_accs)) + 1 if post_ep0_accs else 0
        print(f"  [Trial {trial.number}] WARNING: trial early-stopped within burn-in "
              f"({len(val_acc_log)} epochs). Reporting best post-ep0 val acc.")

    n_epochs_run = len(val_acc_log)
    print(f"  [Trial {trial.number}] epochs_run={n_epochs_run}, "
          f"global_best={global_best_val_acc*100:.2f}% (ep{int(np.argmax(val_acc_log))}), "
          f"burnin_best={burnin_best_val_acc*100:.2f}% (ep{burnin_best_epoch})")

    # MoE collapse check.
    # Uses the report's own is_collapsed flag, whose threshold is relative to the uniform selection
    # rate (mamlpp_pretrain: (1 + 0.8) * top_k / E for top-k routing). The previous absolute test
    # (max selection freq > 0.80) zeroed every perfectly balanced trial with top_k / E >~ 0.7
    # (e.g. E=8, k=6: uniform rate 0.75), i.e. it silently removed part of the M0 search space.
    # A uniform gate (X2) cannot collapse and is not checked (its argmax "dominant expert" is a tie).
    if config.get("use_MOE", False) and config["MOE_gate_mode"] != "uniform":
        max_load = _check_moe_collapse(history, num_experts=config["num_experts"])
        trial.set_user_attr("final_max_expert_load",
                            max_load if max_load is not None else -1.0)
        reports = history.get("routing_reports", [])
        if reports and reports[-1].get("is_collapsed", False):
            print(f"  [Trial {trial.number}] MoE COLLAPSED (max selection freq={max_load:.2f}, "
                  f"threshold={reports[-1]['collapse_freq_threshold']:.2f}). Penalising.")
            return COLLAPSE_PENALTY

    # Save checkpoint (best state across all epochs, for post-hoc analysis)
    model_filename = f"trial_{trial.number}_fold{FOLD_IDX}_best.pt"
    torch.save({
        "trial_num":              trial.number,
        "fold_idx":               FOLD_IDX,
        "model_state_dict":       best_state,
        "config":                 config,
        "global_best_val_acc":    global_best_val_acc,
        "burnin_best_val_acc":    burnin_best_val_acc,
        "burnin_best_epoch":      burnin_best_epoch,
        "n_epochs_run":           n_epochs_run,
        "val_acc_log":            val_acc_log,
        "train_loss_log":         history.get("train_loss_log", []),
    }, RUN_DIR / model_filename)
    print(f"  [Trial {trial.number}] Checkpoint saved: {model_filename}")

    # Per-user adapt-and-eval (the HPO signal) using the best state
    trained_model.load_state_dict(best_state)
    user_metrics: dict = defaultdict(list)
    for batch in val_dl:
        uid     = batch["user_id"]
        metrics = mamlpp_adapt_and_eval(
            trained_model, config, batch["support"], batch["query"])
        user_metrics[uid].append(metrics["acc"])

    per_user_means = [float(np.mean(accs)) for accs in user_metrics.values()]
    mean_acc = float(np.nanmean(per_user_means))

    if config.get("use_MOE", False):
        # routing on the dev episodes after adaptation (what the model predicts with); needed to read
        # X1 vs X2 (routing vs ensembling) and the per-user routing claim
        from MOE.routing_metrics import collect_episode_routing, format_summary
        rs = collect_episode_routing(trained_model, config, val_dl, adapted=True)
        if rs is not None:
            print(f"  [Trial {trial.number}] Routing on dev episodes (adapted):\n{format_summary(rs)}")
            trial.set_user_attr("routing", {k: v for k, v in rs.items() if not isinstance(v, list)})

    trial.set_user_attr("global_best_val_acc",  float(global_best_val_acc))
    trial.set_user_attr("burnin_best_val_acc",  burnin_best_val_acc)
    trial.set_user_attr("burnin_best_epoch",    burnin_best_epoch)
    trial.set_user_attr("n_epochs_run",         n_epochs_run)
    trial.set_user_attr("per_user_val_accs",    per_user_means)
    print(f"  [Trial {trial.number}] Final HPO score (post-burnin adapt-eval): {mean_acc*100:.2f}%")
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

    # ── Load tensor_dict once — reuse for both the flat DL and episodic eval ─
    from ablation_config import tensor_dict_file as _tdp
    tensor_dict_path = _tdp(config)
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
    if "adann_lambda" in config:          # P3: gradient-reversal user head during pretraining
        from arch_variants.adann import adann_pretrain
        trained_model, history = adann_pretrain(model, train_dl, val_dl_flat, config)
    else:
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
        # NOTE: kwarg renamed target_trial_indices -> target_trial_reps in
        # maml_data_pipeline.MetaGestureDataset; this call site was missed by that
        # rename and raised TypeError. Lookup tolerates either config key name.
        target_trial_reps   = config.get("target_trial_reps",
                                    config.get("target_trial_indices")),
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],
        q_query                = config["q_query"],
        num_eval_episodes      = config["num_eval_episodes"],
        is_train               = False,
        seed                   = FIXED_SEED,
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


def _objective_metric(trial: optuna.Trial, config: dict) -> float:
    """
    Objective for the episodic metric learners (P1 family): same cached dev episodes, same per-user
    mean at 10-way 1-shot, same burn-in and pruning as _objective_episodic; no inner loop.
    """
    from MAML.maml_data_pipeline import get_maml_dataloaders
    from arch_variants.metric_learners import metric_pretrain, metric_adapt_and_eval
    from ablation_config import tensor_dict_file as _tdp

    HPO_BURNIN_EPOCHS = 5
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=_tdp(config))
    model = build_model(config)

    def _prune_cb(epoch, val_acc):
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trained_model, history = metric_pretrain(model, config, train_dl, episodic_val_loader=val_dl,
                                             epoch_callback=_prune_cb)
    val_acc_log = history["val_acc_log"]
    post = val_acc_log[HPO_BURNIN_EPOCHS:] or val_acc_log
    trial.set_user_attr("burnin_best_val_acc", float(max(post)) if post else 0.0)
    trial.set_user_attr("n_epochs_run", len(val_acc_log))

    user_metrics: dict = defaultdict(list)
    for batch in val_dl:
        m = metric_adapt_and_eval(trained_model, config, batch["support"], batch["query"])
        user_metrics[batch["user_id"]].append(m["acc"])
    per_user_means = [float(np.mean(a)) for a in user_metrics.values()]
    mean_acc = float(np.nanmean(per_user_means))
    trial.set_user_attr("per_user_val_accs", per_user_means)
    print(f"  [Trial {trial.number}] metric learner ({config['metric_kind']}, {config['metric_backbone']}): "
          f"dev per-user mean {mean_acc*100:.2f}%")
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
    # This is the same dir as ablation_config FYI
    sys.path.append(os.path.join(os.path.dirname(__file__), 'test_eval_files'))
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
        # NOTE: kwarg renamed target_trial_indices -> target_trial_reps in
        # maml_data_pipeline.MetaGestureDataset; this call site was missed by that
        # rename and raised TypeError. Lookup tolerates either config key name.
        target_trial_reps   = config.get("target_trial_reps",
                                    config.get("target_trial_indices")),
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],
        q_query                = config["q_query"],
        num_eval_episodes      = config["num_eval_episodes"],
        is_train               = False,
        seed                   = FIXED_SEED,
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

def _record_timing(trial, ablation_id: str, start: float, state: str) -> None:
    """
    Per-trial cost, stored in the Optuna journal (user attrs) AND printed as one greppable line, for
    every trial, including pruned and crashed ones. hpo_report.py reads the journal; the line is for
    grep: grep -h HPO_TRIAL_TIMING <log dir>/*.out
    """
    import socket
    wall = time.time() - start
    info = {
        "ablation": ablation_id, "trial": trial.number, "state": state,
        "wall_time_s": round(wall, 1), "gpu_hours": round(wall / 3600.0, 4),
        "n_params": LAST_MODEL_N_PARAMS,
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                            if torch.cuda.is_available() else None),
        "emg_repr": os.environ.get("EMG_REPR"),
    }
    if not isinstance(trial, optuna.trial.FixedTrial):
        for k, v in info.items():
            trial.set_user_attr(k, v)
    print("HPO_TRIAL_TIMING " + json.dumps(info), flush=True)


def objective(trial: optuna.Trial, ablation_id: str, profile: dict) -> float:
    trial_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("=" * 80)
    print(f"[{ablation_id} | Trial {trial.number}] Starting  ({profile['note']})")
    print("=" * 80)
    state = "FAIL"
    try:
        result = _objective_dispatch(trial, ablation_id, profile)
        state = "COMPLETE"
        return result
    except optuna.TrialPruned:
        state = "PRUNED"
        raise
    finally:
        _record_timing(trial, ablation_id, trial_start, state)


def _objective_dispatch(trial: optuna.Trial, ablation_id: str, profile: dict) -> float:
    trial_start = time.time()
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
        elif profile["eval_mode"] == "metric":
            result = _objective_metric(trial, config)
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

def study_name_for(ablation_id: str) -> str:
    """Optuna study name; includes the EMG representation so env20 and fs64 studies never mix."""
    emg_repr = os.environ.get("EMG_REPR")
    if not emg_repr:
        raise SystemExit("EMG_REPR is not set (hpo_ablation_launcher.sh --emg-repr env20|fs64)")
    # HPO_STUDY_SUFFIX (launcher --study-suffix) separates e.g. a screening study or an E x k grid
    # from the main study of the same ablation
    suffix = os.environ.get("HPO_STUDY_SUFFIX", "")
    return f"ablation_{ablation_id}_10w1s_kfold7_{emg_repr}_v1" + (f"_{suffix}" if suffix else "")


def run_study(ablation_id: str, profile: dict, n_trials: int = 1) -> optuna.Study:
    sleep_time = random.uniform(0, 10)
    print(f"Staggering start: sleeping {sleep_time:.2f}s …")
    time.sleep(sleep_time)

    use_journal = int(os.environ.get("HPO_USE_JOURNAL", "1"))
    study_name  = study_name_for(ablation_id)
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

    # n_startup_trials: TPE runs pure random search for this many trials before
    # fitting its surrogate model. Set to max(10, len(warm_start)) so TPE waits
    # until all enqueued warm-start points have been evaluated before modelling.
    # With 10 warm-starts and %5 concurrency (~40min trials), wave 1 runs the
    # warm-starts, finishes, and writes to the journal before wave 2 samples —
    # so TPE has real posteriors when it starts suggesting new configs.
    # Previously max(20, len(warm_start)) = 20 burned two full waves on random
    # search and amplified the duplicate-sampling problem.
    n_startup = max(10, len(warm_start))

    # Per-worker TPE seed — CRITICAL for avoiding duplicate configs.
    # Root cause of the all-identical-trials bug: every array task is a fresh
    # process that independently initialises TPESampler(seed=FIXED_SEED=42).
    # During the n_startup random phase, each worker draws the same sequence
    # of random configs from that seed before any results are in the journal
    # (trials take ~40min; the 0-10s stagger is irrelevant at that scale).
    # Fix: derive a unique seed per worker from SLURM_ARRAY_TASK_ID so that
    # concurrent startup trials explore different regions of the search space.
    # Mixing with FIXED_SEED means a preempted task reruns deterministically.
    import hashlib as _hashlib
    _task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    _worker_seed = int(
        _hashlib.md5(f"{_task_id}_{FIXED_SEED}".encode()).hexdigest()[:8], 16
    )
    print(f"TPE worker seed: {_worker_seed} (task_id={_task_id}, fixed_seed={FIXED_SEED})")

    sampler = optuna.samplers.TPESampler(
        seed=_worker_seed,
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
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
    parser.add_argument("--replay-trial", type=int, default=None,
                        help="Re-evaluate this finished trial's exact config (top-3 re-evaluation) "
                             "instead of running a new trial. Needs --seed.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed for --replay-trial.")
    parser.add_argument("--enqueue-json", type=str, default=None,
                        help="JSON list of param dicts to enqueue into the study (e.g. an E x top_k grid "
                             "from hpo_tools.py grid). Workers run queued trials before sampling new ones.")
    parser.add_argument("--enqueue-only", action="store_true",
                        help="With --enqueue-json: enqueue and exit (run once, on the login node, BEFORE "
                             "launching the array, so parallel workers cannot enqueue duplicates).")
    args = parser.parse_args()

    if args.enqueue_json:
        if not args.enqueue_only:
            raise SystemExit("--enqueue-json needs --enqueue-only (enqueue once, then launch the array)")
        _name = study_name_for(args.ablation)
        _study = optuna.create_study(
            study_name=_name, direction="maximize", load_if_exists=True,
            storage=JournalStorage(JournalFileBackend(str(HPO_DB_DIR / f"{_name}.log"))))
        _items = json.loads(Path(args.enqueue_json).read_text())
        if len(_study.trials) > 0:
            raise SystemExit(f"study {_name} already has {len(_study.trials)} trials; refusing to enqueue "
                             f"into it (use a new --study-suffix)")
        for _p in _items:
            _study.enqueue_trial(_p)
        print(f"Enqueued {len(_items)} trials into {_name} ({HPO_DB_DIR / (_name + '.log')})")
        sys.exit(0)

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

    if args.replay_trial is None:
        run_study(ablation_id=ablation_id, profile=profile, n_trials=N_TRIALS)
    else:
        if args.seed is None:
            raise SystemExit("--replay-trial needs --seed")
        SEED_OVERRIDE = args.seed
        VAL_EPISODES_OVERRIDE = 100            # all cached dev episodes per user
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        study_name = study_name_for(ablation_id)
        study = optuna.load_study(
            study_name=study_name,
            storage=JournalStorage(JournalFileBackend(str(HPO_DB_DIR / f"{study_name}.log"))))
        _t = study.trials[args.replay_trial]
        # frozen params of a screening study (HPO_FIXED_PARAMS_JSON) are not Optuna params; restore them
        params = {**_t.user_attrs.get("fixed_params", {}), **_t.params}
        score = objective(optuna.trial.FixedTrial(params), ablation_id, profile)
        out = RUN_DIR / f"reeval_{ablation_id}_trial{args.replay_trial}_seed{args.seed}.json"
        out.write_text(json.dumps({"ablation": ablation_id, "trial": args.replay_trial,
                                   "seed": args.seed, "score": score, "params": params}, indent=1))
        print(f"[replay] trial {args.replay_trial} seed {args.seed}: dev acc {score*100:.2f}% -> {out}")