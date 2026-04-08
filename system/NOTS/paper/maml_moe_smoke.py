# =============================================================================
# SINGLE-RUN DEBUG / ABLATION SCRIPT
# =============================================================================
# This is the non-HPO version of hpo_study.py. All hyperparameters are hardcoded
# as the centroid of the top-10 warm-start trials from the v1 HPO study, with
# categorical params chosen by majority vote and continuous params chosen via
# geometric mean (appropriate for log-scale params).
#
# INTENDED USE:
#   1. Sanity-check that val acc is converging correctly on a full training run.
#   2. Manual ablation: override any param below (e.g. maml_inner_steps) to
#      test whether val acc is driven by that param or is a bug artefact.
#   3. Does NOT use Optuna at all — no sampler, no study, no journal files.
#
# USAGE:
#   python train_single_run.py --model_type DeepCNNLSTM
#   python train_single_run.py --model_type DeepCNNLSTM --fold_idx 0
#
# ABLATION EXAMPLE (override from CLI or just edit the OVERRIDE section below):
#   Scroll down to "ABLATION OVERRIDES" and change any param directly.
# =============================================================================

import os
import copy
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()
print(f"CODE_DIR: {CODE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RUN_DIR:  {RUN_DIR}")

results_save_dir = RUN_DIR
models_save_dir  = RUN_DIR
results_save_dir.mkdir(parents=True, exist_ok=True)
models_save_dir.mkdir(parents=True, exist_ok=True)

user_split_json_filepath = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M")


# ── Imports that live inside your codebase ────────────────────────────────────
from system.MAML.mamlpp import *
from system.MAML.maml_data_pipeline import get_maml_dataloaders
from system.MAML.shared_maml import *
from system.pretraining.pretrain_models import build_model
from system.pretraining.contrastive_net.contrastive_encoder import ContrastiveGestureEncoder


# =============================================================================
# HELPERS  (copied verbatim from hpo_study.py so this file is self-contained)
# =============================================================================

def inject_model_config(config: dict, model_type: str,
                        cnn_base_filters: int = None,
                        lstm_hidden: int = None):
    config["model_type"] = model_type
    config["sequence_length"] = 64
    config["emg_in_ch"]  = 16
    config["imu_in_ch"]  = 72
    config["demo_in_dim"] = 12

    if model_type == "MetaCNNLSTM":
        config.update({
            "cnn_filters": 32, "emg_cnn_layers": 1, "imu_cnn_layers": 1,
            "cnn_kernel": 5, "groupnorm_num_groups": 8,
            "lstm_hidden": 32, "lstm_layers": 1, "bidirectional": False,
            "head_type": "linear",
        })
    elif model_type == "DeepCNNLSTM":
        _cnn_base_filters = cnn_base_filters if cnn_base_filters is not None else 32
        _lstm_hidden      = lstm_hidden      if lstm_hidden      is not None else 64
        config.update({
            "cnn_base_filters": _cnn_base_filters, "cnn_layers": 3,
            "cnn_kernel": 5, "groupnorm_num_groups": 8,
            "lstm_hidden": _lstm_hidden, "lstm_layers": 3, "bidirectional": True,
            "head_type": "mlp",
        })
    elif model_type == "TST":
        config.update({
            "patch_len": 8, "d_model": 64, "n_heads": 4, "n_blocks": 3,
        })
    elif model_type == "ContrastiveNet":
        config.update({"arch_mode": "cnn_attn"})
        config.update({
            "train_reps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "val_reps":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "emg_base_cnn_filters": 64, "emg_cnn_layers": 3,
            "imu_base_cnn_filters": 32, "imu_cnn_layers": 2,
            "cnn_kernel_size": 5, "emg_stride": 1, "imu_stride": 1,
            "groupnorm_num_groups": 8,
            "use_lstm": False, "lstm_hidden": 128, "lstm_layers": 2,
            "use_GlobalAvgPooling": True,
            "attn_pool_heads": 4,
            "embedding_dim": 128, "proj_hidden_dim": 256,
            "num_val_episodes": 20,
            "lr_scheduler": "cosine", "lr_warmup_epochs": 5, "lr_min": 1e-6,
            "grad_clip": 5.0, "log_interval": 100,
        })
    else:
        raise ValueError(f"model_type '{model_type}' not recognised in inject_model_config.")

    return config


def apply_fold_to_config(config, all_splits, fold_idx):
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]


# =============================================================================
# HARDCODED HYPERPARAMETERS
# =============================================================================
# Derived from top-10 v1 warm-start trials:
#   • Categorical / integer params  → majority vote
#   • Continuous params (log-scale) → geometric mean
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ABLATION OVERRIDES — change anything here to run a manual ablation.     │
# │  All other params stay at the centroid values.                           │
# └──────────────────────────────────────────────────────────────────────────┘
ABLATION_OVERRIDES: dict = {
    # Examples (uncomment to activate):
    "maml_inner_steps":      5,      # was 5 at centroid; try 1, 3, 20 to test sensitivity
    "maml_inner_steps_eval": 100,     # was 100 at centroid
    # "maml_alpha_init":       0.001,
    "num_experts":           32,
    "MOE_top_k":             8,
    # "use_lslr_at_eval":      True,
    # "label_smooth":          0.0,    # ablate label smoothing entirely
    # "MOE_aux_coeff":         0.0,    # ablate load-balancing loss
    # "episodes_per_epoch_train": 50,  # reduce for a fast smoke-test run
    # "num_epochs":            5,      # reduce for a fast smoke-test run
}

# ── Centroid values ───────────────────────────────────────────────────────────

# MAML core  (majority vote / geometric mean)
MAML_INNER_STEPS          = 5         # majority vote: 5 appeared 6/10
MAML_INNER_STEPS_EVAL     = 100       # majority vote: 100 appeared 9/10
MAML_ALPHA_INIT           = 0.00165   # geometric mean of top-10
MAML_ALPHA_INIT_EVAL      = 0.0194    # geometric mean of top-10
OUTER_LR                  = 0.000305  # geometric mean of top-10
WEIGHT_DECAY              = 0.000382  # geometric mean of top-10

# Architecture
CNN_BASE_FILTERS          = 128       # majority vote: 128 appeared 8/10
LSTM_HIDDEN               = 128       # majority vote: 128 appeared 9/10
GROUPNORM_NUM_GROUPS      = 4         # majority vote: 4 appeared 9/10
USE_GLOBAL_AVG_POOLING    = False     # majority vote: False appeared 6/10

# Training schedule
META_BATCHSIZE            = 24        # majority vote: 24 appeared 9/10
EPISODES_PER_EPOCH_TRAIN  = 200       # majority vote: 200 appeared 6/10
NUM_EPOCHS                = 50

# Regularisation
LABEL_SMOOTH              = 0.15      # majority vote: 0.15 appeared 7/10

# MAML++
USE_MAML_MSL              = False     # majority vote: False appeared 9/10
MAML_USE_LSLR             = True      # unanimous: True in all 10
USE_LSLR_AT_EVAL          = False     # majority vote: False appeared 8/10

# MoE
NUM_EXPERTS               = 31        # rounded mean of top-10 (mean ≈ 30.6)
MOE_TOP_K                 = 8         # rounded mean of top-10 (mean ≈ 7.8)
MOE_GATE_TEMPERATURE      = 0.652     # geometric mean of top-10
MOE_AUX_COEFF             = 0.046     # geometric mean of top-10
MOE_AUX_LOSS_PLCMT        = "both"    # majority vote: "both" appeared 9/10

# Fixed MoE settings (no meaningful variation in v1)
MOE_CTX_OUT_DIM    = 32
MOE_CTX_HIDDEN_DIM = 32
MOE_DROPOUT        = 0.05


def _apply_overrides(params: dict, overrides: dict) -> dict:
    """Apply ABLATION_OVERRIDES on top of the centroid params. Mutates in-place."""
    for k, v in overrides.items():
        if k not in params:
            raise KeyError(
                f"ABLATION_OVERRIDES key '{k}' not found in params dict. "
                f"Check for typos. Valid keys: {sorted(params.keys())}"
            )
        print(f"  [ABLATION] {k}: {params[k]} → {v}")
        params[k] = v
    return params


# =============================================================================
# MODEL / CONFIG BUILDER  (mirrors build_model_from_trial, no trial object)
# =============================================================================

def build_model_single_run(model_type: str, base_config: dict = None) -> tuple:
    config = copy.deepcopy(base_config) if base_config else {}

    # ── Collect all centroid params into one dict so overrides can be validated ──
    hp = {
        "maml_inner_steps":          MAML_INNER_STEPS,
        "maml_inner_steps_eval":     MAML_INNER_STEPS_EVAL,
        "maml_alpha_init":           MAML_ALPHA_INIT,
        "maml_alpha_init_eval":      MAML_ALPHA_INIT_EVAL,
        "outer_lr":                  OUTER_LR,
        "wd":                        WEIGHT_DECAY,
        "groupnorm_num_groups":      GROUPNORM_NUM_GROUPS,
        "use_GlobalAvgPooling":      USE_GLOBAL_AVG_POOLING,
        "meta_batchsize":            META_BATCHSIZE,
        "episodes_per_epoch_train":  EPISODES_PER_EPOCH_TRAIN,
        "num_epochs":                NUM_EPOCHS,
        "label_smooth":              LABEL_SMOOTH,
        "use_maml_msl":              USE_MAML_MSL,
        "maml_use_lslr":             MAML_USE_LSLR,
        "use_lslr_at_eval":          USE_LSLR_AT_EVAL,
        "num_experts":               NUM_EXPERTS,
        "MOE_top_k":                 MOE_TOP_K,
        "MOE_gate_temperature":      MOE_GATE_TEMPERATURE,
        "MOE_aux_coeff":             MOE_AUX_COEFF,
        "MOE_aux_loss_plcmt":        MOE_AUX_LOSS_PLCMT,
        # Architecture (fixed for DeepCNNLSTM with pretrain_approach='None')
        "cnn_base_filters":          CNN_BASE_FILTERS,
        "lstm_hidden":               LSTM_HIDDEN,
    }

    # Apply any manual ablation overrides
    if ABLATION_OVERRIDES:
        print("\n[ABLATION OVERRIDES ACTIVE]")
        _apply_overrides(hp, ABLATION_OVERRIDES)
        print()

    # Architecture
    config = inject_model_config(
        config, model_type,
        cnn_base_filters=hp["cnn_base_filters"],
        lstm_hidden=hp["lstm_hidden"],
    )
    # inject_model_config sets groupnorm_num_groups to 8 for DeepCNNLSTM;
    # override with the HPO'd centroid value.
    config["groupnorm_num_groups"] = hp["groupnorm_num_groups"]

    config["use_MOE"]             = True
    config["pretrain_approach"]   = "None"
    config["pretrained_model_filename"] = None
    # Task
    config["n_way"]        = 3
    config["k_shot"]       = 1
    config["q_query"]      = 9
    config["num_classes"]  = 10
    config["feature_engr"] = "None"

    # NOTS / paths
    config["NOTS"] = True
    config["user_split_json_filepath"] = user_split_json_filepath
    config["results_save_dir"]         = results_save_dir
    config["models_save_dir"]          = models_save_dir
    config["emg_imu_pkl_full_path"] = f"{CODE_DIR}//dataset//filtered_datasets//metadata_IMU_EMG_allgestures_allusers.pkl"
    config["pwmd_xlsx_filepath"]    = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants with disabilities.xlsx"
    config["pwoutmd_xlsx_filepath"] = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants without disabilities.xlsx"
    config["dfs_save_path"]         = f"{CODE_DIR}/dataset//"
    config["dfs_load_path"]         = f"{CODE_DIR}/dataset/meta-learning-sup-que-ds//"
    config["pretrain_dir"]          = str(CODE_DIR / "pretrain_outputs" / "checkpoints" / "")

    # Debug flags
    config["track_gradient_alignment"]      = False
    config["debug_verbose"]                 = True  # NOTE: Turned this to True for the single run version!
    config["gradient_clip_max_norm"]        = 10.0
    config["num_eval_episodes"]             = 10
    config["debug_one_user_only"]           = False
    config["debug_one_episode"]             = False
    config["debug_five_episodes"]           = False

    # MAML core
    config["meta_batchsize"]            = hp["meta_batchsize"]
    config["maml_inner_steps"]          = hp["maml_inner_steps"]
    config["maml_inner_steps_eval"]     = hp["maml_inner_steps_eval"]
    config["maml_alpha_init"]           = hp["maml_alpha_init"]
    config["maml_alpha_init_eval"]      = hp["maml_alpha_init_eval"]
    config["learning_rate"]             = hp["outer_lr"]
    config["weight_decay"]              = hp["wd"]

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["use_batch_norm"] = False
    config["dropout"]        = 0.1
    config["emg_stride"]     = 1
    config["imu_stride"]     = 1
    config["padding"]        = 0
    config["use_GlobalAvgPooling"] = hp["use_GlobalAvgPooling"]

    # Multimodal
    config["multimodal"]          = True
    config["use_imu"]             = True
    config["use_demographics"]    = False
    config["use_film_x_demo"]     = False
    config["FILM_on_context_or_demo"] = "context"

    # MoE
    config["num_experts"]          = hp["num_experts"]
    config["MOE_top_k"]            = hp["MOE_top_k"]
    config["top_k"]                = hp["MOE_top_k"]
    config["MOE_placement"]        = "encoder"
    config["MOE_gate_temperature"] = hp["MOE_gate_temperature"]
    config["MOE_aux_coeff"]        = hp["MOE_aux_coeff"]
    config["MOE_ctx_out_dim"]      = MOE_CTX_OUT_DIM
    config["MOE_ctx_hidden_dim"]   = MOE_CTX_HIDDEN_DIM
    config["MOE_dropout"]          = MOE_DROPOUT
    config["MOE_expert_expand"]    = 1.0
    config["MOE_mlp_hidden_mult"]  = 1.0
    config["MOE_log_every"]        = 5
    config["MOE_plot_dir"]         = None
    config["gate_type"]            = "context_feature_demo"
    config["expert_architecture"]  = "MLP"
    config["apply_MOE_aux_loss_inner_outer"] = hp["MOE_aux_loss_plcmt"]

    # Training schedule
    config["use_label_shuf_meta_aug"]    = True
    config["num_epochs"]                 = hp["num_epochs"]
    config["episodes_per_epoch_train"]   = hp["episodes_per_epoch_train"]
    config["label_smooth"]               = hp["label_smooth"]

    config["maml_gesture_classes"]   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["target_trial_indices"]   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Optimiser / early-stopping
    config["optimizer"]                 = "adam"
    config["use_earlystopping"]         = True
    config["earlystopping_patience"]    = 8
    config["earlystopping_min_delta"]   = 0.005

    # MAML++ flags
    config["meta_learning"]             = True
    config["num_workers"]               = 8
    config["use_maml_msl"]             = hp["use_maml_msl"]
    config["maml_msl_num_epochs"]       = 0
    config["maml_opt_order"]            = "first"
    config["maml_first_order_to_second_order_epoch"] = 1_000_000
    config["maml_use_lslr"]            = hp["maml_use_lslr"]
    config["enable_inner_loop_optimizable_bn_params"] = False
    config["use_lslr_at_eval"]         = hp["use_lslr_at_eval"]

    # LR scheduler (not active unless use_cosine_outer_lr=True)
    config["use_cosine_outer_lr"]    = False
    config["lr_scheduler_factor"]    = 0.1
    config["lr_scheduler_patience"]  = 6

    # ── Model construction ────────────────────────────────────────────────────
    if model_type in ("MetaCNNLSTM", "DeepCNNLSTM", "TST"):
        if config["use_MOE"]:
            from MOE.MOE_encoder import build_MOE_model
            model = build_MOE_model(config)
        else:
            model = build_model(config)
    elif model_type == "ContrastiveNet":
        model = ContrastiveGestureEncoder(config)
    else:
        raise ValueError(f"model_type '{model_type}' not recognised in build_model_single_run.")

    print(f"--> pretrain_approach='None' — using random initialisation for {model_type}.")

    model.to(config["device"])
    return model, config


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def run_single(model_type: str, fold_idx: int):
    print("=" * 80)
    print(f"  SINGLE-RUN TRAINING  |  model={model_type}  |  fold={fold_idx}")
    print("=" * 80)

    # ── Load splits ───────────────────────────────────────────────────────────
    with open(user_split_json_filepath, "r") as f:
        all_splits = json.load(f)

    # ── Build model + config ──────────────────────────────────────────────────
    model, config = build_model_single_run(model_type, base_config={})
    apply_fold_to_config(config, all_splits, fold_idx)

    print("\nFINAL CONFIG:")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print()

    if config["device"].type == "cpu":
        print("WARNING: training on CPU — this will be extremely slow!")

    # ── Data ─────────────────────────────────────────────────────────────────
    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    episodic_train_loader, episodic_val_loader = get_maml_dataloaders(
        config,
        tensor_dict_path=tensor_dict_path,
    )

    # ── MAML meta-training ───────────────────────────────────────────────────
    run_start = time.time()
    pretrained_model, pretrain_res_dict = mamlpp_pretrain(
        model,
        config,
        episodic_train_loader,
        episodic_val_loader=episodic_val_loader,
    )
    best_val_acc = pretrain_res_dict["best_val_acc"]
    best_state   = pretrain_res_dict["best_state"]
    print(f"\nMeta-training complete. Best val acc = {best_val_acc:.4f}")
    print(f"Total training time: {(time.time() - run_start) / 3600:.2f} hours")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    model_filename = f"single_run_{model_type}_fold{fold_idx}_{timestamp}_best.pt"
    save_path = os.path.join(models_save_dir, model_filename)
    torch.save({
        "model_type":       model_type,
        "fold_idx":         fold_idx,
        "model_state_dict": best_state,
        "config":           config,
        "best_val_acc":     best_val_acc,
        "train_loss_log":   pretrain_res_dict["train_loss_log"],
        "train_acc_log":    pretrain_res_dict["train_acc_log"],
        "val_loss_log":     pretrain_res_dict["val_loss_log"],
        "val_acc_log":      pretrain_res_dict["val_acc_log"],
        "ablation_overrides": ABLATION_OVERRIDES,
    }, save_path)
    print(f"Checkpoint saved to: {save_path}")

    # ── Per-user adaptation & evaluation ─────────────────────────────────────
    model.load_state_dict(best_state)
    user_metrics = defaultdict(list)

    for batch in episodic_val_loader:
        user_id     = batch["user_id"]
        support_set = batch["support"]
        query_set   = batch["query"]
        val_metrics = mamlpp_adapt_and_eval(model, config, support_set, query_set)
        user_metrics[user_id].append(val_metrics["acc"])

    all_user_means = []
    print("\nPer-user results:")
    for user_id, accs in user_metrics.items():
        m_acc = np.mean(accs)
        all_user_means.append(float(m_acc))
        print(f"  User {user_id} | Acc: {m_acc * 100:.2f}%  (over {len(accs)} episodes)")

    mean_acc = np.mean(all_user_means)
    std_acc  = np.std(all_user_means)
    print(f"\nFinal summary | Mean acc: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"Per-user accs (%): {[round(a * 100, 2) for a in all_user_means]}")

    return mean_acc


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-run (non-HPO) MAML++ training for manual debugging and ablation."
    )
    parser.add_argument(
        "--model_type", type=str, default="DeepCNNLSTM",
        choices=["MetaCNNLSTM", "DeepCNNLSTM", "TST", "ContrastiveNet"],
        help="Which model architecture to train.",
    )
    parser.add_argument(
        "--fold_idx", type=int, default=0,
        help="Which cross-validation fold to use (0-indexed).",
    )
    parser.add_argument(
    "--overrides", type=str, default="{}",
    help=(
        'JSON string of ablation overrides, e.g.: '
        '--overrides \'{"maml_inner_steps": 5, "num_epochs": 3}\''),
    )
    # NOTE: Smoke accepts these arugments but doesnt do anything with them... why do we not use them lol
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir",  type=str)
    args = parser.parse_args()

    # Seed everything for reproducibility
    FIXED_SEED = 42
    import random
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    ABLATION_OVERRIDES = json.loads(args.overrides)

    run_single(model_type=args.model_type, fold_idx=args.fold_idx)
