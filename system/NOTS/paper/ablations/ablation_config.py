"""
ablation_config.py
==================
Single source of truth for all ablation hyperparameters and shared utilities.

The "best" config is derived from the M0 HPO warm-start trials. Per the spec:
  - FIXED params use their specified fixed values.
  - TUNE params use the median / mode of the warm-start top-10 trials, favouring
    the settings that appeared most frequently among the top cluster.

Reviewed choices (justify each if a reviewer asks):
  outer_lr              = 3e-4  (median of top trials: 2.6e-4 to 5.3e-4)
  maml_alpha_init       = 1.7e-3 (median of top trials)
  maml_alpha_init_eval  = 0.017  (median of top trials)
  maml_inner_steps      = 5      (most common, spec says keep 5 competitive)
  wd                    = 5.6e-4 (median)
  label_smooth          = 0.15   (mode of top-10)
  episodes_per_epoch    = 200    (mode of top-10)
  num_experts           = 32     (near-mode of top-10, spec says "best val")
  MOE_top_k             = 9      (mode of top-10)
  MOE_gate_temperature  = 0.65   (median)
  MOE_aux_coeff         = 0.023  (median, "lower is better" per spec)
"""

import os
import json
import copy
import warnings
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

# ── Suppress noisy torch weights_only warning ─────────────────────────────────
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Reproducibility seed ──────────────────────────────────────────────────────
FIXED_SEED = 42
# Number of seeds for final reporting (spec: 5)
NUM_FINAL_SEEDS = 5
# Episodic eval episodes (spec: 500 for test)
NUM_TEST_EPISODES = 500
NUM_VAL_EPISODES  = 200

# ── Environment paths (same convention as HPO script) ─────────────────────────
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()

print(f"CODE_DIR: {CODE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RUN_DIR : {RUN_DIR}")

RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── Split file ────────────────────────────────────────────────────────────────
USER_SPLIT_JSON = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"

with open(USER_SPLIT_JSON, "r") as f:
    ALL_SPLITS = json.load(f)

# We use fold 0 for all ablations so every model is evaluated on the same
# held-out test users. Training stability is checked on val_PIDs; final
# reported numbers use test_PIDs. See the ablation spec section 1.3.
FOLD_IDX = 0
TRAIN_PIDS = ALL_SPLITS[FOLD_IDX]["train"]
VAL_PIDS   = ALL_SPLITS[FOLD_IDX]["val"]
TEST_PIDS  = ALL_SPLITS[FOLD_IDX]["test"]

print(f"Using fold {FOLD_IDX}: {len(TRAIN_PIDS)} train | "
      f"{len(VAL_PIDS)} val | {len(TEST_PIDS)} test PIDs")


# =============================================================================
# BASE CONFIG  (shared across all ablations; override in each script)
# =============================================================================

def make_base_config(ablation_id: str) -> dict:
    """
    Returns the shared base config dict.  Each ablation script calls this and
    then modifies only the keys it needs to change.

    Args:
        ablation_id: e.g. "M0", "A1", etc.  Used for save-path naming.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config: dict = {}

    # ── Identity ──────────────────────────────────────────────────────────────
    config["ablation_id"]  = ablation_id
    config["timestamp"]    = timestamp
    config["model_type"]   = "DeepCNNLSTM"   # default; override for special cases

    # ── Paths ─────────────────────────────────────────────────────────────────
    config["user_split_json_filepath"] = str(USER_SPLIT_JSON)
    config["results_save_dir"]         = str(RUN_DIR)
    config["models_save_dir"]          = str(RUN_DIR)
    config["emg_imu_pkl_full_path"]    = str(DATA_DIR / "filtered_datasets"
                                              / "metadata_IMU_EMG_allgestures_allusers.pkl")
    config["dfs_load_path"]  = str(CODE_DIR / "dataset" / "meta-learning-sup-que-ds") + "/"
    config["pretrain_dir"]   = str(CODE_DIR / "pretrain_outputs" / "checkpoints") + "/"
    config["NOTS"] = True   # cluster run flag (keeps old local-path logic from triggering)

    # ── User splits ───────────────────────────────────────────────────────────
    config["train_PIDs"] = TRAIN_PIDS
    config["val_PIDs"]   = VAL_PIDS
    config["test_PIDs"]  = TEST_PIDS

    # ── Input dimensions (FIXED per spec) ─────────────────────────────────────
    config["sequence_length"] = 64
    config["emg_in_ch"]       = 16
    config["imu_in_ch"]       = 72
    config["demo_in_dim"]     = 12

    # ── Modality flags ────────────────────────────────────────────────────────
    config["multimodal"]       = True
    config["use_imu"]          = True
    config["use_demographics"] = False
    config["use_film_x_demo"]  = False
    config["FILM_on_context_or_demo"] = "context"

    # ── Task setup ────────────────────────────────────────────────────────────
    config["n_way"]      = 3    # eval/finetuning: 3-way classification
    config["k_shot"]     = 1
    config["q_query"]    = 9
    config["num_classes"] = 10  # unused by architecture directly; pretrain_num_classes is the
                             # authoritative source for non-MAML head size
    # Pretraining uses all 10 gesture classes (full supervised dataset).
    # The model is built with this many output logits during the pretrain phase.
    # At eval time the head is replaced with a fresh `n_way`-class head.
    config["pretrain_num_classes"] = 10

    # ── Architecture (FIXED per spec) ─────────────────────────────────────────
    config["cnn_base_filters"]    = 128
    config["cnn_layers"]          = 3
    config["cnn_kernel"]          = 5
    config["lstm_hidden"]         = 128
    config["lstm_layers"]         = 3
    config["bidirectional"]       = True
    config["groupnorm_num_groups"] = 4 
    config["use_GlobalAvgPooling"] = False
    config["use_batch_norm"]       = False
    config["dropout"]              = 0.1
    config["head_type"]            = "mlp"
    config["emg_stride"]           = 1
    config["imu_stride"]           = 1
    config["padding"]              = 0

    # ── Best hyperparameters (from HPO warm-start analysis) ───────────────────
    config["learning_rate"]  = 3e-4      # outer_lr
    config["weight_decay"]   = 5.6e-4
    config["label_smooth"]   = 0.15
    config["gradient_clip_max_norm"] = 10.0

    # ── Training schedule ─────────────────────────────────────────────────────
    config["num_epochs"]              = 50
    config["episodes_per_epoch_train"] = 200   # TUNE; best from warm-start mode
    config["num_eval_episodes"]        = NUM_VAL_EPISODES
    config["batch_size"]               = 64    # flat dataloader default
    config["num_workers"]              = 8
    config["optimizer"]                = "adam"
    config["use_cosine_outer_lr"]      = False
    config["lr_scheduler_factor"]      = 0.1
    config["lr_scheduler_patience"]    = 6
    config["use_earlystopping"]        = True
    config["earlystopping_patience"]   = 8
    config["earlystopping_min_delta"]  = 0.005

    # ── MAML core (FIXED + best TUNE per spec) ────────────────────────────────
    config["meta_learning"]           = True
    config["meta_batchsize"]          = 24    # FIXED per spec
    config["maml_inner_steps"]        = 5     # best from warm-start mode
    config["maml_inner_steps_eval"]   = 50    # FIXED per spec
    config["maml_alpha_init"]         = 1.7e-3
    config["maml_alpha_init_eval"]    = 0.017
    config["maml_use_lslr"]           = True   # FIXED per spec
    config["use_lslr_at_eval"]        = False  # FIXED per spec
    config["use_maml_msl"]            = False  # FIXED per spec
    config["maml_msl_num_epochs"]     = 0
    config["maml_opt_order"]          = "first"
    config["maml_first_order_to_second_order_epoch"] = 1_000_000
    config["enable_inner_loop_optimizable_bn_params"] = False

    # ── MoE (FIXED + best TUNE per spec) ─────────────────────────────────────
    config["use_MOE"]                         = True
    config["MOE_placement"]                   = "encoder"   # FIXED per spec
    config["num_experts"]                     = 32
    config["MOE_top_k"]                       = 9
    config["top_k"]                           = config["MOE_top_k"]
    config["MOE_gate_temperature"]            = 0.65
    config["MOE_aux_coeff"]                   = 0.023
    config["MOE_ctx_out_dim"]                 = 32    # FIXED per spec
    config["MOE_ctx_hidden_dim"]              = 32    # FIXED per spec
    config["MOE_dropout"]                     = 0.05  # FIXED per spec
    config["MOE_expert_expand"]               = 1.0
    config["MOE_mlp_hidden_mult"]             = 1.0
    config["MOE_log_every"]                   = 5
    config["MOE_plot_dir"]                    = None
    config["apply_MOE_aux_loss_inner_outer"]  = "inner"  # FIXED per spec (favoured by Optuna)
    # Legacy keys kept for compatibility
    config["gate_type"]              = "context_feature_demo"
    config["expert_architecture"]    = "MLP"

    # ── Gesture / trial selection ─────────────────────────────────────────────
    config["feature_engr"]           = "None"
    config["pretrain_approach"]      = "None"
    config["pretrained_model_filename"] = None
    config["maml_gesture_classes"]   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["available_gesture_classes"] = config["maml_gesture_classes"]
    config["target_trial_indices"]   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Pretrain (flat) dataloader extras
    config["train_reps"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["val_reps"]   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["augment"]    = False
    # For finetuning
    # TODO: I think these should get removed? Idk where they are used... if the eval is episodic we dont need these...
    #config["ft_train_reps"] = [1]
    #config["ft_val_reps"]   = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["ft_label_smooth"] = 0.0

    # ── Augmentation ─────────────────────────────────────────────────────────
    config["use_label_shuf_meta_aug"] = True

    # ── Debug flags ───────────────────────────────────────────────────────────
    config["track_gradient_alignment"] = False
    config["debug_verbose"]            = False
    config["debug_one_user_only"]      = False
    config["debug_one_episode"]        = False
    config["debug_five_episodes"]      = False

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return config


# =============================================================================
# Model builders
# =============================================================================

def build_maml_moe_model(config: dict):
    """Build MAML + MoE model (M0, A5, A8, A9, A12).

    meta_learning=True in base config, so the architecture reads n_way (=3)
    for the classification head — correct for episodic training and eval.
    """
    from MOE.MOE_encoder import build_MOE_model
    model = build_MOE_model(config)
    model.to(config["device"])
    return model


def build_maml_no_moe_model(config: dict):
    """Build MAML + single encoder, no MoE (A3, A4).

    meta_learning=True in base config, so the architecture reads n_way (=3)
    for the classification head — correct for episodic training and eval.
    """
    from pretraining.pretrain_models import build_model
    cfg = copy.deepcopy(config)
    cfg["use_MOE"] = False
    model = build_model(cfg)
    model.to(cfg["device"])
    return model


def build_supervised_moe_model(config: dict):
    """Build supervised (no-MAML) + MoE model (A1).

    meta_learning=False, so the architecture's meta_learning toggle routes to
    pretrain_num_classes (=10) for the head — correct for flat supervised
    pretraining over all gesture classes. At eval time the head is replaced
    with a fresh n_way-class (=3) head by replace_head_for_eval().
    No need to override n_way here; the architecture handles it.
    """
    from MOE.MOE_encoder import build_MOE_model
    cfg = copy.deepcopy(config)
    cfg["use_MOE"] = True
    model = build_MOE_model(cfg)
    model.to(cfg["device"])
    return model


def build_supervised_no_moe_model(config: dict):
    """Build vanilla supervised CNN-LSTM, no MoE (A2, A7).

    meta_learning=False, so the architecture's meta_learning toggle routes to
    pretrain_num_classes (=10) for the head — correct for flat supervised
    pretraining over all gesture classes. At eval time the head is replaced
    with a fresh n_way-class (=3) head by replace_head_for_eval().
    No need to override n_way here; the architecture handles it.
    """
    from pretraining.pretrain_models import build_model
    cfg = copy.deepcopy(config)
    cfg["use_MOE"] = False
    model = build_model(cfg)
    model.to(cfg["device"])
    return model


# =============================================================================
# Head replacement for eval-time transfer learning
# =============================================================================

def replace_head_for_eval(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """
    Replace the pretrained classification head with a fresh `n_way`-class head.

    This is the standard transfer learning protocol: pretrain on all classes,
    then swap in a randomly-initialised head for the target few-shot task.
    The backbone weights are untouched.  Fine-tuning (head_only or full) is
    applied AFTER this replacement by `finetune_and_eval_user`.

    The function inspects `model.head` and replaces the final Linear layer in
    place, preserving any intermediate MLP layers.  Works for both:
      - A simple Linear head  (model.head is nn.Linear)
      - An MLP head           (model.head is nn.Sequential ending in nn.Linear)

    Args:
        model  : pretrained model whose .head attribute will be replaced.
        config : must contain 'n_way' (int) — number of eval classes.

    Returns:
        model with replaced head (same object, modified in-place AND returned).
    """
    import torch.nn as nn

    n_way = int(config["n_way"])
    head  = model.head

    if isinstance(head, nn.Linear):
        in_features  = head.in_features
        model.head   = nn.Linear(in_features, n_way)
        nn.init.xavier_uniform_(model.head.weight)
        nn.init.zeros_(model.head.bias)

    elif isinstance(head, nn.Sequential):
        # Find and replace only the terminal Linear layer
        layers = list(head.children())
        assert isinstance(layers[-1], nn.Linear), (
            f"Expected the last layer of model.head (Sequential) to be nn.Linear, "
            f"got {type(layers[-1])}. Add explicit support for this head type."
        )
        in_features  = layers[-1].in_features
        layers[-1]   = nn.Linear(in_features, n_way)
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        model.head   = nn.Sequential(*layers)

    else:
        raise TypeError(
            f"replace_head_for_eval: model.head is {type(head)}, expected nn.Linear "
            f"or nn.Sequential. Add explicit support for this head type."
        )

    model.head.to(next(model.parameters()).device)
    return model


# =============================================================================
# Seed / reproducibility helpers
# =============================================================================

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Episodic test evaluation
# =============================================================================

def run_episodic_test_eval(model, config: dict, tensor_dict_path: str,
                            test_pids: list, num_episodes: int = NUM_TEST_EPISODES) -> dict:
    """
    Evaluate a MAML model on test_pids using episodic (1-shot 3-way) sampling.
    Returns dict with per-user accs, mean, std.
    """
    from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate, reorient_tensor_dict
    from MAML.mamlpp import mamlpp_adapt_and_eval
    import pickle
    from torch.utils.data import DataLoader
    from collections import defaultdict

    with open(tensor_dict_path, "rb") as f:
        full_dict   = pickle.load(f)
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
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4,
                         collate_fn=maml_mm_collate)

    model.eval()
    user_metrics: dict = defaultdict(list)
    for batch in test_dl:
        uid = batch["user_id"]
        metrics = mamlpp_adapt_and_eval(model, config, batch["support"], batch["query"])
        user_metrics[uid].append(metrics["acc"])

    per_user = {uid: float(np.mean(accs)) for uid, accs in user_metrics.items()}
    vals = list(per_user.values())
    return {
        "per_user_acc": per_user,
        "mean_acc":     float(np.mean(vals)),
        "std_acc":      float(np.std(vals)),
        "num_episodes": num_episodes,
    }


def run_supervised_test_eval(model, config: dict, tensor_dict_path: str,
                              test_pids: list, ft_mode: str,
                              num_episodes: int = NUM_TEST_EPISODES) -> dict:
    """
    Evaluate a non-MAML model using episodic finetune-then-eval.
    ft_mode: 'head_only' | 'full'
    """
    from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate
    from pretraining.pretrain_finetune import finetune_and_eval_user
    import pickle
    from torch.utils.data import DataLoader
    from collections import defaultdict

    with open(tensor_dict_path, "rb") as f:
        full_dict   = pickle.load(f)
    tensor_dict = full_dict["data"]

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
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4,
                         collate_fn=maml_mm_collate)

    model.eval()
    user_metrics: dict = defaultdict(list)
    for batch in test_dl:
        uid     = batch["user_id"]
        support = batch["support"]
        query   = batch["query"]
        metrics = finetune_and_eval_user(
            model, config,
            support_emg=support["emg"], support_imu=support.get("imu"),
            support_labels=support["labels"],
            query_emg=query["emg"],     query_imu=query.get("imu"),
            query_labels=query["labels"],
            mode=ft_mode,
        )
        user_metrics[uid].append(metrics["acc"])

    per_user = {uid: float(np.mean(accs)) for uid, accs in user_metrics.items()}
    vals = list(per_user.values())
    return {
        "per_user_acc": per_user,
        "mean_acc":     float(np.mean(vals)),
        "std_acc":      float(np.std(vals)),
        "ft_mode":      ft_mode,
        "num_episodes": num_episodes,
    }


# =============================================================================
# Save utilities
# =============================================================================

def save_results(results: dict, config: dict, tag: str = ""):
    """Save results dict as JSON to RUN_DIR."""
    import json
    ablation_id = config["ablation_id"]
    ts          = config["timestamp"]
    fname       = f"{ablation_id}_{ts}{('_' + tag) if tag else ''}_results.json"
    fpath       = RUN_DIR / fname
    with open(fpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save_results] Saved to {fpath}")


def save_model_checkpoint(state: dict, config: dict, tag: str = "best"):
    ablation_id = config["ablation_id"]
    ts          = config["timestamp"]
    fname       = f"{ablation_id}_{ts}_{tag}.pt"
    fpath       = RUN_DIR / fname
    torch.save(state, fpath)
    print(f"[save_checkpoint] Saved to {fpath}")
    return str(fpath)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)