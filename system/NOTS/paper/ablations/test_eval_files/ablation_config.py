"""
ablation_config.py
==================
Single source of truth for all ablation hyperparameters and shared utilities.

All tunable hyperparameters are taken from the CONFIRMED best HPO trial:
  Trial 89 of study 'ablation_M0_1s3w_hpo_v1', val_acc = 90.05%
  Log: /scratch/my13/kai/runs/paper/ablations/hpo/M0/trial_89

Parameters sourced directly from the Trial 89 config snapshot in M0_best_log.out.
Each changed value is annotated with its old (pre-Trial-89) value for auditability.

Trial 89 Optuna parameters (the tuned subset):
  outer_lr               = 1.9506e-4   (was 3e-4)
  wd                     = 8.874e-4    (was 5.6e-4)
  maml_inner_steps       = 10          (was 5)
  maml_inner_steps_eval  = 10          (was 50)
  maml_alpha_init        = 9.735e-4    (was 1.7e-3)
  maml_alpha_init_eval   = 5.066e-3    (was 0.017)
  maml_use_lslr          = True        (unchanged)
  use_lslr_at_eval       = False       (unchanged)
  use_maml_msl           = "hybrid"    (was False)
  maml_msl_num_epochs    = 31          (was 0)
  episodes_per_epoch     = 500         (was 200)
  num_experts            = 22          (was 32)
  MOE_top_k              = 9           (unchanged)
  MOE_gate_temperature   = 1.5290      (was 0.65)
  MOE_aux_coeff          = 0.03282     (was 0.023)
  MOE_ctx_out_dim        = 64          (was 32)
  MOE_ctx_hidden_dim     = 32          (unchanged)
  MOE_dropout            = 0.03654     (was 0.05)
  MOE_aux_loss_plcmt     = "outer"     (was "inner")
  label_smooth           = 0.05        (was 0.15)

Architecture params from Trial 89 config snapshot (not Optuna-tuned but
confirmed values from the best run):
  cnn_base_filters       = 64          (was 128)
  lstm_hidden            = 64          (was 128)
  groupnorm_num_groups   = 8           (was 4)
  use_GlobalAvgPooling   = True        (was False)
  use_demographics       = False       (was True)
  num_eval_episodes(val) = 100         (was 200; NUM_VAL_EPISODES kept at 200
                                        for test — see note below)

NOTE on num_eval_episodes: Trial 89 used 100 val episodes during HPO (faster
iteration). We keep NUM_VAL_EPISODES = 200 here for the final ablation runs
to give tighter val accuracy estimates, consistent with ablation spec §1.3.
If you want to exactly reproduce the HPO regime, set NUM_VAL_EPISODES = 100.
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
NUM_VAL_EPISODES  = 200   # see note in module docstring re: Trial 89 using 100

# ── Environment paths (same convention as HPO script) ─────────────────────────
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()

print(f"CODE_DIR: {CODE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RUN_DIR : {RUN_DIR}")

RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── Split file ────────────────────────────────────────────────────────────────
USER_SPLIT_JSON = CODE_DIR / "system" / "fixed_user_splits" / "hpo_strat_kapanji_split.json"

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

    All hyperparameter values reflect Trial 89 (val_acc=90.05%), the confirmed
    best HPO trial from study 'ablation_M0_1s3w_hpo_v1'.

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
    config["user_split_json_filepath"] = str(USER_SPLIT_JSON)  # NOTE: This was only used with HPO. Testing does L2SO
    config["results_save_dir"]         = str(RUN_DIR)
    config["models_save_dir"]          = str(RUN_DIR)
    config["emg_imu_pkl_full_path"]    = str(DATA_DIR / "filtered_datasets"
                                              / "metadata_IMU_EMG_allgestures_allusers.pkl")
    config["dfs_load_path"]  = str(CODE_DIR / "dataset" / "meta-learning-sup-que-ds") + "/"
    config["pretrain_dir"]   = str(CODE_DIR / "pretrain_outputs" / "checkpoints") + "/"
    config["NOTS"] = True   # cluster run flag (keeps old local-path logic from triggering)

    # ── User splits ───────────────────────────────────────────────────────────
    # TODO: Confirm that this is getting overwritten during L2SO
    config["train_PIDs"] = TRAIN_PIDS
    config["val_PIDs"]   = VAL_PIDS
    config["test_PIDs"]  = TEST_PIDS
    config["all_PIDs"]   = TRAIN_PIDS + VAL_PIDS + TEST_PIDS
    config["test_procedure"] = 'hpo_test_split'  # Either 'hpo_test_split' or "L2SO"

    # ── Input dimensions (FIXED per spec) ─────────────────────────────────────
    config["sequence_length"] = 64
    config["emg_in_ch"]       = 16
    config["imu_in_ch"]       = 72
    config["demo_in_dim"]     = 12

    # ── Modality flags ────────────────────────────────────────────────────────
    config["multimodal"]       = True
    config["use_imu"]          = True
    # CHANGED: Trial 89 had use_demographics=False. The HPO apparently found
    # demographics unhelpful (or at least the best trial didn't use them).
    # All ablations should be consistent with this.
    config["use_demographics"] = False   # was True
    config["use_film_x_demo"]  = False
    config["FILM_on_context_or_demo"] = "context"  # TODO: I dont think this is used at all yet?
    ## Also, for the figure at least, we should probably drop ET, Left, and Man from this since otherwise we have 15 cols instead of 12
    #config["demo_dim_labels"] = ["time_disabled", "age", "BMI", "DASH_score", "disability_coding_ET", "disability_coding_MD",
    # "disability_coding_No_Disability", "disability_coding_PN", "disability_coding_SCI", "disability_coding_other", "handedness_Left",
    # "handedness_Right", "gender_Man", "gender_Non-binary", "gender_Woman",
    #]

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

    # ── Architecture ──────────────────────────────────────────────────────────
    # CHANGED: Trial 89 used cnn_base_filters=64, lstm_hidden=64 (was 128 each).
    # These are smaller than the old defaults but are what actually achieved 90.05%.
    config["cnn_base_filters"]    = 64    # was 128
    config["cnn_layers"]          = 3
    config["cnn_kernel"]          = 5
    config["lstm_hidden"]         = 64    # was 128
    config["lstm_layers"]         = 3
    config["bidirectional"]       = True
    # CHANGED: Trial 89 used groupnorm_num_groups=8 (was 4).
    config["groupnorm_num_groups"] = 8    # was 4
    # CHANGED: Trial 89 had use_GlobalAvgPooling=True (was False).
    config["use_GlobalAvgPooling"] = True  # was False
    config["use_batch_norm"]       = False
    config["dropout"]              = 0.1
    config["head_type"]            = "mlp"
    # emg_stride / imu_stride: originally intended as per-expert-CNN stride knobs
    # but were never wired into any model — left here as dead config keys.
    # Commented out so they don't mislead anyone into thinking they do something.
    # config["emg_stride"] = 1
    # config["imu_stride"] = 1
    config["padding"]              = 0

    # ── Shared strided front-end (DeepCNNLSTM_EncoderMOE only) ───────────────
    # front_end_stride > 0  → a single shared Conv1d with that stride is
    #   prepended to the model BEFORE the context projector and expert CNNs.
    #   Both the projector and all experts then see the downsampled signal.
    # front_end_stride == 0 → no front-end layer is created (default; all other
    #   ablations leave this at 0).
    # Only A12 sets this to 20 to handle 2kHz data on a 32GB GPU.
    config["front_end_stride"]     = 0

    # ── Best hyperparameters (Trial 89) ───────────────────────────────────────
    # CHANGED: learning_rate 3e-4 → 1.9506e-4 (Trial 89 Optuna value)
    config["learning_rate"]  = 1.9506115991520216e-4   # was 3e-4
    # CHANGED: weight_decay 5.6e-4 → 8.874e-4 (Trial 89 Optuna value)
    config["weight_decay"]   = 8.873572502558012e-4    # was 5.6e-4
    # CHANGED: label_smooth 0.15 → 0.05 (Trial 89 Optuna value)
    config["label_smooth"]   = 0.05                    # was 0.15
    config["gradient_clip_max_norm"] = 10.0

    # ── Training schedule ─────────────────────────────────────────────────────
    config["num_epochs"]              = 23
    # CHANGED: episodes_per_epoch_train 200 → 500 (Trial 89 Optuna value)
    config["episodes_per_epoch_train"] = 500   # was 200
    config["num_eval_episodes"]        = NUM_VAL_EPISODES
    config["batch_size"]               = 64    # flat dataloader default
    config["num_workers"]              = 8
    config["optimizer"]                = "adam"
    config["use_cosine_outer_lr"]      = False
    config["lr_scheduler_factor"]      = 0.1
    config["lr_scheduler_patience"]    = 6
    config["use_earlystopping"]        = False  # Turned off for testing eval!
    config["earlystopping_patience"]   = 8
    config["earlystopping_min_delta"]  = 0.005

    # ── MAML core (Trial 89 values) ───────────────────────────────────────────
    config["meta_learning"]           = True
    config["meta_batchsize"]          = 24    # FIXED per spec
    # CHANGED: maml_inner_steps 5 → 10 (Trial 89 Optuna value)
    config["maml_inner_steps"]        = 10    # was 5
    # CHANGED: maml_inner_steps_eval 50 → 10 (Trial 89 value; HPO did NOT use
    # a separate eval step count — it used the same value as train)
    config["maml_inner_steps_eval"]   = 10    # was 50
    # CHANGED: maml_alpha_init 1.7e-3 → 9.735e-4 (Trial 89 Optuna value)
    config["maml_alpha_init"]         = 9.734890497675034e-4   # was 1.7e-3
    # CHANGED: maml_alpha_init_eval 0.017 → 5.066e-3 (Trial 89 Optuna value)
    config["maml_alpha_init_eval"]    = 5.06597432775958e-3    # was 0.017
    config["maml_use_lslr"]           = True   # FIXED per spec; unchanged
    config["use_lslr_at_eval"]        = False  # FIXED per spec; unchanged
    # CHANGED: use_maml_msl False → "hybrid" (Trial 89 Optuna value)
    config["use_maml_msl"]            = "hybrid"  # was False
    # CHANGED: maml_msl_num_epochs 0 → 31 (Trial 89 Optuna value)
    config["maml_msl_num_epochs"]     = 31         # was 0
    config["maml_opt_order"]          = "first"
    config["maml_first_order_to_second_order_epoch"] = 1_000_000
    config["enable_inner_loop_optimizable_bn_params"] = False

    # ── MoE (Trial 89 values) ─────────────────────────────────────────────────
    config["use_MOE"]                         = True
    config["MOE_placement"]                   = "encoder"   # FIXED per spec
    # CHANGED: num_experts 32 → 22 (Trial 89 Optuna value)
    config["num_experts"]                     = 22          # was 32
    config["MOE_top_k"]                       = 9           # unchanged
    config["top_k"]                           = config["MOE_top_k"]
    # CHANGED: MOE_gate_temperature 0.65 → 1.5290 (Trial 89 Optuna value)
    config["MOE_gate_temperature"]            = 1.5290172211651742   # was 0.65
    # CHANGED: MOE_aux_coeff 0.023 → 0.03282 (Trial 89 Optuna value)
    config["MOE_aux_coeff"]                   = 0.03282324399711515  # was 0.023
    # CHANGED: MOE_ctx_out_dim 32 → 64 (Trial 89 Optuna value)
    config["MOE_ctx_out_dim"]                 = 64    # was 32
    config["MOE_ctx_hidden_dim"]              = 32    # unchanged
    # CHANGED: MOE_dropout 0.05 → 0.03654 (Trial 89 Optuna value)
    config["MOE_dropout"]                     = 0.03653577545411608  # was 0.05
    config["MOE_expert_expand"]               = 1.0
    config["MOE_mlp_hidden_mult"]             = 1.0
    config["MOE_log_every"]                   = 5
    config["MOE_plot_dir"]                    = None
    # CHANGED: apply_MOE_aux_loss_inner_outer "inner" → "outer" (Trial 89 Optuna value)
    config["apply_MOE_aux_loss_inner_outer"]  = "outer"   # was "inner"
    # Legacy keys kept for compatibility
    config["gate_type"]              = "context_feature_demo"
    config["expert_architecture"]    = "MLP"

    # Added but not used (these need to be fully HPOd/debugged...)
    config["MOE_use_shared_expert"] = False
    config["MOE_importance_coeff"] = 0.0
    config["MOE_routing_signal"] = 'context_proj'
    config["utilization_ratio"] = 0.3  # I think this is not used at all? This is what sets top-k in the other runs

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

    The function inspects `model.head` and constructs a matching fresh head:
      - nn.Linear        → fresh nn.Linear(in_features, n_way)
      - nn.Sequential    → preserve all layers except the terminal Linear,
                           which is replaced with nn.Linear(in_features, n_way)
      - MLPHead          → fresh MLPHead(feat_dim, hidden_dim, n_way, dropout)
                           The entire head is re-initialised (not just the terminal
                           layer) so that no pretrained head weights survive into
                           the few-shot task — consistent with the linear and
                           Sequential protocols above.

    Args:
        model  : pretrained model whose .head attribute will be replaced.
        config : must contain 'n_way' (int) — number of eval classes.

    Returns:
        model with replaced head (same object, modified in-place AND returned).
    """
    import torch.nn as nn
    from pretraining.pretrain_models import MLPHead

    n_way = int(config["n_way"])
    head  = model.head

    if isinstance(head, nn.Linear):
        in_features  = head.in_features
        model.head   = nn.Linear(in_features, n_way)
        nn.init.xavier_uniform_(model.head.weight)
        nn.init.zeros_(model.head.bias)

    elif isinstance(head, MLPHead):
        # Reconstruct the same hidden_dim and dropout from the existing head so
        # the architecture is identical to pretraining, just with n_way outputs.
        # MLPHead.net is: Linear(feat_dim → hidden_dim), GELU, Dropout, Linear(hidden_dim → n_classes)
        net_layers  = list(head.net.children())
        assert isinstance(net_layers[0], nn.Linear), (
            f"replace_head_for_eval: expected MLPHead.net[0] to be nn.Linear, "
            f"got {type(net_layers[0])}."
        )
        assert isinstance(net_layers[2], nn.Dropout), (
            f"replace_head_for_eval: expected MLPHead.net[2] to be nn.Dropout, "
            f"got {type(net_layers[2])}."
        )
        feat_dim   = net_layers[0].in_features
        hidden_dim = net_layers[0].out_features
        dropout    = net_layers[2].p
        model.head = MLPHead(feat_dim, hidden_dim, n_way, dropout)
        # MLPHead uses default PyTorch init (Kaiming uniform for Linear via nn.Linear),
        # which is fine; no need to re-init manually.

    elif isinstance(head, nn.Sequential):
        # Find and replace only the terminal Linear layer
        layers = list(head.children())
        assert isinstance(layers[-1], nn.Linear), (
            f"replace_head_for_eval: expected the last layer of model.head (Sequential) "
            f"to be nn.Linear, got {type(layers[-1])}. Add explicit support for this head type."
        )
        in_features  = layers[-1].in_features
        layers[-1]   = nn.Linear(in_features, n_way)
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        model.head   = nn.Sequential(*layers)

    else:
        raise TypeError(
            f"replace_head_for_eval: model.head is {type(head)}, expected nn.Linear, "
            f"MLPHead, or nn.Sequential. Add explicit support for this head type."
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

    print(f"  [Test] Per-user results ({len(per_user)} users):")
    for uid, acc in sorted(per_user.items()):
        print(f"    user={uid}  acc={acc*100:.2f}%")
    print(f"  [Test] Mean: {float(np.mean(vals))*100:.2f}%  Std: {float(np.std(vals))*100:.2f}%")

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
    from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate, reorient_tensor_dict
    from pretraining.pretrain_finetune import finetune_and_eval_user
    import pickle
    from torch.utils.data import DataLoader
    from collections import defaultdict

    with open(tensor_dict_path, "rb") as f:
        full_dict   = pickle.load(f)
    # Reorient tensors from disk layout (trials, T, C) → (trials, C, T) so that
    # MetaGestureDataset slices yield (C, T) samples, which maml_mm_collate stacks
    # into (B, C, T) — the channel-first layout expected by all Conv1d / LSTM models.
    # run_episodic_test_eval already does this; omitting it here caused the
    # "Expected size 16 but got size 72" RuntimeError in the MOE_encoder cat.
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


# =============================================================================
# Periodic training callbacks (for mamlpp_pretrain hooks)
# =============================================================================

def make_periodic_checkpoint_fn(config: dict):
    """
    Returns a function compatible with mamlpp_pretrain's `periodic_checkpoint_fn` kwarg.
    Signature: fn(model, config, epoch, val_acc, tag) -> None
    Saves a checkpoint to disk immediately (crash insurance).
    """
    def _checkpoint(model, cfg, epoch: int, val_acc: float, tag: str):
        state = {
            "epoch":            epoch,
            "val_acc":          val_acc,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "config":           cfg,
        }
        save_model_checkpoint(state, cfg, tag=tag)
    return _checkpoint


def make_periodic_test_eval_fn(tensor_dict_path: str, test_pids: list):
    """
    Returns a function compatible with mamlpp_pretrain's `periodic_test_eval_fn` kwarg.
    Signature: fn(model, config, epoch) -> None
    Runs run_episodic_test_eval (which prints per-user results internally) on
    the current model weights (NOT the best_state snapshot — this shows where
    the live model is at epoch N, not where the best checkpoint is).

    NOTE: If val_PIDs == test_PIDs (Kapanji split), results here reflect the same
    users driving early stopping — treat these mid-training numbers with caution.
    """
    def _test_eval(model, config, epoch: int):
        print(f"  [PeriodicTestEval] Epoch {epoch} — evaluating on test PIDs: {test_pids}")
        results = run_episodic_test_eval(model, config, tensor_dict_path, test_pids)
        # run_episodic_test_eval already prints per-user + aggregate; nothing extra needed.
        return results
    return _test_eval