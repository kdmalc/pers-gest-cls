"""
maml_train.py
=============
Standalone MAML++ training script — no Optuna, just runs.

Usage
-----
Local (CPU, quick smoke-test):
    python maml_train.py --model_type DeepCNNLSTM

Cluster (GPU, full run):
    python maml_train.py --model_type DeepCNNLSTM --fold 0

Arguments
---------
  --model_type   MetaCNNLSTM | DeepCNNLSTM | TST | ContrastiveNet  (default: DeepCNNLSTM)
  --fold         0 | 1  — which CV fold to run (default: 0)
  --use_MOE      flag — activate Mixture-of-Experts (default: off)
  --MOE_placement  encoder | middle  (default: middle)
  --smoke        flag — tiny config for local debugging (2 epochs, few episodes)
  --no_pretrained  flag — skip loading pretrained weights (random init)

Environment variables (same as the cluster script)
---------------------------------------------------
  CODE_DIR   root of the repo      (default: ./)
  DATA_DIR   data root             (default: ./data)
  RUN_DIR    where outputs go      (default: ./)
"""

import os
import argparse
import copy
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Deterministic but not benchmark-mode (faster on GPU for non-fixed input sizes)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress the weights_only=False warning from torch.load
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paths (from env, same convention as the SLURM script)
# ─────────────────────────────────────────────────────────────────────────────
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR  = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR   = Path(os.environ.get("RUN_DIR",  "./")).resolve()

# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
# This finds the 'system' directory (one level up from 'pretraining') and adds it to the search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MAML.mamlpp import mamlpp_pretrain, mamlpp_adapt_and_eval
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.shared_maml import meta_evaluate
from MAML.MOE_CNN_LSTM import MultimodalCNNLSTMMOE
from pretraining.pretrain_models import build_model as build_baseline_model
from pretraining.contrastive_net.contrastive_encoder import ContrastiveGestureEncoder

from MOE.MOE_encoder import build_MOE_model, load_pretrained_into_MOE


# ─────────────────────────────────────────────────────────────────────────────
# Fixed config
# ─────────────────────────────────────────────────────────────────────────────
# All the values that were previously spread across trial.suggest_* calls are
# collected here with reasonable defaults.
# The only thing you cannot change here without also changing the pretrained
# checkpoint filenames is the architecture block inside inject_model_config().

FIXED_CONFIG = {
    # ── Task ────────────────────────────────────────────────────────────────
    "n_way":           3,       # NOTE: 3-way during meta-train; eval on all 10
    "k_shot":          1,
    "q_query":         9,
    "num_classes":     10,      # total gesture classes in the dataset
    "feature_engr":    "None",

    # ── Data / Modality ──────────────────────────────────────────────────────
    "multimodal":      True,
    "use_imu":         True,
    "use_demographics": False,
    "use_film_x_demo": False,
    "FILM_on_context_or_demo": "context",  # TODO: There's no option to disable this? Is this even happening right now...
    "context_emb_dim": 32,
    "context_pool_type": "mean",

    # ── MAML++ Core ──────────────────────────────────────────────────────────
    "maml_inner_steps":      5,     # inner-loop gradient steps during training
    "maml_inner_steps_eval": 20,    # inner-loop steps at test-time adaptation
    "maml_alpha_init":       0.01,  # inner-loop step size init
    "maml_alpha_init_eval":  0.01,
    "meta_batchsize":        8,     # episodes per outer-loop step

    # ── MAML++ Options ───────────────────────────────────────────────────────
    "use_maml_msl":                         "hybrid",   # True | False | "hybrid"
    "maml_msl_num_epochs":                  10,         # only used when msl=="hybrid"
    "maml_opt_order":                       "first",    # "first" | "second" | "hybrid"
    "maml_first_order_to_second_order_epoch": 1_000_000, # irrelevant when opt_order=="first"
    "maml_use_lslr":                        True,       # learned per-param step sizes
    "use_lslr_at_eval":                     False,
    "enable_inner_loop_optimizable_bn_params": False,
    "track_gradient_alignment":             False,

    # ── Outer-loop Optimizer ─────────────────────────────────────────────────
    "optimizer":        "adam",
    "learning_rate":    5e-4,   # outer LR
    "weight_decay":     1e-4,

    # ── Training Loop ────────────────────────────────────────────────────────
    "num_epochs":               30,
    "episodes_per_epoch_train": 200,
    "use_earlystopping":        True,
    "earlystopping_patience":   8,
    "earlystopping_min_delta":  0.005,
     
    "label_smooth":             0.1,
    "use_label_shuf_meta_aug":  True,
    "gradient_clip_max_norm":   10.0,
    "num_eval_episodes":        10,

    # NOTE: lr_scheduler is not used if use_cosine_outer_lr is False!
    "use_cosine_outer_lr":      False,
    "lr_scheduler_factor":      0.1,
    "lr_scheduler_patience":    6,

    # ── Misc ─────────────────────────────────────────────────────────────────
    "use_batch_norm":   False,
    "dropout":          0.1,
    "emg_stride":       1,
    "imu_stride":       1,
    "padding":          0,
    "use_GlobalAvgPooling": True,
    "num_workers":      4,       # reduce to 0 for local debugging if you get errors
    "verbose":          False,
    "num_total_users":  32,

    "maml_gesture_classes":  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "target_trial_indices":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # ── Pretrained weights ───────────────────────────────────────────────────
    # pretrain_approach controls BOTH whether/how to load pretrained weights
    # AND whether to freeze any part of the network during MAML training.
    # Valid values:
    #   "None"            - random init, no pretrained weights
    #   "full_best"       - load full pretrained model (best checkpoint)
    #   "full_last"       - load full pretrained model (last checkpoint)
    #   "enc_best"        - load encoder only (CNN+LSTM, drop MLP head) from best ckpt
    #   "enc_last"        - load encoder only from last checkpoint
    #   "frozen_enc_best" - load + freeze encoder from best checkpoint  [NOT IMPLEMENTED]
    #   "frozen_enc_last" - load + freeze encoder from last checkpoint  [NOT IMPLEMENTED]
    "pretrain_approach": "None",  # To NOT use pretraining, this needs to be "None"!

    # pretrained_model_filename: stem of the checkpoint filename (without _best.pt / _last.pt).
    # Set to None to use the hardcoded per-model-type defaults in get_pretrain_path().
    # Set to a string like "MetaCNNLSTM_03232026_170503" to override explicitly.
    # Useful when you want to load a non-MOE checkpoint into a MOE model (or vice-versa).
    "pretrained_model_filename": None,

    # ── Debug flags (set via --smoke to override these safely) ───────────────
    "debug_one_user_only":  False,
    "debug_one_episode":    False,
    "debug_five_episodes":  False,

    # ── MOE defaults (only active when --use_MOE is passed) ──────────────────
    "use_MOE":              True,      # overridden by --use_MOE flag
    "MOE_placement":        "encoder",   # "middle" | "encoder"
    "num_experts":          4,
    "MOE_ctx_hidden_dim":   64,
    "MOE_ctx_out_dim":      32,
    "MOE_gate_temperature": 1.0,
    "MOE_top_k":            None,       # None = dense/soft routing (MAML-safe)
    "MOE_expert_expand":    0.75,       # encoder-MOE expert CNN width fraction
    "MOE_mlp_hidden_mult":  1.0,        # middle-MOE expert MLP width multiplier
    "MOE_dropout":          0.1,
    "MOE_aux_coeff":        1e-2,
    "MOE_log_every":        5,          # routing analysis every N epochs (0=never)
    "MOE_plot_dir":         None,       # set to a path string to save heatmaps
}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture injection  (unchanged from original — must match pretrain configs)
# ─────────────────────────────────────────────────────────────────────────────
def inject_model_config(config: dict, model_type: str) -> dict:
    """
    Injects the exact architecture params used during pretraining.
    These MUST match pretrain_configs.py or the weight load will silently fail.
    """
    config["model_type"]      = model_type
    config["sequence_length"] = 64
    config["emg_in_ch"]       = 16
    config["imu_in_ch"]       = 72
    config["demo_in_dim"]     = 12
    config["groupnorm_num_groups"] = 8

    if model_type == "MetaCNNLSTM":
        config.update({
            "cnn_filters":   32, "cnn_kernel": 5, "gn_groups": 8,
            "lstm_hidden":   32, "lstm_layers": 1, "bidirectional": False,
            "head_type":     "linear",
        })
    elif model_type == "DeepCNNLSTM":
        config.update({
            "cnn_base_filters": 32, "cnn_layers": 3,
            "cnn_kernel": 5, "gn_groups": 8,
            "lstm_hidden": 64, "lstm_layers": 3, "bidirectional": True,
            "head_type":   "mlp",
        })
    elif model_type == "TST":
        config.update({
            "patch_len": 8, "d_model": 64, "n_heads": 4, "n_blocks": 3,
        })
    elif model_type == "ContrastiveNet":
        config.update({
            "arch_mode":            "cnn_attn",
            "train_reps":           list(range(1, 11)),
            "val_reps":             list(range(1, 11)),
            "emg_base_cnn_filters": 64, "emg_cnn_layers": 3,
            "imu_base_cnn_filters": 32, "imu_cnn_layers": 2,
            "cnn_kernel_size":      5,
            "emg_stride":           1,  "imu_stride": 1,
            "groupnorm_num_groups": 8,
            "use_lstm":             False,
            "lstm_hidden":          128, "lstm_layers": 2,
            "use_GlobalAvgPooling": True,
            "attn_pool_heads":      4,
            "embedding_dim":        128, "proj_hidden_dim": 256,
            "num_val_episodes":     20,
            "lr_scheduler":         "cosine",
            "lr_warmup_epochs":     5,
            "lr_min":               1e-6,
            "grad_clip":            5.0,
            "log_interval":         100,
        })
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         "Choose MetaCNNLSTM | DeepCNNLSTM | TST | ContrastiveNet.")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained checkpoint paths  (update these when you retrain pretraining)
# ─────────────────────────────────────────────────────────────────────────────
def get_pretrain_path(config
    #model_type: str,
    #best_or_last: str,
    #pretrain_dir: Path,
    #model_filename: str | None = None,
) -> str | None:
    """
    Returns the full path to a pretrained checkpoint.

    Args:
        model_type:   Architecture key (e.g. "MetaCNNLSTM").
        best_or_last: "best" or "last" — which checkpoint variant to load.
        pretrain_dir: Directory containing checkpoint files.
        model_filename:     Optional stem override (e.g. "MetaCNNLSTM_03232026_170503").
                      If provided, bypasses the per-model-type default stem and
                      constructs the path as ``{model_filename}_{best_or_last}.pt``.
                      Pass None to use the hardcoded defaults below.

    Returns:
        Absolute path string, or None if no checkpoint exists for this model.
    """

    model_type = config['model_type']
    # Determine best_or_last from the approach string suffix
    
    if "best" in config["pretrain_approach"]:
        best_or_last = "_best" 
    elif "last" in config["pretrain_approach"]:
        best_or_last = "_last" 
    else:
        best_or_last = ""
    best_or_last = config['??']
    pretrain_dir = config['pretrain_dir']
    model_filename = config['pretrain_model_filename']

    if model_filename is not None:
        # Explicit override — caller knows exactly which checkpoint they want.
        if model_filename[-2:] == "pt":
            fname = model_filename
        else:
            fname = f"{model_filename}.pt"  #_{best_or_last}
        return str(pretrain_dir / fname)
    else:
        if config["use_MOE"]:
            placement_str_abv = config["MOE_placement"][:4]  # Take the first 3 chars
            MOE_str = "MOE"+placement_str_abv+"_"
            if config["MOE_placement"].lower() == "encoder" or config["MOE_placement"].lower() == "enc":
                time_ID = "03272026_124829"
            elif config["MOE_placement"].lower() == "middle" or config["MOE_placement"].lower() == "mid":
                time_ID = "03262026_213056"
            else:
                raise ValueError("Unknown MOE_placement")
            # ── Default stems per model type ─────────────────────────────────────────
            # Update these strings whenever you retrain a new pretrained baseline.
            # NOTE: These time IDs are wrong...
            default_stems = {
            "MetaCNNLSTM":    None,  #f"MetaCNNLSTM_{MOE_str}03232026_170503",
            "DeepCNNLSTM":    f"DeepCNNLSTM_{MOE_str}{time_ID}",
            "TST":            None,  #f"TST_{MOE_str}03232026_163527",
            "ContrastiveNet": None,  #f"ContrastiveNet_attn_{MOE_str}20260325_1810",
            "MOE":            None,  # no pretrained MOE-CNN-LSTM checkpoint exists
        }
        else:
            MOE_str = ""
            # ── Default stems per model type ─────────────────────────────────────────
            # Update these strings whenever you retrain a new pretrained baseline.
            default_stems = {
                "MetaCNNLSTM":    "MetaCNNLSTM_03232026_170503",
                "DeepCNNLSTM":    "DeepCNNLSTM_03232026_165043",
                "TST":            "TST_03232026_163527",
                "ContrastiveNet": "ContrastiveNet_attn_20260325_1810",
                "MOE":            None,  # no pretrained MOE-CNN-LSTM checkpoint exists
            }
        stem = default_stems.get(model_type)
        if stem is None:
            raise ValueError("stem is None")
            return None
        return str(pretrain_dir / f"{stem}{best_or_last}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Model build + pretrained weight loading
# ─────────────────────────────────────────────────────────────────────────────
def build_model(config: dict) -> nn.Module:
    """
    Instantiates the correct model variant (baseline or MOE) and optionally
    loads pretrained backbone weights according to ``config["pretrain_approach"]``.

    pretrain_approach semantics
    ---------------------------
    "None"            → random init (no weights loaded)
    "full_best"       → load all non-head weights from the *best* checkpoint
    "full_last"       → load all non-head weights from the *last* checkpoint
    "enc_best"        → load encoder (CNN + LSTM, no MLP head) from best ckpt
    "enc_last"        → load encoder from last checkpoint
    "frozen_enc_best" → enc_best + freeze the encoder during MAML training
    "frozen_enc_last" → enc_last + freeze the encoder during MAML training

    The checkpoint stem can be overridden via ``config["pretrained_model_filename"]``
    (a string like ``"MetaCNNLSTM_03232026_170503"``); leave as None to use
    the hardcoded per-model-type defaults in get_pretrain_path().
    """
    model_type       = config["model_type"]
    use_MOE          = config.get("use_MOE", False)
    pretrain_approach = config.get("pretrain_approach", "full_best")

    # ── Instantiate ──────────────────────────────────────────────────────────
    if use_MOE and model_type in ("MetaCNNLSTM", "DeepCNNLSTM", "ContrastiveNet"):
        print(f"[build_model] Using MOE variant — placement={config['MOE_placement']}  E={config['num_experts']}")
        model = build_MOE_model(config)
    elif model_type in ("MetaCNNLSTM", "DeepCNNLSTM", "TST"):
        model = build_baseline_model(config)
    elif model_type == "ContrastiveNet":
        model = ContrastiveGestureEncoder(config)
    elif model_type == "MOE":
        model = MultimodalCNNLSTMMOE(config)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    # ── Pretrained weight loading ─────────────────────────────────────────────
    if pretrain_approach == "None":
        print("[build_model] pretrain_approach='None' — using random initialisation.")
    elif pretrain_approach in ("full_best", "full_last",
                               "enc_best",  "enc_last",
                               "frozen_enc_best", "frozen_enc_last"):

        if pretrain_approach.startswith("frozen_enc"):
            raise NotImplementedError(
                f"pretrain_approach='{pretrain_approach}' is not yet implemented. "
                "Freezing the encoder requires plumbing changes in named_param_dict() "
                "so that frozen parameters are excluded from the inner-loop update. "
                "Use 'enc_best' or 'enc_last' for encoder-only loading without freezing."
            )

        # Determine loading scope: full backbone vs encoder only
        enc_only = pretrain_approach.startswith("enc")

        # This should be pulled from the config...
        pretrain_dir = Path(r"/projects/my13/kai/meta-pers-gest/pers-gest-cls/pretrain_outputs/checkpoints/")
        load_path    = get_pretrain_path(config)  #model_type, best_or_last, pretrain_dir, model_filename=model_filename

        if load_path is None:
            print("[build_model] No pretrained checkpoint for this model_type — random init.")
        else:
            scope_label = "encoder (CNN+LSTM, no MLP head)" if enc_only else "full backbone"
            print(f"[build_model] Loading pretrained weights ({scope_label}) from:\n  {load_path}")
            try:
                checkpoint = torch.load(load_path, map_location=config["device"], weights_only=False)
                state_dict = (checkpoint.get("model_state")
                              or checkpoint.get("model_state_dict")
                              or checkpoint)

                if use_MOE and model_type in ("MetaCNNLSTM", "DeepCNNLSTM", "ContrastiveNet"):
                    # MOE-aware transfer: keeps CNN weights, seeds expert CNNs.
                    # enc_only is not separately handled here because load_pretrained_into_MOE
                    # already transfers only the encoder portions by design.
                    model = load_pretrained_into_MOE(
                        MOE_model             = model,
                        pretrained_state_dict = state_dict,
                        placement             = config["MOE_placement"],
                        seed_experts          = True,
                        verbose               = True,
                    )
                else:
                    # Standard transfer.
                    # Always strip the classification/projection head regardless of scope,
                    # since the MAML head is freshly initialised for N-way classification.
                    # For enc_only we additionally strip the MLP body of the backbone.
                    def _keep_key(k: str, enc_only: bool) -> bool:
                        if "head" in k or "projector" in k:
                            return False  # always drop
                        if enc_only and ("mlp" in k or "classifier" in k or "fc" in k):
                            return False  # drop MLP body when enc_only
                        return True

                    filtered = {k: v for k, v in state_dict.items()
                                if _keep_key(k, enc_only)}
                    mdict = model.state_dict()
                    mdict.update(filtered)
                    model.load_state_dict(mdict)
                    print(f"[build_model] Loaded {len(filtered)}/{len(state_dict)} weight tensors ({scope_label}).")

            except FileNotFoundError:
                print(f"\n{'#'*60}")
                print(f"WARNING: Checkpoint not found → {load_path}")
                print(f"Continuing with RANDOM INITIALISATION.")
                print(f"{'#'*60}\n")
    else:
        raise ValueError(
            f"Unknown pretrain_approach='{pretrain_approach}'. "
            "Must be one of: 'None', 'full_best', 'full_last', "
            "'enc_best', 'enc_last', 'frozen_enc_best', 'frozen_enc_last'."
        )

    model.to(config["device"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-user final evaluation  (same as what was inside objective())
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_per_user(model, best_state: dict, episodic_val_loader, config: dict) -> dict:
    """
    Loads the best checkpoint, then runs mamlpp_adapt_and_eval for every
    episode in the val loader, grouping results by user_id.

    Returns a dict with per-user accuracy lists and a summary.
    """
    model.load_state_dict(best_state)
    user_metrics = defaultdict(list)

    for batch in episodic_val_loader:
        user_id     = batch["user_id"]
        support_set = batch["support"]
        query_set   = batch["query"]
        val_metrics = mamlpp_adapt_and_eval(model, config, support_set, query_set)
        user_metrics[user_id].append(val_metrics["acc"])

    per_user_means = {}
    all_means = []
    for uid, accs in user_metrics.items():
        m = float(np.mean(accs))
        per_user_means[uid] = m
        all_means.append(m)
        print(f"  User {uid} | {m*100:.2f}%  ({len(accs)} episodes)")

    mean_acc = float(np.mean(all_means))
    std_acc  = float(np.std(all_means))
    print(f"\n  → Mean: {mean_acc*100:.2f}%  ±  {std_acc*100:.2f}%  ({len(all_means)} users)")
    return {
        "per_user_means": per_user_means,
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# One complete fold
# ─────────────────────────────────────────────────────────────────────────────
def run_fold(model_type: str, fold_idx: int, config: dict,
             all_splits: list, timestamp: str) -> dict:
    """
    Builds model, loads data, runs mamlpp_pretrain, evaluates per user.
    Returns a summary dict for this fold.
    """
    fold_start = time.time()
    print("\n" + "=" * 70)
    print(f"  FOLD {fold_idx}  |  model={model_type}")
    print("=" * 70)

    # Apply fold split
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]

    # Build model fresh for each fold
    model = build_model(config)

    # Data
    tensor_dict_path = os.path.join(
        config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
    )
    episodic_train_loader, episodic_val_loader = get_maml_dataloaders(
        config, tensor_dict_path=tensor_dict_path,
    )

    # MAML++ meta-training
    trained_model, pretrain_res = mamlpp_pretrain(
        model,
        config,
        episodic_train_loader,
        episodic_val_loader=episodic_val_loader,
    )
    best_val_acc = pretrain_res["best_val_acc"]
    best_state   = pretrain_res["best_state"]
    print(f"\n[Fold {fold_idx}] Meta-training done. Best val acc = {best_val_acc*100:.2f}%")

    # Save checkpoint
    save_path = config["models_save_dir"] / f"{model_type}_fold{fold_idx}_{timestamp}_best.pt"
    torch.save({
        "fold_idx":        fold_idx,
        "model_type":      model_type,
        "model_state_dict": best_state,
        "config":          config,
        "best_val_acc":    best_val_acc,
        "train_loss_log":  pretrain_res["train_loss_log"],
        "train_acc_log":   pretrain_res["train_acc_log"],
        "val_loss_log":    pretrain_res["val_loss_log"],
        "val_acc_log":     pretrain_res["val_acc_log"],
        "routing_reports": pretrain_res.get("routing_reports", []),
    }, save_path)
    print(f"[Fold {fold_idx}] Checkpoint saved → {save_path}")

    # Per-user evaluation
    print(f"\n[Fold {fold_idx}] Per-user evaluation:")
    eval_results = evaluate_per_user(trained_model, best_state, episodic_val_loader, config)

    duration = time.time() - fold_start
    print(f"\n[Fold {fold_idx}] Completed in {duration/60:.1f} min")

    return {
        "fold_idx":       fold_idx,
        "best_val_acc":   best_val_acc,
        "mean_user_acc":  eval_results["mean_acc"],
        "std_user_acc":   eval_results["std_acc"],
        "per_user_means": eval_results["per_user_means"],
        "train_loss_log": pretrain_res["train_loss_log"],
        "val_loss_log":   pretrain_res["val_loss_log"],
        "val_acc_log":    pretrain_res["val_acc_log"],
        "routing_reports": pretrain_res.get("routing_reports", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone MAML++ training (no Optuna).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_type", default="DeepCNNLSTM",
                   choices=["MetaCNNLSTM", "DeepCNNLSTM", "TST", "ContrastiveNet"],
                   help="Model architecture to train.")
    p.add_argument("--fold", type=int, default=0, choices=[0, 1],
                   help="Which CV fold to run (0 or 1).")
    p.add_argument("--all_folds", action="store_true",
                   help="Run both folds sequentially and report the mean.")
    p.add_argument("--use_MOE", action="store_true",
                   help="Use Mixture-of-Experts variant of the chosen model.")
    p.add_argument("--MOE_placement", default="middle", choices=["middle", "encoder"],
                   help="Where to insert MOE (middle = between CNN and LSTM).")
    p.add_argument("--smoke", action="store_true",
                   help="Quick smoke-test config: 2 epochs, 10 episodes, CPU-safe.")
    p.add_argument("--pretrain_approach", default=None,
                   choices=["None", "full_best", "full_last",
                            "enc_best", "enc_last",
                            "frozen_enc_best", "frozen_enc_last"],
                   help=(
                       "How to initialise from a pretrained checkpoint. "
                       "Overrides FIXED_CONFIG['pretrain_approach']. "
                       "'None' means random init. "
                       "'full_*' loads the entire backbone; "
                       "'enc_*' loads only CNN+LSTM (drops MLP head). "
                       "'frozen_enc_*' additionally freezes the encoder (not yet implemented)."
                   ))
    p.add_argument("--pretrained_model_filename", default=None, type=str,
                   help=(
                       "Stem of the checkpoint file to load, e.g. "
                       "'MetaCNNLSTM_03232026_170503'. "
                       "Leave unset to use the default checkpoint for the chosen model_type. "
                       "Useful when you want to load a non-MOE checkpoint into a MOE model."
                   ))
    p.add_argument("--no_pretrained", action="store_true",
                   help="Shorthand for --pretrain_approach None (random init).")
    p.add_argument("--num_epochs", type=int, default=None,
                   help="Override num_epochs from FIXED_CONFIG.")
    p.add_argument("--episodes", type=int, default=None,
                   help="Override episodes_per_epoch_train from FIXED_CONFIG.")
    p.add_argument("--n_way", type=int, default=None,
                   help="Override n_way (number of classes per episode).")
    p.add_argument("--inner_steps", type=int, default=None,
                   help="Override maml_inner_steps.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Environment / seed ───────────────────────────────────────────────────
    SEED = 42
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"CODE_DIR: {CODE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"RUN_DIR:  {RUN_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Output dirs ──────────────────────────────────────────────────────────
    results_save_dir = RUN_DIR
    models_save_dir  = RUN_DIR
    results_save_dir.mkdir(parents=True, exist_ok=True)
    models_save_dir.mkdir(parents=True, exist_ok=True)

    # ── Build config ─────────────────────────────────────────────────────────
    config = copy.deepcopy(FIXED_CONFIG)

    # Inject architecture constants (must match pretraining)
    config = inject_model_config(config, args.model_type)

    # NOTS = running on the cluster (Not On Their System)
    config["NOTS"] = True
    config["user_split_json_filepath"] = str(CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json")
    config["results_save_dir"] = results_save_dir
    config["models_save_dir"]  = models_save_dir
    config["emg_imu_pkl_full_path"]  = str(CODE_DIR / "dataset" / "filtered_datasets" / "metadata_IMU_EMG_allgestures_allusers.pkl")
    config["pwmd_xlsx_filepath"]     = str(CODE_DIR / "dataset" / "Biosignal gesture questionnaire for participants with disabilities.xlsx")
    config["pwoutmd_xlsx_filepath"]  = str(CODE_DIR / "dataset" / "Biosignal gesture questionnaire for participants without disabilities.xlsx")
    config["dfs_save_path"]          = str(CODE_DIR / "dataset" / "")
    config["dfs_load_path"]          = str(CODE_DIR / "dataset" / "meta-learning-sup-que-ds" / "")
    config["meta_learning"]          = True
    config["device"]                 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── CLI overrides ─────────────────────────────────────────────────────────
    if args.use_MOE:
        config["use_MOE"]        = True
        config["MOE_placement"]  = args.MOE_placement

    # pretrain_approach: --no_pretrained is a quick alias; explicit flag takes precedence
    if args.no_pretrained:
        config["pretrain_approach"] = "None"
    if args.pretrain_approach is not None:
        config["pretrain_approach"] = args.pretrain_approach
    if args.pretrained_model_filename is not None:
        config["pretrained_model_filename"] = args.pretrained_model_filename

    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs

    if args.episodes is not None:
        config["episodes_per_epoch_train"] = args.episodes

    if args.n_way is not None:
        config["n_way"] = args.n_way

    if args.inner_steps is not None:
        config["maml_inner_steps"] = args.inner_steps

    # ── Smoke-test mode: tiny everything so you can run locally in <2 min ────
    if args.smoke:
        print("\n" + "!" * 60)
        print("  SMOKE MODE — tiny config, CPU, 2 epochs, 10 episodes")
        print("!" * 60 + "\n")
        config.update({
            "num_epochs":               2,
            "episodes_per_epoch_train": 10,
            "maml_inner_steps":         2,
            "maml_inner_steps_eval":    5,
            "meta_batchsize":           2,
            "num_workers":              0,     # avoid multiprocessing issues locally
            "use_earlystopping":        False,
            "maml_use_lslr":            False, # faster without LSLR
            "use_maml_msl":             False,
            "maml_msl_num_epochs":      0,
            "MOE_log_every":            1,     # show routing every epoch in smoke mode
        })

    # ── Print final config summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  MAML++ TRAINING CONFIG")
    print("=" * 70)
    print(f"  model_type:           {args.model_type}")
    print(f"  use_MOE:              {config['use_MOE']}")
    if config["use_MOE"]:
        print(f"  MOE_placement:        {config['MOE_placement']}")
        print(f"  num_experts:          {config['num_experts']}")
        print(f"  MOE_aux_coeff:        {config['MOE_aux_coeff']}")
    print(f"  device:               {config['device']}")
    print(f"  n_way:                {config['n_way']}")
    print(f"  k_shot:               {config['k_shot']}")
    print(f"  maml_inner_steps:     {config['maml_inner_steps']}")
    print(f"  maml_inner_steps_eval:{config['maml_inner_steps_eval']}")
    print(f"  meta_batchsize:       {config['meta_batchsize']}")
    print(f"  episodes/epoch:       {config['episodes_per_epoch_train']}")
    print(f"  num_epochs:           {config['num_epochs']}")
    print(f"  outer_lr:             {config['learning_rate']}")
    print(f"  maml_use_lslr:        {config['maml_use_lslr']}")
    print(f"  use_maml_msl:         {config['use_maml_msl']}")
    print(f"  use_pretrained:       {config['pretrain_approach'] != 'None'}")
    print(f"  pretrain_approach:    {config['pretrain_approach']}")
    model_filename = config.get('pretrained_model_filename')
    print(f"  pretrained_model_filename:  {model_filename if model_filename else '(default for model_type)'}")
    print(f"  fold(s):              {[args.fold] if not args.all_folds else [0, 1]}")
    print("=" * 70 + "\n")

    # ── Load user splits once ────────────────────────────────────────────────
    splits_path = Path(config["user_split_json_filepath"])
    with open(splits_path, "r") as f:
        all_splits = json.load(f)

    folds_to_run = [0, 1] if args.all_folds else [args.fold]

    # ── Run ──────────────────────────────────────────────────────────────────
    all_fold_results = []
    for fold_idx in folds_to_run:
        fold_config = copy.deepcopy(config)   # fresh copy per fold
        result = run_fold(
            model_type = args.model_type,
            fold_idx   = fold_idx,
            config     = fold_config,
            all_splits = all_splits,
            timestamp  = timestamp,
        )
        all_fold_results.append(result)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for r in all_fold_results:
        print(f"  Fold {r['fold_idx']}: "
              f"best_val_acc={r['best_val_acc']*100:.2f}%  "
              f"mean_user_acc={r['mean_user_acc']*100:.2f}% "
              f"± {r['std_user_acc']*100:.2f}%")

    if len(all_fold_results) > 1:
        overall = np.mean([r["mean_user_acc"] for r in all_fold_results])
        print(f"\n  Overall mean across {len(all_fold_results)} folds: {overall*100:.2f}%")

    # Save JSON summary
    summary_path = results_save_dir / f"{args.model_type}_{timestamp}_summary.json"
    summary = {
        "model_type":  args.model_type,
        "use_MOE":     config["use_MOE"],
        "MOE_placement": config.get("MOE_placement"),
        "timestamp":   timestamp,
        "folds":       [
            {k: v for k, v in r.items() if k not in ("train_loss_log", "routing_reports")}
            for r in all_fold_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved → {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()