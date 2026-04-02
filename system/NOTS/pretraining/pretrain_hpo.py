"""
pretrain_hpo.py
===============
Optuna hyperparameter optimisation for supervised pretraining of EMG gesture
classification models.

Supported architectures
-----------------------
  MetaCNNLSTM   — fixed architecture, tune training HPs only
  DeepCNNLSTM   — fixed architecture, tune training HPs + minor arch params
  TST           — arch + training HPs both tuned (transformers are scale-sensitive)
  ContrastiveNet— contrastive/SupCon pretraining via contrastive_train.train()

MoE variants
------------
Pass --use_moe to add MoE routing to any CNN-LSTM backbone (MetaCNNLSTM or
DeepCNNLSTM) or to ContrastiveNet.  top_k is fixed to None (dense / soft
routing) throughout — gradients flow cleanly, MAML-compatible.
MoE routing HPs (temperature, aux_coeff, ctx dims, placement) are searched
alongside training HPs.  The trainers handle aux-loss and routing analysis
every moe_log_every=5 epochs automatically.

Collapse detection
------------------
For CNN-LSTM: pretrain_trainer.py logs routing_reports to history every
moe_log_every epochs. We read the last report and penalise trials where
max_expert_load > COLLAPSE_MAX_LOAD_THRESHOLD.
For ContrastiveNet: same check on logs['routing_reports'] if present.

Usage
-----
  python pretrain_hpo.py --model DeepCNNLSTM --n_trials 5 \
         --tensor_dict /path/to/tensor_dict.pkl

  python pretrain_hpo.py --model DeepCNNLSTM --n_trials 5 --use_moe \
         --tensor_dict /path/to/tensor_dict.pkl

  python pretrain_hpo.py --model ContrastiveNet --n_trials 5 \
         --tensor_dict /path/to/tensor_dict.pkl

  # SLURM (see pretrain_hpo.slurm):
  python pretrain_hpo.py --model DeepCNNLSTM --n_trials 5 \
         --study_name pretrain_hpo_DeepCNNLSTM \
         --journal_path /scratch/.../optuna_dbs/pretrain_hpo_DeepCNNLSTM.log \
         --tensor_dict /path/to/tensor_dict.pkl
"""

import os, sys, argparse, json, time, random, pickle, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# ── Paths — override via env vars (set in SLURM script) ─────────────────────
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()  # CODE_DIR is /projects/my13/kai/meta-pers-gest/pers-gest-cls (on NOTS anyways)
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./runs/pretrain_hpo")).resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE_DIR))

# Only do this if the below package imports dont work
## Create the pretraining path
#PRETRAINING_DIR = CODE_DIR / "system" / "pretraining"
## Append it to sys.path
#sys.path.append(str(PRETRAINING_DIR))
from system.pretraining.pretrain_configs import PRETRAIN_CONFIG, MODEL_CONFIGS
from system.pretraining.pretrain_models import build_model
from system.pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from system.pretraining.pretrain_trainer import pretrain

# ── Collapse detection constants (MoE only) ──────────────────────────────────
COLLAPSE_MAX_LOAD_THRESHOLD = 0.80
COLLAPSE_PENALTY            = 0.0

# ── Fixed study constants ─────────────────────────────────────────────────────
FIXED_SEED  = 42
NUM_EXPERTS = 4

# ─────────────────────────────────────────────────────────────────────────────
# Default user / rep split (fold 0)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_USER_SPLIT = {
    "train_PIDs": PRETRAIN_CONFIG["train_PIDs"],
    "val_PIDs":   PRETRAIN_CONFIG["val_PIDs"],
    "test_PIDs":  PRETRAIN_CONFIG["test_PIDs"],
    "train_reps": PRETRAIN_CONFIG["train_reps"],
    "val_reps":   PRETRAIN_CONFIG["val_reps"],
    "available_gesture_classes": PRETRAIN_CONFIG["available_gesture_classes"],
}

HPO_AUG_DEFAULTS = {
    "augment":       True,
    "aug_noise_std": 0.05,
    "aug_max_shift": 4,
    "aug_ch_drop":   0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared HP suggestion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_training_hps(trial: optuna.Trial) -> dict:
    """HPs common to CNN-LSTM models: lr, wd, dropout, label smooth, batch size."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",  1e-5, 1e-2, log=True),
        "dropout":       trial.suggest_float("dropout",       0.0,  0.4,  step=0.05),
        "label_smooth":  trial.suggest_float("label_smooth",  0.0,  0.2,  step=0.05),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128]),
        "optimizer":     trial.suggest_categorical("optimizer",  ["adamw", "adam"]),
    }


def _suggest_moe_hps(trial: optuna.Trial) -> dict:
    """
    MoE routing HPs.  top_k is fixed to None (dense routing) — you are
    enforcing this in the rest of the codebase too.  Both snake_case and
    UPPER_case variants of each key are injected because different modules
    use different conventions (MOE_encoder.py uses UPPER, pretrain_trainer.py
    uses lower, and one line reads the bare 'top_k' key).
    """
    placement   = trial.suggest_categorical("MOE_placement",      ["encoder", "middle"])
    gate_temp   = trial.suggest_float("MOE_gate_temperature",     0.5, 8.0, log=True)
    aux_coeff   = trial.suggest_float("MOE_aux_coeff",            1e-4, 1.0, log=True)
    ctx_out_dim = trial.suggest_categorical("MOE_ctx_out_dim",    [16, 32, 64, 128])
    ctx_hidden  = trial.suggest_categorical("MOE_ctx_hidden_dim", [32, 64, 128])
    moe_dropout = trial.suggest_float("MOE_dropout",              0.0, 0.3)
    expert_exp  = trial.suggest_float("MOE_expert_expand",        0.5, 1.5)
    mlp_mult    = trial.suggest_float("MOE_mlp_hidden_mult",      0.5, 2.0)

    return {
        "use_MOE":              True,
        "num_experts":          NUM_EXPERTS,
        # snake_case (pretrain_configs / pretrain_trainer)
        "MOE_placement":        placement,
        "MOE_gate_temperature": gate_temp,
        "MOE_aux_coeff":        aux_coeff,
        "MOE_ctx_out_dim":      ctx_out_dim,
        "MOE_ctx_hidden_dim":   ctx_hidden,
        "MOE_dropout":          moe_dropout,
        "MOE_expert_expand":    expert_exp,
        "MOE_mlp_hidden_mult":  mlp_mult,
        "MOE_top_k":            None,   # dense routing, fixed
        "top_k":                None,   # pretrain_trainer.py line 144 reads this key
        # UPPER_case (MOE_encoder.py)
        "MOE_placement":        placement,
        "MOE_gate_temperature": gate_temp,
        "MOE_aux_coeff":        aux_coeff,
        "MOE_ctx_out_dim":      ctx_out_dim,
        "MOE_ctx_hidden_dim":   ctx_hidden,
        "MOE_dropout":          moe_dropout,
        "MOE_expert_expand":    expert_exp,
        "MOE_mlp_hidden_mult":  mlp_mult,
        "MOE_top_k":            None,   # dense routing, fixed
        # Routing analysis (RoutingCollector already in pretrain_trainer.py)
        "MOE_log_every":        5,
        "MOE_plot_dir":         None,
    }


def _fixed_training_schedule() -> dict:
    return {
        "num_epochs":         40,
        "warmup_epochs":      3,
        "use_early_stopping": True,
        "es_patience":        7,
        "es_min_delta":       1e-4,
        "use_scheduler":      True,
        "grad_clip":          5.0,
        "use_amp":            False,
        "num_workers":        4,
    }


def _check_moe_collapse(history_or_logs: dict) -> float | None:
    """
    Read max_expert_load from the last routing_report in a history/logs dict.
    Tries several plausible key paths to be robust to RoutingAnalyzer changes.
    Returns the max_load (0-1) if found, else None.
    """
    reports = history_or_logs.get("routing_reports", [])
    if not reports:
        return None
    last = reports[-1]
    for key in ("max_expert_load", "dominant_fraction", "max_load"):
        if key in last:
            return float(last[key])
    for key in ("max_expert_load", "dominant_fraction", "max_load"):
        if key in last.get("load_balance", {}):
            return float(last["load_balance"][key])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Objective: MetaCNNLSTM
# ─────────────────────────────────────────────────────────────────────────────

def objective_metacnnlstm(trial: optuna.Trial, tensor_dict: dict,
                           tensor_dict_path: str, user_split: dict,
                           device: str, use_moe: bool) -> float:
    config = {
        **MODEL_CONFIGS["MetaCNNLSTM"],
        **user_split,
        **HPO_AUG_DEFAULTS,
        **_suggest_training_hps(trial),
        **_fixed_training_schedule(),
        "device":        device,
        "lstm_hidden":   trial.suggest_categorical("lstm_hidden",   [64, 128, 256]),
        "cnn_filters":   trial.suggest_categorical("cnn_filters",   [32, 64, 128]),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "head_type":     "linear",
        "use_MOE":       False,
    }
    if use_moe:
        config.update(_suggest_moe_hps(trial))
        from MOE.MOE_encoder import build_MOE_model
        model = build_MOE_model(config)
    else:
        model = build_model(config)

    train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict)
    _, history = pretrain(model, train_dl, val_dl, config, save_path=None)
    val_acc = float(history.get("best_val_acc", 0.0))

    if use_moe:
        max_load = _check_moe_collapse(history)
        trial.set_user_attr("final_max_expert_load", max_load if max_load is not None else -1.0)
        if max_load is not None and max_load > COLLAPSE_MAX_LOAD_THRESHOLD:
            print(f"  [Trial {trial.number}] MoE COLLAPSED (max_load={max_load:.2f}). Penalising.")
            return COLLAPSE_PENALTY

    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Objective: DeepCNNLSTM
# ─────────────────────────────────────────────────────────────────────────────

def objective_deepcnnlstm(trial: optuna.Trial, tensor_dict: dict,
                           tensor_dict_path: str, user_split: dict,
                           device: str, use_moe: bool) -> float:
    config = {
        **MODEL_CONFIGS["DeepCNNLSTM"],
        **user_split,
        **HPO_AUG_DEFAULTS,
        **_suggest_training_hps(trial),
        **_fixed_training_schedule(),
        "device":           device,
        "cnn_base_filters": trial.suggest_categorical("cnn_base_filters", [32, 64]),
        "lstm_hidden":      trial.suggest_categorical("lstm_hidden",      [64, 128, 256]),
        "bidirectional":    trial.suggest_categorical("bidirectional",    [True, False]),
        "head_type":        trial.suggest_categorical("head_type",        ["linear", "mlp"]),
        "use_MOE":          False,
    }
    if use_moe:
        config.update(_suggest_moe_hps(trial))
        from MOE.MOE_encoder import build_MOE_model
        model = build_MOE_model(config)
    else:
        model = build_model(config)

    train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict)
    _, history = pretrain(model, train_dl, val_dl, config, save_path=None)
    val_acc = float(history.get("best_val_acc", 0.0))

    if use_moe:
        max_load = _check_moe_collapse(history)
        trial.set_user_attr("final_max_expert_load", max_load if max_load is not None else -1.0)
        if max_load is not None and max_load > COLLAPSE_MAX_LOAD_THRESHOLD:
            print(f"  [Trial {trial.number}] MoE COLLAPSED (max_load={max_load:.2f}). Penalising.")
            return COLLAPSE_PENALTY

    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Objective: TST
# ─────────────────────────────────────────────────────────────────────────────

def objective_tst(trial: optuna.Trial, tensor_dict: dict,
                  tensor_dict_path: str, user_split: dict,
                  device: str, use_moe: bool) -> float:
    if use_moe:
        raise ValueError("MOE is not supported for TST — no CNN encoder to attach experts to.")

    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    n_heads = trial.suggest_categorical("n_heads", {32: [2, 4], 64: [4, 8], 128: [4, 8]}[d_model])

    config = {
        **MODEL_CONFIGS["TST"],
        **user_split,
        **HPO_AUG_DEFAULTS,
        **_suggest_training_hps(trial),
        **_fixed_training_schedule(),
        "device":        device,
        "d_model":       d_model,
        "n_heads":       n_heads,
        "d_ff":          trial.suggest_categorical("d_ff",      [128, 256, 512]),
        "n_blocks":      trial.suggest_int("n_blocks", 2, 6),
        "patch_len":     trial.suggest_categorical("patch_len", [4, 8, 16]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 3, 15),
        "head_type":     trial.suggest_categorical("head_type", ["linear", "mlp"]),
        "use_MOE":       False,
    }

    train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict)
    model = build_model(config)
    _, history = pretrain(model, train_dl, val_dl, config, save_path=None)
    return float(history.get("best_val_acc", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Objective: ContrastiveNet
# ─────────────────────────────────────────────────────────────────────────────

def objective_contrastivenet(trial: optuna.Trial, tensor_dict: dict,
                              tensor_dict_path: str, user_split: dict,
                              device: str, use_moe: bool) -> float:
    """
    ContrastiveGestureEncoder (or ContrastiveEncoderMOE when use_moe=True)
    trained with SupCon/Siamese loss via contrastive_train.train().

    Key interface facts from reading the code:
      - get_contrastive_dataloaders() takes tensor_dict_path (string path),
        NOT a pre-loaded dict. It loads internally.
      - train() reads config['tensor_dict_path'] and config['run_dir'] directly.
      - train() returns (best_val_1nn_acc, best_state, logs).
      - The objective metric is best_val_1nn_acc — 1-shot prototypical val
        accuracy over unseen users. This directly proxies MAML downstream.
      - When use_moe=True, train() must build ContrastiveEncoderMOE instead
        of ContrastiveGestureEncoder. See note at bottom of this function.
    """
    from system.pretraining.contrastive_net.contrastive_train import train as contrastive_train
    from system.pretraining.contrastive_net.contrastive_config import get_config as get_contrastive_config

    # Start from the full canonical config so no key is missing
    config = get_contrastive_config()

    # ── User / rep split ──────────────────────────────────────────────────────
    config.update({
        "train_PIDs": user_split["train_PIDs"],
        "val_PIDs":   user_split["val_PIDs"],
        "test_PIDs":  user_split["test_PIDs"],
        "train_reps": user_split.get("train_reps", config["train_reps"]),
        "val_reps":   user_split.get("val_reps",   config["val_reps"]),
    })

    # ── Paths that train() reads directly from config ─────────────────────────
    config["tensor_dict_path"] = tensor_dict_path
    config["run_dir"]          = str(RUN_DIR)
    config["device"]           = device

    # ── Fixed architecture (matches inject_model_config + contrastive_config) ─
    config.update({
        "emg_in_ch":            16,
        "imu_in_ch":            72,
        "use_imu":              True,
        "sequence_length":      64,
        "num_classes":          10,
        "gesture_labels":       list(range(10)),
        "emg_cnn_layers":       3,
        "imu_cnn_layers":       2,
        "cnn_kernel_size":      5,
        "emg_stride":           1,
        "imu_stride":           1,
        "groupnorm_num_groups": 8,
        "attn_pool_heads":      4,
        "use_GlobalAvgPooling": True,
        "lstm_layers":          2,
    })

    # ── Searched: architecture ────────────────────────────────────────────────
    arch_mode = trial.suggest_categorical("arch_mode", ["cnn_attn", "cnn_lstm"])
    config.update({
        "arch_mode":            arch_mode,
        "use_lstm":             arch_mode == "cnn_lstm",
        "emg_base_cnn_filters": trial.suggest_categorical(
            "emg_base_cnn_filters", [32, 64, 128]),
        "imu_base_cnn_filters": trial.suggest_categorical(
            "imu_base_cnn_filters", [16, 32, 64]),
        "embedding_dim":        trial.suggest_categorical(
            "embedding_dim", [64, 128, 256]),
        "proj_hidden_dim":      trial.suggest_categorical(
            "proj_hidden_dim", [128, 256, 512]),
        "lstm_hidden":          trial.suggest_categorical(
            "lstm_hidden", [64, 128, 256]),
    })

    # ── Searched: training ────────────────────────────────────────────────────
    config.update({
        "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":      trial.suggest_float("weight_decay",  1e-5, 1e-2, log=True),
        "dropout":           trial.suggest_float("dropout",        0.0,  0.4,  step=0.05),
        "optimizer":         trial.suggest_categorical("optimizer", ["adamw", "adam"]),
        "batch_construction": "balanced",
        "samples_per_class": trial.suggest_categorical("samples_per_class", [4, 6, 8]),
        "classes_per_batch": trial.suggest_categorical("classes_per_batch", [6, 8, 10]),
    })

    # ── Searched: contrastive loss ────────────────────────────────────────────
    config.update({
        "loss_mode":          trial.suggest_categorical("loss_mode", ["supcon", "siamese"]),
        "supcon_temperature": trial.suggest_float("supcon_temperature", 0.05, 0.5, log=True),
    })

    # ── Fixed training schedule (short HPO budget) ────────────────────────────
    config.update({
        "num_epochs":              40,
        "lr_warmup_epochs":        3,
        "use_earlystopping":       True,
        "earlystopping_patience":  7,
        "earlystopping_min_delta": 0.002,
        "lr_scheduler":            "cosine",
        "grad_clip":               5.0,
        "num_workers":             4,
        "num_val_episodes":        20,
        "val_support_shots":       1,
        "val_query_per_class":     9,
        # Run linear probe less often during HPO to save time
        "epochs_between_linprob":  20,
        "linprob_epochs":          30,
        "linprob_lr":              1e-2,
        "seed":                    FIXED_SEED,
        "use_MOE":                 False,
    })

    # ── MoE (optional) ───────────────────────────────────────────────────────
    if use_moe:
        config.update(_suggest_moe_hps(trial))
        # NOTE: contrastive_train.train() currently always instantiates
        # ContrastiveGestureEncoder(config). To support MoE, add this one-line
        # check near the "# ---- Model ----" comment in contrastive_train.py:
        #
        #   if config.get('use_moe', False):
        #       from MOE.MOE_encoder import build_MOE_model
        #       model = build_MOE_model(config)
        #   else:
        #       model = ContrastiveGestureEncoder(config)
        #
        # ContrastiveEncoderMOE already has the same forward/encode/
        # get_prototypes/predict interface as ContrastiveGestureEncoder, so
        # the rest of contrastive_train.py needs no other changes.

    # ── Run training ──────────────────────────────────────────────────────────
    # train() returns (best_val_1nn_acc, best_state, logs)
    best_val_1nn_acc, _best_state, logs = contrastive_train(config, fold_idx=0)

    val_acc = float(best_val_1nn_acc)
    trial.set_user_attr("best_val_1nn_acc", val_acc)

    if use_moe:
        max_load = _check_moe_collapse(logs)
        trial.set_user_attr("final_max_expert_load",
                            max_load if max_load is not None else -1.0)
        if max_load is not None and max_load > COLLAPSE_MAX_LOAD_THRESHOLD:
            print(f"  [Trial {trial.number}] MoE COLLAPSED (max_load={max_load:.2f}). Penalising.")
            return COLLAPSE_PENALTY

    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

OBJECTIVE_MAP = {
    "MetaCNNLSTM":    objective_metacnnlstm,
    "DeepCNNLSTM":    objective_deepcnnlstm,
    "TST":            objective_tst,
    "ContrastiveNet": objective_contrastivenet,
}


def objective(trial: optuna.Trial, model_type: str, tensor_dict: dict,
              tensor_dict_path: str, user_split: dict,
              device: str, use_moe: bool) -> float:
    try:
        return OBJECTIVE_MAP[model_type](
            trial, tensor_dict, tensor_dict_path, user_split, device, use_moe
        )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  [Trial {trial.number}] ERROR: {e}")
        import traceback; traceback.print_exc()
        raise optuna.TrialPruned(f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Study runner
# ─────────────────────────────────────────────────────────────────────────────

def run_study(
    model_type: str, tensor_dict: dict, tensor_dict_path: str,
    n_trials: int, study_name: str, journal_path: str,
    user_split: dict = None, device: str = None, use_moe: bool = False,
) -> optuna.Study:
    if user_split is None:
        user_split = DEFAULT_USER_SPLIT
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sleep_time = random.uniform(0, 10)
    print(f"[Worker] Staggering start by {sleep_time:.1f}s …")
    time.sleep(sleep_time)

    os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
    storage = JournalStorage(JournalFileBackend(journal_path))
    time.sleep(random.uniform(0, 5))

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        sampler        = TPESampler(seed=FIXED_SEED),
        direction      = "maximize",
        load_if_exists = True,
        pruner         = optuna.pruners.MedianPruner(
            n_startup_trials = max(5, n_trials // 4),
            n_warmup_steps   = 3,
        ),
    )

    study.optimize(
        lambda trial: objective(trial, model_type, tensor_dict, tensor_dict_path,
                                user_split, device, use_moe),
        n_trials       = n_trials,
        gc_after_trial = True,
    )
    return study


def print_best(study: optuna.Study, model_type: str, use_moe: bool):
    try:
        best = study.best_trial
    except ValueError:
        print("[HPO] No completed trials yet.")
        return
    print(f"\n{'='*65}")
    print(f"BEST TRIAL #{best.number}  |  {model_type}  |  MoE={use_moe}")
    print(f"  val_acc : {best.value * 100:.2f}%")
    if use_moe:
        ml = best.user_attrs.get("final_max_expert_load", -1.0)
        print(f"  max_load: {ml:.3f}  ({'COLLAPSED' if ml > COLLAPSE_MAX_LOAD_THRESHOLD else 'OK'})")
    print("  HPs:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:35s}: {v}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str, required=True, choices=list(OBJECTIVE_MAP))
    parser.add_argument("--tensor_dict",  type=str, required=True)
    parser.add_argument("--n_trials",     type=int, default=5)
    parser.add_argument("--use_moe",      action="store_true", default=False)
    parser.add_argument("--study_name",   type=str, default=None)
    parser.add_argument("--journal_path", type=str, default=None)
    parser.add_argument("--device",       type=str, default=None)
    args = parser.parse_args()

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"Loading tensor dict: {args.tensor_dict}")
    with open(args.tensor_dict, "rb") as f:
        full = pickle.load(f)
    tensor_dict = full["data"] if "data" in full else full
    print(f"Loaded {len(tensor_dict)} subjects.")

    moe_suffix   = "_moe" if args.use_moe else ""
    study_name   = args.study_name or f"pretrain_hpo_{args.model}{moe_suffix}"
    db_dir       = os.environ.get("OPTUNA_DB_DIR",
                                  "/scratch/my13/kai/meta-pers-gest/optuna_dbs")
    os.makedirs(db_dir, exist_ok=True)
    journal_path = args.journal_path or os.path.join(db_dir, f"{study_name}.log")
    device       = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"Study   : {study_name}")
    print(f"Model   : {args.model}  |  MoE={args.use_moe}")
    print(f"Journal : {journal_path}")
    print(f"Trials  : {args.n_trials} (this worker)")
    print(f"Device  : {device}")
    if args.use_moe:
        print(f"Experts : {NUM_EXPERTS} (fixed, dense routing, top_k=None)")
        print(f"Collapse: >{COLLAPSE_MAX_LOAD_THRESHOLD:.0%} max expert load → penalised")
    print(f"{'='*65}\n")

    study = run_study(
        model_type       = args.model,
        tensor_dict      = tensor_dict,
        tensor_dict_path = args.tensor_dict,
        n_trials         = args.n_trials,
        study_name       = study_name,
        journal_path     = journal_path,
        device           = device,
        use_moe          = args.use_moe,
    )

    print_best(study, args.model, args.use_moe)

    try:
        best     = study.best_trial
        out_path = RUN_DIR / f"{study_name}_best_hps.json"
        with open(out_path, "w") as f:
            json.dump(best.params, f, indent=2, default=str)
        print(f"Best HPs saved → {out_path}")
    except Exception as e:
        print(f"Could not save best HPs: {e}")