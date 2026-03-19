"""
pretrain_hpo.py
===============
Optuna hyperparameter optimization for the three pretrain architectures.

Philosophy:
  - MetaCNNLSTM and DeepCNNLSTM: architecture is FIXED (matching published designs).
    Only tune training hyperparameters (lr, wd, dropout, label_smooth, batch_size).
  - TST: architecture CAN be tuned (d_model, n_heads, n_blocks, patch_len) because
    the transformer is much more sensitive to scale than CNN-LSTMs.

Each trial:
  - Trains for a short budget (40 epochs or until early stop at patience=7)
  - Returns val accuracy as the objective

Optuna dashboard:
    optuna-dashboard sqlite:///hpo_results.db

Usage:
    python pretrain_hpo.py --model MetaCNNLSTM --n_trials 50 --tensor_dict path/to/dict.pkl
    python pretrain_hpo.py --model TST --n_trials 100 --tensor_dict path/to/dict.pkl
"""

import argparse
import torch
import optuna
from optuna.samplers import TPESampler

from pretrain_models import build_model, DEFAULT_CONFIGS
from pretrain_data_pipeline import get_pretrain_dataloaders
from pretrain_trainer import pretrain


# ─────────────────────────────────────────────────────────────────────────────
# Default user / trial split — first fold only for HPO
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_USER_SPLIT = {
    "train_PIDs": [
        "P102","P114","P119","P005","P107","P126","P132","P112",
        "P103","P125","P127","P010","P128","P111","P118",
        "P124","P110","P116","P108","P104","P122","P131","P106","P115",
    ],
    "val_PIDs":  ["P011","P006","P105","P109"],
    "test_PIDs": ["P008","P004","P123","P121"],

    # 1-indexed trial/repetition numbers (NOT class labels)
    "train_reps": list(range(1, 9)),  # trials 1-8
    "val_reps":   [9, 10],            # trials 9-10

    # 0-indexed gesture class labels
    "available_gesture_classes": list(range(0, 10)),
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared augmentation defaults used in every HPO trial
# ─────────────────────────────────────────────────────────────────────────────

HPO_AUG_DEFAULTS = {
    "augment":       True,
    "aug_noise_std": 0.05,
    "aug_max_shift": 4,
    "aug_ch_drop":   0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Objective functions
# ─────────────────────────────────────────────────────────────────────────────

def objective_metacnnlstm(trial: optuna.Trial, tensor_dict_path: str, user_split: dict, device: str):
    """
    MetaCNNLSTM: fixed architecture, tune only training hyperparams.
    Architecture: 1 Conv + LSTM + linear head — fixed to match Meta paper.
    """
    config = {
        **DEFAULT_CONFIGS["MetaCNNLSTM"],
        **user_split,
        **HPO_AUG_DEFAULTS,
        "device": device,
        # ── Tuned: training hyperparameters ──────────────────────────────────
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",  1e-5, 1e-2, log=True),
        "dropout":       trial.suggest_float("dropout",       0.0,  0.4,  step=0.05),
        "label_smooth":  trial.suggest_float("label_smooth",  0.0,  0.2,  step=0.05),
        "batch_size":    trial.suggest_categorical("batch_size",    [32, 64, 128]),
        "lstm_hidden":   trial.suggest_categorical("lstm_hidden",   [64, 128, 256]),
        "cnn_filters":   trial.suggest_categorical("cnn_filters",   [32, 64, 128]),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "head_type":     "linear",   # FIXED per Meta paper
        # ── Fixed: short training budget for HPO ─────────────────────────────
        "num_epochs":         40,
        "warmup_epochs":      3,
        "use_early_stopping": True,
        "es_patience":        7,
        "use_scheduler":      True,
        "num_workers":        4,
    }

    train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict_path)
    model = build_model(config)
    _, history = pretrain(model, train_dl, val_dl, config, save_path=None)
    return history.get('best_val_acc', 0.0)


def objective_deepcnnlstm(trial: optuna.Trial, tensor_dict_path: str, user_split: dict, device: str):
    """
    DeepCNNLSTM: architecture mostly fixed; tune training + minor arch params.
    """
    config = {
        **DEFAULT_CONFIGS["DeepCNNLSTM"],
        **user_split,
        **HPO_AUG_DEFAULTS,
        "device": device,
        # ── Tuned ────────────────────────────────────────────────────────────
        "learning_rate":    trial.suggest_float("learning_rate",   5e-5, 5e-3, log=True),
        "weight_decay":     trial.suggest_float("weight_decay",    1e-5, 1e-2, log=True),
        "dropout":          trial.suggest_float("dropout",         0.1,  0.5,  step=0.05),
        "label_smooth":     trial.suggest_float("label_smooth",    0.0,  0.2,  step=0.05),
        "batch_size":       trial.suggest_categorical("batch_size",       [32, 64, 128]),
        "cnn_base_filters": trial.suggest_categorical("cnn_base_filters", [32, 64]),
        "lstm_hidden":      trial.suggest_categorical("lstm_hidden",      [64, 128, 256]),
        "bidirectional":    trial.suggest_categorical("bidirectional",    [True, False]),
        "head_type":        trial.suggest_categorical("head_type",        ["linear", "mlp"]),
        # ── Fixed ─────────────────────────────────────────────────────────────
        "num_epochs":         40,
        "warmup_epochs":      3,
        "use_early_stopping": True,
        "es_patience":        7,
        "use_scheduler":      True,
        "num_workers":        4,
    }

    train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict_path)
    model = build_model(config)
    _, history = pretrain(model, train_dl, val_dl, config, save_path=None)
    return history.get('best_val_acc', 0.0)


def objective_tst(trial: optuna.Trial, tensor_dict_path: str, user_split: dict, device: str):
    """
    TST: both architecture AND training hyperparams are tuned.
    Transformers are much more sensitive to scale than CNNs.

    Key constraint: d_model must be divisible by n_heads.
    """
    d_model_choice = trial.suggest_categorical("d_model", [32, 64, 128])
    valid_heads = {32: [2, 4], 64: [4, 8], 128: [4, 8]}
    n_heads = trial.suggest_categorical("n_heads", valid_heads[d_model_choice])

    config = {
        **DEFAULT_CONFIGS["TST"],
        **user_split,
        **HPO_AUG_DEFAULTS,
        "device": device,
        # ── Architecture ──────────────────────────────────────────────────────
        "d_model":  d_model_choice,
        "n_heads":  n_heads,
        "d_ff":     trial.suggest_categorical("d_ff",     [128, 256, 512]),
        "n_blocks": trial.suggest_int("n_blocks", 2, 6),
        "patch_len":trial.suggest_categorical("patch_len",[4, 8, 16]),
        # ── Training hyperparameters ──────────────────────────────────────────
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",  1e-5, 5e-3, log=True),
        "dropout":       trial.suggest_float("dropout",       0.0,  0.4,  step=0.05),
        "label_smooth":  trial.suggest_float("label_smooth",  0.0,  0.2,  step=0.05),
        "batch_size":    trial.suggest_categorical("batch_size",    [32, 64, 128]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 3, 15),
        "head_type":     trial.suggest_categorical("head_type",     ["linear", "mlp"]),
        # ── Fixed ─────────────────────────────────────────────────────────────
        "num_epochs":         40,
        "use_early_stopping": True,
        "es_patience":        7,
        "use_scheduler":      True,
        "num_workers":        4,
    }

    train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict_path)
    model = build_model(config)
    _, history = pretrain(model, train_dl, val_dl, config, save_path=None)
    return history.get('best_val_acc', 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

OBJECTIVE_MAP = {
    "MetaCNNLSTM": objective_metacnnlstm,
    "DeepCNNLSTM": objective_deepcnnlstm,
    "TST":         objective_tst,
}


def run_hpo(
    model_type: str,
    tensor_dict_path: str,
    n_trials: int = 50,
    user_split: dict = None,
    device: str = None,
    storage: str = None,
    study_name: str = None,
):
    if user_split is None:
        user_split = DEFAULT_USER_SPLIT
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if study_name is None:
        study_name = f"pretrain_hpo_{model_type}"
    if storage is None:
        # Optuna Journal (file-based) is preferred over SQLite for single-machine runs
        # because it avoids locking issues. Switch to MySQL for multi-node HPO.
        storage = f"sqlite:///hpo_{model_type}.db"

    objective_fn = OBJECTIVE_MAP[model_type]
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        sampler        = sampler,
        direction      = "maximize",
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective_fn(trial, tensor_dict_path, user_split, device),
        n_trials          = n_trials,
        n_jobs            = 1,   # increase only if you have multiple GPUs
        show_progress_bar = True,
    )

    print(f"\n{'='*60}")
    print(f"HPO complete — {model_type}")
    print(f"Best val accuracy: {study.best_value * 100:.2f}%")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    return study


def get_best_config(model_type: str, study: optuna.Study) -> dict:
    """
    Build a full config dict from the best HPO trial.
    Returns a config ready for build_model() and get_pretrain_dataloaders().
    """
    base = DEFAULT_CONFIGS[model_type].copy()
    base.update(study.best_params)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain HPO with Optuna")
    parser.add_argument("--model",       type=str, required=True, choices=list(OBJECTIVE_MAP.keys()))
    parser.add_argument("--tensor_dict", type=str, required=True, help="Path to tensor_dict.pkl")
    parser.add_argument("--n_trials",    type=int, default=50,    help="Number of Optuna trials")
    parser.add_argument("--device",      type=str, default=None,  help="cuda or cpu")
    parser.add_argument("--storage",     type=str, default=None,  help="Optuna storage URL")
    args = parser.parse_args()

    run_hpo(
        model_type       = args.model,
        tensor_dict_path = args.tensor_dict,
        n_trials         = args.n_trials,
        device           = args.device,
        storage          = args.storage,
    )
