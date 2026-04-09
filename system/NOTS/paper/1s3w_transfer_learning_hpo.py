# =============================================================================
# WARM-START CONFIGURATION
# =============================================================================
# NOTE: Not using currently!
# Paste top-N param dicts from a previous study here once you have them.
# Keys not actively suggest_*'d in this study are automatically dropped.
WARM_START_PARAMS: list[dict] = [
    # --- Paste warm-start trial dicts here when available ---
]

import os
N_TRIALS = int(os.environ.get("N_TRIALS", 100))
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

def apply_fold_to_config(config: dict, all_splits: dict, fold_idx: int) -> None:
    """Mutate config in-place to set train/val/test PIDs for the given fold."""
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]

# ---------------------------------------------------------------------------
# Imports from the pretraining system
# ---------------------------------------------------------------------------
from system.pretraining.pretrain_models        import build_model
from system.pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from system.pretraining.pretrain_trainer       import pretrain
from system.pretraining.pretrain_finetune      import evaluate_all_val_users

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

#############################################################

def build_config_from_trial(trial, eval_mode: str) -> dict:
    """
    Build a full training + eval config from an Optuna trial.

    eval_mode:
        'zero_shot'  — HPO objective is zero-shot query accuracy
        'full'       — HPO objective is 1-shot full-finetuning accuracy
        'head_only'  — HPO objective is 1-shot head-only finetuning accuracy

    Hyperparameters swept:
        Pretraining:
            learning_rate, weight_decay, label_smooth, num_epochs,
            batch_size, groupnorm_num_groups, augment,
            cnn_base_filters, lstm_hidden
        Finetuning (not used for zero_shot):
            ft_lr, ft_steps, ft_optimizer
    """
    config = {}

    # ── Architecture ─────────────────────────────────────────────────────────
    config["model_type"]    = "DeepCNNLSTM"
    config["sequence_length"] = 64
    config["emg_in_ch"]     = 16
    config["imu_in_ch"]     = 72
    config["demo_in_dim"]   = 12

    config["cnn_base_filters"] = trial.suggest_categorical("cnn_base_filters", [64, 96, 128])
    config["lstm_hidden"]      = trial.suggest_categorical("lstm_hidden",      [64, 128])
    config["cnn_layers"]       = 3
    config["cnn_kernel"]       = 5
    config["groupnorm_num_groups"] = trial.suggest_categorical("groupnorm_num_groups", [4, 8])
    config["lstm_layers"]      = 3
    config["bidirectional"]    = True
    config["head_type"]        = "mlp"
    config["dropout"]          = 0.1

    # ── Task setup (matches MAML HPO for fair comparison) ────────────────────
    config["n_way"]       = 3
    config["k_shot"]      = 1
    config["q_query"]     = 9
    config["num_classes"] = 10

    # ── Modality ─────────────────────────────────────────────────────────────
    config["use_imu"]         = True
    config["use_demographics"] = False
    config["use_MOE"]         = False   # This is the no-MOE ablation

    # ── Data ─────────────────────────────────────────────────────────────────
    config["NOTS"] = True
    config["user_split_json_filepath"] = user_split_json_filepath
    config["results_save_dir"] = results_save_dir
    config["models_save_dir"]  = models_save_dir
    config["emg_imu_pkl_full_path"] = f"{CODE_DIR}//dataset//filtered_datasets//metadata_IMU_EMG_allgestures_allusers.pkl"
    config["pwmd_xlsx_filepath"]    = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants with disabilities.xlsx"
    config["pwoutmd_xlsx_filepath"] = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants without disabilities.xlsx"
    config["dfs_save_path"]    = f"{CODE_DIR}/dataset//"
    config["dfs_load_path"]    = f"{CODE_DIR}/dataset/meta-learning-sup-que-ds//"

    # Trial/rep splits: reps 1-8 for training, 9-10 for val within pretrain loader.
    # Support/query split for finetuning eval mirrors the 1-shot protocol:
    #   rep 9 → support (the 1 labelled example the user provides)
    #   rep 10 → query  (held-out for evaluation)
    # This is consistent with the MAML episodic setup.
    config["train_reps"]      = [1, 2, 3, 4, 5, 6, 7, 8]
    config["val_reps"]        = [9, 10]
    config["ft_support_reps"] = [9]   # 1-shot support rep
    config["ft_query_reps"]   = [10]  # query rep (held out)

    config["available_gesture_classes"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["maml_gesture_classes"]      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["target_trial_indices"]      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # ── Pretraining optimisation ─────────────────────────────────────────────
    config["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    config["weight_decay"]  = trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True)
    config["label_smooth"]  = trial.suggest_categorical("label_smooth", [0.0, 0.05, 0.1, 0.15, 0.2])
    config["num_epochs"]    = trial.suggest_categorical("num_epochs", [50, 75, 100])
    config["batch_size"]    = trial.suggest_categorical("batch_size", [32, 64, 128])
    config["augment"]       = trial.suggest_categorical("augment", [True, False])

    config["optimizer"]        = "adamw"
    config["use_scheduler"]    = True
    config["warmup_epochs"]    = 5
    config["grad_clip"]        = 5.0
    config["use_early_stopping"] = True
    config["es_patience"]      = 10
    config["es_min_delta"]     = 0.001
    config["num_workers"]      = 8
    config["use_amp"]          = False  # Keep deterministic for HPO
    config["MOE_log_every"]    = 0      # No MoE logging needed
    config["MOE_plot_dir"]     = None

    # ── Finetuning hyperparameters (only used when eval_mode != 'zero_shot') ─
    if eval_mode != 'zero_shot':
        config["ft_lr"]           = trial.suggest_float("ft_lr",  1e-4, 1e-2, log=True)
        config["ft_steps"]        = trial.suggest_categorical("ft_steps", [10, 20, 50, 100])
        config["ft_optimizer"]    = trial.suggest_categorical("ft_optimizer", ["adam", "sgd"])
        config["ft_weight_decay"] = 0.0
    else:
        # Still need these keys present so evaluate_all_val_users doesn't KeyError
        # in case it's called; they won't be used for zero_shot mode.
        config["ft_lr"]           = 1e-3
        config["ft_steps"]        = 0
        config["ft_optimizer"]    = "adam"
        config["ft_weight_decay"] = 0.0

    # ── Eval ─────────────────────────────────────────────────────────────────
    config["num_eval_episodes"] = 10
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return config


#############################################################
# Load splits once at module level (same pattern as MAML HPO)
#############################################################
with open(user_split_json_filepath, "r") as f:
    ALL_SPLITS = json.load(f)
NUM_FOLDS = 1

tensor_dict_path_global = None  # set in __main__, read by objective

def objective(trial, eval_mode: str):
    """
    Optuna objective for non-MAML, non-MOE DeepCNNLSTM.

    Steps per fold:
      1. Build config + model from trial params
      2. Run supervised pretrain() on train_PIDs/train_reps
      3. Evaluate on val_PIDs using evaluate_all_val_users()
         in the requested eval_mode (zero_shot / full / head_only)
      4. Return mean val accuracy across users as the objective
    """
    fold_mean_accs      = []
    pretrain_val_accs   = []
    all_fold_user_accs  = []

    for fold_idx in range(NUM_FOLDS):
        fold_start_time = time.time()

        print("=" * 80)
        print(f"[Trial {trial.number}] Starting fold {fold_idx + 1}/{NUM_FOLDS} | eval_mode={eval_mode}")
        print("=" * 80)

        config = build_config_from_trial(trial, eval_mode)
        apply_fold_to_config(config, ALL_SPLITS, fold_idx)

        print("\nCONFIG:")
        print(config)
        print("\n")

        if config["device"].type == "cpu":
            print("WARNING: Running HPO on CPU — switch to GPU for reasonable runtime!")

        # ── 1. Build model ────────────────────────────────────────────────────
        model = build_model(config)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Trial {trial.number}] Model has {n_params:,} trainable parameters.")
        model.to(config["device"])

        # ── 2. Supervised pretraining ─────────────────────────────────────────
        # Load tensor_dict (caller sets tensor_dict_path_global in __main__)
        import pickle
        with open(tensor_dict_path_global, 'rb') as f:
            full_dict = pickle.load(f)
        tensor_dict = full_dict['data'] if 'data' in full_dict else full_dict

        train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)

        # Save model checkpoint for this trial
        model_filename = os.path.join(str(models_save_dir),
                                      f"trial_{trial.number}_fold_{fold_idx}_best.pt")

        trained_model, history = pretrain(
            model, train_dl, val_dl, config,
            save_path=model_filename,
        )

        best_val_acc = history['best_val_acc']
        pretrain_val_accs.append(float(best_val_acc))
        print(f"[Trial {trial.number} | Fold {fold_idx}] Pretrain done. "
              f"Best pretrain val acc = {best_val_acc:.4f} (supervised, in-distribution)")

        # ── 3. Episodic few-shot evaluation on val users ──────────────────────
        # Load best checkpoint weights before eval
        ckpt = torch.load(model_filename, map_location=config["device"], weights_only=False)
        trained_model.load_state_dict(ckpt["model_state_dict"])

        user_results = evaluate_all_val_users(
            trained_model, config, tensor_dict, mode=eval_mode,
        )

        overall = user_results.pop('__overall__')
        mean_acc_ratio = overall['mean_acc']
        std_acc_ratio  = overall['std_acc']

        all_user_means = [v['mean_acc'] for v in user_results.values()]
        user_acc_percentages = [round(a * 100, 2) for a in all_user_means]

        fold_duration = time.time() - fold_start_time
        print(f"[Trial {trial.number} | Fold {fold_idx}] User accs (%): {user_acc_percentages}")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Mean acc: "
              f"{mean_acc_ratio*100:.2f}% ± {std_acc_ratio*100:.2f}%")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Finished in {fold_duration:.2f} seconds.")

        fold_mean_accs.append(mean_acc_ratio)
        all_fold_user_accs.append(all_user_means)

    clean_fold_accs = [float(f) for f in fold_mean_accs]
    overall_mean_acc = float(np.nanmean(clean_fold_accs))

    trial.set_user_attr("fold_mean_accs",       fold_mean_accs)
    trial.set_user_attr("fold_user_accs",        all_fold_user_accs)
    trial.set_user_attr("mean_pretrain_val_acc", float(np.nanmean(pretrain_val_accs)))

    return overall_mean_acc


def _build_warm_start_params(raw_params: dict, trial_suggest_keys: set) -> dict:
    return {k: v for k, v in raw_params.items() if k in trial_suggest_keys}


# Keys actively suggest_*'d in build_config_from_trial for each eval_mode.
# Update this if you add/remove any trial.suggest_* calls above.
SUGGEST_KEYS_ZEROSHOT = {
    "cnn_base_filters", "lstm_hidden", "groupnorm_num_groups",
    "learning_rate", "weight_decay", "label_smooth",
    "num_epochs", "batch_size", "augment",
}
SUGGEST_KEYS_FINETUNE = SUGGEST_KEYS_ZEROSHOT | {
    "ft_lr", "ft_steps", "ft_optimizer",
}


def run_study(study_name: str, storage_path: str, eval_mode: str, n_trials: int = 1):
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

    suggest_keys = (SUGGEST_KEYS_FINETUNE
                    if eval_mode != 'zero_shot'
                    else SUGGEST_KEYS_ZEROSHOT)

    n_startup = max(20, len(WARM_START_PARAMS))
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

    if WARM_START_PARAMS and len(study.trials) == 0:
        print(f"Warm-starting with {len(WARM_START_PARAMS)} trials...")
        for i, raw_params in enumerate(WARM_START_PARAMS):
            filtered = _build_warm_start_params(raw_params, suggest_keys)
            study.enqueue_trial(filtered)
            print(f"  Enqueued warm-start trial {i}: {filtered}")
    elif WARM_START_PARAMS and len(study.trials) > 0:
        print(f"Study already has {len(study.trials)} trials — skipping warm-start enqueue.")

    study.optimize(
        lambda trial: objective(trial, eval_mode),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPO for non-MAML, non-MOE DeepCNNLSTM (pretrain + few-shot eval)."
    )
    parser.add_argument(
        "--eval_mode", type=str, default="full",
        choices=["zero_shot", "full", "head_only"],
        help=(
            "What to optimise:\n"
            "  zero_shot — maximise query acc with NO finetuning\n"
            "  full      — maximise 1-shot acc after full finetuning\n"
            "  head_only — maximise 1-shot acc after head-only finetuning"
        ),
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir",  type=str, default=None)
    args = parser.parse_args()

    db_dir = "/scratch/my13/kai/meta-pers-gest/optuna_dbs"
    os.makedirs(db_dir, exist_ok=True)

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    # Set global tensor_dict path (loaded inside objective per trial to avoid pickling issues)
    tensor_dict_path_global = os.path.join(
        str(CODE_DIR), "dataset", "meta-learning-sup-que-ds", "segfilt_rts_tensor_dict.pkl"
    )

    study_name   = f"1s3w_transfer_learning_DeepCNNLSTM_4fcv_{args.eval_mode}_hpo"
    journal_path = os.path.join(db_dir, f"{study_name}.log")

    print(f"Starting HPO Study: {study_name}")
    print(f"Journal Path: {journal_path}")
    print(f"Eval mode: {args.eval_mode}")

    run_study(
        study_name=study_name,
        storage_path=journal_path,
        eval_mode=args.eval_mode,
        n_trials=N_TRIALS,
    )