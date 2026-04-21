"""
a11_eval_hpo_extended.py
========================
Extended HPO for A11 (Meta pretrained model) eval hyperparameters.

Why this exists
---------------
The original A11 HPO (ablation_hpo.py) searched:
  ft_lr    : [1e-5, 1e-2]  log-uniform   → best was 0.01 (HIT UPPER BOUND)
  ft_steps : {10, 25, 50, 100}           → best was 100  (HIT UPPER BOUND)

Both hyperparameters hit the boundary of the search space, meaning the true
optimum was never found. This script re-runs the search with a wider range.

This script is COMPLETELY SEPARATE from ablation_hpo.py:
  - Different study name   : ablation_A11_eval_hpo_v2
  - Different journal file : ablation_A11_eval_hpo_v2.log
  - No modifications to the existing A11 v1 study

The model backbone is FIXED (Meta's pretrained weights). We are only searching
for the best adaptation HPs — this is not training HPO.

New search space
----------------
  ft_lr    : [1e-4, 1.0]  log-uniform
              For a frozen backbone with only a linear head, high LRs are
              perfectly reasonable. 1.0 is not unusual for head-only tuning
              with a small support set (1-shot, 3-way = 3 support examples).
              We include 1e-4 as the lower bound to cover the possibility that
              the lower end of the original range was also suboptimal.

  ft_steps : {50, 100, 150, 200, 250, 300}
              Start from 50 — we already know below 50 is not optimal.
              300 is generous but cheap (head-only FT is fast).

Usage (one trial per job — driven by SLURM array via a11_eval_hpo_launcher.sh):
  python a11_eval_hpo_extended.py

  # With explicit overrides:
  python a11_eval_hpo_extended.py --n_trials 1 --hpo_db_dir /path/to/dbs

SLURM
-----
Use a11_eval_hpo_launcher.sh to submit an array job.
Each trial takes ~5-15 min (head-only FT is fast).
Recommend 40-60 total trials.
"""

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend

# =============================================================================
# Environment / paths  (same pattern as ablation_hpo.py)
# =============================================================================

FIXED_SEED = 42
N_TRIALS   = int(os.environ.get("N_TRIALS", 1))

CODE_DIR   = Path(os.environ.get("CODE_DIR",   "./")).resolve()
DATA_DIR   = Path(os.environ.get("DATA_DIR",   "./data")).resolve()
RUN_DIR    = Path(os.environ.get("RUN_DIR",    "./")).resolve()
HPO_DB_DIR = Path(os.environ.get("HPO_DB_DIR", "/scratch/my13/kai/meta-pers-gest/optuna_dbs")).resolve()

for _p in [
    CODE_DIR,
    CODE_DIR / "system",
    CODE_DIR / "system" / "MAML",
    CODE_DIR / "system" / "MOE",
    CODE_DIR / "system" / "pretraining",
]:
    sys.path.insert(0, str(_p))

# Meta repo path for DiscreteGesturesArchitecture
NEUROMOTOR_REPO = Path(os.environ.get(
    "NEUROMOTOR_REPO",
    "/projects/my13/div-emg/generic-neuromotor-interface"
)).resolve()
sys.path.insert(0, str(NEUROMOTOR_REPO))

RUN_DIR.mkdir(parents=True, exist_ok=True)
HPO_DB_DIR.mkdir(parents=True, exist_ok=True)

print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"CODE_DIR       : {CODE_DIR}")
print(f"HPO_DB_DIR     : {HPO_DB_DIR}")
print(f"RUN_DIR        : {RUN_DIR}")

# =============================================================================
# Constants — must match ablation_hpo.py exactly
# =============================================================================

META_CHECKPOINT_PATH = Path("/rhf/allocations/my13/emg_models/discrete_gestures/model_checkpoint.ckpt")
EMG_2KHZ_PKL_PATH    = "/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/segfilt_2khz_emg_tensor_dict.pkl"
EMG_2KHZ_IN_CH       = 16
EMG_2KHZ_SEQ_LEN     = 4300

USER_SPLIT_JSON = (CODE_DIR / "system" / "fixed_user_splits"
                   / "4kfcv_splits_shared_test.json")
FOLD_IDX = 0

# Optuna study name — different from v1 so there is zero collision risk
STUDY_NAME = "ablation_A11_eval_hpo_v2"

# =============================================================================
# Data split loading (identical to ablation_hpo.py)
# =============================================================================

with open(USER_SPLIT_JSON, "r") as f:
    ALL_SPLITS = json.load(f)


def apply_fold_to_config(config: dict) -> dict:
    split = ALL_SPLITS[FOLD_IDX]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    return config


# =============================================================================
# Optuna objective
# =============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Objective for A11 extended eval HPO.

    Extended search space:
      ft_lr    : [1e-4, 1.0]  log-uniform  (previous upper bound was 1e-2)
      ft_steps : {50, 100, 150, 200, 250, 300}  (previous max was 100)

    Everything else is identical to _objective_a11 in ablation_hpo.py.
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from pretraining.pretrain_finetune import finetune_and_eval_user
    from torch.utils.data import DataLoader

    trial_start = time.time()
    print("=" * 80)
    print(f"[A11-v2 | Trial {trial.number}] Starting")
    print("=" * 80)

    # ── Suggest extended HPs ──────────────────────────────────────────────────
    ft_lr    = trial.suggest_float("ft_lr",    1e-4, 1.0, log=True)
    ft_steps = trial.suggest_categorical("ft_steps", [50, 100, 150, 200, 250, 300])

    print(f"  ft_lr={ft_lr:.2e}, ft_steps={ft_steps}")

    # ── Minimal config ────────────────────────────────────────────────────────
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
        "use_imu":                False,
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

    # ── Load Meta model ───────────────────────────────────────────────────────
    from A10_A11_A12_meta_pretrained import MetaEMGWrapper
    model = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
    model.to(device)

    # ── Load tensor_dict ──────────────────────────────────────────────────────
    with open(EMG_2KHZ_PKL_PATH, "rb") as f:
        full_dict = pickle.load(f)
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
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=maml_mm_collate)

    user_metrics: dict = defaultdict(list)
    for batch in val_dl:
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

    elapsed = time.time() - trial_start
    trial.set_user_attr("ft_lr",              ft_lr)
    trial.set_user_attr("ft_steps",           ft_steps)
    trial.set_user_attr("per_user_val_accs",  per_user_means)

    print(f"[A11-v2 | Trial {trial.number}] ft_lr={ft_lr:.2e}, ft_steps={ft_steps} "
          f"→ val acc={mean_acc*100:.2f}%  ({elapsed:.1f}s)")
    return mean_acc


# =============================================================================
# Study runner
# =============================================================================

def run_study(n_trials: int = 1) -> optuna.Study:
    # Stagger workers to avoid journal write collisions at startup
    sleep_time = random.uniform(0, 10)
    print(f"Staggering start: sleeping {sleep_time:.2f}s …")
    time.sleep(sleep_time)

    use_journal  = int(os.environ.get("HPO_USE_JOURNAL", "1"))
    journal_path = HPO_DB_DIR / f"{STUDY_NAME}.log"

    if use_journal:
        lock_obj = JournalFileBackend(str(journal_path))
        storage  = JournalStorage(lock_obj)
        print(f"Journal storage : {journal_path}")
    else:
        storage = optuna.storages.InMemoryStorage()
        print("InMemoryStorage (debug mode — results not persisted).")

    time.sleep(random.uniform(0, 10))

    # Per-worker TPE seed to prevent duplicate configs in concurrent array jobs
    _task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    _worker_seed = int(
        hashlib.md5(f"{_task_id}_{FIXED_SEED}".encode()).hexdigest()[:8], 16
    )
    print(f"TPE worker seed : {_worker_seed} (task_id={_task_id})")

    # Warm-start: seed the study with the best known config from v1 HPO.
    # ft_lr=0.01 and ft_steps=100 were the v1 optima (both boundary-hitting),
    # so enqueuing them as trial 0 ensures we evaluate them in the v2 study
    # for a direct apples-to-apples comparison, then let TPE explore beyond.
    WARM_START = [
        {"ft_lr": 0.01,  "ft_steps": 100},   # v1 best — boundary, include for reference
        {"ft_lr": 0.1,   "ft_steps": 150},   # one step beyond v1 boundary in both dims
        {"ft_lr": 0.01,  "ft_steps": 200},   # same LR, more steps
        {"ft_lr": 0.1,   "ft_steps": 100},   # higher LR, same steps as v1 best
    ]

    sampler = optuna.samplers.TPESampler(
        seed             = _worker_seed,
        n_startup_trials = max(10, len(WARM_START)),
        n_ei_candidates  = 24,
        multivariate     = True,
    )

    study = optuna.create_study(
        study_name    = STUDY_NAME,
        direction     = "maximize",
        storage       = storage,
        load_if_exists= True,
        sampler       = sampler,
    )

    if len(study.trials) == 0:
        print(f"Warm-starting study with {len(WARM_START)} trials …")
        for i, params in enumerate(WARM_START):
            study.enqueue_trial(params)
            print(f"  Enqueued warm-start {i}: {params}")
    else:
        print(f"Study already has {len(study.trials)} trials — skipping warm-start.")

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extended A11 eval HPO (wider ft_lr and ft_steps ranges)."
    )
    parser.add_argument(
        "--n_trials", type=int, default=N_TRIALS,
        help="Number of Optuna trials to run in this job. Default: N_TRIALS env var.",
    )
    parser.add_argument(
        "--hpo_db_dir", type=str, default=None,
        help="Override HPO_DB_DIR env var.",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Override RUN_DIR env var.",
    )
    args = parser.parse_args()

    if args.hpo_db_dir:
        HPO_DB_DIR = Path(args.hpo_db_dir).resolve()
        HPO_DB_DIR.mkdir(parents=True, exist_ok=True)
    if args.out_dir:
        RUN_DIR = Path(args.out_dir).resolve()
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    print(f"\n{'='*70}")
    print(f"  A11 Extended Eval HPO  (study: {STUDY_NAME})")
    print(f"  N_TRIALS   : {args.n_trials}")
    print(f"  HPO_DB_DIR : {HPO_DB_DIR}")
    print(f"  RUN_DIR    : {RUN_DIR}")
    print(f"  Search space:")
    print(f"    ft_lr    : [1e-4, 1.0]  log-uniform")
    print(f"    ft_steps : {{50, 100, 150, 200, 250, 300}}")
    print(f"{'='*70}\n")

    run_study(n_trials=args.n_trials)