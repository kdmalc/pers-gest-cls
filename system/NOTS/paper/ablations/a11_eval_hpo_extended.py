"""
A11_eval_hpo_extended.py
========================
Extended HPO + paper curve for A11 (Meta pretrained model) eval HPs.

Two modes
---------
1. HPO mode  (default, driven by SLURM array via eval_hp_launchers.sh A11)
   Runs Optuna TPE over ft_lr and ft_steps with wider ranges than v1.
   Study name: ablation_A11_eval_hpo_v2  (different from v1 — zero collision)

2. Paper curve mode  (--paper-curve --ft-lr <best>)
   Fixes ft_lr to the best value from HPO and sweeps ft_steps over
   PAPER_STEPS_GRID = {1, 3, 5, 10, 15, 25, 50, 100, 150, 200}.

Why this exists
---------------
The original A11 HPO (_objective_a11 in ablation_hpo.py) searched:
  ft_lr    : [1e-5, 1e-2]      -> best was 0.01  (HIT UPPER BOUND)
  ft_steps : {10, 25, 50, 100} -> best was 100   (HIT UPPER BOUND)
Both hit the search boundary. True optimum was never found.

New HPO search space
--------------------
  ft_lr    : [1e-4, 1.0]  log-uniform
  ft_steps : {50, 100, 150, 200, 250, 300}

Paper curve step grid (for figure)
-----------------------------------
  PAPER_STEPS_GRID = {1, 3, 5, 10, 15, 25, 50, 100, 150, 200}

Workflow
--------
  # Step 1: extended HPO (50 trials recommended)
  bash eval_hp_launchers.sh A11 --n-trials 50

  # Step 2: inspect study, note best ft_lr, generate paper curve
  bash eval_hp_launchers.sh A11_CURVE --ft-lr <best_ft_lr>

SLURM
-----
HPO mode    : array job, ~10-15 min per trial
Paper curve : single job, ~25 min total
"""

import argparse
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
# Environment / paths
# =============================================================================

FIXED_SEED = 42
N_TRIALS   = int(os.environ.get("N_TRIALS", 1))

CODE_DIR   = Path(os.environ.get("CODE_DIR",   "./")).resolve()
DATA_DIR   = Path(os.environ.get("DATA_DIR",   "./data")).resolve()
RUN_DIR    = Path(os.environ.get("RUN_DIR",    "./")).resolve()
HPO_DB_DIR = Path(os.environ.get(
    "HPO_DB_DIR", "/scratch/my13/kai/meta-pers-gest/optuna_dbs"
)).resolve()

for _p in [
    CODE_DIR,
    CODE_DIR / "system",
    CODE_DIR / "system" / "MAML",
    CODE_DIR / "system" / "MOE",
    CODE_DIR / "system" / "pretraining",
]:
    sys.path.insert(0, str(_p))

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
# Constants
# =============================================================================

META_CHECKPOINT_PATH = Path(
    "/rhf/allocations/my13/emg_models/discrete_gestures/model_checkpoint.ckpt"
)
EMG_2KHZ_PKL_PATH = (
    "/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/"
    "meta-learning-sup-que-ds/segfilt_2khz_emg_tensor_dict.pkl"
)
EMG_2KHZ_IN_CH   = 16
EMG_2KHZ_SEQ_LEN = 4300

USER_SPLIT_JSON = (
    CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"
)
FOLD_IDX = 0

STUDY_NAME = "ablation_A11_eval_hpo_v2"

# Paper figure step grid. Includes low step counts to show the full
# adaptation trajectory and sample-efficiency story vs M0.
PAPER_STEPS_GRID = [1, 3, 5, 10, 15, 25, 50, 100, 150, 200]

NUM_VAL_EPISODES = 200
DEFAULT_VAL_PIDS = ["P011", "P006", "P105", "P109"]

# =============================================================================
# Data split loading
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
# Shared config builder
# =============================================================================

def _build_a11_config(ft_lr: float, ft_steps: int) -> dict:
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
    return config


# =============================================================================
# Shared eval logic
# =============================================================================

def _eval_a11(config: dict) -> tuple[float, list[float]]:
    """
    Run episodic val eval for A11 at the ft_lr and ft_steps in config.
    Returns (mean_val_acc, per_user_means).
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from pretraining.pretrain_finetune import finetune_and_eval_user
    from torch.utils.data import DataLoader
    from A10_A11_A12_meta_pretrained import MetaEMGWrapper

    device = config["device"]
    model  = MetaEMGWrapper(META_CHECKPOINT_PATH, freeze_backbone=True)
    model.to(device)

    with open(EMG_2KHZ_PKL_PATH, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = config["val_PIDs"],
        target_gesture_classes  = config["maml_gesture_classes"],
        target_trial_indices    = config["target_trial_indices"],
        n_way                   = config["n_way"],
        k_shot                  = config["k_shot"],
        q_query                 = config.get("q_query", None),
        num_eval_episodes       = config["num_eval_episodes"],
        is_train                = False,
        seed                    = FIXED_SEED,
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
            support_emg    = support["emg"],   support_imu=None,
            support_labels = support["labels"],
            query_emg      = query["emg"],     query_imu=None,
            query_labels   = query["labels"],
            mode           = "head_only",
        )
        user_metrics[uid].append(metrics["acc"])

    per_user_means = [float(np.mean(accs)) for accs in user_metrics.values()]
    return float(np.nanmean(per_user_means)), per_user_means


# =============================================================================
# Optuna objective (HPO mode)
# =============================================================================

def objective(trial: optuna.Trial) -> float:
    trial_start = time.time()
    print("=" * 80)
    print(f"[A11-v2 | Trial {trial.number}] Starting")
    print("=" * 80)

    ft_lr    = trial.suggest_float("ft_lr",    1e-4, 1.0, log=True)
    ft_steps = trial.suggest_categorical("ft_steps", [50, 100, 150, 200, 250, 300])

    print(f"  ft_lr={ft_lr:.2e}, ft_steps={ft_steps}")

    config              = _build_a11_config(ft_lr, ft_steps)
    mean_acc, per_user  = _eval_a11(config)

    elapsed = time.time() - trial_start
    trial.set_user_attr("ft_lr",             ft_lr)
    trial.set_user_attr("ft_steps",          ft_steps)
    trial.set_user_attr("per_user_val_accs", per_user)

    print(f"[A11-v2 | Trial {trial.number}] ft_lr={ft_lr:.2e}, ft_steps={ft_steps} "
          f"-> val acc={mean_acc*100:.2f}%  ({elapsed:.1f}s)")
    return mean_acc


# =============================================================================
# Paper curve runner
# =============================================================================

def run_paper_curve(
    ft_lr:            float,
    out_dir:          Path,
    val_pids:         list[str],
    num_val_episodes: int,
) -> None:
    """
    Sweep ft_steps over PAPER_STEPS_GRID at fixed ft_lr.
    Writes partial results after each step count — preemption-safe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    n_configs = len(PAPER_STEPS_GRID)

    print(f"\nA11 PAPER CURVE: {n_configs} step values at fixed ft_lr={ft_lr:.2e}")
    print(f"Steps grid : {PAPER_STEPS_GRID}")
    print(f"Output dir : {out_dir}")
    print()

    results     = []
    best_acc    = -1.0
    best_steps  = None
    sweep_start = time.time()

    for i, ft_steps in enumerate(PAPER_STEPS_GRID):
        t0 = time.time()
        print(f"[{i+1:>2}/{n_configs}] ft_steps={ft_steps:>4} ...", end="", flush=True)

        config = _build_a11_config(ft_lr, ft_steps)
        config["num_eval_episodes"] = num_val_episodes
        config["val_PIDs"]          = val_pids

        mean_acc, per_user = _eval_a11(config)
        elapsed = time.time() - t0

        print(f"  acc={mean_acc*100:.2f}%  ({elapsed:.1f}s)")
        results.append({
            "ft_steps":      ft_steps,
            "ft_lr":         ft_lr,
            "mean_acc":      mean_acc,
            "per_user_accs": per_user,
        })

        if mean_acc > best_acc:
            best_acc   = mean_acc
            best_steps = ft_steps

        partial_output = {
            "ablation_id":          "A11",
            "mode":                 "paper_curve",
            "ft_lr":                ft_lr,
            "steps_grid":           PAPER_STEPS_GRID,
            "val_pids":             val_pids,
            "num_val_episodes":     num_val_episodes,
            "n_configs_total":      n_configs,
            "n_configs_done":       i + 1,
            "best_steps_so_far":    best_steps,
            "best_mean_acc_so_far": best_acc,
            "results":              results,
        }
        out_path = out_dir / f"A11_paper_curve_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(partial_output, f, indent=2)

    total_elapsed = time.time() - sweep_start
    print(f"\n{'='*70}")
    print(f"[A11] Paper curve complete in {total_elapsed/60:.1f} min")
    print(f"  Best steps : {best_steps}")
    print(f"  Best acc   : {best_acc*100:.2f}%")
    print(f"  Results    : {out_path}")
    print(f"{'='*70}")

    print(f"\nNext step — run M0 paper curve and plot both on the same axes.")


# =============================================================================
# HPO study runner
# =============================================================================

def run_study(n_trials: int = 1) -> optuna.Study:
    sleep_time = random.uniform(0, 10)
    print(f"Staggering start: sleeping {sleep_time:.2f}s ...")
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

    _task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    _worker_seed = int(
        hashlib.md5(f"{_task_id}_{FIXED_SEED}".encode()).hexdigest()[:8], 16
    )
    print(f"TPE worker seed : {_worker_seed} (task_id={_task_id})")

    WARM_START = [
        {"ft_lr": 0.01,  "ft_steps": 100},   # v1 best — boundary, reference
        {"ft_lr": 0.1,   "ft_steps": 150},   # beyond v1 in both dims
        {"ft_lr": 0.01,  "ft_steps": 200},   # same LR, more steps
        {"ft_lr": 0.1,   "ft_steps": 100},   # higher LR, same steps as v1
    ]

    sampler = optuna.samplers.TPESampler(
        seed             = _worker_seed,
        n_startup_trials = max(10, len(WARM_START)),
        n_ei_candidates  = 24,
        multivariate     = True,
    )

    study = optuna.create_study(
        study_name     = STUDY_NAME,
        direction      = "maximize",
        storage        = storage,
        load_if_exists = True,
        sampler        = sampler,
    )

    if len(study.trials) == 0:
        print(f"Warm-starting with {len(WARM_START)} trials ...")
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
        description="Extended A11 eval HPO and paper curve generator."
    )
    parser.add_argument("--n-trials",    type=int, default=N_TRIALS, dest="n_trials")
    parser.add_argument("--hpo-db-dir",  type=str, default=None,     dest="hpo_db_dir")
    parser.add_argument("--out-dir",     type=str, default=None,     dest="out_dir")
    parser.add_argument("--val-pids",    type=str, nargs="+",
                        default=DEFAULT_VAL_PIDS, dest="val_pids")
    parser.add_argument("--num-val-episodes", type=int,
                        default=NUM_VAL_EPISODES, dest="num_val_episodes")

    parser.add_argument(
        "--paper-curve", action="store_true", dest="paper_curve",
        help="Generate paper figure curve at fixed --ft-lr. Run after HPO.",
    )
    parser.add_argument(
        "--ft-lr", type=float, default=None, dest="ft_lr",
        help="Fixed ft_lr for --paper-curve. Take from study.best_trial.",
    )

    args = parser.parse_args()

    if args.hpo_db_dir:
        HPO_DB_DIR = Path(args.hpo_db_dir).resolve()
        HPO_DB_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else RUN_DIR

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    if args.paper_curve:
        assert args.ft_lr is not None, \
            "--ft-lr is required for --paper-curve."
        print(f"\n{'='*70}")
        print(f"  A11 Paper Curve")
        print(f"  ft_lr      : {args.ft_lr:.2e}")
        print(f"  Steps grid : {PAPER_STEPS_GRID}")
        print(f"  Val PIDs   : {args.val_pids}")
        print(f"  Out dir    : {out_dir}")
        print(f"{'='*70}\n")
        run_paper_curve(
            ft_lr            = args.ft_lr,
            out_dir          = out_dir,
            val_pids         = args.val_pids,
            num_val_episodes = args.num_val_episodes,
        )
    else:
        print(f"\n{'='*70}")
        print(f"  A11 Extended Eval HPO  (study: {STUDY_NAME})")
        print(f"  N_TRIALS   : {args.n_trials}")
        print(f"  HPO_DB_DIR : {HPO_DB_DIR}")
        print(f"  Search space:")
        print(f"    ft_lr    : [1e-4, 1.0]  log-uniform")
        print(f"    ft_steps : {{50, 100, 150, 200, 250, 300}}")
        print(f"{'='*70}\n")
        run_study(n_trials=args.n_trials)