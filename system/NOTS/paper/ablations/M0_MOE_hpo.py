"""
M0_MOE_hpo.py
==========
Optuna HPO for the MoE architecture and loss hyperparameters.

IMPORTANT — execution model
────────────────────────────
This script runs EXACTLY ONE Optuna trial per invocation and then exits.
It is designed to be called from hpo_ablation_launcher.sh as one task in a
SLURM array job, where each array task is one trial.  All tasks share a
single JournalStorage (.log file) so TPE learns from completed trials as the
array progresses.

This mirrors the exact pattern used by ablation_hpo.py for all other ablations.

Environment variables consumed (set by the launcher via --export):
    HPO_USE_JOURNAL : "1" = use JournalFileBackend (production, default)
                      "0" = use InMemoryStorage (debug, results discarded)
    HPO_DB_DIR      : path to directory for Optuna journal files
    RUN_DIR         : per-trial output directory (set by launcher per task)
    CODE_DIR        : root of the codebase
    DATA_DIR        : root of the data directory
    N_TRIALS        : always "1" when called from launcher (1 trial per task)

CLI args:
    --ablation M0_MOE_hpo   (must be passed; the launcher always passes --ablation)
    --dry-run            print sampled config without training

Study name  : moe_hpo_1s3w_hpo_v1
Journal file: $HPO_DB_DIR/moe_hpo_1s3w_hpo_v1.log

Scavenge / preemption safety
─────────────────────────────
- JournalFileBackend writes each completed trial result atomically.
  A killed task loses its own (incomplete) trial only; all previously
  completed trials in the journal are safe.
- SIGTERM handler writes a best-summary JSON before SIGKILL arrives.
  HPC schedulers typically give 30-60s between SIGTERM and SIGKILL.
- Resubmitting the array job continues from the existing journal automatically
  via load_if_exists=True.

Search space
────────────
  num_experts            : {4, 8, 12, 16, 20, 24, 28, 32, 36, 40}
  utilization_ratio      : {0.1, 0.2, 0.3, 0.4, 0.5}
                           top_k = max(1, round(num_experts * ratio))
                           Both utilization_ratio and effective top_k are
                           stored in the trial config so downstream code that
                           reads config["top_k"] or config["MOE_top_k"] works
                           without modification.
  MOE_routing_signal     : {linear_proj_input, context_proj,
                            context_proj_with_demo}  (last only if use_demographics)
  MOE_gate_temperature   : {0.5, 1.0, 2.0}
  MOE_use_shared_expert  : {True, False}
  MOE_aux_coeff          : log-uniform [1e-3, 1e-1]
  MOE_importance_coeff   : log-uniform [1e-3, 1e-1]
  MOE_ctx_hidden_dim     : {32, 64, 128}
  MOE_ctx_out_dim        : {16, 32, 64}

Constraint: expected samples per expert per batch >= MIN_SAMPLES_PER_EXPERT.
Violated trials are pruned (not failed) so TPE avoids that region going forward.
The constraint uses meta_batchsize (number of MAML episodes per outer step),
which is the operationally correct batch size for expert load — not the flat
supervised batch_size.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.storages.journal import JournalStorage, JournalFileBackend
import torch

# ── Path setup (mirrors ablation_hpo.py) ────────────────────────────────────
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))

sys.path.append(os.path.join(os.path.dirname(__file__), 'test_eval_files'))
from ablation_config import (
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STUDY_NAME   = "moe_hpo_1s3w_hpo_v1"
JOURNAL_NAME = "moe_hpo_1s3w_hpo_v1.log"

# Minimum expected samples routed to each expert per batch step.
# (k / E) * meta_batchsize >= this threshold.
# meta_batchsize is the number of MAML episodes per outer step — the
# operationally correct batch size for expert load (not the flat batch_size).
MIN_SAMPLES_PER_EXPERT = 2


# ─────────────────────────────────────────────────────────────────────────────
# Storage factory  (mirrors ablation_hpo.py pattern exactly)
# ─────────────────────────────────────────────────────────────────────────────

def make_storage(use_journal: bool,
                 hpo_db_dir: str) -> optuna.storages.BaseStorage:
    """
    Production  (use_journal=True):
        JournalStorage backed by a shared .log file in hpo_db_dir.
        All array tasks append to the same file; JournalFileBackend handles
        concurrent writes safely via file locking.  Survives preemption.

    Debug       (use_journal=False):
        InMemoryStorage — results discarded when the process exits.
        Used when HPO_USE_JOURNAL=0 (launcher --debug mode).
    """
    if use_journal:
        Path(hpo_db_dir).mkdir(parents=True, exist_ok=True)
        journal_path = os.path.join(hpo_db_dir, JOURNAL_NAME)
        backend = JournalFileBackend(journal_path)
        print(f"[MOE HPO] JournalStorage: {journal_path}")
        return JournalStorage(backend)
    else:
        print("[MOE HPO] InMemoryStorage (debug — results will not be saved)")
        return optuna.storages.InMemoryStorage()


# ─────────────────────────────────────────────────────────────────────────────
# Constraint helper
# ─────────────────────────────────────────────────────────────────────────────

def _expected_samples_per_expert(num_experts: int,
                                  top_k: int,
                                  batch_size: int) -> float:
    """
    Average number of samples routed to each expert per forward pass.
    top_k is always an int here (dense routing is not offered in this search space).
    """
    return (top_k / num_experts) * batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Search space sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_moe_hparams(trial: optuna.Trial, base_config: dict) -> dict:
    """
    Sample one set of MoE hyperparameters and merge with base_config.
    Raises optuna.exceptions.TrialPruned if the (E, k, B) combination
    violates the minimum-samples-per-expert constraint.

    top_k is derived from num_experts * utilization_ratio rather than sampled
    directly, so the search space is fixed and consistent across all trials
    (required for TPE to build a valid probabilistic model).

    Both "utilization_ratio" and the effective "top_k" / "MOE_top_k" are
    written into the returned config so all downstream code that reads
    config["top_k"] or config["MOE_top_k"] works without any changes.
    """
    # ── Core structure ───────────────────────────────────────────────────────
    num_experts = trial.suggest_categorical(
        "num_experts", [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    )

    # Sample utilization ratio; top_k is derived, not sampled directly.
    # This keeps the Optuna search space fixed regardless of num_experts,
    # which is required for TPE to function correctly.
    ratio = trial.suggest_categorical(
        "utilization_ratio", [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    top_k = max(1, round(num_experts * ratio))

    # ── Constraint ───────────────────────────────────────────────────────────
    # Use meta_batchsize: number of MAML episodes per outer step.
    # This is the correct batch size for expert load — each episode is one
    # forward pass through the MoE, so meta_batchsize determines how many
    # samples are distributed across experts per step.
    meta_batchsize = base_config["meta_batchsize"]
    expected = _expected_samples_per_expert(num_experts, top_k, meta_batchsize)
    if expected < MIN_SAMPLES_PER_EXPERT:
        raise optuna.exceptions.TrialPruned(
            f"Constraint violated: E={num_experts}, ratio={ratio}, "
            f"k={top_k}, B={meta_batchsize} "
            f"-> {expected:.2f} samples/expert/batch < {MIN_SAMPLES_PER_EXPERT}"
        )

    # ── Routing signal ───────────────────────────────────────────────────────
    routing_choices = ["linear_proj_input", "context_proj"]
    if base_config["use_demographics"]:
        routing_choices.append("context_proj_with_demo")
    routing_signal = trial.suggest_categorical(
        "MOE_routing_signal", routing_choices
    )

    # ── Gate temperature ─────────────────────────────────────────────────────
    gate_temp = trial.suggest_categorical(
        "MOE_gate_temperature", [0.5, 1.0, 2.0]
    )

    # ── Shared expert ────────────────────────────────────────────────────────
    use_shared = trial.suggest_categorical(
        "MOE_use_shared_expert", [True, False]
    )

    # ── Aux loss weights ─────────────────────────────────────────────────────
    aux_coeff = trial.suggest_float("MOE_aux_coeff", 1e-3, 1e-1, log=True)
    imp_coeff = trial.suggest_float("MOE_importance_coeff", 1e-3, 1e-1, log=True)

    # ── Router capacity ──────────────────────────────────────────────────────
    ctx_hidden = trial.suggest_categorical("MOE_ctx_hidden_dim", [32, 64, 128])
    ctx_out    = trial.suggest_categorical("MOE_ctx_out_dim",    [16, 32, 64])

    # ── Merge ────────────────────────────────────────────────────────────────
    config = copy.deepcopy(base_config)
    config.update({
        "num_experts":           num_experts,
        "utilization_ratio":     ratio,   # logged for analysis; not read by model
        "MOE_top_k":             top_k,   # effective integer top_k
        "top_k":                 top_k,   # alias kept in sync (ablation_config sets both)
        "MOE_routing_signal":    routing_signal,
        "MOE_gate_temperature":  gate_temp,
        "MOE_use_shared_expert": use_shared,
        "MOE_aux_coeff":         aux_coeff,
        "MOE_importance_coeff":  imp_coeff,
        "MOE_ctx_hidden_dim":    ctx_hidden,
        "MOE_ctx_out_dim":       ctx_out,
    })
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, base_config: dict,
              tensor_dict_path: str) -> float:
    """
    Sample hyperparams, train one MAML++ run, return best val acc.
    Val accuracy (not test accuracy) is the HPO signal; test set is held out.
    """
    try:
        config = sample_moe_hparams(trial, base_config)
    except optuna.exceptions.TrialPruned:
        raise

    trial_seed = FIXED_SEED + trial.number
    set_seeds(trial_seed)
    config["seed"] = trial_seed

    print(f"\n[MOE HPO | trial {trial.number}] Hyperparameters:")
    for k in ["num_experts", "utilization_ratio", "MOE_top_k", "MOE_routing_signal",
              "MOE_gate_temperature", "MOE_use_shared_expert",
              "MOE_aux_coeff", "MOE_importance_coeff",
              "MOE_ctx_hidden_dim", "MOE_ctx_out_dim"]:
        print(f"  {k}: {config[k]}")

    try:
        model = build_maml_moe_model(config)
    except Exception as e:
        print(f"[MOE HPO | trial {trial.number}] Model build failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Model build failed: {e}")

    try:
        train_dl, val_dl = get_maml_dataloaders(
            config, tensor_dict_path=tensor_dict_path
        )
    except Exception as e:
        print(f"[MOE HPO | trial {trial.number}] Dataloader build failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Dataloader build failed: {e}")

    try:
        from MAML.mamlpp import mamlpp_pretrain
        _trained_model, train_history = mamlpp_pretrain(
            model, config, train_dl, episodic_val_loader=val_dl,
        )
        val_acc = float(train_history["best_val_acc"])
    except Exception as e:
        print(f"[MOE HPO | trial {trial.number}] Training failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Training failed: {e}")

    print(f"[MOE HPO | trial {trial.number}] val_acc = {val_acc:.4f}")
    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Best summary  (written after every completed trial)
# ─────────────────────────────────────────────────────────────────────────────

def save_best_summary(study: optuna.Study, output_dir: Path) -> None:
    """
    Write a best-trial JSON to output_dir.  Called after the trial completes
    and also by the SIGTERM handler so you always have an up-to-date snapshot.
    Multiple array tasks write to the same file; the last writer wins, which
    is fine since each write is the current global best from the shared journal.
    """
    if study.best_trial is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    best = {
        "study_name":        study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_val_acc":      study.best_value,
        "best_params":       study.best_params,
    }
    summary_path = output_dir / "moe_hpo_best.json"
    with open(summary_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[MOE HPO] Best summary -> {summary_path}")
    print(f"[MOE HPO] Best val acc so far : {study.best_value:.4f}")
    print(f"[MOE HPO] Best params so far  : {study.best_params}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoE HPO — runs exactly one Optuna trial per invocation."
    )
    p.add_argument(
        "--ablation", type=str, default="M0_MOE_hpo",
        help="Passed by the launcher; used for logging only. "
             "Study name and journal path are fixed constants in this script.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Sample and print one config without running training.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Environment (set by launcher via --export) ───────────────────────────
    use_journal = os.environ.get("HPO_USE_JOURNAL", "1") == "1"
    hpo_db_dir  = os.environ.get("HPO_DB_DIR",
                  str(Path(RUN_DIR).parent.parent / "optuna_dbs"))
    run_dir     = Path(os.environ.get(
        "RUN_DIR", str(Path(RUN_DIR) / "moe_hpo_debug")
    ))

    print(f"\n{'='*60}")
    print(f"[MOE HPO] ablation   : {args.ablation}")
    print(f"[MOE HPO] study      : {STUDY_NAME}")
    print(f"[MOE HPO] journal    : {os.path.join(hpo_db_dir, JOURNAL_NAME)}")
    print(f"[MOE HPO] run_dir    : {run_dir}")
    print(f"[MOE HPO] use_journal: {use_journal}")
    print(f"[MOE HPO] CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[MOE HPO] GPU        : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # ── Base config ──────────────────────────────────────────────────────────
    base_config = make_base_config(ablation_id="M0")
    tensor_dict_path = os.path.join(
        base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
    )

    # ── Storage ──────────────────────────────────────────────────────────────
    storage = make_storage(use_journal, hpo_db_dir)

    # ── Study ────────────────────────────────────────────────────────────────
    # load_if_exists=True: all 500 array tasks share one study; each appends
    # exactly one trial to the shared journal.  TPE automatically benefits
    # from earlier completed trials when sampling later ones.
    sampler = TPESampler(seed=FIXED_SEED)
    study = optuna.create_study(
        study_name     = STUDY_NAME,
        direction      = "maximize",
        sampler        = sampler,
        storage        = storage,
        load_if_exists = True,
    )

    # ── SIGTERM handler ───────────────────────────────────────────────────────
    # Scavenge sends SIGTERM before SIGKILL (typically 30-60s gap on NOTS).
    # We use the window to write the best-summary JSON.
    # The current trial's weights are not recoverable if mid-training —
    # that is unavoidable.  All completed trials in the journal are already safe.
    def _sigterm_handler(signum, frame):
        print("\n[MOE HPO] SIGTERM received — writing best summary before SIGKILL.")
        try:
            save_best_summary(study, run_dir)
        except Exception as e:
            print(f"[MOE HPO] Could not write summary on SIGTERM: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        dummy_trial = study.ask()
        try:
            config = sample_moe_hparams(dummy_trial, base_config)
            print("[DRY RUN] Sampled config:")
            for k in ["num_experts", "utilization_ratio", "MOE_top_k", "top_k",
                      "MOE_routing_signal", "MOE_gate_temperature",
                      "MOE_use_shared_expert", "MOE_aux_coeff",
                      "MOE_importance_coeff", "MOE_ctx_hidden_dim", "MOE_ctx_out_dim"]:
                print(f"  {k}: {config[k]}")
        except optuna.exceptions.TrialPruned as e:
            print(f"[DRY RUN] Trial pruned during sampling: {e}")
        finally:
            study.tell(dummy_trial, state=optuna.trial.TrialState.FAIL)
        return

    # ── Run exactly one trial ─────────────────────────────────────────────────
    study.optimize(
        lambda trial: objective(trial, base_config, tensor_dict_path),
        n_trials = 1,
        catch    = (RuntimeError, ValueError, torch.cuda.OutOfMemoryError),
    )

    # ── Write summary after trial completes ───────────────────────────────────
    save_best_summary(study, run_dir)
    print(f"\n[MOE HPO] Trial complete. Exiting.")


if __name__ == "__main__":
    main()