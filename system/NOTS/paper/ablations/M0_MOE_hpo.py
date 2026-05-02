"""
M0_MOE_hpo.py
=============
Optuna HPO for 1-shot 10-way MAML+MoE.

Searches BOTH MoE structure HPs and the MAML HPs most likely to shift at
10-way (inner_steps, learning rates). Everything else (backbone architecture,
MSL schedule, LSLR config) is fixed.

IMPORTANT — execution model
────────────────────────────
Runs EXACTLY ONE Optuna trial per invocation and exits.
Called from hpo_ablation_launcher.sh as one SLURM array task per trial.
All tasks share a single JournalStorage (.log file) so TPE learns across them.

Environment variables (set by launcher via --export):
    HPO_USE_JOURNAL : "1" = JournalFileBackend (production)
                      "0" = InMemoryStorage (debug — results discarded)
    HPO_DB_DIR      : directory for Optuna journal files
    RUN_DIR         : per-trial output directory
    CODE_DIR        : root of the codebase
    DATA_DIR        : root of the data directory

CLI args:
    --ablation M0_MOE_hpo   (passed by launcher; used for logging only)
    --dry-run               print sampled config without running training

Study name  : moe_hpo_1s10w_v1
Journal file: $HPO_DB_DIR/moe_hpo_1s10w_v1.log

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL SEARCH SPACE REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAML HPs  (re-tuned — old Trial 89 values were for 1s3w, not 1s10w)
─────────────────────────────────────────────────────────────────────
  outer_lr              log-uniform [5e-5, 5e-4]
                        Meta-optimizer learning rate. Controls how fast the
                        meta-initialisation moves on each outer step.

  weight_decay          log-uniform [1e-5, 1e-3]
                        L2 regularisation on outer parameters. Helps prevent
                        meta-overfitting, particularly at 10-way.

  maml_inner_steps      categorical {5, 10, 15, 20, 25}
                        Number of gradient steps in the inner loop (both train
                        AND eval — always matched). More steps = better
                        per-task adaptation but slower training and higher risk
                        of meta-overfitting (the model learns to overfit the
                        support set rather than meta-learn). If you previously
                        saw performance peak at epoch 1 and decay, that was
                        caused by too many inner steps — the meta-gradient was
                        optimising for fitting the support set rather than
                        generalising. 50 is intentionally excluded for this
                        reason. inner_steps_eval is ALWAYS set equal to
                        inner_steps (matched train/eval protocol).

  maml_alpha_init       log-uniform [1e-4, 1e-2]
                        Inner-loop step size during training. Operates in
                        tandem with LSLR (which learns per-parameter per-step
                        multipliers on top of this base rate).

  maml_alpha_init_eval  log-uniform [1e-4, 5e-2]
                        Inner-loop step size at evaluation / fine-tuning time.
                        Slightly wider range than train alpha because test-time
                        adaptation can tolerate a more aggressive step (no
                        meta-gradient to worry about).

MoE structure HPs
─────────────────
  num_experts           categorical {2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32}
                        Number of CNN expert encoders. Encoder-placement MoE
                        forward-pass cost scales linearly with this. Small
                        values (2-8) test whether fewer, more specialised
                        experts may generalise better at 10-way.

  utilization_ratio     categorical {0.1, 0.2, 0.3, 0.4, 0.5}
                        Fraction of experts selected per sample. top_k =
                        max(1, round(num_experts * ratio)). Sampled as a ratio
                        (not top_k directly) so the search space dimension is
                        fixed regardless of num_experts (required for TPE).

  MOE_gate_temperature  log-uniform [0.3, 3.0]
                        Softmax temperature on gate logits. Higher = flatter
                        routing (more expert sharing). Lower = sharper routing
                        (more specialisation). Wider range than 1s3w because
                        10-way may benefit from flatter routing so the harder
                        classification signal gets more expert coverage.

  MOE_use_shared_expert {True, False}
                        Adds an always-on universal expert summed with the
                        weighted mixture. Stabilises early training and may
                        reduce collapse by giving the router less pressure to
                        always use one expert.

  MOE_routing_signal    {linear_proj_input, context_proj}
                        Router architecture. linear_proj_input is simpler
                        (GAP + one linear). context_proj is richer (lightweight
                        CNN + two-layer MLP).

MoE loss HPs
────────────
  MOE_aux_coeff         log-uniform [1e-3, 5e-1]
                        Weight on Switch-style load-balance loss (f_i * P_i).
                        Penalises uneven dispatch counts. Wider upper bound
                        because small expert counts + 10-way may need stronger
                        regularisation to prevent collapse.

  MOE_importance_coeff  log-uniform [1e-3, 5e-1]
                        Weight on Shazeer importance loss (CV² of per-expert
                        soft weight sums). Catches ordinal ranking dominance
                        that the Switch loss misses — the specific failure mode
                        you observed (large dispatch imbalance with near-uniform
                        soft weights).

MoE router capacity HPs
────────────────────────
  MOE_ctx_hidden_dim    categorical {32, 64, 128}
  MOE_ctx_out_dim       categorical {16, 32, 64}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXED (not HPO'd)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  n_way=10, k_shot=1, q_query=3   task definition
  meta_batchsize=24               fixed per spec
  maml_use_lslr=True              always on
  use_lslr_at_eval=False          fixed
  use_maml_msl="hybrid"           MSL annealing always on
  maml_msl_num_epochs=31          fixed
  label_smooth=0.05               unanimous in prior runs
  MOE_placement="encoder"         fixed per spec
  cnn_base_filters, lstm_hidden   fixed backbone
  MOE_dropout=0.05                not a primary driver
  apply_MOE_aux_loss_inner_outer="outer"  correct for MAML

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
META-OVERFITTING SAFEGUARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A trial is pruned if train_acc - val_acc > META_OVERFIT_THRESHOLD at any
epoch after META_OVERFIT_GRACE_EPOCHS. This catches the "epoch 1 peaks then
decays" failure mode without wasting remaining wall time. The gap is logged
every epoch regardless so you can see it in the SLURM output.
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

# ── Path setup ───────────────────────────────────────────────────────────────
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

# New study name — completely separate from the old 1s3w journal.
# If you ever want to re-run 1s3w HPO, it will write to a different file.
STUDY_NAME   = "moe_hpo_1s10w_v1"
JOURNAL_NAME = "moe_hpo_1s10w_v1.log"

# Task definition — the whole point of this script.
HPO_N_WAY   = 10
HPO_K_SHOT  = 1
HPO_Q_QUERY = 3    # 3 query samples per class → 30 total per episode.
                    # Matches roughly the same total query set size as 1s3w
                    # (q_query=9 → 27 total), keeping per-trial wall time
                    # comparable. More would be cleaner signal but 3× slower.

# Minimum expected samples routed to each expert per forward pass.
MIN_SAMPLES_PER_EXPERT = 2

# ── HPO training schedule ─────────────────────────────────────────────────────
HPO_NUM_EPOCHS         = 15    # directional signal; full runs use more
HPO_EPISODES_PER_EPOCH = 300   # reduced for wall-time budget

# ── Collapse detection overrides ──────────────────────────────────────────────
HPO_COLLAPSE_GRACE_EPOCHS    = 5
HPO_COLLAPSE_LOG_EVERY       = 3
HPO_COLLAPSE_ABORT_THRESHOLD = 0.80

# ── Meta-overfitting safeguard ────────────────────────────────────────────────
# Prune if train_acc - val_acc exceeds this after the grace period.
META_OVERFIT_THRESHOLD    = 0.35   # 35 percentage-point train/val gap
META_OVERFIT_GRACE_EPOCHS = 5      # don't prune before this epoch


# ─────────────────────────────────────────────────────────────────────────────
# Storage factory
# ─────────────────────────────────────────────────────────────────────────────

def make_storage(use_journal: bool, hpo_db_dir: str) -> optuna.storages.BaseStorage:
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

def _expected_samples_per_expert(num_experts: int, top_k: int, batch_size: int) -> float:
    return (top_k / num_experts) * batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Search space sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_hparams(trial: optuna.Trial, base_config: dict) -> dict:
    """
    Sample one complete set of hyperparameters for a 1-shot 10-way MAML+MoE run.

    MAML HPs are included in the search because the Trial 89 values were tuned
    for 1s3w and will not transfer reliably to 1s10w.

    top_k is derived from num_experts * utilization_ratio (not sampled directly)
    so the Optuna search space dimension is fixed regardless of num_experts,
    which is required for TPE to build a valid probabilistic model.
    """

    # ── MAML HPs ─────────────────────────────────────────────────────────────

    outer_lr     = trial.suggest_float("outer_lr",     5e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # 50 inner steps intentionally excluded: caused "epoch 1 peaks then decays"
    # meta-overfitting in prior runs. The meta-overfit safeguard would catch it
    # anyway, but excluding it avoids wasting those trial slots entirely.
    inner_steps = trial.suggest_categorical("maml_inner_steps", [5, 10, 15, 20, 25])

    alpha_init      = trial.suggest_float("maml_alpha_init",      1e-4, 1e-2,  log=True)
    alpha_init_eval = trial.suggest_float("maml_alpha_init_eval", 1e-4, 5e-2,  log=True)

    # ── MoE structure HPs ─────────────────────────────────────────────────────

    num_experts = trial.suggest_categorical(
        "num_experts", [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
    )

    ratio = trial.suggest_categorical(
        "utilization_ratio", [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    top_k = max(1, round(num_experts * ratio))

    # ── Constraint: enough samples per expert per batch ───────────────────────
    meta_batchsize = base_config["meta_batchsize"]
    expected = _expected_samples_per_expert(num_experts, top_k, meta_batchsize)
    if expected < MIN_SAMPLES_PER_EXPERT:
        raise optuna.exceptions.TrialPruned(
            f"Constraint violated: E={num_experts}, ratio={ratio}, k={top_k}, "
            f"B={meta_batchsize} -> {expected:.2f} samples/expert/batch "
            f"< {MIN_SAMPLES_PER_EXPERT}"
        )

    # Wider temperature range — 10-way may benefit from flatter routing so the
    # harder classification signal spreads across more experts.
    gate_temp = trial.suggest_float("MOE_gate_temperature", 0.3, 3.0, log=True)

    use_shared = trial.suggest_categorical("MOE_use_shared_expert", [True, False])

    # Demographics option excluded (use_demographics=False in config).
    routing_choices = ["linear_proj_input", "context_proj"]
    if base_config.get("use_demographics", False):
        routing_choices.append("context_proj_with_demo")
    routing_signal = trial.suggest_categorical("MOE_routing_signal", routing_choices)

    # ── MoE loss HPs ──────────────────────────────────────────────────────────
    # Wider upper bound (5e-1 vs 1e-1) — small expert counts + 10-way may need
    # stronger load-balancing to resist collapse.
    aux_coeff = trial.suggest_float("MOE_aux_coeff",        1e-3, 5e-1, log=True)
    imp_coeff = trial.suggest_float("MOE_importance_coeff", 1e-3, 5e-1, log=True)

    # ── Router capacity HPs ───────────────────────────────────────────────────
    ctx_hidden = trial.suggest_categorical("MOE_ctx_hidden_dim", [32, 64, 128])
    ctx_out    = trial.suggest_categorical("MOE_ctx_out_dim",    [16, 32, 64])

    # ── Build config ──────────────────────────────────────────────────────────
    config = copy.deepcopy(base_config)
    config.update({
        # Task — override ablation_config's 1s3w defaults
        "n_way":    HPO_N_WAY,
        "k_shot":   HPO_K_SHOT,
        "q_query":  HPO_Q_QUERY,

        # MAML
        "learning_rate":         outer_lr,
        "weight_decay":          weight_decay,
        "maml_inner_steps":      inner_steps,
        "maml_inner_steps_eval": inner_steps,   # always matched — no exceptions
        "maml_alpha_init":       alpha_init,
        "maml_alpha_init_eval":  alpha_init_eval,

        # MoE structure
        "num_experts":           num_experts,
        "utilization_ratio":     ratio,      # logged only; model reads MOE_top_k
        "MOE_top_k":             top_k,
        "MOE_gate_temperature":  gate_temp,
        "MOE_use_shared_expert": use_shared,
        "MOE_routing_signal":    routing_signal,

        # MoE losses
        "MOE_aux_coeff":         aux_coeff,
        "MOE_importance_coeff":  imp_coeff,

        # Router capacity
        "MOE_ctx_hidden_dim":    ctx_hidden,
        "MOE_ctx_out_dim":       ctx_out,

        # HPO training schedule
        "num_epochs":               HPO_NUM_EPOCHS,
        "episodes_per_epoch_train": HPO_EPISODES_PER_EPOCH,

        # HPO collapse detection overrides (read by mamlpp_pretrain)
        "hpo_collapse_grace_epochs":    HPO_COLLAPSE_GRACE_EPOCHS,
        "hpo_collapse_log_every":       HPO_COLLAPSE_LOG_EVERY,
        "hpo_collapse_abort_threshold": HPO_COLLAPSE_ABORT_THRESHOLD,
    })
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Meta-overfitting check
# ─────────────────────────────────────────────────────────────────────────────

def _check_meta_overfit(train_acc_log: list, val_acc_log: list) -> tuple[bool, float]:
    """
    Returns (is_overfitting, max_gap) where max_gap is the largest
    train_acc - val_acc observed across all epochs after the grace period.

    Uses the max gap (not the most recent) because the failure mode typically
    forms early and persists — checking the max is more robust than a single
    epoch snapshot.
    """
    n = min(len(train_acc_log), len(val_acc_log))
    if n <= META_OVERFIT_GRACE_EPOCHS:
        return False, 0.0

    gaps = [train_acc_log[i] - val_acc_log[i] for i in range(META_OVERFIT_GRACE_EPOCHS, n)]
    max_gap = max(gaps) if gaps else 0.0
    return max_gap > META_OVERFIT_THRESHOLD, max_gap


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, base_config: dict, tensor_dict_path: str) -> float:
    """
    Sample hyperparams, train one MAML++ run, return best val acc.
    Val accuracy (not test accuracy) is the HPO signal; test set is held out.

    Pruning conditions (trial excluded from TPE surrogate model):
      1. (E, k, B) constraint violated at sample time.
      2. MoE collapse confirmed before any val checkpoint was ever recorded.
         (If collapse fires after a valid checkpoint, return that partial signal.)
      3. Meta-overfitting: train_acc - val_acc > META_OVERFIT_THRESHOLD after
         the grace period. These trials wasted compute — the inner loop was
         overfitting the support set rather than meta-learning.
    """
    try:
        config = sample_hparams(trial, base_config)
    except optuna.exceptions.TrialPruned:
        raise

    trial_seed = FIXED_SEED + trial.number
    set_seeds(trial_seed)
    config["seed"] = trial_seed

    print(f"\n[MOE HPO | trial {trial.number}] Hyperparameters:")
    log_keys = [
        "n_way", "k_shot", "q_query",
        "learning_rate", "weight_decay",
        "maml_inner_steps", "maml_alpha_init", "maml_alpha_init_eval",
        "num_experts", "utilization_ratio", "MOE_top_k",
        "MOE_gate_temperature", "MOE_use_shared_expert", "MOE_routing_signal",
        "MOE_aux_coeff", "MOE_importance_coeff",
        "MOE_ctx_hidden_dim", "MOE_ctx_out_dim",
        "num_epochs", "episodes_per_epoch_train",
    ]
    for k in log_keys:
        print(f"  {k}: {config[k]}")

    try:
        model = build_maml_moe_model(config)
    except Exception as e:
        print(f"[MOE HPO | trial {trial.number}] Model build failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Model build failed: {e}")

    try:
        train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)
    except Exception as e:
        print(f"[MOE HPO | trial {trial.number}] Dataloader build failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Dataloader build failed: {e}")

    try:
        from MAML.mamlpp import mamlpp_pretrain
        _trained_model, train_history = mamlpp_pretrain(
            model, config, train_dl, episodic_val_loader=val_dl,
        )
    except Exception as e:
        print(f"[MOE HPO | trial {trial.number}] Training failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Training failed: {e}")

    moe_collapsed = train_history.get("moe_collapsed", False)
    val_acc       = float(train_history["best_val_acc"])

    # ── Check 1: collapse before any val checkpoint ───────────────────────────
    if moe_collapsed and val_acc <= -1.0:
        print(f"[MOE HPO | trial {trial.number}] MoE collapsed before first val "
              f"checkpoint. Pruning.")
        raise optuna.exceptions.TrialPruned(
            f"MoE collapsed (best_val_acc={val_acc:.4f}, never set)."
        )

    # ── Check 2: meta-overfitting ─────────────────────────────────────────────
    train_acc_log = train_history.get("train_acc_log", [])
    val_acc_log   = train_history.get("val_acc_log", [])

    # Always log the per-epoch gap so it's visible in SLURM output.
    n = min(len(train_acc_log), len(val_acc_log))
    if n > 0:
        gaps = [train_acc_log[i] - val_acc_log[i] for i in range(n)]
        print(f"[MOE HPO | trial {trial.number}] Train-val acc gap per epoch: "
              f"{[f'{g:+.3f}' for g in gaps]}")

    is_overfit, max_gap = _check_meta_overfit(train_acc_log, val_acc_log)
    if is_overfit:
        print(f"[MOE HPO | trial {trial.number}] Meta-overfitting detected: "
              f"max train-val gap = {max_gap:.3f} > threshold {META_OVERFIT_THRESHOLD}. "
              f"Pruning (inner_steps={config['maml_inner_steps']}, "
              f"outer_lr={config['learning_rate']:.2e}).")
        raise optuna.exceptions.TrialPruned(
            f"Meta-overfitting: max train-val gap = {max_gap:.3f}."
        )

    # ── Normal return ─────────────────────────────────────────────────────────
    if moe_collapsed:
        print(f"[MOE HPO | trial {trial.number}] MoE collapsed but returning "
              f"partial val_acc={val_acc:.4f} (pre-collapse signal is meaningful).")
    else:
        print(f"[MOE HPO | trial {trial.number}] val_acc = {val_acc:.4f}")

    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Best summary
# ─────────────────────────────────────────────────────────────────────────────

def save_best_summary(study: optuna.Study, output_dir: Path) -> None:
    """
    Write best-trial JSON to output_dir. Called after each trial and on SIGTERM.
    Multiple array tasks write to the same file; last writer wins (safe, since
    each write reflects the current global best from the shared journal).
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
        description="MoE HPO (1-shot 10-way) — one Optuna trial per invocation."
    )
    p.add_argument("--ablation", type=str, default="M0_MOE_hpo",
                   help="Passed by launcher; used for logging only.")
    p.add_argument("--dry-run", action="store_true",
                   help="Sample and print one config without running training.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    use_journal = os.environ.get("HPO_USE_JOURNAL", "1") == "1"
    hpo_db_dir  = os.environ.get("HPO_DB_DIR",
                  str(Path(RUN_DIR).parent.parent / "optuna_dbs"))
    run_dir     = Path(os.environ.get("RUN_DIR", str(Path(RUN_DIR) / "moe_hpo_debug")))

    print(f"\n{'='*60}")
    print(f"[MOE HPO] ablation   : {args.ablation}")
    print(f"[MOE HPO] study      : {STUDY_NAME}")
    print(f"[MOE HPO] journal    : {os.path.join(hpo_db_dir, JOURNAL_NAME)}")
    print(f"[MOE HPO] run_dir    : {run_dir}")
    print(f"[MOE HPO] use_journal: {use_journal}")
    print(f"[MOE HPO] task       : {HPO_K_SHOT}-shot {HPO_N_WAY}-way  "
          f"(q_query={HPO_Q_QUERY}, total query={HPO_Q_QUERY * HPO_N_WAY}/episode)")
    print(f"[MOE HPO] schedule   : {HPO_NUM_EPOCHS} epochs × "
          f"{HPO_EPISODES_PER_EPOCH} episodes/epoch")
    print(f"[MOE HPO] CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[MOE HPO] GPU        : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    base_config = make_base_config(ablation_id="M0")
    tensor_dict_path = os.path.join(
        base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
    )

    storage = make_storage(use_journal, hpo_db_dir)

    sampler = TPESampler(seed=FIXED_SEED)
    study = optuna.create_study(
        study_name     = STUDY_NAME,
        direction      = "maximize",
        sampler        = sampler,
        storage        = storage,
        load_if_exists = True,
    )

    def _sigterm_handler(signum, frame):
        print("\n[MOE HPO] SIGTERM received — writing best summary before SIGKILL.")
        try:
            save_best_summary(study, run_dir)
        except Exception as e:
            print(f"[MOE HPO] Could not write summary on SIGTERM: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    if args.dry_run:
        dummy_trial = study.ask()
        try:
            config = sample_hparams(dummy_trial, base_config)
            print("[DRY RUN] Sampled config:")
            for k in [
                "n_way", "k_shot", "q_query",
                "learning_rate", "weight_decay",
                "maml_inner_steps", "maml_alpha_init", "maml_alpha_init_eval",
                "num_experts", "utilization_ratio", "MOE_top_k",
                "MOE_gate_temperature", "MOE_use_shared_expert", "MOE_routing_signal",
                "MOE_aux_coeff", "MOE_importance_coeff",
                "MOE_ctx_hidden_dim", "MOE_ctx_out_dim",
                "num_epochs", "episodes_per_epoch_train",
            ]:
                print(f"  {k}: {config[k]}")
        except optuna.exceptions.TrialPruned as e:
            print(f"[DRY RUN] Trial pruned during sampling: {e}")
        finally:
            study.tell(dummy_trial, state=optuna.trial.TrialState.FAIL)
        return

    study.optimize(
        lambda trial: objective(trial, base_config, tensor_dict_path),
        n_trials = 1,
        catch    = (RuntimeError, ValueError, torch.cuda.OutOfMemoryError),
    )

    save_best_summary(study, run_dir)
    print(f"\n[MOE HPO] Trial complete. Exiting.")


if __name__ == "__main__":
    main()