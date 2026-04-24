"""
MOE_hpo.py
==========
Optuna-based HPO sweep over the MoE architecture and loss hyperparameters.

Sweeps the following axes (see SEARCH SPACE section below):
  - num_experts            : number of experts {4, 8, 16}
  - MOE_top_k              : routing sparsity {1, 2, 4, None (dense)}
  - MOE_routing_signal     : router design {linear_proj_input, context_proj,
                                            context_proj_with_demo}
  - MOE_gate_temperature   : softmax temperature {0.5, 1.0, 2.0}
  - MOE_use_shared_expert  : always-on universal expert {True, False}
  - MOE_aux_coeff          : Switch / KL load-balance loss weight
  - MOE_importance_coeff   : Shazeer importance loss weight (top-k only)
  - MOE_ctx_hidden_dim     : context projector hidden dim (routing signal
                             dependent)
  - MOE_ctx_out_dim        : routing vector dim (routing signal dependent)

Constraint enforced during sampling:
  "Expected samples per expert per batch" ≥ MIN_SAMPLES_PER_EXPERT_PER_BATCH.
  This prevents degenerate configurations (e.g., E=16, k=1) where individual
  experts receive essentially no gradient signal per update step.
  See _expected_samples_per_expert() for the formula.

Usage:
    python MOE_hpo.py                     # 200 trials, default config
    python MOE_hpo.py --n-trials 50       # quick exploratory run
    python MOE_hpo.py --study-name my_run # named study (resumes if exists)

Results are saved to:
    {RUN_DIR}/hpo/moe_hpo_results.json     # full trial log
    {RUN_DIR}/hpo/moe_hpo_best.json        # best trial config + score
"""

import os, sys, copy, json, argparse, time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))

sys.path.append(os.path.join(os.path.dirname(__file__), 'test_eval_files'))
from ablation_config import (
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HPO_DIR = Path(RUN_DIR) / "hpo"
HPO_DIR.mkdir(parents=True, exist_ok=True)

# Minimum expected samples per expert per batch.
# If a (E, k, batch_size) combination would route fewer than this many samples
# to each expert on average per training step, the trial is rejected during
# sampling.  This prevents pathological sparsity at small data scale.
# Formula: (k / E) * batch_size >= MIN_SAMPLES_PER_EXPERT_PER_BATCH
# With batch_size ~32 and MIN=2: E=16,k=1 → (1/16)*32=2 (just passes)
#                                 E=16,k=1 batch_size=16 → 1 (rejected)
# Adjust MIN_SAMPLES_PER_EXPERT_PER_BATCH to match your actual episodic batch size.
MIN_SAMPLES_PER_EXPERT_PER_BATCH = 2


# ─────────────────────────────────────────────────────────────────────────────
# Constraint helper
# ─────────────────────────────────────────────────────────────────────────────

def _expected_samples_per_expert(num_experts: int,
                                  top_k,
                                  batch_size: int) -> float:
    """
    Expected number of samples routed to each expert per batch step.

    For dense routing (top_k=None) all experts always receive all samples
    (as weighted inputs), so this constraint is vacuous — return infinity.
    For top-k: each sample activates k experts, so on average each expert
    receives (k / E) * B samples per step.

    Args:
        num_experts : E
        top_k       : k (int) or None for dense
        batch_size  : B

    Returns:
        float — expected samples per expert per batch (inf for dense).
    """
    if top_k is None:
        return float("inf")
    return (top_k / num_experts) * batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Search space sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_moe_hparams(trial: optuna.Trial, base_config: dict) -> dict:
    """
    Sample a single set of MoE hyperparameters and merge with base_config.

    Raises optuna.exceptions.TrialPruned if the sampled combination violates
    the minimum samples-per-expert constraint.  Optuna counts pruned trials
    against the trial budget, so this will reduce effective trial count
    slightly if many degenerate combos are sampled; with TPE this stabilises
    quickly after the first ~20 trials.

    SEARCH SPACE
    ────────────
    num_experts : {4, 8, 16}
        24 experts (previous best HPO result) was too many for your data scale.
        The observed 1300x dispatch imbalance is a direct consequence of having
        more "slots" than natural gesture clusters, giving the router too many
        near-equivalent choices and making collapse almost inevitable.

    MOE_top_k : {1, 2, 4, None}
        None = dense / soft routing (all experts weighted each step).
        The constraint filter will reject (E=4, k=1) only if batch_size < 8,
        so k=1 is viable with E=4 or E=8 at normal batch sizes.
        Note: k=1 with MAML can cause gradient instability; if you observe
        this, restrict to k >= 2 or use dense routing.

    MOE_routing_signal : {"linear_proj_input", "context_proj",
                          "context_proj_with_demo"}
        "linear_proj_input"      : GAP raw input → single Linear.
                                   Cheapest; closest to Shazeer/Switch standard.
        "context_proj"           : CNN → GAP → MLP (original default).
        "context_proj_with_demo" : context_proj + demographics embedding.
                                   Only sample this if use_demographics=True
                                   in your config; if not, it is excluded.

    MOE_gate_temperature : {0.5, 1.0, 2.0}
        < 1 : sharper routing (more winner-takes-all)
        > 1 : flatter routing (more uniform expert usage)
        High temperature is a soft alternative to the importance loss for
        preventing collapse; the two interact and both are worth sweeping.

    MOE_use_shared_expert : {True, False}
        Always-on universal expert (DeepSeek-V2 style).

    MOE_aux_coeff : log-uniform in [1e-3, 1e-1]
        Weight for Switch-style / KL load-balance loss.
        If collapse persists even with importance loss, try pushing this higher.

    MOE_importance_coeff : log-uniform in [1e-3, 1e-1]
        Weight for Shazeer importance loss (top-k only; ignored for dense).
        Usually set equal to or slightly higher than aux_coeff.

    MOE_ctx_hidden_dim : {32, 64, 128}
        Hidden dim of the context projector MLP.
        Only meaningful for "context_proj" and "context_proj_with_demo".
        For "linear_proj_input" this value is ignored.

    MOE_ctx_out_dim : {16, 32, 64}
        Routing vector dimensionality (input to MOEGate.linear).
        Larger = more routing expressivity; smaller = fewer router params.
    """
    # ── Core MoE structure ───────────────────────────────────────────────────
    num_experts = trial.suggest_categorical("num_experts", [4, 8, 16])

    top_k_choices = [1, 2, 4, None]
    # Only offer top-k values that are < num_experts (dense is always valid)
    top_k_choices = [k for k in top_k_choices if k is None or k < num_experts]
    top_k = trial.suggest_categorical("MOE_top_k", top_k_choices)

    # ── Constraint: minimum samples per expert per batch ─────────────────────
    episodic_batch_size = base_config.get("episode_batch_size",
                          base_config.get("batch_size", 32))
    expected = _expected_samples_per_expert(num_experts, top_k, episodic_batch_size)
    if expected < MIN_SAMPLES_PER_EXPERT_PER_BATCH:
        raise optuna.exceptions.TrialPruned(
            f"Rejected: E={num_experts}, k={top_k}, B={episodic_batch_size} → "
            f"{expected:.2f} samples/expert/batch < {MIN_SAMPLES_PER_EXPERT_PER_BATCH}"
        )

    # ── Routing signal ───────────────────────────────────────────────────────
    use_demographics = base_config.get("use_demographics", False)
    routing_signal_choices = ["linear_proj_input", "context_proj"]
    if use_demographics:
        routing_signal_choices.append("context_proj_with_demo")
    routing_signal = trial.suggest_categorical(
        "MOE_routing_signal", routing_signal_choices
    )

    # ── Gate temperature ─────────────────────────────────────────────────────
    gate_temp = trial.suggest_categorical("MOE_gate_temperature", [0.5, 1.0, 2.0])

    # ── Shared expert ────────────────────────────────────────────────────────
    use_shared = trial.suggest_categorical("MOE_use_shared_expert", [True, False])

    # ── Aux loss weights ─────────────────────────────────────────────────────
    aux_coeff = trial.suggest_float("MOE_aux_coeff", 1e-3, 1e-1, log=True)
    # Importance loss is only meaningful for top-k routing.
    # For dense routing we still sample it (it's ignored in compute_moe_aux_loss),
    # so Optuna can track the parameter across trials uniformly.
    imp_coeff = trial.suggest_float("MOE_importance_coeff", 1e-3, 1e-1, log=True)

    # ── Routing module capacity ──────────────────────────────────────────────
    # Only meaningfully used by context_proj variants; harmless to sample for
    # linear_proj_input (those params won't be instantiated for that router).
    ctx_hidden = trial.suggest_categorical("MOE_ctx_hidden_dim", [32, 64, 128])
    ctx_out    = trial.suggest_categorical("MOE_ctx_out_dim",    [16, 32, 64])

    # ── Merge into config ────────────────────────────────────────────────────
    hparams = {
        "num_experts":           num_experts,
        "MOE_top_k":             top_k,
        "MOE_routing_signal":    routing_signal,
        "MOE_gate_temperature":  gate_temp,
        "MOE_use_shared_expert": use_shared,
        "MOE_aux_coeff":         aux_coeff,
        "MOE_importance_coeff":  imp_coeff,
        "MOE_ctx_hidden_dim":    ctx_hidden,
        "MOE_ctx_out_dim":       ctx_out,
    }
    config = copy.deepcopy(base_config)
    config.update(hparams)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, base_config: dict,
              tensor_dict_path: str) -> float:
    """
    Optuna objective.  Returns mean validation accuracy (to maximise).

    We use validation accuracy (not test accuracy) as the HPO signal.
    Test set is held out until final model selection.

    Training uses MAML++ on the train_PIDs, and evaluation uses episodic
    val accuracy over val_PIDs (same episodic protocol as the main runs).
    """
    try:
        config = sample_moe_hparams(trial, base_config)
    except optuna.exceptions.TrialPruned:
        raise   # let Optuna handle it

    trial_seed = FIXED_SEED + trial.number
    set_seeds(trial_seed)
    config["seed"] = trial_seed

    print(f"\n[HPO Trial {trial.number}] Params:")
    hpo_keys = [
        "num_experts", "MOE_top_k", "MOE_routing_signal",
        "MOE_gate_temperature", "MOE_use_shared_expert",
        "MOE_aux_coeff", "MOE_importance_coeff",
        "MOE_ctx_hidden_dim", "MOE_ctx_out_dim",
    ]
    for k in hpo_keys:
        print(f"  {k}: {config[k]}")

    try:
        model = build_maml_moe_model(config)
    except Exception as e:
        print(f"[HPO Trial {trial.number}] Model build failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Model build failed: {e}")

    try:
        train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)
    except Exception as e:
        print(f"[HPO Trial {trial.number}] Dataloader build failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Dataloader build failed: {e}")

    try:
        from MAML.mamlpp import mamlpp_pretrain
        trained_model, train_history = mamlpp_pretrain(
            model, config, train_dl, episodic_val_loader=val_dl,
        )
        val_acc = float(train_history["best_val_acc"])
    except Exception as e:
        print(f"[HPO Trial {trial.number}] Training failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Training failed: {e}")

    print(f"[HPO Trial {trial.number}] val_acc = {val_acc:.4f}")
    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_hpo_results(study: optuna.Study, output_dir: Path) -> None:
    """Save full trial log and best-trial summary to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full trial log
    trial_log = []
    for t in study.trials:
        trial_log.append({
            "number":   t.number,
            "value":    t.value,
            "state":    str(t.state),
            "params":   t.params,
            "duration": t.duration.total_seconds() if t.duration else None,
        })
    log_path = output_dir / "moe_hpo_results.json"
    with open(log_path, "w") as f:
        json.dump(trial_log, f, indent=2)
    print(f"[HPO] Full trial log saved to {log_path}")

    # Best trial summary
    if study.best_trial is not None:
        best = {
            "best_trial_number": study.best_trial.number,
            "best_val_acc":      study.best_value,
            "best_params":       study.best_params,
        }
        best_path = output_dir / "moe_hpo_best.json"
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"[HPO] Best trial saved to {best_path}")
        print(f"[HPO] Best val acc: {study.best_value:.4f}")
        print(f"[HPO] Best params:  {study.best_params}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MoE HPO sweep")
    p.add_argument("--n-trials",    type=int,   default=200,
                   help="Number of Optuna trials (default: 200)")
    p.add_argument("--study-name",  type=str,   default="moe_hpo",
                   help="Optuna study name (used for resuming)")
    p.add_argument("--storage",     type=str,   default=None,
                   help="Optuna storage URL for distributed HPO "
                        "(e.g. sqlite:///hpo.db). Default: in-memory.")
    p.add_argument("--timeout",     type=float, default=None,
                   help="Maximum total wall time in seconds (optional).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Base config (same as M0 but we will override MoE params per trial) ──
    base_config = make_base_config(ablation_id="M0_hpo")
    # HPO uses val_PIDs for evaluation, NOT test_PIDs.
    # test_PIDs must remain completely held out during HPO.
    print("\n[HPO] Base config:")
    print(f"  train_PIDs : {base_config.get('train_PIDs')}")
    print(f"  val_PIDs   : {base_config.get('val_PIDs')}")
    print(f"  test_PIDs  : {base_config.get('test_PIDs')} (HELD OUT — not touched)")

    tensor_dict_path = os.path.join(
        base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
    )

    print(f"\n[HPO] Starting study '{args.study_name}' with {args.n_trials} trials.")
    print(f"[HPO] Results will be saved to {HPO_DIR}")
    print(f"[HPO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[HPO] GPU: {torch.cuda.get_device_name(0)}")

    # ── Create or resume study ───────────────────────────────────────────────
    sampler = TPESampler(seed=FIXED_SEED)
    study = optuna.create_study(
        study_name     = args.study_name,
        direction      = "maximize",
        sampler        = sampler,
        storage        = args.storage,
        load_if_exists = True,   # resume if study already exists in storage
    )

    # ── Run optimisation ─────────────────────────────────────────────────────
    study.optimize(
        lambda trial: objective(trial, base_config, tensor_dict_path),
        n_trials  = args.n_trials,
        timeout   = args.timeout,
        # catch common recoverable errors so HPO continues past a bad trial
        catch     = (RuntimeError, ValueError, torch.cuda.OutOfMemoryError),
        show_progress_bar = True,
    )

    # ── Save results ─────────────────────────────────────────────────────────
    save_hpo_results(study, HPO_DIR)

    # ── Print summary ────────────────────────────────────────────────────────
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL]

    print(f"\n{'='*70}")
    print(f"[HPO] Study complete.")
    print(f"  Completed trials : {len(completed)}")
    print(f"  Pruned trials    : {len(pruned)}")
    print(f"  Failed trials    : {len(failed)}")
    if study.best_trial is not None:
        print(f"\n  Best val acc : {study.best_value:.4f}")
        print(f"  Best params  :")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()