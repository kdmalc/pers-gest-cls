"""
M0_full_model.py
================
Ablation M0: Full Model — MAML + MoE  [PRIMARY RESULT]

Supports two test procedures via --test-procedure CLI arg (or config default):
  'hpo_test_split' : Fixed 24/4/4 split, single run at FIXED_SEED.
                     Used during HPO / early development.
  'L2SO'           : Leave-2-Subjects-Out over all N subjects.
                     For fold i: test=subjects[i], val=subjects[(i+1) % N].
                     One run per fold, paired t-test across folds for stats.

Parallelism:
  L2SO folds are submitted as SEPARATE SLURM jobs by eval_launcher.sh.
  Each job passes --fold-idx <i> to run exactly one fold.
  If --fold-idx is NOT passed, all folds run sequentially (legacy / local dev).

Training : Episodic dataloader
Evaluation: Episodic (1-shot 3-way), 500 episodes over test_PIDs
Reported  : single result (hpo_test_split) or mean ± std across folds (L2SO)
"""

import os, sys, copy, json, argparse
import numpy as np
import torch

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    make_periodic_checkpoint_fn, make_periodic_test_eval_fn,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0: Full Model (MAML + MoE)")
    parser.add_argument(
        "--test-procedure",
        choices=["hpo_test_split", "L2SO"],
        default=None,
        help=(
            "Override config['test_procedure']. "
            "'hpo_test_split' = fixed 24/4/4 split, single run (debugging/HPO). "
            "'L2SO' = Leave-2-Subjects-Out, one fold per subject (default for final results)."
        ),
    )
    parser.add_argument(
        "--fold-idx",
        type=int,
        default=None,
        help=(
            "If set, run only this single L2SO fold index (0-based) and exit. "
            "Used by eval_launcher.sh to run each fold as a separate SLURM job. "
            "Ignored when --test-procedure is hpo_test_split."
        ),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = make_base_config(ablation_id="M0")
    # Apply CLI override if provided; otherwise the config default (L2SO) stands.
    if args.test_procedure is not None:
        config["test_procedure"] = args.test_procedure
    return config


def run_one_fold(fold_id: str, seed: int, config: dict) -> dict:
    """
    Train and evaluate a single fold. config must already have
    train_PIDs, val_PIDs, test_PIDs set correctly before calling.
    """
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[M0 | {fold_id} | seed={seed}] Parameters: {n_params:,}")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    from MAML.mamlpp import mamlpp_pretrain
    checkpoint_fn = make_periodic_checkpoint_fn(config)
    test_eval_fn  = make_periodic_test_eval_fn(tensor_dict_path, config["test_PIDs"])

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
        periodic_checkpoint_fn=checkpoint_fn,
        periodic_test_eval_fn=test_eval_fn,
        checkpoint_every=10,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[M0 | {fold_id} | seed={seed}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "fold_id":           fold_id,
            "seed":              seed,
            "model_state_dict":  train_history["best_state"],
            "config":            config,
            "best_val_acc":      best_val_acc,
            "train_loss_log":    train_history["train_loss_log"],
            "val_acc_log":       train_history["val_acc_log"],
        },
        config,
        tag=f"{fold_id}_seed{seed}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[M0 | {fold_id} | seed={seed}] Test acc: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")

    return {
        "fold_id":      fold_id,
        "seed":         seed,
        "test_PID":     config["test_PIDs"],
        "val_PID":      config["val_PIDs"],
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test procedure: fixed HPO split (single run)
# ─────────────────────────────────────────────────────────────────────────────

def run_hpo_test_split(config: dict) -> dict:
    """Single run at FIXED_SEED on the fixed train/val/test split."""
    print(f"\n{'='*70}")
    print(f"[M0] hpo_test_split: single run (seed={FIXED_SEED})")
    print(f"{'='*70}")
    # PIDs are already set in make_base_config for the fixed split
    return run_one_fold(
        fold_id=f"fixed_seed{FIXED_SEED}",
        seed=FIXED_SEED,
        config=copy.deepcopy(config),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test procedure: Leave-2-Subjects-Out
# ─────────────────────────────────────────────────────────────────────────────

def build_l2so_folds(all_pids: list) -> list:
    """
    For fold i:
      test subject   = all_pids[i]
      val subject    = all_pids[(i + 1) % N]   (round-robin, deterministic)
      train subjects = everyone else
    """
    n = len(all_pids)
    folds = []
    for i in range(n):
        test_pid   = all_pids[i]
        val_pid    = all_pids[(i + 1) % n]
        train_pids = [p for p in all_pids if p != test_pid and p != val_pid]
        folds.append({
            "fold_idx":   i,
            "test_pid":   test_pid,
            "val_pid":    val_pid,
            "train_pids": train_pids,
        })
    return folds


def run_single_l2so_fold(config: dict, fold: dict) -> dict:
    """Run and save results for a single L2SO fold dict (as built by build_l2so_folds)."""
    fold_idx = fold["fold_idx"]
    print(f"\n{'='*70}")
    print(f"[M0] L2SO fold {fold_idx}  "
          f"test={fold['test_pid']}  val={fold['val_pid']}")
    print(f"{'='*70}")

    fold_config = copy.deepcopy(config)
    fold_config["train_PIDs"] = fold["train_pids"]
    fold_config["val_PIDs"]   = [fold["val_pid"]]
    fold_config["test_PIDs"]  = [fold["test_pid"]]

    result = run_one_fold(
        fold_id=f"l2so_fold{fold_idx:02d}_test{fold['test_pid']}",
        seed=FIXED_SEED,
        config=fold_config,
    )
    return result


def run_l2so_all_folds(config: dict) -> list:
    """
    L2SO procedure — runs ALL folds sequentially.
    Only used when --fold-idx is NOT passed (local dev / legacy).
    For cluster runs, the launcher submits one job per fold via --fold-idx.
    """
    all_pids = config["all_PIDs"]
    assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."
    folds = build_l2so_folds(all_pids)
    return [run_single_l2so_fold(config, fold) for fold in folds]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = build_config(args)
    test_procedure = config["test_procedure"]

    print("\nM0 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"\nTest procedure : {test_procedure}")
    if args.fold_idx is not None:
        print(f"Fold index     : {args.fold_idx}  (single-fold SLURM job)")

    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'. "
        f"Must be 'hpo_test_split' or 'L2SO'."
    )

    # ── hpo_test_split: single run, ignore --fold-idx ─────────────────────────
    if test_procedure == "hpo_test_split":
        if args.fold_idx is not None:
            print(f"[M0] WARNING: --fold-idx {args.fold_idx} is ignored for hpo_test_split.")
        result = run_hpo_test_split(config)
        summary = {
            "ablation_id":     "M0",
            "description":     "Full Model: MAML + MoE",
            "test_procedure":  "hpo_test_split",
            "seed":            FIXED_SEED,
            "n_params":        result["n_params"],
            "result":          result,
            "test_acc":        result["test_results"]["mean_acc"],
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }
        save_results(summary, config, tag="summary")

        print(f"\n{'='*70}")
        print(f"[M0] FINAL (hpo_test_split): {summary['test_acc']*100:.2f}%  "
              f"single run, seed={FIXED_SEED}")
        print(f"     {config['n_way']}-way {config['k_shot']}-shot")
        print(f"{'='*70}")

    # ── L2SO: one fold per SLURM job (preferred) or all sequential (fallback) ──
    else:
        all_pids = config["all_PIDs"]
        assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."
        folds = build_l2so_folds(all_pids)

        if args.fold_idx is not None:
            # ── Single-fold path: used by eval_launcher.sh ─────────────────────
            assert 0 <= args.fold_idx < len(folds), (
                f"--fold-idx {args.fold_idx} is out of range for {len(folds)} subjects "
                f"(valid: 0 – {len(folds)-1})."
            )
            fold   = folds[args.fold_idx]
            result = run_single_l2so_fold(config, fold)
            summary = {
                "ablation_id":    "M0",
                "description":    "Full Model: MAML + MoE",
                "test_procedure": "L2SO",
                "fold_idx":       args.fold_idx,
                "seed":           FIXED_SEED,
                "n_params":       result["n_params"],
                "fold_result":    result,
                "test_acc":       result["test_results"]["mean_acc"],
                "config_snapshot": {k: str(v) for k, v in config.items()},
            }
            save_results(summary, config, tag=f"fold{args.fold_idx:02d}_summary")

            print(f"\n{'='*70}")
            print(f"[M0] FOLD {args.fold_idx} RESULT (L2SO): "
                  f"{result['test_results']['mean_acc']*100:.2f}%  "
                  f"test={fold['test_pid']}  seed={FIXED_SEED}")
            print(f"     {config['n_way']}-way {config['k_shot']}-shot")
            print(f"{'='*70}")

        else:
            # ── All-folds path: sequential fallback for local dev ───────────────
            print("[M0] WARNING: --fold-idx not set. Running all L2SO folds sequentially.")
            print("     On the cluster, prefer submitting one job per fold via eval_launcher.sh.")
            all_results = run_l2so_all_folds(config)
            test_accs   = [r["test_results"]["mean_acc"] for r in all_results]
            summary = {
                "ablation_id":     "M0",
                "description":     "Full Model: MAML + MoE",
                "test_procedure":  "L2SO",
                "seed":            FIXED_SEED,
                "n_params":        all_results[0]["n_params"],
                "fold_results":    all_results,
                "mean_test_acc":   float(np.mean(test_accs)),
                "std_test_acc":    float(np.std(test_accs)),
                "num_folds":       len(all_results),
                "config_snapshot": {k: str(v) for k, v in config.items()},
            }
            save_results(summary, config, tag="summary")

            print(f"\n{'='*70}")
            print(f"[M0] FINAL (L2SO, all folds sequential): "
                  f"{summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
            print(f"     over {summary['num_folds']} L2SO folds (one per test subject)")
            print(f"     {config['n_way']}-way {config['k_shot']}-shot")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()