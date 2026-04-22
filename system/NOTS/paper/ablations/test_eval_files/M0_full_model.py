"""
M0_full_model.py
================
Ablation M0: Full Model — MAML + MoE  [PRIMARY RESULT]

Supports two test procedures via config['test_procedure']:
  'hpo_test_split' : Fixed 24/4/4 split (fold 0 only), multi-seed loop.
                     Used during HPO / early development.
  'L2SO'           : Leave-2-Subjects-Out over all N subjects.
                     For fold i: test=subjects[i], val=subjects[(i+1) % N].
                     One run per fold, paired t-test across folds for stats.

Training : Episodic dataloader
Evaluation: Episodic (1-shot 3-way), 500 episodes over test_PIDs
Reported  : mean ± std across seeds (hpo_test_split) or folds (L2SO)
"""

import os, sys, copy, json
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
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def build_config() -> dict:
    config = make_base_config(ablation_id="M0")
    # No changes — M0 is the full model with all defaults.
    return config


def run_one_fold(fold_id: str, seed: int, config: dict) -> dict:
    """
    Train and evaluate a single fold/seed combination.
    config must already have train_PIDs, val_PIDs, test_PIDs set
    correctly for this fold before calling this function.
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
    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
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
# Test procedure: fixed HPO split (multi-seed, same 24/4/4 split)
# ─────────────────────────────────────────────────────────────────────────────

def run_hpo_test_split(config: dict) -> list:
    """
    Original fixed-split procedure. Loops over NUM_FINAL_SEEDS seeds,
    each training from scratch on the same train/val/test split.
    """
    results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[M0] hpo_test_split: seed {seed_idx+1}/{NUM_FINAL_SEEDS} (seed={actual_seed})")
        print(f"{'='*70}")
        fold_config = copy.deepcopy(config)
        # PIDs are already set in make_base_config for fold 0
        result = run_one_fold(
            fold_id=f"fixed_seed{actual_seed}",
            seed=actual_seed,
            config=fold_config,
        )
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test procedure: Leave-2-Subjects-Out
# ─────────────────────────────────────────────────────────────────────────────

def build_l2so_folds(all_pids: list) -> list:
    """
    For fold i:
      test subject  = all_pids[i]
      val subject   = all_pids[(i + 1) % N]   (round-robin, deterministic)
      train subjects = everyone else

    Returns a list of dicts: [{test_pid, val_pid, train_pids}, ...]
    """
    n = len(all_pids)
    folds = []
    for i in range(n):
        test_pid  = all_pids[i]
        val_pid   = all_pids[(i + 1) % n]
        train_pids = [p for p in all_pids if p != test_pid and p != val_pid]
        folds.append({
            "fold_idx":   i,
            "test_pid":   test_pid,
            "val_pid":    val_pid,
            "train_pids": train_pids,
        })
    return folds


def run_l2so(config: dict) -> list:
    """
    L2SO procedure. One training run per fold (no seed loop).
    Statistical power comes from the N subject-level test accuracies.
    """
    all_pids = config["all_PIDs"]
    assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."

    folds = build_l2so_folds(all_pids)
    results = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        print(f"\n{'='*70}")
        print(f"[M0] L2SO fold {fold_idx+1}/{len(folds)}  "
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
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = build_config()
    test_procedure = config["test_procedure"]

    print("\nM0 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"\nTest procedure: {test_procedure}")

    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'. "
        f"Must be 'hpo_test_split' or 'L2SO'."
    )

    if test_procedure == "hpo_test_split":
        all_results = run_hpo_test_split(config)
        test_accs = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":     "M0",
            "description":     "Full Model: MAML + MoE",
            "test_procedure":  "hpo_test_split",
            "n_params":        all_results[0]["n_params"],
            "fold_results":    all_results,
            "mean_test_acc":   float(np.mean(test_accs)),
            "std_test_acc":    float(np.std(test_accs)),
            "num_seeds":       NUM_FINAL_SEEDS,
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }

    else:  # L2SO
        all_results = run_l2so(config)
        # One accuracy per test subject — this is the distribution for your paired t-test
        test_accs = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":     "M0",
            "description":     "Full Model: MAML + MoE",
            "test_procedure":  "L2SO",
            "n_params":        all_results[0]["n_params"],
            "fold_results":    all_results,
            "mean_test_acc":   float(np.mean(test_accs)),
            "std_test_acc":    float(np.std(test_accs)),
            "num_folds":       len(all_results),
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }

    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[M0] FINAL ({test_procedure}): "
          f"{summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
    if test_procedure == "L2SO":
        print(f"     over {summary['num_folds']} L2SO folds (one per test subject)")
    else:
        print(f"     over {summary['num_seeds']} seeds, fixed split")
    print(f"     {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()