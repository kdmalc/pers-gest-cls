"""
fewshot_grid.py
===============
Few-Shot Grid Sweep: M0 model trained and evaluated at each (k_shot, n_way) combo.

Grid:
  k_shot : [1, 3, 5]
  n_way  : [3, 5, 10]
  -> 9 jobs total (one per cell), all run in parallel via the launcher.

This script runs a SINGLE (k_shot, n_way) pair per invocation, specified via
--k-shot and --n-way. The eval launcher submits one job per grid cell.

Why train-per-cell rather than train-once-eval-many:
  MAML's inner loop is optimized for the (k, n) regime it trains in. A model
  trained at 1-shot 3-way is not meaningfully comparable to one trained at
  5-shot 10-way when tested in a different regime. Each cell is its own model.

Note: The (k=1, n=3) cell is identical to M0 by construction. It is included
here for grid completeness and as a self-consistency check.

All hyperparameters are fixed at M0 best values (Trial 89). This is not HPO.

Training : Episodic dataloader
Evaluation: Episodic, 500 episodes over test_PIDs
Seed     : FIXED_SEED (single run per cell, consistent with other ablations)

Output: one results JSON per cell, saved to RUN_DIR, tagged with k{K}_n{N}.
        Aggregate offline into the grid table / heatmap figure.
"""

import argparse
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
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import mamlpp_pretrain

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Keep in sync with GRID_K_SHOTS / GRID_N_WAYS in eval_ablation_launcher.sh.
GRID_K_SHOTS = [1, 3, 5]
GRID_N_WAYS  = [3, 5, 10]

# q_query is held fixed across all grid cells (standard practice).
# Changing k_shot does NOT change the number of query samples.
Q_QUERY_FIXED = 9


def build_config(k_shot: int, n_way: int) -> dict:
    config = make_base_config(ablation_id=f"grid_k{k_shot}_n{n_way}")

    config["k_shot"]  = k_shot
    config["n_way"]   = n_way
    config["q_query"] = Q_QUERY_FIXED

    # n_way also controls the MAML head size at training time, so the model
    # built from this config will have an n_way-class head during meta-training.
    # This is intentional: each cell trains its own model in the correct regime.

    print(f"[grid] k_shot={k_shot}  n_way={n_way}  q_query={Q_QUERY_FIXED}")
    return config


def run(k_shot: int, n_way: int) -> dict:
    config = build_config(k_shot, n_way)
    set_seeds(FIXED_SEED)
    config["seed"] = FIXED_SEED

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[grid k={k_shot} n={n_way} | seed={FIXED_SEED}] Parameters: {n_params:,}")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[grid k={k_shot} n={n_way} | seed={FIXED_SEED}] Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "k_shot":            k_shot,
            "n_way":             n_way,
            "seed":              FIXED_SEED,
            "model_state_dict":  train_history["best_state"],
            "config":            config,
            "best_val_acc":      best_val_acc,
            "train_loss_log":    train_history["train_loss_log"],
            "val_acc_log":       train_history["val_acc_log"],
        },
        config,
        tag=f"k{k_shot}_n{n_way}_seed{FIXED_SEED}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[grid k={k_shot} n={n_way} | seed={FIXED_SEED}] Test acc: "
          f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")

    result = {
        "ablation_id":     f"grid_k{k_shot}_n{n_way}",
        "description":     f"Few-Shot Grid: k_shot={k_shot}, n_way={n_way}",
        "k_shot":          k_shot,
        "n_way":           n_way,
        "q_query":         Q_QUERY_FIXED,
        "seed":            FIXED_SEED,
        "best_val_acc":    float(best_val_acc),
        "test_results":    test_results,
        "n_params":        n_params,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"k{k_shot}_n{n_way}_final")

    print(f"\n{'='*70}")
    print(f"[grid] FINAL k={k_shot} n={n_way}: {test_results['mean_acc']*100:.2f}%")
    print(f"  q_query={Q_QUERY_FIXED}  seed={FIXED_SEED}")
    print(f"{'='*70}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot grid sweep (M0) — one (k_shot, n_way) cell per job."
    )
    parser.add_argument("--k-shot", type=int, required=True,
                        help=f"Number of support shots. Must be one of {GRID_K_SHOTS}.")
    parser.add_argument("--n-way", type=int, required=True,
                        help=f"Number of classes. Must be one of {GRID_N_WAYS}.")
    args = parser.parse_args()

    assert args.k_shot in GRID_K_SHOTS, (
        f"--k-shot {args.k_shot} is not in the defined grid {GRID_K_SHOTS}. "
        f"Update GRID_K_SHOTS in both fewshot_grid.py and eval_ablation_launcher.sh "
        f"if you want to add a new value."
    )
    assert args.n_way in GRID_N_WAYS, (
        f"--n-way {args.n_way} is not in the defined grid {GRID_N_WAYS}. "
        f"Update GRID_N_WAYS in both fewshot_grid.py and eval_ablation_launcher.sh "
        f"if you want to add a new value."
    )

    run(args.k_shot, args.n_way)


if __name__ == "__main__":
    main()