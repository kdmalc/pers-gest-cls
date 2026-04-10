"""
A5_expert_count_sweep.py
========================
Ablation A5: MAML + MoE — Expert Count Sweep (Mountain Curve)

Sweeps num_experts over {4, 8, 12, 16, 20, 24, 32, 40} with MOE_top_k set to
~num_experts/3 (rounded) as specified. All other hyperparameters are FIXED at M0
best values. This is not HPO — do not re-tune other hyperparameters per K value.

Output: per-expert-count mean ± std accuracy → the "mountain curve" figure.

Training : Episodic dataloader
Evaluation: Episodic (1-shot 3-way)
Seeds    : NUM_FINAL_SEEDS per expert count (matches the spec's error bars)
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
from MAML.mamlpp import mamlpp_pretrain

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Spec: sweep {4, 8, 12, 16, 20, 24, 32, 40}
EXPERT_COUNTS = [4, 8, 12, 16, 20, 24, 32, 40]


def topk_for_experts(num_experts: int) -> int:
    """Spec: top_k ≈ num_experts / 3, rounded. Minimum of 1."""
    return max(1, round(num_experts / 3))


def build_config(num_experts: int) -> dict:
    config = make_base_config(ablation_id=f"A5_E{num_experts}")
    top_k = topk_for_experts(num_experts)

    config["num_experts"]      = num_experts
    config["MOE_top_k"]        = top_k
    config["top_k"]            = top_k

    print(f"[A5] num_experts={num_experts}, top_k={top_k}")
    return config


def run_one_seed(num_experts: int, seed: int, config: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[A5 E={num_experts} | seed={seed}] Parameters: {n_params:,}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[A5 E={num_experts} | seed={seed}] Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "num_experts":      num_experts,
            "seed":             seed,
            "model_state_dict": train_history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
        },
        config,
        tag=f"E{num_experts}_seed{seed}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[A5 E={num_experts} | seed={seed}] Test acc: "
          f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")

    return {
        "num_experts":  num_experts,
        "top_k":        config["MOE_top_k"],
        "seed":         seed,
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


def main():
    all_expert_results = {}

    for num_experts in EXPERT_COUNTS:
        print(f"\n{'#'*70}")
        print(f"[A5] Expert count sweep: num_experts = {num_experts}")
        print(f"{'#'*70}")

        config = build_config(num_experts)
        seed_results = []

        for seed_idx in range(NUM_FINAL_SEEDS):
            actual_seed = FIXED_SEED + seed_idx
            print(f"\n{'='*70}")
            print(f"[A5 E={num_experts}] Seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
            print(f"{'='*70}")
            result = run_one_seed(num_experts, actual_seed, config)
            seed_results.append(result)

        test_accs = [r["test_results"]["mean_acc"] for r in seed_results]
        expert_summary = {
            "num_experts":   num_experts,
            "top_k":         topk_for_experts(num_experts),
            "n_params":      seed_results[0]["n_params"],
            "seed_results":  seed_results,
            "mean_test_acc": float(np.mean(test_accs)),
            "std_test_acc":  float(np.std(test_accs)),
        }
        all_expert_results[num_experts] = expert_summary

        # Save intermediate results after each expert count
        # (so partial results survive if the job is killed)
        _partial_config = copy.deepcopy(config)
        _partial_config["ablation_id"] = "A5"
        save_results(
            {"partial_sweep": all_expert_results},
            _partial_config,
            tag=f"partial_E{num_experts}",
        )

        print(f"\n[A5 E={num_experts}] DONE: "
              f"{expert_summary['mean_test_acc']*100:.2f}% ± {expert_summary['std_test_acc']*100:.2f}%")

    # Final summary — this is the mountain curve data
    sweep_summary = {
        "ablation_id":    "A5",
        "description":    "MAML + MoE: Expert Count Sweep (Mountain Curve)",
        "expert_counts":  EXPERT_COUNTS,
        "results":        all_expert_results,
        # Flatten for easy plotting
        "plot_data": {
            "x_num_experts":  EXPERT_COUNTS,
            "y_mean_acc":     [all_expert_results[e]["mean_test_acc"] for e in EXPERT_COUNTS],
            "y_std_acc":      [all_expert_results[e]["std_test_acc"]  for e in EXPERT_COUNTS],
            "top_k_values":   [topk_for_experts(e) for e in EXPERT_COUNTS],
        },
        "num_seeds":      NUM_FINAL_SEEDS,
    }

    final_config = make_base_config(ablation_id="A5")
    save_results(sweep_summary, final_config, tag="final_sweep")

    print(f"\n{'='*70}")
    print("[A5] MOUNTAIN CURVE RESULTS:")
    print(f"{'Experts':>10}  {'top_k':>6}  {'Acc (mean±std)':>20}")
    print("-" * 42)
    for e in EXPERT_COUNTS:
        r = all_expert_results[e]
        print(f"{e:>10}  {topk_for_experts(e):>6}  "
              f"{r['mean_test_acc']*100:>8.2f}% ± {r['std_test_acc']*100:.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
