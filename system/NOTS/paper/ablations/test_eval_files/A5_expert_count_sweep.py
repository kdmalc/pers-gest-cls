"""
A5_expert_count_sweep.py
========================
Ablation A5: MAML + MoE — Expert Count Sweep (Mountain Curve)

Sweeps num_experts over {4, 8, 12, 16, 20, 24, 32, 40} with MOE_top_k set to
~num_experts/3 (rounded) as specified. All other hyperparameters are FIXED at M0
best values. This is not HPO — do not re-tune other hyperparameters per K value.

This script runs a SINGLE expert count per invocation, specified via --num-experts.
The eval launcher submits one job per expert count in parallel.

Output: per-expert-count accuracy saved to RUN_DIR, tagged with the expert count.
        The launcher's job array produces one result file per expert count, which
        you then aggregate offline into the mountain curve figure.

Training : Episodic dataloader
Evaluation: Episodic (1-shot 3-way), 500 episodes over test_PIDs
Seed     : FIXED_SEED (single run per expert count, consistent with other ablations)
           To add error bars, change NUM_SEEDS below to NUM_FINAL_SEEDS and the
           launcher will need to be updated to account for longer wall time.
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

# Keep this in sync with EXPERT_COUNTS in eval_ablation_launcher.sh.
# Defined here for documentation and for the assert below.
EXPERT_COUNTS = [4, 8, 12, 16, 20, 24, 32, 40]


def topk_for_experts(num_experts: int) -> int:
    """Spec: top_k = num_experts / 3, rounded. Minimum of 1."""
    return max(1, round(num_experts / 3))


def build_config(num_experts: int) -> dict:
    config = make_base_config(ablation_id=f"A5_E{num_experts}")
    top_k = topk_for_experts(num_experts)
    config["num_experts"] = num_experts
    config["MOE_top_k"]   = top_k
    config["top_k"]       = top_k
    print(f"[A5] num_experts={num_experts}, top_k={top_k}")
    return config


def run(num_experts: int) -> dict:
    config = build_config(num_experts)
    set_seeds(FIXED_SEED)
    config["seed"] = FIXED_SEED

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[A5 E={num_experts} | seed={FIXED_SEED}] Parameters: {n_params:,}")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[A5 E={num_experts} | seed={FIXED_SEED}] Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "num_experts":       num_experts,
            "seed":              FIXED_SEED,
            "model_state_dict":  train_history["best_state"],
            "config":            config,
            "best_val_acc":      best_val_acc,
            "train_loss_log":    train_history["train_loss_log"],
            "val_acc_log":       train_history["val_acc_log"],
        },
        config,
        tag=f"E{num_experts}_seed{FIXED_SEED}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[A5 E={num_experts} | seed={FIXED_SEED}] Test acc: "
          f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")

    result = {
        "ablation_id":   f"A5_E{num_experts}",
        "description":   f"MAML + MoE: Expert Count Sweep, num_experts={num_experts}",
        "num_experts":   num_experts,
        "top_k":         config["MOE_top_k"],
        "seed":          FIXED_SEED,
        "best_val_acc":  float(best_val_acc),
        "test_results":  test_results,
        "n_params":      n_params,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"E{num_experts}_final")

    print(f"\n{'='*70}")
    print(f"[A5 E={num_experts}] FINAL: {test_results['mean_acc']*100:.2f}%")
    print(f"  top_k={config['MOE_top_k']}  seed={FIXED_SEED}")
    print(f"{'='*70}")

    return result


def main():
    parser = argparse.ArgumentParser(description="A5 expert count sweep — one expert count per job")
    parser.add_argument("--num-experts", type=int, required=True,
                        help=f"Number of experts to run. Must be one of {EXPERT_COUNTS}.")
    args = parser.parse_args()

    assert args.num_experts in EXPERT_COUNTS, (
        f"--num-experts {args.num_experts} is not in the defined sweep {EXPERT_COUNTS}. "
        f"Update EXPERT_COUNTS in both A5_expert_count_sweep.py and eval_ablation_launcher.sh "
        f"if you want to add a new value."
    )

    run(args.num_experts)


if __name__ == "__main__":
    main()