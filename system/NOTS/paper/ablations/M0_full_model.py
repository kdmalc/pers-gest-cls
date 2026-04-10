"""
M0_full_model.py
================
Ablation M0: Full Model — MAML + MoE  [PRIMARY RESULT]

This is the full proposed model. All hyperparameters are at their best HPO-derived
values. All other ablations are defined as deviations from this config.
Do not modify this script.

Training : Episodic dataloader
Evaluation: Episodic (1-shot 3-way), 500 episodes over test_PIDs
Reported  : mean ± std over NUM_FINAL_SEEDS seeds
"""

import os, sys, copy, json, time
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


def build_config() -> dict:
    config = make_base_config(ablation_id="M0")
    # No changes — M0 is the full model with all defaults.
    return config


def run_one_seed(seed: int, config: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[M0 | seed={seed}] Parameters: {n_params:,}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[M0 | seed={seed}] Training complete. Best val acc = {best_val_acc:.4f}")

    # Save checkpoint
    save_model_checkpoint(
        {
            "seed": seed,
            "model_state_dict": train_history["best_state"],
            "config": config,
            "best_val_acc": best_val_acc,
            "train_loss_log": train_history["train_loss_log"],
            "val_acc_log":    train_history["val_acc_log"],
        },
        config,
        tag=f"seed{seed}_best",
    )

    # Load best weights for test eval
    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[M0 | seed={seed}] Test acc: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")

    return {
        "seed":         seed,
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


def main():
    config = build_config()
    print("\nM0 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    all_seed_results = []
    for seed in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed
        print(f"\n{'='*70}")
        print(f"[M0] Running seed {seed+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_one_seed(actual_seed, config)
        all_seed_results.append(result)

    test_accs = [r["test_results"]["mean_acc"] for r in all_seed_results]
    summary = {
        "ablation_id":     "M0",
        "description":     "Full Model: MAML + MoE",
        "n_params":        all_seed_results[0]["n_params"],
        "seed_results":    all_seed_results,
        "mean_test_acc":   float(np.mean(test_accs)),
        "std_test_acc":    float(np.std(test_accs)),
        "num_seeds":       NUM_FINAL_SEEDS,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[M0] FINAL: {summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
    print(f"     over {NUM_FINAL_SEEDS} seeds, {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
