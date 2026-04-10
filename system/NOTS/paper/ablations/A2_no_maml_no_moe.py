"""
A2_no_maml_no_moe.py
=====================
Ablation A2: No-MAML + No-MoE (Vanilla CNN-LSTM Baseline)

Lower bound. Standard CNN-LSTM trained with supervised cross-entropy, no
meta-learning, no MoE. Shows that naive deep learning is insufficient for
cross-subject few-shot EMG gesture recognition.

Changes from M0:
  - No MAML.
  - No MoE (single CNN-LSTM encoder).
  - Flat training dataloader.
  - Episodic evaluation with head-only and full fine-tuning (same as A1).

Training : Flat dataloader
Evaluation: Episodic (1-shot 3-way), finetune before eval
"""

import os, sys, copy, json
import numpy as np
import torch
import pickle

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config, build_supervised_no_moe_model,
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS,
    run_supervised_test_eval, save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

FT_STEPS = 50
FT_LR    = 1e-3


def build_config() -> dict:
    config = make_base_config(ablation_id="A2")

    # ── Remove MAML ───────────────────────────────────────────────────────────
    config["meta_learning"] = False

    # ── Remove MoE ────────────────────────────────────────────────────────────
    config["use_MOE"] = False

    # ── Flat dataloader settings ──────────────────────────────────────────────
    config["batch_size"] = 64
    config["train_reps"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["val_reps"]   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["augment"]    = False

    # ── Fine-tuning config ────────────────────────────────────────────────────
    config["ft_steps"]        = FT_STEPS
    config["ft_lr"]           = FT_LR
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]

    return config


def run_one_seed(seed: int, config: dict, tensor_dict: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_supervised_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[A2 | seed={seed}] Parameters: {n_params:,}")

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert n_classes == config["num_classes"], (
        f"Expected {config['num_classes']} classes, got {n_classes}."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    best_val_acc = max(history["val_acc_log"]) if history["val_acc_log"] else float("nan")
    print(f"[A2 | seed={seed}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "seed":             seed,
            "model_state_dict": history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
            "train_loss_log":   history.get("train_loss_log", []),
            "val_acc_log":      history.get("val_acc_log", []),
        },
        config,
        tag=f"seed{seed}_best",
    )

    trained_model.load_state_dict(history["best_state"])

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

    head_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="head_only",
    )
    full_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="full",
    )

    print(f"[A2 | seed={seed}] Test head-only: {head_results['mean_acc']*100:.2f}% "
          f"± {head_results['std_acc']*100:.2f}%")
    print(f"[A2 | seed={seed}] Test full-ft  : {full_results['mean_acc']*100:.2f}% "
          f"± {full_results['std_acc']*100:.2f}%")

    return {
        "seed":             seed,
        "best_val_acc":     float(best_val_acc),
        "test_head_only":   head_results,
        "test_full_ft":     full_results,
        "n_params":         n_params,
    }


def main():
    config = build_config()
    print("\nA2 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict   = pickle.load(f)
    tensor_dict = full_dict["data"]

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[A2] Running seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_one_seed(actual_seed, config, tensor_dict)
        all_seed_results.append(result)

    head_accs = [r["test_head_only"]["mean_acc"] for r in all_seed_results]
    full_accs = [r["test_full_ft"]["mean_acc"]   for r in all_seed_results]

    summary = {
        "ablation_id":         "A2",
        "description":         "No-MAML + No-MoE (Vanilla CNN-LSTM Baseline)",
        "n_params":            all_seed_results[0]["n_params"],
        "seed_results":        all_seed_results,
        "mean_test_head_only": float(np.mean(head_accs)),
        "std_test_head_only":  float(np.std(head_accs)),
        "mean_test_full_ft":   float(np.mean(full_accs)),
        "std_test_full_ft":    float(np.std(full_accs)),
        "ft_steps":            FT_STEPS,
        "ft_lr":               FT_LR,
        "num_seeds":           NUM_FINAL_SEEDS,
        "config_snapshot":     {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[A2] FINAL head-only: {summary['mean_test_head_only']*100:.2f}% "
          f"± {summary['std_test_head_only']*100:.2f}%")
    print(f"[A2] FINAL full-ft  : {summary['mean_test_full_ft']*100:.2f}% "
          f"± {summary['std_test_full_ft']*100:.2f}%")
    print(f"     over {NUM_FINAL_SEEDS} seeds, {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
