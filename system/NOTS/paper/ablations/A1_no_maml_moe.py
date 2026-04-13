"""
A1_no_maml_moe.py
=================
Ablation A1: No-MAML + MoE (Supervised MoE)

Isolates the contribution of MAML. The MoE encoder is identical to M0, but
training uses a flat dataloader with standard cross-entropy (no meta-learning).
At eval time we finetune on the 1-shot support set before episodic evaluation,
reporting BOTH head-only and full fine-tuning results (per spec).

Changes from M0:
  - No MAML inner/outer loop.
  - Flat (standard supervised) training dataloader.
  - All MoE components identical to M0.

Training : Flat dataloader
Evaluation: Episodic (1-shot 3-way), finetune before eval
            → reported as (head_only, full_ft)
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
    make_base_config, build_supervised_moe_model,
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS,
    run_supervised_test_eval, save_results, save_model_checkpoint, count_parameters,
    replace_head_for_eval,
    RUN_DIR,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain
import pickle

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Finetuning protocol (spec: 50 gradient steps, both modes reported)
FT_STEPS  = 50
FT_LR     = 1e-3   # reasonable default; same order as maml_alpha_init_eval


def build_config() -> dict:
    config = make_base_config(ablation_id="A1")

    # ── Remove MAML — not used for training ───────────────────────────────────
    config["meta_learning"] = False

    # ── Use learning_rate as single LR (no inner/outer split) ────────────────
    # outer_lr from M0 best config is the training LR here.
    # (maml_alpha_init etc. are never referenced in pretrain_trainer.py)

    # ── Flat dataloader settings ──────────────────────────────────────────────
    config["batch_size"]  = 64
    config["train_reps"]  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["val_reps"]    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["augment"]     = False

    # ── Fine-tuning config (applied at eval time) ─────────────────────────────
    config["ft_steps"]        = FT_STEPS
    config["ft_lr"]           = FT_LR
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]

    # ── MoE stays identical to M0 ─────────────────────────────────────────────
    # (already set in make_base_config)

    return config


def run_one_seed(seed: int, config: dict, tensor_dict: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_supervised_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[A1 | seed={seed}] Parameters: {n_params:,}")

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert n_classes == config["pretrain_num_classes"], (
        f"Flat dataloader returned {n_classes} classes but pretrain_num_classes="
        f"{config['pretrain_num_classes']}. Check your gesture class list."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    best_val_acc = max(history["val_acc_log"]) if history["val_acc_log"] else float("nan")
    print(f"[A1 | seed={seed}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "seed":              seed,
            "model_state_dict":  history["best_state"],
            "config":            config,
            "best_val_acc":      best_val_acc,
            "train_loss_log":    history.get("train_loss_log", []),
            "val_acc_log":       history.get("val_acc_log", []),
        },
        config,
        tag=f"seed{seed}_best",
    )

    trained_model.load_state_dict(history["best_state"])

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

    # Both fine-tuning modes (spec requirement for A1)
    head_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="head_only",
    )
    full_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="full",
    )

    print(f"[A1 | seed={seed}] Test head-only: {head_results['mean_acc']*100:.2f}% "
          f"± {head_results['std_acc']*100:.2f}%")
    print(f"[A1 | seed={seed}] Test full-ft  : {full_results['mean_acc']*100:.2f}% "
          f"± {full_results['std_acc']*100:.2f}%")

    return {
        "seed":                  seed,
        "best_val_acc":          float(best_val_acc),
        "test_head_only":        head_results,
        "test_full_ft":          full_results,
        "n_params":              n_params,
    }


def main():
    config = build_config()
    print("\nA1 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    # Load tensor_dict once — reused across seeds
    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict   = pickle.load(f)
    tensor_dict = full_dict["data"]

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[A1] Running seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_one_seed(actual_seed, config, tensor_dict)
        all_seed_results.append(result)

    head_accs = [r["test_head_only"]["mean_acc"] for r in all_seed_results]
    full_accs = [r["test_full_ft"]["mean_acc"]   for r in all_seed_results]

    summary = {
        "ablation_id":          "A1",
        "description":          "No-MAML + MoE (Supervised MoE)",
        "n_params":             all_seed_results[0]["n_params"],
        "seed_results":         all_seed_results,
        "mean_test_head_only":  float(np.mean(head_accs)),
        "std_test_head_only":   float(np.std(head_accs)),
        "mean_test_full_ft":    float(np.mean(full_accs)),
        "std_test_full_ft":     float(np.std(full_accs)),
        "ft_steps":             FT_STEPS,
        "ft_lr":                FT_LR,
        "num_seeds":            NUM_FINAL_SEEDS,
        "config_snapshot":      {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[A1] FINAL head-only: {summary['mean_test_head_only']*100:.2f}% "
          f"± {summary['std_test_head_only']*100:.2f}%")
    print(f"[A1] FINAL full-ft  : {summary['mean_test_full_ft']*100:.2f}% "
          f"± {summary['std_test_full_ft']*100:.2f}%")
    print(f"     over {NUM_FINAL_SEEDS} seeds, {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()