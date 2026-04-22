# A2_no_maml_no_moe.py
"""
A2_no_maml_no_moe.py
=====================
Ablation A2: No-MAML + No-MoE (Vanilla CNN-LSTM Baseline)

Changes from M0:
  - No MAML.
  - No MoE (single CNN-LSTM encoder).
  - Flat training dataloader.
  - Episodic evaluation with head-only and full fine-tuning (same as A1).

test_procedure:
  'hpo_test_split' : Fixed split, multi-seed loop (development/HPO).
  'L2SO'           : Leave-2-Subjects-Out over all_PIDs. One run per fold.
"""

import os, sys, copy, json
import numpy as np
import torch
import pickle

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
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
from MAML.maml_data_pipeline import reorient_tensor_dict

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

FT_STEPS = 50
FT_LR    = 1e-3


def build_config() -> dict:
    config = make_base_config(ablation_id="A2")
    config["meta_learning"]   = False
    config["use_MOE"]         = False
    config["batch_size"]      = 64
    config["train_reps"]      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["val_reps"]        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["augment"]         = False
    config["ft_steps"]        = FT_STEPS
    config["ft_lr"]           = FT_LR
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]
    return config


def run_one_fold(fold_id: str, seed: int, config: dict, tensor_dict: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    print(f"\n[A2 | {fold_id} | seed={seed}]")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    model = build_supervised_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Parameters : {n_params:,}")

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert n_classes == config["num_classes"], (
        f"Expected {config['num_classes']} classes, got {n_classes}."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else float("nan")
    print(f"[A2 | {fold_id}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "fold_id":           fold_id,
            "seed":              seed,
            "model_state_dict":  trained_model.state_dict(),
            "config":            config,
            "best_val_acc":      best_val_acc,
            "train_loss_log":    history["train_loss"],
            "val_acc_log":       history["val_acc"],
        },
        config,
        tag=f"{fold_id}_seed{seed}_best",
    )

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

    head_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="head_only",
    )
    full_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="full",
    )

    print(f"[A2 | {fold_id}] Test head-only: {head_results['mean_acc']*100:.2f}% "
          f"± {head_results['std_acc']*100:.2f}%")
    print(f"[A2 | {fold_id}] Test full-ft  : {full_results['mean_acc']*100:.2f}% "
          f"± {full_results['std_acc']*100:.2f}%")

    return {
        "fold_id":        fold_id,
        "seed":           seed,
        "test_PID":       config["test_PIDs"],
        "val_PID":        config["val_PIDs"],
        "best_val_acc":   float(best_val_acc),
        "test_head_only": head_results,
        "test_full_ft":   full_results,
        "n_params":       n_params,
    }


def run_hpo_test_split(config: dict, tensor_dict: dict) -> list:
    results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[A2] hpo_test_split: seed {seed_idx+1}/{NUM_FINAL_SEEDS} (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_one_fold(
            fold_id=f"fixed_seed{actual_seed}",
            seed=actual_seed,
            config=copy.deepcopy(config),
            tensor_dict=tensor_dict,
        )
        results.append(result)
    return results


def build_l2so_folds(all_pids: list) -> list:
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


def run_l2so(config: dict, tensor_dict: dict) -> list:
    all_pids = config["all_PIDs"]
    assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."

    folds = build_l2so_folds(all_pids)
    results = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        print(f"\n{'='*70}")
        print(f"[A2] L2SO fold {fold_idx+1}/{len(folds)}  "
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
            tensor_dict=tensor_dict,
        )
        results.append(result)

    return results


def main():
    config = build_config()
    test_procedure = config["test_procedure"]

    print("\nA2 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"\nTest procedure: {test_procedure}")

    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'. Must be 'hpo_test_split' or 'L2SO'."
    )

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    if test_procedure == "hpo_test_split":
        all_results = run_hpo_test_split(config, tensor_dict)
        head_accs = [r["test_head_only"]["mean_acc"] for r in all_results]
        full_accs = [r["test_full_ft"]["mean_acc"]   for r in all_results]
        summary = {
            "ablation_id":         "A2",
            "description":         "No-MAML + No-MoE (Vanilla CNN-LSTM Baseline)",
            "test_procedure":      "hpo_test_split",
            "n_params":            all_results[0]["n_params"],
            "fold_results":        all_results,
            "mean_test_head_only": float(np.mean(head_accs)),
            "std_test_head_only":  float(np.std(head_accs)),
            "mean_test_full_ft":   float(np.mean(full_accs)),
            "std_test_full_ft":    float(np.std(full_accs)),
            "ft_steps":            FT_STEPS,
            "ft_lr":               FT_LR,
            "num_seeds":           NUM_FINAL_SEEDS,
            "config_snapshot":     {k: str(v) for k, v in config.items()},
        }
    else:  # L2SO
        all_results = run_l2so(config, tensor_dict)
        head_accs = [r["test_head_only"]["mean_acc"] for r in all_results]
        full_accs = [r["test_full_ft"]["mean_acc"]   for r in all_results]
        summary = {
            "ablation_id":         "A2",
            "description":         "No-MAML + No-MoE (Vanilla CNN-LSTM Baseline)",
            "test_procedure":      "L2SO",
            "n_params":            all_results[0]["n_params"],
            "fold_results":        all_results,
            "mean_test_head_only": float(np.mean(head_accs)),
            "std_test_head_only":  float(np.std(head_accs)),
            "mean_test_full_ft":   float(np.mean(full_accs)),
            "std_test_full_ft":    float(np.std(full_accs)),
            "ft_steps":            FT_STEPS,
            "ft_lr":               FT_LR,
            "num_folds":           len(all_results),
            "config_snapshot":     {k: str(v) for k, v in config.items()},
        }

    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[A2] FINAL head-only ({test_procedure}): "
          f"{summary['mean_test_head_only']*100:.2f}% ± {summary['std_test_head_only']*100:.2f}%")
    print(f"[A2] FINAL full-ft   ({test_procedure}): "
          f"{summary['mean_test_full_ft']*100:.2f}% ± {summary['std_test_full_ft']*100:.2f}%")
    if test_procedure == "L2SO":
        print(f"     over {summary['num_folds']} L2SO folds (one per test subject)")
    else:
        print(f"     over {summary['num_seeds']} seeds, fixed split")
    print(f"     {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()