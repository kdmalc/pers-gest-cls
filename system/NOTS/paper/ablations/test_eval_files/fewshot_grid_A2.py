"""
fewshot_grid_A2.py
==================
Few-Shot Grid Sweep: A2 model (No-MAML + No-MoE, vanilla CNN-LSTM) trained and
evaluated at each (k_shot, n_way) combo, with BOTH head_only and full fine-tuning.

Grid:
  k_shot : [1, 3, 5]
  n_way  : [3, 5, 10]
  -> 9 jobs total (one per cell), all run in parallel via the launcher.

This script runs a SINGLE (k_shot, n_way) pair per invocation, specified via
--k-shot and --n-way. The eval launcher submits one job per grid cell.

Differences from fewshot_grid.py (M0):
  - Uses build_supervised_no_moe_model instead of build_maml_moe_model.
  - Training uses get_pretrain_dataloaders (flat supervised) instead of
    get_maml_dataloaders (episodic).
  - Training uses pretrain() instead of mamlpp_pretrain().
  - Evaluation uses run_supervised_test_eval (finetune-then-eval) instead of
    run_episodic_test_eval (MAML adapt-and-eval).
  - Evaluation is done TWICE per cell: ft_mode='head_only' and ft_mode='full',
    matching the A2 ablation protocol.
  - The pretrain head is built with pretrain_num_classes (10-way); at eval time
    replace_head_for_eval() swaps in a fresh n_way-class head before finetuning.

Why train-per-cell:
  The supervised model is trained on a flat dataset of all 10 classes. However,
  ft_steps and ft_lr were tuned for the base (k=1, n=3) regime (A2 defaults).
  More importantly, at eval time the episodic sampler draws exactly k_shot
  support samples and evaluates n_way-class accuracy — these change meaningfully
  with (k, n). Training per-cell keeps the pretraining regime consistent with
  the eval regime (same n_way head is what gets fine-tuned into).

  NOTE: Unlike MAML, supervised pretraining is not directly sensitive to (k, n)
  during training itself — the flat dataloader doesn't use episodes. The main
  reason to retrain per-cell is to ensure the replacement head dimension matches
  n_way at training time, and for a fair comparison against the M0 grid where
  each cell IS a separately trained model.

Note: The (k=1, n=3) cell is identical to A2 by construction. It is included
here for grid completeness and as a self-consistency check.

All hyperparameters are fixed at A2 values (same as M0 Trial 89 base, with
meta_learning=False, use_MOE=False). This is not HPO.

Training : Flat supervised dataloader (all 10 gesture classes)
Evaluation: Episodic, 500 episodes over test_PIDs
           Both head_only and full fine-tuning reported per cell.
Seed     : FIXED_SEED (single run per cell, consistent with other ablations)

Output: one results JSON per cell, saved to RUN_DIR, tagged with k{K}_n{N}.
        Aggregate offline into the grid table / heatmap figure.
"""

import argparse
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
    set_seeds, FIXED_SEED,
    run_supervised_test_eval, save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain
from MAML.maml_data_pipeline import reorient_tensor_dict

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Keep in sync with GRID_K_SHOTS / GRID_N_WAYS in eval_launcher.sh (grid_A2 block).
GRID_K_SHOTS = [1, 3, 5]
GRID_N_WAYS  = [3, 5, 10]

# q_query is held fixed across all grid cells (standard practice).
Q_QUERY_FIXED = 9

# Fine-tuning hyperparameters — kept identical to the base A2 ablation.
FT_STEPS = 25
FT_LR    = 1e-3


def build_config(k_shot: int, n_way: int) -> dict:
    config = make_base_config(ablation_id=f"grid_A2_k{k_shot}_n{n_way}")

    # Supervised / no-MoE flags — same as A2.
    config["meta_learning"]   = False
    config["use_MOE"]         = False
    config["batch_size"]      = 64
    config["train_reps"]      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["val_reps"]        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config["augment"]         = False

    # Fine-tuning settings.
    config["ft_steps"]        = FT_STEPS
    config["ft_lr"]           = FT_LR
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]

    # Grid cell-specific task dimensions.
    # n_way controls replace_head_for_eval() at eval time.
    # k_shot / q_query control the episodic eval sampler.
    config["k_shot"]  = k_shot
    config["n_way"]   = n_way
    config["q_query"] = Q_QUERY_FIXED

    print(f"[grid_A2] k_shot={k_shot}  n_way={n_way}  q_query={Q_QUERY_FIXED}")
    return config


def run(k_shot: int, n_way: int) -> dict:
    config = build_config(k_shot, n_way)
    set_seeds(FIXED_SEED)
    config["seed"] = FIXED_SEED

    print(f"\n[grid_A2 k={k_shot} n={n_way} | seed={FIXED_SEED}]")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    model = build_supervised_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Parameters : {n_params:,}")

    # ── Load tensor dict once; reused for both pretrain and eval dataloaders ──
    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    # ── Flat supervised pretraining ───────────────────────────────────────────
    # get_pretrain_dataloaders uses pretrain_num_classes (10) for the head.
    # n_way is NOT used during pretraining — only at eval via replace_head_for_eval.
    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict=tensor_dict)
    assert n_classes == config["num_classes"], (
        f"Expected {config['num_classes']} classes from dataloader, got {n_classes}."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else float("nan")
    print(f"[grid_A2 k={k_shot} n={n_way}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "k_shot":            k_shot,
            "n_way":             n_way,
            "seed":              FIXED_SEED,
            "model_state_dict":  trained_model.state_dict(),
            "config":            config,
            "best_val_acc":      best_val_acc,
            "train_loss_log":    history["train_loss"],
            "val_acc_log":       history["val_acc"],
        },
        config,
        tag=f"k{k_shot}_n{n_way}_seed{FIXED_SEED}_best",
    )

    # ── Episodic eval: head-only fine-tuning ──────────────────────────────────
    head_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="head_only",
    )
    # ── Episodic eval: full fine-tuning ───────────────────────────────────────
    full_results = run_supervised_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"],
        ft_mode="full",
    )

    print(f"[grid_A2 k={k_shot} n={n_way}] Test head-only: "
          f"{head_results['mean_acc']*100:.2f}% ± {head_results['std_acc']*100:.2f}%")
    print(f"[grid_A2 k={k_shot} n={n_way}] Test full-ft  : "
          f"{full_results['mean_acc']*100:.2f}% ± {full_results['std_acc']*100:.2f}%")

    result = {
        "ablation_id":     f"grid_A2_k{k_shot}_n{n_way}",
        "description":     f"Few-Shot Grid A2 (No-MAML/No-MoE): k_shot={k_shot}, n_way={n_way}",
        "k_shot":          k_shot,
        "n_way":           n_way,
        "q_query":         Q_QUERY_FIXED,
        "seed":            FIXED_SEED,
        "best_val_acc":    float(best_val_acc),
        "test_head_only":  head_results,
        "test_full_ft":    full_results,
        "n_params":        n_params,
        "ft_steps":        FT_STEPS,
        "ft_lr":           FT_LR,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"k{k_shot}_n{n_way}_final")

    print(f"\n{'='*70}")
    print(f"[grid_A2] FINAL k={k_shot} n={n_way}:")
    print(f"  head-only : {head_results['mean_acc']*100:.2f}%  ±  {head_results['std_acc']*100:.2f}%")
    print(f"  full-ft   : {full_results['mean_acc']*100:.2f}%  ±  {full_results['std_acc']*100:.2f}%")
    print(f"  q_query={Q_QUERY_FIXED}  seed={FIXED_SEED}  ft_steps={FT_STEPS}  ft_lr={FT_LR}")
    print(f"{'='*70}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot grid sweep (A2: No-MAML/No-MoE) — one (k_shot, n_way) cell per job."
    )
    parser.add_argument("--k-shot", type=int, required=True,
                        help=f"Number of support shots. Must be one of {GRID_K_SHOTS}.")
    parser.add_argument("--n-way", type=int, required=True,
                        help=f"Number of classes. Must be one of {GRID_N_WAYS}.")
    args = parser.parse_args()

    assert args.k_shot in GRID_K_SHOTS, (
        f"--k-shot {args.k_shot} is not in the defined grid {GRID_K_SHOTS}. "
        f"Update GRID_K_SHOTS in both fewshot_grid_A2.py and eval_launcher.sh "
        f"if you want to add a new value."
    )
    assert args.n_way in GRID_N_WAYS, (
        f"--n-way {args.n_way} is not in the defined grid {GRID_N_WAYS}. "
        f"Update GRID_N_WAYS in both fewshot_grid_A2.py and eval_launcher.sh "
        f"if you want to add a new value."
    )

    run(args.k_shot, args.n_way)


if __name__ == "__main__":
    main()