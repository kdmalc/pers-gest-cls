"""
maml_eval_hp_sweep.py
=====================
Post-hoc eval HP sweep for trained MAML checkpoints (M0, A12, etc.).

Two modes
---------
1. 2D sweep  (--sweep-alpha, run FIRST)
   Sweeps maml_inner_steps_eval x maml_alpha_init_eval jointly on the val set.
   Uses HPO_STEPS_GRID — starts at 50 since you already know lower is suboptimal.
   Purpose: find the best (steps*, alpha*) for this checkpoint.

2. Paper curve  (--paper-curve --alpha <best>, run AFTER the 2D sweep)
   Fixes alpha to the best value found in the 2D sweep and sweeps a broader
   step range for the paper figure, including low step counts to show the
   sample-efficiency story vs A11.
   Uses PAPER_STEPS_GRID = {1, 3, 5, 10, 15, 25, 50, 100, 150, 200}.

Why two modes?
--------------
During HPO you already know <50 steps is suboptimal, so there is no reason
to waste cluster time evaluating them. But for the paper figure you WANT
those low step counts to show that MAML adapts faster than the baseline.
The two grids serve different purposes and must not be conflated.

Workflow
--------
  # Step 1: find best (steps, alpha) jointly
  bash eval_hp_launchers.sh M0_SWEEP --checkpoint /path/to/ckpt.pt --sweep-alpha

  # Step 2: inspect JSON, note best_alpha_so_far, then generate paper curve:
  bash eval_hp_launchers.sh M0_CURVE --checkpoint /path/to/ckpt.pt --alpha <best>

  # A12: identical, just pass --ablation-id A12

Usage (direct)
--------------
  python maml_eval_hp_sweep.py \\
      --checkpoint /path/to/trial_64_fold0_best.pt \\
      --ablation-id M0 \\
      --sweep-alpha \\
      --out-dir /scratch/.../eval_hp_sweep/M0

  python maml_eval_hp_sweep.py \\
      --checkpoint /path/to/trial_64_fold0_best.pt \\
      --ablation-id M0 \\
      --paper-curve --alpha 0.00672 \\
      --out-dir /scratch/.../eval_hp_sweep/M0

Output
------
  <out_dir>/eval_hp_sweep_<ablation_id>_<mode>_<timestamp>.json

  Partial results are written after every config — preemption-safe.

SLURM
-----
Single-GPU job, no array needed.
2D sweep    : ~2-3h  (8 steps x 9 alphas x ~2 min each)
Paper curve : ~25 min (10 steps x ~2.5 min each)
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# =============================================================================
# Paths
# =============================================================================

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./")).resolve()

for _p in [
    CODE_DIR,
    CODE_DIR / "system",
    CODE_DIR / "system" / "MAML",
    CODE_DIR / "system" / "MOE",
    CODE_DIR / "system" / "pretraining",
]:
    sys.path.insert(0, str(_p))

print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"CODE_DIR       : {CODE_DIR}")
print(f"RUN_DIR        : {RUN_DIR}")

# =============================================================================
# Step grids
# =============================================================================

# 2D sweep (--sweep-alpha). Starts at 50 — below that already known suboptimal.
HPO_STEPS_GRID = [50, 75, 100, 125, 150, 175, 200, 250]

# Paper figure (--paper-curve). Full trajectory including low step counts.
PAPER_STEPS_GRID = [1, 3, 5, 10, 15, 25, 50, 100, 150, 200]

# Alpha grid for 2D sweep. Centered around M0's HPO value of 0.006717.
# Covers below and above — at high step counts the optimal alpha tends smaller.
ALPHA_EVAL_GRID = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030]

NUM_VAL_EPISODES = 200
DEFAULT_VAL_PIDS = ["P011", "P006", "P105", "P109"]


# =============================================================================
# Checkpoint loader
# =============================================================================

def load_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    """
    Load a trained MAML checkpoint and reconstruct the model.
    Returns (model, config) with weights loaded and model on GPU.

    Expected checkpoint keys:
        checkpoint["model_state_dict"]
        checkpoint["config"]
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    use_moe = config.get("use_MOE", False)
    if use_moe:
        from MOE.MOE_encoder import build_MOE_model
        model = build_MOE_model(config)
    else:
        from pretraining.pretrain_models import build_model
        model = build_model(config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Global best val acc : "
          f"{checkpoint.get('global_best_val_acc', checkpoint.get('best_val_acc', 'N/A'))}")
    print(f"  Trained inner steps : {config.get('maml_inner_steps')}")
    print(f"  Trained alpha eval  : {config.get('maml_alpha_init_eval')}")

    return model, config


# =============================================================================
# Single (steps, alpha) evaluation
# =============================================================================

def eval_one_config(
    model:            torch.nn.Module,
    base_config:      dict,
    inner_steps_eval: int,
    alpha_eval:       float,
    val_pids:         list[str],
    tensor_dict_path: str,
    num_val_episodes: int,
    fixed_seed:       int,
) -> dict:
    """
    Run episodic val eval for one (inner_steps_eval, alpha_eval) combination.

    Model weights are never modified — each episode starts from the same
    fixed init.

    use_lslr_at_eval is forced False. The LSLR was trained for a specific
    inner_steps count; using it at a different count is invalid and would
    confound the sweep signal.
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from MAML.mamlpp import mamlpp_adapt_and_eval
    from torch.utils.data import DataLoader

    eval_config = copy.deepcopy(base_config)
    eval_config["maml_inner_steps_eval"] = inner_steps_eval
    eval_config["maml_alpha_init_eval"]  = alpha_eval
    eval_config["use_lslr_at_eval"]      = False
    eval_config["val_PIDs"]              = val_pids

    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = val_pids,
        target_gesture_classes  = eval_config["maml_gesture_classes"],
        target_trial_indices    = eval_config["target_trial_indices"],
        n_way                   = eval_config["n_way"],
        k_shot                  = eval_config["k_shot"],
        q_query                 = eval_config.get("q_query", None),
        num_eval_episodes       = num_val_episodes,
        is_train                = False,
        seed                    = fixed_seed,
        use_label_shuf_meta_aug = False,
    )
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=maml_mm_collate)

    user_accs:   dict = defaultdict(list)
    user_losses: dict = defaultdict(list)

    for batch in val_dl:
        uid     = batch["user_id"]
        metrics = mamlpp_adapt_and_eval(
            model, eval_config, batch["support"], batch["query"]
        )
        user_accs[uid].append(float(metrics["acc"]))
        if "loss" in metrics:
            user_losses[uid].append(float(metrics["loss"]))

    per_user_acc  = {uid: float(np.mean(accs))   for uid, accs   in user_accs.items()}
    per_user_loss = {uid: float(np.mean(losses))  for uid, losses in user_losses.items()
                     if losses}

    all_accs   = list(per_user_acc.values())
    all_losses = list(per_user_loss.values())

    return {
        "inner_steps_eval": inner_steps_eval,
        "alpha_eval":       alpha_eval,
        "mean_acc":         float(np.mean(all_accs)),
        "std_acc":          float(np.std(all_accs)),
        "mean_loss":        float(np.mean(all_losses)) if all_losses else None,
        "std_loss":         float(np.std(all_losses))  if all_losses else None,
        "per_user_acc":     per_user_acc,
        "per_user_loss":    per_user_loss,
    }


# =============================================================================
# Main sweep runner
# =============================================================================

def run_sweep(
    checkpoint_path:  Path,
    ablation_id:      str,
    out_dir:          Path,
    val_pids:         list[str],
    num_val_episodes: int,
    sweep_alpha:      bool,
    paper_curve:      bool,
    fixed_alpha:      float | None,
    fixed_seed:       int = 42,
) -> None:
    """
    Run the sweep and save results to JSON in out_dir.
    Exactly one of sweep_alpha or paper_curve must be True.
    """
    assert sweep_alpha != paper_curve, \
        "Exactly one of --sweep-alpha or --paper-curve must be set."
    if paper_curve:
        assert fixed_alpha is not None, \
            "--alpha must be provided when using --paper-curve."

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag  = "2d_sweep" if sweep_alpha else "paper_curve"

    model, base_config = load_checkpoint(checkpoint_path)

    if ablation_id == "A12":
        from ablation_hpo import EMG_2KHZ_PKL_PATH
        tensor_dict_path = str(EMG_2KHZ_PKL_PATH)
    else:
        tensor_dict_path = os.path.join(
            base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
        )

    trained_alpha = float(base_config.get("maml_alpha_init_eval", 0.0))
    trained_steps = int(base_config.get("maml_inner_steps", 10))

    if sweep_alpha:
        steps_grid = HPO_STEPS_GRID
        alpha_grid = ALPHA_EVAL_GRID
        print(f"\n2D SWEEP: {len(steps_grid)} step values x "
              f"{len(alpha_grid)} alpha values = "
              f"{len(steps_grid) * len(alpha_grid)} configurations.")
    else:
        steps_grid = PAPER_STEPS_GRID
        alpha_grid = [fixed_alpha]
        print(f"\nPAPER CURVE: {len(steps_grid)} step values at "
              f"fixed alpha={fixed_alpha:.6f}.")

    grid      = [(s, a) for s in steps_grid for a in alpha_grid]
    n_configs = len(grid)

    print(f"Ablation         : {ablation_id}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Val PIDs         : {val_pids}")
    print(f"Val episodes     : {num_val_episodes}")
    print(f"Trained steps    : {trained_steps}")
    print(f"Trained alpha    : {trained_alpha:.6f}")
    print(f"Total configs    : {n_configs}")
    print(f"Output dir       : {out_dir}")
    print()

    results          = []
    best_acc         = -1.0
    best_steps       = None
    best_alpha_found = None
    sweep_start      = time.time()

    for i, (steps, alpha) in enumerate(grid):
        t0 = time.time()
        print(f"[{i+1:>3}/{n_configs}] steps={steps:>4}, alpha={alpha:.5f} ...",
              end="", flush=True)

        result = eval_one_config(
            model            = model,
            base_config      = base_config,
            inner_steps_eval = steps,
            alpha_eval       = alpha,
            val_pids         = val_pids,
            tensor_dict_path = tensor_dict_path,
            num_val_episodes = num_val_episodes,
            fixed_seed       = fixed_seed,
        )

        elapsed = time.time() - t0
        print(f"  acc={result['mean_acc']*100:.2f}%  ({elapsed:.1f}s)")
        results.append(result)

        if result["mean_acc"] > best_acc:
            best_acc         = result["mean_acc"]
            best_steps       = steps
            best_alpha_found = alpha

        partial_output = {
            "ablation_id":          ablation_id,
            "mode":                 mode_tag,
            "checkpoint":           str(checkpoint_path),
            "val_pids":             val_pids,
            "num_val_episodes":     num_val_episodes,
            "trained_steps":        trained_steps,
            "trained_alpha":        trained_alpha,
            "steps_grid":           steps_grid,
            "alpha_grid":           alpha_grid,
            "fixed_seed":           fixed_seed,
            "n_configs_total":      n_configs,
            "n_configs_done":       i + 1,
            "best_steps_so_far":    best_steps,
            "best_alpha_so_far":    best_alpha_found,
            "best_mean_acc_so_far": best_acc,
            "results":              results,
        }
        out_path = out_dir / f"eval_hp_sweep_{ablation_id}_{mode_tag}_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(partial_output, f, indent=2)

    total_elapsed = time.time() - sweep_start

    print(f"\n{'='*70}")
    print(f"[{ablation_id}] {mode_tag} complete in {total_elapsed/60:.1f} min")
    print(f"  Best steps : {best_steps}")
    print(f"  Best alpha : {best_alpha_found:.5f}")
    print(f"  Best acc   : {best_acc*100:.2f}%")
    print(f"  Results    : {out_path}")
    print(f"{'='*70}")

    if sweep_alpha:
        print(f"\nNext step — generate paper curve with best alpha:")
        print(f"  bash eval_hp_launchers.sh M0_CURVE \\")
        print(f"      --checkpoint {checkpoint_path} \\")
        print(f"      --alpha {best_alpha_found:.6f}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-hoc eval HP sweep for trained MAML checkpoints."
    )
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument(
        "--ablation-id", type=str, required=True,
        choices=["M0", "A3", "A4", "A5", "A8", "A12"],
        dest="ablation_id",
    )
    parser.add_argument("--out-dir",  type=str, default=None, dest="out_dir")
    parser.add_argument("--val-pids", type=str, nargs="+",
                        default=DEFAULT_VAL_PIDS, dest="val_pids")
    parser.add_argument("--num-val-episodes", type=int,
                        default=NUM_VAL_EPISODES, dest="num_val_episodes")
    parser.add_argument("--fixed-seed", type=int, default=42, dest="fixed_seed")
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Fixed alpha for --paper-curve. "
             "Take best_alpha_so_far from the 2D sweep JSON.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--sweep-alpha", action="store_true", dest="sweep_alpha",
        help="2D sweep over HPO_STEPS_GRID x ALPHA_EVAL_GRID. Run this first.",
    )
    mode_group.add_argument(
        "--paper-curve", action="store_true", dest="paper_curve",
        help="Paper figure curve at fixed --alpha over PAPER_STEPS_GRID. "
             "Run this after --sweep-alpha.",
    )

    args = parser.parse_args()

    run_sweep(
        checkpoint_path  = Path(args.checkpoint).resolve(),
        ablation_id      = args.ablation_id,
        out_dir          = Path(args.out_dir).resolve() if args.out_dir else RUN_DIR,
        val_pids         = args.val_pids,
        num_val_episodes = args.num_val_episodes,
        sweep_alpha      = args.sweep_alpha,
        paper_curve      = args.paper_curve,
        fixed_alpha      = args.alpha,
        fixed_seed       = args.fixed_seed,
    )