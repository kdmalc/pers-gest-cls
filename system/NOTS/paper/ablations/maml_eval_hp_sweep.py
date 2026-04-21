"""
maml_eval_hp_sweep.py
=====================
Post-hoc eval HP sweep for trained MAML checkpoints (M0, A12, etc.).

Purpose
-------
After training HPO, the best checkpoint was selected with maml_inner_steps_eval
matched to maml_inner_steps (e.g. 10). At test time you can run MORE inner steps
without any retraining cost — the meta-init is fixed, you're just changing how
hard you push the adaptation. This script sweeps two eval-time HPs jointly:

  maml_inner_steps_eval  : how many gradient steps to take at eval time
  maml_alpha_init_eval   : the per-step learning rate used during eval adaptation

This is SEPARATE from ablation_hpo.py and does NOT touch any Optuna journal.
It writes a self-contained JSON results file to RUN_DIR.

What this is NOT
----------------
This is not a second round of training HPO. The model weights are frozen.
We are characterising how the fixed init responds to different adaptation
budgets. This is the same kind of sweep as the M0_inner_steps_sweep.json
you already ran, but now also sweeping the eval LR.

Usage
-----
  python maml_eval_hp_sweep.py \\
      --checkpoint /path/to/trial_64_fold0_best.pt \\
      --ablation_id M0 \\
      --out_dir /scratch/.../eval_hp_sweep/M0

  # To also vary eval LR (recommended):
  python maml_eval_hp_sweep.py \\
      --checkpoint /path/to/trial_64_fold0_best.pt \\
      --ablation_id M0 \\
      --sweep_alpha  \\
      --out_dir /scratch/.../eval_hp_sweep/M0

  # A12 is identical — just pass the A12 checkpoint and --ablation_id A12:
  python maml_eval_hp_sweep.py \\
      --checkpoint /path/to/A12_best.pt \\
      --ablation_id A12 \\
      --sweep_alpha \\
      --out_dir /scratch/.../eval_hp_sweep/A12

Output
------
  <out_dir>/eval_hp_sweep_<ablation_id>_<timestamp>.json

  Contains per-(steps, alpha) result dicts with mean_acc, std_acc, per_user_acc.
  Load in your analysis notebook to find the best combo and plot the surface.

SLURM
-----
This is a single-GPU job — no array needed.
Typical wall time: ~15-30 min for the default grid (depends on num_val_episodes
and the number of grid points).

Example sbatch flags:
  --time=01:30:00
  --mem=32G
  --gres=gpu:1
  --cpus-per-task=10
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
# Paths — resolved from environment (same pattern as ablation_hpo.py)
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
print(f"DATA_DIR       : {DATA_DIR}")
print(f"RUN_DIR        : {RUN_DIR}")

# =============================================================================
# Sweep grid definition
# =============================================================================

# Inner steps to sweep. Extend beyond your previous maximum of 100.
# The goal is to find where accuracy plateaus — do not stop early.
INNER_STEPS_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250]

# Eval alpha grid. Your HPO found 0.006717 as best at 10-step eval.
# At 100+ steps you likely want a smaller LR (risk of overshooting with many steps).
# Cover a range below and above the HPO value to map the sensitivity.
ALPHA_EVAL_GRID = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030]

# Number of val episodes. Use 200 to match your existing M0 inner-steps sweep JSON.
NUM_VAL_EPISODES = 200

# Val PIDs — must match what was used in the existing sweep for comparability.
# These are the four val users from your M0 sweep JSON.
DEFAULT_VAL_PIDS = ["P011", "P006", "P105", "P109"]


# =============================================================================
# Checkpoint loader
# =============================================================================

def load_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    """
    Load a trained MAML checkpoint and reconstruct the model.
    Returns (model, config) with model weights loaded and model on GPU.

    The checkpoint is expected to have:
        checkpoint["model_state_dict"]  : state dict
        checkpoint["config"]            : the training config dict

    We reconstruct the model using the same builder as the training scripts
    so that architecture is guaranteed to match the saved weights.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    # Build model using the same factory as training
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

    print(f"  Checkpoint loaded. Global best val acc: "
          f"{checkpoint.get('global_best_val_acc', checkpoint.get('best_val_acc', 'N/A'))}")
    print(f"  Config: maml_inner_steps={config.get('maml_inner_steps')}, "
          f"maml_alpha_init_eval={config.get('maml_alpha_init_eval')}")

    return model, config


# =============================================================================
# Single (steps, alpha) evaluation
# =============================================================================

def eval_one_config(
    model: torch.nn.Module,
    base_config: dict,
    inner_steps_eval: int,
    alpha_eval: float,
    val_pids: list[str],
    tensor_dict_path: str,
    num_val_episodes: int,
    fixed_seed: int,
) -> dict:
    """
    Run episodic val eval for one (inner_steps_eval, alpha_eval) combination.

    The model weights are never modified — mamlpp_adapt_and_eval deepcopies
    internally (or uses a temporary copy) for each episode so the base init
    is always the same starting point.

    Returns a dict with mean_acc, std_acc, per_user_acc, mean_loss, std_loss.
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from MAML.mamlpp import mamlpp_adapt_and_eval
    from torch.utils.data import DataLoader

    # Build a config copy with the eval HPs overridden
    eval_config = copy.deepcopy(base_config)
    eval_config["maml_inner_steps_eval"] = inner_steps_eval
    eval_config["maml_alpha_init_eval"]  = alpha_eval
    # Ensure we are using the eval LR, not LSLR during eval
    # (LSLR was trained for a specific inner_steps; it's not valid at a different
    # step count unless re-tuned. For the sweep, fix use_lslr_at_eval=False so
    # we get a clean signal from the scalar alpha.)
    eval_config["use_lslr_at_eval"] = False

    # Load tensor_dict and build val dataset
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, eval_config)

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = val_pids,
        target_gesture_classes = eval_config["maml_gesture_classes"],
        target_trial_indices   = eval_config["target_trial_indices"],
        n_way                  = eval_config["n_way"],
        k_shot                 = eval_config["k_shot"],
        q_query                = eval_config.get("q_query", None),
        num_eval_episodes      = num_val_episodes,
        is_train               = False,
        seed                   = fixed_seed,
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

    per_user_acc  = {uid: float(np.mean(accs)) for uid, accs in user_accs.items()}
    per_user_loss = {uid: float(np.mean(losses)) for uid, losses in user_losses.items() if losses}

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
    checkpoint_path: Path,
    ablation_id: str,
    out_dir: Path,
    val_pids: list[str],
    num_val_episodes: int,
    sweep_alpha: bool,
    fixed_seed: int = 42,
) -> None:
    """
    Run the full (inner_steps x alpha) grid sweep and save results to JSON.

    If sweep_alpha=False, only inner_steps is swept with the checkpoint's
    trained alpha value (same behaviour as the existing M0 sweep JSON).

    If sweep_alpha=True, the full 2D grid is swept. This is recommended —
    the optimal alpha changes substantially as inner_steps increases.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    model, base_config = load_checkpoint(checkpoint_path)
    base_config["val_PIDs"] = val_pids

    # Tensor dict path: use the config's dfs_load_path + standard filename,
    # or the 2kHz path for A12.
    if ablation_id == "A12":
        from ablation_hpo import EMG_2KHZ_PKL_PATH
        tensor_dict_path = str(EMG_2KHZ_PKL_PATH)
    else:
        tensor_dict_path = os.path.join(
            base_config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
        )

    trained_alpha = float(base_config.get("maml_alpha_init_eval", 0.0))
    trained_steps = int(base_config.get("maml_inner_steps", 10))

    # Build the grid of (steps, alpha) pairs to evaluate
    if sweep_alpha:
        alpha_grid = ALPHA_EVAL_GRID
        print(f"\nSweeping {len(INNER_STEPS_GRID)} step values × "
              f"{len(alpha_grid)} alpha values = "
              f"{len(INNER_STEPS_GRID) * len(alpha_grid)} configurations.")
    else:
        # Only sweep steps; use the trained alpha value as-is
        alpha_grid = [trained_alpha]
        print(f"\nSweeping {len(INNER_STEPS_GRID)} step values at fixed "
              f"trained alpha={trained_alpha:.6f}.")

    grid = [(s, a) for s in INNER_STEPS_GRID for a in alpha_grid]
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

    results = []
    best_acc   = -1.0
    best_steps = None
    best_alpha = None

    sweep_start = time.time()

    for i, (steps, alpha) in enumerate(grid):
        t0 = time.time()
        print(f"[{i+1:>3}/{n_configs}] steps={steps:>4}, alpha={alpha:.5f} ...", end="", flush=True)

        result = eval_one_config(
            model          = model,
            base_config    = base_config,
            inner_steps_eval = steps,
            alpha_eval     = alpha,
            val_pids       = val_pids,
            tensor_dict_path = tensor_dict_path,
            num_val_episodes = num_val_episodes,
            fixed_seed     = fixed_seed,
        )

        elapsed = time.time() - t0
        print(f"  acc={result['mean_acc']*100:.2f}%  ({elapsed:.1f}s)")

        results.append(result)

        if result["mean_acc"] > best_acc:
            best_acc   = result["mean_acc"]
            best_steps = steps
            best_alpha = alpha

        # Checkpoint partial results after every config so progress is not
        # lost if the job is preempted. Overwrite the same file each time.
        partial_output = {
            "ablation_id":        ablation_id,
            "checkpoint":         str(checkpoint_path),
            "val_pids":           val_pids,
            "num_val_episodes":   num_val_episodes,
            "trained_steps":      trained_steps,
            "trained_alpha":      trained_alpha,
            "sweep_alpha":        sweep_alpha,
            "inner_steps_grid":   INNER_STEPS_GRID,
            "alpha_grid":         alpha_grid,
            "fixed_seed":         fixed_seed,
            "n_configs_total":    n_configs,
            "n_configs_done":     i + 1,
            "best_steps_so_far":  best_steps,
            "best_alpha_so_far":  best_alpha,
            "best_mean_acc_so_far": best_acc,
            "results":            results,
        }
        out_path = out_dir / f"eval_hp_sweep_{ablation_id}_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(partial_output, f, indent=2)

    total_elapsed = time.time() - sweep_start

    print(f"\n{'='*70}")
    print(f"[{ablation_id}] Sweep complete in {total_elapsed/60:.1f} min")
    print(f"  Best steps : {best_steps}")
    print(f"  Best alpha : {best_alpha:.5f}")
    print(f"  Best acc   : {best_acc*100:.2f}%")
    print(f"  Results    : {out_path}")
    print(f"{'='*70}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-hoc eval HP sweep for trained MAML checkpoints."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained .pt checkpoint file.",
    )
    parser.add_argument(
        "--ablation_id", type=str, required=True,
        choices=["M0", "A3", "A4", "A5", "A8", "A12"],
        help="Ablation ID (determines data path and model builder).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Directory to write sweep results JSON. Defaults to RUN_DIR env var.",
    )
    parser.add_argument(
        "--val_pids", type=str, nargs="+", default=DEFAULT_VAL_PIDS,
        help="Val PIDs to evaluate on. Default: P011 P006 P105 P109.",
    )
    parser.add_argument(
        "--num_val_episodes", type=int, default=NUM_VAL_EPISODES,
        help=f"Number of val episodes per config. Default: {NUM_VAL_EPISODES}.",
    )
    parser.add_argument(
        "--sweep_alpha", action="store_true",
        help="If set, sweep maml_alpha_init_eval over ALPHA_EVAL_GRID in addition "
             "to inner_steps. Recommended — the optimal alpha changes with step count.",
    )
    parser.add_argument(
        "--fixed_seed", type=int, default=42,
        help="Random seed for episode sampling. Default: 42.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else RUN_DIR

    run_sweep(
        checkpoint_path  = Path(args.checkpoint).resolve(),
        ablation_id      = args.ablation_id,
        out_dir          = out_dir,
        val_pids         = args.val_pids,
        num_val_episodes = args.num_val_episodes,
        sweep_alpha      = args.sweep_alpha,
        fixed_seed       = args.fixed_seed,
    )