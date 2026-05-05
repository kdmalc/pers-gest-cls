"""
fewshot_grid_A4.py
==================
Few-Shot Grid Sweep: A4 model (MAML + No-MoE, single encoder matched to ALL M0
experts combined) trained and evaluated at each (k_shot, n_way) combo.

Grid:
  k_shot : [1, 3, 5]
  n_way  : [3, 5, 10]
  -> 9 jobs total (one per cell), all run in parallel via the launcher.

This script runs a SINGLE (k_shot, n_way) pair per invocation, specified via
--k-shot and --n-way. The eval launcher submits one job per grid cell.

Why A4 and not A3 for the grid?
  A4 is the critical capacity-controlled ablation: architecturally identical to
  A2 but trained with MAML. It is the direct counterpart to the M0 grid (which
  adds MoE on top) and the A2 grid (which replaces MAML with supervised). The
  three grids together answer:
    M0  grid : MoE + MAML        (full system)
    A4  grid : no-MoE + MAML     (isolates MoE)
    A2  grid : no-MoE + no-MAML  (isolates both)

Test procedure:
  hpo_test_split ONLY. A4 trains with MAML (mamlpp_pretrain), which is
  significantly slower than the supervised A2 training. Running a full L2SO
  across the 9-cell grid would require ~9 * 32 = 288 MAML training runs — not
  feasible within cluster resource budgets. The (k=1, n=3) cell is identical to
  the canonical A4 ablation (hpo_test_split) by construction and serves as a
  self-consistency check.

Differences from fewshot_grid.py (M0 grid):
  - Uses build_maml_no_moe_model (no MoE) instead of build_maml_moe_model.
  - CNN encoder is capacity-matched to ALL M0 experts combined via
    compute_matched_filters_for_ablation (same as standalone A4).
  - No MoE aux loss, no expert routing — pure MAML + single encoder.

Differences from fewshot_grid_A2.py (A2 grid):
  - Training uses get_maml_dataloaders + mamlpp_pretrain (episodic MAML)
    instead of get_pretrain_dataloaders + pretrain (flat supervised).
  - Evaluation uses run_episodic_test_eval (MAML adapt-and-eval) instead of
    run_supervised_test_eval (finetune-then-eval).
  - No head_only vs. full fine-tuning split at eval time — MAML adaptation
    updates all parameters by definition (inner loop runs on full model).
  - test_procedure is fixed to hpo_test_split (see note above).

Why train-per-cell:
  MAML training IS directly sensitive to (k_shot, n_way): the episodic dataloader
  samples exactly k_shot support examples and evaluates n_way-class accuracy at
  every episode during both training and validation. The meta-objective the model
  is optimised for changes with the task distribution, so each (k, n) cell
  genuinely requires a separately trained model. This is the same reason the M0
  grid retrains per cell.

Note: The (k=1, n=3) cell is identical to A4 (hpo_test_split) by construction.
It is included here for grid completeness and as a self-consistency check.

All non-(k, n) hyperparameters are fixed at A4 / Trial 89 values. This is not
HPO — we are only varying the few-shot task dimensions.

Training : Episodic MAML (mamlpp_pretrain), identical regime to standalone A4.
Evaluation: Episodic, 500 episodes over test_PIDs (NUM_TEST_EPISODES).
Seed     : FIXED_SEED (single run per cell, consistent with other ablations).

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
    make_base_config, build_maml_no_moe_model,
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    make_periodic_checkpoint_fn, make_periodic_test_eval_fn,
    compute_matched_filters_for_ablation,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import mamlpp_pretrain

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Keep in sync with GRID_K_SHOTS / GRID_N_WAYS in eval_launcher.sh (grid_A4 block).
GRID_K_SHOTS = [1, 3, 5]
GRID_N_WAYS  = [3, 5, 10]

# q_query is held fixed across all grid cells (standard practice; mirrors M0 grid).
Q_QUERY_FIXED = 9


def build_config(k_shot: int, n_way: int) -> dict:
    config = make_base_config(ablation_id=f"grid_A4_k{k_shot}_n{n_way}")

    # A4 flags — MAML, no MoE. Mirrors standalone A4 exactly.
    config["meta_learning"] = True
    config["use_MOE"]       = False

    # Capacity-match CNN encoder to ALL M0 experts combined — same as standalone A4.
    match_info = compute_matched_filters_for_ablation(
        ablation_id=f"grid_A4_k{k_shot}_n{n_way}",
        ablation_config=config,
        match_target="all_experts",
    )
    config["cnn_base_filters"] = match_info["matched_filters"]

    # Log param-match metadata for auditing (mirrors standalone A4).
    config["_param_match_target"]     = "all_experts_cnn"
    config["_m0_total_params"]        = match_info["m0_total_params"]
    config["_m0_all_expert_params"]   = match_info["m0_all_expert_params"]
    config["_m0_one_expert_params"]   = match_info["m0_one_expert_params"]
    config["_a4_matched_cnn_params"]  = match_info["matched_cnn_params"]
    config["_a4_total_params"]        = match_info["matched_total_params"]
    config["_a4_param_ratio"]         = match_info["param_ratio"]

    # Fix test procedure to hpo_test_split — L2SO is not feasible for the full
    # 9-cell MAML grid (see module docstring for reasoning).
    config["test_procedure"] = "hpo_test_split"

    # Grid cell-specific task dimensions — these drive the MAML episodic
    # dataloader AND the episodic eval sampler, so they must match.
    config["k_shot"]  = k_shot
    config["n_way"]   = n_way
    config["q_query"] = Q_QUERY_FIXED

    print(f"[grid_A4] k_shot={k_shot}  n_way={n_way}  q_query={Q_QUERY_FIXED}")
    print(f"[grid_A4] cnn_base_filters={config['cnn_base_filters']}  "
          f"param_ratio={match_info['param_ratio']:.4f}")
    return config


def run(k_shot: int, n_way: int) -> dict:
    config = build_config(k_shot, n_way)
    set_seeds(FIXED_SEED)
    config["seed"] = FIXED_SEED

    print(f"\n[grid_A4 k={k_shot} n={n_way} | seed={FIXED_SEED}]")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")
    print(f"  cnn_base_filters : {config['cnn_base_filters']}  "
          f"(match_target=all_experts, ratio={config['_a4_param_ratio']:.4f})")

    model = build_maml_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Parameters : {n_params:,}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
        periodic_checkpoint_fn=make_periodic_checkpoint_fn(config),
        periodic_test_eval_fn=make_periodic_test_eval_fn(
            tensor_dict_path, config["test_PIDs"]
        ),
        checkpoint_every=10,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[grid_A4 k={k_shot} n={n_way}] Training complete. "
          f"Best val acc = {best_val_acc:.4f}")

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

    print(f"[grid_A4 k={k_shot} n={n_way}] Test acc: "
          f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")

    result = {
        "ablation_id":        f"grid_A4_k{k_shot}_n{n_way}",
        "description":        (
            f"Few-Shot Grid A4 (MAML/No-MoE, all-expert capacity-matched): "
            f"k_shot={k_shot}, n_way={n_way}"
        ),
        "k_shot":             k_shot,
        "n_way":              n_way,
        "q_query":            Q_QUERY_FIXED,
        "seed":               FIXED_SEED,
        "best_val_acc":       float(best_val_acc),
        "test_results":       test_results,
        "n_params":           n_params,
        "cnn_base_filters":   config["cnn_base_filters"],
        "param_match_target": config["_param_match_target"],
        "matched_cnn_params": config["_a4_matched_cnn_params"],
        "param_ratio":        config["_a4_param_ratio"],
        "config_snapshot":    {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"k{k_shot}_n{n_way}_final")

    print(f"\n{'='*70}")
    print(f"[grid_A4] FINAL k={k_shot} n={n_way}:")
    print(f"  test acc : {test_results['mean_acc']*100:.2f}%  "
          f"±  {test_results['std_acc']*100:.2f}%")
    print(f"  q_query={Q_QUERY_FIXED}  seed={FIXED_SEED}")
    print(f"  cnn_base_filters={config['cnn_base_filters']}  "
          f"param_ratio={config['_a4_param_ratio']:.4f}")
    print(f"{'='*70}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot grid sweep (A4: MAML/No-MoE, all-expert capacity-matched) "
            "— one (k_shot, n_way) cell per job."
        )
    )
    parser.add_argument("--k-shot", type=int, required=True,
                        help=f"Number of support shots. Must be one of {GRID_K_SHOTS}.")
    parser.add_argument("--n-way", type=int, required=True,
                        help=f"Number of classes. Must be one of {GRID_N_WAYS}.")
    args = parser.parse_args()

    assert args.k_shot in GRID_K_SHOTS, (
        f"--k-shot {args.k_shot} is not in the defined grid {GRID_K_SHOTS}. "
        f"Update GRID_K_SHOTS in both fewshot_grid_A4.py and eval_launcher.sh "
        f"if you want to add a new value."
    )
    assert args.n_way in GRID_N_WAYS, (
        f"--n-way {args.n_way} is not in the defined grid {GRID_N_WAYS}. "
        f"Update GRID_N_WAYS in both fewshot_grid_A4.py and eval_launcher.sh "
        f"if you want to add a new value."
    )

    run(args.k_shot, args.n_way)


if __name__ == "__main__":
    main()