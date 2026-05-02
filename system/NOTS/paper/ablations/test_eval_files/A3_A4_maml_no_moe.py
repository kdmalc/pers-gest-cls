# A3_A4_maml_no_moe.py
"""
A3_A4_maml_no_moe.py
=====================
Ablations A3 and A4: MAML + No-MoE

A3: Single encoder matched to ONE M0 expert (natural small baseline).
    This answers: "what does a single-expert-capacity model + MAML achieve?"

A4: Single encoder matched to ALL M0 experts combined (CRITICAL capacity ablation).
    This answers: "does MoE help beyond just having more parameters?"
    A4 architecture is identical to A2 — only the training regime differs (MAML vs supervised).
    Comparing A4 vs A2 isolates MAML. Comparing A4 vs M0 isolates MoE.

LSTM and head are identical to M0 and are NOT part of the matching equation.
Only the CNN encoder (what M0 implements as an expert pool) is scaled.

test_procedure:
  'hpo_test_split' : Fixed split, single seed run (FIXED_SEED).
  'L2SO'           : Leave-2-Subjects-Out over all_PIDs. One run per fold.
"""

import os, sys, copy, json, argparse
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


# ─────────────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────────────

def build_config_a3() -> dict:
    """
    A3: MAML + single encoder sized to match ONE M0 expert.
    Natural small baseline — shows what MAML alone achieves at low capacity.
    cnn_base_filters will be at or near the M0 default (64), since we're
    matching just one expert's CNN params.
    """
    config = make_base_config(ablation_id="A3")
    config["use_MOE"] = False

    match_info = compute_matched_filters_for_ablation(
        ablation_id="A3",
        ablation_config=config,
        match_target="one_expert",
    )
    config["cnn_base_filters"] = match_info["matched_filters"]

    config["_param_match_target"]          = "one_expert_cnn"
    config["_m0_total_params"]             = match_info["m0_total_params"]
    config["_m0_all_expert_params"]        = match_info["m0_all_expert_params"]
    config["_m0_one_expert_params"]        = match_info["m0_one_expert_params"]
    config["_a3_matched_cnn_params"]       = match_info["matched_cnn_params"]
    config["_a3_total_params_after_match"] = match_info["matched_total_params"]
    config["_a3_param_ratio"]              = match_info["param_ratio"]

    return config


def build_config_a4() -> dict:
    """
    A4: MAML + single encoder sized to match ALL M0 experts combined.
    Critical capacity-controlled ablation. Architecturally identical to A2
    (same cnn_base_filters, same LSTM, same head). Only the training regime
    differs: A4 uses MAML, A2 uses supervised pretraining.
    """
    config = make_base_config(ablation_id="A4")
    config["use_MOE"] = False

    match_info = compute_matched_filters_for_ablation(
        ablation_id="A4",
        ablation_config=config,
        match_target="all_experts",
    )
    config["cnn_base_filters"] = match_info["matched_filters"]

    config["_param_match_target"]          = "all_experts_cnn"
    config["_m0_total_params"]             = match_info["m0_total_params"]
    config["_m0_all_expert_params"]        = match_info["m0_all_expert_params"]
    config["_m0_one_expert_params"]        = match_info["m0_one_expert_params"]
    config["_a4_matched_cnn_params"]       = match_info["matched_cnn_params"]
    config["_a4_total_params_after_match"] = match_info["matched_total_params"]
    config["_a4_param_ratio"]              = match_info["param_ratio"]

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Core training + eval for one fold
# ─────────────────────────────────────────────────────────────────────────────

def run_one_fold(ablation_id: str, fold_id: str, seed: int, config: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    match_target = config.get("_param_match_target", "unknown")
    param_ratio_key = f"_{ablation_id.lower()}_param_ratio"
    param_ratio = config.get(param_ratio_key, float("nan"))

    print(f"\n[{ablation_id} | {fold_id} | seed={seed}]")
    print(f"  train_PIDs       : {config['train_PIDs']}")
    print(f"  val_PIDs         : {config['val_PIDs']}")
    print(f"  test_PIDs        : {config['test_PIDs']}")
    print(f"  cnn_base_filters : {config['cnn_base_filters']}  (match_target={match_target})")
    print(f"  CNN param ratio  : {param_ratio:.4f}")

    model = build_maml_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Total parameters : {n_params:,}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
        periodic_checkpoint_fn=make_periodic_checkpoint_fn(config),
        periodic_test_eval_fn=make_periodic_test_eval_fn(tensor_dict_path, config["test_PIDs"]),
        checkpoint_every=10,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[{ablation_id} | {fold_id}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "fold_id":           fold_id,
            "seed":              seed,
            "model_state_dict":  train_history["best_state"],
            "config":            config,
            "best_val_acc":      best_val_acc,
            "train_loss_log":    train_history["train_loss_log"],
            "val_acc_log":       train_history["val_acc_log"],
        },
        config,
        tag=f"{fold_id}_seed{seed}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[{ablation_id} | {fold_id}] Test acc: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")

    return {
        "fold_id":      fold_id,
        "seed":         seed,
        "test_PID":     config["test_PIDs"],
        "val_PID":      config["val_PIDs"],
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test procedures
# ─────────────────────────────────────────────────────────────────────────────

def run_hpo_test_split(ablation_id: str, config: dict) -> list:
    print(f"\n{'='*70}")
    print(f"[{ablation_id}] hpo_test_split: single run (seed={FIXED_SEED})")
    print(f"{'='*70}")
    result = run_one_fold(
        ablation_id=ablation_id,
        fold_id=f"fixed_seed{FIXED_SEED}",
        seed=FIXED_SEED,
        config=copy.deepcopy(config),
    )
    return [result]


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


def run_l2so(ablation_id: str, config: dict) -> list:
    all_pids = config["all_PIDs"]
    assert len(all_pids) >= 3, "Need at least 3 subjects for L2SO."

    folds = build_l2so_folds(all_pids)
    results = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        print(f"\n{'='*70}")
        print(f"[{ablation_id}] L2SO fold {fold_idx+1}/{len(folds)}  "
              f"test={fold['test_pid']}  val={fold['val_pid']}")
        print(f"{'='*70}")

        fold_config = copy.deepcopy(config)
        fold_config["train_PIDs"] = fold["train_pids"]
        fold_config["val_PIDs"]   = [fold["val_pid"]]
        fold_config["test_PIDs"]  = [fold["test_pid"]]

        result = run_one_fold(
            ablation_id=ablation_id,
            fold_id=f"l2so_fold{fold_idx:02d}_test{fold['test_pid']}",
            seed=FIXED_SEED,
            config=fold_config,
        )
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(ablation_id: str, config: dict, description: str):
    test_procedure = config["test_procedure"]

    print(f"\n{ablation_id} CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"\nTest procedure: {test_procedure}")

    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'."
    )

    param_ratio_key = f"_{ablation_id.lower()}_param_ratio"
    matched_cnn_key = f"_{ablation_id.lower()}_matched_cnn_params"

    if test_procedure == "hpo_test_split":
        all_results = run_hpo_test_split(ablation_id, config)
        test_accs = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":        ablation_id,
            "description":        description,
            "test_procedure":     "hpo_test_split",
            "n_params":           all_results[0]["n_params"],
            "fold_results":       all_results,
            "mean_test_acc":      float(np.mean(test_accs)),
            "std_test_acc":       float(np.std(test_accs)),
            "seed":               FIXED_SEED,
            "cnn_base_filters":   config["cnn_base_filters"],
            "param_match_target": config["_param_match_target"],
            "matched_cnn_params": config[matched_cnn_key],
            "param_ratio":        config[param_ratio_key],
            "config_snapshot":    {k: str(v) for k, v in config.items()},
        }
    else:  # L2SO
        all_results = run_l2so(ablation_id, config)
        test_accs = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":        ablation_id,
            "description":        description,
            "test_procedure":     "L2SO",
            "n_params":           all_results[0]["n_params"],
            "fold_results":       all_results,
            "mean_test_acc":      float(np.mean(test_accs)),
            "std_test_acc":       float(np.std(test_accs)),
            "num_folds":          len(all_results),
            "cnn_base_filters":   config["cnn_base_filters"],
            "param_match_target": config["_param_match_target"],
            "matched_cnn_params": config[matched_cnn_key],
            "param_ratio":        config[param_ratio_key],
            "config_snapshot":    {k: str(v) for k, v in config.items()},
        }

    save_results(summary, config, tag="summary")

    param_ratio = config[param_ratio_key]
    print(f"\n{'='*70}")
    print(f"[{ablation_id}] FINAL ({test_procedure}): "
          f"{summary['mean_test_acc']*100:.2f}%")
    if test_procedure == "L2SO":
        print(f"     ± {summary['std_test_acc']*100:.2f}%  "
              f"over {summary['num_folds']} L2SO folds")
    else:
        print(f"     single run, seed={FIXED_SEED}")
    print(f"     cnn_base_filters={config['cnn_base_filters']}  "
          f"(match_target={config['_param_match_target']}, ratio={param_ratio:.4f})")
    print(f"     {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=["A3", "A4"],
        required=True,
        help=(
            "A3: MAML, single encoder matched to ONE M0 expert (small baseline). "
            "A4: MAML, single encoder matched to ALL M0 experts combined (critical ablation)."
        ),
    )
    args = parser.parse_args()

    if args.ablation == "A3":
        config_a3 = build_config_a3()
        run_ablation("A3", config_a3,
                     "MAML + No-MoE (single encoder matched to ONE M0 expert)")

    elif args.ablation == "A4":
        config_a4 = build_config_a4()
        run_ablation("A4", config_a4,
                     "MAML + No-MoE (single encoder matched to ALL M0 experts) ← CRITICAL")


if __name__ == "__main__":
    main()