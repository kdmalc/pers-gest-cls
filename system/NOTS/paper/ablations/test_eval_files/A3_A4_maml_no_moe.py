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

test_procedure: 'kfold7' only (28/4 outer split, 7 folds of 20/4/4; --fold-idx 0..6).
"""

import os, sys, copy, json, argparse, functools
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
    tensor_dict_file,
    make_base_config, build_maml_no_moe_model,
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    make_periodic_checkpoint_fn, make_periodic_test_eval_fn,
    compute_matched_filters_for_ablation,
    RUN_DIR,
    run_kfold, add_fold_arg, kfold_tag, resolve_fold_ids, add_smoke_arg, apply_smoke,
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

    tensor_dict_path = tensor_dict_file(config)
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
# Orchestration  (protocol: kfold7 only; legacy protocols removed 2026-10-21)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(ablation_id: str, config: dict, description: str, fold_idx=None):
    assert config["test_procedure"] == "kfold7", config["test_procedure"]

    print(f"\n{ablation_id} CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    param_ratio_key = f"_{ablation_id.lower()}_param_ratio"
    matched_cnn_key = f"_{ablation_id.lower()}_matched_cnn_params"

    all_results = run_kfold(
        config, fold_idx,
        functools.partial(run_one_fold, ablation_id),
    )
    test_accs = [r["test_results"]["mean_acc"] for r in all_results]
    summary = {
        "ablation_id":        ablation_id,
        "description":        description,
        "test_procedure":     "kfold7",
        "fold_idxs":          resolve_fold_ids(fold_idx),
        "n_params":           all_results[0]["n_params"],
        "fold_results":       all_results,
        "mean_test_acc":      float(np.mean(test_accs)),
        "std_test_acc":       float(np.std(test_accs)),
        "num_folds":          len(all_results),
        "seed":               FIXED_SEED,
        "cnn_base_filters":   config["cnn_base_filters"],
        "param_match_target": config["_param_match_target"],
        "matched_cnn_params": config[matched_cnn_key],
        "param_ratio":        config[param_ratio_key],
        "config_snapshot":    {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag=kfold_tag(fold_idx))

    param_ratio = config[param_ratio_key]
    print(f"\n{'='*70}")
    print(f"[{ablation_id}] FINAL (kfold7 {summary['fold_idxs']}): "
          f"{summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%  "
          f"over {summary['num_folds']} fold(s)")
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
    add_fold_arg(parser)
    add_smoke_arg(parser)
    args = parser.parse_args()

    if args.ablation == "A3":
        run_ablation("A3", apply_smoke(build_config_a3(), args.smoke),
                     "MAML + No-MoE (single encoder matched to ONE M0 expert)",
                     fold_idx=args.fold_idx)
    else:
        run_ablation("A4", apply_smoke(build_config_a4(), args.smoke),
                     "MAML + No-MoE (single encoder matched to ALL M0 experts) ← CRITICAL",
                     fold_idx=args.fold_idx)


if __name__ == "__main__":
    main()
