# A3_A4_maml_no_moe.py
"""
A3_A4_maml_no_moe.py
=====================
Ablations A3 and A4: MAML + No-MoE

A3: Single encoder, natural (smaller) parameter count.
A4: Single encoder scaled to match M0 expert CNN parameter count. CRITICAL ablation.

test_procedure:
  'hpo_test_split' : Fixed split, single seed run (FIXED_SEED).
  'L2SO'           : Leave-2-Subjects-Out over all_PIDs. One run per fold.
"""

import os, sys, copy, json, argparse
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config, build_maml_moe_model, build_maml_no_moe_model,
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import mamlpp_pretrain
from pretraining.pretrain_models import build_model

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter matching utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_moe_encoder_params(moe_model: nn.Module) -> int:
    if hasattr(moe_model, "experts"):
        return sum(p.numel() for p in moe_model.experts.parameters() if p.requires_grad)
    expert_params = 0
    for name, module in moe_model.named_modules():
        if "expert" in name.lower() and len(list(module.children())) == 0:
            expert_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    assert expert_params > 0, (
        "Could not find 'experts' submodule in MoE model. "
        "Inspect the model architecture and update count_moe_encoder_params()."
    )
    return expert_params


def find_matched_cnn_filters(target_cnn_params: int, config_template: dict,
                              lstm_hidden: int, search_range=(64, 512)) -> int:
    from pretraining.pretrain_models import build_model as _build
    import math

    gn_groups = config_template["groupnorm_num_groups"]

    def _cnn_param_count(filters: int) -> int:
        cfg = copy.deepcopy(config_template)
        cfg["cnn_base_filters"] = filters
        cfg["lstm_hidden"]      = lstm_hidden
        cfg["use_MOE"]          = False
        m = _build(cfg)
        if hasattr(m, "cnn") or hasattr(m, "conv"):
            cnn_mod = getattr(m, "cnn", None) or getattr(m, "conv", None)
            return sum(p.numel() for p in cnn_mod.parameters() if p.requires_grad)
        total = 0
        for name, p in m.named_parameters():
            if p.requires_grad and "lstm" not in name.lower() and "head" not in name.lower():
                total += p.numel()
        return total

    lo, hi = search_range
    lo = math.ceil(lo / gn_groups) * gn_groups

    best_filters = lo
    best_diff = abs(_cnn_param_count(lo) - target_cnn_params)

    for f in range(lo, hi + 1, gn_groups):
        diff = abs(_cnn_param_count(f) - target_cnn_params)
        if diff < best_diff:
            best_diff = diff
            best_filters = f

    return best_filters


# ─────────────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────────────

def build_config_a3() -> dict:
    config = make_base_config(ablation_id="A3")
    config["use_MOE"] = False
    return config


def build_config_a4(moe_config: dict) -> dict:
    config = make_base_config(ablation_id="A4")
    config["use_MOE"] = False

    moe_model = build_maml_moe_model(moe_config)
    m0_total_params  = count_parameters(moe_model)
    m0_expert_params = count_moe_encoder_params(moe_model)
    del moe_model

    print(f"\n[A4] M0 total params       : {m0_total_params:,}")
    print(f"[A4] M0 expert CNN params  : {m0_expert_params:,}  (A4 encoder must match this)")

    matched_filters = find_matched_cnn_filters(
        target_cnn_params=m0_expert_params,
        config_template=config,
        lstm_hidden=config["lstm_hidden"],
    )
    config["cnn_base_filters"] = matched_filters

    a4_model = build_maml_no_moe_model(config)
    a4_total_params = count_parameters(a4_model)
    del a4_model

    print(f"[A4] Selected cnn_base_filters = {matched_filters}")
    print(f"[A4] A4 total params           = {a4_total_params:,}")
    print(f"[A4] Parameter ratio (A4 / M0 experts) = "
          f"{a4_total_params / max(m0_expert_params, 1):.3f}  (target ≈ 1.0)")

    config["_a4_matched_filters"] = matched_filters
    config["_m0_expert_params"]   = m0_expert_params
    config["_a4_total_params"]    = a4_total_params

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Core training + eval for one fold
# ─────────────────────────────────────────────────────────────────────────────

def run_one_fold(ablation_id: str, fold_id: str, seed: int, config: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    print(f"\n[{ablation_id} | {fold_id} | seed={seed}]")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    model = build_maml_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Parameters : {n_params:,}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
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
    """Single deterministic run on the fixed train/val/test split."""
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
        f"Unknown test_procedure '{test_procedure}'. Must be 'hpo_test_split' or 'L2SO'."
    )

    if test_procedure == "hpo_test_split":
        all_results = run_hpo_test_split(ablation_id, config)
        test_accs = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":     ablation_id,
            "description":     description,
            "test_procedure":  "hpo_test_split",
            "n_params":        all_results[0]["n_params"],
            "fold_results":    all_results,
            "mean_test_acc":   float(np.mean(test_accs)),
            "std_test_acc":    float(np.std(test_accs)),
            "seed":            FIXED_SEED,
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }
    else:  # L2SO
        all_results = run_l2so(ablation_id, config)
        test_accs = [r["test_results"]["mean_acc"] for r in all_results]
        summary = {
            "ablation_id":     ablation_id,
            "description":     description,
            "test_procedure":  "L2SO",
            "n_params":        all_results[0]["n_params"],
            "fold_results":    all_results,
            "mean_test_acc":   float(np.mean(test_accs)),
            "std_test_acc":    float(np.std(test_accs)),
            "num_folds":       len(all_results),
            "config_snapshot": {k: str(v) for k, v in config.items()},
        }

    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[{ablation_id}] FINAL ({test_procedure}): "
          f"{summary['mean_test_acc']*100:.2f}%")
    if test_procedure == "L2SO":
        print(f"     ± {summary['std_test_acc']*100:.2f}%  "
              f"over {summary['num_folds']} L2SO folds (one per test subject)")
    else:
        print(f"     single run, seed={FIXED_SEED}, fixed split")
    print(f"     {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=["A3", "A4"],
        required=True,  # no default — must be explicit to avoid running the wrong thing
        help="Which ablation to run: A3 (natural params) or A4 (parameter-matched to M0 experts).",
    )
    args = parser.parse_args()

    if args.ablation == "A3":
        config_a3 = build_config_a3()
        run_ablation("A3", config_a3, "MAML + No-MoE (Single Expert, Reduced Parameters)")

    elif args.ablation == "A4":
        moe_config = make_base_config(ablation_id="M0_ref")
        config_a4  = build_config_a4(moe_config)
        run_ablation("A4", config_a4,
                     "MAML + No-MoE (Parameter-Matched Encoder) ← CRITICAL")


if __name__ == "__main__":
    main()