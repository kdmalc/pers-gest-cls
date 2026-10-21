# A2_no_maml_no_moe.py
"""
A2_no_maml_no_moe.py
=====================
Ablation A2: No-MAML + No-MoE (Vanilla CNN-LSTM, Parameter-Matched to ALL M0 Experts)

Changes from M0:
  - No MAML (flat supervised pretraining via pretrain_trainer).
  - No MoE: single CNN-LSTM encoder instead of expert pool.
  - cnn_base_filters is grid-searched so the single encoder's CNN-only param
    count matches the SUM of all M0 expert CNN params. LSTM and head are
    identical to M0 and are excluded from the matching equation.

This is architecturally identical to A4 (same encoder size, same LSTM, same head).
The only difference is training regime: A2 uses supervised pretraining, A4 uses MAML.
Comparing A2 vs A4 isolates the contribution of MAML at equal model capacity.
Comparing A4 vs M0 isolates the contribution of MoE at equal total expert capacity.

test_procedure: 'kfold7' only (28/4 outer split, 7 folds of 20/4/4).

CLI args:
  --fold-idx {0..6}   run one fold (cluster: one job per fold); omitted = all 7 sequentially
"""

import os, sys, copy, json, argparse, functools
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
    tensor_dict_file,
    make_base_config, build_supervised_no_moe_model,
    set_seeds, FIXED_SEED,
    run_supervised_test_eval, save_results, save_model_checkpoint, count_parameters,
    compute_matched_filters_for_ablation,
    RUN_DIR,
    run_kfold, add_fold_arg, kfold_tag, resolve_fold_ids,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain
from MAML.maml_data_pipeline import reorient_tensor_dict

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def build_config() -> dict:
    """
    A2 config: M0's HPO values inherited; only ablation-defining flags set.

    Eval-time adaptation mirrors M0's MAML inner-loop eval exactly:
      ft_steps      = maml_inner_steps_eval  (same number of gradient steps)
      ft_lr         = maml_alpha_init_eval   (same per-step learning rate)
      ft_optimizer  = "sgd"                  (MAML inner loop is SGD:
                                              theta' = theta - alpha * grad)
      ft_weight_decay = 0.0                  (MAML inner loop has no WD)
    """
    config = make_base_config(ablation_id="A2")

    # ── Ablation-defining overrides ───────────────────────────────────────────
    config["meta_learning"] = False
    config["use_MOE"]       = False

    # ── Eval-time adaptation: mirror M0's MAML inner-loop eval exactly ───────
    config["ft_steps"]        = config["maml_inner_steps_eval"]  # = 10
    config["ft_lr"]           = config["maml_alpha_init_eval"]   # = 5.066e-3
    config["ft_optimizer"]    = "sgd"   # MAML inner loop is SGD: theta' = theta - alpha*grad
    config["ft_weight_decay"] = 0.0    # MAML inner loop has no weight decay

    # ── Parameter matching ────────────────────────────────────────────────────
    # Target: SUM of all M0 expert CNN params (not one expert — that's A3).
    # LSTM and head are excluded from both sides of the match so we're only
    # scaling the CNN encoder to have the same total capacity as M0's expert pool.
    match_info = compute_matched_filters_for_ablation(
        ablation_id="A2",
        ablation_config=config,
        match_target="all_experts",
    )
    config["cnn_base_filters"] = match_info["matched_filters"]

    # Stash matching metadata for the saved config snapshot / auditing
    config["_param_match_target"]          = "all_experts_cnn"
    config["_m0_total_params"]             = match_info["m0_total_params"]
    config["_m0_all_expert_params"]        = match_info["m0_all_expert_params"]
    config["_m0_one_expert_params"]        = match_info["m0_one_expert_params"]
    config["_a2_matched_cnn_params"]       = match_info["matched_cnn_params"]
    config["_a2_total_params_after_match"] = match_info["matched_total_params"]
    config["_a2_param_ratio"]              = match_info["param_ratio"]

    return config


def run_one_fold(fold_id: str, seed: int, config: dict, tensor_dict: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    print(f"\n[A2 | {fold_id} | seed={seed}]")
    print(f"  train_PIDs          : {config['train_PIDs']}")
    print(f"  val_PIDs            : {config['val_PIDs']}")
    print(f"  test_PIDs           : {config['test_PIDs']}")
    print(f"  cnn_base_filters    : {config['cnn_base_filters']}  (matched to all M0 experts)")
    print(f"  CNN param ratio     : {config['_a2_param_ratio']:.4f}")

    model = build_supervised_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Total parameters    : {n_params:,}")

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert n_classes == config["pretrain_num_classes"], (
        f"Expected {config['pretrain_num_classes']} classes (pretrain_num_classes), got {n_classes}."
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

    tensor_dict_path = tensor_dict_file(config)

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


# Protocol: kfold7 only (28/4 outer split, 7 folds of 20/4/4). Legacy protocols removed 2026-10-21.


def main():
    parser = argparse.ArgumentParser(description="A2: No-MAML + No-MoE")
    add_fold_arg(parser)
    args = parser.parse_args()

    config = build_config()
    assert config["test_procedure"] == "kfold7", config["test_procedure"]
    print(f"[A2] ft_lr={config['ft_lr']:.4e}  ft_steps={config['ft_steps']}  "
          f"ft_optimizer={config['ft_optimizer']}  (mirroring M0 MAML inner-loop eval)")

    tensor_dict_path = tensor_dict_file(config)
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    all_results = run_kfold(
        config, args.fold_idx,
        functools.partial(run_one_fold, tensor_dict=tensor_dict),
    )
    head_accs = [r["test_head_only"]["mean_acc"] for r in all_results]
    full_accs = [r["test_full_ft"]["mean_acc"]   for r in all_results]
    summary = {
        "ablation_id":             "A2",
        "description":             "No-MAML + No-MoE (CNN matched to ALL M0 experts combined)",
        "test_procedure":          "kfold7",
        "fold_idxs":               resolve_fold_ids(args.fold_idx),
        "seed":                    FIXED_SEED,
        "n_params":                all_results[0]["n_params"],
        "fold_results":            all_results,
        "mean_test_head_only":     float(np.mean(head_accs)),
        "std_test_head_only":      float(np.std(head_accs)),
        "mean_test_full_ft":       float(np.mean(full_accs)),
        "std_test_full_ft":        float(np.std(full_accs)),
        "ft_steps":                config["ft_steps"],
        "ft_lr":                   config["ft_lr"],
        "ft_optimizer":            config["ft_optimizer"],
        "ft_weight_decay":         config["ft_weight_decay"],
        "num_folds":               len(all_results),
        "cnn_base_filters":        config["cnn_base_filters"],
        "param_match_target":      "all_experts_cnn",
        "m0_all_expert_params":    config["_m0_all_expert_params"],
        "matched_cnn_params":      config["_a2_matched_cnn_params"],
        "param_ratio":             config["_a2_param_ratio"],
        "config_snapshot":         {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag=kfold_tag(args.fold_idx))

    print(f"\n{'='*70}")
    print(f"[A2] FINAL head-only (kfold7 {summary['fold_idxs']}): "
          f"{summary['mean_test_head_only']*100:.2f}% ± {summary['std_test_head_only']*100:.2f}%")
    print(f"[A2] FINAL full-ft   (kfold7 {summary['fold_idxs']}): "
          f"{summary['mean_test_full_ft']*100:.2f}% ± {summary['std_test_full_ft']*100:.2f}%")
    print(f"     ft_lr={config['ft_lr']:.4e}  ft_steps={config['ft_steps']}  ft_optimizer={config['ft_optimizer']}")
    print(f"     cnn_base_filters={config['cnn_base_filters']}  "
          f"(matched to ALL M0 experts, ratio={config['_a2_param_ratio']:.4f})")
    print(f"     {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
