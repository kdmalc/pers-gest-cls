# A1_no_maml_moe.py
"""
A1_no_maml_moe.py
=================
Ablation A1: No-MAML + MoE (Supervised MoE)

Isolates the contribution of MAML. The MoE encoder is identical to M0, but
training uses a flat dataloader with standard cross-entropy (no meta-learning).
At eval time we finetune on the 1-shot support set before episodic evaluation,
reporting BOTH head-only and full fine-tuning results (per spec).

Changes from M0 (make_base_config):
  - meta_learning = False  → flat supervised training, not episodic
  - ft_steps     = maml_inner_steps_eval  (mirrors MAML inner-loop step count)
  - ft_lr        = maml_alpha_init_eval   (mirrors MAML inner-loop eval LR)
  - ft_optimizer = "sgd"                  (mirrors MAML inner-loop update rule:
                                           θ' = θ - α∇L is vanilla SGD)
  - ft_weight_decay = 0.0                 (MAML inner loop has no weight decay)

All HPO-tuned hyperparameters (lr, wd, label_smooth, architecture, MoE params,
etc.) are inherited unchanged from make_base_config() / M0's Trial 89 values.
Do NOT add overrides here for anything that was tuned during HPO.

The eval-time adaptation setup directly mirrors M0's MAML inner-loop eval so
that the only difference between M0 and A1 is whether the loss surface was
shaped by meta-learning — isolating exactly MAML's contribution.

Training : Flat dataloader (get_pretrain_dataloaders)
Evaluation: Episodic (1-shot 3-way), finetune before eval
            → reported as (head_only, full_ft)

test_procedure: 'kfold7' only (28/4 outer split, 7 folds of 20/4/4; --fold-idx 0..6,
                omitted = all 7 sequentially). Legacy protocols removed 2026-10-21.
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
    make_base_config, build_supervised_moe_model,
    set_seeds, FIXED_SEED,
    run_supervised_test_eval, save_results, save_model_checkpoint, count_parameters,
    replace_head_for_eval,
    RUN_DIR,
    run_kfold, add_fold_arg, kfold_tag, resolve_fold_ids, add_seed_arg,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain
from MAML.maml_data_pipeline import reorient_tensor_dict

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def build_config() -> dict:
    """
    A1 config: M0's HPO values unchanged; only ablation-defining flags set.

    Eval-time adaptation mirrors M0's MAML inner-loop eval setup exactly:
      ft_steps      = maml_inner_steps_eval  (same number of gradient steps)
      ft_lr         = maml_alpha_init_eval   (same per-step learning rate)
      ft_optimizer  = "sgd"                  (MAML inner loop is SGD:
                                              theta' = theta - alpha * grad)
      ft_weight_decay = 0.0                  (MAML inner loop has no WD)

    This means the only difference between M0 and A1 at eval time is whether
    the model's loss surface was shaped by meta-learning (M0) or standard
    supervised training (A1) — isolating exactly MAML's contribution.
    """
    config = make_base_config(ablation_id="A1")

    # ── Ablation-defining override ────────────────────────────────────────────
    # Disable meta-learning → flat supervised training loop instead of MAML.
    config["meta_learning"] = False

    # ── Eval-time adaptation: mirror M0's MAML inner-loop eval exactly ───────
    # maml_inner_steps_eval and maml_alpha_init_eval are set in make_base_config
    # from Trial 89 values and are guaranteed to exist.
    config["ft_steps"]        = config["maml_inner_steps_eval"]   # = 10
    config["ft_lr"]           = config["maml_alpha_init_eval"]    # = 5.066e-3
    config["ft_optimizer"]    = "sgd"    # MAML inner loop is SGD: theta' = theta - alpha*grad
    config["ft_weight_decay"] = 0.0     # MAML inner loop has no weight decay

    return config


def run_one_fold(fold_id: str, seed: int, config: dict, tensor_dict: dict) -> dict:
    """
    Train and evaluate one fold. config must already have
    train_PIDs, val_PIDs, test_PIDs set correctly before calling.
    """
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    print(f"\n[A1 | {fold_id} | seed={seed}]")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    model = build_supervised_moe_model(config)
    n_params = count_parameters(model)
    print(f"  Parameters : {n_params:,}")

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert n_classes == config["pretrain_num_classes"], (
        f"Flat dataloader returned {n_classes} classes but pretrain_num_classes="
        f"{config['pretrain_num_classes']}."
    )

    sample_batch = next(iter(train_dl))
    labels = sample_batch["labels"]
    assert labels.min() >= 0 and labels.max() < config["pretrain_num_classes"], (
        f"Label range [{labels.min()}, {labels.max()}] out of bounds for "
        f"pretrain_num_classes={config['pretrain_num_classes']}."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else float("nan")
    print(f"[A1 | {fold_id}] Training complete. Best val acc = {best_val_acc:.4f}")

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

    print(f"[A1 | {fold_id}] Test head-only: {head_results['mean_acc']*100:.2f}% "
          f"± {head_results['std_acc']*100:.2f}%")
    print(f"[A1 | {fold_id}] Test full-ft  : {full_results['mean_acc']*100:.2f}% "
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main  (protocol: kfold7 only -- 28/4 outer split, 7 folds of 20/4/4)
# Legacy 'hpo_test_split' / 'L2SO' were removed from this script on 2026-10-21;
# see git history to reproduce old numbers.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="A1: No-MAML + MoE")
    add_fold_arg(parser)
    add_seed_arg(parser)
    args = parser.parse_args()

    config = build_config()
    assert config["test_procedure"] == "kfold7", config["test_procedure"]

    print("\nA1 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    tensor_dict_path = tensor_dict_file(config)
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    all_results = run_kfold(
        config, args.fold_idx,
        functools.partial(run_one_fold, tensor_dict=tensor_dict),
        seed=args.seed,
    )
    head_accs = [r["test_head_only"]["mean_acc"] for r in all_results]
    full_accs = [r["test_full_ft"]["mean_acc"]   for r in all_results]
    summary = {
        "ablation_id":         "A1",
        "description":         "No-MAML + MoE (Supervised MoE)",
        "test_procedure":      "kfold7",
        "fold_idxs":           resolve_fold_ids(args.fold_idx),
        "seed":                args.seed,
        "n_params":            all_results[0]["n_params"],
        "fold_results":        all_results,
        "mean_test_head_only": float(np.mean(head_accs)),
        "std_test_head_only":  float(np.std(head_accs)),
        "mean_test_full_ft":   float(np.mean(full_accs)),
        "std_test_full_ft":    float(np.std(full_accs)),
        "ft_steps":            config["ft_steps"],
        "ft_lr":               config["ft_lr"],
        "ft_optimizer":        config["ft_optimizer"],
        "ft_weight_decay":     config["ft_weight_decay"],
        "num_folds":           len(all_results),
        "config_snapshot":     {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag=kfold_tag(args.fold_idx, args.seed))

    print(f"\n{'='*70}")
    print(f"[A1] FINAL head-only (kfold7 {summary['fold_idxs']}): "
          f"{summary['mean_test_head_only']*100:.2f}% Â± {summary['std_test_head_only']*100:.2f}%")
    print(f"[A1] FINAL full-ft   (kfold7 {summary['fold_idxs']}): "
          f"{summary['mean_test_full_ft']*100:.2f}% Â± {summary['std_test_full_ft']*100:.2f}%")
    print(f"     ft_steps={config['ft_steps']}  ft_lr={config['ft_lr']:.4e}"
          f"  ft_optimizer={config['ft_optimizer']}")
    print(f"     {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
