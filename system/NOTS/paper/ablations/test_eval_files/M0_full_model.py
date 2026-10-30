"""
M0_full_model.py
================
Ablation M0: Full Model — MAML + MoE  [PRIMARY RESULT]

Protocol (kfold7): 28/4 outer split (4 HPO-val users never tested), then 7 folds of
20 train / 4 val / 4 test over the 28. One SLURM job per fold (--fold-idx 0..6);
without --fold-idx all 7 folds run sequentially (local dev).

Training  : Episodic dataloader
Evaluation: Episodic, test_PIDs of the fold
Reported  : per-fold result, mean ± std across folds
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
    tensor_dict_file,
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    make_periodic_checkpoint_fn, make_periodic_test_eval_fn,
    RUN_DIR,
    apply_kfold_fold, KFOLD_N_FOLDS,
    add_smoke_arg, apply_smoke, add_seed_arg, kfold_tag, add_dump_logits_arg,
)
from MAML.maml_data_pipeline import get_maml_dataloaders

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0: Full Model (MAML + MoE)")
    parser.add_argument(
        "--fold-idx",
        type=int,
        default=None,
        help="kfold7 fold index 0..6 (one SLURM job per fold). Omit to run all folds sequentially.",
    )
    add_smoke_arg(parser)
    add_seed_arg(parser)
    add_dump_logits_arg(parser)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = apply_smoke(make_base_config(ablation_id="M0"), args.smoke)
    config["dump_query_logits"] = bool(args.dump_logits)
    return config


def run_one_fold(fold_id: str, seed: int, config: dict) -> dict:
    """
    Train and evaluate a single fold. config must already have
    train_PIDs, val_PIDs, test_PIDs set correctly before calling.
    """
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[M0 | {fold_id} | seed={seed}] Parameters: {n_params:,}")
    print(f"  train_PIDs : {config['train_PIDs']}")
    print(f"  val_PIDs   : {config['val_PIDs']}")
    print(f"  test_PIDs  : {config['test_PIDs']}")

    tensor_dict_path = tensor_dict_file(config)
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    from MAML.mamlpp import mamlpp_pretrain
    checkpoint_fn = make_periodic_checkpoint_fn(config)
    test_eval_fn  = make_periodic_test_eval_fn(tensor_dict_path, config["test_PIDs"])

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
        periodic_checkpoint_fn=checkpoint_fn,
        periodic_test_eval_fn=test_eval_fn,
        checkpoint_every=10,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[M0 | {fold_id} | seed={seed}] Training complete. Best val acc = {best_val_acc:.4f}")

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
    print(f"[M0 | {fold_id} | seed={seed}] Test acc: {test_results['mean_acc']*100:.2f}% "
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


def main():
    args   = parse_args()
    config = build_config(args)

    print("\nM0 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    fold_ids = [args.fold_idx] if args.fold_idx is not None else list(range(KFOLD_N_FOLDS))
    assert all(0 <= i < KFOLD_N_FOLDS for i in fold_ids), (
        f"--fold-idx must be in [0, {KFOLD_N_FOLDS - 1}]")
    results = []
    for i in fold_ids:
        fold_config = apply_kfold_fold(config, i)
        print(f"\n[M0] kfold7 fold {i}  test={fold_config['test_PIDs']}  val={fold_config['val_PIDs']}")
        results.append(run_one_fold(fold_id=f"kfold7_fold{i}", seed=args.seed, config=fold_config))
    accs = [r["test_results"]["mean_acc"] for r in results]
    summary = {
        "ablation_id":     "M0",
        "description":     "Full Model: MAML + MoE",
        "test_procedure":  "kfold7",
        "fold_idxs":       fold_ids,
        "seed":            args.seed,
        "n_params":        results[0]["n_params"],
        "fold_results":    results,
        "mean_test_acc":   float(np.mean(accs)),
        "std_test_acc":    float(np.std(accs)),
        "num_folds":       len(results),
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    tag = kfold_tag(fold_ids[0] if len(fold_ids) == 1 else None, args.seed)
    save_results(summary, config, tag=tag)
    print(f"[M0] FINAL kfold7 {fold_ids}: {summary['mean_test_acc']*100:.2f}% "
          f"+/- {summary['std_test_acc']*100:.2f}% over {len(results)} fold(s)")


if __name__ == "__main__":
    main()
