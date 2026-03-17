"""
run_pretrain.py
===============
Entry point: train one or all three architectures and evaluate them.

Example usage:
    # Train all three with default configs
    python run_pretrain.py --tensor_dict data/tensor_dict.pkl --all

    # Train just MetaCNNLSTM
    python run_pretrain.py --tensor_dict data/tensor_dict.pkl --model MetaCNNLSTM

    # Train with a specific config JSON (output from HPO)
    python run_pretrain.py --tensor_dict data/tensor_dict.pkl --model TST --config best_tst_config.json
"""

import os
import json
import argparse
import torch
import pickle

from pretrain_models import build_model
from pretrain_data_pipeline import get_pretrain_dataloaders
from pretrain_trainer import pretrain
from pretrain_eval import run_full_eval
from pretrain_configs import PRETRAIN_CONFIG, MODEL_CONFIGS


def train_one_model(model_type: str, tensor_dict_path: str, config_overrides: dict = None, save_dir: str = "checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    # Build config
    config = {
        **PRETRAIN_CONFIG,
        **MODEL_CONFIGS[model_type],
        "model_type": model_type,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if config_overrides:
        config.update(config_overrides)

    # Data
    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict_path)
    print(f"[run] n_classes = {n_classes}")

    # Model
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[run] {model_type} | {n_params:,} trainable parameters")

    # Train
    save_path = os.path.join(save_dir, f"{model_type}_best.pt")
    model, history = pretrain(model, train_dl, val_dl, config, save_path=save_path)

    # Save history
    hist_path = os.path.join(save_dir, f"{model_type}_history.json")
    json.dump(
        {k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
         for k, vals in history.items() if k != 'clf'},
        open(hist_path, 'w'), indent=2
    )
    print(f"[run] History saved → {hist_path}")

    return model, history, config


def eval_one_model(model, config: dict, tensor_dict_path: str, save_dir: str = "eval_results", method: str = "pca"):
    os.makedirs(save_dir, exist_ok=True)
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)
    results = run_full_eval(model, tensor_dict, config, save_dir=save_dir, method=method)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # LOCAL LAPTOP: C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\meta-learning-sup-que-ds\\maml_tensor_dict.pkl
    # REMOTE NOTS: /projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/maml_tensor_dict.pkl
    # NOTE: required=True ignores default lol 
    parser.add_argument("--tensor_dict", type=str, required=False, default='C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\meta-learning-sup-que-ds\\maml_tensor_dict.pkl')
    parser.add_argument("--model",       type=str, default="TST",
                        choices=["MetaCNNLSTM", "DeepCNNLSTM", "TST"])
    parser.add_argument("--all",         action="store_true",
                        help="Train all three models sequentially")
    parser.add_argument("--config",      type=str, default=None,
                        help="Path to JSON config override")
    parser.add_argument("--eval_method", type=str, default="pca",
                        choices=["pca", "tsne"])
    parser.add_argument("--save_dir",    type=str, default="pretrain_outputs")
    args = parser.parse_args()

    config_overrides = {}
    if args.config:
        with open(args.config) as f:
            config_overrides = json.load(f)

    models_to_run = ["MetaCNNLSTM", "DeepCNNLSTM", "TST"] if args.all else [args.model]
    assert None not in models_to_run, "Specify --model or --all"

    all_results = {}
    for mtype in models_to_run:
        print(f"\n{'#'*70}")
        print(f"# Training: {mtype}")
        print(f"{'#'*70}\n")

        model, history, config = train_one_model(
            mtype, args.tensor_dict,
            config_overrides = config_overrides,
            save_dir         = os.path.join(args.save_dir, "checkpoints"),
        )

        results = eval_one_model(
            model, config, args.tensor_dict,
            save_dir = os.path.join(args.save_dir, "eval", mtype),
            method   = args.eval_method,
        )
        all_results[mtype] = results

    # Final comparison table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'In-Dist Probe':>15} {'OOD Probe':>12}")
    print(f"{'-'*50}")
    for mtype, res in all_results.items():
        ind = res['probe_in_dist']['accuracy'] * 100
        ood = res['probe_out_dist']['accuracy'] * 100
        print(f"{mtype:<20} {ind:>14.1f}% {ood:>11.1f}%")
    print(f"{'='*60}")
