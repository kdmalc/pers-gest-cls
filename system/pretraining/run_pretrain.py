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

    # Train a MoE model and emit routing analysis every 5 epochs + after training
    python run_pretrain.py --tensor_dict data/tensor_dict.pkl --model DeepCNNLSTM --config moe_config.json

MoE notes:
    When config["use_moe"] = True the following extra behaviour is activated:
      - pretrain_trainer.py logs routing reports every moe_log_every epochs (already wired).
      - After training, run_full_eval() calls run_moe_routing_eval() on both the train
        and val DataLoaders using the *best* checkpoint weights.  Plots go to:
            {save_dir}/eval/{model}/moe_routing/{train|val}/
        and the raw RoutingRecord is saved as a .pt file for offline inspection.
    Relevant config keys (all have sensible defaults):
        use_moe        : bool  — master switch
        num_experts    : int   — number of MoE experts (required)
        moe_aux_coeff  : float — load-balancing loss coefficient (default 1e-2)
        moe_log_every  : int   — epoch interval for in-training routing logs (default 5)
        moe_plot_dir   : str   — overrides default plot dir for in-training plots
        demo_dim_labels: list  — optional human-readable labels for demographics dims
"""
 
import os
import json
import argparse
import torch
import pickle
from datetime import datetime
 
from pretrain_models import build_model
from pretrain_data_pipeline import get_pretrain_dataloaders
from pretrain_trainer import pretrain
from pretrain_eval import run_full_eval, viz_from_checkpoint
from pretrain_configs import PRETRAIN_CONFIG, MODEL_CONFIGS


def get_run_timestamp() -> str:
    """Return a timestamp string for use in checkpoint filenames: MMDDYYYY_HHMMSS."""
    return datetime.now().strftime("%m%d%Y_%H%M%S")


def make_checkpoint_paths(save_dir: str, model_type: str, timestamp: str) -> dict:
    """
    Build the standardised checkpoint paths for a single run.

    Naming convention: {ModelName}_{MMDDYYYY_HHMMSS}_{type}.pt

    Returns a dict with keys:
        'best' : path for the best-val-loss checkpoint (saved during training)
        'last' : path for the final-epoch checkpoint   (saved after training)
    """
    base = f"{model_type}_{timestamp}"
    return {
        "best": os.path.join(save_dir, f"{base}_best.pt"),
        "last": os.path.join(save_dir, f"{base}_last.pt"),
    }


def load_tensor_dict(tensor_dict_path: str) -> dict:
    """Single place in the entire codebase that touches the pkl file."""
    with open(tensor_dict_path, 'rb') as f:
        full_dict = pickle.load(f)
    tensor_dict = full_dict['data'] if 'data' in full_dict else full_dict
    print(f"[load_tensor_dict] Loaded {len(tensor_dict)} subjects from {tensor_dict_path}")
    return tensor_dict
 
 
def train_one_model(
    model_type: str,
    tensor_dict: dict,          # pre-loaded — no path, no re-loading
    config: dict,
    save_dir: str = "checkpoints",
):
    """
    Train a single pretrain model. Caller builds config and loads tensor_dict.

    Saves two checkpoints to save_dir:
        {model_type}_{timestamp}_best.pt  — best validation loss (saved by trainer)
        {model_type}_{timestamp}_last.pt  — weights at the final epoch

    Also builds and returns the DataLoaders so they can be forwarded to
    run_full_eval() without being reconstructed a second time.

    Returns:
        model       : trained model (final-epoch state)
        history     : training history dict
        ckpt_paths  : dict with keys 'best' and 'last'
        train_dl    : training DataLoader (reuse in eval to avoid rebuilding)
        val_dl      : validation DataLoader (reuse in eval to avoid rebuilding)
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = get_run_timestamp()
    ckpt_paths = make_checkpoint_paths(save_dir, model_type, timestamp)

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    print(f"[train_one_model] {model_type} | n_classes={n_classes}")
 
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_one_model] {model_type} | {n_params:,} trainable parameters")
    print(f"[train_one_model] Checkpoints → best: {ckpt_paths['best']}")
    print(f"[train_one_model]               last: {ckpt_paths['last']}")

    # pretrain() saves the best checkpoint internally via save_path.
    # After it returns, we separately save the last (final-epoch) model.
    model, history = pretrain(model, train_dl, val_dl, config, save_path=ckpt_paths["best"])

    # ── Save last-epoch checkpoint ────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config":           config,
            "epoch":            config.get("num_epochs", "unknown"),
            "checkpoint_type":  "last",
        },
        ckpt_paths["last"],
    )
    print(f"[train_one_model] Last checkpoint saved → {ckpt_paths['last']}")

    hist_path = os.path.join(save_dir, f"{model_type}_{timestamp}_history.json")
    json.dump(
        {k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
         for k, vals in history.items() if k not in ('clf', 'routing_reports')},
        open(hist_path, 'w'), indent=2
    )
    print(f"[train_one_model] History saved → {hist_path}")

    # ── Save routing_reports separately (not JSON-serialisable inline) ────────
    if history.get('routing_reports'):
        import pickle as _pkl
        rr_path = os.path.join(save_dir, f"{model_type}_{timestamp}_routing_reports.pkl")
        with open(rr_path, 'wb') as f:
            _pkl.dump(history['routing_reports'], f)
        print(f"[train_one_model] Routing reports saved → {rr_path}")

    return model, history, ckpt_paths, train_dl, val_dl
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dict", type=str,
                        default="C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\segfilt_rts_tensor_dict.pkl")
    parser.add_argument("--model",       type=str, default="DeepCNNLSTM",
                        choices=["MetaCNNLSTM", "DeepCNNLSTM", "TST"])
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--config",      type=str, default=None)
    parser.add_argument("--eval_method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--save_dir",    type=str, default="pretrain_outputs")
    # ── Standalone viz from a saved checkpoint (skips training + probing) ─────
    parser.add_argument(
        "--viz_checkpoint", type=str, default=None,
        help="Path to a .pt checkpoint. If provided, skip training and just produce "
             "the latent-space visualisation for the specified split.",
    )
    parser.add_argument(
        "--viz_split", type=str, default="both", choices=["train", "val", "both"],
        help="Which users to visualise when --viz_checkpoint is set (default: both).",
    )
    args = parser.parse_args()
 
    config_overrides = {}
    if args.config:
        with open(args.config) as f:
            config_overrides = json.load(f)
 
    # ── Load data ONCE ────────────────────────────────────────────────────────
    tensor_dict = load_tensor_dict(args.tensor_dict)
 
    # ── Standalone checkpoint viz (no training, no probing) ──────────────────
    if args.viz_checkpoint:
        mtype  = args.model  # must specify which model arch the checkpoint is for
        config = {
            **PRETRAIN_CONFIG,
            **MODEL_CONFIGS[mtype],
            "model_type": mtype,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            **config_overrides,
        }
        viz_from_checkpoint(
            checkpoint_path = args.viz_checkpoint,
            tensor_dict     = tensor_dict,
            config          = config,
            save_dir        = os.path.join(args.save_dir, "eval", mtype),
            method          = args.eval_method,
            split           = args.viz_split,
        )
        print("\n[run_pretrain] Checkpoint viz complete. Exiting.")
        import sys; sys.exit(0)
 
    models_to_run = ["MetaCNNLSTM", "DeepCNNLSTM", "TST"] if args.all else [args.model]
 
    all_results = {}
    for mtype in models_to_run:
        print(f"\n{'#'*70}\n# Training: {mtype}\n{'#'*70}\n")

        # ── Build config ──────────────────────────────────────────────────────
        # Wire moe_plot_dir into config so the trainer knows where to write
        # in-training routing plots.  The eval-time plots go to a sibling dir
        # (handled inside run_full_eval → run_moe_routing_eval).
        ckpt_dir  = os.path.join(args.save_dir, "checkpoints")
        eval_dir  = os.path.join(args.save_dir, "eval", mtype)
        moe_plot_dir = os.path.join(eval_dir, "MOE_routing_training") \
            if config_overrides.get('use_MOE', PRETRAIN_CONFIG.get('use_MOE', False)) \
               or MODEL_CONFIGS.get(mtype, {}).get('use_MOE', False) \
            else None

        config = {
            **PRETRAIN_CONFIG,
            **MODEL_CONFIGS[mtype],
            "model_type":    mtype,
            "device":        "cuda" if torch.cuda.is_available() else "cpu",
            "MOE_plot_dir":  moe_plot_dir,   # None when use_moe=False → trainer skips plots
            **config_overrides,
        }

        # ── Train ─────────────────────────────────────────────────────────────
        # train_one_model now returns (model, history, ckpt_paths, train_dl, val_dl).
        # We forward the DataLoaders directly into run_full_eval so they are not
        # rebuilt a second time — particularly important for large datasets and
        # MoE runs where routing eval needs a full loader pass.
        model, history, ckpt_paths, train_dl, val_dl = train_one_model(
            mtype, tensor_dict, config,
            save_dir = ckpt_dir,
        )

        # ── Eval ──────────────────────────────────────────────────────────────
        # Pass the DataLoaders in so run_full_eval can use them for MoE routing
        # analysis without needing to reconstruct them from tensor_dict.
        results = run_full_eval(
            model, tensor_dict, config,
            save_dir = eval_dir,
            method   = args.eval_method,
            train_dl = train_dl,
            val_dl   = val_dl,
        )
        all_results[mtype] = results
 
    print(f"\n{'='*60}\nFINAL COMPARISON\n{'='*60}")
    print(f"{'Model':<20} {'In-Dist Probe':>15} {'OOD Probe':>12}")
    print(f"{'-'*50}")
    for mtype, res in all_results.items():
        ind = res['probe_in_dist']['accuracy'] * 100
        ood = res['probe_out_dist']['accuracy'] * 100
        print(f"{mtype:<20} {ind:>14.1f}% {ood:>11.1f}%")
    print(f"{'='*60}")