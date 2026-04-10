"""
A3_A4_maml_no_moe.py
=====================
Ablations A3 and A4: MAML + No-MoE

A3: Single encoder with its natural (smaller) parameter count.
    Shows MoE contribution without controlling for parameter count.

A4: CRITICAL — Single encoder scaled to match the combined parameter count of
    all K experts in M0. Isolates the routing/specialisation mechanism of MoE
    from its parameter budget. The most important ablation in the paper.

Both use:
  Training : Episodic dataloader
  Evaluation: Episodic (1-shot 3-way)

A4 methodology:
  1. Build M0 and count only encoder parameters (the K-expert CNN block).
  2. Scale cnn_base_filters (and optionally lstm_hidden) so a single CNN-LSTM
     matches that count as closely as possible.
  3. Print exact counts before training so you can verify and note in the paper.

NOTE: "Encoder parameters" in the MoE model means the K CNN expert networks.
The LSTM and head are shared, so we only scale the CNN block width for A4.
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
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import mamlpp_pretrain
from pretraining.pretrain_models import build_model

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Parameter counting utilities
# =============================================================================

def count_moe_encoder_params(moe_model: nn.Module) -> int:
    """
    Count only the parameters in the expert CNN networks of the MoE encoder.
    These are the parameters A4 must match.

    We look for 'experts' submodule which holds the K CNN expert networks
    in the EncoderMOE variant.
    """
    # Try direct attribute access first (EncoderMOE layout)
    if hasattr(moe_model, "experts"):
        return sum(p.numel() for p in moe_model.experts.parameters() if p.requires_grad)
    # Fallback: sum all submodules whose name contains 'expert'
    expert_params = 0
    for name, module in moe_model.named_modules():
        if "expert" in name.lower() and len(list(module.children())) == 0:
            expert_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    if expert_params == 0:
        raise RuntimeError(
            "Could not find 'experts' submodule in MoE model. "
            "Inspect the model architecture and update count_moe_encoder_params()."
        )
    return expert_params


def find_matched_cnn_filters(target_cnn_params: int, config_template: dict,
                              lstm_hidden: int, search_range=(64, 512)) -> int:
    """
    Binary-search for cnn_base_filters such that the resulting CNN block's
    parameter count is as close as possible to target_cnn_params.

    We keep lstm_hidden fixed (changing depth confounds the comparison per spec).
    Returns the best cnn_base_filters value found.
    """
    from pretraining.pretrain_models import build_model as _build

    def _cnn_param_count(filters: int) -> int:
        cfg = copy.deepcopy(config_template)
        cfg["cnn_base_filters"] = filters
        cfg["lstm_hidden"]      = lstm_hidden
        cfg["use_MOE"]          = False
        m = _build(cfg)
        # Count only the CNN part (before LSTM)
        if hasattr(m, "cnn") or hasattr(m, "conv"):
            cnn_mod = getattr(m, "cnn", None) or getattr(m, "conv", None)
            return sum(p.numel() for p in cnn_mod.parameters() if p.requires_grad)
        # Fallback: count all non-LSTM, non-head params
        total = 0
        for name, p in m.named_parameters():
            if p.requires_grad and "lstm" not in name.lower() and "head" not in name.lower():
                total += p.numel()
        return total

    lo, hi = search_range
    best_filters = lo
    best_diff = abs(_cnn_param_count(lo) - target_cnn_params)

    # Coarse sweep
    for f in range(lo, hi + 1, 16):
        diff = abs(_cnn_param_count(f) - target_cnn_params)
        if diff < best_diff:
            best_diff = diff
            best_filters = f

    # Fine sweep around best
    for f in range(max(lo, best_filters - 16), min(hi, best_filters + 16) + 1):
        diff = abs(_cnn_param_count(f) - target_cnn_params)
        if diff < best_diff:
            best_diff = diff
            best_filters = f

    return best_filters


# =============================================================================
# Config builders
# =============================================================================

def build_config_a3() -> dict:
    config = make_base_config(ablation_id="A3")
    # ── Remove MoE, keep MAML ─────────────────────────────────────────────────
    config["use_MOE"] = False
    # cnn_base_filters / lstm_hidden unchanged — natural (smaller) param count
    return config


def build_config_a4(moe_config: dict) -> dict:
    """
    Build A4 config with cnn_base_filters scaled to match M0 expert param count.
    Must be called AFTER the M0 model has been built so we can count its params.
    """
    config = make_base_config(ablation_id="A4")
    config["use_MOE"] = False

    # Build M0 model just to measure expert parameters
    moe_model = build_maml_moe_model(moe_config)
    m0_total_params  = count_parameters(moe_model)
    m0_expert_params = count_moe_encoder_params(moe_model)
    del moe_model  # free GPU memory

    print(f"\n[A4] M0 total params       : {m0_total_params:,}")
    print(f"[A4] M0 expert CNN params  : {m0_expert_params:,}  (A4 encoder must match this)")

    # Find cnn_base_filters that gives ~m0_expert_params in a single CNN block
    matched_filters = find_matched_cnn_filters(
        target_cnn_params=m0_expert_params,
        config_template=config,
        lstm_hidden=config["lstm_hidden"],  # FIXED per spec — don't change depth
    )
    config["cnn_base_filters"] = matched_filters

    # Build A4 model to verify the match
    a4_model = build_maml_no_moe_model(config)
    a4_total_params = count_parameters(a4_model)
    del a4_model

    print(f"[A4] Selected cnn_base_filters = {matched_filters}")
    print(f"[A4] A4 total params           = {a4_total_params:,}")
    print(f"[A4] Parameter ratio (A4 / M0 experts) = "
          f"{a4_total_params / max(m0_expert_params, 1):.3f}  (target ≈ 1.0)")
    print(f"[A4] NOTE: Report both M0 expert count ({m0_expert_params:,}) "
          f"and A4 total count ({a4_total_params:,}) in the paper.\n")

    config["_a4_matched_filters"]      = matched_filters
    config["_m0_expert_params"]        = m0_expert_params
    config["_a4_total_params"]         = a4_total_params

    return config


# =============================================================================
# Common training loop (shared by A3 and A4)
# =============================================================================

def run_one_seed(ablation_id: str, seed: int, config: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_no_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[{ablation_id} | seed={seed}] Parameters: {n_params:,}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[{ablation_id} | seed={seed}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "seed":             seed,
            "model_state_dict": train_history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
            "train_loss_log":   train_history["train_loss_log"],
            "val_acc_log":      train_history["val_acc_log"],
        },
        config,
        tag=f"seed{seed}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[{ablation_id} | seed={seed}] Test acc: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")

    return {
        "seed":         seed,
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


def run_ablation(ablation_id: str, config: dict, description: str):
    print(f"\n{ablation_id} CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[{ablation_id}] Running seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_one_seed(ablation_id, actual_seed, config)
        all_seed_results.append(result)

    test_accs = [r["test_results"]["mean_acc"] for r in all_seed_results]
    summary = {
        "ablation_id":     ablation_id,
        "description":     description,
        "n_params":        all_seed_results[0]["n_params"],
        "seed_results":    all_seed_results,
        "mean_test_acc":   float(np.mean(test_accs)),
        "std_test_acc":    float(np.std(test_accs)),
        "num_seeds":       NUM_FINAL_SEEDS,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[{ablation_id}] FINAL: {summary['mean_test_acc']*100:.2f}% "
          f"± {summary['std_test_acc']*100:.2f}%")
    print(f"     over {NUM_FINAL_SEEDS} seeds, {config['n_way']}-way {config['k_shot']}-shot")
    print(f"{'='*70}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["A3", "A4", "both"], default="both",
                        help="Which ablation to run. 'both' runs A3 then A4.")
    args = parser.parse_args()

    if args.ablation in ("A3", "both"):
        config_a3 = build_config_a3()
        run_ablation("A3", config_a3,
                     "MAML + No-MoE (Single Expert, Reduced Parameters)")

    if args.ablation in ("A4", "both"):
        # A4 needs the M0 config to measure expert parameter target
        moe_config = make_base_config(ablation_id="M0_ref")
        config_a4 = build_config_a4(moe_config)
        run_ablation("A4", config_a4,
                     "MAML + No-MoE (Parameter-Matched Encoder) ← CRITICAL")


if __name__ == "__main__":
    main()
