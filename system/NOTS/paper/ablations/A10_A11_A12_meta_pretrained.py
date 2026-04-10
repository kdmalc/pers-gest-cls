"""
A10_A11_A12_meta_pretrained.py
================================
Ablations A10, A11, A12: Meta Pretrained EMG Foundation Model Comparison

⚠️  DIFFERENT DATA: All three ablations use 2000 Hz EMG-only data (no IMU).
    Results are reported in a SEPARATE figure with explicit callout that the
    data format differs from all other ablations (M0, A1–A9).

A10: Meta's pretrained EMG model, zero-shot (no fine-tuning).
A11: Meta's pretrained EMG model, 1-shot fine-tuning (head-only and full-FT).
A12: Our full MAML + MoE model trained on the same 2kHz EMG-only data.
     This is the apples-to-apples control ensuring method differences, not
     data format differences, drive any gap vs A10/A11.

STUB STATUS:
  A10 and A11 require Meta's pretrained weights and their inference API.
  A12 is fully implementable — it is M0 with use_imu=False and the 2kHz data.

See the TODO blocks below for what you need to fill in.

Usage:
    python A10_A11_A12_meta_pretrained.py --ablation A12   # run A12 (ready to go)
    python A10_A11_A12_meta_pretrained.py --ablation A10   # stub — will raise NotImplementedError
    python A10_A11_A12_meta_pretrained.py --ablation A11   # stub — will raise NotImplementedError
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
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS,
    run_episodic_test_eval, run_supervised_test_eval,
    save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import mamlpp_pretrain

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Data configuration for 2kHz EMG-only ─────────────────────────────────────
# TODO: Set these to match your 2kHz EMG-only dataset paths and dimensions.
# These WILL differ from the standard dataset paths used in M0–A9.
EMG_2KHZ_PKL_PATH  = None  # e.g. CODE_DIR / "dataset" / "2khz_emg_only.pkl"
EMG_2KHZ_IN_CH     = None  # e.g. 16 (confirm with your sensor config at 2kHz)
EMG_2KHZ_SEQ_LEN   = None  # e.g. 128 (2kHz × window_size_s — depends on your windowing)

# ── Meta pretrained model ─────────────────────────────────────────────────────
# TODO: Set to the path of Meta's pretrained checkpoint and their model class.
META_CHECKPOINT_PATH = None  # e.g. "/scratch/.../meta_emg_foundation.pt"

# Fine-tuning settings for A11
A11_FT_LR    = 1e-4   # spec: tune this (log-uniform [1e-5, 1e-2]) — use a reasonable default
A11_FT_STEPS = 50     # spec: tune over {10, 25, 50, 100} — 50 is a safe default


# =============================================================================
# A12: MAML + MoE on 2kHz EMG-only  (FULLY IMPLEMENTED)
# =============================================================================

def build_config_a12() -> dict:
    """
    A12 is M0 with use_imu=False and the 2kHz EMG-only data.
    Per spec: separate full HPO run is recommended, but we use the same best
    hyperparameters from M0 as a starting point. Note that input dimensionality
    has changed (EMG-only), so M0's hyperparameters may not be optimal.
    Flag this caveat in the paper.
    """
    if EMG_2KHZ_PKL_PATH is None:
        raise ValueError(
            "EMG_2KHZ_PKL_PATH is not set. "
            "Edit the top of this file to set the path to your 2kHz EMG-only dataset."
        )
    if EMG_2KHZ_IN_CH is None or EMG_2KHZ_SEQ_LEN is None:
        raise ValueError(
            "EMG_2KHZ_IN_CH and EMG_2KHZ_SEQ_LEN must be set. "
            "Edit the top of this file with your 2kHz dataset dimensions."
        )

    config = make_base_config(ablation_id="A12")

    # Switch to 2kHz EMG-only data
    config["use_imu"]          = False
    config["emg_in_ch"]        = EMG_2KHZ_IN_CH
    config["sequence_length"]  = EMG_2KHZ_SEQ_LEN
    config["dfs_load_path"]    = str(Path(EMG_2KHZ_PKL_PATH).parent) + "/"

    # Per spec: do NOT transfer M0's hyperparameters blindly — input dim changed.
    # These are sensible defaults; ideally run a separate HPO sweep for A12.
    # The spec says "separate full HPO run" — flag this in the paper if skipped.

    return config


def run_a12_one_seed(seed: int, config: dict) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[A12 | seed={seed}] Parameters: {n_params:,}")
    print(f"[A12 | seed={seed}] Input: EMG-only {config['emg_in_ch']}ch @ 2kHz, "
          f"seq_len={config['sequence_length']}")

    pkl_name = Path(EMG_2KHZ_PKL_PATH).name
    tensor_dict_path = str(Path(EMG_2KHZ_PKL_PATH))
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[A12 | seed={seed}] Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "seed":             seed,
            "model_state_dict": train_history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
        },
        config,
        tag=f"seed{seed}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])
    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[A12 | seed={seed}] Test: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")

    return {
        "seed":         seed,
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


def run_a12():
    config = build_config_a12()
    print("\nA12 CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[A12] Seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_a12_one_seed(actual_seed, config)
        all_seed_results.append(result)

    test_accs = [r["test_results"]["mean_acc"] for r in all_seed_results]
    summary = {
        "ablation_id":    "A12",
        "description":    "MAML + MoE on 2kHz EMG-only (apples-to-apples control for A10/A11)",
        "data_note":      "2kHz EMG-only — different data format from M0-A9",
        "n_params":       all_seed_results[0]["n_params"],
        "seed_results":   all_seed_results,
        "mean_test_acc":  float(np.mean(test_accs)),
        "std_test_acc":   float(np.std(test_accs)),
        "num_seeds":      NUM_FINAL_SEEDS,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[A12] FINAL: {summary['mean_test_acc']*100:.2f}% ± {summary['std_test_acc']*100:.2f}%")
    print(f"      ⚠  2kHz EMG-only data — compare only against A10/A11, not M0")
    print(f"{'='*70}")


# =============================================================================
# A10: Meta pretrained EMG model, zero-shot  (STUB)
# =============================================================================

def run_a10():
    raise NotImplementedError(
        "A10 is a stub. To implement:\n"
        "  1. Set META_CHECKPOINT_PATH at the top of this file.\n"
        "  2. Import Meta's model class and load the checkpoint.\n"
        "  3. Wrap their model's forward pass in a function that accepts\n"
        "     (x_emg, x_imu=None, demographics=None) and returns logits,\n"
        "     so it's compatible with run_episodic_test_eval.\n"
        "  4. Call run_episodic_test_eval with use_imu=False and the 2kHz data.\n"
        "     Note: A10 does NOT fine-tune — the model is used directly.\n"
        "\n"
        "IMPORTANT: The episodic eval wrapper (mamlpp_adapt_and_eval) runs\n"
        "inner-loop gradient steps at eval time. For A10 (zero-shot), you need\n"
        "a version that skips the inner loop and evaluates directly on query.\n"
        "This requires a separate eval function — write one or use\n"
        "run_supervised_test_eval with ft_steps=0.\n"
        "\n"
        "Meta's EMG foundation model API / checkpoint location:\n"
        "  → TODO: Confirm with your PI where the weights live and what\n"
        "    their inference interface looks like.\n"
    )


# =============================================================================
# A11: Meta pretrained EMG model, 1-shot fine-tune  (STUB)
# =============================================================================

def run_a11():
    raise NotImplementedError(
        "A11 is a stub. To implement:\n"
        "  1. Complete A10 first (you need the same model loading code).\n"
        "  2. Fine-tune the loaded Meta model on the 1-shot support set\n"
        "     using run_supervised_test_eval with ft_mode='head_only' and\n"
        "     ft_mode='full' (report both, per spec).\n"
        "  3. Set ft_steps to one of {10, 25, 50, 100} (spec says tune this;\n"
        "     use A11_FT_STEPS at the top of this file as default).\n"
        "  4. Set ft_lr from the spec's range [1e-5, 1e-2] — use A11_FT_LR.\n"
        "\n"
        "NOTE: The spec says tune ft_lr (log-uniform [1e-5, 1e-2]) and\n"
        "ft_steps ({10,25,50,100}) with N=50 Optuna trials. The current\n"
        "hardcoded defaults (A11_FT_LR=1e-4, A11_FT_STEPS=50) are a starting\n"
        "point but are NOT HPO-tuned. If the paper requires it, add a small\n"
        "Optuna sweep here.\n"
    )


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=["A10", "A11", "A12"],
        required=True,
        help="Which ablation to run.",
    )
    args = parser.parse_args()

    if args.ablation == "A10":
        run_a10()
    elif args.ablation == "A11":
        run_a11()
    elif args.ablation == "A12":
        run_a12()


if __name__ == "__main__":
    main()
