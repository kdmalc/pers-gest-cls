"""
A13_modality_ablation.py
========================
Ablation A13: modality ablation for M0 (MAML + MoE).

Requested by R1, R2, R4 and the meta-reviewer. Contribution 1 claims
multimodal fusion matters, and the paper contains no modality ablation.

Conditions (one SLURM job each, --condition selects):
    both      : control. Should reproduce the fixed-split M0 number (88.4%).
                Run it -- it is the only thing that proves the masking harness
                did not change the pipeline.
    emg_only  : IMU channels zero-masked
    imu_only  : EMG channels zero-masked

MASK, DO NOT REMOVE
-------------------
Channels are zero-filled rather than deleted, so input width and therefore
parameter count are unchanged. The paper is scrupulous about parameter
matching, and an unmatched ablation invites the obvious question. The cost,
which must be disclosed: masked channels are dead capacity, so this is not
identical to a purpose-built unimodal model.

SPLIT
-----
Fixed 24/4/4 HPO split, because L2SO is 16 runs per condition. Consequences,
all of which need stating in the response:
  - these cells CANNOT enter the paired RM-ANOVA (not evaluable per-participant
    across all 32 participants)
  - they must be compared against the FIXED-SPLIT baseline (88.4%), NOT the
    L2SO headline (86.7%)
  - label them preliminary single-split and commit L2SO to camera-ready

HYPERPARAMETERS
---------------
Reused from the fused model, unchanged. State the bias direction, because it
runs in our favour: hyperparameters were tuned for fused input, so the
unimodal conditions are handicapped. Do not lean on small margins.

IF IMU-ONLY MATCHES OR BEATS FUSED
----------------------------------
Plausible, and the result is then robust: the handicap points the other way,
so it cannot be explained away by tuning. Report it straight. The consequence
is that the fusion claim at L34-38 narrows from blanket complementarity to
specific failure modes (static and low-movement gestures). Cheaper reported by
us than found by a reviewer.

Usage (NOTS):
    python A13_modality_ablation.py --condition emg_only
    python A13_modality_ablation.py --condition imu_only
    python A13_modality_ablation.py --condition both
"""

import os
import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config, build_maml_moe_model,
    set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    make_periodic_checkpoint_fn, make_periodic_test_eval_fn,
)
from MAML.maml_data_pipeline import get_maml_dataloaders

CONDITIONS = ("both", "emg_only", "imu_only")

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def build_config(condition: str) -> dict:
    assert condition in CONDITIONS, f"--condition must be one of {CONDITIONS}"

    config = make_base_config(ablation_id=f"A13_{condition}")

    # REQUIRED: get_maml_dataloaders reads config["seed"] directly and
    # make_base_config does not set it (M0_full_model.py sets it explicitly at
    # line 89). Omitting this raises KeyError: 'seed' at the dataloader build,
    # i.e. AFTER model construction and any pre-flight checks have printed --
    # which is why the mask verification passed and the job still died.
    config["seed"] = FIXED_SEED

    # The ablation-defining flag. Everything else inherits M0's HPO values.
    config["modality_mask"] = condition

    # Fixed split only. L2SO is 16 runs per condition; see docstring.
    config["test_procedure"] = "hpo_test_split"

    # use_imu stays True in ALL conditions. This is deliberate: it keeps the
    # 88-channel input and the parameter count identical across conditions.
    # Setting use_imu=False would drop 72 channels and change per-expert layer 1
    # from C=88 to C=16, i.e. ~23k of ~233k params per expert, ~507k over 22
    # experts -- exactly the unmatched comparison we are avoiding.
    assert config["use_imu"] is True, (
        "use_imu must remain True for masked modality ablations, otherwise the "
        "parameter count changes and the comparison is no longer matched."
    )

    print(f"[A13] condition        : {condition}")
    print(f"[A13] modality_mask    : {config['modality_mask']}")
    print(f"[A13] use_imu          : {config['use_imu']} (channels masked, not removed)")
    print(f"[A13] emg_in_ch        : {config['emg_in_ch']}")
    print(f"[A13] imu_in_ch        : {config['imu_in_ch']}")
    print(f"[A13] test_procedure   : {config['test_procedure']}")
    print(f"[A13] n_way / k_shot   : {config['n_way']} / {config['k_shot']}")
    return config


def verify_masking(config: dict, tensor_dict_path: str) -> dict:
    """
    Pull one episode and confirm the mask did what we think it did BEFORE
    spending a training job on it. A silently-inert mask would produce a
    fused-performance number labelled as unimodal, which is the worst
    possible failure here.
    """
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    import pickle
    from torch.utils.data import DataLoader

    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = config["test_PIDs"],
        target_gesture_classes  = config["maml_gesture_classes"],
        target_trial_reps       = config["target_trial_reps"],
        n_way                   = config["n_way"],
        k_shot                  = config["k_shot"],
        q_query                 = config["q_query"],
        num_eval_episodes       = 2,
        is_train                = False,
        seed                    = FIXED_SEED,
        use_label_shuf_meta_aug = False,
        modality_mask           = config["modality_mask"],
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=maml_mm_collate)

    batch = next(iter(dl))
    emg = batch["support"]["emg"]
    imu = batch["support"].get("imu")

    emg_energy = float(emg.abs().sum())
    imu_energy = float(imu.abs().sum()) if imu is not None else 0.0
    cond = config["modality_mask"]

    print(f"\n[A13] mask verification ({cond}):")
    print(f"        support EMG shape {tuple(emg.shape)}  sum|x| = {emg_energy:.4f}")
    if imu is not None:
        print(f"        support IMU shape {tuple(imu.shape)}  sum|x| = {imu_energy:.4f}")

    if cond == "emg_only":
        assert imu_energy == 0.0, f"emg_only but IMU energy is {imu_energy}, expected 0"
        assert emg_energy > 0.0,  "emg_only but EMG energy is 0"
    elif cond == "imu_only":
        assert emg_energy == 0.0, f"imu_only but EMG energy is {emg_energy}, expected 0"
        assert imu_energy > 0.0,  "imu_only but IMU energy is 0"
    else:
        assert emg_energy > 0.0 and imu_energy > 0.0, "control condition has a zeroed modality"

    print("        PASS")

    # Also surface the realised episode shape, which is what lets us report Q
    # correctly instead of assuming q_query. See maml_data_pipeline changes.
    if ds.episode_shape_log:
        rec = ds.episode_shape_log[0]
        print(f"        realised episode: {rec['n_classes_realised']} classes, "
              f"n_support={rec['n_support']}, n_query={rec['n_query']}, "
              f"q_per_class={rec['q_per_class']:.1f}")
    return {"emg_energy": emg_energy, "imu_energy": imu_energy}


def run(condition: str) -> dict:
    config = build_config(condition)
    set_seeds(FIXED_SEED)

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

    mask_check = verify_masking(config, tensor_dict_path)

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"\n[A13 | {condition}] Parameters: {n_params:,}")
    print("       (must be IDENTICAL across all three conditions -- that is the "
          "point of masking rather than removing.)")

    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    from MAML.mamlpp import mamlpp_pretrain
    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
        periodic_checkpoint_fn=make_periodic_checkpoint_fn(config),
        periodic_test_eval_fn=make_periodic_test_eval_fn(
            tensor_dict_path, config["test_PIDs"]),
        checkpoint_every=10,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[A13 | {condition}] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "condition":        condition,
            "seed":             FIXED_SEED,
            "model_state_dict": train_history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
        },
        config,
        tag=f"A13_{condition}_seed{FIXED_SEED}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])
    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )

    result = {
        "ablation_id":     f"A13_{condition}",
        "description":     f"Modality ablation ({condition}), masked channels, fixed split",
        "condition":       condition,
        "modality_mask":   condition,
        "test_procedure":  "hpo_test_split",
        "seed":            FIXED_SEED,
        "n_params":        n_params,
        "mask_check":      mask_check,
        "best_val_acc":    float(best_val_acc),
        "test_results":    test_results,
        "test_acc":        test_results["mean_acc"],
        "caveats": [
            "Channels are zero-masked, not removed; masked channels are dead "
            "capacity, so this is not a purpose-built unimodal model.",
            "Fixed 24/4/4 split: cannot enter the paired RM-ANOVA and must be "
            "compared against the fixed-split baseline (88.4%), not L2SO (86.7%).",
            "Hyperparameters were tuned for the fused model, so unimodal "
            "conditions are handicapped. Bias runs against the ablations.",
        ],
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"A13_{condition}_final")

    print(f"\n{'='*70}")
    print(f"[A13] FINAL {condition}: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")
    print(f"      {config['n_way']}-way {config['k_shot']}-shot, fixed split, "
          f"seed={FIXED_SEED}")
    print(f"      params={n_params:,}   compare against fixed-split M0, NOT 86.7%")
    print(f"{'='*70}")
    return result


def main():
    ap = argparse.ArgumentParser(description="A13: modality ablation for M0.")
    ap.add_argument("--condition", choices=list(CONDITIONS), required=True)
    args = ap.parse_args()
    run(args.condition)


if __name__ == "__main__":
    main()
