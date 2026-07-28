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
Fixed 24/4/4 HPO split, because L2SO is 32 runs per condition (one per subject:
test = subjects[i], val = subjects[(i+1) % 32]) -- not 16, as an earlier version
of this docstring said. Consequences, all of which need stating in the response:
  - these cells CANNOT enter the paired RM-ANOVA (not evaluable per-participant
    across all 32 participants)
  - they must be compared against the FIXED-SPLIT baseline from THIS harness
    (the `both` control), NOT the published 88.4% and NOT the L2SO headline
    (86.7%). Three runs of the identical fixed-split config have produced
    88.46 / 87.58 / 90.68, a ~3-point spread, so the `both` control is the only
    legitimate reference for the unimodal cells.
  - label them preliminary single-split and commit L2SO to camera-ready

VOCABULARY SIZE
---------------
`--n-way` is a CLI argument, defaulting to the config value (3). The first A13
batch ran 3-way only, which is the weakest available test of a fusion claim:
3-way is near ceiling (both=87.6%, emg_only=88.7%) and the ~3-point same-config
spread above is larger than that difference, so 3-way cannot resolve it. Sweep
3/5/10-way instead -- 10-way is also the deployment vocabulary.

1-way is rejected explicitly (see the argparse validator): with a single class
the label is constant and chance accuracy is 100%, so it measures nothing.

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


def build_config(condition: str, n_way: int = None, k_shot: int = None) -> dict:
    assert condition in CONDITIONS, f"--condition must be one of {CONDITIONS}"

    ablation_id = f"A13_{condition}"
    if n_way is not None:
        ablation_id += f"_n{n_way}"
    if k_shot is not None and int(k_shot) != 1:
        ablation_id += f"_k{k_shot}"
    config = make_base_config(ablation_id=ablation_id)

    # REQUIRED: get_maml_dataloaders reads config["seed"] directly and
    # make_base_config does not set it (M0_full_model.py sets it explicitly at
    # line 89). Omitting this raises KeyError: 'seed' at the dataloader build,
    # i.e. AFTER model construction and any pre-flight checks have printed --
    # which is why the mask verification passed and the job still died.
    config["seed"] = FIXED_SEED

    # The ablation-defining flag. Everything else inherits M0's HPO values.
    config["modality_mask"] = condition

    # ── Task size ─────────────────────────────────────────────────────────────
    # Overridable so the modality question can be asked at more than one
    # vocabulary size. These are the ONLY task-shape keys an A13 run may change;
    # every HPO-tuned hyperparameter stays at its M0 value.
    if n_way is not None:
        config["n_way"] = int(n_way)
    if k_shot is not None:
        config["k_shot"] = int(k_shot)

    # q_query is nominal. The eval path assigns every non-support repetition to
    # the query set, so the realised per-class count is (n_reps - k_shot). Record
    # both so the caption can state the realised number rather than the config
    # value -- this is the disclosure promised in the R2 W1 response.
    n_reps = len(config["target_trial_reps"])
    config["realised_q_per_class"] = n_reps - int(config["k_shot"])

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
    print(f"[A13] q_query (nominal): {config['q_query']}  -> realised per class: "
          f"{config['realised_q_per_class']} ({len(config['target_trial_reps'])} reps "
          f"- {config['k_shot']} support)")
    print(f"[A13] chance level     : {100.0 / config['n_way']:.1f}%")
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


def run(condition: str, n_way: int = None, k_shot: int = None) -> dict:
    config = build_config(condition, n_way=n_way, k_shot=k_shot)
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
        tag=f"{config['ablation_id']}_seed{FIXED_SEED}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])
    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )

    result = {
        "ablation_id":     config["ablation_id"],
        "description":     (f"Modality ablation ({condition}), masked channels, "
                            f"fixed split, {config['n_way']}-way "
                            f"{config['k_shot']}-shot"),
        "condition":       condition,
        "modality_mask":   condition,
        "test_procedure":  "hpo_test_split",
        "n_way":           int(config["n_way"]),
        "k_shot":          int(config["k_shot"]),
        "q_query_nominal": int(config["q_query"]),
        "q_per_class_realised": int(config["realised_q_per_class"]),
        "chance_level":    1.0 / float(config["n_way"]),
        "seed":            FIXED_SEED,
        "n_params":        n_params,
        "mask_check":      mask_check,
        "best_val_acc":    float(best_val_acc),
        "test_results":    test_results,
        "test_acc":        test_results["mean_acc"],
        "caveats": [
            "Channels are zero-masked, not removed; masked channels are dead "
            "capacity, so this is not a purpose-built unimodal model.",
            "Fixed 24/4/4 split: cannot enter the paired RM-ANOVA. Compare "
            "against the `both` control FROM THIS HARNESS at the same n_way, "
            "not against the published 88.4% and not against L2SO (86.7%): "
            "three runs of the identical fixed-split config have spanned "
            "88.46 / 87.58 / 90.68.",
            "Hyperparameters were tuned for the fused model at 3-way, so "
            "unimodal conditions and larger vocabularies are handicapped. Bias "
            "runs against the ablations.",
            f"Realised query count is {config['realised_q_per_class']} per class "
            f"({len(config['target_trial_reps'])} reps - {config['k_shot']} "
            f"support), not the nominal q_query={config['q_query']}.",
        ],
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"{config['ablation_id']}_final")

    print(f"\n{'='*70}")
    print(f"[A13] FINAL {condition}: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")
    print(f"      {config['n_way']}-way {config['k_shot']}-shot, fixed split, "
          f"seed={FIXED_SEED}")
    print(f"      chance={100.0 / config['n_way']:.1f}%   "
          f"q/class realised={config['realised_q_per_class']}")
    print(f"      params={n_params:,}   compare against the `both` control at "
          f"{config['n_way']}-way, NOT 88.4% and NOT 86.7%")
    print(f"{'='*70}")
    return result


def _n_way_arg(value: str) -> int:
    """n_way validator.

    1-way is rejected rather than silently allowed: with a single class every
    label is identical, chance accuracy is 100%, and the resulting number is not
    a discrimination measurement at all. 2-way is the minimum meaningful task.
    """
    n = int(value)
    if n < 2:
        raise argparse.ArgumentTypeError(
            f"--n-way must be >= 2, got {n}. A 1-way task has a constant label "
            "and 100% chance accuracy, so it cannot measure discrimination. "
            "Use 2 for the easiest meaningful setting, 3 for the paper headline, "
            "or 10 for the deployment vocabulary."
        )
    return n


def main():
    ap = argparse.ArgumentParser(description="A13: modality ablation for M0.")
    ap.add_argument("--condition", choices=list(CONDITIONS), required=True)
    ap.add_argument("--n-way", type=_n_way_arg, default=None,
                    help="Vocabulary size. Default: config value (3). "
                         "Sweep 3 5 10 -- 3-way alone is near ceiling and "
                         "cannot resolve the modality difference.")
    ap.add_argument("--k-shot", type=int, default=None,
                    help="Support examples per class. Default: config value (1).")
    args = ap.parse_args()
    if args.k_shot is not None and args.k_shot < 1:
        ap.error(f"--k-shot must be >= 1, got {args.k_shot}")
    run(args.condition, n_way=args.n_way, k_shot=args.k_shot)


if __name__ == "__main__":
    main()
