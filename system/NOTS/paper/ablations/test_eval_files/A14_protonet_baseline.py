"""
A14_protonet_baseline.py
========================
Prototypical Networks baseline on our own backbone, for parameter
comparability.

Asked by R1 (W1) and R4 (W2), and it also discharges a claim the paper makes
without evidence: §2 (L110-112) asserts that metric-learning approaches are
fragile with 1-shot physiological signals, about a method we never ran. Running
it either supports the claim or corrects it. Either outcome is better than the
current state.

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
The ProtoNet implementation already exists in
`system/nonparametric/eval_knn_proto.py` (`proto_raw`, `proto_pca`,
`proto_encoded`, plus kNN tracks). It has no `__main__` and no CLI, so it could
only be driven from a notebook and could not be scheduled as a SLURM job. This
is a thin runner around `run_all_conditions`, not a reimplementation.

WHICH ROW ANSWERS THE REVIEWERS
-------------------------------
`proto_encoded` -- ProtoNet on the features of our meta-trained encoder. That
is the parameter-comparable comparison: identical backbone, identical support
budget, the only difference being a nearest-prototype readout in place of
meta-learned adaptation. The `proto_raw` / `proto_pca` / kNN tracks come along
for free and are informative context, but they are not backbone-matched and
should not be presented as the headline.

ENCODER INTERFACE
-----------------
`eval_knn_proto.neural_encode_batch` calls `encoder.eval()` and then
`encoder(emg, imu)`, accepting either `features` or `(features, aux)`. Our M0
model exposes `backbone(x_emg, x_imu, demographics=None, return_routing=False)`
returning `(l3, [l1, l2, l3])` -- but `model.backbone` is a bound method, so it
has no `.eval()`. The docstring's `encoder=model.backbone` therefore does not
work as written. `_BackboneEncoder` below is a small nn.Module adapter that
does.

Usage (NOTS):
    python A14_protonet_baseline.py --checkpoint /path/to/best_M0_model.pt
    python A14_protonet_baseline.py --checkpoint ... --n-way 3 --shots 1 3 5
    python A14_protonet_baseline.py --no-encoder          # raw/PCA tracks only
"""

import os
import sys
import copy
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))
sys.path.insert(0, str(CODE_DIR / "system" / "nonparametric"))

from ablation_config import (
    make_base_config, build_maml_moe_model, set_seeds, FIXED_SEED,
    save_results, count_parameters,
)
from MAML.maml_data_pipeline import reorient_tensor_dict

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class _BackboneEncoder(nn.Module):
    """
    Adapter exposing our M0 backbone as an nn.Module with the signature
    `eval_knn_proto.neural_encode_batch` expects.

    Returns the pooled final-layer feature (l3), which is the representation the
    classification head consumes -- so ProtoNet is scored on exactly the
    features the meta-learned readout would have seen.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        assert hasattr(model, "backbone"), (
            f"{type(model).__name__} has no .backbone(); cannot extract features."
        )

    @torch.no_grad()
    def forward(self, x_emg, x_imu=None, demographics=None):
        out = self.model.backbone(x_emg, x_imu, demographics, return_routing=False)
        # backbone returns (l3, [l1, l2, l3]); neural_encode_batch accepts a
        # (features, aux) tuple and takes the first element.
        return out


def load_encoder(checkpoint: str, config: dict):
    model = build_maml_moe_model(config)
    ckpt = torch.load(checkpoint, map_location=config["device"])
    state = ckpt.get("model_state_dict", ckpt.get("best_state", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[A14] WARNING load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected")
        print(f"      missing[:5]={list(missing)[:5]}")
        print(f"      unexpected[:5]={list(unexpected)[:5]}")
    model.to(config["device"]).eval()
    print(f"[A14] Loaded encoder checkpoint: {checkpoint}")
    print(f"[A14] Backbone parameters: {count_parameters(model):,}")
    return _BackboneEncoder(model).to(config["device"]).eval()


def main():
    ap = argparse.ArgumentParser(
        description="A14: Prototypical Networks baseline on the matched backbone.")
    ap.add_argument("--checkpoint", default=None,
                    help="M0 checkpoint for the proto_encoded track. Omit with "
                         "--no-encoder to run raw/PCA tracks only.")
    ap.add_argument("--no-encoder", action="store_true")
    ap.add_argument("--n-way", type=int, default=3,
                    help="3 matches Table 1/4's headline protocol; 10 matches the "
                         "deployment vocabulary.")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--knn-metric", default="l1")
    ap.add_argument("--pca-level", default="per_sample",
                    choices=["global", "per_class", "per_sample"])
    args = ap.parse_args()

    if not args.no_encoder and args.checkpoint is None:
        ap.error("--checkpoint is required unless --no-encoder is passed.")

    base = make_base_config(ablation_id=f"A14_protonet_n{args.n_way}")
    set_seeds(FIXED_SEED)

    from eval_knn_proto import BASE_CONFIG, run_all_conditions

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["n_way"]      = args.n_way
    cfg["shots"]      = args.shots
    cfg["seed"]       = FIXED_SEED
    cfg["device"]     = base["device"]
    cfg["use_imu"]    = base["use_imu"]
    cfg["knn_metric"] = args.knn_metric
    cfg["pca_level"]  = args.pca_level
    # Evaluate on the SAME held-out users as the ablation suite, so the number is
    # comparable to Table 1/4 rather than to eval_knn_proto's own default list.
    cfg["eval_PIDs"]  = list(base["test_PIDs"])
    cfg["available_gesture_classes"] = list(base["maml_gesture_classes"])
    cfg["all_rep_indices"] = list(base["target_trial_reps"])

    print(f"[A14] n_way={cfg['n_way']}  shots={cfg['shots']}")
    print(f"[A14] eval_PIDs={cfg['eval_PIDs']}  (matched to ablation test split)")
    print(f"[A14] knn_metric={cfg['knn_metric']}  pca_level={cfg['pca_level']}")

    tensor_dict_path = os.path.join(base["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, base)

    encoder = None if args.no_encoder else load_encoder(args.checkpoint, base)

    results = run_all_conditions(
        tensor_dict, cfg,
        shot_conditions=cfg["shots"],
        encoder=encoder,
        verbose=True,
    )

    # Flatten for storage; run_all_conditions keys by k_shot.
    flat = {}
    for k, per_method in results.items():
        for method, r in per_method.items():
            if isinstance(r, dict) and "mean_acc" in r:
                flat[f"{method}_k{k}"] = {
                    "mean_acc": float(r["mean_acc"]),
                    "std_acc":  float(r.get("std_acc", float("nan"))),
                }

    headline = flat.get(f"proto_encoded_k1")
    out = {
        "ablation_id":   f"A14_protonet_n{args.n_way}",
        "description":   ("Prototypical Networks on the meta-trained M0 backbone "
                          "(proto_encoded), plus raw/PCA and kNN tracks for context."),
        "checkpoint":    args.checkpoint,
        "n_way":         args.n_way,
        "shots":         args.shots,
        "eval_PIDs":     cfg["eval_PIDs"],
        "knn_metric":    args.knn_metric,
        "pca_level":     args.pca_level,
        "results":       flat,
        "headline_proto_encoded_1shot": headline,
        "caveats": [
            "proto_encoded is the backbone-matched comparison and the row that "
            "answers R1 W1 / R4 W2. proto_raw, proto_pca and the kNN tracks are "
            "not backbone-matched and are context only.",
            "The encoder is meta-trained with MAML++, so proto_encoded reuses a "
            "representation shaped by episodic adaptation. It isolates the "
            "readout (nearest prototype vs meta-learned adaptation), not the "
            "training procedure.",
            "Evaluated on the fixed-split test users, so not directly comparable "
            "to Table 1's L2SO numbers.",
        ],
        "config_snapshot": {k: str(v) for k, v in cfg.items()},
    }
    save_results(out, base, tag=f"A14_protonet_n{args.n_way}")

    print(f"\n{'='*70}")
    if headline:
        print(f"[A14] HEADLINE proto_encoded, {args.n_way}-way 1-shot: "
              f"{headline['mean_acc']*100:.2f}%")
        print(f"      This is the row to quote. Compare against fixed-split M0.")
        print(f"      Then either support or correct the L110-112 claim about "
              f"metric-learning fragility.")
    else:
        print("[A14] No proto_encoded result (ran with --no-encoder?).")
        print("      The raw/PCA tracks alone do NOT answer R1 W1 -- they are not "
              "backbone-matched.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
