"""
A15_metatrain_user_sweep.py
===========================
Meta-training user sweep: accuracy as a function of how many users are in the
meta-training set.

Asked by R2 (Q3) and the meta-reviewer (Q10). The response currently brackets
this with the two endpoints already in the paper -- 1 meta-training user
(Subject-Specific EncoderMoE, 64.6%) and 24 (86.7%) -- and offers "one
intermediate point if a slot is free". This script produces the intermediate
points.

It also bears directly on meta-review Q10's first question: is the
cross-subject advantage just a sample-size effect? A sweep that is still
climbing at N=24 says more data would help; one that has plateaued by N=16 says
the benefit is structural rather than volume-driven. Either answer is more
useful than the current text.

DESIGN
------
Subsample the meta-TRAIN user list to N users. Validation and test users are
untouched, so every point in the sweep is evaluated on the identical test
subjects and the only thing varying is meta-training set size.

Subsampling is seeded and deterministic per (N, seed). Small N is
high-variance -- WHICH users you draw matters as much as how many -- so
`--subsample-seeds` runs several draws per N and reports mean and spread. With
one draw per N, an N=8 point can land anywhere in a wide band and a
non-monotonic sweep will look like a finding when it is sampling noise. Use at
least 3 seeds for any N below 16 that you intend to report.

Endpoints are deliberately re-runnable: N=24 should reproduce the fixed-split
M0 number, which is the control confirming the subsampling harness is inert.

Usage (NOTS):
    python A15_metatrain_user_sweep.py --n-users 8
    python A15_metatrain_user_sweep.py --n-users 16 --subsample-seeds 0 1 2
    python A15_metatrain_user_sweep.py --n-users 24            # control
"""

import os
import sys
import copy
import random
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
    make_base_config, build_maml_moe_model, set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    make_periodic_checkpoint_fn, make_periodic_test_eval_fn,
)
from MAML.maml_data_pipeline import get_maml_dataloaders

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def subsample_train_users(all_train_pids: list, n_users: int, subsample_seed: int) -> list:
    """Deterministic seeded subsample. Sorted first so the draw is reproducible
    regardless of the incoming list order."""
    pool = sorted(all_train_pids)
    assert 1 <= n_users <= len(pool), (
        f"--n-users {n_users} out of range: {len(pool)} meta-training users available."
    )
    rng = random.Random(10_000 + 97 * subsample_seed + n_users)
    return sorted(rng.sample(pool, n_users))


def run_one(n_users: int, subsample_seed: int) -> dict:
    config = make_base_config(ablation_id=f"A15_N{n_users}_s{subsample_seed}")

    # REQUIRED: get_maml_dataloaders reads config["seed"] directly and
    # make_base_config does not set it (M0_full_model.py sets it explicitly at
    # line 89). Omitting this raises KeyError: 'seed' at the dataloader build,
    # i.e. AFTER model construction and any pre-flight checks have printed --
    # which is why the mask verification passed and the job still died.
    config["seed"] = FIXED_SEED
    config["test_procedure"] = "hpo_test_split"
    set_seeds(FIXED_SEED)

    full_train = list(config["train_PIDs"])
    chosen = subsample_train_users(full_train, n_users, subsample_seed)
    config["train_PIDs"] = chosen

    print(f"\n{'#'*70}")
    print(f"# A15  N={n_users} meta-training users  subsample_seed={subsample_seed}")
    print(f"{'#'*70}")
    print(f"[A15] meta-train ({len(chosen)}/{len(full_train)}): {chosen}")
    print(f"[A15] val   (unchanged): {config['val_PIDs']}")
    print(f"[A15] test  (unchanged): {config['test_PIDs']}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"[A15] Parameters: {n_params:,} (constant across the sweep)")

    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    from MAML.mamlpp import mamlpp_pretrain
    trained_model, hist = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
        periodic_checkpoint_fn=make_periodic_checkpoint_fn(config),
        periodic_test_eval_fn=make_periodic_test_eval_fn(
            tensor_dict_path, config["test_PIDs"]),
        checkpoint_every=10,
    )
    save_model_checkpoint(
        {"n_users": n_users, "subsample_seed": subsample_seed,
         "train_PIDs": chosen, "seed": FIXED_SEED,
         "model_state_dict": hist["best_state"], "config": config,
         "best_val_acc": hist["best_val_acc"]},
        config, tag=f"A15_N{n_users}_s{subsample_seed}_best",
    )
    trained_model.load_state_dict(hist["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )

    result = {
        "ablation_id":     f"A15_N{n_users}_s{subsample_seed}",
        "description":     "Meta-training user sweep: N meta-train users, fixed val/test.",
        "n_users":         n_users,
        "subsample_seed":  subsample_seed,
        "train_PIDs":      chosen,
        "val_PIDs":        config["val_PIDs"],
        "test_PIDs":       config["test_PIDs"],
        "n_params":        n_params,
        "best_val_acc":    float(hist["best_val_acc"]),
        "test_results":    test_results,
        "test_acc":        test_results["mean_acc"],
        "caveats": [
            "Only the meta-training set size varies; val and test users are held "
            "fixed across the sweep.",
            "Small N is high-variance in WHICH users are drawn as well as how "
            "many. Do not report a single draw below N=16.",
            "Fixed 24/4/4 split: outside the paired RM-ANOVA. N=24 should "
            "reproduce the fixed-split M0 number and is the harness control.",
        ],
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"A15_N{n_users}_s{subsample_seed}_final")

    print(f"[A15] N={n_users} s={subsample_seed}: "
          f"{test_results['mean_acc']*100:.2f}% ± {test_results['std_acc']*100:.2f}%")
    return result


def main():
    ap = argparse.ArgumentParser(description="A15: meta-training user sweep.")
    ap.add_argument("--n-users", type=int, required=True)
    ap.add_argument("--subsample-seeds", type=int, nargs="+", default=[0],
                    help="Draws per N. Use >=3 for N<16 before reporting.")
    args = ap.parse_args()

    results = [run_one(args.n_users, s) for s in args.subsample_seeds]
    accs = [r["test_acc"] for r in results]

    print(f"\n{'='*70}")
    print(f"[A15] N={args.n_users} over {len(accs)} subsample seed(s): "
          f"{np.mean(accs)*100:.2f}%"
          + (f" ± {np.std(accs)*100:.2f}% (across draws)" if len(accs) > 1 else ""))
    if len(accs) == 1 and args.n_users < 16:
        print("      NOTE: single draw at N<16. Treat as indicative only; add "
              "--subsample-seeds 0 1 2 before reporting this point.")
    print(f"      Endpoints already in the paper: N=1 -> 64.6%, N=24 -> 86.7% (L2SO)")
    print(f"      This sweep is fixed-split, so compare its N=24 point to 88.4%.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
