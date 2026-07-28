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
    """Deterministic seeded subsample, returned in the ORIGINAL config order.

    ORDER MATTERS -- THIS IS WHY THE N=24 CONTROL FAILED
    ----------------------------------------------------
    The previous version drew from `sorted(all_train_pids)` and returned
    `sorted(...)`, so every A15 run trained on an alphabetically sorted PID list
    while M0, the few-shot grid and A13 all used the split-file order
    (P011, P010, P008, P006, ...). PID order changes the episode-sampling
    sequence, which changes how the RNG stream is consumed, which changes the
    training trajectory. Same seed, different result. That is why the N=24
    control -- which subsamples 24 of 24 and should therefore be a no-op --
    returned 90.68% against 88.46% for the published grid run.

    Fix: sort only for the DRAW (so the draw itself is order-independent and
    reproducible), then map the selection back onto the incoming order. At
    N = len(pool) the returned list is now identical to `all_train_pids`, so the
    control is a genuine no-op.
    """
    pool_sorted = sorted(all_train_pids)
    assert 1 <= n_users <= len(pool_sorted), (
        f"--n-users {n_users} out of range: {len(pool_sorted)} meta-training users available."
    )
    rng = random.Random(10_000 + 97 * subsample_seed + n_users)
    chosen = set(rng.sample(pool_sorted, n_users))
    # Preserve config order so N=len(pool) reproduces the M0 PID sequence exactly.
    return [pid for pid in all_train_pids if pid in chosen]


def draw_is_degenerate(all_train_pids: list, n_users: int) -> bool:
    """True when subsampling cannot vary the user set (N == pool size).

    At N=24 of 24 every `subsample_seed` selects the same 24 users, so the runs
    are bit-identical -- A15_N24 s0/s1/s2 all returned 0.9067592592592594. They
    are one run reported three times, not three replicates, and must not carry a
    +/- across seeds.
    """
    return n_users >= len(all_train_pids)


def run_one(n_users: int, subsample_seed: int,
            vary_training_seed: bool = False) -> dict:
    config = make_base_config(ablation_id=f"A15_N{n_users}_s{subsample_seed}")

    # REQUIRED: get_maml_dataloaders reads config["seed"] directly and
    # make_base_config does not set it (M0_full_model.py sets it explicitly at
    # line 89). Omitting this raises KeyError: 'seed' at the dataloader build,
    # i.e. AFTER model construction and any pre-flight checks have printed --
    # which is why the mask verification passed and the job still died.
    config["test_procedure"] = "hpo_test_split"

    full_train = list(config["train_PIDs"])
    chosen = subsample_train_users(full_train, n_users, subsample_seed)
    degenerate = draw_is_degenerate(full_train, n_users)

    # Seeding policy, made explicit because it determines what the +/- means.
    #
    #   vary_training_seed=False (default): the training seed is pinned to
    #     FIXED_SEED and only the user DRAW varies with subsample_seed. The
    #     spread across seeds is then "which users you got", which is the
    #     quantity the sweep is about. At N == pool size the draw cannot vary,
    #     so all seeds collapse to one run -- see draw_is_degenerate().
    #
    #   vary_training_seed=True: the training seed also moves, giving genuine
    #     independent replicates. This is the only way to get a spread at
    #     N=24, and it is what quantifies the ~3-point same-config
    #     reproducibility band (88.46 / 87.58 / 90.68 observed across three
    #     nominally identical fixed-split runs).
    training_seed = FIXED_SEED + subsample_seed if vary_training_seed else FIXED_SEED
    config["seed"] = training_seed
    set_seeds(training_seed)

    config["train_PIDs"] = chosen

    print(f"\n{'#'*70}")
    print(f"# A15  N={n_users} meta-training users  subsample_seed={subsample_seed}")
    print(f"{'#'*70}")
    print(f"[A15] meta-train ({len(chosen)}/{len(full_train)}): {chosen}")
    print(f"[A15] val   (unchanged): {config['val_PIDs']}")
    print(f"[A15] test  (unchanged): {config['test_PIDs']}")
    print(f"[A15] training seed    : {training_seed} "
          f"(vary_training_seed={vary_training_seed})")
    print(f"[A15] PID order        : preserved from config "
          f"({'IDENTICAL to M0 -- control is a true no-op' if degenerate else 'subset of the M0 order'})")
    if degenerate:
        print(f"[A15] NOTE: N={n_users} == pool size, so the draw cannot vary. "
              f"Seeds are {'training replicates' if vary_training_seed else 'IDENTICAL runs'}.")

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
        "training_seed":   training_seed,
        "vary_training_seed": bool(vary_training_seed),
        "draw_is_degenerate": bool(degenerate),
        "replicate_kind":  ("training_seed" if vary_training_seed else
                            ("none_draw_degenerate" if degenerate else "user_draw")),
        "train_PIDs":      chosen,
        "train_PIDs_order_preserved": True,
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
            "Fixed 24/4/4 split: outside the paired RM-ANOVA. N=24 is the "
            "harness control and should reproduce the fixed-split M0 number. "
            "PID order is now preserved from config, so at N=24 the training "
            "PID sequence is identical to M0's -- an earlier version sorted it, "
            "which changed the RNG stream and made the control read 90.68% "
            "against the grid's 88.46%.",
            ("At N == pool size the user draw cannot vary, so subsample seeds "
             "give bit-identical runs. Report N=24 as a single run unless "
             "--vary-training-seed was passed."
             if degenerate else
             "Spread across subsample seeds reflects WHICH users were drawn, "
             "not training stochasticity (training seed is pinned)."),
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
    ap.add_argument("--vary-training-seed", action="store_true",
                    help="Also move the training seed with each subsample seed, "
                         "giving genuine independent replicates instead of "
                         "user-draw replicates. Required to get any spread at "
                         "N == pool size, and the way to measure the "
                         "same-config reproducibility band.")
    args = ap.parse_args()

    from ablation_config import TRAIN_PIDS
    degenerate = draw_is_degenerate(list(TRAIN_PIDS), args.n_users)

    if degenerate and len(args.subsample_seeds) > 1 and not args.vary_training_seed:
        print(f"\n[A15] REFUSING to run {len(args.subsample_seeds)} identical jobs.")
        print(f"      N={args.n_users} == the {len(TRAIN_PIDS)}-user meta-training pool, "
              f"so every subsample seed selects the same users and every run is")
        print(f"      bit-identical. This already happened: A15_N24 s0/s1/s2 all "
              f"returned 0.9067592592592594.")
        print(f"      Either run a single seed:      --n-users {args.n_users} --subsample-seeds 0")
        print(f"      or ask for real replicates:    --n-users {args.n_users} "
              f"--subsample-seeds {' '.join(map(str, args.subsample_seeds))} --vary-training-seed")
        raise SystemExit(2)

    results = [run_one(args.n_users, s, vary_training_seed=args.vary_training_seed)
               for s in args.subsample_seeds]
    accs = [r["test_acc"] for r in results]

    print(f"\n{'='*70}")
    print(f"[A15] N={args.n_users} over {len(accs)} subsample seed(s): "
          f"{np.mean(accs)*100:.2f}%"
          + (f" ± {np.std(accs)*100:.2f}% "
             f"(across {'training seeds' if args.vary_training_seed else 'draws'})"
             if len(accs) > 1 else ""))
    if len(accs) > 1 and len(set(accs)) == 1:
        print("      WARNING: all runs returned the identical accuracy. These are "
              "not replicates -- check draw_is_degenerate in the JSONs.")
    if len(accs) == 1 and args.n_users < 16:
        print("      NOTE: single draw at N<16. Treat as indicative only; add "
              "--subsample-seeds 0 1 2 before reporting this point.")
    print(f"      Endpoints already in the paper: N=1 -> 64.6% "
          f"(A8 Subject-Specific EncoderMoE; A7 Subject-Specific Supervised is "
          f"72.2% -- do not swap these), N=24 -> 86.7% (L2SO).")
    print(f"      This sweep is fixed-split. Compare its N=24 point to the "
          f"fixed-split number, and note the same-config band is ~3 points.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
