# A16_upper_bound.py
"""
A16_upper_bound.py
==================
Reviewer ARaC's requested UPPER BOUND: fine-tune an already-pretrained,
parameter-matched model on (as close as possible to) ALL of a test user's own
data, and evaluate on data it never trained on.

WHAT ARaC ASKED, AND WHAT THIS SCRIPT DOES
------------------------------------------
Two questions in the review:

  (1) "Does the Subject-Specific Transfer Learning model pretrain on the larger
      dataset of all users, or is it trained from scratch on a single user?"
      Answer: from scratch, on one user, on ONE rep (10 samples). A7/A8 set
      train_PIDs = [pid]; there is no cross-subject phase at all. This is a
      factual answer for the rebuttal text -- no experiment needed.

  (2) "It would be nice to see an upper bound achieved with fine-tuning your own
      parameter-matched pre-trained model on all of a test user's training data."
      That is this script.

WHY THIS IS NOT ALREADY IN THE PAPER
------------------------------------
A4 is not it: A4 is MAML-trained and MAML-adapted, isolating whether MoE helps
beyond raw capacity.

A2 is the closest relative -- same cross-subject supervised pretraining, same
parameter matching -- but it is reported ONLY at K=1, and its fine-tune is
deliberately crippled to match MAML's adaptation budget so the comparison is
fair (ft_steps = maml_inner_steps_eval, ft_lr = maml_alpha_init_eval). Correct
for the headline table; wrong for a ceiling, because it caps BOTH the data and
the optimisation budget at what MAML gets.

A16 changes exactly two things versus the existing A2/M0 eval path:
  1. Adaptation data : 9 of the user's reps instead of K=1.
  2. Adaptation budget: a properly tuned Adam fine-tune instead of the
                        MAML-mirrored ~10-25 step budget.
Backbone, pretraining, and parameter matching are UNTOUCHED. Nothing is
retrained -- see below.

NO RETRAINING: PER-TEST-SUBJECT CHECKPOINTS
--------------------------------------------
Only the eval-time protocol changed, so the existing L2SO checkpoints are
reused directly. In L2SO fold i, all_PIDs[i] was the held-out test subject, so
fold i's checkpoint is exactly "the parameter-matched model pretrained on all
users EXCEPT this one". A16 loads, per test subject, the fold checkpoint that
held that subject out. Fine-tuning that model on that subject's data is
therefore clean -- the subject was never in pretraining.

Use find_a16_checkpoints.py to build the manifest. It also screens out the
April A2 checkpoints, which predate parameter matching (~0.6M params vs the
matched ~6.1M) and would understate the CNN-LSTM ceiling by a wide margin.

THE SPLIT: LEAVE-ONE-REP-OUT
-----------------------------
Each user has 10 reps per gesture. "Fine-tune on all of it" cannot literally
mean train AND evaluate on the same reps -- that reports training accuracy.
Some held-out data is unavoidable.

Rather than fixing one arbitrary split, A16 uses leave-one-rep-out: each of the
10 reps takes a turn as the held-out eval rep while the other 9 form the
adaptation set; results are averaged over all 10 folds. 9 adaptation reps is
the most any single-held-out-rep protocol can give the model, which is the
generous reading ARaC is asking for, and there is no split proportion for a
reviewer to call arbitrary.

FINE-TUNE HYPERPARAMETERS
-------------------------
An under-trained upper bound is not an upper bound. ft_lr and ft_steps are
chosen by a small grid run on VAL subjects and then frozen for the TEST
subjects, so the reported bound is not oracle-tuned on the numbers being
reported. Each val subject uses its own fold checkpoint, same as test.

BASES
-----
  --base A2   parameter-matched supervised CNN-LSTM ("generic CNN-LSTM doing
              transfer learning" -- what ARaC literally asked about)
  --base M0   the full EncoderMoE, plain-fine-tuned rather than MAML-adapted
              ("how close is K-shot adaptation to this model's own ceiling")
  --base both run them back to back in one job

Usage:
    python find_a16_checkpoints.py                       # build the manifest
    python A16_upper_bound.py --base both --manifest .../a16_manifest.json
    python A16_upper_bound.py --base A2   --manifest ... --smoke
"""

import os, sys, copy, json, random, pickle, argparse
from itertools import combinations

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
    make_base_config,
    build_supervised_no_moe_model, build_maml_moe_model,
    compute_matched_filters_for_ablation,
    set_seeds, FIXED_SEED,
    save_results, count_parameters,
    RUN_DIR,
)
from pretraining.pretrain_finetune import finetune_and_eval_user
from MAML.maml_data_pipeline import reorient_tensor_dict

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Protocol constants
# =============================================================================

ALL_REPS = list(range(1, 11))   # 1-indexed rep numbers, 10 reps per gesture

# Fine-tune grid searched on VAL subjects only.
FT_LR_GRID    = [3e-4, 1e-3, 3e-3]
FT_STEPS_GRID = [50, 200]

FT_LR_DEFAULT    = 1e-3
FT_STEPS_DEFAULT = 200

DEFAULT_MANIFEST = "/scratch/my13/kai/runs/paper/ablations/eval/a16_manifest.json"


# =============================================================================
# Episode construction
# =============================================================================

def build_episode(tensor_dict, pid, gesture_classes,
                  support_rep_nums, query_rep_nums, device):
    """
    One episode for a single subject.

    support_rep_nums / query_rep_nums are 1-INDEXED rep numbers; conversion to
    0-indexed tensor positions happens here and nowhere else.

    tensor_dict must already have been through reorient_tensor_dict(), so each
    slot is (trials, C, T) and no permute is needed.
    """
    overlap = set(support_rep_nums) & set(query_rep_nums)
    assert not overlap, (
        f"Support/query rep overlap {sorted(overlap)} for pid={pid}. "
        f"The upper bound is meaningless if adaptation sees the eval rep."
    )

    sup_emg, sup_imu, sup_lbl = [], [], []
    qry_emg, qry_imu, qry_lbl = [], [], []

    for local_label, cls in enumerate(gesture_classes):
        slot     = tensor_dict[pid][cls]
        emg_all  = slot["emg"]
        imu_all  = slot.get("imu", None)
        n_trials = emg_all.shape[0]

        for rep in support_rep_nums:
            idx = rep - 1
            assert 0 <= idx < n_trials, (
                f"support rep {rep} out of range for pid={pid} class={cls} "
                f"({n_trials} trials available)."
            )
            sup_emg.append(emg_all[idx].float())
            sup_lbl.append(local_label)
            if imu_all is not None:
                sup_imu.append(imu_all[idx].float())

        for rep in query_rep_nums:
            idx = rep - 1
            assert 0 <= idx < n_trials, (
                f"query rep {rep} out of range for pid={pid} class={cls} "
                f"({n_trials} trials available)."
            )
            qry_emg.append(emg_all[idx].float())
            qry_lbl.append(local_label)
            if imu_all is not None:
                qry_imu.append(imu_all[idx].float())

    return {
        "support_emg":    torch.stack(sup_emg).to(device),
        "support_labels": torch.tensor(sup_lbl, dtype=torch.long).to(device),
        "query_emg":      torch.stack(qry_emg).to(device),
        "query_labels":   torch.tensor(qry_lbl, dtype=torch.long).to(device),
        "support_imu":    torch.stack(sup_imu).to(device) if sup_imu else None,
        "query_imu":      torch.stack(qry_imu).to(device) if qry_imu else None,
    }


def enumerate_class_combos(gesture_classes, n_way, num_combos, seed):
    """
    All C(len(classes), n_way) class combinations, shuffled with a fixed seed
    and truncated. n_way=3 over 10 classes -> 120 combos, so --num-combos 120
    covers the complete space exactly once rather than sampling it.
    """
    combos = [list(c) for c in combinations(sorted(gesture_classes), n_way)]
    rng = random.Random(seed)
    rng.shuffle(combos)
    if num_combos < len(combos):
        return combos[:num_combos]
    if num_combos > len(combos):
        print(f"[A16] --num-combos {num_combos} exceeds the {len(combos)} distinct "
              f"{n_way}-way combinations; evaluating all {len(combos)} once.")
    return combos


# =============================================================================
# Checkpoint loading
# =============================================================================

def load_manifest(path):
    p = Path(path)
    assert p.exists(), (
        f"Manifest not found: {p}\n"
        f"Build it first:  python find_a16_checkpoints.py"
    )
    with open(p) as f:
        blob = json.load(f)
    return blob["manifest"]


def build_model_for_base(base, config):
    if base == "A2":
        return build_supervised_no_moe_model(config).to(config["device"])
    if base == "M0":
        return build_maml_moe_model(config).to(config["device"])
    raise ValueError(f"Unknown base '{base}'.")


def load_subject_checkpoint(base, pid, manifest, config):
    """
    Load the L2SO fold checkpoint that held `pid` out of pretraining.

    A head-shaped mismatch is tolerated: replace_head_for_eval() discards the
    pretrained head per episode anyway. A mismatch anywhere else means the
    backbone is not the pretrained one, which would silently invalidate the
    result -- so that raises.
    """
    entry = manifest.get(base, {}).get(pid)
    assert entry is not None, (
        f"No {base} checkpoint for test subject {pid} in the manifest. "
        f"Run find_a16_checkpoints.py and check its coverage report."
    )
    ckpt_path = Path(entry["path"])
    assert ckpt_path.exists(), f"Manifest points at a missing file: {ckpt_path}"

    model = build_model_for_base(base, config)
    ckpt  = torch.load(ckpt_path, map_location=config["device"], weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("best_state", ckpt))

    model_sd = model.state_dict()
    filtered, dropped = {}, []
    for k, v in state.items():
        if k in model_sd and hasattr(v, "shape") and v.shape != model_sd[k].shape:
            dropped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v

    non_head = [k for k, _, _ in dropped if "head" not in k]
    assert not non_head, (
        f"Shape mismatch outside the classifier head for {base}/{pid}: "
        f"{non_head}. The backbone would not be the pretrained one. "
        f"Refusing to continue."
    )

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # A checkpoint whose trunk did not actually load is worse than no result.
    trunk_missing = [k for k in missing if "head" not in k]
    assert len(trunk_missing) == 0, (
        f"{len(trunk_missing)} trunk tensors missing after load for "
        f"{base}/{pid} (e.g. {trunk_missing[:5]}). Checkpoint/architecture "
        f"mismatch -- refusing to report a bogus upper bound."
    )

    n_params = count_parameters(model)
    print(f"  [load] {base}/{pid}: {ckpt_path.name}  "
          f"{n_params:,} params  fold={entry.get('fold_idx')}  "
          f"dropped_head={len(dropped)}  unexpected={len(unexpected)}")

    # Unexpected keys are tensors present in the checkpoint with no counterpart
    # in the model, so load_state_dict silently discards them. For M0 these are
    # expected to be MAML++ inner-loop machinery (LSLR per-layer/per-step alphas,
    # MSL state) which plain Adam fine-tuning genuinely does not need. But if a
    # real backbone weight ever lands here under an unrecognised name, the arm
    # would be fine-tuning a partly-default network and still look plausible.
    # Print a prefix histogram so that stays visible rather than assumed.
    unexpected_prefixes = {}
    for k in unexpected:
        head = k.split(".")[0]
        unexpected_prefixes[head] = unexpected_prefixes.get(head, 0) + 1
    if unexpected:
        summary = ", ".join(f"{p}:{c}" for p, c in
                            sorted(unexpected_prefixes.items(),
                                   key=lambda kv: -kv[1])[:8])
        print(f"         unexpected-key prefixes -> {summary}")
        print(f"         examples: {list(unexpected)[:4]}")

    # Anything that looks like a weight/bias on a conv/lstm/expert module is NOT
    # inner-loop bookkeeping and should not be silently discarded.
    suspicious = [k for k in unexpected
                  if any(t in k.lower() for t in
                         ("conv", "lstm", "expert", "encoder", "proj"))
                  and k.split(".")[-1] in ("weight", "bias")]
    assert not suspicious, (
        f"{len(suspicious)} unexpected checkpoint keys look like real backbone "
        f"weights, not MAML inner-loop state (e.g. {suspicious[:5]}). These are "
        f"being discarded, which would leave part of the network at its default "
        f"initialisation. Refusing to report an upper bound from a partly-"
        f"untrained model."
    )

    del ckpt, state
    return model, {
        "path":       str(ckpt_path),
        "fold_idx":   entry.get("fold_idx"),
        "timestamp":  entry.get("timestamp"),
        "n_params":   n_params,
        "dropped_head_tensors": len(dropped),
        "n_unexpected": len(unexpected),
        "unexpected_prefixes": unexpected_prefixes,
    }


# =============================================================================
# Evaluation: leave-one-rep-out
# =============================================================================

def eval_subject_lo1o(model, config, tensor_dict, pid, combos,
                      ft_mode, ft_lr, ft_steps, held_out_reps, label=""):
    """
    Leave-one-rep-out for ONE subject with ONE (already subject-appropriate)
    model. Returns per-fold means and the subject mean.
    """
    cfg = copy.deepcopy(config)
    cfg["ft_lr"]           = float(ft_lr)
    cfg["ft_steps"]        = int(ft_steps)
    cfg["ft_optimizer"]    = "adam"
    cfg["ft_weight_decay"] = float(config["weight_decay"])

    fold_means = {}
    for held_out in held_out_reps:
        adapt_reps = [r for r in ALL_REPS if r != held_out]
        fold_accs = []
        for classes in combos:
            ep = build_episode(tensor_dict, pid, classes,
                               support_rep_nums=adapt_reps,
                               query_rep_nums=[held_out],
                               device=cfg["device"])
            metrics = finetune_and_eval_user(
                model, cfg,
                support_emg    = ep["support_emg"],
                support_imu    = ep["support_imu"],
                support_labels = ep["support_labels"],
                query_emg      = ep["query_emg"],
                query_imu      = ep["query_imu"],
                query_labels   = ep["query_labels"],
                mode           = ft_mode,
            )
            fold_accs.append(metrics["acc"])
        fold_means[held_out] = float(np.mean(fold_accs))
        print(f"    [{label}] pid={pid} held_out_rep={held_out:>2} {ft_mode:<9} "
              f"lr={ft_lr:.1e} steps={ft_steps:<4} "
              f"acc={fold_means[held_out]*100:6.2f}%  (n_combos={len(combos)})")

    subject_mean = float(np.mean(list(fold_means.values())))
    print(f"  [{label}] pid={pid}  SUBJECT MEAN over {len(held_out_reps)} "
          f"folds = {subject_mean*100:.2f}%")
    return subject_mean, fold_means


def eval_cohort(base, manifest, config, tensor_dict, pids, combos,
                ft_mode, ft_lr, ft_steps, held_out_reps, label=""):
    """
    Run leave-one-rep-out across a set of subjects, loading each subject's own
    fold checkpoint. Aggregates at the subject level, matching every other
    ablation in the paper.
    """
    per_subject_mean, per_subject_folds, load_info = {}, {}, {}

    for pid in pids:
        model, info = load_subject_checkpoint(base, pid, manifest, config)
        load_info[pid] = info
        mean, folds = eval_subject_lo1o(
            model, config, tensor_dict, pid, combos,
            ft_mode, ft_lr, ft_steps, held_out_reps, label=label,
        )
        per_subject_mean[pid]  = mean
        per_subject_folds[pid] = folds
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vals = list(per_subject_mean.values())
    return {
        "base":                 base,
        "ft_mode":              ft_mode,
        "ft_lr":                float(ft_lr),
        "ft_steps":             int(ft_steps),
        "held_out_reps":        held_out_reps,
        "n_folds":              len(held_out_reps),
        "n_combos_per_fold":    len(combos),
        "per_subject_mean":     per_subject_mean,
        "per_subject_per_fold": per_subject_folds,
        "checkpoint_info":      load_info,
        "mean_acc":             float(np.mean(vals)),
        "std_acc":              float(np.std(vals)),
        "n_subjects":           len(vals),
    }


def tune_ft_hparams(base, manifest, config, tensor_dict, val_pids, args):
    """
    Grid over (ft_lr, ft_steps) on VAL subjects, using a cheap subset of folds
    and combos. The winner is frozen for the TEST subjects.
    """
    combos = enumerate_class_combos(config["maml_gesture_classes"],
                                    config["n_way"], args.tune_combos, FIXED_SEED)
    tune_folds = ALL_REPS[:args.tune_folds]

    print(f"\n{'='*74}")
    print(f"[A16/{base}] STAGE 1 -- fine-tune HP grid on VAL subjects {val_pids}")
    print(f"      folds={tune_folds}  combos={len(combos)}  "
          f"grid={len(FT_LR_GRID)}x{len(FT_STEPS_GRID)}")
    print(f"{'='*74}")

    grid = []
    for lr in FT_LR_GRID:
        for steps in FT_STEPS_GRID:
            res = eval_cohort(base, manifest, config, tensor_dict, val_pids,
                              combos, ft_mode="full", ft_lr=lr, ft_steps=steps,
                              held_out_reps=tune_folds, label="tune")
            grid.append(res)
            print(f"  [tune/{base}] lr={lr:.1e} steps={steps:<4} "
                  f"-> {res['mean_acc']*100:.2f}%")

    best = max(grid, key=lambda r: r["mean_acc"])
    print(f"\n[A16/{base}] Selected ft_lr={best['ft_lr']:.1e} "
          f"ft_steps={best['ft_steps']}  (val mean {best['mean_acc']*100:.2f}%)")
    return best["ft_lr"], best["ft_steps"], grid


# =============================================================================
# Config
# =============================================================================

def build_config(base: str, args) -> dict:
    """
    Reproduces the architecture flags each base was pretrained under, so the
    checkpoint loads into an identical model.
    """
    config = make_base_config(ablation_id=f"A16_{base}")
    config["n_way"]             = args.n_way
    config["k_shot"]            = 1        # unused; episodes built directly
    config["ft_label_smooth"]   = 0.0
    config["target_trial_reps"] = ALL_REPS

    if base == "A2":
        config["meta_learning"] = False
        config["use_MOE"]       = False
        match_info = compute_matched_filters_for_ablation(
            ablation_id="A16_A2", ablation_config=config,
            match_target="all_experts",
        )
        config["cnn_base_filters"]      = match_info["matched_filters"]
        config["_param_match_target"]   = "all_experts_cnn"
        config["_m0_all_expert_params"] = match_info["m0_all_expert_params"]
        config["_matched_cnn_params"]   = match_info["matched_cnn_params"]
        config["_param_ratio"]          = match_info["param_ratio"]
    elif base == "M0":
        config["meta_learning"] = True
        config["use_MOE"]       = True
    else:
        raise ValueError(f"Unknown base '{base}'.")

    return config


# =============================================================================
# Per-base driver
# =============================================================================

def run_base(base, args, manifest, tensor_dict_cache):
    set_seeds(FIXED_SEED)
    config = build_config(base, args)

    test_pids = args.test_pids or config["test_PIDs"]
    val_pids  = args.val_pids  or config["val_PIDs"]
    config["test_PIDs"] = list(test_pids)
    config["val_PIDs"]  = list(val_pids)

    # make_base_config defaults test_procedure to "L2SO", but A16 evaluates a
    # fixed set of test subjects using per-subject fold checkpoints -- the
    # effective protocol is the fixed split. Left unset, assert_protocol_consistent
    # fires and the saved JSON claims "L2SO" while reporting 4 subjects, which is
    # exactly the mismatch that already contaminated the fewshot_grid JSONs.
    config["test_procedure"] = (
        "L2SO" if len(test_pids) == len(config["all_PIDs"]) else "hpo_test_split"
    )

    held_out_reps = ALL_REPS if not args.smoke else ALL_REPS[:2]

    print(f"\n{'#'*74}")
    print(f"# A16 base={base}")
    print(f"#   protocol : leave-one-rep-out (9 adapt reps / 1 eval rep), "
          f"{len(held_out_reps)} folds")
    print(f"#   test PIDs: {test_pids}")
    print(f"#   val  PIDs: {val_pids}  (fine-tune HP selection only)")
    print(f"#   modes    : {args.modes}")
    print(f"{'#'*74}")

    # Load data once and reuse across bases.
    if tensor_dict_cache.get("td") is None:
        tensor_dict_path = os.path.join(config["dfs_load_path"],
                                        "segfilt_rts_tensor_dict.pkl")
        with open(tensor_dict_path, "rb") as f:
            full_dict = pickle.load(f)
        tensor_dict_cache["td"] = reorient_tensor_dict(full_dict, config)
    tensor_dict = tensor_dict_cache["td"]

    # ── Stage 1: fine-tune HP selection ──────────────────────────────────────
    grid = None
    if args.ft_lr is not None and args.ft_steps is not None:
        ft_lr, ft_steps = args.ft_lr, args.ft_steps
        print(f"[A16/{base}] Using supplied ft_lr={ft_lr:.1e} ft_steps={ft_steps}; "
              f"skipping tuning.")
    elif args.stage in ("tune", "both"):
        ft_lr, ft_steps, grid = tune_ft_hparams(base, manifest, config,
                                                tensor_dict, val_pids, args)
    else:
        ft_lr, ft_steps = FT_LR_DEFAULT, FT_STEPS_DEFAULT
        print(f"[A16/{base}] No tuning stage; using "
              f"ft_lr={ft_lr:.1e} ft_steps={ft_steps}.")

    if args.stage == "tune":
        save_results({"ablation_id": config["ablation_id"], "stage": "tune",
                      "base": base, "grid_results": grid,
                      "selected_ft_lr": ft_lr, "selected_ft_steps": ft_steps,
                      "val_PIDs": val_pids},
                     config, tag=f"{base}_tune")
        return None

    # ── Stage 2: the upper bound on test subjects ────────────────────────────
    combos = enumerate_class_combos(config["maml_gesture_classes"],
                                    config["n_way"], args.num_combos, FIXED_SEED)
    print(f"\n{'='*74}")
    print(f"[A16/{base}] STAGE 2 -- leave-one-rep-out upper bound on TEST "
          f"subjects {test_pids}")
    print(f"      combos/fold={len(combos)}  ft_lr={ft_lr:.1e}  ft_steps={ft_steps}")
    print(f"{'='*74}")

    cells = []
    for mode in args.modes:
        cells.append(eval_cohort(base, manifest, config, tensor_dict, test_pids,
                                 combos, ft_mode=mode, ft_lr=ft_lr,
                                 ft_steps=ft_steps, held_out_reps=held_out_reps,
                                 label="run"))

    summary = {
        "ablation_id":  config["ablation_id"],
        "description": (f"Upper bound (reviewer ARaC): {base} parameter-matched "
                        f"pretrained initialisation, fine-tuned leave-one-rep-out "
                        f"(9 adapt reps / 1 held-out eval rep, averaged over "
                        f"{len(held_out_reps)} folds). No retraining: each test "
                        f"subject uses the L2SO fold checkpoint that held it out."),
        "base":          base,
        "protocol":      "leave_one_rep_out",
        "n_folds":       len(held_out_reps),
        "held_out_reps": held_out_reps,
        "seed":          FIXED_SEED,
        "n_way":         config["n_way"],
        "modes":         args.modes,
        "num_combos":    len(combos),
        "ft_lr":         ft_lr,
        "ft_steps":      ft_steps,
        "ft_optimizer":  "adam",
        "ft_hparam_selection": ("val_subject_grid" if grid is not None
                                else "supplied_or_default"),
        "ft_grid_results": grid,
        "cells":         cells,
        "test_PIDs":     list(test_pids),
        "val_PIDs":      list(val_pids),
        "manifest_path": str(args.manifest),
        "config_snapshot": {k: str(v) for k, v in config.items()},
        "note_for_rebuttal": (
            "Compare cells[*].mean_acc against the existing K=1 number for the "
            "same base on the same test subjects. That gap is the answer to "
            "ARaC's question: how much accuracy is left on the table by using "
            "1 shot instead of the user's whole training set."
        ),
    }
    save_results(summary, config, tag=f"{base}_summary")

    print(f"\n{'='*74}")
    print(f"[A16/{base}] FINAL -- leave-one-rep-out ({len(held_out_reps)} folds)")
    for c in cells:
        print(f"  {c['ft_mode']:<9}  {c['mean_acc']*100:6.2f}% +/- "
              f"{c['std_acc']*100:5.2f}%   (n_subjects={c['n_subjects']})")
    print(f"{'='*74}")
    return summary


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="A16: upper bound via leave-one-rep-out fine-tuning on a "
                    "test user's own data (reviewer ARaC)."
    )
    ap.add_argument("--base", choices=["A2", "M0", "both"], default="both")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="Checkpoint manifest from find_a16_checkpoints.py.")
    ap.add_argument("--stage", choices=["tune", "run", "both"], default="both")
    ap.add_argument("--test-pids", nargs="+", default=None,
                    help="Defaults to the paper's fixed test split.")
    ap.add_argument("--val-pids", nargs="+", default=None,
                    help="Used only for fine-tune HP selection.")
    ap.add_argument("--n-way", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["full"],
                    choices=["full", "head_only"])
    ap.add_argument("--num-combos", type=int, default=40,
                    help="Class combinations per fold. 120 = complete 3-way set.")
    ap.add_argument("--tune-combos", type=int, default=10)
    ap.add_argument("--tune-folds", type=int, default=3)
    ap.add_argument("--ft-lr", type=float, default=None)
    ap.add_argument("--ft-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to prove the pipeline executes end to end.")
    args = ap.parse_args()

    if args.smoke:
        args.num_combos  = 2
        args.tune_combos = 1
        args.tune_folds  = 1
        print("[A16] SMOKE MODE -- results are not meaningful.")

    manifest = load_manifest(args.manifest)
    bases = ["A2", "M0"] if args.base == "both" else [args.base]

    # Fail fast on coverage before spending GPU time.
    probe_cfg = make_base_config(ablation_id="A16_probe")
    want_test = args.test_pids or probe_cfg["test_PIDs"]
    want_val  = args.val_pids  or probe_cfg["val_PIDs"]
    for base in bases:
        have = set(manifest.get(base, {}))
        need = set(want_test) | (set(want_val) if args.stage != "run" else set())
        missing = sorted(need - have)
        assert not missing, (
            f"Manifest has no {base} checkpoint for: {missing}\n"
            f"Covered: {sorted(have)}\n"
            f"Re-run find_a16_checkpoints.py, or pass --test-pids/--val-pids "
            f"restricted to covered subjects."
        )
    print(f"[A16] Manifest coverage OK for bases {bases}.")

    tensor_dict_cache = {"td": None}
    summaries = {}
    for base in bases:
        summaries[base] = run_base(base, args, manifest, tensor_dict_cache)

    if args.stage != "tune" and len(bases) > 1:
        print(f"\n{'#'*74}")
        print(f"# A16 COMBINED")
        print(f"{'#'*74}")
        for base in bases:
            s = summaries.get(base)
            if not s:
                continue
            for c in s["cells"]:
                print(f"  {base:<3} {c['ft_mode']:<9} "
                      f"{c['mean_acc']*100:6.2f}% +/- {c['std_acc']*100:5.2f}%")
        print(f"  Compare each against that base's existing K=1 number.")


if __name__ == "__main__":
    main()
