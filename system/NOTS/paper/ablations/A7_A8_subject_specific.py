"""
A7_A8_subject_specific.py
==========================
Ablations A7 and A8: Subject-Specific Models

Both ablations enforce a strict 1-sample-per-class information budget, identical
to M0's test-time adaptation budget. M0 gets 1 sample/class as its support set.
A7 and A8 get exactly the same: 1 sample/class. No more, no less.

A7: Subject-Specific CNN-LSTM (Fair 1-shot Baseline)
    - Pretrain: flat supervised on 1 rep/class × 10 classes = 10 samples.
    - Eval: replace 10-class head with fresh 3-class head, fine-tune head-only
      on the same 1-shot support set (3 samples), evaluate on query set.
    - This is fair to M0: both see exactly 1 sample/class of the target subject.

A8: Subject-Specific MAML + MoE (Fair 1-shot Baseline)
    - No pretraining. Random init.
    - Eval: MAML inner-loop adapt on 1-shot support set (3 samples), eval on query.
    - MAML pretraining is impossible with 1 sample/class — the bi-level objective
      requires a support/query split within each episode, which needs >1 sample/class.

────────────────────────────────────────────────────────────────────────────────
Rep split design (addresses NeurIPS reviewer fairness concern):
────────────────────────────────────────────────────────────────────────────────
We have 10 reps per class. The 1-shot constraint says the model (or its support
set) may only see 1 rep/class of the target subject. The remaining 9 reps are
the query pool.

Problem: if we always use rep 1 as support, our accuracy estimate has higher
variance than cross-subject models (which average over random rep-as-support
across 500 episodes). To make the comparison symmetric, we vary which rep is
the "support/train rep" across the NUM_FINAL_SEEDS seed runs:

    seed run 0 → support rep = rep 1  (0-indexed trial 0)
    seed run 1 → support rep = rep 2  (0-indexed trial 1)
    seed run 2 → support rep = rep 3  (0-indexed trial 2)
    ...
    seed run k → support rep = rep (k+1)  (0-indexed trial k)

This averages over the "which rep is support" variance across seeds, exactly as
cross-subject models average over it across episodes. The 1-shot constraint is
preserved: within each seed run, the model sees exactly 1 rep/class.

Concretely for each seed run:
    support_trial_idx  = seed_idx % 10          (0-indexed)
    query_trial_indices = all other 9 indices    (0-indexed)

Eval episodes within a seed run:
    - Support is fixed (always the designated rep for this seed).
    - Task (which 3 gesture classes) varies across episodes → C(10,3) = 120
      distinct class combinations → plenty of episodic variance.

────────────────────────────────────────────────────────────────────────────────
Why we do NOT use MetaGestureDataset for subject-specific eval:
────────────────────────────────────────────────────────────────────────────────
MetaGestureDataset._build_episode() shuffles the available trial pool and
assigns the first k_shot trials as support. Passing all 10 trials would
randomly assign any trial as support — violating the "support = training rep"
contract for A7, and the "support = fixed budget rep" contract for A8.

Instead, we build support/query tensors explicitly. This makes the split
100% transparent and auditable, with no hidden sampling behavior.

────────────────────────────────────────────────────────────────────────────────
Early stopping for A7:
────────────────────────────────────────────────────────────────────────────────
With only 10 training samples (1 rep × 10 classes), val_reps = train_reps
means early stopping would just measure memorization of the training set, which
is meaningless. We disable early stopping and run for a fixed number of epochs
(A7_PRETRAIN_EPOCHS), analogous to A8's fixed inner-loop steps. This makes
the two ablations directly comparable in terms of training budget.

Usage:
    python A7_A8_subject_specific.py --ablation A7
    python A7_A8_subject_specific.py --ablation A8
    python A7_A8_subject_specific.py --ablation both
"""

import os, sys, copy, json, argparse
import numpy as np
import torch
import pickle
import random
from collections import defaultdict

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config, build_supervised_no_moe_model, build_maml_moe_model,
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS, NUM_TEST_EPISODES,
    save_results, count_parameters, RUN_DIR,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain
from pretraining.pretrain_finetune import finetune_and_eval_user
from MAML.maml_data_pipeline import maml_mm_collate, reorient_tensor_dict
from MAML.mamlpp import mamlpp_adapt_and_eval

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_TRIAL_INDICES_0INDEXED = list(range(10))   # 0-indexed positions into (num_trials, T, C)
NUM_REPS = len(ALL_TRIAL_INDICES_0INDEXED)     # 10

# A7: fixed training schedule, no early stopping (see docstring)
A7_PRETRAIN_EPOCHS = 100

from ablation_config import TEST_PIDS


# =============================================================================
# Shared helper: build explicit support + query tensors for one episode
# =============================================================================

def build_ss_episode(
    tensor_dict: dict,
    pid: str,
    gesture_classes: list,       # the n_way classes selected for this episode
    support_trial_idx: int,      # 0-indexed; which trial is the support sample
    query_trial_indices: list,   # 0-indexed; which trials form the query pool
    use_imu: bool,
    device: torch.device,
) -> dict:
    """
    Build one subject-specific 1-shot N-way episode with an EXPLICIT rep split.

    Support: exactly trial `support_trial_idx` for each gesture class.
    Query:   all trials in `query_trial_indices` for each gesture class.

    Labels are remapped to 0 … n_way-1 in the order of `gesture_classes`.

    Returns a dict with keys:
        support_emg    : (n_way, C_emg, T)
        support_imu    : (n_way, C_imu, T) or None
        support_labels : (n_way,)   long, values in 0 … n_way-1
        query_emg      : (n_way * len(query_trial_indices), C_emg, T)
        query_imu      : same shape or None
        query_labels   : (n_way * len(query_trial_indices),) long
    """
    assert support_trial_idx not in query_trial_indices, (
        f"support_trial_idx={support_trial_idx} appears in query_trial_indices={query_trial_indices}. "
        "Support and query must be disjoint."
    )

    support_emg_list, support_imu_list, support_label_list = [], [], []
    query_emg_list,   query_imu_list,   query_label_list   = [], [], []

    for local_label, cls in enumerate(gesture_classes):
        slot    = tensor_dict[pid][cls]
        emg_all = slot["emg"]   # (num_trials, C, T) after reorient_tensor_dict
        imu_all = slot["imu"]   # (num_trials, C_imu, T) or None

        num_trials = emg_all.shape[0]
        assert support_trial_idx < num_trials, (
            f"support_trial_idx={support_trial_idx} out of range for "
            f"pid={pid}, class={cls} which has {num_trials} trials."
        )

        # Support: single fixed trial
        support_emg_list.append(emg_all[support_trial_idx].float())
        if imu_all is not None:
            support_imu_list.append(imu_all[support_trial_idx].float())
        support_label_list.append(local_label)

        # Query: all designated query trials
        for q_idx in query_trial_indices:
            assert q_idx < num_trials, (
                f"query_trial_idx={q_idx} out of range for "
                f"pid={pid}, class={cls} which has {num_trials} trials."
            )
            query_emg_list.append(emg_all[q_idx].float())
            if imu_all is not None:
                query_imu_list.append(imu_all[q_idx].float())
            query_label_list.append(local_label)

    support_emg    = torch.stack(support_emg_list).to(device)        # (n_way, C, T)
    query_emg      = torch.stack(query_emg_list).to(device)          # (n_way*q, C, T)
    support_labels = torch.tensor(support_label_list, dtype=torch.long).to(device)
    query_labels   = torch.tensor(query_label_list,   dtype=torch.long).to(device)

    support_imu = None
    query_imu   = None
    if support_imu_list:
        support_imu = torch.stack(support_imu_list).to(device)
        query_imu   = torch.stack(query_imu_list).to(device)

    return {
        "support_emg":    support_emg,
        "support_imu":    support_imu,
        "support_labels": support_labels,
        "query_emg":      query_emg,
        "query_imu":      query_imu,
        "query_labels":   query_labels,
    }


def build_ss_eval_episodes(
    tensor_dict: dict,
    pid: str,
    config: dict,
    support_trial_idx: int,
    query_trial_indices: list,
    num_episodes: int,
    seed: int,
) -> list:
    """
    Generate `num_episodes` subject-specific eval episodes for `pid`.

    Support is always `support_trial_idx` (fixed for this seed run).
    Task (which n_way classes) is sampled without replacement from the 10
    gesture classes, giving C(10,3) = 120 distinct class combinations.
    Episodes cycle through these combinations deterministically.

    Returns:
        list of episode dicts (keys: support_emg, support_imu, support_labels,
        query_emg, query_imu, query_labels).
    """
    gesture_classes = config["maml_gesture_classes"]
    n_way           = config["n_way"]
    use_imu         = config["use_imu"]
    device          = config["device"]

    # Pre-enumerate all C(n_classes, n_way) class combinations deterministically
    from itertools import combinations
    all_combos = list(combinations(sorted(gesture_classes), n_way))

    # Seed a local RNG so episodes are reproducible but independent of global state
    ep_rng = random.Random(seed)
    ep_rng.shuffle(all_combos)

    # Cycle through combos to fill num_episodes
    episodes = []
    for ep_idx in range(num_episodes):
        classes = list(all_combos[ep_idx % len(all_combos)])
        ep = build_ss_episode(
            tensor_dict, pid, classes,
            support_trial_idx, query_trial_indices,
            use_imu, device,
        )
        episodes.append(ep)

    return episodes


# =============================================================================
# A7: Subject-Specific CNN-LSTM
# =============================================================================

def build_config_a7() -> dict:
    config = make_base_config(ablation_id="A7")
    config["subject_specific_model"] = True
    config["meta_learning"]          = False
    config["use_MOE"]                = False
    config["batch_size"]             = 10   # exactly the dataset size (1 rep × 10 classes)

    # Fixed training schedule — no early stopping (see module docstring).
    config["num_epochs"]          = A7_PRETRAIN_EPOCHS
    config["use_earlystopping"]   = False

    # Head-only fine-tuning at eval. The backbone learned subject-specific
    # features from 10-sample pretraining. Touching the backbone with 3 support
    # samples would overwrite those features — strictly worse.
    config["ft_steps"]        = 50
    config["ft_lr"]           = 1e-3
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]

    return config


def run_a7_one_subject(
    pid: str,
    seed_idx: int,      # 0-indexed seed run (determines which rep is support)
    seed: int,          # actual RNG seed value
    config: dict,
    tensor_dict: dict,
) -> dict:
    """
    Train and evaluate A7 for one subject under one seed run.

    Information budget: 1 rep/class. Same as M0.

    The support rep rotates with seed_idx so that across NUM_FINAL_SEEDS runs
    we average over the same "which rep is support" variance as cross-subject
    episodic eval. Within a run, support is fixed; task (class subset) varies.

    Step 1 — Pretrain backbone:
        Flat supervised training on the support rep only (10 samples, 10-class head).

    Step 2 — Episodic eval:
        For each episode, sample n_way classes from the 10 gesture classes.
        Support  = support rep, selected n_way classes (1 sample/class, same as training).
        Query    = all remaining 9 reps, selected n_way classes (9 samples/class).
        Fine-tune head-only on the 3 support samples, then eval on query.
        finetune_and_eval_user handles head replacement internally on each episode.
    """
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    # ── Determine which rep is support for this seed run ──────────────────────
    support_trial_idx  = seed_idx % NUM_REPS
    query_trial_indices = [i for i in ALL_TRIAL_INDICES_0INDEXED if i != support_trial_idx]

    # 1-indexed rep number for logging / pretrain dataloader (which uses 1-indexed rep nums)
    support_rep_num = support_trial_idx + 1

    print(f"[A7 | {pid} | seed_idx={seed_idx}] "
          f"Support rep: {support_rep_num} (trial_idx={support_trial_idx}) | "
          f"Query reps: {[i+1 for i in query_trial_indices]}")

    # ── Step 1: Flat pretrain on support rep only ─────────────────────────────
    # train_PIDs and val_PIDs both = [pid] — subject-specific.
    # val_reps = train_reps = [support_rep_num] because we have no other data
    # to validate on. Early stopping is disabled (see build_config_a7), so
    # val loss is only logged for diagnostics and does not affect training.
    config["train_PIDs"] = [pid]
    config["val_PIDs"]   = [pid]
    config["train_reps"] = [support_rep_num]
    config["val_reps"]   = [support_rep_num]

    model = build_supervised_no_moe_model(config)

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert len(train_dl.dataset) > 0, (
        f"[A7 | {pid} | seed_idx={seed_idx}] Empty train set. "
        f"Check that pid={pid} and rep_num={support_rep_num} exist in tensor_dict."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    # Load best state. With early stopping disabled this is the final epoch,
    # but pretrain() still tracks best train loss so we load it for consistency.

    # ── Step 2: Episodic eval ─────────────────────────────────────────────────
    # Build episodes explicitly — no MetaGestureDataset — so the support/query
    # rep split is enforced deterministically (see module docstring).
    episodes = build_ss_eval_episodes(
        tensor_dict, pid, config,
        support_trial_idx   = support_trial_idx,
        query_trial_indices = query_trial_indices,
        num_episodes        = NUM_TEST_EPISODES,
        seed                = seed,
    )

    episode_accs = []
    for ep in episodes:
        # finetune_and_eval_user replaces the head internally on its deep copy.
        # We do NOT call replace_head_for_eval here — that would be redundant
        # and would reinitialise the head a second time inside the function.
        metrics = finetune_and_eval_user(
            trained_model, config,
            support_emg    = ep["support_emg"],
            support_imu    = ep["support_imu"],
            support_labels = ep["support_labels"],
            query_emg      = ep["query_emg"],
            query_imu      = ep["query_imu"],
            query_labels   = ep["query_labels"],
            mode           = "head_only",
        )
        episode_accs.append(metrics["acc"])

    mean_acc = float(np.mean(episode_accs))
    print(f"[A7 | {pid} | seed_idx={seed_idx}] "
          f"Acc: {mean_acc*100:.2f}%  ({len(episode_accs)} eps, "
          f"support_rep={support_rep_num})")

    return {
        "pid":               pid,
        "seed":              seed,
        "seed_idx":          seed_idx,
        "support_rep_num":   support_rep_num,
        "mean_acc":          mean_acc,
        "n_episodes":        len(episode_accs),
        "n_params":          count_parameters(model),
    }


# =============================================================================
# A8: Subject-Specific MAML + MoE (random init, no pretraining)
# =============================================================================

def build_config_a8() -> dict:
    config = make_base_config(ablation_id="A8")
    config["subject_specific_model"] = True
    # All other keys (n_way, k_shot, q_query, MAML hparams, MoE hparams)
    # inherit from make_base_config — identical to M0 so the only variable
    # is the absence of cross-subject pretraining.
    return config


def run_a8_one_subject(
    pid: str,
    seed_idx: int,
    seed: int,
    config: dict,
    tensor_dict: dict,
) -> dict:
    """
    Evaluate A8 for one subject under one seed run.

    Information budget: 1 rep/class. Same as M0.

    No pretraining. Random init. MAML inner-loop adapt on the 1-shot support
    set (n_way samples), eval on query. The gap M0 vs A8 isolates exactly the
    value of cross-subject meta-training.

    Support rep rotates with seed_idx (same logic as A7) so variance sources
    are symmetric across ablations.
    """
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    # ── Determine which rep is support for this seed run ──────────────────────
    support_trial_idx   = seed_idx % NUM_REPS
    query_trial_indices = [i for i in ALL_TRIAL_INDICES_0INDEXED if i != support_trial_idx]
    support_rep_num     = support_trial_idx + 1

    print(f"[A8 | {pid} | seed_idx={seed_idx}] "
          f"Support rep: {support_rep_num} (trial_idx={support_trial_idx}) | "
          f"Query reps: {[i+1 for i in query_trial_indices]}")

    model = build_maml_moe_model(config)

    # ── Episodic eval ─────────────────────────────────────────────────────────
    episodes = build_ss_eval_episodes(
        tensor_dict, pid, config,
        support_trial_idx   = support_trial_idx,
        query_trial_indices = query_trial_indices,
        num_episodes        = NUM_TEST_EPISODES,
        seed                = seed,
    )

    episode_accs = []
    for ep in episodes:
        # mamlpp_adapt_and_eval expects a batch dict with support/query sub-dicts
        # each containing 'emg', 'imu', 'labels'. We construct that here to match
        # what maml_mm_collate would produce from a DataLoader batch.
        batch = {
            "support": {
                "emg":    ep["support_emg"],
                "imu":    ep["support_imu"],
                "labels": ep["support_labels"],
            },
            "query": {
                "emg":    ep["query_emg"],
                "imu":    ep["query_imu"],
                "labels": ep["query_labels"],
            },
        }
        metrics = mamlpp_adapt_and_eval(model, config, batch["support"], batch["query"])
        episode_accs.append(metrics["acc"])

    mean_acc = float(np.mean(episode_accs))
    print(f"[A8 | {pid} | seed_idx={seed_idx}] "
          f"Acc: {mean_acc*100:.2f}%  ({len(episode_accs)} eps, "
          f"support_rep={support_rep_num})")

    return {
        "pid":             pid,
        "seed":            seed,
        "seed_idx":        seed_idx,
        "support_rep_num": support_rep_num,
        "mean_acc":        mean_acc,
        "n_episodes":      len(episode_accs),
        "n_params":        count_parameters(model),
    }


# =============================================================================
# Run orchestration
# =============================================================================

def run_subject_specific_ablation(
    ablation_id: str,
    subject_runner,
    config: dict,
    description: str,
    tensor_dict: dict,
):
    """
    Outer loop: iterate over TEST_PIDS × NUM_FINAL_SEEDS.

    Subject-specific models are evaluated on TEST_PIDS only — there is no
    cross-subject pretraining phase, so train/val PIDs are irrelevant here.
    Each (pid, seed_idx) pair trains (A7) or skips training (A8) from scratch.

    Aggregation:
        per_subject_mean[pid] = mean over seed runs for that pid
        mean_acc              = mean over per_subject_mean values
        std_acc               = std  over per_subject_mean values
    """
    print(f"\n{ablation_id} CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    all_results = []
    for pid in TEST_PIDS:
        for seed_idx in range(NUM_FINAL_SEEDS):
            actual_seed = FIXED_SEED + seed_idx
            print(f"\n{'='*70}")
            print(f"[{ablation_id}] PID={pid}  seed_idx={seed_idx+1}/{NUM_FINAL_SEEDS}  "
                  f"(seed={actual_seed}, support_rep={seed_idx % NUM_REPS + 1})")
            print(f"{'='*70}")
            result = subject_runner(
                pid, seed_idx, actual_seed, config, tensor_dict
            )
            all_results.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    pid_to_accs = defaultdict(list)
    for r in all_results:
        pid_to_accs[r["pid"]].append(r["mean_acc"])

    per_subject_mean = {
        pid: float(np.mean(accs))
        for pid, accs in pid_to_accs.items()
        if accs
    }
    subject_means = list(per_subject_mean.values())

    summary = {
        "ablation_id":      ablation_id,
        "description":      description,
        "n_params":         next(r["n_params"] for r in all_results),
        "all_results":      all_results,
        "per_subject_mean": per_subject_mean,
        "mean_acc":         float(np.mean(subject_means)) if subject_means else float("nan"),
        "std_acc":          float(np.std(subject_means))  if subject_means else float("nan"),
        "n_subjects":       len(per_subject_mean),
        "num_seeds":        NUM_FINAL_SEEDS,
        "config_snapshot":  {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[{ablation_id}] FINAL across {len(per_subject_mean)} subjects: "
          f"{summary['mean_acc']*100:.2f}% ± {summary['std_acc']*100:.2f}%")
    print(f"[{ablation_id}] Support rep per seed: "
          + ", ".join(f"seed{i}→rep{i % NUM_REPS + 1}" for i in range(NUM_FINAL_SEEDS)))
    print(f"{'='*70}")

    return summary


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["A7", "A8", "both"], default="both")
    args = parser.parse_args()

    # Load tensor_dict once and reuse for both ablations. reorient_tensor_dict
    # transposes EMG/IMU from (trials, T, C) → (trials, C, T) in-place and is
    # idempotent, so calling it twice is safe.
    #
    # We use A7's config to get the path (both ablations share the same dataset).
    config_for_path = make_base_config(ablation_id="path_resolve")
    tensor_dict_path = os.path.join(
        config_for_path["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
    )
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    # reorient_tensor_dict modifies in-place and returns tensor_dict for convenience.
    # After this call, all EMG/IMU tensors are (num_trials, C, T), ready for the model.
    tensor_dict = reorient_tensor_dict(full_dict, config_for_path)

    if args.ablation in ("A7", "both"):
        config_a7 = build_config_a7()
        run_subject_specific_ablation(
            "A7", run_a7_one_subject, config_a7,
            description="Subject-Specific CNN-LSTM (Fair 1-shot Baseline)",
            tensor_dict=tensor_dict,
        )

    if args.ablation in ("A8", "both"):
        config_a8 = build_config_a8()
        run_subject_specific_ablation(
            "A8", run_a8_one_subject, config_a8,
            description="Subject-Specific MAML + MoE (Fair 1-shot Baseline, Random Init)",
            tensor_dict=tensor_dict,
        )


if __name__ == "__main__":
    main()