# A7_A8_subject_specific.py
"""
A7_A8_subject_specific.py
==========================
Ablations A7 and A8: Subject-Specific Models

[original docstring preserved — see rep split design notes above]

test_procedure:
  'hpo_test_split' : Evaluate only on the fixed TEST_PIDS (development).
  'L2SO'           : Evaluate on ALL all_PIDs so subject coverage matches
                     the L2SO cross-subject ablations (M0, A1–A4).
                     Rep-level logic is unchanged — this only controls
                     which subjects are included in the outer loop.

NOTE: Subject-specific models have no cross-subject training phase.
The subject-level loop here is over evaluation subjects only.
Val/test splits are over REPS within each subject, not over subjects.
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

ALL_TRIAL_INDICES_0INDEXED = list(range(10))
NUM_REPS = len(ALL_TRIAL_INDICES_0INDEXED)
A7_PRETRAIN_EPOCHS = 100


# ─────────────────────────────────────────────────────────────────────────────
# Episode building helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_ss_episode(
    tensor_dict, pid, gesture_classes, support_trial_idx,
    query_trial_indices, use_imu, device,
):
    assert support_trial_idx not in query_trial_indices, (
        f"support_trial_idx={support_trial_idx} appears in query_trial_indices={query_trial_indices}."
    )

    support_emg_list, support_imu_list, support_label_list = [], [], []
    query_emg_list,   query_imu_list,   query_label_list   = [], [], []

    for local_label, cls in enumerate(gesture_classes):
        slot    = tensor_dict[pid][cls]
        emg_all = slot["emg"]
        imu_all = slot["imu"]

        num_trials = emg_all.shape[0]
        assert support_trial_idx < num_trials, (
            f"support_trial_idx={support_trial_idx} out of range for "
            f"pid={pid}, class={cls} which has {num_trials} trials."
        )

        support_emg_list.append(emg_all[support_trial_idx].float())
        if imu_all is not None:
            support_imu_list.append(imu_all[support_trial_idx].float())
        support_label_list.append(local_label)

        for q_idx in query_trial_indices:
            assert q_idx < num_trials, (
                f"query_trial_idx={q_idx} out of range for pid={pid}, class={cls}."
            )
            query_emg_list.append(emg_all[q_idx].float())
            if imu_all is not None:
                query_imu_list.append(imu_all[q_idx].float())
            query_label_list.append(local_label)

    support_emg    = torch.stack(support_emg_list).to(device)
    query_emg      = torch.stack(query_emg_list).to(device)
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


def build_ss_eval_episodes(tensor_dict, pid, config, support_trial_idx,
                            query_trial_indices, num_episodes, seed):
    gesture_classes = config["maml_gesture_classes"]
    n_way           = config["n_way"]
    use_imu         = config["use_imu"]
    device          = config["device"]

    from itertools import combinations
    all_combos = list(combinations(sorted(gesture_classes), n_way))

    ep_rng = random.Random(seed)
    ep_rng.shuffle(all_combos)

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


# ─────────────────────────────────────────────────────────────────────────────
# A7 config + per-subject runner (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_config_a7() -> dict:
    config = make_base_config(ablation_id="A7")
    config["subject_specific_model"]  = True
    config["meta_learning"]           = False
    config["use_MOE"]                 = False
    config["batch_size"]              = 10
    config["num_epochs"]              = A7_PRETRAIN_EPOCHS
    config["use_earlystopping"]       = True
    config["earlystopping_patience"]  = 20
    config["ft_steps"]                = 50
    config["ft_lr"]                   = 1e-3
    config["ft_optimizer"]            = "adam"
    config["ft_weight_decay"]         = config["weight_decay"]
    return config


def run_a7_one_subject(pid, seed_idx, seed, config, tensor_dict):
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    train_trial_idx    = seed_idx % NUM_REPS
    val_trial_idx      = (seed_idx + 1) % NUM_REPS
    test_trial_indices = [
        i for i in ALL_TRIAL_INDICES_0INDEXED
        if i != train_trial_idx and i != val_trial_idx
    ]
    assert len(test_trial_indices) == NUM_REPS - 2

    train_rep_num     = train_trial_idx + 1
    val_rep_num       = val_trial_idx   + 1
    support_trial_idx = train_trial_idx

    print(f"[A7 | {pid} | seed_idx={seed_idx}] "
          f"Train rep: {train_rep_num} | Val rep: {val_rep_num} | "
          f"Test reps: {[i+1 for i in test_trial_indices]}")

    config["train_PIDs"] = [pid]
    config["val_PIDs"]   = [pid]
    config["train_reps"] = [train_rep_num]
    config["val_reps"]   = [val_rep_num]

    model = build_supervised_no_moe_model(config)
    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    assert len(train_dl.dataset) > 0, (
        f"[A7 | {pid} | seed_idx={seed_idx}] Empty train set."
    )

    trained_model, history = pretrain(model, train_dl, val_dl, config)

    episodes = build_ss_eval_episodes(
        tensor_dict, pid, config,
        support_trial_idx   = support_trial_idx,
        query_trial_indices = test_trial_indices,
        num_episodes        = NUM_TEST_EPISODES,
        seed                = seed,
    )

    episode_accs = []
    for ep in episodes:
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
    print(f"[A7 | {pid} | seed_idx={seed_idx}] Acc: {mean_acc*100:.2f}%")

    return {
        "pid":           pid,
        "seed":          seed,
        "seed_idx":      seed_idx,
        "train_rep_num": train_rep_num,
        "val_rep_num":   val_rep_num,
        "mean_acc":      mean_acc,
        "n_episodes":    len(episode_accs),
        "n_params":      count_parameters(model),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A8 config + per-subject runner (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_config_a8() -> dict:
    config = make_base_config(ablation_id="A8")
    config["subject_specific_model"] = True
    return config


def run_a8_one_subject(pid, seed_idx, seed, config, tensor_dict):
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    support_trial_idx   = seed_idx % NUM_REPS
    query_trial_indices = [i for i in ALL_TRIAL_INDICES_0INDEXED if i != support_trial_idx]
    support_rep_num     = support_trial_idx + 1

    print(f"[A8 | {pid} | seed_idx={seed_idx}] "
          f"Support rep: {support_rep_num} | Query reps: {[i+1 for i in query_trial_indices]}")

    model = build_maml_moe_model(config)

    episodes = build_ss_eval_episodes(
        tensor_dict, pid, config,
        support_trial_idx   = support_trial_idx,
        query_trial_indices = query_trial_indices,
        num_episodes        = NUM_TEST_EPISODES,
        seed                = seed,
    )

    episode_accs = []
    for ep in episodes:
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
    print(f"[A8 | {pid} | seed_idx={seed_idx}] Acc: {mean_acc*100:.2f}%")

    return {
        "pid":             pid,
        "seed":            seed,
        "seed_idx":        seed_idx,
        "support_rep_num": support_rep_num,
        "mean_acc":        mean_acc,
        "n_episodes":      len(episode_accs),
        "n_params":        count_parameters(model),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_subject_specific_ablation(ablation_id, subject_runner, config,
                                   description, tensor_dict):
    """
    Outer loop: iterate over eval subjects × NUM_FINAL_SEEDS.

    Which subjects are evaluated depends on test_procedure:
      'hpo_test_split' → config['test_PIDs']  (fixed small set)
      'L2SO'           → config['all_PIDs']   (all subjects, matches cross-subject coverage)
    """
    test_procedure = config["test_procedure"]
    assert test_procedure in ("hpo_test_split", "L2SO"), (
        f"Unknown test_procedure '{test_procedure}'."
    )

    eval_pids = config["test_PIDs"] if test_procedure == "hpo_test_split" \
                else config["all_PIDs"]

    print(f"\n{ablation_id} CONFIG (test_procedure={test_procedure}):")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    print(f"Evaluating on {len(eval_pids)} subjects: {eval_pids}")

    all_results = []
    for pid in eval_pids:
        for seed_idx in range(NUM_FINAL_SEEDS):
            actual_seed = FIXED_SEED + seed_idx
            print(f"\n{'='*70}")
            print(f"[{ablation_id}] PID={pid}  seed_idx={seed_idx+1}/{NUM_FINAL_SEEDS}  "
                  f"(seed={actual_seed})")
            print(f"{'='*70}")
            result = subject_runner(pid, seed_idx, actual_seed, config, tensor_dict)
            all_results.append(result)

    pid_to_accs = defaultdict(list)
    for r in all_results:
        pid_to_accs[r["pid"]].append(r["mean_acc"])

    per_subject_mean = {
        pid: float(np.mean(accs))
        for pid, accs in pid_to_accs.items()
    }
    subject_means = list(per_subject_mean.values())

    summary = {
        "ablation_id":      ablation_id,
        "description":      description,
        "test_procedure":   test_procedure,
        "n_params":         next(r["n_params"] for r in all_results),
        "all_results":      all_results,
        "per_subject_mean": per_subject_mean,
        "mean_acc":         float(np.mean(subject_means)),
        "std_acc":          float(np.std(subject_means)),
        "n_subjects":       len(per_subject_mean),
        "num_seeds":        NUM_FINAL_SEEDS,
        "config_snapshot":  {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[{ablation_id}] FINAL ({test_procedure}) across {len(per_subject_mean)} subjects: "
          f"{summary['mean_acc']*100:.2f}% ± {summary['std_acc']*100:.2f}%")
    print(f"{'='*70}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["A7", "A8", "both"], default="both")
    args = parser.parse_args()

    config_for_path = make_base_config(ablation_id="path_resolve")
    tensor_dict_path = os.path.join(
        config_for_path["dfs_load_path"], "segfilt_rts_tensor_dict.pkl"
    )
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
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