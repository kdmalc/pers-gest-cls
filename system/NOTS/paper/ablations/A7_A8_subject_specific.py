"""
A7_A8_subject_specific.py
==========================
Ablations A7 and A8: Subject-Specific Models

A7: Subject-Specific CNN-LSTM (Supervised Oracle Ceiling)
    Trains and tests on THE SAME subject (no cross-subject generalisation).
    Oracle ceiling: shows how much the cross-subject setup costs.
    Train: flat per-subject, Test: episodic per-subject held-out data.

A8: Subject-Specific MAML + MoE
    Full M0 architecture but trained and evaluated within a single subject.
    Compares to A7 (does MAML/MoE help even within-subject?) and M0
    (what is the cross-subject generalisation cost?).

Subject split strategy (per spec):
  - 80% trials for training, 20% held-out trials for episodic test evaluation.
  - Consistent with the evaluation split used elsewhere.
  - We split by trial index: trials [1..8] → train, trials [9,10] → held-out.
    This gives a clean temporal split (later recordings are harder / closer to real use).

  NOTE: The spec says "held-out gestures or held-out trials — be consistent".
  We use held-out TRIALS (rep indices 9 & 10) because:
    (a) it preserves the full gesture vocabulary for training, which is needed for
        3-way episodic eval at test time;
    (b) it mirrors how real deployment works (all gestures known, but new recordings).
  Confirm this with your PI before the paper submission.

Usage:
    python A7_A8_subject_specific.py --ablation A7   # run A7 only
    python A7_A8_subject_specific.py --ablation A8   # run A8 only
    python A7_A8_subject_specific.py --ablation both # run A7 then A8
"""

import os, sys, copy, json, argparse
import numpy as np
import torch
import pickle
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
    save_results, save_model_checkpoint, count_parameters, RUN_DIR,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain
from pretraining.pretrain_finetune import finetune_and_eval_user
from MAML.maml_data_pipeline import get_maml_dataloaders, MetaGestureDataset, maml_mm_collate
from MAML.mamlpp import mamlpp_pretrain, mamlpp_adapt_and_eval
from torch.utils.data import DataLoader

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Trial split — train on reps 1-8, eval on reps 9-10
TRAIN_TRIAL_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]
HELD_OUT_TRIAL_INDICES = [9, 10]

# We evaluate over ALL subjects in our split (train + val + test)
# because per the spec A7/A8 train on each subject independently.
# Use the full subject set so we can compare within-subject vs cross-subject.
from ablation_config import TRAIN_PIDS, VAL_PIDS, TEST_PIDS
ALL_SUBJECT_PIDS = TRAIN_PIDS + VAL_PIDS + TEST_PIDS


# =============================================================================
# A7: Subject-Specific CNN-LSTM (Oracle Ceiling)
# =============================================================================

def build_config_a7() -> dict:
    config = make_base_config(ablation_id="A7")
    config["subject_specific_model"] = True
    config["meta_learning"] = False
    config["use_MOE"]       = False
    config["batch_size"]    = 64

    # Fine-tuning at eval time (1-shot support before query evaluation)
    config["ft_steps"]        = 50
    config["ft_lr"]           = 1e-3
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]

    return config


def run_a7_one_subject(pid: str, seed: int, config: dict,
                       tensor_dict: dict) -> dict:
    """Train and evaluate one subject for A7."""
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    # Restrict to this single subject
    config["train_PIDs"] = [pid]
    config["val_PIDs"]   = [pid]
    config["train_reps"] = TRAIN_TRIAL_INDICES
    config["val_reps"]   = HELD_OUT_TRIAL_INDICES

    model = build_supervised_no_moe_model(config)

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    if len(train_dl.dataset) == 0:
        print(f"[A7 | {pid} | seed={seed}] WARNING: empty train set — skipping.")
        return {"pid": pid, "seed": seed, "skipped": True}

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    trained_model.load_state_dict(history["best_state"])

    # Episodic eval on held-out trials of the SAME subject
    test_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = [pid],
        target_gesture_classes = config["maml_gesture_classes"],
        target_trial_indices   = HELD_OUT_TRIAL_INDICES,
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],
        q_query                = config.get("q_query", None),
        num_eval_episodes      = NUM_TEST_EPISODES,
        is_train               = False,
        seed                   = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2,
                         collate_fn=maml_mm_collate)

    episode_accs = []
    for batch in test_dl:
        support = batch["support"]
        query   = batch["query"]
        metrics = finetune_and_eval_user(
            trained_model, config,
            support_emg=support["emg"], support_imu=support.get("imu"),
            support_labels=support["labels"],
            query_emg=query["emg"],     query_imu=query.get("imu"),
            query_labels=query["labels"],
            mode="full",  # A7 spec: use full fine-tuning
        )
        episode_accs.append(metrics["acc"])

    mean_acc = float(np.mean(episode_accs)) if episode_accs else float("nan")
    print(f"[A7 | {pid} | seed={seed}] Acc: {mean_acc*100:.2f}%  ({len(episode_accs)} eps)")

    return {
        "pid":          pid,
        "seed":         seed,
        "mean_acc":     mean_acc,
        "n_episodes":   len(episode_accs),
        "best_val_acc": float(max(history["val_acc_log"])) if history["val_acc_log"] else float("nan"),
        "n_params":     count_parameters(model),
    }


# =============================================================================
# A8: Subject-Specific MAML + MoE
# =============================================================================

def build_config_a8() -> dict:
    config = make_base_config(ablation_id="A8")
    config["subject_specific_model"] = True
    # Full M0 config, but train/test will be overridden per subject
    return config


def run_a8_one_subject(pid: str, seed: int, config: dict,
                       tensor_dict_path: str) -> dict:
    """Train and evaluate one subject for A8."""
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    # Restrict to this single subject
    config["train_PIDs"]          = [pid]
    config["val_PIDs"]            = [pid]
    config["target_trial_indices"] = TRAIN_TRIAL_INDICES  # inner loop uses train trials

    model = build_maml_moe_model(config)

    # Per-subject episodic dataloaders
    # Note: MetaGestureDataset needs at least n_way gesture classes and k_shot trials.
    # With only one subject and 10 classes, 3-way 1-shot is feasible.
    try:
        train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)
    except Exception as e:
        print(f"[A8 | {pid} | seed={seed}] WARNING: dataloader failed ({e}) — skipping.")
        return {"pid": pid, "seed": seed, "skipped": True}

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    trained_model.load_state_dict(train_history["best_state"])

    # Eval on held-out trials of the SAME subject
    with open(tensor_dict_path, "rb") as f:
        full_dict   = pickle.load(f)
    tensor_dict = full_dict["data"]

    test_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = [pid],
        target_gesture_classes = config["maml_gesture_classes"],
        target_trial_indices   = HELD_OUT_TRIAL_INDICES,
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],
        q_query                = config.get("q_query", None),
        num_eval_episodes      = NUM_TEST_EPISODES,
        is_train               = False,
        seed                   = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2,
                         collate_fn=maml_mm_collate)

    episode_accs = []
    for batch in test_dl:
        metrics = mamlpp_adapt_and_eval(
            trained_model, config, batch["support"], batch["query"]
        )
        episode_accs.append(metrics["acc"])

    mean_acc = float(np.mean(episode_accs)) if episode_accs else float("nan")
    print(f"[A8 | {pid} | seed={seed}] Acc: {mean_acc*100:.2f}%  ({len(episode_accs)} eps)")

    return {
        "pid":          pid,
        "seed":         seed,
        "mean_acc":     mean_acc,
        "n_episodes":   len(episode_accs),
        "best_val_acc": float(train_history["best_val_acc"]),
        "n_params":     count_parameters(model),
    }


# =============================================================================
# Run logic
# =============================================================================

def run_subject_specific_ablation(ablation_id: str, subject_runner, config: dict,
                                   description: str, **runner_kwargs):
    """
    Run a subject-specific ablation over all subjects and all seeds.
    """
    print(f"\n{ablation_id} CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    all_results = []
    for pid in ALL_SUBJECT_PIDS:
        for seed_idx in range(NUM_FINAL_SEEDS):
            actual_seed = FIXED_SEED + seed_idx
            print(f"\n{'='*70}")
            print(f"[{ablation_id}] PID={pid} seed {seed_idx+1}/{NUM_FINAL_SEEDS}")
            print(f"{'='*70}")
            result = subject_runner(pid, actual_seed, config, **runner_kwargs)
            all_results.append(result)

    # Aggregate: mean over seeds per subject, then mean over subjects
    pid_to_accs = defaultdict(list)
    for r in all_results:
        if not r.get("skipped"):
            pid_to_accs[r["pid"]].append(r["mean_acc"])

    per_subject_mean = {pid: float(np.mean(accs)) for pid, accs in pid_to_accs.items()
                        if accs}
    subject_means = list(per_subject_mean.values())

    summary = {
        "ablation_id":      ablation_id,
        "description":      description,
        "n_params":         next((r["n_params"] for r in all_results if not r.get("skipped")), None),
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
    print(f"{'='*70}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["A7", "A8", "both"], default="both")
    args = parser.parse_args()

    if args.ablation in ("A7", "both"):
        config_a7 = build_config_a7()
        tensor_dict_path = os.path.join(config_a7["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
        with open(tensor_dict_path, "rb") as f:
            full_dict   = pickle.load(f)
        tensor_dict = full_dict["data"]

        run_subject_specific_ablation(
            "A7", run_a7_one_subject, config_a7,
            description="Subject-Specific CNN-LSTM (Supervised Oracle Ceiling)",
            tensor_dict=tensor_dict,
        )

    if args.ablation in ("A8", "both"):
        config_a8 = build_config_a8()
        tensor_dict_path = os.path.join(config_a8["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

        run_subject_specific_ablation(
            "A8", run_a8_one_subject, config_a8,
            description="Subject-Specific MAML + MoE",
            tensor_dict_path=tensor_dict_path,
        )


if __name__ == "__main__":
    main()
