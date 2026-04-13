"""
A7_A8_subject_specific.py
==========================
Ablations A7 and A8: Subject-Specific Models

Both ablations enforce a strict 1-sample-per-class information budget, identical
to M0's test-time adaptation budget. M0 gets 1 sample/class as its support set.
A7 and A8 get exactly the same: 1 sample/class. No more, no less.

A7: Subject-Specific CNN-LSTM (Fair 1-shot Baseline)
    - Pretrain: flat supervised on 1 sample/class × 10 classes = 10 samples.
    - Eval: replace 10-class head with fresh 3-class head, fine-tune head-only
      on the same 1-shot support set (3 samples), evaluate on query set.
    - This is fair to M0: both see exactly 1 sample/class of the target subject.

A8: Subject-Specific MAML + MoE (Fair 1-shot Baseline)
    - No pretraining. Random init.
    - Eval: MAML inner-loop adapt on 1-shot support set (3 samples), eval on query.
    - Fair to M0: same architecture, same adaptation budget, no prior subject info.
    - MAML pretraining is impossible here — bi-level objective requires a
      support/query split within each episode, which needs >1 sample/class.

Trial indexing:
    TRAIN_TRIAL_INDICES    = [1]         — the single rep used everywhere
    HELD_OUT_TRIAL_INDICES = [2..10]     — query pool only (never used for learning)

    For A7: backbone trains on rep 1. The episodic sampler draws its 1-shot
    support from rep 1 (the same data — it is all we have) and query from
    reps 2–10. The support set IS the training set, just as M0's support set
    is the entirety of what it knows about the target subject.

Usage:
    python A7_A8_subject_specific.py --ablation A7
    python A7_A8_subject_specific.py --ablation A8
    python A7_A8_subject_specific.py --ablation both
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
    save_results, count_parameters, RUN_DIR,
    replace_head_for_eval,
)
from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
from pretraining.pretrain_trainer import pretrain
from pretraining.pretrain_finetune import finetune_and_eval_user
from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate
from MAML.mamlpp import mamlpp_adapt_and_eval
from torch.utils.data import DataLoader

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Trial indices ──────────────────────────────────────────────────────────────
# We have 1 sample/class. That is rep 1. That is all we get.
# Reps 2–10 are the query pool — never used for any form of learning.
TRAIN_TRIAL_INDICES    = [1]
HELD_OUT_TRIAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

from ablation_config import TRAIN_PIDS, VAL_PIDS, TEST_PIDS
ALL_SUBJECT_PIDS = TRAIN_PIDS + VAL_PIDS + TEST_PIDS


# =============================================================================
# A7: Subject-Specific CNN-LSTM
# =============================================================================

def build_config_a7() -> dict:
    config = make_base_config(ablation_id="A7")
    config["subject_specific_model"] = True
    config["meta_learning"] = False
    config["use_MOE"]       = False
    config["batch_size"]    = 10   # exactly the dataset size (1 rep × 10 classes)

    # Head-only fine-tuning at eval. The backbone learned subject-specific
    # features from the 10-sample pretraining. We only fit the fresh 3-class
    # head on the 3 support samples — touching the backbone would overwrite
    # those features with 3 samples, which is strictly worse.
    config["ft_steps"]        = 50
    config["ft_lr"]           = 1e-3
    config["ft_optimizer"]    = "adam"
    config["ft_weight_decay"] = config["weight_decay"]

    return config


def run_a7_one_subject(pid: str, seed: int, config: dict,
                       tensor_dict: dict) -> dict:
    """
    Train and evaluate A7 for one subject.

    Information budget: 1 sample/class. Same as M0.

    Step 1 — Pretrain backbone:
        Flat supervised training on rep 1 only (10 samples total, 10-class head).

    Step 2 — Replace head:
        Swap the 10-class pretrain head for a fresh 3-class head.

    Step 3 — Episodic eval:
        For each episode, MetaGestureDataset samples a 3-way task.
        Support = 1 sample/class drawn from TRAIN_TRIAL_INDICES (rep 1) — the
        same rep the backbone trained on. This is intentional: the support set
        IS the 1 sample/class we have. Query = 9 samples/class from
        HELD_OUT_TRIAL_INDICES (reps 2–10, never seen during learning).
        Fine-tune head-only on the 3 support samples, then eval on query.
    """
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    config["train_PIDs"] = [pid]
    config["val_PIDs"]   = [pid]
    config["train_reps"] = TRAIN_TRIAL_INDICES    # [1] — 1 sample/class, all we have
    config["val_reps"]   = TRAIN_TRIAL_INDICES    # same data; val loss is a training monitor only

    model = build_supervised_no_moe_model(config)

    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    if len(train_dl.dataset) == 0:
        print(f"[A7 | {pid} | seed={seed}] WARNING: empty train set — skipping.")
        return {"pid": pid, "seed": seed, "skipped": True}

    trained_model, history = pretrain(model, train_dl, val_dl, config)
    trained_model.load_state_dict(history["best_state"])

    # Replace 10-class pretrain head with a fresh 3-class head.
    trained_model = replace_head_for_eval(trained_model, config)

    # Episodic eval.
    # Pass all 10 reps to MetaGestureDataset so it can construct valid
    # support (k_shot=1 from rep 1) + query (q_query=9 from reps 2–10) splits.
    # The sampler handles the split internally; no learning occurs on query.
    test_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = [pid],
        target_gesture_classes = config["maml_gesture_classes"],
        target_trial_indices   = TRAIN_TRIAL_INDICES + HELD_OUT_TRIAL_INDICES,
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],    # 1
        q_query                = config["q_query"],   # 9
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
            support_emg    = support["emg"],
            support_imu    = support.get("imu"),
            support_labels = support["labels"],
            query_emg      = query["emg"],
            query_imu      = query.get("imu"),
            query_labels   = query["labels"],
            mode           = "head_only",
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
    return config


def run_a8_one_subject(pid: str, seed: int, config: dict,
                       tensor_dict_path: str) -> dict:
    """
    Evaluate A8 for one subject.

    Information budget: 1 sample/class. Same as M0.

    No pretraining. Random init. MAML inner-loop adapt on the 1-shot support
    set (3 samples), eval on query (27 samples). No gradients on query.

    The gap M0 vs A8 isolates exactly the value of cross-subject pretraining.
    """
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = build_maml_moe_model(config)

    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = full_dict["data"]

    test_ds = MetaGestureDataset(
        tensor_dict,
        target_pids            = [pid],
        target_gesture_classes = config["maml_gesture_classes"],
        target_trial_indices   = TRAIN_TRIAL_INDICES + HELD_OUT_TRIAL_INDICES,
        n_way                  = config["n_way"],
        k_shot                 = config["k_shot"],    # 1
        q_query                = config["q_query"],   # 9
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
            model, config, batch["support"], batch["query"]
        )
        episode_accs.append(metrics["acc"])

    mean_acc = float(np.mean(episode_accs)) if episode_accs else float("nan")
    print(f"[A8 | {pid} | seed={seed}] Acc: {mean_acc*100:.2f}%  ({len(episode_accs)} eps)")

    return {
        "pid":        pid,
        "seed":       seed,
        "mean_acc":   mean_acc,
        "n_episodes": len(episode_accs),
        "n_params":   count_parameters(model),
    }


# =============================================================================
# Run logic
# =============================================================================

def run_subject_specific_ablation(ablation_id: str, subject_runner, config: dict,
                                   description: str, **runner_kwargs):
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

    pid_to_accs = defaultdict(list)
    for r in all_results:
        if not r.get("skipped"):
            pid_to_accs[r["pid"]].append(r["mean_acc"])

    per_subject_mean = {pid: float(np.mean(accs)) for pid, accs in pid_to_accs.items() if accs}
    subject_means    = list(per_subject_mean.values())

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
            full_dict = pickle.load(f)
        tensor_dict = full_dict["data"]

        run_subject_specific_ablation(
            "A7", run_a7_one_subject, config_a7,
            description="Subject-Specific CNN-LSTM (Fair 1-shot Baseline)",
            tensor_dict=tensor_dict,
        )

    if args.ablation in ("A8", "both"):
        config_a8 = build_config_a8()
        tensor_dict_path = os.path.join(config_a8["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

        run_subject_specific_ablation(
            "A8", run_a8_one_subject, config_a8,
            description="Subject-Specific MAML + MoE (Fair 1-shot Baseline)",
            tensor_dict=tensor_dict_path,
        )


if __name__ == "__main__":
    main()