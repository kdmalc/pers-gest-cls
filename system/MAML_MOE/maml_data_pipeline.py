"""
maml_data_pipeline.py
=====================
Meta-learning (MAML) episodic dataloader for EMG gesture classification.

────────────────────────────────────────────────────────────────────────────────
Tensor-dict layout (new, as of bug-fix — matches pretrain_data_pipeline.py):

  tensor_dict[pid][gesture_class] → dict with fields:
    'emg'         : Tensor (num_trials, seq_len, num_channels)   shape e.g. (10, 64, 16)
    'imu'         : Tensor (num_trials, seq_len, imu_channels)   or None
    'demo'        : Tensor (demographic feature vector)
    'gest_ID'     : int — same as the outer key; 0-indexed gesture class label
    'rep_indices' : list — 1-indexed trial/repetition numbers present in this entry

  Key types:
    pid           : str,  e.g. "P102"
    gesture_class : int,  0-indexed gesture class label (what the model predicts),
                    e.g. 0 … 9.  *** NOT a repetition number ***

────────────────────────────────────────────────────────────────────────────────
Terminology (please keep distinct throughout this file):
  gesture_class / class_label : int 0 … (n_classes-1)  ← identity of the gesture
  trial_idx                   : int 0 … (num_trials-1)  ← which recording of that gesture
                                 (0-indexed position inside the (num_trials, T, C) tensor)
  rep_num / rep_index         : int 1 … 10  (1-indexed, stored in 'rep_indices');
                                 NOT used directly in MAML — we just iterate trial_idx.

────────────────────────────────────────────────────────────────────────────────
MAML train/test split strategy:
  - Split is BY USER (participant): train_PIDs vs val_PIDs.
  - ALL gesture classes for a withheld user are withheld (no intra-class split).
  - Intra-subject rep splits (e.g. hold out reps 9 & 10) are supported via
    `target_trial_indices` for legacy / ablation purposes, but are NOT the
    default MAML strategy.  Pass target_trial_indices=None to use all trials.

────────────────────────────────────────────────────────────────────────────────
Config keys consumed here:
  train_PIDs              : list[str]  — participant IDs for meta-train
  val_PIDs                : list[str]  — participant IDs for meta-val
  maml_gesture_classes    : list[int]  — 0-indexed class labels to sample from
                             (replaces old 'maml_reps'; the name 'reps' was a
                              misnomer — these are CLASS labels, not rep numbers)
  maml_reps               : list[int]  — LEGACY alias for maml_gesture_classes;
                             honoured if maml_gesture_classes is absent
  n_way                   : int  — number of classes per episode
  k_shot                  : int  — support shots per class
  q_query                 : int or None  — query shots per class (None = all remaining)
  episodes_per_epoch_train: int
  num_eval_episodes       : int  — episodes per val user (pre-computed)
  seed                    : int
  num_workers             : int
  use_label_shuf_meta_aug : bool — shuffle label assignment order (augmentation)
"""

import os
import torch
import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========================================
# 1. COLLATE FUNCTION
# ==========================================

def maml_mm_collate(batch):
    """
    batch is always a list of exactly 1 episode (DataLoader batch_size=1).

    Each sample stored in episode['support'] / episode['query'] has:
      'emg'  : Tensor (num_channels, seq_len)  i.e. (C, T)
      'imu'  : Tensor (imu_channels, seq_len)  or None
      'demo' : Tensor (demo_dim,)
      'label': int (local, 0-indexed within episode)

    Output tensors are (B, C, T) — channel-first, as expected by Conv1d / LSTM.
    """
    episode = batch[0]

    def stack_samples(sample_list):
        if not sample_list:
            return None

        # emg: list of (C, T)  →  stack → (B, C, T)
        emg = torch.stack([s["emg"] for s in sample_list], dim=0)

        imu = None
        if sample_list[0]["imu"] is not None:
            imu = torch.stack([s["imu"] for s in sample_list], dim=0)

        demo   = torch.stack([s["demo"]  for s in sample_list], dim=0).float()
        labels = torch.as_tensor([s["label"] for s in sample_list], dtype=torch.long)

        return {"emg": emg, "imu": imu, "demo": demo, "labels": labels}

    return {
        "support":   stack_samples(episode["support"]),
        "query":     stack_samples(episode["query"]),
        "user_id":   episode.get("user_id"),
        "label_map": episode.get("label_map"),
    }


# ==========================================
# 2. THE META-LEARNING DATASET
# ==========================================

class MetaGestureDataset(Dataset):
    """
    Episodic dataset for N-way K-shot meta-learning over EMG gestures.

    Each episode:
      - Samples one participant at random from `target_pids`.
      - Samples `n_way` gesture classes from that participant's available classes.
      - For each class, randomly splits available trials into support (k_shot)
        and query (q_query, or all remaining if q_query is None).

    Args:
        tensor_dict          : the loaded data dict (already extracted from
                               full_dict['data']).
        target_pids          : participant IDs to draw episodes from.
        target_gesture_classes : list of 0-indexed gesture class labels to sample.
                               Only classes present for a given user are actually used.
        target_trial_indices : optional list of 0-indexed trial positions to restrict
                               to (legacy intra-subject rep split).  Pass None to use
                               ALL available trials (default MAML behaviour).
        n_way, k_shot, q_query : standard few-shot episode dimensions.
        episodes_per_epoch   : number of episodes constituting one training epoch.
        is_train             : if True → episodes generated on-the-fly each __getitem__;
                               if False → episodes pre-computed once and cached.
        seed                 : RNG seed for reproducible val cache and debug episodes.
        num_eval_episodes    : episodes pre-computed per val user.
        debug_one_episode    : repeat a single fixed episode (overfit sanity check).
        debug_five_episodes  : repeat five fixed episodes.
        debug_one_user_only  : when debug is active, use the same user for all episodes.
        use_label_shuf_meta_aug : randomise label→class assignment (data augmentation).
    """

    def __init__(
        self,
        tensor_dict,
        target_pids,
        target_gesture_classes,          # 0-indexed class labels (NOT rep numbers)
        target_trial_indices=None,       # 0-indexed; None → use all trials
        n_way=10,
        k_shot=1,
        q_query=9,
        episodes_per_epoch=1000,
        is_train=True,
        seed=42,
        num_eval_episodes=10,
        debug_one_episode=False,
        debug_five_episodes=False,
        debug_one_user_only=False,
        use_label_shuf_meta_aug=True,
    ):
        self.data = {pid: tensor_dict[pid] for pid in target_pids if pid in tensor_dict}
        self.pids = list(self.data.keys())

        # Gesture classes to sample from (0-indexed class labels).
        self.target_gesture_classes = list(target_gesture_classes)

        # Optional trial-index restriction (legacy intra-subject split).
        # None means "use all available trials for this class".
        self.target_trial_indices = target_trial_indices  # list[int] | None

        self.n_way               = n_way
        self.k_shot              = k_shot
        self.q_query             = q_query
        self.episodes_per_epoch  = episodes_per_epoch
        self.is_train            = is_train
        self.num_eval_episodes   = num_eval_episodes
        self.debug_one_episode   = debug_one_episode
        self.debug_five_episodes = debug_five_episodes
        self.debug_one_user_only = debug_one_user_only
        self.use_label_shuf_meta_aug = use_label_shuf_meta_aug

        # ── Debug episode cache ──────────────────────────────────────────────
        self.debug_episodes = []
        if self.debug_one_episode or self.debug_five_episodes:
            mode_name = "ONE task" if self.debug_one_episode else "FIVE tasks"
            user_mode = "Single User" if self.debug_one_user_only else "Unique Users"
            print(
                f"!!! DEBUG MODE: {mode_name} | {user_mode} | (n_way={n_way}) !!!",
                flush=True,
            )

            debug_rng     = random.Random(seed)
            num_to_create = 1 if self.debug_one_episode else 5

            if self.debug_one_user_only:
                selected_user  = debug_rng.choice(self.pids)
                target_user_ids = [selected_user] * num_to_create
            else:
                target_user_ids = debug_rng.sample(
                    self.pids, min(num_to_create, len(self.pids))
                )

            for i, user_id in enumerate(target_user_ids):
                available = self._available_classes_for_user(user_id)
                classes   = debug_rng.sample(available, min(self.n_way, len(available)))
                if not self.use_label_shuf_meta_aug:
                    classes = sorted(classes)

                label_map = {c: idx for idx, c in enumerate(classes)}
                ep        = self._build_episode(
                    user_id, classes, label_map, debug_rng, is_train=True
                )
                self.debug_episodes.append(ep)

                fingerprint = ep["support"][0]["emg"].abs().sum().item()
                print(
                    f"  > Episode {i} Fixed (User {user_id}) | Map: {label_map} "
                    f"| Fingerprint: {fingerprint:.4f}",
                    flush=True,
                )

        # ── Normal val cache ─────────────────────────────────────────────────
        self.val_episodes_cache = []
        if not self.is_train and not self.debug_one_episode and not self.debug_five_episodes:
            self._precompute_val_episodes(seed)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _available_classes_for_user(self, user_id: str) -> list:
        """
        Return the intersection of target_gesture_classes and the class labels
        actually present in tensor_dict for this user.

        tensor_dict[user_id] keys are 0-indexed gesture class labels (int).
        """
        return [
            cls for cls in self.target_gesture_classes
            if cls in self.data[user_id]
        ]

    def _available_trial_indices_for_class(self, user_id: str, class_label: int) -> list:
        """
        Return the trial (0-indexed) positions available for this user × class,
        optionally restricted to self.target_trial_indices.

        The tensor has shape (num_trials, T, C); valid indices are 0 … num_trials-1.
        If target_trial_indices is None, all trials are used.
        """
        emg_tensor  = self.data[user_id][class_label]["emg"]  # (num_trials, T, C)
        num_trials  = emg_tensor.shape[0]
        all_indices = list(range(num_trials))

        if self.target_trial_indices is None:
            return all_indices

        # Keep only indices that fall within the allowed set AND within bounds.
        return [
            idx for idx in self.target_trial_indices
            if 0 <= idx < num_trials
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Episode construction
    # ──────────────────────────────────────────────────────────────────────────

    def _precompute_val_episodes(self, seed: int):
        """Pre-generate validation episodes so they are identical every epoch."""
        val_rng = random.Random(seed)
        for user_id in self.pids:
            available = self._available_classes_for_user(user_id)
            for _ in range(self.num_eval_episodes):
                classes = val_rng.sample(available, min(self.n_way, len(available)))
                if not self.use_label_shuf_meta_aug:
                    classes = sorted(classes)
                label_map = {c: i for i, c in enumerate(classes)}
                episode   = self._build_episode(
                    user_id, classes, label_map, val_rng, is_train=False
                )
                self.val_episodes_cache.append(episode)

    def _build_episode(
        self,
        user_id: str,
        classes: list,          # ordered list of class_labels for this episode
        label_map: dict,        # class_label → local episode label (0-indexed)
        rng_instance,
        is_train: bool,
    ) -> dict:
        """
        Build one N-way K-shot episode for the given user and class selection.

        Data access pattern (NEW dict layout):
          emg_all = tensor_dict[user_id][class_label]['emg']  # (num_trials, T, C)
          one trial: emg_all[trial_idx]                       # (T, C)

        The collate fn expects individual samples to be (T, C); it handles
        the permutation to (B, C, T).
        """
        support_samples, query_samples = [], []

        for class_label in classes:
            local_label = label_map[class_label]

            # ── Fetch tensors for this user × class ──────────────────────────
            slot        = self.data[user_id][class_label]
            emg_all     = slot["emg"]               # (num_trials, T, C)
            imu_all     = slot.get("imu", None)     # (num_trials, T, C_imu) or None
            demo        = slot["demo"]               # (demo_dim,)

            # ── Determine which trial indices to use ─────────────────────────
            trial_indices = self._available_trial_indices_for_class(user_id, class_label)

            if len(trial_indices) < self.k_shot + 1:
                # Not enough trials for at least 1 support + 1 query; skip class.
                # (Should not happen with a well-formed dataset and sane k_shot.)
                continue

            # Shuffle to randomise which trials become support vs query
            rng_instance.shuffle(trial_indices)

            sup_idx = trial_indices[: self.k_shot]
            if is_train and self.q_query is not None:
                qry_idx = trial_indices[self.k_shot : self.k_shot + self.q_query]
            else:
                # Eval or q_query=None: use all remaining trials for query
                qry_idx = trial_indices[self.k_shot :]

            for idx in sup_idx:
                support_samples.append({
                    "emg":          emg_all[idx],                              # (T, C)
                    "imu":          imu_all[idx] if imu_all is not None else None,
                    "demo":         demo,
                    "label":        local_label,
                    "global_class": class_label,
                })
            for idx in qry_idx:
                query_samples.append({
                    "emg":          emg_all[idx],                              # (T, C)
                    "imu":          imu_all[idx] if imu_all is not None else None,
                    "demo":         demo,
                    "label":        local_label,
                    "global_class": class_label,
                })

        # Shuffle so the model sees a random label order, not class-sequential
        rng_instance.shuffle(support_samples)
        rng_instance.shuffle(query_samples)

        return {
            "support":   support_samples,
            "query":     query_samples,
            "user_id":   user_id,
            "label_map": label_map,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self):
        if self.debug_one_episode or self.debug_five_episodes:
            return self.episodes_per_epoch if self.is_train else len(self.debug_episodes)
        return self.episodes_per_epoch if self.is_train else len(self.val_episodes_cache)

    def __getitem__(self, idx):
        # ── Debug mode ───────────────────────────────────────────────────────
        if self.debug_one_episode or self.debug_five_episodes:
            if self.is_train:
                return random.choice(self.debug_episodes)
            else:
                return self.debug_episodes[idx % len(self.debug_episodes)]

        # ── Normal val: return pre-computed deterministic episode ─────────────
        if not self.is_train:
            return self.val_episodes_cache[idx]

        # ── Normal train: generate episode on-the-fly ─────────────────────────
        user_id   = random.choice(self.pids)
        available = self._available_classes_for_user(user_id)
        classes   = random.sample(available, min(self.n_way, len(available)))
        if not self.use_label_shuf_meta_aug:
            classes = sorted(classes)

        label_map = {c: i for i, c in enumerate(classes)}
        return self._build_episode(user_id, classes, label_map, random, is_train=True)


# ==========================================
# 3. DATALOADER BUILDER
# ==========================================

def get_maml_dataloaders(config, tensor_dict_path):
    """
    Build episodic train and val DataLoaders for MAML.

    Config keys:
      maml_gesture_classes  : list[int]  0-indexed gesture class labels to sample.
                              LEGACY alias: 'maml_reps' (honoured if the above absent).
                              NOTE: despite the old name 'maml_reps', these are CLASS
                              LABELS (0-indexed), not repetition numbers.
      target_trial_indices  : list[int] | None  0-indexed trial positions to use.
                              None (default) → all available trials used.
    """
    with open(tensor_dict_path, "rb") as f:
        full_dict   = pickle.load(f)
    # New dict layout: top level has 'data' key plus metadata.
    tensor_dict = full_dict["data"]

    # ── Re-orient Data to Contiguous (B, C, T) once ──────────────────────────
    # Needs to be contiguous for the CNN
    print("[maml_data_pipeline] Re-orienting data tensors to (trials, channels, seq_len)...")
    for pid in tensor_dict:
        for gest_class in tensor_dict[pid]:
            # EMG: (trials, seq, chan) -> (trials, chan, seq) ----> (trials, 64, 16) -> (trials, 16, 64)
            emg = tensor_dict[pid][gest_class]['emg']
            if emg.shape[1] != config['emg_in_ch']: # Double check it's "sideways"
                tensor_dict[pid][gest_class]['emg'] = emg.permute(0, 2, 1).contiguous()
            
            # IMU: (trials, seq, chan) -> (trials, chan, seq) ----> (trials, 64, 72) -> (trials, 72, 64)
            imu = tensor_dict[pid][gest_class].get('imu')
            if imu is not None and imu.shape[1] != config['imu_in_ch']:
                 tensor_dict[pid][gest_class]['imu'] = imu.permute(0, 2, 1).contiguous()

    num_workers    = int(config.get("num_workers", 4))
    use_label_shuf = config.get("use_label_shuf_meta_aug", True)

    # ── Resolve gesture class list ────────────────────────────────────────────
    # 'maml_gesture_classes' is the canonical key.
    # 'maml_reps' is kept as a legacy alias — historically the name was misleading
    # (it stored class labels, not repetition numbers).
    if "maml_gesture_classes" in config:
        gesture_classes = config["maml_gesture_classes"]
    elif "maml_reps" in config:
        gesture_classes = config["maml_reps"]
        # Emit a one-time reminder to rename the config key.
        print(
            "[get_maml_dataloaders] WARNING: config key 'maml_reps' is a legacy alias "
            "for gesture CLASS labels (0-indexed). Please rename to 'maml_gesture_classes' "
            "to avoid confusion with repetition/trial numbers."
        )
    else:
        raise KeyError(
            "Config must contain 'maml_gesture_classes' (or legacy 'maml_reps') — "
            "a list of 0-indexed gesture class labels to sample episodes from."
        )

    # ── Optional trial-index restriction (legacy intra-subject rep split) ─────
    # Pass a list of 0-indexed trial positions to restrict which trials are used,
    # or leave as None to use all available trials (standard MAML by-user split).
    target_trial_indices = config.get("target_trial_indices", None)

    train_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = config["train_PIDs"],
        target_gesture_classes  = gesture_classes,
        target_trial_indices    = target_trial_indices,
        n_way                   = config["n_way"],
        k_shot                  = config["k_shot"],
        q_query                 = config["q_query"],
        episodes_per_epoch      = config["episodes_per_epoch_train"],
        is_train                = True,
        debug_one_episode       = config.get("debug_one_episode",   False),
        debug_five_episodes     = config.get("debug_five_episodes",  False),
        debug_one_user_only     = config.get("debug_one_user_only",  False),
        use_label_shuf_meta_aug = use_label_shuf,
    )

    # Val uses train_PIDs in debug mode (to guarantee the same fixed episodes exist)
    val_pids = (
        config["train_PIDs"]
        if (config.get("debug_one_episode") or config.get("debug_five_episodes"))
        else config["val_PIDs"]
    )

    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = val_pids,
        target_gesture_classes  = gesture_classes,
        target_trial_indices    = target_trial_indices,
        n_way                   = config["n_way"],
        k_shot                  = config["k_shot"],
        q_query                 = config.get("q_query", None),
        num_eval_episodes       = config.get("num_eval_episodes", 10),
        is_train                = False,
        seed                    = config.get("seed", 42),
        debug_one_episode       = config.get("debug_one_episode",   False),
        debug_five_episodes     = config.get("debug_five_episodes",  False),
        debug_one_user_only     = config.get("debug_one_user_only",  False),
        use_label_shuf_meta_aug = use_label_shuf,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size      = 1,
        shuffle         = True,
        num_workers     = num_workers,
        collate_fn      = maml_mm_collate,
        worker_init_fn  = worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = maml_mm_collate,
    )

    return train_dl, val_dl