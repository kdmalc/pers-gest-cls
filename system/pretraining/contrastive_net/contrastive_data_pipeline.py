"""
contrastive_data_pipeline.py

Flat-batch DataLoader for contrastive training.
No episodic structure — just large, balanced batches where each batch
contains M samples per gesture class, enabling rich positive/negative pairs.

Batch construction modes:
  'balanced' (recommended): Exactly `samples_per_class` samples from each of
                             `classes_per_batch` gesture classes. Every batch
                             has a guaranteed positive for every anchor.
  'random'                : Standard random sampling. Simpler but may produce
                             batches with no positives for rare classes.

Validation DataLoader:
  Uses pre-generated episodic structure (1-shot prototyping) to evaluate
  cross-user transfer accuracy — exactly matching the test-time protocol.

Data format (from existing tensor_dict):
  tensor_dict[pid][gesture_id] = {
      'emg'  : Tensor (N_trials, T, C_emg)   → we permute to (N, C, T)
      'imu'  : Tensor (N_trials, T, C_imu) | None
      'demo' : Tensor (demo_dim,)
  }
"""

import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# 1. FLAT TRAINING DATASET
# ============================================================

class FlatGestureDataset(Dataset):
    """
    Returns individual gesture windows as flat samples.
    Each item: {'emg', 'imu', 'demo', 'label', 'user_id'}

    This is intentionally dead simple — all the magic is in the Sampler.
    """

    def __init__(self, tensor_dict: dict, target_pids: list,
                 target_gestures: list, use_imu: bool = False):
        self.use_imu = use_imu
        self.samples = []  # List of (pid, gesture_id, trial_idx)
        self.data    = {}

        for pid in target_pids:
            if pid not in tensor_dict:
                continue
            self.data[pid] = {}
            for gid in target_gestures:
                if gid not in tensor_dict[pid]:
                    continue
                entry = tensor_dict[pid][gid]
                n_trials = entry['emg'].shape[0]
                self.data[pid][gid] = entry
                for i in range(n_trials):
                    self.samples.append((pid, gid, i))

        # Build class index: label → list of sample indices (into self.samples)
        # We use (pid, gesture_id) as the "full class" identity,
        # but gesture_id alone is the label for contrastive grouping.
        self.gesture_to_indices = {}
        for idx, (pid, gid, _) in enumerate(self.samples):
            self.gesture_to_indices.setdefault(gid, []).append(idx)

        self.unique_gestures = sorted(self.gesture_to_indices.keys())
        self.all_pids = target_pids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        pid, gid, trial_i = self.samples[idx]
        entry = self.data[pid][gid]

        # EMG: (T, C_emg) → (C_emg, T)
        emg = entry['emg'][trial_i].permute(1, 0) if entry['emg'][trial_i].dim() == 2 \
              else entry['emg'][trial_i].T

        imu = None
        if self.use_imu and entry.get('imu') is not None:
            raw_imu = entry['imu'][trial_i]
            imu = raw_imu.permute(1, 0) if raw_imu.dim() == 2 else raw_imu.T

        demo  = entry['demo'].float()
        label = gid  # gesture class (int)

        return {
            'emg':     emg.float(),
            'imu':     imu,
            'demo':    demo,
            'label':   label,
            'user_id': pid,
        }


# ============================================================
# 2. BALANCED BATCH SAMPLER
# ============================================================

class BalancedGestureSampler(Sampler):
    """
    Yields batch indices such that each batch contains exactly:
        samples_per_class × classes_per_batch  samples

    Within each batch, every included gesture class has exactly
    `samples_per_class` samples, drawn uniformly at random from all
    users. This guarantees at least (samples_per_class - 1) positives
    per anchor, enabling rich SupCon gradients.

    If a class has fewer than `samples_per_class` available trials,
    we sample with replacement for that class.
    """

    def __init__(self, dataset: FlatGestureDataset,
                 samples_per_class: int,
                 classes_per_batch: int,
                 steps_per_epoch: int = 500):
        self.dataset           = dataset
        self.samples_per_class = samples_per_class
        self.classes_per_batch = min(classes_per_batch, len(dataset.unique_gestures))
        self.steps_per_epoch   = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            # Pick which gesture classes appear in this batch
            chosen_gestures = random.sample(
                self.dataset.unique_gestures,
                self.classes_per_batch
            )

            batch_indices = []
            for gid in chosen_gestures:
                pool = self.dataset.gesture_to_indices[gid]
                if len(pool) >= self.samples_per_class:
                    chosen = random.sample(pool, self.samples_per_class)
                else:
                    chosen = random.choices(pool, k=self.samples_per_class)
                batch_indices.extend(chosen)

            # Shuffle within batch so model doesn't see sorted labels
            random.shuffle(batch_indices)
            yield batch_indices


# ============================================================
# 3. COLLATE FOR FLAT BATCHES
# ============================================================

def flat_collate(batch):
    """
    Collates a flat list of sample dicts into tensors.
    Handles None IMU by returning None for the whole batch.
    """
    emg    = torch.stack([s['emg'] for s in batch], dim=0)
    labels = torch.tensor([s['label'] for s in batch], dtype=torch.long)
    demo   = torch.stack([s['demo'] for s in batch], dim=0)
    # My user_ids are strings like "P100" so we cannot directly convert to torch.long (integers)
    #user_ids = torch.tensor([s['user_id'] for s in batch], dtype=torch.long)
    user_ids = [s['user_id'] for s in batch]

    imu = None
    if batch[0]['imu'] is not None:
        imu = torch.stack([s['imu'] for s in batch], dim=0)

    return {
        'emg':      emg,
        'imu':      imu,
        'demo':     demo,
        'labels':   labels,
        'user_ids': user_ids,
    }


# ============================================================
# 4. VALIDATION EPISODIC DATASET (1-shot prototyping)
# ============================================================

class EpisodicValDataset(Dataset):
    """
    Pre-generates N episodes per validation user.
    Each episode:
      - support: k_shot samples per class (used to build prototypes)
      - query:   remaining samples per class

    This exactly mirrors test-time inference so val accuracy is meaningful.
    """

    def __init__(self, tensor_dict: dict, target_pids: list,
                 target_gestures: list, n_way: int = 10,
                 k_shot: int = 1, q_query: int = 9,
                 num_episodes_per_user: int = 20,
                 seed: int = 42, use_imu: bool = False):
        self.use_imu    = use_imu
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.q_query    = q_query

        data = {pid: tensor_dict[pid] for pid in target_pids if pid in tensor_dict}
        rng  = random.Random(seed)
        self.episodes = []

        for pid in data:
            available = [g for g in target_gestures if g in data[pid]]
            if len(available) < n_way:
                continue

            for _ in range(num_episodes_per_user):
                classes   = rng.sample(available, n_way)
                label_map = {c: i for i, c in enumerate(classes)}

                support_samples, query_samples = [], []

                for gid in classes:
                    local_label = label_map[gid]
                    entry  = data[pid][gid]
                    n_tri  = entry['emg'].shape[0]
                    idxs   = list(range(n_tri))
                    rng.shuffle(idxs)

                    sup_idx = idxs[:k_shot]
                    qry_idx = idxs[k_shot: k_shot + q_query]

                    def make_sample(i):
                        emg = entry['emg'][i]
                        emg = emg.permute(1, 0) if emg.dim() == 2 else emg.T
                        imu = None
                        if use_imu and entry.get('imu') is not None:
                            raw = entry['imu'][i]
                            imu = raw.permute(1, 0) if raw.dim() == 2 else raw.T
                        return {
                            'emg':   emg.float(),
                            'imu':   imu,
                            'demo':  entry['demo'].float(),
                            'label': local_label,
                        }

                    support_samples.extend([make_sample(i) for i in sup_idx])
                    query_samples.extend([make_sample(i) for i in qry_idx])

                rng.shuffle(support_samples)
                rng.shuffle(query_samples)
                self.episodes.append({
                    'support':   support_samples,
                    'query':     query_samples,
                    'user_id':   pid,
                    'label_map': label_map,
                    'n_way':     n_way,
                })

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def episodic_collate(batch):
    """Collates a list of episode dicts (batch_size=1)."""
    ep = batch[0]

    def stack(samples):
        emg    = torch.stack([s['emg'] for s in samples])
        labels = torch.tensor([s['label'] for s in samples], dtype=torch.long)
        demo   = torch.stack([s['demo'] for s in samples])
        imu = None
        if samples[0]['imu'] is not None:
            imu = torch.stack([s['imu'] for s in samples])
        return {'emg': emg, 'imu': imu, 'demo': demo, 'labels': labels}

    return {
        'support':   stack(ep['support']),
        'query':     stack(ep['query']),
        'user_id':   ep['user_id'],
        'label_map': ep['label_map'],
        'n_way':     ep['n_way'],
    }


# ============================================================
# 5. DATALOADER BUILDER
# ============================================================

def get_contrastive_dataloaders(config: dict, tensor_dict_path: str):
    """
    Returns (train_dl, val_dl) for contrastive training.

    train_dl: Flat balanced batches — no episodic structure.
    val_dl  : Pre-generated 1-shot prototyping episodes per user.
    """
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)

    use_imu          = config.get('use_imu', False)
    num_workers      = config.get('num_workers', 4)
    batch_mode       = config.get('batch_construction', 'balanced')
    spc              = config.get('samples_per_class', 6)
    cpb              = config.get('classes_per_batch', 10)
    gestures         = config.get('gesture_labels', list(range(1, 11)))
    steps_per_epoch  = config.get('steps_per_epoch_train', 500)

    # ---- Train ----
    train_ds = FlatGestureDataset(
        tensor_dict,
        target_pids=config['train_PIDs'],
        target_gestures=gestures,
        use_imu=use_imu,
    )

    if batch_mode == 'balanced':
        sampler   = BalancedGestureSampler(train_ds, spc, cpb, steps_per_epoch)
        train_dl  = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=flat_collate,
            worker_init_fn=worker_init_fn,
        )
    else:
        eff_bs = spc * cpb
        train_dl = DataLoader(
            train_ds,
            batch_size=eff_bs,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=flat_collate,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    # ---- Val ----
    val_ds = EpisodicValDataset(
        tensor_dict,
        target_pids=config['val_PIDs'],
        target_gestures=gestures,
        n_way=config.get('num_classes', 10),
        k_shot=config.get('val_support_shots', 1),
        q_query=config.get('val_query_per_class', 9),
        num_episodes_per_user=config.get('num_val_episodes', 20),
        seed=config.get('seed', 42),
        use_imu=use_imu,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=episodic_collate,
    )

    return train_dl, val_dl
