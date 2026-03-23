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
      'emg'  : Tensor (N_trials, T, C_emg) on disk → normalized to (N_trials, C_emg, T)
      'imu'  : Tensor (N_trials, T, C_imu) on disk → normalized to (N_trials, C_imu, T) | None
      'demo' : Tensor (demo_dim,)
  }

  normalize_tensor_dict() is called once in get_contrastive_dataloaders(), immediately
  after pickle.load(). All downstream code can assume (N_trials, C, T) channel-first order.
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


KNOWN_CHANNEL_COUNTS = {16, 72}


def normalize_tensor_dict(tensor_dict: dict) -> None:
    """
    Normalizes all EMG/IMU tensors to (N_trials, C, T) in-place,
    immediately after loading from disk.

    This is the single, authoritative place where channel ordering is fixed.
    No downstream code needs to think about axis ordering again.

    Raw shapes expected from disk:
        emg : (N_trials, T, C_emg)  →  (N_trials, C_emg, T)
        imu : (N_trials, T, C_imu)  →  (N_trials, C_imu, T)  [if present]

    Raises ValueError immediately if a shape is unrecognizable, so data
    issues surface at load time rather than as cryptic collate errors.
    """
    def _to_channel_first(tensor: torch.Tensor, key: str, pid, gid) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError(
                f"tensor_dict[{pid}][{gid}]['{key}']: expected 3-D (N, T, C) or "
                f"(N, C, T), got shape {tuple(tensor.shape)}."
            )
        N, d1, d2 = tensor.shape
        if d2 in KNOWN_CHANNEL_COUNTS:
            return tensor.transpose(1, 2).contiguous()   # (N, T, C) → (N, C, T)
        elif d1 in KNOWN_CHANNEL_COUNTS:
            return tensor.contiguous()                    # already (N, C, T)
        else:
            raise ValueError(
                f"tensor_dict[{pid}][{gid}]['{key}'] shape {tuple(tensor.shape)}: "
                f"neither dim[1]={d1} nor dim[2]={d2} is in "
                f"KNOWN_CHANNEL_COUNTS={KNOWN_CHANNEL_COUNTS}. "
                "Update KNOWN_CHANNEL_COUNTS or check your data pipeline."
            )

    for pid, gestures in tensor_dict.items():
        for gid, entry in gestures.items():
            entry['emg'] = _to_channel_first(entry['emg'], 'emg', pid, gid)
            if entry.get('imu') is not None:
                entry['imu'] = _to_channel_first(entry['imu'], 'imu', pid, gid)


# ============================================================
# 1. FLAT TRAINING DATASET
# ============================================================

class FlatGestureDataset(Dataset):
    """
    Returns individual gesture windows as flat samples.
    Each item: {'emg', 'imu', 'demo', 'label', 'user_id'}
    """

    def __init__(self, tensor_dict: dict, target_pids: list,
                 target_gestures: list, target_reps: list = None, use_imu: bool = False):
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
                
                # Filter by 1-indexed target reps
                if target_reps is not None:
                    valid_trials = [r - 1 for r in target_reps if (r - 1) < n_trials]
                else:
                    valid_trials = list(range(n_trials))

                for i in valid_trials:
                    self.samples.append((pid, gid, i))

        # Build class index: label → list of sample indices (into self.samples)
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

        # EMG already (C, T) — normalized to channel-first at load time
        emg = entry['emg'][trial_i]

        imu = None
        if self.use_imu and entry.get('imu') is not None:
            imu = entry['imu'][trial_i]

        demo  = entry['demo'].float()
        label = gid  # gesture class (int)

        return {
            'emg':     emg.float(),
            'imu':     imu.float() if imu is not None else None,
            'demo':    demo,
            'label':   label,
            'user_id': pid,
        }


# ============================================================
# 2. BALANCED BATCH SAMPLER
# ============================================================

class BalancedGestureSampler(Sampler):
    """
    Yields batch indices ensuring guaranteed positives per batch.
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

            random.shuffle(batch_indices)
            yield batch_indices


# ============================================================
# 3. COLLATE FOR FLAT BATCHES
# ============================================================

def flat_collate(batch):
    emg    = torch.stack([s['emg'] for s in batch], dim=0)
    labels = torch.tensor([s['label'] for s in batch], dtype=torch.long)
    demo   = torch.stack([s['demo'] for s in batch], dim=0)
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
    """

    def __init__(self, tensor_dict: dict, target_pids: list,
                 target_gestures: list, target_reps: list = None, n_way: int = 10,
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
                    
                    if target_reps is not None:
                        idxs = [r - 1 for r in target_reps if (r - 1) < n_tri]
                    else:
                        idxs = list(range(n_tri))
                        
                    rng.shuffle(idxs)

                    sup_idx = idxs[:k_shot]
                    qry_idx = idxs[k_shot: k_shot + q_query]

                    def make_sample(i):
                        # EMG/IMU already (C, T) — normalized at load time
                        emg = entry['emg'][i]
                        imu = None
                        if use_imu and entry.get('imu') is not None:
                            imu = entry['imu'][i]
                        return {
                            'emg':   emg.float(),
                            'imu':   imu.float() if imu is not None else None,
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

def get_contrastive_dataloaders(config: dict, tensor_dict_path: str,
                                return_val_flat: bool = False):
    """
    Returns dataloaders for contrastive training.

    Args:
        config              : experiment config dict
        tensor_dict_path    : path to the pickled tensor dict
        return_val_flat     : if False (default), returns (train_dl, val_episodic_dl)
                              if True, returns  (train_dl, val_episodic_dl, val_flat_dl)
                              val_flat_dl is a flat balanced loader over val_PIDs using
                              the same batch construction as train_dl. It is used to
                              compute val SupCon loss (same objective as training,
                              no backprop) which is the direct overfitting signal.
    """
    with open(tensor_dict_path, 'rb') as f:
        full_dict = pickle.load(f)
        tensor_dict = full_dict['data']

    # Normalize ALL tensors to (N_trials, C, T) once, right here.
    # Every Dataset and collate_fn below can assume clean channel-first data.
    normalize_tensor_dict(tensor_dict)

    use_imu          = config.get('use_imu', False)
    num_workers      = config.get('num_workers', 4)
    batch_mode       = config.get('batch_construction', 'balanced')
    spc              = config.get('samples_per_class', 6)
    cpb              = config.get('classes_per_batch', 10)
    gestures         = config.get('gesture_labels', list(range(10)))
    steps_per_epoch  = config.get('steps_per_epoch_train', 500)

    train_reps       = config.get('train_reps', None)
    val_reps         = config.get('val_reps', None)

    # ---- Train flat loader ----
    train_ds = FlatGestureDataset(
        tensor_dict,
        target_pids=config['train_PIDs'],
        target_gestures=gestures,
        target_reps=train_reps,
        use_imu=use_imu,
    )

    if batch_mode == 'balanced':
        sampler  = BalancedGestureSampler(train_ds, spc, cpb, steps_per_epoch)
        train_dl = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=flat_collate,
            worker_init_fn=worker_init_fn,
        )
    else:
        eff_bs   = spc * cpb
        train_dl = DataLoader(
            train_ds,
            batch_size=eff_bs,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=flat_collate,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    # ---- Val episodic loader (1-NN prototyping accuracy) ----
    val_ds = EpisodicValDataset(
        tensor_dict,
        target_pids=config['val_PIDs'],
        target_gestures=gestures,
        target_reps=val_reps,
        n_way=config.get('num_classes', 10),
        k_shot=config.get('val_support_shots', 1),
        q_query=config.get('val_query_per_class', 9),
        num_episodes_per_user=config.get('num_val_episodes', 20),
        seed=config.get('seed', 42),
        use_imu=use_imu,
    )

    val_episodic_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=episodic_collate,
    )

    if not return_val_flat:
        return train_dl, val_episodic_dl

    # ---- Val flat loader (val SupCon loss — same format as train_dl) ----
    # Built from val_PIDs + val_reps with the same balanced batch construction
    # as the training loader. steps_per_epoch is scaled down proportionally
    # to val set size to avoid spending too long on val loss computation.
    val_flat_ds = FlatGestureDataset(
        tensor_dict,
        target_pids=config['val_PIDs'],
        target_gestures=gestures,
        target_reps=val_reps,
        use_imu=use_imu,
    )

    n_train_users = max(len(config['train_PIDs']), 1)
    n_val_users   = max(len(config['val_PIDs']), 1)
    val_steps     = max(1, round(steps_per_epoch * n_val_users / n_train_users))

    if batch_mode == 'balanced':
        val_flat_sampler = BalancedGestureSampler(val_flat_ds, spc, cpb, val_steps)
        val_flat_dl = DataLoader(
            val_flat_ds,
            batch_sampler=val_flat_sampler,
            num_workers=num_workers,
            collate_fn=flat_collate,
            worker_init_fn=worker_init_fn,
        )
    else:
        eff_bs      = spc * cpb
        val_flat_dl = DataLoader(
            val_flat_ds,
            batch_size=eff_bs,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=flat_collate,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    return train_dl, val_episodic_dl, val_flat_dl