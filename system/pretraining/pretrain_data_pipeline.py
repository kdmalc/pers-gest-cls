"""
pretrain_data_pipeline.py
=========================
Standard supervised pretraining dataloader for EMG gesture classification.

Structure:
  - Reads the same tensor_dict pickle used by the MAML pipeline
  - Returns (emg, imu, labels) batches for standard cross-entropy training
  - Strict train/val/test user split to prevent data leakage into MAML evaluation

Expected tensor_dict layout (same as MetaGestureDataset):
  tensor_dict[pid][gesture_id]['emg']   → Tensor (n_trials, T, C_emg) or (n_trials, C_emg, T)
  tensor_dict[pid][gesture_id]['imu']   → Tensor or None
  tensor_dict[pid][gesture_id]['demo']  → Tensor (demographic vector)

All samples are flattened into a single flat dataset across users and gestures.
Labels are GLOBAL gesture IDs (not episode-relative), which is correct for pretraining.

Data augmentation (light, appropriate for small EMG datasets):
  - Gaussian noise injection
  - Random temporal shift (circular)
  - Channel dropout (randomly zeros out 1-2 EMG channels)
These are disabled at val/test time.
"""

import pickle
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
    """
    Ensures the tensor is in (N, C, T) format by checking for 
    known channel counts (EMG=16, IMU=72).
    """
    if x.dim() != 3:
        return x
    
    # If the LAST dimension matches 16 or 72, it is currently (N, T, C)
    # and needs to be permuted to (N, C, T).
    if x.shape[2] in [16, 72]:
        # Only permute if the middle dimension isn't already a channel count.
        # This prevents double-flipping if time and channel count are the same.
        #if x.shape[1] not in [16, 72]:
        return x.permute(0, 2, 1).contiguous()
            
    return x

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add zero-mean Gaussian noise. std is relative to signal std."""
    noise = torch.randn_like(x) * (x.std() * std + 1e-8)
    return x + noise

def _temporal_shift(x: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    """
    Circular shift along the time dimension.
    x: (C, T)
    """
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=shift, dims=-1)

def _channel_dropout(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """
    Zero out entire channels independently with probability drop_prob.
    x: (C, T)
    """
    mask = (torch.rand(x.size(0), 1) > drop_prob).float().to(x.device)
    return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PretrainGestureDataset(Dataset):
    def __init__(
        self,
        tensor_dict: dict,
        target_pids: list,
        target_reps: list,  # Rename parameter
        use_imu: bool = False,
        augment: bool = False,
        noise_std: float = 0.05,
        max_shift: int = 4,
        ch_drop_prob: float = 0.10,
        n_classes: int = 10
    ):
        self.use_imu = use_imu
        self.augment = augment
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.ch_drop = ch_drop_prob
        self.n_classes = n_classes

        # 1. Dynamically find all unique gestures to build the proper label mapping
        all_gestures = set()
        for pid in target_pids:
            if pid in tensor_dict:
                all_gestures.update(tensor_dict[pid].keys())
        sorted_gestures = sorted(list(all_gestures))
        
        # Now label_map properly maps the actual classes 0 through 9
        self.label_map = {g: i for i, g in enumerate(sorted_gestures)}
        
        self.samples = []
        for pid in target_pids:
            if pid not in tensor_dict: continue
            for gest, slot in tensor_dict[pid].items():
                emg_data = ensure_channel_first(slot['emg'])
                imu_data = ensure_channel_first(slot['imu']) if slot.get('imu') is not None else None
                label = self.label_map[gest]
                
                # 2. Slice based on repetitions (Assumes config uses 1-indexed rep counts)
                for rep in target_reps:
                    idx = rep - 1  # Convert to 0-indexed array position
                    if idx < 0 or idx >= emg_data.shape[0]:
                        continue  # Skip if this trial doesn't exist for the user
                    
                    self.samples.append((emg_data[idx], imu_data[idx] if imu_data is not None else None, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emg, imu, label = self.samples[idx]

        # Clone to avoid in-place modification of cached tensors
        emg = emg.clone().float()

        if self.augment:
            emg = _gaussian_noise(emg, self.noise_std)
            emg = _temporal_shift(emg, self.max_shift)
            emg = _channel_dropout(emg, self.ch_drop)

        if imu is not None:
            imu = imu.clone().float()
            if self.augment:
                imu = _gaussian_noise(imu, self.noise_std)

        return {
            "emg":   emg,
            "imu":   imu,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_collate(batch):
    emgs   = torch.stack([b["emg"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    imus   = None
    if batch[0]["imu"] is not None:
        imus = torch.stack([b["imu"] for b in batch])
    return {"emg": emgs, "imu": imus, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def get_pretrain_dataloaders(config: dict, tensor_dict_path: str):
    """
    Returns (train_dl, val_dl, n_classes).

    config keys used:
      train_PIDs, val_PIDs
      train_gesture_range, valtest_gesture_range
      use_imu, batch_size, num_workers
      augment (bool, default True for train)
      noise_std, max_shift, ch_drop_prob
    """
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)

    # TODO: Is it applying all these data augs? Is this creating MORE samples or is it augmenting the data I passed in? Presumably in place?... Does it randomly choose one/some?
    train_ds = PretrainGestureDataset(
        tensor_dict,
        target_pids     = config["train_PIDs"],
        target_reps     = config["train_reps"],
        use_imu         = config.get("use_imu", False),
        augment         = config.get("augment", True),
        noise_std       = config.get("noise_std", 0.05),
        max_shift       = config.get("max_shift", 4),
        ch_drop_prob    = config.get("ch_drop_prob", 0.10),
    )

    val_ds = PretrainGestureDataset(
        tensor_dict,
        target_pids     = config["val_PIDs"],
        target_reps     = config["val_reps"],
        use_imu         = config.get("use_imu", False),
        augment         = False,   # NEVER augment val
    )

    nw = int(config.get("num_workers", 4))
    bs = int(config.get("batch_size", 64))

    train_dl = DataLoader(
        train_ds,
        batch_size  = bs,
        shuffle     = True,
        num_workers = nw,
        collate_fn  = pretrain_collate,
        pin_memory  = True,
        drop_last   = True,   # avoids BN issues with tiny last batch
    )

    val_dl = DataLoader(
        val_ds,
        batch_size  = bs * 2,
        shuffle     = False,
        num_workers = nw,
        collate_fn  = pretrain_collate,
        pin_memory  = True,
    )

    return train_dl, val_dl, train_ds.n_classes
