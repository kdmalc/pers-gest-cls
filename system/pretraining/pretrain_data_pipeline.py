"""
pretrain_data_pipeline.py
=========================
Standard supervised pretraining dataloader for EMG gesture classification.

────────────────────────────────────────────────────────────────────────────────
NEW tensor_dict layout (as of bug-fix):
  tensor_dict[pid][gesture_class]  →  dict with fields:
    'emg'          : Tensor (num_trials, seq_len, num_channels)  → (10, 64, 16)
    'imu'          : Tensor (num_trials, seq_len, imu_channels)  → (10, 64, 72) or None
    'demo'         : Tensor (demographic vector)
    'enc_gest_ID'  : encoded gesture ID (internal, not used here)
    'gest_ID'      : int — same as the outer key; 0-indexed gesture class label (0..9)
    'enc_pid'      : encoded participant ID (internal, not used here)
    'rep_indices'  : 1-indexed trial/repetition numbers present in this entry

  Key types:
    pid          : str,  e.g. "P102"
    gesture_class: int,  0-indexed gesture class label, i.e. 0 … (n_classes-1)
                   *** NOT a repetition number, NOT an encoded ID ***

────────────────────────────────────────────────────────────────────────────────
Terminology used throughout this file — please keep these DISTINCT:
  gesture_class / class_label : integer 0 … (n_classes-1)  ← what we predict
  trial_num / rep_num         : integer 1 … 10  (1-indexed) ← one recording of a gesture

  "train_reps" / "val_reps" in the config refer to TRIAL/REP NUMBERS (1-indexed),
  NOT to class labels.  They control which of the 10 repetitions are used for
  training vs. validation within each participant.

────────────────────────────────────────────────────────────────────────────────
Data augmentation (applied stochastically per __getitem__, not offline):
  - Gaussian noise injection
  - Random temporal shift (circular)
  - Channel dropout (randomly zeros out individual EMG channels)
  These are DISABLED at val/test time.  Augmentation does NOT increase the
  dataset size — it randomly perturbs each sample every time it is drawn,
  so the effective diversity grows across epochs.
────────────────────────────────────────────────────────────────────────────────
Config keys consumed here (aligned with BASE_CONFIG):
  train_PIDs                : list[str]  — participant IDs for training
  val_PIDs                  : list[str]  — participant IDs for validation
  train_reps                : list[int]  — 1-indexed trial numbers used for train
  val_reps                  : list[int]  — 1-indexed trial numbers used for val
  available_gesture_classes : list[int]  — 0-indexed class labels to include
  use_imu                   : bool
  batch_size                : int
  num_workers               : int
  augment                   : bool       — applied to train set only
  aug_noise_std             : float
  aug_max_shift             : int
  aug_ch_drop               : float
"""

import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Tensor layout helper
# ─────────────────────────────────────────────────────────────────────────────

def to_channel_first_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a single sample from (seq_len, num_channels) → (num_channels, seq_len).

    The new tensor_dict stores data as (num_trials, seq_len, num_channels).
    After slicing out one trial we get (seq_len, num_channels), i.e. (T, C).
    Models expect (C, T).  This function handles that final permute.

    Args:
        x: Tensor of shape (T, C)  — a single trial, channel-last

    Returns:
        Tensor of shape (C, T)  — channel-first, ready for Conv1d / LSTM
    """
    assert x.dim() == 2, f"Expected 2D tensor (T, C), got shape {x.shape}"
    return x.permute(1, 0).contiguous()  # (T, C) → (C, T)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers  (all operate on a single 2-D sample: (C, T))
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add zero-mean Gaussian noise scaled to the signal's own std."""
    noise = torch.randn_like(x) * (x.std() * std + 1e-8)
    return x + noise


def _temporal_shift(x: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    """
    Circular shift along the time axis (dim=-1).
    Shift amount is sampled uniformly from [-max_shift, +max_shift].

    Args:
        x: Tensor (C, T)
    """
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=shift, dims=-1)


def _channel_dropout(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """
    Zero out entire EMG channels independently with probability drop_prob.

    Args:
        x: Tensor (C, T)
    """
    mask = (torch.rand(x.size(0), 1) > drop_prob).float().to(x.device)
    return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PretrainGestureDataset(Dataset):
    """
    Flat dataset over participants × gesture classes × repetition trials.

    Each item is ONE trial (one repetition of one gesture by one participant).
    The label is the gesture class label (0-indexed integer, 0 … n_classes-1).

    Indexing into tensor_dict:
      tensor_dict[pid][gesture_class]['emg']  →  (num_trials, seq_len, num_channels)
      We select the trial corresponding to a specific 1-indexed rep_num via:
          trial_idx = rep_num - 1   (converts 1-indexed → 0-indexed array position)

    Args:
        tensor_dict              : loaded data dict (already extracted from full_dict['data'])
        target_pids              : participant IDs to include
        target_rep_nums          : 1-indexed trial/repetition numbers to include
                                   e.g. [1,2,...,8] for train, [9,10] for val
        available_gesture_classes: 0-indexed class labels to include (default: all present)
        use_imu                  : whether to load IMU data
        augment                  : whether to apply stochastic augmentation in __getitem__
        aug_noise_std            : Gaussian noise scale (relative to signal std)
        aug_max_shift            : max circular time shift in samples
        aug_ch_drop              : per-channel zero-out probability
    """

    def __init__(
        self,
        tensor_dict: dict,
        target_pids: list,
        target_rep_nums: list,           # 1-indexed trial numbers, e.g. [1..8] or [9,10]
        available_gesture_classes: list = None,  # 0-indexed class labels
        use_imu: bool = False,
        augment: bool = False,
        aug_noise_std: float = 0.05,
        aug_max_shift: int = 4,
        aug_ch_drop: float = 0.10,
    ):
        self.use_imu       = use_imu
        self.augment       = augment
        self.aug_noise_std = aug_noise_std
        self.aug_max_shift = aug_max_shift
        self.aug_ch_drop   = aug_ch_drop

        # ── Determine which gesture class labels to include ──────────────────
        # Keys in tensor_dict[pid] are 0-indexed integer class labels.
        # We allow the caller to restrict to a subset; default = all present.
        if available_gesture_classes is not None:
            gesture_classes = sorted(available_gesture_classes)
        else:
            # Collect all class labels present across target PIDs
            all_classes = set()
            for pid in target_pids:
                if pid in tensor_dict:
                    all_classes.update(tensor_dict[pid].keys())
            gesture_classes = sorted(all_classes)

        self.n_classes = len(gesture_classes)

        # ── Build flat sample list ───────────────────────────────────────────
        # Each entry: (emg_trial, imu_trial_or_None, class_label)
        #   emg_trial : Tensor (seq_len, num_channels)  — channel-last, 1 trial
        #   imu_trial : Tensor (seq_len, imu_channels) or None
        #   class_label: int  0 … (n_classes-1)
        #
        # We do NOT permute to (C, T) here; we do it in __getitem__ so that
        # cached tensors stay in the original layout and cloning is cheap.

        self.samples = []
        skipped_trials = 0

        for pid in target_pids:
            if pid not in tensor_dict:
                continue

            for class_label in gesture_classes:
                if class_label not in tensor_dict[pid]:
                    continue

                slot    = tensor_dict[pid][class_label]
                emg_all = slot['emg']   # (num_trials, seq_len, num_channels)
                imu_all = slot.get('imu', None)  # (num_trials, seq_len, imu_channels) or None

                num_trials_available = emg_all.shape[0]  # should be 10

                for rep_num in target_rep_nums:
                    # rep_num is 1-indexed (1 … 10); convert to 0-indexed array position
                    trial_idx = rep_num - 1

                    if trial_idx < 0 or trial_idx >= num_trials_available:
                        skipped_trials += 1
                        continue  # This rep_num doesn't exist for this pid/gesture

                    emg_trial = emg_all[trial_idx]   # (seq_len, num_channels) = (T, C)
                    imu_trial = imu_all[trial_idx] if imu_all is not None else None

                    self.samples.append((emg_trial, imu_trial, class_label))

        if skipped_trials > 0:
            print(f"[PretrainGestureDataset] Warning: skipped {skipped_trials} "
                  f"(pid, gesture_class, rep_num) combos where trial_idx was out of range.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emg_tc, imu_tc, class_label = self.samples[idx]

        # Clone before any in-place-style ops; convert (T, C) → (C, T)
        emg = to_channel_first_2d(emg_tc.clone().float())   # (C_emg, T)

        if self.augment:
            emg = _gaussian_noise(emg, self.aug_noise_std)
            emg = _temporal_shift(emg, self.aug_max_shift)
            emg = _channel_dropout(emg, self.aug_ch_drop)

        imu = None
        if self.use_imu and imu_tc is not None:
            imu = to_channel_first_2d(imu_tc.clone().float())  # (C_imu, T)
            if self.augment:
                imu = _gaussian_noise(imu, self.aug_noise_std)

        return {
            "emg":   emg,                                      # (C_emg, T)
            "imu":   imu,                                      # (C_imu, T) or None
            "label": torch.tensor(class_label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_collate(batch):
    emgs   = torch.stack([b["emg"]   for b in batch])   # (B, C_emg, T)
    labels = torch.stack([b["label"] for b in batch])   # (B,)
    imus   = None
    if batch[0]["imu"] is not None:
        imus = torch.stack([b["imu"] for b in batch])   # (B, C_imu, T)
    return {"emg": emgs, "imu": imus, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def get_pretrain_dataloaders(config: dict, tensor_dict: dict):
    """
    Build train and val DataLoaders for supervised pretraining.
    Caller is responsible for loading tensor_dict from disk (load once, reuse).

    Returns:
        train_dl : DataLoader
        val_dl   : DataLoader
        n_classes: int
    """
    gesture_classes = config["available_gesture_classes"]

    train_ds = PretrainGestureDataset(
        tensor_dict,
        target_pids               = config["train_PIDs"],
        target_rep_nums           = config["train_reps"],
        available_gesture_classes = gesture_classes,
        use_imu                   = config["use_imu"],
        augment                   = config.get("augment", False),
        aug_noise_std             = config.get("aug_noise_std", 0.05),
        aug_max_shift             = config.get("aug_max_shift", 4),
        aug_ch_drop               = config.get("aug_ch_drop", 0.10),
    )
    val_ds = PretrainGestureDataset(
        tensor_dict,
        target_pids               = config["val_PIDs"],
        target_rep_nums           = config["val_reps"],
        available_gesture_classes = gesture_classes,
        use_imu                   = config["use_imu"],
        augment                   = False,
    )

    nw = int(config["num_workers"])
    bs = int(config["batch_size"])

    # NOTE: Why is the val bs twice the train bs... bs for val shouldnt even matter right...
    train_dl = DataLoader(train_ds, batch_size=bs,   shuffle=True,  num_workers=nw,
                          collate_fn=pretrain_collate, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs*2, shuffle=False, num_workers=nw,
                          collate_fn=pretrain_collate, pin_memory=True)

    print(f"[get_pretrain_dataloaders] "
          f"train: {len(train_ds)} samples ({len(config['train_PIDs'])} PIDs, reps {config['train_reps']}) | "
          f"val: {len(val_ds)} samples ({len(config['val_PIDs'])} PIDs, reps {config['val_reps']}) | "
          f"n_classes={train_ds.n_classes}")

    return train_dl, val_dl, train_ds.n_classes
