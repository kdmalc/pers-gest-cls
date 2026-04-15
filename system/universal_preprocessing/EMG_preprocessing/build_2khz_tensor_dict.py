"""
build_2khz_tensor_dict.py
=========================
Build a 2 kHz EMG-only tensor_dict .pkl file for use in A10/A11/A12 ablations.

This produces a file in exactly the same format as your existing
`segfilt_rts_tensor_dict.pkl`, but:
  - EMG is kept at 2000 Hz (no downsampling to 64 points)
  - All trials are resampled to a fixed length TARGET_LEN_SAMPLES at 2000 Hz
  - No linear envelope is applied
  - No IMU data is included (Meta's model is EMG-only)

Pipeline
--------
  raw segmented CSVs (2000 Hz, 16 channels, variable ~3-6 s per trial)
      │
      ▼
  BPF 20–450 Hz  +  mean subtraction
      │
      ▼
  per-gesture std normalisation  (std ≈ 1 across all channels in that trial)
      │
      ▼
  resample every trial to TARGET_LEN_SAMPLES at 2000 Hz
  using scipy.signal.resample  (polyphase, preserves 0–450 Hz content)
      │
      ▼
  reshape to tensor (num_trials, TARGET_LEN_SAMPLES, 16)  ← (T, C), channel-last
      │                                                      matches your existing format
      ▼
  save as pkl:  {"data": tensor_dict}
      └── tensor_dict[pid][gesture_class_int] = {
              "emg":         Tensor (num_trials, TARGET_LEN_SAMPLES, 16)
              "imu":         None
              "demo":        Tensor (zeros placeholder — not used by Meta model)
              "gest_ID":     int  (0-indexed gesture class label)
              "enc_gest_ID": int  (same as gest_ID here)
              "enc_pid":     int  (0-indexed participant index)
              "rep_indices": list[int]  (1-indexed trial numbers present)
          }

Why resample instead of truncate
---------------------------------
  Meta's Conv1d was trained with kernel=21, stride=10 at 2000 Hz.  Those
  filters learned MUAP-like waveforms at 2 kHz temporal resolution.
  Resampling to a fixed number of 2 kHz samples preserves that spectral
  content.  Truncating would throw away real signal; resampling at a lower
  fs (e.g. 1000 Hz) would shift the filter's effective receptive field and
  partially invalidate the pretrained weights.

  We pick TARGET_LEN_SAMPLES based on the SURVEY step below:
    1. Run with DRY_RUN = True to print the trial-length distribution.
    2. Pick a TARGET_LEN_SAMPLES ≤ 10th-percentile trial length.
    3. Run with DRY_RUN = False to build and save the pkl.

Usage
-----
  # Step 1: survey your trial lengths
  python build_2khz_tensor_dict.py --dry-run

  # Step 2: set TARGET_LEN_SAMPLES below (or pass --target-len), then build
  python build_2khz_tensor_dict.py

  # Or override target length from the command line:
  python build_2khz_tensor_dict.py --target-len 2000
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample as scipy_resample

# ── Import your shared_processing helpers ─────────────────────────────────────
# Adjust this path to wherever shared_processing.py lives on your cluster.
SHARED_PROCESSING_DIR = Path(
    "/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/universal_preprocessing/EMG_preprocessing"
)
sys.path.insert(0, str(SHARED_PROCESSING_DIR))

from shared_processing import (
    load_segraw_data,
    apply_filter_to_nested_dict,
    normalize_gestures_by_std_any_channels,
)


# =============================================================================
# ── CONFIG — edit these before running ───────────────────────────────────────
# =============================================================================

# TODO: Confirm that this is what I want... idk where I actually want the data
RAW_DATA_DIR = Path(
    "/rhf/allocations/my13/div-emg/data/segmented_raw_data"
)
SAVE_PATH = Path(
    "/rhf/allocations/my13/div-emg/dataset/segfilt_2khz_emg_tensor_dict.pkl"
)

FS = 2000  # Hz — do not change; this is what the raw data was recorded at

# Target sequence length in samples at 2000 Hz.
# Set this AFTER running --dry-run to inspect your distribution.
# Rule of thumb: set to the 10th-percentile trial length or slightly below.
# 2000 samples = 1.0 s at 2 kHz.  Adjust based on your dry-run output.
TARGET_LEN_SAMPLES = 2000  # TODO: confirm after dry-run

# Participant lists — keep in sync with your main pipeline
PIDS_IMPAIRED   = [
    'P102','P103','P104','P105','P106','P107','P108','P109','P110','P111',
    'P112','P114','P115','P116','P118','P119','P121','P122','P123','P124',
    'P125','P126','P127','P128','P131','P132',
]
PIDS_UNIMPAIRED = ['P004','P005','P006','P008','P010','P011']
PIDS_ALL        = PIDS_IMPAIRED + PIDS_UNIMPAIRED

# Gesture name → 0-indexed class label mapping.
# Must match the mapping used in your existing tensor_dict exactly.
# Check your existing dataset or ablation_config to confirm this order.
GESTURE_TO_CLASS = {
    'close':         0,
    'delete':        1,
    'duplicate':     2,
    'move':          3,
    'open':          4,
    'pan':           5,
    'rotate':        6,
    'select-single': 7,
    'zoom-in':       8,
    'zoom-out':      9,
}


# =============================================================================
# ── Survey: print trial-length distribution (dry run) ────────────────────────
# =============================================================================

def survey_trial_lengths(nested_dict: dict) -> None:
    """
    Print statistics on raw trial lengths so you can choose TARGET_LEN_SAMPLES.
    Call this with DRY_RUN=True before committing to a target length.
    """
    lengths = []
    for pid in nested_dict:
        for gesture in nested_dict[pid]:
            for trial in nested_dict[pid][gesture]:
                # nested_dict[pid][gesture][trial]["EMG"] is list of 16 channels,
                # each channel is a list of timepoints.
                ch0 = nested_dict[pid][gesture][trial]["EMG"][0]
                lengths.append(len(ch0))

    lengths = np.array(lengths)
    print("\n=== Trial Length Distribution (samples at 2000 Hz) ===")
    print(f"  N trials  : {len(lengths)}")
    print(f"  Min       : {lengths.min():,}  ({lengths.min()/FS*1000:.0f} ms)")
    print(f"  p5        : {int(np.percentile(lengths,  5)):,}  ({np.percentile(lengths,  5)/FS*1000:.0f} ms)")
    print(f"  p10       : {int(np.percentile(lengths, 10)):,}  ({np.percentile(lengths, 10)/FS*1000:.0f} ms)")
    print(f"  p25       : {int(np.percentile(lengths, 25)):,}  ({np.percentile(lengths, 25)/FS*1000:.0f} ms)")
    print(f"  Median    : {int(np.percentile(lengths, 50)):,}  ({np.percentile(lengths, 50)/FS*1000:.0f} ms)")
    print(f"  p75       : {int(np.percentile(lengths, 75)):,}  ({np.percentile(lengths, 75)/FS*1000:.0f} ms)")
    print(f"  Max       : {lengths.max():,}  ({lengths.max()/FS*1000:.0f} ms)")
    print(f"  Std       : {lengths.std():.0f} samples  ({lengths.std()/FS*1000:.0f} ms)")
    print()
    print("Recommendation: set TARGET_LEN_SAMPLES to the p10 value above,")
    print("or slightly below it to avoid edge cases.  E.g.:")
    print(f"  TARGET_LEN_SAMPLES = {int(np.percentile(lengths, 10)) // 100 * 100}  "
          f"(rounded down to nearest 100)")
    print()

    # Also check: are there any gestures where ALL trials are very short?
    print("=== Per-gesture min trial length ===")
    for pid in list(nested_dict.keys())[:3]:  # spot check first 3 participants
        for gesture in nested_dict[pid]:
            trial_lens = [
                len(nested_dict[pid][gesture][t]["EMG"][0])
                for t in nested_dict[pid][gesture]
            ]
            print(f"  {pid}/{gesture}: min={min(trial_lens):,}  max={max(trial_lens):,}")
    print("  (showing first 3 participants only — check more if needed)")


# =============================================================================
# ── Resample a single trial to TARGET_LEN_SAMPLES ────────────────────────────
# =============================================================================

def resample_trial(emg_channels: list, target_len: int) -> np.ndarray:
    """
    Resample one gesture trial to exactly target_len samples.

    Args:
        emg_channels : list of 16 lists, each of length T_original.
                       i.e. emg_channels[ch][t]
        target_len   : desired output length in samples

    Returns:
        np.ndarray of shape (target_len, 16) — channel-last, matching
        your tensor_dict's (seq_len, num_channels) convention.
    """
    n_channels = len(emg_channels)
    t_original = len(emg_channels[0])

    # Stack to (n_channels, T_original)
    arr = np.array(emg_channels, dtype=np.float32)  # (16, T_orig)

    if t_original == target_len:
        # No resampling needed — just reshape
        return arr.T  # (T, 16)

    # scipy.signal.resample operates along axis=-1 by default (last axis).
    # arr shape: (16, T_orig) → resampled: (16, target_len)
    arr_resampled = scipy_resample(arr, target_len, axis=1)  # (16, target_len)

    return arr_resampled.T.astype(np.float32)  # (target_len, 16)


# =============================================================================
# ── Build the tensor_dict ─────────────────────────────────────────────────────
# =============================================================================

def build_tensor_dict(ppd_dict: dict, target_len: int) -> dict:
    """
    Convert the preprocessed nested dict into the tensor_dict format
    expected by MetaGestureDataset and your existing eval pipeline.

    tensor_dict[pid][gesture_class_int] = {
        "emg"        : Tensor (num_trials, target_len, 16)   — (N, T, C)
        "imu"        : None
        "demo"       : Tensor of zeros  (placeholder; not used)
        "gest_ID"    : int  (0-indexed class label)
        "enc_gest_ID": int  (same as gest_ID)
        "enc_pid"    : int  (participant index, 0-indexed)
        "rep_indices": list[int]  (1-indexed trial numbers)
    }

    Args:
        ppd_dict   : nested dict after BPF + std normalisation
        target_len : fixed sequence length to resample all trials to

    Returns:
        tensor_dict
    """
    tensor_dict = {}

    pid_list = sorted(ppd_dict.keys())

    for pid_idx, pid in enumerate(pid_list):
        if pid not in GESTURE_TO_CLASS.keys().__class__(GESTURE_TO_CLASS):
            # pid is always in tensor_dict — this guard is for gesture names
            pass

        tensor_dict[pid] = {}
        print(f"  Processing {pid} ({pid_idx+1}/{len(pid_list)})...")

        for gesture_name, gesture_class in GESTURE_TO_CLASS.items():
            if gesture_name not in ppd_dict[pid]:
                print(f"    WARNING: gesture '{gesture_name}' not found for {pid}, skipping.")
                continue

            trial_dict = ppd_dict[pid][gesture_name]
            trial_keys = sorted(trial_dict.keys())  # sort for reproducibility

            trial_arrays = []
            for trial_key in trial_keys:
                emg_channels = trial_dict[trial_key]["EMG"]  # list of 16 lists
                arr = resample_trial(emg_channels, target_len)  # (target_len, 16)
                trial_arrays.append(arr)

            # Stack: (num_trials, target_len, 16)
            emg_tensor = torch.from_numpy(
                np.stack(trial_arrays, axis=0)
            ).float()  # (num_trials, T, C)

            assert emg_tensor.shape == (len(trial_keys), target_len, 16), (
                f"Unexpected shape for {pid}/{gesture_name}: {emg_tensor.shape}. "
                f"Expected ({len(trial_keys)}, {target_len}, 16)."
            )

            tensor_dict[pid][gesture_class] = {
                "emg":         emg_tensor,
                "imu":         None,
                "demo":        torch.zeros(12),  # placeholder; Meta model ignores this
                "gest_ID":     gesture_class,
                "enc_gest_ID": gesture_class,
                "enc_pid":     pid_idx,
                "rep_indices": [int(k) if str(k).isdigit() else i+1
                                for i, k in enumerate(trial_keys)],
            }

    return tensor_dict


# =============================================================================
# ── Verification ─────────────────────────────────────────────────────────────
# =============================================================================

def verify_tensor_dict(tensor_dict: dict, target_len: int) -> None:
    """Spot-check the built tensor_dict for shape and value sanity."""
    print("\n=== Verification ===")

    pids = list(tensor_dict.keys())
    print(f"Participants: {len(pids)}")

    sample_pid = pids[0]
    classes = list(tensor_dict[sample_pid].keys())
    print(f"Gesture classes for {sample_pid}: {sorted(classes)}")

    # Shape check
    for pid in pids:
        for cls in tensor_dict[pid]:
            emg = tensor_dict[pid][cls]["emg"]
            assert emg.shape[1] == target_len, (
                f"Shape mismatch: {pid}/class{cls} emg.shape={emg.shape}, "
                f"expected T={target_len}"
            )
            assert emg.shape[2] == 16, (
                f"Channel count wrong: {pid}/class{cls} emg.shape={emg.shape}"
            )

    # Value sanity: check std is reasonable (should be ~1 per trial after normalisation)
    sample_cls  = sorted(tensor_dict[sample_pid].keys())[0]
    sample_emg  = tensor_dict[sample_pid][sample_cls]["emg"]  # (N, T, 16)
    per_trial_stds = sample_emg.std(dim=[1, 2])  # one std per trial
    print(f"\nSample {sample_pid}/class{sample_cls}:")
    print(f"  emg shape     : {sample_emg.shape}")
    print(f"  per-trial std : mean={per_trial_stds.mean():.3f}, "
          f"min={per_trial_stds.min():.3f}, max={per_trial_stds.max():.3f}")
    print(f"  (should be ~1.0 after per-gesture std normalisation)")

    # Check for NaN/Inf
    for pid in pids:
        for cls in tensor_dict[pid]:
            emg = tensor_dict[pid][cls]["emg"]
            assert not torch.isnan(emg).any(), f"NaN in {pid}/class{cls}"
            assert not torch.isinf(emg).any(), f"Inf in {pid}/class{cls}"

    print("\nAll shape, value, and NaN/Inf checks passed.")


# =============================================================================
# ── Main ─────────────────────────────────────────────────────────────────────
# =============================================================================

def main(dry_run: bool, target_len: int):
    print(f"RAW_DATA_DIR       : {RAW_DATA_DIR}")
    print(f"SAVE_PATH          : {SAVE_PATH}")
    print(f"TARGET_LEN_SAMPLES : {target_len} samples = {target_len/FS*1000:.0f} ms at {FS} Hz")
    print(f"DRY_RUN            : {dry_run}")
    print()

    # ── Step 1: Load raw segmented CSVs ──────────────────────────────────────
    print("Step 1/4: Loading raw EMG CSVs (this takes ~50 minutes)...")
    t0 = time.time()

    nested_dict = load_segraw_data(
        pIDs          = PIDS_ALL,
        data_dir_path = str(RAW_DATA_DIR),
        modalities    = ["E"],
        expt_types    = ["experimenter-defined"],
        num_emg_channels = 16,
    )

    print(f"  Done in {time.time()-t0:.1f}s. Participants: {list(nested_dict.keys())}")

    # ── Dry run: survey lengths and exit ─────────────────────────────────────
    if dry_run:
        survey_trial_lengths(nested_dict)
        print("Dry run complete. Set TARGET_LEN_SAMPLES in the script based on")
        print("the distribution above, then re-run without --dry-run.")
        return

    # ── Step 2: BPF 20–450 Hz + mean subtraction ─────────────────────────────
    print("\nStep 2/4: BPF 20–450 Hz + mean subtraction...")
    t0 = time.time()

    filt_dict = apply_filter_to_nested_dict(
        nested_dict,
        normalization_method = "MEANSUBTRACTION",
        already_BPFd         = False,
    )

    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Step 3: Per-gesture std normalisation ─────────────────────────────────
    print("\nStep 3/4: Per-gesture std normalisation...")
    t0 = time.time()

    ppd_dict = normalize_gestures_by_std_any_channels(filt_dict)

    print(f"  Done in {time.time()-t0:.1f}s")

    # Quick sanity check on std
    sample_pid     = list(ppd_dict.keys())[0]
    sample_gesture = list(ppd_dict[sample_pid].keys())[0]
    sample_trial   = list(ppd_dict[sample_pid][sample_gesture].keys())[0]
    sample_data    = ppd_dict[sample_pid][sample_gesture][sample_trial]["EMG"]
    flat = [v for ch in sample_data for v in ch]
    print(f"  Spot-check std (should be ~1.0): {np.std(flat):.4f}")

    # ── Step 4: Build tensor_dict ──────────────────────────────────────────────
    print(f"\nStep 4/4: Building tensor_dict (target_len={target_len} samples)...")
    t0 = time.time()

    tensor_dict = build_tensor_dict(ppd_dict, target_len=target_len)

    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Verify ────────────────────────────────────────────────────────────────
    verify_tensor_dict(tensor_dict, target_len=target_len)

    # ── Save ──────────────────────────────────────────────────────────────────
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {SAVE_PATH} ...")
    t0 = time.time()

    with open(SAVE_PATH, "wb") as f:
        pickle.dump({"data": tensor_dict}, f, protocol=4)

    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"\nFile saved: {SAVE_PATH}")
    print(f"Set EMG_2KHZ_PKL_PATH = Path('{SAVE_PATH}') in A10_A11_A12_meta_pretrained.py")
    print(f"Set EMG_2KHZ_SEQ_LEN  = {target_len} in A10_A11_A12_meta_pretrained.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 2kHz EMG tensor_dict for Meta model ablations.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only load data and print trial-length distribution. Does not build or save the pkl."
    )
    parser.add_argument(
        "--target-len", type=int, default=TARGET_LEN_SAMPLES,
        help=f"Target sequence length in samples at 2000 Hz (default: {TARGET_LEN_SAMPLES})."
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, target_len=args.target_len)