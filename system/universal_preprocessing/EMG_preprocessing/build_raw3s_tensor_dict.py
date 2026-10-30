"""
Build segfilt_raw3s_tensor_dict.pkl: the fused tensor dict with EMG replaced by native-rate EMG, cropped
to a fixed 3.0 s window (EMG-RAW, docs/specs/SPEC_2khz_emg.md §3). Read by the learned strided stem
(config["emg_stem"] = "raw_strided"; set automatically by EMG_REPR=raw3s).

    python build_raw3s_tensor_dict.py --root <segmented_raw_data> \
        --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl \
        --out dataset/meta-learning-sup-que-ds/segfilt_raw3s_tensor_dict.pkl

Per trial: the fs64 preprocessing (DC removal, 20-450 Hz band-pass, per-trial std = 1; emg_fs64.preprocess),
then a 6144-sample (3.0 s at 2048 Hz) window centred on the activity centroid, i.e. the energy-weighted
mean time of the band-passed signal (summed over channels, 0.1 s moving average). Zero padding where the
window leaves the trial. Stored as float16 (the stem casts to float32), shape (10, 6144, 16) per
(pid, class), like every other dict on disk ((trials, T, C); reorient_tensor_dict flips it).

DEVIATION FROM THE SPEC (deliberate): the spec re-crops the IMU to the same absolute window. Here the
IMU is taken UNCHANGED from the fused dict, exactly as fs64 does, so EMG-RAW vs EMG-FS64 differ ONLY in
the EMG input (the question the ablation asks), and the IMU confound the spec warns about does not arise.
Raises on any mismatch (missing rep, label order, pid set). No trial is skipped.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emg_fs64 as F  # noqa: E402
from raw_emg_io import GESTURE_TO_CLASS, N_REPS, list_emg_files, read_emg_csv  # noqa: E402

WINDOW_S = 3.0
SMOOTH_S = 0.1


def crop_window(x_lc: np.ndarray, fs: float = F.FS_EMG, window_s: float = WINDOW_S) -> np.ndarray:
    """Preprocessed (L, C) -> (W, C) window centred on the activity centroid, zero-padded."""
    W = int(round(window_s * fs))
    L = x_lc.shape[0]
    energy = (x_lc ** 2).sum(axis=1)
    k = max(1, int(round(SMOOTH_S * fs)))
    energy = np.convolve(energy, np.ones(k) / k, mode="same")
    if not np.isfinite(energy).all() or energy.sum() <= 0:
        raise ValueError("degenerate trial: no finite positive energy")
    centre = int(round((np.arange(L) * energy).sum() / energy.sum()))
    start = centre - W // 2
    out = np.zeros((W, x_lc.shape[1]), dtype=np.float64)
    src_lo, src_hi = max(0, start), min(L, start + W)
    out[src_lo - start: src_hi - start] = x_lc[src_lo:src_hi]
    return out


def trial_to_raw3s(emg_lc: np.ndarray, fs: float = F.FS_EMG) -> np.ndarray:
    """Raw trial (L, 16) -> (6144, 16) float16."""
    return crop_window(F.preprocess(emg_lc, fs), fs).astype(np.float16)


def build(root, fused_path, out_path, fs=F.FS_EMG, log=print):
    with open(fused_path, "rb") as f:
        fused = pickle.load(f)
    data = fused["data"]
    for pid in sorted(data):
        files = list_emg_files(root, pid)
        for name, cls in GESTURE_TO_CLASS.items():
            entry = data[pid][cls]
            if entry["gesture_id"] != name:
                raise ValueError(f"{pid} class {cls}: fused dict says {entry['gesture_id']!r}, expected {name!r}")
            if list(entry["rep_indices"]) != list(range(1, N_REPS + 1)):
                raise ValueError(f"{pid}/{name}: fused rep_indices {entry['rep_indices']}")
            x = np.stack([trial_to_raw3s(read_emg_csv(path), fs) for _, path, _ in files[name]])
            entry["emg"] = torch.from_numpy(x)                  # (10, 6144, 16) float16
        log(f"{pid}: raw windows built")
    fused["emg_repr"] = "raw3s"
    fused["fs_emg"] = fs
    fused["raw_window"] = {"seconds": WINDOW_S, "samples": int(round(WINDOW_S * fs)),
                           "centre": "activity centroid (energy-weighted, 0.1 s smoothing)",
                           "preprocess": "emg_fs64.preprocess (DC, 20-450 Hz band-pass, per-trial std=1)",
                           "imu": "unchanged from the fused dict (same as fs64)"}
    with open(out_path, "wb") as f:
        pickle.dump(fused, f, protocol=4)
    log(f"wrote {out_path}")
    return fused


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fused-dict", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.root, a.fused_dict, a.out)
