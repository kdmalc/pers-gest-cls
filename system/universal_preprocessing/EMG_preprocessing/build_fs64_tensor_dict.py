"""
Build segfilt_fs64_tensor_dict.pkl: the fused tensor dict with EMG replaced by the EMG-FS64 feature
sequence (docs/SPEC_2khz_emg.md). IMU, demographics, labels and rep ids are taken UNCHANGED from the
existing fused dict (joined by pid / class / rep), so the episode cache and every consumer work as is.

    python build_fs64_tensor_dict.py --root <segmented_raw_data> \
        --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl \
        --out dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl

Feature standardisation (one mean/std per feature TYPE, shared over channels and steps) uses the 4 DEV
users only. They appear in no fold, so no train/val/test information enters the features. The
constants are stored in the file (`feature_norm`).

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
sys.path.insert(0, str(HERE.parents[1] / "fixed_user_splits"))
import emg_fs64 as F  # noqa: E402
import kfold_split  # noqa: E402
from raw_emg_io import GESTURE_TO_CLASS, N_REPS, list_emg_files, read_emg_csv  # noqa: E402


def build(root, fused_path, out_path, dev_pids=None, fs=F.FS_EMG, log=print):
    dev_pids = list(kfold_split.hpo_val_pids()) if dev_pids is None else list(dev_pids)
    with open(fused_path, "rb") as f:
        fused = pickle.load(f)
    data = fused["data"]
    feats = {}                                           # (pid, cls) -> (10, 64, 112) float32
    for pid in sorted(data):
        files = list_emg_files(root, pid)
        for name, cls in GESTURE_TO_CLASS.items():
            entry = data[pid][cls]
            if entry["gesture_id"] != name:
                raise ValueError(f"{pid} class {cls}: fused dict says {entry['gesture_id']!r}, "
                                 f"expected {name!r}")
            if list(entry["rep_indices"]) != list(range(1, N_REPS + 1)):
                raise ValueError(f"{pid}/{name}: fused rep_indices {entry['rep_indices']}")
            feats[(pid, cls)] = np.stack(
                [F.trial_to_fs64(read_emg_csv(path), fs) for _, path, _ in files[name]])
        log(f"{pid}: features built")

    # per feature-type standardisation from the dev users only
    dev = np.concatenate([feats[(p, c)].reshape(-1, 16, F.N_FEATURES)
                          for p in dev_pids for c in range(10)], axis=0)    # (n, 16, F)
    mean, std = dev.mean(axis=(0, 1)), dev.std(axis=(0, 1))
    if not (std > 0).all():
        raise ValueError("a feature type has zero variance on the dev users")
    for (pid, cls), x in feats.items():
        z = (x.reshape(N_REPS, F.N_STEPS, 16, F.N_FEATURES) - mean) / std
        data[pid][cls]["emg"] = torch.from_numpy(z.reshape(N_REPS, F.N_STEPS, -1).astype(np.float32))
    fused["emg_repr"] = "fs64"
    fused["fs_emg"] = fs
    fused["feature_names"] = list(F.FEATURE_NAMES)
    fused["feature_norm"] = {"mean": mean.tolist(), "std": std.tolist(),
                             "computed_from_dev_pids": dev_pids}
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
