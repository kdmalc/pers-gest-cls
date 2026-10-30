"""
gesture_descriptors.py
======================
Per (user, gesture) movement and body-region descriptors from the RAW segmented CSVs (Box), for the
failure analysis (failure_analysis.py) and the high- vs low-movement analysis. The tensor dicts cannot
be used for movement magnitude: their trials are scale-normalised (per-trial IMU spread is ~0.5-0.8 for
every gesture), so how MUCH a gesture moves is lost there. Region shares are fine in either.

Per trial (raw IMU ~150 Hz, raw EMG 2048 Hz):
  gyro_s         mean |angular velocity| of IMU sensor s over the trial (per-trial median removed: bias)
  emg_rms_c      RMS of EMG channel c after a 20-450 Hz band-pass (4th-order Butterworth, zero phase)
  duration_s     trial length
Per (user, gesture) = median over the 10 reps:
  motion         sum_s gyro_s (whole-body rotational activity; the movement-magnitude variable)
  imu_share_<R>  share of motion per IMU region (6 regions; hands have no IMU, body_regions.py)
  emg_rel_c      emg_rms_c / the user's median emg_rms_c over all 100 trials (removes per-user electrode gain)
  emg_share_<R>  share of sum_c emg_rel_c per region (8 regions): which region this gesture recruits
                 MORE than the user's typical gesture
  emg_level      mean_c emg_rel_c (relative effort)
Labels (pre-stated here, before any accuracy is looked at):
  movement_level        tertile of log(motion) over all (user, gesture) pairs: low | mid | high
  movement_level_user   the same within each user's 10 gestures (3 low / 4 mid / 3 high)
  side                  IMU motion share of the right side: > 0.65 R, < 0.35 L, else bilateral
  emg_level_region      argmax of the EMG share merged over sides: hand | forearm | upper_arm | shoulder
  imu_region            argmax of imu_share (6 regions)
  distal_emg_share      hand + forearm EMG share
  finger_like           movement_level == low and distal_emg_share in the top tertile over all pairs
                        (wrist / finger gestures: muscle activity without much limb rotation)

    python system/analysis/gesture_descriptors.py --root "<Box>/segmented_raw_data" --out <dir>/gesture_descriptors.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for rel in ("system", "system/universal_preprocessing/EMG_preprocessing"):
    if str(REPO / rel) not in sys.path:
        sys.path.insert(0, str(REPO / rel))

from arch_variants.body_regions import REGIONS, N_IMU  # noqa: E402

FS_EMG = 2048.0
LEVELS = ("hand", "forearm", "upper_arm", "shoulder")


def read_imu_csv(path) -> np.ndarray:
    """-> (L, 72): IMU1..8, 9, 11, 13, 15 x (ax ay az vx vy vz), the tensor dict's sensor order."""
    df = pd.read_csv(path, index_col=0)
    slots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15]
    cols = [f"IMU{s}_{a}" for s in slots for a in ("ax", "ay", "az", "vx", "vy", "vz")]
    if list(df.columns) != cols:
        raise ValueError(f"{path}: unexpected IMU columns {list(df.columns)[:4]}...")
    return df.to_numpy(dtype=np.float64)


def imu_path_for(emg_path: Path) -> Path:
    p = emg_path.with_name(emg_path.name.replace("_EMG_", "_IMU_"))
    if not p.exists():
        raise FileNotFoundError(f"no IMU file next to {emg_path.name}: expected {p.name}")
    return p


def trial_features(emg_lc: np.ndarray, imu_lc: np.ndarray) -> dict:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [20.0, 450.0], btype="band", fs=FS_EMG, output="sos")
    e = sosfiltfilt(sos, emg_lc - emg_lc.mean(axis=0, keepdims=True), axis=0)
    rms = np.sqrt((e ** 2).mean(axis=0))                                   # (16,)
    gyro = imu_lc.reshape(len(imu_lc), N_IMU, 6)[:, :, 3:]                 # (L, 12, 3)
    gyro = gyro - np.median(gyro, axis=0, keepdims=True)
    gmag = np.linalg.norm(gyro, axis=2).mean(axis=0)                       # (12,)
    return {"emg_rms": rms, "gyro": gmag, "duration_s": len(emg_lc) / FS_EMG}


def user_descriptors(root, pid: str, log=print) -> pd.DataFrame:
    from raw_emg_io import GESTURE_TO_CLASS, list_emg_files, read_emg_csv
    files = list_emg_files(root, pid)
    rows = []
    for g, items in files.items():
        for rep, path, _ in items:
            f = trial_features(read_emg_csv(path), read_imu_csv(imu_path_for(path)))
            rows.append({"user": pid, "gesture": g, "cls": GESTURE_TO_CLASS[g], "rep": rep - 1, **f})
    df = pd.DataFrame(rows)
    emg = np.stack(df["emg_rms"].to_numpy())
    rel = emg / np.median(emg, axis=0, keepdims=True)                     # per-user channel gain removed
    gyro = np.stack(df["gyro"].to_numpy())
    out = []
    for (g, cls), idx in df.groupby(["gesture", "cls"]).groups.items():
        idx = np.asarray(list(idx))
        r = np.median(rel[idx], axis=0)
        gy = np.median(gyro[idx], axis=0)
        row = {"user": pid, "gesture": g, "cls": int(cls), "motion": float(gy.sum()),
               "motion_max_sensor": float(gy.max()), "emg_level": float(r.mean()),
               "duration_s": float(np.median(df.loc[idx, "duration_s"]))}
        tot_e, tot_i = r.sum(), gy.sum()
        for reg in REGIONS:
            row[f"emg_share_{reg.name}"] = float(r[list(reg.emg)].sum() / tot_e)
            if reg.imu:
                row[f"imu_share_{reg.name}"] = float(gy[list(reg.imu)].sum() / tot_i)
        out.append(row)
    log(f"  {pid}: {len(df)} trials")
    return pd.DataFrame(out)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lm = np.log(df["motion"])
    q1, q2 = np.quantile(lm, [1 / 3, 2 / 3])
    df["movement_level"] = np.where(lm <= q1, "low", np.where(lm <= q2, "mid", "high"))
    ranks = df.groupby("user")["motion"].rank(method="first")
    df["movement_level_user"] = np.where(ranks <= 3, "low", np.where(ranks <= 7, "mid", "high"))
    r_share = sum(df[f"imu_share_{r.name}"] for r in REGIONS if r.side == "R" and r.imu)
    df["imu_share_R"] = r_share
    df["side"] = np.where(r_share > 0.65, "R", np.where(r_share < 0.35, "L", "bilateral"))
    lvl = {L: sum(df[f"emg_share_{r.name}"] for r in REGIONS if r.level == L) for L in LEVELS}
    for L, v in lvl.items():
        df[f"emg_share_level_{L}"] = v
    df["emg_level_region"] = pd.DataFrame(lvl).idxmax(axis=1)
    imu_cols = [f"imu_share_{r.name}" for r in REGIONS if r.imu]
    df["imu_region"] = df[imu_cols].idxmax(axis=1).str.replace("imu_share_", "", regex=False)
    df["distal_emg_share"] = lvl["hand"] + lvl["forearm"]
    top = np.quantile(df["distal_emg_share"], 2 / 3)
    df["finger_like"] = (df["movement_level"] == "low") & (df["distal_emg_share"] > top)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="segmented_raw_data folder (Box)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--users", nargs="*", default=None, help="default: every user in the fs64 tensor dict")
    a = ap.parse_args()
    if a.users:
        users = a.users
    else:
        import pickle
        with open(REPO / "dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl", "rb") as f:
            users = sorted(pickle.load(f)["data"])
    t0 = time.time()
    df = add_labels(pd.concat([user_descriptors(a.root, u) for u in users], ignore_index=True))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"saved {a.out}: {len(df)} (user, gesture) rows from {len(users)} users in {time.time() - t0:.0f}s")
    print(df["movement_level"].value_counts().to_dict(), df["side"].value_counts().to_dict(),
          df["emg_level_region"].value_counts().to_dict(), "finger_like:", int(df["finger_like"].sum()))


if __name__ == "__main__":
    main()
