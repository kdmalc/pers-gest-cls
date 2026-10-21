"""
Reading the raw segmented EMG CSVs (Box: .../2024_UIST_dataset/upload/segmented_raw_data/<PID>/).

File name: <ddhhmmss-ish timestamp>_<PID>_experimenter-defined_EMG_<gesture>_<rep>.csv, columns
",EMG1..EMG16" (first column = sample index). The 10 canonical gestures map to classes 0..9
alphabetically, matching the fused tensor dict's LabelEncoder.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

GESTURE_TO_CLASS = {
    "close": 0, "delete": 1, "duplicate": 2, "move": 3, "open": 4,
    "pan": 5, "rotate": 6, "select-single": 7, "zoom-in": 8, "zoom-out": 9,
}
N_REPS = 10
# DATA ANOMALY (found 2026-10-21): the raw folder P005 contains 560 files whose NAMES say P105 (a
# different cohort: timestamps 2306/2307 vs P105's own folder 2309, different bytes, no file named
# P005). The original preprocessing keyed on the FOLDER (load_segraw_data), so P005's tensors come
# from these files. We keep that, explicitly and only for this pair; any other mismatch raises.
# Needs Kai's confirmation that the P005 folder really is P005 (see docs/KAI_TODO.md).
KNOWN_NAME_MISMATCH = {"P005": "P105"}
_EMG_NAME = re.compile(r"^(?P<ts>\d+)_(?P<pid>P\d+)_experimenter-defined_EMG_(?P<gesture>.+)_(?P<rep>\d+)\.csv$")


def timestamp_seconds(ts: str) -> int:
    """Last 8 digits = dd hh mm ss of the recording (verified: reps of a gesture are 4-7 s apart)."""
    return ((int(ts[-8:-6]) * 24 + int(ts[-6:-4])) * 60 + int(ts[-4:-2])) * 60 + int(ts[-2:])


def list_emg_files(root, pid: str) -> dict:
    """-> {gesture: [(rep, path, t_seconds), ...] sorted by rep}, canonical gestures only.
    Raises unless every canonical gesture has exactly reps 1..10."""
    d = Path(root) / pid
    if not d.is_dir():
        raise FileNotFoundError(d)
    out = {g: [] for g in GESTURE_TO_CLASS}
    for f in d.iterdir():
        m = _EMG_NAME.match(f.name)
        if m and m["gesture"] in out:
            allowed = {pid} | ({KNOWN_NAME_MISMATCH[pid]} if pid in KNOWN_NAME_MISMATCH else set())
            if m["pid"] not in allowed:
                raise ValueError(f"{f}: file name PID != folder PID")
            out[m["gesture"]].append((int(m["rep"]), f, timestamp_seconds(m["ts"])))
    for g, items in out.items():
        items.sort()
        if [r for r, _, _ in items] != list(range(1, N_REPS + 1)):
            raise ValueError(f"{pid}/{g}: reps {[r for r, _, _ in items]} != 1..{N_REPS}")
    return out


def read_emg_csv(path) -> np.ndarray:
    """-> (L, 16) float64."""
    df = pd.read_csv(path, index_col=0)
    if list(df.columns) != [f"EMG{i}" for i in range(1, 17)]:
        raise ValueError(f"{path}: unexpected columns {list(df.columns)[:4]}...")
    return df.to_numpy(dtype=np.float64)
