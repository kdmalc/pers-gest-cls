"""
Diagnostic: what does the data say about repetition order / sessions?  (audit finding 4.4)

Reads the metadata pickle (DataFrame with at least Participant, Gesture_ID, Gesture_Num)
and prints, per participant: the row-order structure of (Gesture_ID, Gesture_Num) trials,
the longest run of consecutive trials of one gesture, whether Gesture_Num is monotone
within each gesture block, and any column whose name suggests time/session/date.

CAVEAT printed at the end: file row order is only evidence about recording order if the
file was never sorted. The authoritative answer is the data-collection protocol / lab
notes; this script only tells us what the file can and cannot support.

Usage (cluster):  python inspect_rep_structure.py /scratch/my13/kai/meta-pers-gest/data/filtered_datasets/metadata_IMU_EMG_allgestures_allusers.pkl
"""
import re
import sys

import pandas as pd

REQUIRED = ["Participant", "Gesture_ID", "Gesture_Num"]
TIMEISH = re.compile(r"time|date|session|stamp|clock|day|block|order|index", re.I)


def main(path: str) -> None:
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {path}, got {type(obj)}")
    df = obj
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}; columns are {list(df.columns)[:40]}")

    print(f"rows={len(df)}  columns={len(df.columns)}")
    timeish = [c for c in df.columns if TIMEISH.search(str(c))]
    print(f"columns whose names suggest time/session/order: {timeish}")
    print(f"index type: {type(df.index).__name__}, monotone increasing: {df.index.is_monotonic_increasing}")

    for pid, g in df.groupby("Participant", sort=True):
        # one entry per trial = consecutive rows with the same (Gesture_ID, Gesture_Num)
        key = list(zip(g["Gesture_ID"].tolist(), g["Gesture_Num"].tolist()))
        trials = [k for i, k in enumerate(key) if i == 0 or k != key[i - 1]]
        gestures = [t[0] for t in trials]
        runs, cur = [], 1
        for i in range(1, len(gestures)):
            if gestures[i] == gestures[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        n_gest = len(set(gestures))
        mono = all(
            [n for gid, n in trials if gid == gg] == sorted(n for gid, n in trials if gid == gg)
            for gg in set(gestures)
        )
        reps_per_gest = {gg: sum(1 for t in trials if t[0] == gg) for gg in set(gestures)}
        layout = "BLOCKED (all reps of a gesture contiguous)" if max(runs) >= 10 else "INTERLEAVED/ROUND-ROBIN"
        print(f"{pid}: trials={len(trials)} gestures={n_gest} reps/gesture(min,max)="
              f"({min(reps_per_gest.values())},{max(reps_per_gest.values())}) "
              f"longest same-gesture run={max(runs)} Gesture_Num monotone={mono} -> {layout}")

    print("\nCAVEAT: row order is evidence of recording order only if the file was never sorted "
          "(Gesture_Num is sorted ascending at tensor-save time, tensor_saving_with_morphology.ipynb). "
          "No wall-clock or session column was found unless listed above; session count and "
          "donning/doffing must come from the collection protocol / lab notes.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
