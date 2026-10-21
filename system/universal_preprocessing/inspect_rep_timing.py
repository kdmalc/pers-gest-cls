"""
Recording-time structure of the raw segmented EMG files, from the timestamps in the FILE NAMES.

    python inspect_rep_timing.py "<...>/2024_UIST_dataset/upload/segmented_raw_data"

File name: <prefix><dd><hh><mm><ss>_<PID>_experimenter-defined_EMG_<gesture>_<rep>.csv. For each
participant (the 10 canonical gestures only) it prints the session length, the longest span between
the first and last repetition of one gesture, the median gap between consecutive repetitions, whether
repetition numbers increase with time, and the number of contiguous gesture blocks (10 = every
gesture's repetitions were recorded back-to-back).
"""
import collections
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

CANONICAL = {"close", "delete", "duplicate", "move", "open", "pan", "rotate", "select-single",
             "zoom-in", "zoom-out"}
PAT = re.compile(r"^(\d+)_(P\d+)_experimenter-defined_EMG_(.+)_(\d+)\.csv$")


def secs(ts: str) -> int:
    return ((int(ts[-8:-6]) * 24 + int(ts[-6:-4])) * 60 + int(ts[-4:-2])) * 60 + int(ts[-2:])


def main(root: str):
    root = Path(root)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.name[0] == "P" and p.name[1:].isdigit()):
        ev = sorted((secs(m[1]), m[3], int(m[4])) for f in d.iterdir()
                    if (m := PAT.match(f.name)) and m[3] in CANONICAL)
        if not ev:
            raise ValueError(f"{d.name}: no canonical-gesture EMG files")
        by = collections.defaultdict(list)
        for t, g, r in ev:
            by[g].append((t, r))
        spans = [max(t for t, _ in v) - min(t for t, _ in v) for v in by.values()]
        gaps = [b[0] - a[0] for v in by.values() for a, b in zip(sorted(v), sorted(v)[1:])]
        mono = all([r for _, r in sorted(v)] == sorted(r for _, r in v) for v in by.values())
        seq = [g for _, g, _ in ev]
        blocks = sum(1 for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1])
        rows.append((d.name, len(ev), len(by), (ev[-1][0] - ev[0][0]) / 60, max(spans),
                     st.median(gaps), mono, blocks))
    print("pid  files gestures session_min max_gesture_span_s median_rep_gap_s monotone blocks")
    for r in rows:
        print(*r)
    full = [r for r in rows if r[2] == 10]
    print(f"\n{len(full)} participants with all 10 canonical gestures")
    print("session minutes: median %.1f range %.1f-%.1f" % (
        np.median([r[3] for r in full]), min(r[3] for r in full), max(r[3] for r in full)))
    print("longest span of one gesture's 10 reps (s): median %.0f range %.0f-%.0f" % (
        np.median([r[4] for r in full]), min(r[4] for r in full), max(r[4] for r in full)))
    print("median gap between consecutive reps (s): median %.1f range %.1f-%.1f" % (
        np.median([r[5] for r in full]), min(r[5] for r in full), max(r[5] for r in full)))
    print("rep numbers increase with time for everyone:", all(r[6] for r in full),
          "| every gesture recorded as one contiguous block:", all(r[7] == 10 for r in full))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
