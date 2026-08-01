#!/usr/bin/env python
"""
find_a16_checkpoints.py
=======================
Locate, validate, and index the pretrained A2 / M0 checkpoints that A16 needs,
and write a manifest mapping  (base, test_pid) -> checkpoint path.

WHY THIS IS NOT JUST `ls`
-------------------------
The eval tree contains A2 checkpoints from two different architectures:

    A2_20260427..30_fixed_seed42_seed42_best.pt      ~2.4 MB   (~0.6M params)
    A2_20260502_*_l2so_foldNN_testPXXX_seed42_best.pt ~24.7 MB  (~6.1M params)

The small ones PREDATE parameter matching (compute_matched_filters_for_ablation
with match_target="all_experts"). Using them would report an upper bound from a
model roughly 10x smaller than M0, which is exactly the comparison the reviewer
is asking us to control for. This script reads the parameter count out of each
state_dict and refuses to put an off-size checkpoint in the manifest without
--allow-mismatched-params.

WHY PER-TEST-SUBJECT CHECKPOINTS
--------------------------------
A16 fine-tunes on a test user's own data. That user must not have been in the
pretraining set, or the "upper bound" is contaminated. The L2SO runs already
produced exactly the right artefact: fold i excluded all_PIDs[i] as test and
all_PIDs[i+1] as val. So for test subject P004 we must load fold28_testP004,
for P104 fold29, and so on. This script builds that mapping from the checkpoint
metadata (config["test_PIDs"]), falling back to the filename if the config key
is missing.

Usage:
    # scan the default NOTS locations, print a report, write the manifest
    python find_a16_checkpoints.py

    # scan somewhere else / write elsewhere
    python find_a16_checkpoints.py \
        --roots /scratch/my13/kai/runs/paper/ablations/eval \
        --out   /scratch/my13/kai/runs/paper/ablations/eval/a16_manifest.json

    # just look, don't write
    python find_a16_checkpoints.py --dry-run
"""

import os, sys, json, re, argparse
from pathlib import Path
from collections import defaultdict

import torch

DEFAULT_ROOTS = [
    "/scratch/my13/kai/runs/paper/ablations/eval",
    "/projects/my13/kai/meta-pers-gest/pers-gest-cls/models/final_eval_models",
]

DEFAULT_OUT = "/scratch/my13/kai/runs/paper/ablations/eval/a16_manifest.json"

# Filenames look like:
#   A2_20260502_1549_l2so_fold28_testP004_seed42_best.pt
#   M0_20260428_1409_fixed_seed42_seed42_best.pt
FNAME_RE = re.compile(
    r"^(?P<base>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{4})_(?P<rest>.+)\.pt$"
)
FOLD_RE  = re.compile(r"l2so_fold(?P<fold>\d+)_test(?P<pid>P\d+)")

# Expected trainable-parameter counts, used only as a sanity band. These are
# derived from the observed checkpoint sizes (float32: bytes / 4 ~ params) and
# are deliberately loose -- the point is to separate 0.6M from 6.1M, not to
# assert an exact count.
PARAM_BANDS = {
    "A2": (3_000_000, 12_000_000),   # parameter-matched to ALL M0 experts
    "M0": (3_000_000, 12_000_000),   # full EncoderMoE
}


def count_state_dict_params(state_dict) -> int:
    return sum(v.numel() for v in state_dict.values()
               if hasattr(v, "numel"))


def inspect_checkpoint(path: Path) -> dict | None:
    """
    Read one checkpoint's metadata without keeping the weights in memory.
    Returns None if the file is not a recognisable ablation checkpoint.
    """
    m = FNAME_RE.match(path.name)
    if not m:
        return None

    base_from_name = m.group("base")
    if base_from_name not in ("A2", "M0"):
        return None

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {
            "path": str(path), "base": base_from_name, "error": repr(e),
            "usable": False,
        }

    state = ckpt.get("model_state_dict", ckpt.get("best_state", None))
    if state is None and isinstance(ckpt, dict) and \
            all(hasattr(v, "numel") for v in ckpt.values()):
        state = ckpt   # bare state_dict

    info = {
        "path":       str(path),
        "filename":   path.name,
        "base":       base_from_name,
        "timestamp":  f"{m.group('date')}_{m.group('time')}",
        "size_bytes": path.stat().st_size,
    }

    if state is None:
        info.update({"usable": False,
                     "error": f"no state_dict; top-level keys={list(ckpt)[:8]}"})
        del ckpt
        return info

    info["n_params"] = count_state_dict_params(state)

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    info["cnn_base_filters"] = cfg.get("cnn_base_filters")
    info["use_MOE"]          = cfg.get("use_MOE")
    info["meta_learning"]    = cfg.get("meta_learning")
    info["ablation_id"]      = cfg.get("ablation_id", ckpt.get("ablation_id"))
    info["fold_id"]          = ckpt.get("fold_id")
    info["best_val_acc"]     = ckpt.get("best_val_acc")

    # Test PID: prefer the config, fall back to the filename.
    test_pids = cfg.get("test_PIDs")
    if isinstance(test_pids, (list, tuple)) and len(test_pids) == 1:
        info["test_pid"] = test_pids[0]
        info["test_pid_source"] = "config"
    else:
        fm = FOLD_RE.search(path.name)
        if fm:
            info["test_pid"] = fm.group("pid")
            info["fold_idx"] = int(fm.group("fold"))
            info["test_pid_source"] = "filename"
        else:
            info["test_pid"] = None
            info["test_pid_source"] = None

    fm = FOLD_RE.search(path.name)
    if fm:
        info["fold_idx"] = int(fm.group("fold"))
    info["is_l2so"] = fm is not None

    lo, hi = PARAM_BANDS[base_from_name]
    info["params_in_band"] = bool(lo <= info["n_params"] <= hi)
    info["usable"] = bool(info["params_in_band"] and info["test_pid"])

    del ckpt, state
    return info


def scan(roots, verbose=True):
    found = []
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            if verbose:
                print(f"[scan] SKIP (missing): {rp}")
            continue
        if verbose:
            print(f"[scan] {rp}")
        for path in sorted(rp.rglob("*.pt")):
            info = inspect_checkpoint(path)
            if info is not None:
                found.append(info)
    return found


def build_manifest(found, allow_mismatched=False):
    """
    (base, test_pid) -> newest usable checkpoint.

    Newest is by embedded timestamp, so a re-run supersedes an earlier one.
    Non-L2SO checkpoints are never selected: they have no single held-out test
    subject, so fine-tuning on any specific user risks that user having been in
    the pretraining set.
    """
    by_key = defaultdict(list)
    for info in found:
        if not info.get("is_l2so"):
            continue
        if not info.get("test_pid"):
            continue
        if not info.get("params_in_band") and not allow_mismatched:
            continue
        by_key[(info["base"], info["test_pid"])].append(info)

    manifest = {}
    for (base, pid), entries in by_key.items():
        entries.sort(key=lambda e: e["timestamp"])
        chosen = entries[-1]
        manifest.setdefault(base, {})[pid] = {
            "path":       chosen["path"],
            "timestamp":  chosen["timestamp"],
            "n_params":   chosen["n_params"],
            "fold_idx":   chosen.get("fold_idx"),
            "n_candidates": len(entries),
        }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-mismatched-params", action="store_true",
                    help="Include checkpoints outside the expected parameter "
                         "band. Only do this if you know why they differ.")
    args = ap.parse_args()

    found = scan(args.roots)
    print(f"\n[find] Inspected {len(found)} candidate checkpoint(s).\n")

    # ── Report: group by base, flag the architecture split ───────────────────
    by_base = defaultdict(list)
    for info in found:
        by_base[info["base"]].append(info)

    for base in sorted(by_base):
        entries = by_base[base]
        print(f"{'='*78}")
        print(f"  {base}: {len(entries)} checkpoint(s)")
        print(f"{'='*78}")

        param_groups = defaultdict(list)
        for e in entries:
            param_groups[e.get("n_params", -1)].append(e)

        if len(param_groups) > 1:
            print(f"  !! {len(param_groups)} DISTINCT parameter counts found. "
                  f"These are different architectures:")
            for n, group in sorted(param_groups.items()):
                lo, hi = PARAM_BANDS[base]
                flag = "OK " if lo <= n <= hi else "OFF"
                print(f"     [{flag}] {n:>12,} params  x{len(group):<3} "
                      f"e.g. {Path(group[0]['path']).name}")
            print(f"     Expected band for {base}: "
                  f"{PARAM_BANDS[base][0]:,} - {PARAM_BANDS[base][1]:,}")
            print(f"     Off-band checkpoints predate parameter matching and "
                  f"are EXCLUDED from the manifest.")
            print()

        l2so = [e for e in entries if e.get("is_l2so")]
        print(f"  L2SO per-fold checkpoints : {len(l2so)}")
        print(f"  non-L2SO (fixed split etc): {len(entries) - len(l2so)}  "
              f"(never selected -- no single held-out subject)")

        broken = [e for e in entries if e.get("error")]
        if broken:
            print(f"  !! {len(broken)} unreadable:")
            for e in broken[:5]:
                print(f"     {Path(e['path']).name}: {e['error'][:80]}")
        print()

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = build_manifest(found, allow_mismatched=args.allow_mismatched_params)

    print(f"{'='*78}")
    print(f"  MANIFEST")
    print(f"{'='*78}")
    for base in sorted(manifest):
        pids = manifest[base]
        print(f"  {base}: {len(pids)} subject(s) covered")
        for pid in sorted(pids):
            e = pids[pid]
            dupe = f"  ({e['n_candidates']} candidates, took newest)" \
                   if e["n_candidates"] > 1 else ""
            print(f"     {pid}  fold{e['fold_idx']:>2}  {e['n_params']:>10,}p  "
                  f"{e['timestamp']}{dupe}")
    if not manifest:
        print("  EMPTY -- no usable per-fold checkpoints found.")
        print("  A16 will have to pretrain inline (--base A2 supports this).")

    # ── Coverage check against the paper's 4 test subjects ───────────────────
    PAPER_TEST_PIDS = ["P004", "P104", "P105", "P121"]
    print(f"\n  Coverage of the paper's fixed test split {PAPER_TEST_PIDS}:")
    for base in ("A2", "M0"):
        have = manifest.get(base, {})
        missing = [p for p in PAPER_TEST_PIDS if p not in have]
        status = "COMPLETE" if not missing else f"MISSING {missing}"
        print(f"     {base}: {status}")

    if not args.dry_run:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump({"manifest": manifest, "all_inspected": found}, f, indent=2)
        print(f"\n[find] Wrote manifest to {outp}")
        print(f"[find] Pass it to A16 with:  --manifest {outp}")
    else:
        print(f"\n[find] --dry-run: manifest not written.")


if __name__ == "__main__":
    main()
