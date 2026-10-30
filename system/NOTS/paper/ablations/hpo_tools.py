"""
hpo_tools.py
============
Read-only helpers for the Optuna journals written by ablation_hpo.py. Runs on the login node (CPU,
needs only optuna + numpy from the env).

  report  one line per study: trials by state, best dev score, wall time per trial, GPU-hours
          python hpo_tools.py report --db-dir /scratch/my13/kai/runs/paper/sub2/hpo/optuna_dbs
          python hpo_tools.py report --db-dir ... --trials ablation_M0_10w1s_kfold7_fs64_v1   # per trial
          python hpo_tools.py report --db-dir ... --csv hpo_summary.csv

  grid    build an enqueue file that varies some params around one trial (e.g. the E x top_k sweep)
          python hpo_tools.py grid --db-dir ... --study ablation_M0_10w1s_kfold7_fs64_v1 --trial best \
              --vary num_experts=4,8,16,22,36 --vary MOE_top_k=1,2,4,9 --out m0_exk_grid.json
          then: python ablation_hpo.py --ablation M0 --enqueue-json m0_exk_grid.json --enqueue-only
          (with HPO_STUDY_SUFFIX=exk set) and launch the array with --study-suffix exk.

  freeze  write chosen params of one trial to JSON, to freeze them in a screening study
          python hpo_tools.py freeze --db-dir ... --study ablation_A4_10w1s_kfold7_fs64_v1 --trial best \
              --keys outer_lr,wd,maml_inner_steps,maml_alpha_init,maml_alpha_init_eval --out shared_maml.json
          then launch with  --fixed-params /full/path/shared_maml.json

Wall time per trial comes from the trial's "wall_time_s" user attr (ablation_hpo.py, every trial
from 2026-10-30 on). Older trials have no such attr; for those the report uses
datetime_complete - datetime_start and marks the column "(from timestamps)". Running trials have
no end time and are counted, not timed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def _load_studies(db_dir: Path):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalStorage
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for log in sorted(db_dir.glob("*.log")):
        storage = JournalStorage(JournalFileBackend(str(log)))
        for s in optuna.get_all_study_summaries(storage):
            yield log, optuna.load_study(study_name=s.study_name, storage=storage)


def _wall(t):
    if "wall_time_s" in t.user_attrs:
        return float(t.user_attrs["wall_time_s"]), "attr"
    if t.datetime_start is not None and t.datetime_complete is not None:
        return (t.datetime_complete - t.datetime_start).total_seconds(), "timestamps"
    return None, None


def report(args):
    rows = []
    for log, study in _load_studies(Path(args.db_dir)):
        if args.trials and study.study_name != args.trials:
            continue
        by_state = {}
        walls, sources, gpu_h = [], set(), 0.0
        for t in study.trials:
            by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
            w, src = _wall(t)
            if w is not None:
                gpu_h += w / 3600.0
                sources.add(src)
                if t.state.name == "COMPLETE":
                    walls.append(w)
        done = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
        best = max(done, key=lambda t: t.value) if done else None
        row = {
            "study": study.study_name,
            "complete": by_state.get("COMPLETE", 0), "pruned": by_state.get("PRUNED", 0),
            "fail": by_state.get("FAIL", 0), "running": by_state.get("RUNNING", 0),
            "waiting": by_state.get("WAITING", 0),
            "best_dev_acc": round(best.value * 100, 2) if best else None,
            "best_trial": best.number if best else None,
            "median_h_complete": round(float(np.median(walls)) / 3600, 2) if walls else None,
            "max_h_complete": round(float(np.max(walls)) / 3600, 2) if walls else None,
            "gpu_hours_total": round(gpu_h, 1),
            "time_source": ",".join(sorted(sources)) or "-",
        }
        rows.append(row)
        if args.trials:
            print(f"{'trial':>5} {'state':>9} {'dev_acc':>8} {'hours':>6} {'epochs':>6} {'params':>10}  host")
            for t in study.trials:
                w, _ = _wall(t)
                print(f"{t.number:>5} {t.state.name:>9} "
                      f"{(f'{t.value*100:.2f}' if t.value is not None else '-'):>8} "
                      f"{(f'{w/3600:.2f}' if w is not None else '-'):>6} "
                      f"{str(t.user_attrs.get('n_epochs_run', '-')):>6} "
                      f"{str(t.user_attrs.get('n_params', '-')):>10}  {t.user_attrs.get('host', '-')}")
    if not rows:
        sys.exit(f"no studies found in {args.db_dir}" + (f" named {args.trials}" if args.trials else ""))
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"\nTOTAL GPU-hours over these studies: {sum(r['gpu_hours_total'] for r in rows):.1f} "
          f"(timed trials only; RUNNING trials not included)")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


def grid(args):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalStorage
    storage = JournalStorage(JournalFileBackend(str(Path(args.db_dir) / f"{args.study}.log")))
    study = optuna.load_study(study_name=args.study, storage=storage)
    if args.trial == "best":
        base = study.best_trial.params
    else:
        base = study.trials[int(args.trial)].params
    axes = []
    for spec in args.vary:
        key, vals = spec.split("=", 1)
        if key not in base:
            sys.exit(f"{key} is not a parameter of the base trial (params: {sorted(base)})")
        axes.append((key, [type(base[key])(v) for v in vals.split(",")]))
    items = []
    for combo in itertools.product(*[v for _, v in axes]):
        p = dict(base)
        p.update({k: v for (k, _), v in zip(axes, combo)})
        if "num_experts" in p and "MOE_top_k" in p and p["MOE_top_k"] > p["num_experts"]:
            continue
        items.append(p)
    Path(args.out).write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} configs to {args.out} (base: trial {args.trial} of {args.study})")


def freeze(args):
    """Write {name: value} of chosen params of one trial: the frozen shared HPs for a screening study."""
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalStorage
    storage = JournalStorage(JournalFileBackend(str(Path(args.db_dir) / f"{args.study}.log")))
    study = optuna.load_study(study_name=args.study, storage=storage)
    t = study.best_trial if args.trial == "best" else study.trials[int(args.trial)]
    params = {**t.user_attrs.get("fixed_params", {}), **t.params}
    keys = params.keys() if args.keys == "all" else args.keys.split(",")
    missing = [k for k in keys if k not in params]
    if missing:
        sys.exit(f"{missing} not in trial {t.number}'s params {sorted(params)} "
                 f"(conditional params such as maml_msl_num_epochs exist only when active)")
    out = {k: params[k] for k in keys}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {out}  (trial {t.number} of {args.study}, dev acc {t.value*100:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--db-dir", required=True)
    r.add_argument("--trials", default=None, help="study name: also print one line per trial")
    r.add_argument("--csv", default=None)
    g = sub.add_parser("grid")
    g.add_argument("--db-dir", required=True)
    g.add_argument("--study", required=True)
    g.add_argument("--trial", default="best")
    g.add_argument("--vary", action="append", required=True, help="key=v1,v2,... (repeatable)")
    g.add_argument("--out", required=True)
    z = sub.add_parser("freeze")
    z.add_argument("--db-dir", required=True)
    z.add_argument("--study", required=True)
    z.add_argument("--trial", default="best")
    z.add_argument("--keys", required=True, help="comma-separated param names, or 'all'")
    z.add_argument("--out", required=True)
    a = ap.parse_args()
    {"report": report, "grid": grid, "freeze": freeze}[a.cmd](a)
