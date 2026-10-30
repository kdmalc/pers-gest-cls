"""
Export the best HPO configuration(s) from an Optuna journal (docs/specs/SPEC_hpo.md).

Replays each trial's sampled parameters through the SAME `build_config_from_trial` the HPO used
(optuna FixedTrial), so there is no hand-written trial-param -> config-key mapping to drift.

    # 1. list/export the top-k candidates of a finished study
    python hpo_export_best.py candidates --ablation M0 --top-k 3 --out cands_M0.json
    # 2. after re-evaluating each candidate with 3 seeds (ablation_hpo.py --replay-trial T --seed S),
    #    pick the winner by mean over seeds and write it into the committed config file
    python hpo_export_best.py finalize --ablation M0 --reeval-dir <dir with reeval_*.json> \
           --out models/hpo_best_models/best_hpo_configs.json

The finalized file maps ablation id -> {"overrides": {config key: value}, "provenance": {...}} and is
what `ablation_config.apply_hpo_config(..., mode="tuned"|"frozen")` consumes. Overrides are the keys
whose value differs from `make_base_config(ablation)` (so the file stays small and auditable).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "test_eval_files"))

import optuna  # noqa: E402
from optuna.storages.journal import JournalFileBackend, JournalStorage  # noqa: E402


def _jsonable(v):
    import torch
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.device):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _load_study(ablation: str, journal: str = None):
    import ablation_hpo as H
    name = H.study_name_for(ablation)   # needs EMG_REPR in the environment
    path = Path(journal) if journal else H.HPO_DB_DIR / f"{name}.log"
    if not path.exists():
        raise FileNotFoundError(path)
    study = optuna.load_study(study_name=name, storage=JournalStorage(JournalFileBackend(str(path))))
    return H, study, name


def overrides_for(H, ablation: str, params: dict) -> dict:
    """The config a trial with these params used, as overrides relative to the base config."""
    from ablation_config import make_base_config
    profile = H.ABLATION_PROFILES[ablation]
    cfg = H.build_config_from_trial(optuna.trial.FixedTrial(params), ablation, profile)
    H.apply_fold_to_config(cfg)
    base = make_base_config(ablation)
    skip = {"train_PIDs", "val_PIDs", "test_PIDs", "device", "timestamp", "results_save_dir",
            "models_save_dir", "val_episode_source", "val_episodes_per_user", "num_epochs",
            "use_earlystopping", "earlystopping_patience", "earlystopping_min_delta",
            "num_workers", "ablation_id"}
    out = {}
    for k, v in cfg.items():
        if k in skip:
            continue
        if k not in base or _jsonable(base[k]) != _jsonable(v):
            out[k] = _jsonable(v)
    return out


def _all_params(t) -> dict:
    """Optuna params plus the frozen ones of a screening study (ablation_hpo HPO_FIXED_PARAMS_JSON)."""
    return {**t.user_attrs.get("fixed_params", {}), **t.params}


def cmd_candidates(a):
    H, study, name = _load_study(a.ablation, a.journal)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit(f"{name}: no COMPLETE trials")
    done.sort(key=lambda t: t.value, reverse=True)
    cands = []
    for rank, t in enumerate(done[: a.top_k]):
        p = _all_params(t)
        cands.append({"rank": rank, "trial": t.number, "value": t.value, "params": p,
                      "overrides": overrides_for(H, a.ablation, p)})
        print(f"rank {rank}: trial {t.number}  dev acc {t.value*100:.2f}%")
    Path(a.out).write_text(json.dumps({"ablation": a.ablation, "study": name,
                                       "n_complete": len(done), "candidates": cands}, indent=1))
    print(f"wrote {a.out}  ({len(done)} complete trials in study)")


def cmd_finalize(a):
    files = sorted(glob.glob(os.path.join(a.reeval_dir, f"reeval_{a.ablation}_trial*_seed*.json")))
    if not files:
        raise SystemExit(f"no reeval_{a.ablation}_trial*_seed*.json in {a.reeval_dir}")
    by_trial = {}
    for f in files:
        r = json.loads(Path(f).read_text())
        by_trial.setdefault(r["trial"], []).append(r)
    summary = {t: {"mean": sum(x["score"] for x in rs) / len(rs), "n_seeds": len(rs),
                   "scores": [x["score"] for x in rs]} for t, rs in by_trial.items()}
    for t, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"trial {t}: mean dev acc {s['mean']*100:.2f}% over {s['n_seeds']} seeds {s['scores']}")
    if len({s["n_seeds"] for s in summary.values()}) != 1:
        raise SystemExit("candidates have different numbers of re-eval seeds; finish them first")
    best_trial = max(summary, key=lambda t: summary[t]["mean"])
    H, study, name = _load_study(a.ablation, a.journal)
    params = _all_params(study.trials[best_trial])
    out_path = Path(a.out)
    allcfg = json.loads(out_path.read_text()) if out_path.exists() else {}
    allcfg[a.ablation] = {
        "overrides": overrides_for(H, a.ablation, params),
        "provenance": {"study": name, "trial": best_trial, "reeval": summary[best_trial],
                       "all_candidates": summary, "params": params},
    }
    out_path.write_text(json.dumps(allcfg, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_path}: {a.ablation} <- trial {best_trial}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--ablation", required=True)
    c.add_argument("--top-k", type=int, default=3)
    c.add_argument("--journal", default=None)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_candidates)
    f = sub.add_parser("finalize")
    f.add_argument("--ablation", required=True)
    f.add_argument("--reeval-dir", required=True)
    f.add_argument("--journal", default=None)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_finalize)
    args = ap.parse_args()
    args.fn(args)
