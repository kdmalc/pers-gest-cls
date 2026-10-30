"""
hybrid_fusion.py
================
H1 hybrid late fusion (docs/specs/SPEC_riemannian.md): a trained model's per-query logits combined with
Riemannian MDM, score = (1 - w) * z(logits) + w * z(-d_MDM), z = per-query standardisation over the
n_way classes. w is chosen by LEAVE-ONE-USER-OUT over the 28 test users (w for user u is the grid value
that maximises the mean accuracy of the other 27), so no user's own labels pick its weight.
w = 0 is the model alone, w = 1 is MDM alone. Laptop, no training; needs the model's --dump-logits npz
files (one per fold) and a trial_distances.py npz of the same EMG representation.

    python system/analysis/hybrid_fusion.py --dist <dir>/dist_fs64.npz \
        --neural "M0=<eval>/kfold7/M0/fs64/fold*/M0_*_query_logits.npz" --out <dir>/hybrid_M0_fs64.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for rel in ("system", "system/MAML", "system/analysis"):
    if str(REPO / rel) not in sys.path:
        sys.path.insert(0, str(REPO / rel))

from failure_analysis import EpisodeIndex, load_neural, zscore_rows  # noqa: E402
from trial_distances import DistanceBank  # noqa: E402

W_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


def per_user_acc(bank, store, folds, ref="mdm"):
    """-> {user: (len(W_GRID),) accuracy}."""
    from episodes import episode_cache as ec
    hits, counts = {}, {}
    for fold in folds:
        for ep in ec.load("test", fold).select(10, 1):
            ix = EpisodeIndex(ep)
            if f"{ep.id}|logits" not in store:
                raise KeyError(f"no logits for {ep.id}")
            if not np.array_equal(store[f"{ep.id}|labels"], ix.y_query):
                raise ValueError(f"{ep.id}: stored labels do not match the cache's query order")
            zn = zscore_rows(store[f"{ep.id}|logits"].astype(np.float64))
            zm = zscore_rows(-bank.episode_distances(ix, ref))
            h = np.array([((1 - w) * zn + w * zm).argmax(1) == ix.y_query for w in W_GRID]).sum(axis=1)
            hits[ep.user] = hits.get(ep.user, 0) + h
            counts[ep.user] = counts.get(ep.user, 0) + len(ix.y_query)
    return {u: hits[u] / counts[u] for u in hits}


def louo(acc: dict) -> dict:
    users = sorted(acc)
    A = np.stack([acc[u] for u in users])                     # (U, W)
    chosen, out = [], []
    for i in range(len(users)):
        others = np.delete(A, i, axis=0).mean(axis=0)
        j = int(others.argmax())
        chosen.append(float(W_GRID[j]))
        out.append(A[i, j])
    return {"louo_acc": float(np.mean(out) * 100), "louo_sd_users": float(np.std(out) * 100),
            "w_chosen": chosen, "per_user_louo": dict(zip(users, map(float, np.asarray(out) * 100))),
            "grid_acc": dict(zip(map(float, W_GRID), map(float, A.mean(axis=0) * 100))),
            "model_alone": float(A[:, 0].mean() * 100), "mdm_alone": float(A[:, -1].mean() * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--neural", required=True, help="NAME=GLOB of the model's --dump-logits npz files")
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(7)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name, store = load_neural(a.neural)
    res = louo(per_user_acc(DistanceBank(a.dist), store, a.folds))
    res["model"] = name
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{name}: alone {res['model_alone']:.1f} | MDM alone {res['mdm_alone']:.1f} | "
          f"hybrid (LOUO w) {res['louo_acc']:.1f} +/- {res['louo_sd_users']:.1f}; "
          f"w chosen: {sorted(set(res['w_chosen']))}")


if __name__ == "__main__":
    main()
