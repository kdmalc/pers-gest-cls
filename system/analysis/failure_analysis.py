"""
failure_analysis.py
===================
Which queries does each method get wrong, do the methods fail on DIFFERENT queries, and can gesture
properties (movement magnitude, body region, side) predict where covariance wins or loses?
docs/specs/SPEC_riemannian.md section 3 (pre-stated analyses). Laptop, no training.

Inputs
  --dist         trial_distances.py npz (classical 1-shot nearest-neighbour methods)
  --descriptors  gesture_descriptors.py csv (raw-CSV movement / region descriptors), optional
  --neural NAME=GLOB   per-query logits dumped by the eval scripts (--dump-logits), one npz per fold;
                 repeatable. Episodes are joined by id and the stored labels are checked.
Every 10-way 1-shot cached TEST episode of every fold (28 users) unless --max-per-user is given.

Outputs (--out dir)
  queries.csv            one row per (episode, query): user, fold, class, rep gap, correct_<m>, margin_<m>
  user_gesture_acc.csv   accuracy per (user, gesture, method) + descriptors
  summary.json           per-method accuracy; pairwise complementarity (oracle-or, both-wrong, error phi);
                         z-scored distance-level fusions; accuracy by descriptor label; the Delta(MDM - X)
                         association with descriptors (Spearman) and its leave-one-user-out prediction
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for rel in ("system", "system/MAML", "system/analysis"):
    if str(REPO / rel) not in sys.path:
        sys.path.insert(0, str(REPO / rel))

from trial_distances import DistanceBank  # noqa: E402

LABEL_COLS = ("movement_level", "movement_level_user", "side", "emg_level_region", "imu_region", "finger_like")
CONT_COLS = ("log_motion", "emg_level", "distal_emg_share", "laterality", "duration_s")
FUSIONS = {"mdm+td_knn": ("mdm", "td_knn"), "mdm+dollar_b": ("mdm", "dollar_b"),
           "mdm+knn_l1": ("mdm", "knn_l1"), "mdm+seg_mdm": ("mdm", "seg_mdm"),
           "mdm_emg+mdm_imu+td_knn": ("mdm_emg", "mdm_imu", "td_knn")}


class EpisodeIndex:
    """The label / class / rep layout of a cached episode without loading any signal (same order as
    episodes.episode_numpy.materialize)."""

    def __init__(self, ep):
        self.episode_id, self.user, self.n_way, self.k_shot = ep.id, ep.user, ep.n_way, ep.k_shot
        ys, cs, rs, yq, cq, rq = [], [], [], [], [], []
        for label, cls in enumerate(ep.classes):
            for r in ep.support[label]:
                ys.append(label); cs.append(cls); rs.append(r)
            for r in ep.query(label):
                yq.append(label); cq.append(cls); rq.append(r)
        self.y_support, self.cls_support, self.rep_support = map(np.asarray, (ys, cs, rs))
        self.y_query, self.cls_query, self.rep_query = map(np.asarray, (yq, cq, rq))


def zscore_rows(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-12)


def load_neural(spec: str) -> tuple:
    name, pattern = spec.split("=", 1)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"--neural {name}: no files match {pattern}")
    store = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            store[k] = z[k]
    return name, store


def collect(bank: DistanceBank, neural: dict, folds, max_per_user=None) -> pd.DataFrame:
    from episodes import episode_cache as ec
    rows = []
    for fold in folds:
        es = ec.load("test", fold)
        for ep in es.select(10, 1, max_per_user=max_per_user):
            ix = EpisodeIndex(ep)
            scores = {m: -bank.episode_distances(ix, m) for m in bank.methods}
            for name, store in neural.items():
                key = f"{ep.id}|logits"
                if key not in store:
                    raise KeyError(f"{name}: no logits for episode {ep.id}")
                if not np.array_equal(store[f"{ep.id}|labels"], ix.y_query):
                    raise ValueError(f"{name}: stored labels of {ep.id} do not match the cache's query order")
                scores[name] = store[key]
            for fname, parts in FUSIONS.items():
                if all(p in scores for p in parts):
                    scores[fname] = sum(zscore_rows(scores[p]) for p in parts)
            for name in neural:
                if "mdm" in scores:
                    scores[f"mdm+{name}"] = zscore_rows(scores["mdm"]) + zscore_rows(scores[name])
            rep_of = {int(l): int(r) for l, r in zip(ix.y_support, ix.rep_support)}
            for qi, (y, c, r) in enumerate(zip(ix.y_query, ix.cls_query, ix.rep_query)):
                row = {"fold": fold, "episode": ep.id, "user": ep.user, "label": int(y), "cls": int(c),
                       "rep": int(r), "rep_gap": abs(int(r) - rep_of[int(y)])}
                for m, s in scores.items():
                    sq = s[qi]
                    others = np.delete(sq, y)
                    row[f"correct_{m}"] = int(sq.argmax() == y)
                    row[f"margin_{m}"] = float((sq[y] - others.max()) / (sq.std() + 1e-12))
                rows.append(row)
    return pd.DataFrame(rows)


def user_mean(df: pd.DataFrame, col: str) -> float:
    return float(df.groupby("user")[col].mean().mean() * 100)


def complementarity(q: pd.DataFrame, methods) -> list:
    out = []
    for a, b in combinations(methods, 2):
        ca, cb = q[f"correct_{a}"], q[f"correct_{b}"]
        tmp = q[["user"]].assign(orr=(ca | cb), both_wrong=((1 - ca) & (1 - cb)))
        ea, eb = 1 - ca, 1 - cb
        phi = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else float("nan")
        out.append({"a": a, "b": b, "acc_a": user_mean(q, f"correct_{a}"), "acc_b": user_mean(q, f"correct_{b}"),
                    "oracle_or": user_mean(tmp, "orr"), "both_wrong": user_mean(tmp, "both_wrong"),
                    "error_phi": phi})
    return out


def by_label(ug: pd.DataFrame, methods) -> dict:
    out = {}
    for col in LABEL_COLS:
        if col not in ug:
            continue
        g = ug.groupby(col)
        out[col] = {str(k): {"n_cells": int(len(v)), **{m: float(v[f"acc_{m}"].mean() * 100) for m in methods}}
                    for k, v in g}
    return out


def delta_analysis(ug: pd.DataFrame, ref: str, others, n_perm: int = 200, seed: int = 0) -> dict:
    """Delta = acc(ref) - acc(X) per (user, gesture). Spearman with each continuous descriptor, and a
    leave-one-user-out ridge prediction of Delta from all descriptors (vs a within-user permutation null)."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    rng = np.random.default_rng(seed)
    X = ug[list(CONT_COLS)].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    users = ug["user"].to_numpy()
    out = {}

    def louo(y):
        pred = np.zeros_like(y)
        for u in np.unique(users):
            tr, te = users != u, users == u
            pred[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr]).predict(X[te])
        return spearmanr(pred, y).correlation

    for m in others:
        d = (ug[f"acc_{ref}"] - ug[f"acc_{m}"]).to_numpy(dtype=float) * 100
        res = {"mean_delta": float(d.mean()), "spearman": {}}
        for j, c in enumerate(CONT_COLS):
            r = spearmanr(X[:, j], d)
            res["spearman"][c] = {"rho": float(r.correlation), "p": float(r.pvalue)}
        obs = louo(d)
        null = []
        for _ in range(n_perm):
            dp = d.copy()
            for u in np.unique(users):            # shuffle Delta across gestures within each user
                idx = np.where(users == u)[0]
                dp[idx] = dp[rng.permutation(idx)]
            null.append(louo(dp))
        res["louo_rho"] = float(obs)
        res["louo_null_p95"] = float(np.quantile(null, 0.95))
        res["louo_perm_p"] = float((np.sum(np.asarray(null) >= obs) + 1) / (n_perm + 1))
        out[m] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--descriptors", default=None)
    ap.add_argument("--neural", action="append", default=[], help="NAME=GLOB of --dump-logits npz files")
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(7)))
    ap.add_argument("--max-per-user", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bank = DistanceBank(a.dist)
    neural = dict(load_neural(s) for s in a.neural)
    q = collect(bank, neural, a.folds, a.max_per_user)
    q.to_csv(out / "queries.csv", index=False)
    methods = [c[len("correct_"):] for c in q.columns if c.startswith("correct_")]
    base = [m for m in methods if m in bank.methods or m in neural]

    summary = {"n_queries": int(len(q)), "n_users": int(q["user"].nunique()), "folds": a.folds,
               "accuracy": {m: user_mean(q, f"correct_{m}") for m in methods},
               "accuracy_by_rep_gap": {m: q.groupby("rep_gap")[f"correct_{m}"].mean().mul(100).round(1).to_dict()
                                       for m in base},
               "complementarity": complementarity(q, base)}

    ug = q.groupby(["user", "cls"])[[f"correct_{m}" for m in methods]].mean().reset_index()
    ug.columns = ["user", "cls"] + [f"acc_{m}" for m in methods]
    if a.descriptors:
        desc = pd.read_csv(a.descriptors)
        desc["log_motion"] = np.log(desc["motion"])
        desc["laterality"] = (desc["imu_share_R"] - 0.5).abs()
        n0 = len(ug)
        ug = ug.merge(desc, on=["user", "cls"], how="left", validate="one_to_one")
        if ug["motion"].isna().any():
            raise ValueError(f"descriptors missing for {int(ug['motion'].isna().sum())} of {n0} (user, gesture) cells")
        summary["by_label"] = by_label(ug, base)
        summary["delta_vs_mdm"] = delta_analysis(ug, "mdm", [m for m in base if m != "mdm"], n_perm=a.n_perm)
    ug.to_csv(out / "user_gesture_acc.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{summary['n_users']} users, {summary['n_queries']} queries")
    for m, v in sorted(summary["accuracy"].items(), key=lambda kv: -kv[1]):
        print(f"  {m:24s} {v:5.1f}")
    print("complementarity with mdm (oracle-or / both-wrong / error phi):")
    for c in summary["complementarity"]:
        if "mdm" in (c["a"], c["b"]):
            print(f"  {c['a']:>10s} vs {c['b']:<10s} {c['oracle_or']:5.1f} {c['both_wrong']:5.1f} {c['error_phi']:.2f}")
    if "by_label" in summary:
        for col in ("movement_level", "emg_level_region", "side", "finger_like"):
            print(f"by {col}:")
            for k, v in summary["by_label"][col].items():
                print(f"  {k:10s} n={v['n_cells']:3d} " + " ".join(f"{m}={v[m]:.1f}" for m in base))
        print("Delta(mdm - X) predicted from descriptors, leave-one-user-out Spearman (perm p):")
        for m, v in summary["delta_vs_mdm"].items():
            print(f"  {m:12s} mean delta {v['mean_delta']:+5.1f}  rho {v['louo_rho']:+.2f} (null95 {v['louo_null_p95']:+.2f}, p {v['louo_perm_p']:.3f})")


if __name__ == "__main__":
    main()
