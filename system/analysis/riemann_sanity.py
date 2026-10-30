"""
riemann_sanity.py
=================
Sanity checks for the Riemannian MDM baseline (91.4% at 10-way 1-shot on fs64, 28 users, 2026-10-30), run
on the same cached test episodes (all 7 folds) the cluster used:

  (a) fused       the baseline as run on the cluster (EMG + IMU MDM distances summed): must reproduce
  (b) shuffled    support labels permuted within each episode: must fall to chance (10%), else leakage
  (c) emg_only    MDM on the EMG block only
  (d) imu_only    MDM on the IMU block only
  (e) rep gap     accuracy of (a) as a function of |rep(query) - rep(support of the true class)|. The 10 reps
                  of a gesture were recorded back to back (4-7 s apart), so a steep fall with the gap would mean
                  part of the accuracy is slow drift within a recording block, not gesture shape.

    python system/analysis/riemann_sanity.py --emg-repr fs64 --per-user 50 --out <json>
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for rel in ("system", "system/MAML"):
    sys.path.insert(0, str(REPO / rel))

DICTS = {"env20": ("segfilt_rts_tensor_dict.pkl", 16), "fs64": ("segfilt_fs64_tensor_dict.pkl", 112)}


def _covs(x):
    from pyriemann.estimation import Covariances
    return Covariances(estimator="oas").fit_transform(x.astype(np.float64))


def mdm_dist(sup, y_sup, qry):
    """(n_query, n_classes) Riemannian distances to each class mean, columns ordered by np.unique(y_sup)."""
    from pyriemann.classification import MDM
    m = MDM(metric="riemann").fit(_covs(sup), y_sup)
    return m.transform(_covs(qry)), m.classes_


def run(emg_repr: str, per_user: int, seed: int = 0):
    from episodes import episode_cache as ec
    from episodes.episode_numpy import materialize
    from MAML.maml_data_pipeline import reorient_tensor_dict
    name, ch = DICTS[emg_repr]
    d = pickle.load(open(REPO / "dataset" / "meta-learning-sup-que-ds" / name, "rb"))
    td = reorient_tensor_dict(d, {"emg_in_ch": ch, "imu_in_ch": 72})
    rng = np.random.default_rng(seed)

    acc = defaultdict(lambda: defaultdict(list))        # condition -> user -> per-episode acc
    gap_hits = defaultdict(lambda: [0, 0])              # |rep gap| -> [correct, total] for (a)
    t0 = time.time()
    n_ep = 0
    for fold in range(7):
        es = ec.load("test", fold)
        for ep in es.select(10, 1, max_per_user=per_user):
            m = materialize(ep, td)
            de, cls = mdm_dist(m.emg_support, m.y_support, m.emg_query)
            di, _ = mdm_dist(m.imu_support, m.y_support, m.imu_query)
            yq = m.y_query
            pred_f = cls[(de + di).argmin(1)]
            acc["fused"][m.user].append(float((pred_f == yq).mean()))
            acc["emg_only"][m.user].append(float((cls[de.argmin(1)] == yq).mean()))
            acc["imu_only"][m.user].append(float((cls[di.argmin(1)] == yq).mean()))
            ys = rng.permutation(m.y_support)
            ds, cs = mdm_dist(m.emg_support, ys, m.emg_query)
            dsi, _ = mdm_dist(m.imu_support, ys, m.imu_query)
            acc["shuffled"][m.user].append(float((cs[(ds + dsi).argmin(1)] == yq).mean()))
            rep_of_label = {int(l): int(r) for l, r in zip(m.y_support, m.rep_support)}
            for yt, rq, p in zip(yq, m.rep_query, pred_f):
                g = abs(int(rq) - rep_of_label[int(yt)])
                gap_hits[g][0] += int(p == yt)
                gap_hits[g][1] += 1
            n_ep += 1
            if n_ep % 200 == 0:
                print(f"  {emg_repr}: {n_ep} episodes, {time.time()-t0:.0f}s", flush=True)

    out = {"emg_repr": emg_repr, "per_user": per_user, "n_episodes": n_ep, "conditions": {}}
    for cond, by_user in acc.items():
        pu = np.array([np.mean(v) for v in by_user.values()])
        out["conditions"][cond] = {"mean": float(pu.mean() * 100), "sd_users": float(pu.std() * 100),
                                   "min_user": float(pu.min() * 100), "n_users": int(pu.size)}
    out["rep_gap"] = {int(g): {"acc": c / t * 100, "n": t} for g, (c, t) in sorted(gap_hits.items())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emg-repr", choices=sorted(DICTS), default="fs64")
    ap.add_argument("--per-user", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("CODE_DIR", str(REPO))
    res = run(a.emg_repr, a.per_user)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for c, v in res["conditions"].items():
        print(f"{a.emg_repr} {c:9s}: {v['mean']:.1f} +/- {v['sd_users']:.1f} (worst user {v['min_user']:.1f}, n={v['n_users']})")
    print("accuracy vs |rep gap|:", {g: round(v["acc"], 1) for g, v in res["rep_gap"].items()})
