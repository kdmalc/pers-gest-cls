"""
cov_cluster_gate.py
===================
Gate before any covariance-plus-clustering claim (docs/specs/SPEC_riemannian.md section 2; rules there
were written before the first run). Laptop, no training, all 32 users of the tensor dict.

Per user and modality, the Riemannian mean of the user's trial covariances (OAS) is the user's
"covariance signature"; users are compared by the affine-invariant distance (EMG + IMU summed) and,
for k-means, by standardised log-Euclidean vectors reduced by PCA.

  G1 identifiability  signature from reps 0-4 vs reps 5-9: is a user's nearest other-half signature its own?
                      (rank-1 rate; chance 1/32)
  G2 cluster structure  best silhouette over K = 2..6 (k-means, PCA space) vs a Gaussian null with the
                      same mean and covariance (200 draws): observed > null 95th percentile
  G3 stability        adjusted Rand index between the clusterings of the two halves, at the best K
  G4 shared motifs    per (user, gesture) mean covariance after re-centring each user at its own mean:
                      is a user's gesture closer to other users' SAME referent (e.g. their "close") than
                      to their other referents? Cross-user nearest-referent rate vs chance 10%
                      (the covariance version of the ability-cluster study's H1)

    python system/analysis/cov_cluster_gate.py --emg-repr fs64 --out <dir>/cov_cluster_gate_fs64.json
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

from trial_distances import load_tensor_dict, user_trials, oas_covs  # noqa: E402

K_RANGE = range(2, 7)


def _mean(covs):
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(covs)


def _airm(a, b):
    from pyriemann.utils.distance import distance_riemann
    return float(distance_riemann(a, b))


def _logvec(m):
    from pyriemann.utils.base import logm
    from pyriemann.utils.tangentspace import upper
    return upper(logm(m)[None])[0]


def signatures(td, users, reps=None):
    """-> list of (emg_mean, imu_mean) per user over the given rep positions (None = all)."""
    out = []
    for u in users:
        idx, emg, imu = user_trials(td[u])
        keep = np.ones(len(idx), bool) if reps is None else np.isin(idx[:, 1], reps)
        out.append((_mean(oas_covs(emg[keep])), _mean(oas_covs(imu[keep]))))
    return out


def pca_space(vecs, var=0.9):
    from sklearn.decomposition import PCA
    X = np.asarray(vecs)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    p = PCA(n_components=min(len(X) - 1, X.shape[1])).fit(X)
    k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), var) + 1)
    return p.transform(X)[:, :k]


def best_silhouette(Z, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = (-1.0, None, None)
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)
        s = float(silhouette_score(Z, lab))
        if s > best[0]:
            best = (s, k, lab)
    return best


def run(emg_repr: str, n_null: int = 200, seed: int = 0) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    td = load_tensor_dict(emg_repr)
    users = sorted(td)
    rng = np.random.default_rng(seed)
    full = signatures(td, users)
    half_a = signatures(td, users, reps=list(range(5)))
    half_b = signatures(td, users, reps=list(range(5, 10)))

    # G1: identifiability
    n = len(users)
    D = np.array([[_airm(a[0], b[0]) + _airm(a[1], b[1]) for b in half_b] for a in half_a])
    rank1 = float(np.mean(D.argmin(axis=1) == np.arange(n)))

    # G2: cluster structure vs a Gaussian null
    Z = pca_space([np.concatenate([_logvec(e), _logvec(i)]) for e, i in full])
    sil, k_best, labels = best_silhouette(Z, seed)
    mu, cov = Z.mean(0), np.cov(Z.T)
    null = [best_silhouette(rng.multivariate_normal(mu, cov, size=n), seed)[0] for _ in range(n_null)]

    # G3: stability across halves at the best K
    za = pca_space([np.concatenate([_logvec(e), _logvec(i)]) for e, i in half_a])
    zb = pca_space([np.concatenate([_logvec(e), _logvec(i)]) for e, i in half_b])
    la = KMeans(n_clusters=k_best, n_init=20, random_state=seed).fit_predict(za)
    lb = KMeans(n_clusters=k_best, n_init=20, random_state=seed).fit_predict(zb)
    ari = float(adjusted_rand_score(la, lb))

    # G4: shared referent motifs across users after per-user re-centring
    from pyriemann.utils.base import invsqrtm
    gest = {}                                              # (user, cls) -> (emg, imu) re-centred means
    for u, (me, mi) in zip(users, full):
        idx, emg, imu = user_trials(td[u])
        ce, ci = oas_covs(emg), oas_covs(imu)
        we, wi = invsqrtm(me), invsqrtm(mi)
        for c in np.unique(idx[:, 0]):
            k = idx[:, 0] == c
            gest[(u, int(c))] = (we @ _mean(ce[k]) @ we, wi @ _mean(ci[k]) @ wi)
    keys = sorted(gest)
    hits, total = 0, 0
    per_user = {}
    for (u, c) in keys:
        others = [(v, d) for (v, d) in keys if v != u]
        dist = np.array([_airm(gest[(u, c)][0], gest[o][0]) + _airm(gest[(u, c)][1], gest[o][1]) for o in others])
        h = int(others[int(dist.argmin())][1] == c)
        hits += h; total += 1
        per_user.setdefault(u, []).append(h)
    g4 = hits / total

    return {"emg_repr": emg_repr, "n_users": n,
            "G1_identifiability_rank1": rank1, "G1_chance": 1.0 / n,
            "G2_silhouette": sil, "G2_best_k": int(k_best), "G2_null_p95": float(np.quantile(null, 0.95)),
            "G2_null_mean": float(np.mean(null)), "G2_pca_dims": int(Z.shape[1]),
            "G3_split_half_ari": ari,
            "G4_cross_user_same_referent_rate": g4, "G4_chance": 0.1,
            "G4_per_user": {u: float(np.mean(v)) for u, v in per_user.items()},
            "clusters": {u: int(l) for u, l in zip(users, labels)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emg-repr", required=True, choices=("env20", "fs64"))
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.emg_repr, a.n_null)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("clusters", "G4_per_user")}, indent=1))


if __name__ == "__main__":
    main()
