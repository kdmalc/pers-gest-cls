"""
trial_distances.py
==================
Within-user trial-to-trial distance matrices for the nearest-neighbour classical methods. At 1-shot
every one of these methods classifies a query by its distance to each class's single support trial, and
that distance depends on the two trials only, not on the episode. So one (100 x 100) matrix per user and
method gives every cached 1-shot episode's scores by lookup: the full 28-user analysis takes minutes
instead of the hours per-episode refitting takes (riemann_sanity.py: ~1.5 h for fs64 at 50 episodes/user).

Methods (each reproduces nonparametric/classical.py at K=1; tests/test_riemann_analysis.py checks):
  mdm       Riemannian (affine-invariant) distance of the OAS covariances, EMG + IMU summed (riemann_mdm)
  mdm_emg   EMG term only;  mdm_imu   IMU term only
  seg_mdm   the same on N_SEG time segments, summed over segments and modalities (riemann_seg_mdm)
  knn_l1    L1 on the flattened EMG+IMU tensor (knn_l1)
  td_knn    squared Euclidean on mean/std-over-time features (td_knn)
  dollar_b  `$B`: distance of the query to the TEMPLATE in the template's own PCA basis (dollar_b);
            asymmetric, D[template, query]

    python system/analysis/trial_distances.py --emg-repr fs64 --out <dir>/dist_fs64.npz

Output npz: for each user u, "u|index" ((n, 2) array of (class, rep)) and "u|<method>" (n, n) with
D[i, j] = distance from support/template trial i to query trial j.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for rel in ("system", "system/MAML"):
    if str(REPO / rel) not in sys.path:
        sys.path.insert(0, str(REPO / rel))

DICTS = {"env20": ("segfilt_rts_tensor_dict.pkl", 16), "fs64": ("segfilt_fs64_tensor_dict.pkl", 112)}
METHODS = ("mdm", "mdm_emg", "mdm_imu", "seg_mdm", "knn_l1", "td_knn", "dollar_b")


def load_tensor_dict(emg_repr: str) -> dict:
    from MAML.maml_data_pipeline import reorient_tensor_dict
    name, ch = DICTS[emg_repr]
    with open(REPO / "dataset" / "meta-learning-sup-que-ds" / name, "rb") as f:
        d = pickle.load(f)
    return reorient_tensor_dict(d, {"emg_in_ch": ch, "imu_in_ch": 72})


def user_trials(td_user: dict):
    """-> index (n, 2) of (class, rep), emg (n, Ce, T), imu (n, Ci, T); classes in sorted order."""
    idx, emg, imu = [], [], []
    for cls in sorted(td_user):
        slot = td_user[cls]
        for r in range(slot["emg"].shape[0]):
            idx.append((cls, r))
            emg.append(slot["emg"][r].numpy())
            imu.append(slot["imu"][r].numpy())
    return np.asarray(idx), np.stack(emg).astype(np.float64), np.stack(imu).astype(np.float64)


def oas_covs(x: np.ndarray) -> np.ndarray:
    from pyriemann.estimation import Covariances
    return Covariances(estimator="oas").fit_transform(x)


def airm_pairwise(covs: np.ndarray) -> np.ndarray:
    """(n, C, C) SPD -> (n, n) affine-invariant distances sqrt(sum log^2 eig(A^-1/2 B A^-1/2))."""
    S = torch.as_tensor(covs, dtype=torch.float64)
    w, V = torch.linalg.eigh(S)
    isq = (V * w.clamp_min(1e-12).rsqrt().unsqueeze(-2)) @ V.transpose(-1, -2)
    n = S.shape[0]
    D = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        M = isq[i] @ S @ isq[i]
        ev = torch.linalg.eigvalsh(0.5 * (M + M.transpose(-1, -2))).clamp_min(1e-12)
        D[i] = ev.log().pow(2).sum(-1).sqrt()
    D = 0.5 * (D + D.T)
    D.fill_diagonal_(0.0)
    return D.numpy()


def segment_edges(T: int, n_seg: int) -> np.ndarray:
    return np.linspace(0, T, n_seg + 1).round().astype(int)


def dollar_b_pairwise(x: np.ndarray, n_components: int = 50) -> np.ndarray:
    """x: (n, C, T) fused trials. D[i, j] = `$B` distance of trial j to template i (classical.dollar_b)."""
    X = torch.as_tensor(x, dtype=torch.float64)
    n, C, T = X.shape
    n_pc = min(n_components, C - 1)
    D = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        mean = X[i].mean(dim=1, keepdim=True)
        xc = X[i] - mean
        w, v = torch.linalg.eigh(xc @ xc.T / (T - 1))
        U = v[:, -n_pc:].flip(-1)                                      # (C, P), descending
        proj_t = (U.T @ xc).reshape(-1)
        proj = torch.einsum("cp,nct->npt", U, X - mean).reshape(n, -1)
        D[i] = (proj - proj_t).abs().sum(dim=1)
    return D.numpy()


def user_distances(td_user: dict, n_seg: int = 3, methods=METHODS) -> dict:
    idx, emg, imu = user_trials(td_user)
    out = {"index": idx}
    need_mdm = {"mdm", "mdm_emg", "mdm_imu"} & set(methods)
    if need_mdm:
        de, di = airm_pairwise(oas_covs(emg)), airm_pairwise(oas_covs(imu))
        out.update(mdm=de + di, mdm_emg=de, mdm_imu=di)
    if "seg_mdm" in methods:
        T = emg.shape[2]
        D = 0.0
        for a, b in zip(segment_edges(T, n_seg)[:-1], segment_edges(T, n_seg)[1:]):
            D = D + airm_pairwise(oas_covs(emg[:, :, a:b])) + airm_pairwise(oas_covs(imu[:, :, a:b]))
        out["seg_mdm"] = D
    fused = np.concatenate([emg, imu], axis=1)
    if "knn_l1" in methods:
        f = torch.as_tensor(fused.reshape(len(fused), -1))
        out["knn_l1"] = torch.cdist(f, f, p=1).numpy()
    if "td_knn" in methods:
        feat = np.concatenate([fused.mean(axis=2), fused.std(axis=2)], axis=1)
        f = torch.as_tensor(feat)
        out["td_knn"] = torch.cdist(f, f).pow(2).numpy()
    if "dollar_b" in methods:
        out["dollar_b"] = dollar_b_pairwise(fused)
    return {k: v for k, v in out.items() if k == "index" or k in methods}


def compute_all(emg_repr: str, users=None, methods=METHODS, log=print) -> dict:
    td = load_tensor_dict(emg_repr)
    flat = {}
    t0 = time.time()
    for u in sorted(td) if users is None else users:
        res = user_distances(td[u], methods=methods)
        for k, v in res.items():
            flat[f"{u}|{k}"] = v
        log(f"  {u}: done ({time.time() - t0:.0f}s)")
    return flat


class DistanceBank:
    """Lookup of 1-shot episode scores from a trial_distances npz."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.data = {k: z[k] for k in z.files}
        self.users = sorted({k.split("|")[0] for k in self.data})
        self.methods = sorted({k.split("|")[1] for k in self.data} - {"index"})
        self._pos = {}
        for u in self.users:
            self._pos[u] = {(int(c), int(r)): i for i, (c, r) in enumerate(self.data[f"{u}|index"])}

    def episode_distances(self, mep, method: str) -> np.ndarray:
        """(n_query, n_way) distances of each query to each class's support trial (K=1 only)."""
        if mep.k_shot != 1:
            raise ValueError("DistanceBank covers 1-shot episodes only (K>1 needs class means)")
        pos = self._pos[mep.user]
        D = self.data[f"{mep.user}|{method}"]
        s_idx = np.empty(mep.n_way, dtype=int)
        for lab, c, r in zip(mep.y_support, mep.cls_support, mep.rep_support):
            s_idx[int(lab)] = pos[(int(c), int(r))]
        q_idx = np.array([pos[(int(c), int(r))] for c, r in zip(mep.cls_query, mep.rep_query)])
        return D[np.ix_(s_idx, q_idx)].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emg-repr", choices=sorted(DICTS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    a = ap.parse_args()
    flat = compute_all(a.emg_repr, methods=a.methods)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **flat)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
