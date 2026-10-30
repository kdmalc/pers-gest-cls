"""
Gate 0 of the body-region line (docs/specs/SPEC_body_regions.md): how much accuracy is there to gain
from choosing body regions per user? Laptop, no training, dev users only.

Balanced-fusion 1-NN L1 (notebook 01) is additive across channels, so the distance of any channel
subset is an exact sum of per-channel distances and all 255 subsets of the 8 regions can be scored.

    python system/nonparametric/region_headroom.py --out <results.json>   (run from the repo root)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

EMG_FEATURES = 7
N_CHANNELS = 28                  # 16 EMG channels + 12 IMU sensors, the unit of a partition


def channel_distances(emg_a, imu_a, emg_b, imu_b) -> np.ndarray:
    """(na, C, T) arrays -> (na, nb, 28) balanced L1 distance per EMG channel and IMU sensor."""
    na, nb = len(emg_a), len(emg_b)
    emg_dim = emg_a.shape[1] * emg_a.shape[2]
    imu_dim = imu_a.shape[1] * imu_a.shape[2]
    out = np.zeros((na, nb, N_CHANNELS), dtype=np.float32)
    de = np.abs(emg_a[:, None] - emg_b[None]).sum(axis=3) / np.sqrt(emg_dim)    # (na, nb, 112)
    di = np.abs(imu_a[:, None] - imu_b[None]).sum(axis=3) / np.sqrt(imu_dim)    # (na, nb, 72)
    out[..., :16] = de.reshape(na, nb, 16, EMG_FEATURES).sum(axis=3)
    out[..., 16:] = di.reshape(na, nb, 12, 6).sum(axis=3)
    return out


def anatomical_groups() -> list:
    from arch_variants.body_regions import REGIONS
    return [list(r.emg) + [16 + s for s in r.imu] for r in REGIONS]


def random_groups(rng) -> list:
    """A random partition with the anatomical shape: six groups of 2 EMG + 2 IMU, two of 2 EMG."""
    emg, imu = rng.permutation(16), 16 + rng.permutation(12)
    groups = [[*emg[2 * g:2 * g + 2], *imu[2 * g:2 * g + 2]] for g in range(6)]
    groups += [[*emg[12:14]], [*emg[14:16]]]
    return [[int(c) for c in g] for g in groups]


def subset_masks(n_regions: int = 8) -> np.ndarray:
    """(2^n - 1, n) boolean; row i is the subset with bits i + 1. The last row is every region."""
    ids = np.arange(1, 2 ** n_regions)
    return ((ids[:, None] >> np.arange(n_regions)) & 1).astype(bool)


def to_regions(d_ch: np.ndarray, groups: list) -> np.ndarray:
    """(..., 28) -> (..., n_regions)."""
    return np.stack([d_ch[..., g].sum(axis=-1) for g in groups], axis=-1)


def subset_accuracy(d_reg, y_support, y_query, masks) -> np.ndarray:
    """d_reg (n_ep, nq, ns, R) -> (n_subsets, n_ep) accuracy per episode."""
    out = np.zeros((len(masks), len(d_reg)), dtype=np.float32)
    idx = np.arange(len(d_reg))[:, None]
    for i, m in enumerate(masks):
        pred = y_support[idx, d_reg[..., m].sum(axis=-1).argmin(axis=2)]
        out[i] = (pred == y_query).mean(axis=1)
    return out


def cv_oracle(acc, allowed=None) -> float:
    """acc (n_subsets, n_ep): pick on odd episodes, score on even, and the reverse."""
    allowed = np.arange(len(acc)) if allowed is None else allowed
    a, b = acc[allowed][:, ::2], acc[allowed][:, 1::2]
    return float((b[a.mean(axis=1).argmax()].mean() + a[b.mean(axis=1).argmax()].mean()) / 2)


def leave_one_user_out_global(accs: dict) -> dict:
    """One subset for everyone, picked on the other users. -> {user: accuracy}."""
    users = list(accs)
    out = {}
    for u in users:
        pooled = np.mean([accs[v].mean(axis=1) for v in users if v != u], axis=0)
        out[u] = float(accs[u][pooled.argmax()].mean())
    return out


def support_selector(d_reg, spreads, ref, y_support, y_query, m: int) -> np.ndarray:
    """
    Per episode, keep the m regions whose support spread is largest relative to `ref` (the same
    region's spread in other users). Uses the support set only. -> accuracy per episode (n_ep,).
    """
    keep = np.argsort(-(spreads / ref), axis=1)[:, :m]
    out = np.zeros(len(d_reg), dtype=np.float32)
    for e in range(len(d_reg)):
        pred = y_support[e][d_reg[e][..., keep[e]].sum(axis=-1).argmin(axis=1)]
        out[e] = (pred == y_query[e]).mean()
    return out


def _load_dev(n_way: int, k_shot: int):
    from episodes import episode_cache as ec
    from episodes.episode_numpy import materialize
    from MAML.maml_data_pipeline import reorient_tensor_dict
    d = pickle.load(open("dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl", "rb"))
    td = reorient_tensor_dict(d, {"emg_in_ch": 112, "imu_in_ch": 72})
    per = {}
    for ep in ec.load("dev", None).select(n_way, k_shot):
        m = materialize(ep, td)
        b = per.setdefault(ep.user, {"q": [], "s": [], "ys": [], "yq": []})
        b["q"].append(channel_distances(m.emg_query, m.imu_query, m.emg_support, m.imu_support))
        b["s"].append(channel_distances(m.emg_support, m.imu_support, m.emg_support, m.imu_support))
        b["ys"].append(m.y_support)
        b["yq"].append(m.y_query)
    return {u: {k: np.stack(v) for k, v in b.items()} for u, b in per.items()}


def _spread(d_ss_reg) -> np.ndarray:
    """(n_ep, ns, ns, R) -> (n_ep, R): mean off-diagonal distance among support samples."""
    ns = d_ss_reg.shape[1]
    return d_ss_reg.sum(axis=(1, 2)) / (ns * (ns - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-way", type=int, default=10)
    ap.add_argument("--k-shot", type=int, default=1)
    ap.add_argument("--n-random", type=int, default=200)
    args = ap.parse_args()

    from arch_variants.body_regions import REGION_NAMES
    data = _load_dev(args.n_way, args.k_shot)
    users = sorted(data)
    masks = subset_masks(8)
    every = len(masks) - 1
    sizes = masks.sum(axis=1)
    single = [int(np.flatnonzero((sizes == 1) & masks[:, r])[0]) for r in range(8)]

    def score(groups):
        return {u: subset_accuracy(to_regions(data[u]["q"], groups), data[u]["ys"], data[u]["yq"], masks)
                for u in users}

    groups = anatomical_groups()
    accs = score(groups)
    res = {"users": users, "cell": [args.n_way, args.k_shot], "regions": list(REGION_NAMES),
           "n_episodes": {u: len(data[u]["q"]) for u in users}}

    res["all_channels"] = {u: float(accs[u][every].mean()) for u in users}
    res["single_region"] = {REGION_NAMES[r]: {u: float(accs[u][single[r]].mean()) for u in users} for r in range(8)}
    res["oracle_in_sample"] = {u: float(accs[u].mean(axis=1).max()) for u in users}
    res["oracle_in_sample_subset"] = {
        u: [REGION_NAMES[r] for r in np.flatnonzero(masks[int(accs[u].mean(axis=1).argmax())])] for u in users}
    res["cv_oracle"] = {u: cv_oracle(accs[u]) for u in users}
    res["global_loo"] = leave_one_user_out_global(accs)
    res["budget_cv_oracle"] = {m: {u: cv_oracle(accs[u], np.flatnonzero(sizes <= m)) for u in users}
                               for m in range(1, 9)}

    spreads = {u: _spread(to_regions(data[u]["s"], groups)) for u in users}
    sel = {}
    for m in range(1, 9):
        sel[m] = {}
        for u in users:
            ref = np.mean([spreads[v].mean(axis=0) for v in users if v != u], axis=0)
            sel[m][u] = float(support_selector(to_regions(data[u]["q"], groups), spreads[u], ref,
                                               data[u]["ys"], data[u]["yq"], m).mean())
    res["support_selector"] = sel

    rng = np.random.default_rng(0)
    null = []
    for i in range(args.n_random):
        a = score(random_groups(rng))
        null.append(float(np.mean([cv_oracle(a[u]) for u in users])))
    res["random_cv_oracle"] = null

    mean = lambda d: float(np.mean(list(d.values())))
    base, cv = mean(res["all_channels"]), mean(res["cv_oracle"])
    n_better = sum(res["cv_oracle"][u] > res["all_channels"][u] for u in users)
    p95 = float(np.percentile(null, 95))
    h3 = mean(res["budget_cv_oracle"][4])
    h4 = max(mean(sel[m]) for m in range(1, 5))
    res["rules"] = {
        "H1_accuracy": bool(cv - base >= 0.02 and n_better >= 3),
        "H2_anatomy": bool(cv > p95),
        "H3_reduction": bool(h3 >= base - 0.01),
        "H4_selector": bool(h4 >= base - 0.02),
    }
    print(f"all-channel {base*100:.1f} | per user {[round(res['all_channels'][u]*100, 1) for u in users]}")
    print(f"in-sample oracle {mean(res['oracle_in_sample'])*100:.1f} | CV oracle {cv*100:.1f} "
          f"({n_better}/{len(users)} users better) | global LOO {mean(res['global_loo'])*100:.1f}")
    print(f"random CV oracle mean {np.mean(null)*100:.1f}, p95 {p95*100:.1f}, max {max(null)*100:.1f}")
    print("single region:", {k: round(mean(v) * 100, 1) for k, v in res["single_region"].items()})
    print("budget CV oracle:", {m: round(mean(v) * 100, 1) for m, v in res["budget_cv_oracle"].items()})
    print("support selector:", {m: round(mean(v) * 100, 1) for m, v in sel.items()})
    print("in-sample subsets:", res["oracle_in_sample_subset"])
    print("rules:", res["rules"])
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.path[:0] = ["system", "system/MAML"]
    os.environ.setdefault("CODE_DIR", os.getcwd())
    os.environ.setdefault("RUN_DIR", os.environ.get("TEMP", "/tmp"))
    os.environ.setdefault("HPO_CONFIG", "legacy")
    main()
