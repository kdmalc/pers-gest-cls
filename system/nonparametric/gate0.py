"""
gate0.py
========
The laptop pre-screen for a personalisation idea, before it gets any GPU time (docs/specs/SPEC_gate0.md).
Generalises region_headroom.py (the body-region Gate 0): dev users only, the cached 10-way 1-shot
episodes, balanced 1-NN L1 over the 28 channels (16 EMG, 12 IMU sensors), a matched random control,
pre-stated thresholds.

A **candidate** is a function that turns support-set information into per-channel weights for the 1-NN
distance (the distance is additive over channels, so any weighting is one matrix product):

    candidate(view, others, **kw) -> weights, shape (28,) or (n_ep, 28), non-negative

  view    the evaluated user's SUPPORT-ONLY data: {"s": (n_ep, ns, ns, 28) support-support channel
          distances, "ys": (n_ep, ns) support labels, "n_ep"}. Query distances and labels are not in it,
          so a candidate cannot peek at what it is scored on (the body-region "oracle" lesson).
  others  {user: full bundle} of the other dev users, usable as "training data" for reference statistics.

The **null** permutes the candidate's weights across channels per episode, so it keeps the exact
sparsity and scale and asks only: does it matter WHICH channels got the weight?

Rules (stated here, before any candidate runs):
  H1  mean gain over the uniform baseline >= 2 points, and at least 3 of the 4 dev users improve
  H2  the candidate beats the 95th percentile of its own permutation null
A candidate passes Gate 0 only if both hold. Passing earns a deep screen (P1 / M0 variant); failing
closes the idea at the data level.

    python system/nonparametric/gate0.py --candidate support_spread:m=14 --out results.json
    python system/nonparametric/gate0.py --candidate-path my_idea.py:weights --out results.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys

import numpy as np

N_CHANNELS = 28
EMG_CHANNELS = np.arange(16)
IMU_CHANNELS = np.arange(16, 28)
DEFAULT_DICT = "dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl"

RULES = dict(min_gain=0.02, min_users_frac=0.75, null_percentile=95.0)


# ── data ─────────────────────────────────────────────────────────────────────
def load_dev_bundles(n_way: int = 10, k_shot: int = 1, dict_path: str = DEFAULT_DICT) -> dict:
    """{user: {"q": (n_ep, nq, ns, 28), "s": (n_ep, ns, ns, 28), "ys": (n_ep, ns), "yq": (n_ep, nq)}}."""
    from episodes import episode_cache as ec
    from episodes.episode_numpy import materialize
    from MAML.maml_data_pipeline import reorient_tensor_dict
    from nonparametric.region_headroom import channel_distances
    with open(dict_path, "rb") as f:
        td = reorient_tensor_dict(pickle.load(f), {"emg_in_ch": 112, "imu_in_ch": 72})
    per: dict = {}
    for ep in ec.load("dev", None).select(n_way, k_shot):
        m = materialize(ep, td)
        b = per.setdefault(ep.user, {"q": [], "s": [], "ys": [], "yq": []})
        b["q"].append(channel_distances(m.emg_query, m.imu_query, m.emg_support, m.imu_support))
        b["s"].append(channel_distances(m.emg_support, m.imu_support, m.emg_support, m.imu_support))
        b["ys"].append(m.y_support)
        b["yq"].append(m.y_query)
    return {u: {k: np.stack(v) for k, v in b.items()} for u, b in per.items()}


def support_view(bundle: dict) -> dict:
    return {"s": bundle["s"], "ys": bundle["ys"], "n_ep": len(bundle["s"])}


# ── scoring ──────────────────────────────────────────────────────────────────
def as_episode_weights(w, n_ep: int) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        w = np.broadcast_to(w, (n_ep, N_CHANNELS))
    if w.shape != (n_ep, N_CHANNELS):
        raise ValueError(f"weights must be (28,) or ({n_ep}, 28), got {w.shape}")
    if (w < 0).any() or not (w.sum(axis=1) > 0).all():
        raise ValueError("weights must be non-negative with a positive sum in every episode")
    return w


def episode_accuracy(bundle: dict, w) -> np.ndarray:
    """Weighted 1-NN accuracy per episode, (n_ep,)."""
    q, ys, yq = bundle["q"], bundle["ys"], bundle["yq"]
    w = as_episode_weights(w, len(q))
    d = (q * w[:, None, None, :]).sum(axis=-1)                     # (n_ep, nq, ns)
    pred = np.take_along_axis(ys, d.argmin(axis=2), axis=1)         # (n_ep, nq)
    return (pred == yq).mean(axis=1)


def permute_channels(w: np.ndarray, rng) -> np.ndarray:
    """The matched null: each episode's weight vector permuted across channels."""
    return np.stack([row[rng.permutation(N_CHANNELS)] for row in w])


# ── built-in candidates (examples; the uniform one must score exactly the baseline) ──
def cand_uniform(view, others):
    return np.ones(N_CHANNELS)


def cand_emg_only(view, others):
    w = np.zeros(N_CHANNELS); w[EMG_CHANNELS] = 1.0
    return w


def cand_imu_only(view, others):
    w = np.zeros(N_CHANNELS); w[IMU_CHANNELS] = 1.0
    return w


def _spread(s: np.ndarray) -> np.ndarray:
    """(n_ep, ns, ns, 28) -> (n_ep, 28): mean off-diagonal support distance per channel."""
    ns = s.shape[1]
    return s.sum(axis=(1, 2)) / max(ns * (ns - 1), 1)


def cand_support_spread(view, others, m: int = 14):
    """Keep the m channels whose support spread is largest relative to the other users' mean spread.
    (The body-region selector, at channel level. It failed there; kept as the worked example.)"""
    ref = np.mean([_spread(b["s"]).mean(axis=0) for b in others.values()], axis=0)
    ratio = _spread(view["s"]) / np.maximum(ref, 1e-12)
    keep = np.argsort(-ratio, axis=1)[:, :m]
    w = np.zeros((view["n_ep"], N_CHANNELS))
    np.put_along_axis(w, keep, 1.0, axis=1)
    return w


CANDIDATES = {"uniform": cand_uniform, "emg_only": cand_emg_only, "imu_only": cand_imu_only,
              "support_spread": cand_support_spread}


# ── the gate ─────────────────────────────────────────────────────────────────
def run_gate0(data: dict, candidate, n_random: int = 200, rules: dict = RULES, seed: int = 0,
              name: str = "candidate", **kw) -> dict:
    users = sorted(data)
    rng = np.random.default_rng(seed)
    base, cand, weights = {}, {}, {}
    for u in users:
        others = {v: data[v] for v in users if v != u}
        w = as_episode_weights(candidate(support_view(data[u]), others, **kw), len(data[u]["q"]))
        weights[u] = w
        base[u] = float(episode_accuracy(data[u], np.ones(N_CHANNELS)).mean())
        cand[u] = float(episode_accuracy(data[u], w).mean())
    null = [float(np.mean([episode_accuracy(data[u], permute_channels(weights[u], rng)).mean() for u in users]))
            for _ in range(n_random)]

    mean = lambda d: float(np.mean(list(d.values())))
    gain = mean(cand) - mean(base)
    n_better = int(sum(cand[u] > base[u] for u in users))
    need = math.ceil(rules["min_users_frac"] * len(users))
    pct = float(np.percentile(null, rules["null_percentile"])) if null else float("nan")
    h1 = bool(gain >= rules["min_gain"] and n_better >= need)
    h2 = bool(null and mean(cand) > pct)
    res = {"candidate": name, "kwargs": kw, "users": users, "rules": rules,
           "n_episodes": {u: int(len(data[u]["q"])) for u in users},
           "baseline": base, "candidate_acc": cand, "gain": gain, "n_users_better": n_better,
           "null": null, "null_percentile_value": pct,
           "mean_nonzero_channels": float(np.mean([(weights[u] > 0).sum(axis=1).mean() for u in users])),
           "H1_gain": h1, "H2_beats_null": h2, "pass": bool(h1 and h2)}
    return res


def format_report(res: dict) -> str:
    users = res["users"]
    rows = [f"Gate 0 — {res['candidate']} {res['kwargs'] or ''}".rstrip(),
            f"  uniform 1-NN   {np.mean(list(res['baseline'].values()))*100:5.1f}  per user "
            f"{[round(res['baseline'][u]*100, 1) for u in users]}",
            f"  candidate      {np.mean(list(res['candidate_acc'].values()))*100:5.1f}  per user "
            f"{[round(res['candidate_acc'][u]*100, 1) for u in users]}  "
            f"({res['mean_nonzero_channels']:.1f} channels weighted on average)",
            f"  gain {res['gain']*100:+.1f} pts, {res['n_users_better']}/{len(users)} users better  -> H1 {res['H1_gain']}"]
    if res["null"]:
        rows.append(f"  permutation null: mean {np.mean(res['null'])*100:.1f}, "
                    f"p{res['rules']['null_percentile']:.0f} {res['null_percentile_value']*100:.1f}, "
                    f"max {max(res['null'])*100:.1f} (n={len(res['null'])})  -> H2 {res['H2_beats_null']}")
    rows.append(f"  PASS = {res['pass']}")
    return "\n".join(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_candidate(spec: str):
    """'name' or 'name:k=v,k2=v2' from CANDIDATES."""
    name, _, kws = spec.partition(":")
    if name not in CANDIDATES:
        raise SystemExit(f"unknown candidate {name!r}; built in: {sorted(CANDIDATES)}")
    kw = {}
    for item in filter(None, kws.split(",")):
        k, v = item.split("=")
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return name, CANDIDATES[name], kw


def _load_candidate_path(spec: str):
    """'path/to/file.py:function_name'."""
    path, _, fn = spec.rpartition(":")
    s = importlib.util.spec_from_file_location("gate0_candidate", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return f"{os.path.basename(path)}:{fn}", getattr(mod, fn), {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--candidate", help="built-in name, optionally name:k=v,...")
    g.add_argument("--candidate-path", help="file.py:function")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-way", type=int, default=10)
    ap.add_argument("--k-shot", type=int, default=1)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dict", default=DEFAULT_DICT)
    args = ap.parse_args()
    name, fn, kw = _parse_candidate(args.candidate) if args.candidate else _load_candidate_path(args.candidate_path)
    data = load_dev_bundles(args.n_way, args.k_shot, args.dict)
    res = run_gate0(data, fn, n_random=args.n_random, seed=args.seed, name=name, **kw)
    res["cell"] = [args.n_way, args.k_shot]
    print(format_report(res))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.path[:0] = ["system", "system/MAML"]
    os.environ.setdefault("CODE_DIR", os.getcwd())
    os.environ.setdefault("RUN_DIR", os.environ.get("TEMP", "/tmp"))
    os.environ.setdefault("HPO_CONFIG", "legacy")
    main()
