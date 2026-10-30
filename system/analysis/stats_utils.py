"""
Statistics for per-user results (docs/specs/SPEC_stats.md). numpy + scipy only.

The unit of analysis is the USER. Episodes are never treated as independent samples.
"""
from collections import defaultdict

import numpy as np
from scipy import stats


def per_user_acc(rows) -> dict:
    """rows: per-episode dicts with user, n_correct, n_query. -> {user: pooled-over-episodes acc}.
    All episodes of a cell share Q, so this equals the mean of the episode accuracies."""
    correct, total = defaultdict(int), defaultdict(int)
    for r in rows:
        correct[r["user"]] += r["n_correct"]
        total[r["user"]] += r["n_query"]
    if not total:
        raise ValueError("no rows")
    return {u: correct[u] / total[u] for u in sorted(total)}


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean, resampling USERS with replacement."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("need a 1-d array of at least 2 users")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_compare(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired per-user comparison of A vs B (same user order). Two-sided Wilcoxon (Pratt)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError(f"need equal-length 1-d arrays (>=2), got {a.shape} {b.shape}")
    d = a - b
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    if np.all(d == 0):
        p = 1.0          # defined case: no difference at all (scipy raises on all-zero input)
    else:
        p = float(stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided").pvalue)
    sd = float(d.std(ddof=1))
    return {
        "n_users": int(d.size), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "p": p,
        "d_z": float(mean / sd) if sd > 0 else float("inf") if mean > 0 else
               float("-inf") if mean < 0 else 0.0,
        "sd_diff": sd, "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
    }


def holm(pvals) -> list:
    """Holm-Bonferroni adjusted p-values (monotone), returned in the input order."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def cluster_bootstrap(per_user: dict, fold_of_user: dict, n_boot: int = 10000,
                      seed: int = 0, alpha: float = 0.05):
    """Sensitivity CI of the 28-user mean resampling FOLDS (users in a fold share one model)."""
    if set(per_user) != set(fold_of_user):
        raise ValueError("per_user and fold_of_user must cover the same users")
    folds = sorted(set(fold_of_user.values()))
    by_fold = {f: np.array([per_user[u] for u in per_user if fold_of_user[u] == f]) for f in folds}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(folds), size=len(folds))
        means[i] = np.concatenate([by_fold[folds[j]] for j in pick]).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(list(per_user.values()))), float(lo), float(hi)


def mde(sd_diff: float, n: int = 28, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean paired difference (same units as sd_diff), paired-t power."""
    from scipy.optimize import brentq
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    def pw(d_z):
        nc = d_z * np.sqrt(n)
        return (1 - stats.nct.cdf(t_crit, df=n - 1, nc=nc)) + stats.nct.cdf(-t_crit, df=n - 1, nc=nc)

    d_z = brentq(lambda d: pw(d) - power, 1e-3, 5.0)
    return float(d_z * sd_diff)
