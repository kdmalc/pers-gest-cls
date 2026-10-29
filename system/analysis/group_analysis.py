"""
Per-ability-group analysis of the MoE gain (docs/SPEC_group_analysis.md). numpy + scipy only.

Question: does the gain of the MoE model over its no-MoE twin (default M0 - A4) differ across the
three ability groups? Unit of analysis is the USER; groups are small (5/9/14 test users), so
everything is reported as effect sizes with bootstrap intervals, never as a p-value claim.

Input is the output of `aggregate.aggregate()`: {method: {cell: {user: acc}}} (accuracies in
[0, 1], per-user, already averaged over train seeds). Gains are reported in accuracy points
(x100) by `rows_for_table`; the functions here work in the input units.
"""
import numpy as np
from scipy import stats

from analysis import stats_utils as su

GROUP_ORDER = ("Non-disabled Kapanji", "Disabled Kapanji", "Disabled Non-Kapanji")


# ─────────────────────────────────────────────────────────────────────────────
def load_groups(split=None):
    """-> (groups, fold_of_user). Groups are restricted to users that are TESTED in some fold,
    so the HPO-only users never enter. `split` defaults to the committed 28/4 split."""
    if split is None:
        from episodes import episode_cache as ec
        split = ec.kfold_split.load_split()
    fold_of_user = {u: f["fold"] for f in split["folds"] for u in f["test"]}
    groups = {g: sorted(u for u in split["groups"][g] if u in fold_of_user) for g in GROUP_ORDER}
    seen = [u for g in groups.values() for u in g]
    if len(seen) != len(set(seen)):
        raise ValueError("a user appears in two ability groups")
    if set(seen) != set(fold_of_user):
        raise ValueError(f"groups do not cover the tested users: "
                         f"{sorted(set(fold_of_user) - set(seen))}")
    return groups, fold_of_user


def _hedges_dz(d) -> float:
    """Paired standardised effect mean/sd with the small-sample (Hedges) correction."""
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() == 0 else float("inf") * np.sign(d.mean())
    n = d.size
    return float(d.mean() / sd * (1 - 3 / (4 * (n - 1) - 1)))


def user_gains(agg, cell, treat="M0", ref="A4", users=None) -> dict:
    """{user: acc_treat - acc_ref}. Raises if either method lacks a requested user."""
    for m in (treat, ref):
        if m not in agg or cell not in agg[m]:
            raise KeyError(f"{m!r} has no results for cell {cell}")
    users = sorted(set(agg[treat][cell]) & set(agg[ref][cell])) if users is None else list(users)
    for u in users:
        for m in (treat, ref):
            if u not in agg[m][cell]:
                raise ValueError(f"user {u} missing for {m} at cell {cell}")
    return {u: agg[treat][cell][u] - agg[ref][cell][u] for u in users}


def group_gain(agg, cell, treat="M0", ref="A4", split=None, n_boot=10000, seed=0) -> dict:
    """Per-group and overall gain summary for one cell.

    -> {"groups": {g: {n, mean_gain, ci_lo, ci_hi, d_z, sd_gain, mde, ref_mean, ref_sd_users,
                       wins, ties, losses}},
        "overall": {...same keys...}, "gain": {user: gain}, "groups_users": {g: [users]},
        "fold_of_user": {...}}"""
    groups, fold_of_user = load_groups(split)
    gain = user_gains(agg, cell, treat, ref, users=sorted(fold_of_user))
    ref_acc = {u: agg[ref][cell][u] for u in gain}

    def summ(users):
        d = np.array([gain[u] for u in users])
        r = np.array([ref_acc[u] for u in users])
        mean, lo, hi = su.bootstrap_ci(d, n_boot=n_boot, seed=seed)
        sd = float(d.std(ddof=1))
        return {"n": len(users), "mean_gain": mean, "ci_lo": lo, "ci_hi": hi,
                "d_z": _hedges_dz(d), "sd_gain": sd,
                "mde": su.mde(sd, n=len(users)) if sd > 0 else 0.0,
                "ref_mean": float(r.mean()), "ref_sd_users": float(r.std(ddof=1)),
                "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                "losses": int((d < 0).sum())}

    return {"groups": {g: summ(us) for g, us in groups.items()},
            "overall": summ(sorted(gain)), "gain": gain, "groups_users": groups,
            "fold_of_user": fold_of_user}


def group_contrast(res, g_a, g_b, n_boot=10000, seed=0, alpha=0.05):
    """Difference of mean gain between two groups (a - b) with a stratified bootstrap CI:
    users are resampled with replacement inside each group independently."""
    a = np.array([res["gain"][u] for u in res["groups_users"][g_a]])
    b = np.array([res["gain"][u] for u in res["groups_users"][g_b]])
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    lo, hi = np.quantile(ma - mb, [alpha / 2, 1 - alpha / 2])
    return {"a": g_a, "b": g_b, "diff": float(a.mean() - b.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi)}


def all_group_contrasts(res, **kw) -> list:
    g = [x for x in GROUP_ORDER if x in res["groups_users"]]
    return [group_contrast(res, g[i], g[j], **kw) for i in range(len(g)) for j in range(i + 1, len(g))]


def difficulty_gain_corr(res, agg, cell, ref="A4", n_boot=10000, seed=0, alpha=0.05) -> dict:
    """Group-agnostic check: Spearman correlation between a user's no-MoE accuracy ("difficulty")
    and their gain, over all tested users, with a user-bootstrap CI. Negative rho = the gain is
    larger for users the baseline finds hard."""
    users = sorted(res["gain"])
    x = np.array([agg[ref][cell][u] for u in users])
    y = np.array([res["gain"][u] for u in users])
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        i = rng.integers(0, len(users), size=len(users))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        rhos.append(stats.spearmanr(x[i], y[i]).statistic)
    lo, hi = np.quantile(rhos, [alpha / 2, 1 - alpha / 2])
    return {"rho": rho, "ci_lo": float(lo), "ci_hi": float(hi), "n_users": len(users)}


def fold_cluster_ci(res, n_boot=10000, seed=0) -> tuple:
    """Sensitivity: CI of the overall mean gain resampling FOLDS (users in a fold share a model)."""
    return su.cluster_bootstrap(res["gain"], res["fold_of_user"], n_boot=n_boot, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
def rows_for_table(res, scale=100.0) -> list:
    """One dict per group (+ overall) with gains in points; ready for csv / markdown."""
    rows = []
    for name, s in list(res["groups"].items()) + [("All tested users", res["overall"])]:
        rows.append({"group": name, "n_users": s["n"],
                     "ref_acc": s["ref_mean"] * scale, "ref_sd_users": s["ref_sd_users"] * scale,
                     "gain": s["mean_gain"] * scale, "ci_lo": s["ci_lo"] * scale,
                     "ci_hi": s["ci_hi"] * scale, "d_z": s["d_z"], "mde": s["mde"] * scale,
                     "W/T/L": f"{s['wins']}/{s['ties']}/{s['losses']}"})
    return rows


def to_markdown(rows) -> str:
    head = ["group", "n_users", "ref_acc", "ref_sd_users", "gain", "ci_lo", "ci_hi", "d_z", "mde",
            "W/T/L"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(r[h] if isinstance(r[h], str) else
                                     (str(r[h]) if h == "n_users" else f"{r[h]:.2f}")
                                     for h in head) + " |")
    return "\n".join(out)
