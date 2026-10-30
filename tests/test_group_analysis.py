"""Per-ability-group gain analysis (docs/specs/SPEC_group_analysis.md) on synthetic per-user results."""
import numpy as np
import pytest

from analysis import group_analysis as ga
from episodes import episode_cache as ec

CELL = (10, 1)


@pytest.fixture(scope="module")
def split():
    return ec.kfold_split.load_split()


def _agg(split, gain_by_group, noise=0.0, seed=0, ref_acc=0.6):
    """Synthetic aggregate(): M0 = A4 + planted per-group gain (+ noise)."""
    rng = np.random.default_rng(seed)
    groups, _ = ga.load_groups(split)
    a4, m0 = {}, {}
    for g, users in groups.items():
        for u in users:
            a4[u] = ref_acc + rng.normal(0, 0.05)
            m0[u] = a4[u] + gain_by_group[g] + rng.normal(0, noise)
    return {"A4": {CELL: a4}, "M0": {CELL: m0}}


def test_groups_cover_exactly_the_28_tested_users(split):
    groups, fold_of_user = ga.load_groups(split)
    assert [len(groups[g]) for g in ga.GROUP_ORDER] == [5, 9, 14]
    assert set(fold_of_user) == {u for g in groups.values() for u in g}
    assert not set(split["hpo_val"]) & set(fold_of_user)       # HPO-only users never enter


def test_planted_gain_in_one_group_is_recovered(split):
    gains = {"Non-disabled Kapanji": 0.0, "Disabled Kapanji": 0.0, "Disabled Non-Kapanji": 0.10}
    agg = _agg(split, gains, noise=0.01)
    res = ga.group_gain(agg, CELL, split=split, n_boot=2000)
    g = res["groups"]
    assert g["Disabled Non-Kapanji"]["mean_gain"] == pytest.approx(0.10, abs=0.02)
    assert g["Disabled Non-Kapanji"]["ci_lo"] > 0.05
    assert abs(g["Non-disabled Kapanji"]["mean_gain"]) < 0.03
    c = ga.group_contrast(res, "Disabled Non-Kapanji", "Non-disabled Kapanji", n_boot=2000)
    assert c["ci_lo"] > 0.03 and c["diff"] == pytest.approx(0.10, abs=0.03)


def test_flat_gain_gives_overlapping_cis_and_zero_contrast(split):
    agg = _agg(split, {g: 0.05 for g in ga.GROUP_ORDER}, noise=0.01, seed=3)
    res = ga.group_gain(agg, CELL, split=split, n_boot=2000)
    for c in ga.all_group_contrasts(res, n_boot=2000):
        assert c["ci_lo"] < 0 < c["ci_hi"]
    assert res["overall"]["mean_gain"] == pytest.approx(0.05, abs=0.01)
    assert res["overall"]["wins"] == 28


def test_missing_user_or_method_raises(split):
    agg = _agg(split, {g: 0.0 for g in ga.GROUP_ORDER})
    del agg["M0"][CELL]["P102"]
    with pytest.raises(ValueError, match="P102"):
        ga.group_gain(agg, CELL, split=split)
    with pytest.raises(KeyError):
        ga.group_gain(agg, (3, 1), split=split)


def test_extra_hpo_users_are_ignored(split):
    agg = _agg(split, {g: 0.02 for g in ga.GROUP_ORDER})
    for m in agg:
        agg[m][CELL]["P005"] = 0.9 if m == "M0" else 0.1       # HPO-only user: must not count
    res = ga.group_gain(agg, CELL, split=split, n_boot=500)
    assert res["overall"]["n"] == 28 and "P005" not in res["gain"]
    assert res["overall"]["mean_gain"] == pytest.approx(0.02, abs=1e-9)


def test_difficulty_gain_corr_detects_hard_user_benefit(split):
    groups, _ = ga.load_groups(split)
    rng = np.random.default_rng(0)
    users = sorted(u for g in groups.values() for u in g)
    a4 = {u: float(rng.uniform(0.3, 0.8)) for u in users}
    m0 = {u: a4[u] + 0.3 * (0.8 - a4[u]) for u in users}       # gain shrinks as baseline improves
    agg = {"A4": {CELL: a4}, "M0": {CELL: m0}}
    res = ga.group_gain(agg, CELL, split=split, n_boot=500)
    r = ga.difficulty_gain_corr(res, agg, CELL, n_boot=500)
    assert r["rho"] == pytest.approx(-1.0) and r["ci_hi"] < -0.9


def test_table_rows_are_in_points_and_render(split):
    agg = _agg(split, {g: 0.04 for g in ga.GROUP_ORDER})
    res = ga.group_gain(agg, CELL, split=split, n_boot=500)
    rows = ga.rows_for_table(res)
    assert [r["group"] for r in rows][-1] == "All tested users" and len(rows) == 4
    assert rows[-1]["gain"] == pytest.approx(4.0, abs=1e-6)
    assert ga.to_markdown(rows).count("\n") == 5
    assert isinstance(ga.fold_cluster_ci(res, n_boot=200)[0], float)
