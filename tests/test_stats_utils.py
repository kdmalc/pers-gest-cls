"""Statistics + aggregator (docs/SPEC_stats.md) on synthetic results with known answers."""
import json

import numpy as np
import pytest
from scipy import stats

from analysis import aggregate as agg
from analysis import stats_utils as su
from episodes import episode_cache as ec


# ─────────────────────────────────────────────────────────────────────────────
# stats_utils
# ─────────────────────────────────────────────────────────────────────────────
def test_per_user_acc_pools_episodes_per_user():
    rows = [{"user": "a", "n_correct": 3, "n_query": 9}, {"user": "a", "n_correct": 6, "n_query": 9},
            {"user": "b", "n_correct": 0, "n_query": 9}]
    assert su.per_user_acc(rows) == {"a": 0.5, "b": 0.0}
    with pytest.raises(ValueError):
        su.per_user_acc([])


def test_bootstrap_ci_covers_true_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    hits = 0
    for t in range(200):
        x = rng.normal(0.6, 0.1, size=28)
        m, lo, hi = su.bootstrap_ci(x, n_boot=500, seed=t)
        hits += lo <= 0.6 <= hi
    assert 0.88 <= hits / 200 <= 0.99          # nominal 95%, small-n percentile bootstrap
    x = rng.normal(size=28)
    assert su.bootstrap_ci(x, seed=3) == su.bootstrap_ci(x, seed=3)


def test_paired_compare_matches_scipy_and_reports_wtl():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.4, 0.8, 28)
    a = b + rng.normal(0.03, 0.02, 28)
    r = su.paired_compare(a, b)
    assert r["p"] == pytest.approx(stats.wilcoxon(a, b, zero_method="pratt").pvalue)
    assert r["wins"] + r["ties"] + r["losses"] == 28
    assert r["ci_lo"] < r["mean_diff"] < r["ci_hi"]
    assert r["d_z"] == pytest.approx(r["mean_diff"] / r["sd_diff"])


def test_paired_compare_identical_inputs_gives_p_one():
    x = np.linspace(0.1, 0.9, 28)
    r = su.paired_compare(x, x)
    assert r["p"] == 1.0 and r["ties"] == 28 and r["d_z"] == 0.0


def test_paired_compare_rejects_bad_shapes():
    with pytest.raises(ValueError):
        su.paired_compare([1, 2, 3], [1, 2])


def test_holm_known_example_and_monotone():
    adj = su.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert su.holm([0.5]) == [0.5]
    adj2 = su.holm(np.random.default_rng(2).uniform(size=20))
    order = np.argsort(np.random.default_rng(2).uniform(size=20))
    assert all(adj2[order[i]] <= adj2[order[i + 1]] + 1e-12 for i in range(19))


def test_cluster_bootstrap_requires_same_users_and_brackets_mean():
    users = [f"u{i}" for i in range(28)]
    per_user = {u: 0.5 + 0.01 * (i % 7) for i, u in enumerate(users)}
    fold_of = {u: i % 7 for i, u in enumerate(users)}
    m, lo, hi = su.cluster_bootstrap(per_user, fold_of, n_boot=500)
    assert lo <= m <= hi
    with pytest.raises(ValueError):
        su.cluster_bootstrap(per_user, {u: 0 for u in users[:-1]})


def test_mde_matches_textbook_dz_for_n28():
    d_z = su.mde(1.0, n=28)
    assert 0.52 < d_z < 0.58                    # ~0.55 for n=28, alpha .05, power .8
    assert su.mde(0.05, n=28) == pytest.approx(0.05 * d_z)


# ─────────────────────────────────────────────────────────────────────────────
# aggregator
# ─────────────────────────────────────────────────────────────────────────────
def _write_results(path, method, bias, seed=0, drop_ids=0, tamper_hash=False, cell=(3, 1)):
    fold_results = []
    for k in range(7):
        es = ec.load("test", k)
        eps = es.select(*cell)
        rows = []
        for i, e in enumerate(eps):
            nq = e.n_way * e.n_query_per_class
            rows.append({"id": e.id, "user": e.user, "n_way": e.n_way, "k_shot": e.k_shot,
                         "n_query": nq, "n_correct": min(nq, int(nq * bias) + (i % 2))})
        if drop_ids and k == 3:
            rows = rows[drop_ids:]
        src = es.source_block()
        if tamper_hash:
            src = dict(src, sha256="0" * 64)
        fold_results.append({"seed": seed, "test_results": {"episode_source": src, "per_episode": rows}})
    path.write_text(json.dumps({"ablation_id": method, "smoke": False, "fold_results": fold_results}))
    return path


def test_aggregate_builds_per_user_tables_and_comparisons(tmp_path):
    pa = _write_results(tmp_path / "a.json", "MA", 0.80)
    pb = _write_results(tmp_path / "b.json", "MB", 0.55)
    a = agg.aggregate([pa, pb])
    assert set(a) == {"MA/test_results", "MB/test_results"}
    users = a["MA/test_results"][(3, 1)]
    assert len(users) == 28
    table = agg.summary_table(a, (3, 1), n_boot=200)
    assert table[0]["method"] == "MA/test_results" and table[0]["mean"] > table[1]["mean"]
    cmp = agg.compare_to(a, (3, 1), "MB/test_results", ["MA/test_results"], n_boot=200)
    assert cmp[0]["mean_diff"] > 0.2 and cmp[0]["wins"] == 28 and cmp[0]["p_holm"] <= 1.0


def test_aggregate_averages_seeds(tmp_path):
    p0 = _write_results(tmp_path / "s0.json", "MA", 0.80, seed=0)
    p1 = _write_results(tmp_path / "s1.json", "MA", 0.60, seed=1)
    a = agg.aggregate([p0, p1])["MA/test_results"][(3, 1)]
    single = agg.aggregate([p0])["MA/test_results"][(3, 1)]
    assert np.mean(list(a.values())) < np.mean(list(single.values()))


def test_aggregate_rejects_hash_mismatch(tmp_path):
    p = _write_results(tmp_path / "t.json", "MA", 0.7, tamper_hash=True)
    with pytest.raises(ValueError, match="episode hash"):
        agg.aggregate([p])


def test_aggregate_rejects_missing_or_extra_episode_ids(tmp_path):
    pa = _write_results(tmp_path / "a.json", "MA", 0.8)
    pb = _write_results(tmp_path / "b.json", "MB", 0.6, drop_ids=2)
    with pytest.raises(ValueError, match="id sets differ"):
        agg.aggregate([pa, pb])


def test_aggregate_allows_declared_subset_methods(tmp_path):
    pa = _write_results(tmp_path / "a.json", "MA", 0.8)
    pb = _write_results(tmp_path / "b.json", "MB", 0.6, drop_ids=2)
    out = agg.aggregate([pa, pb], subset_ok={"MB/test_results"})
    assert len(out["MB/test_results"][(3, 1)]) == 28


def test_aggregate_rejects_missing_users(tmp_path):
    obj = json.loads(_write_results(tmp_path / "a.json", "MA", 0.8).read_text())
    obj["fold_results"] = obj["fold_results"][:6]
    (tmp_path / "short.json").write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="users missing"):
        agg.aggregate([tmp_path / "short.json"], expect_folds=range(6))


def test_collect_blocks_requires_contract(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"ablation_id": "X", "smoke": False, "test_results": {"mean_acc": 0.5}}))
    with pytest.raises(ValueError, match="no block"):
        agg.collect_blocks(p)


def test_aggregate_rejects_smoke_results(tmp_path):
    p = _write_results(tmp_path / "s.json", "MA", 0.8)
    obj = json.loads(p.read_text())
    obj["smoke"] = True
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="SMOKE"):
        agg.aggregate([p])
