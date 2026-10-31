"""The Gate 0 harness (nonparametric/gate0.py) on synthetic channel distances."""
import numpy as np
import pytest

from nonparametric import gate0 as g0


def _bundle(rng, n_ep=12, n_way=4, nq=3, signal_channel=None, signal=3.0):
    """Random per-channel distances; if signal_channel is set, that channel is small for the true class."""
    ns = n_way
    ys = np.tile(np.arange(n_way), (n_ep, 1))
    yq = np.tile(np.repeat(np.arange(n_way), nq // n_way + 1)[:nq], (n_ep, 1))
    q = rng.uniform(0.5, 1.5, size=(n_ep, nq, ns, g0.N_CHANNELS))
    if signal_channel is not None:
        for e in range(n_ep):
            for i in range(nq):
                q[e, i, yq[e, i], signal_channel] -= signal
        q = np.maximum(q, 0.0)
    s = rng.uniform(0.5, 1.5, size=(n_ep, ns, ns, g0.N_CHANNELS))
    return {"q": q, "s": s, "ys": ys, "yq": yq}


def _data(seed=0, **kw):
    rng = np.random.default_rng(seed)
    return {f"P{i}": _bundle(rng, **kw) for i in range(4)}


def test_uniform_candidate_equals_baseline_and_its_null_is_degenerate():
    res = g0.run_gate0(_data(), g0.cand_uniform, n_random=20, name="uniform")
    assert res["candidate_acc"] == res["baseline"]
    assert all(abs(v - np.mean(list(res["baseline"].values()))) < 1e-12 for v in res["null"])
    assert not res["H1_gain"] and not res["H2_beats_null"] and not res["pass"]


def test_planted_channel_passes_and_random_channel_fails():
    data = _data(signal_channel=5)

    def right(view, others):
        w = np.zeros(g0.N_CHANNELS); w[5] = 1.0
        return w

    def wrong(view, others):
        w = np.zeros(g0.N_CHANNELS); w[6] = 1.0
        return w

    good = g0.run_gate0(data, right, n_random=50, name="right")
    bad = g0.run_gate0(data, wrong, n_random=50, name="wrong")
    assert good["pass"] and good["n_users_better"] == 4
    assert all(a == 1.0 for a in good["candidate_acc"].values())
    assert not bad["pass"]
    assert good["mean_nonzero_channels"] == 1.0


def test_candidate_cannot_see_the_evaluated_users_queries():
    def peek(view, others):
        return view["q"]

    with pytest.raises(KeyError):
        g0.run_gate0(_data(), peek, n_random=1)
    seen = {}

    def inspect(view, others):
        seen["view_keys"] = set(view)
        seen["n_others"] = len(others)
        return np.ones(g0.N_CHANNELS)

    g0.run_gate0(_data(), inspect, n_random=1)
    assert seen["view_keys"] == {"s", "ys", "n_ep"} and seen["n_others"] == 3


def test_episode_weights_validation():
    b = _data()["P0"]
    with pytest.raises(ValueError, match="28"):
        g0.episode_accuracy(b, np.ones(27))
    with pytest.raises(ValueError, match="non-negative"):
        g0.episode_accuracy(b, -np.ones(g0.N_CHANNELS))
    with pytest.raises(ValueError, match="non-negative"):
        g0.episode_accuracy(b, np.zeros(g0.N_CHANNELS))
    per_ep = np.ones((len(b["q"]), g0.N_CHANNELS))
    assert np.array_equal(g0.episode_accuracy(b, per_ep), g0.episode_accuracy(b, np.ones(g0.N_CHANNELS)))


def test_permutation_null_keeps_sparsity_and_scale():
    rng = np.random.default_rng(0)
    w = np.zeros((5, g0.N_CHANNELS)); w[:, :7] = np.arange(1, 8)
    p = g0.permute_channels(w, rng)
    assert p.shape == w.shape
    assert np.array_equal(np.sort(p, axis=1), np.sort(w, axis=1))
    assert not np.array_equal(p, w)


def test_builtin_support_spread_shapes_and_cli_parse():
    data = _data()
    others = {u: data[u] for u in list(data)[1:]}
    w = g0.cand_support_spread(g0.support_view(data["P0"]), others, m=10)
    assert w.shape == (12, g0.N_CHANNELS) and ((w > 0).sum(axis=1) == 10).all()
    name, fn, kw = g0._parse_candidate("support_spread:m=10")
    assert name == "support_spread" and fn is g0.cand_support_spread and kw == {"m": 10}
    assert set(g0.CANDIDATES) >= {"uniform", "emg_only", "imu_only", "support_spread"}
    assert "PASS" in g0.format_report(g0.run_gate0(data, fn, n_random=3, name=name, **kw))
