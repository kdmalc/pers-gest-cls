"""
tests/test_riemannian.py
========================
The Riemannian / covariance code (docs/specs/SPEC_riemannian.md): SPD utilities, the X7 / X8 / X9 / R1
model variants, SPDNet, the new classical methods, and the analysis tools (trial distance bank,
failure analysis, hybrid fusion).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from test_arch_variants import _moe, _x, _episode
from test_models_functional import arch_keys

pytest.importorskip("pyriemann")


# ─────────────────────────────────────────────────────────────────────────────
# SPD utilities
# ─────────────────────────────────────────────────────────────────────────────
def _spd(n=4, C=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, C, 3 * C, generator=g, dtype=torch.float64)
    return A @ A.transpose(1, 2) / (3 * C) + 0.1 * torch.eye(C, dtype=torch.float64)


def test_shrunk_cov_is_spd_and_rejects_bad_rho():
    from arch_variants.spd import shrunk_cov
    x = torch.randn(3, 8, 5, dtype=torch.float64)          # C > T: the sample covariance is singular
    S = shrunk_cov(x, 0.1)
    assert torch.linalg.eigvalsh(S).min() > 0
    assert torch.allclose(S, S.transpose(1, 2))
    with pytest.raises(ValueError):
        shrunk_cov(x, 0.0)


def test_triu_vec_roundtrip_and_norm():
    from arch_variants.spd import triu_vec, untriu_vec
    S = _spd()
    v = triu_vec(S)
    assert v.shape == (4, 15)
    assert torch.allclose(untriu_vec(v, 5), S)
    assert torch.allclose(v.norm(dim=1), S.flatten(1).norm(dim=1))      # sqrt(2) weighting


def test_newton_schulz_sqrt_matches_eigh():
    from arch_variants.spd import sqrtm_ns, sym_fn
    S = _spd()
    assert torch.allclose(sqrtm_ns(S, iters=20), sym_fn(S, torch.sqrt), atol=1e-6)


def test_support_whitener_whitens_the_support_mean():
    from arch_variants.spd import shrunk_cov, support_whitener
    x = torch.randn(10, 6, 40, dtype=torch.float64) * torch.arange(1, 7, dtype=torch.float64).view(1, 6, 1)
    W = support_whitener(x, 0.1)
    R = shrunk_cov(x, 0.1).mean(0)
    assert torch.allclose(W @ R @ W, torch.eye(6, dtype=torch.float64), atol=1e-8)


def test_covpool_shapes_and_gradients():
    from arch_variants.spd import CovPool
    for norm in ("sqrt", "log"):
        cp = CovPool(12, 4, norm=norm, rho=0.1).double()
        h = torch.randn(3, 20, 12, dtype=torch.float64, requires_grad=True)
        out = cp(h)
        assert out.shape == (3, 10)
        out.sum().backward()
        assert torch.isfinite(h.grad).all() and torch.isfinite(cp.proj.weight.grad).all()
    with pytest.raises(ValueError):
        CovPool(12, 4, norm="bogus")


# ─────────────────────────────────────────────────────────────────────────────
# Model variants
# ─────────────────────────────────────────────────────────────────────────────
def _forward_ok(model, x, support=None):
    from MAML.mamlpp import task_support
    sup = support if support is not None else {"emg": x[0], "imu": x[1]}
    with task_support(model, sup, "cpu"):
        out = model(*x, return_routing=True) if hasattr(model, "num_experts") else model(*x)
    logits = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(logits).all()
    return out


def test_x7_router_reads_the_input_covariance(tiny_config):
    model, cfg = _moe(tiny_config, MOE_routing_signal="cov_tangent", cov_shrinkage=0.1)
    x = _x(cfg)
    _, info = _forward_ok(model, x)
    assert info["gate_weights"].shape == (x[0].shape[0], cfg["num_experts"])
    # scaling one channel changes the covariance, hence the routing
    x2 = (x[0].clone(), x[1].clone()); x2[0][:, 0] *= 5
    _, info2 = model(*x2, return_routing=True)
    assert not torch.allclose(info["gate_weights_soft"], info2["gate_weights_soft"])


def test_x7t_needs_and_uses_the_support_set(tiny_config):
    from MAML.mamlpp import task_support
    model, cfg = _moe(tiny_config, MOE_routing_signal="cov_tangent_task", cov_shrinkage=0.1)
    assert model.needs_task_support
    x = _x(cfg)
    with pytest.raises(RuntimeError, match="support set"):
        model(*x)
    s1 = {"emg": x[0], "imu": x[1]}
    s2 = {"emg": x[0] * 3 + 1, "imu": x[1]}
    with task_support(model, s1, "cpu"):
        a = model(*x)
    with task_support(model, s2, "cpu"):
        b = model(*x)
    assert not torch.allclose(a, b)


def _synthetic_users(cfg, n_users=4, n_cls=3, n_rep=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {f"U{u}": {c: {"emg": torch.randn(n_rep, cfg["emg_in_ch"], cfg["sequence_length"], generator=g) * (1 + u),
                          "imu": torch.randn(n_rep, cfg["imu_in_ch"], cfg["sequence_length"], generator=g)}
                      for c in range(n_cls)} for u in range(n_users)}


@pytest.mark.parametrize("whiten", [False, True])
def test_x8_centroid_gate_requires_init_then_routes(tiny_config, whiten):
    from MOE.MOE_encoder import init_cov_centroids, CovCentroidGate
    model, cfg = _moe(tiny_config, MOE_routing_signal="cov_centroid", cov_shrinkage=0.1,
                      MOE_cov_expert_whiten=whiten)
    assert isinstance(model.gate, CovCentroidGate)
    x = _x(cfg)
    with pytest.raises(RuntimeError, match="not initialised"):
        model(*x)
    init_cov_centroids(model, _synthetic_users(cfg, n_users=cfg["num_experts"] + 1), None or
                       [f"U{u}" for u in range(cfg["num_experts"] + 1)], log=lambda *_: None)
    assert bool(model.gate.initialised)
    _, info = _forward_ok(model, x)
    w = info["gate_weights_soft"]
    assert torch.allclose(w.sum(-1), torch.ones(len(w)))
    # gradients reach the centroids (through the gate and, with whitening, through the experts' inputs)
    out, info = model(*x, return_routing=True)
    out.sum().backward()
    assert model.gate.centroids.grad is not None and torch.isfinite(model.gate.centroids.grad).all()
    # the init survives a state_dict round trip
    model2, _ = _moe(tiny_config, MOE_routing_signal="cov_centroid", cov_shrinkage=0.1,
                     MOE_cov_expert_whiten=whiten)
    model2.load_state_dict(model.state_dict())
    assert bool(model2.gate.initialised)


def test_x8_init_refuses_too_few_users(tiny_config):
    from MOE.MOE_encoder import init_cov_centroids
    model, cfg = _moe(tiny_config, MOE_routing_signal="cov_centroid", cov_shrinkage=0.1,
                      MOE_cov_expert_whiten=False)
    td = _synthetic_users(cfg, n_users=cfg["num_experts"] - 1)
    with pytest.raises(ValueError, match="experts"):
        init_cov_centroids(model, td, sorted(td), log=lambda *_: None)


def test_whiten_without_centroid_routing_raises(tiny_config):
    with pytest.raises(ValueError, match="cov_centroid"):
        _moe(tiny_config, MOE_cov_expert_whiten=True)


@pytest.mark.parametrize("norm", ["sqrt", "log"])
def test_x9_cov_pool_changes_the_feature_size(tiny_config, norm):
    model, cfg = _moe(tiny_config, feature_pool="cov", cov_pool_dim=5, cov_pool_norm=norm, cov_shrinkage=0.1)
    assert model.feat_dim == 15
    feat, _ = model.backbone(*_x(cfg))
    assert feat.shape[-1] == 15
    _forward_ok(model, _x(cfg))


def test_x9_rejects_transformer(tiny_config):
    with pytest.raises(ValueError, match="transformer"):
        _moe(tiny_config, feature_pool="cov", cov_pool_dim=5, cov_pool_norm="sqrt", cov_shrinkage=0.1,
             temporal_model="transformer")


def _a4(cfg, **over):
    from pretraining.pretrain_models import build_model
    c = arch_keys(dict(cfg))
    c.update(use_MOE=False, model_type="DeepCNNLSTM", **over)
    return build_model(c), c


@pytest.mark.parametrize("builder", ["moe", "a4"])
def test_r1_euclidean_alignment_is_invariant_to_modality_gain(tiny_config, builder):
    """EA whitens each modality by the support set's mean covariance, so a gain change of a whole
    modality (support and query alike) leaves the aligned inputs, hence the logits, unchanged.
    (A general channel mixing A only maps the aligned inputs to a rotation of themselves.)"""
    from MAML.mamlpp import task_support
    over = dict(input_align="ea", cov_shrinkage=0.1)
    model, cfg = (_moe(tiny_config, **over) if builder == "moe" else _a4(tiny_config, **over))
    model.eval()
    assert model.needs_task_support
    x = _x(cfg)
    with pytest.raises(RuntimeError, match="support set"):
        model(*x)
    with task_support(model, {"emg": x[0], "imu": x[1]}, "cpu"):
        a = model(*x)
    xs = (x[0] * 7.0, x[1] * 0.2)                         # every channel's gain changes uniformly
    with task_support(model, {"emg": xs[0], "imu": xs[1]}, "cpu"):
        b = model(*xs)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    assert torch.allclose(a, b, atol=1e-4)


def test_a4_cov_pool(tiny_config):
    model, cfg = _a4(tiny_config, feature_pool="cov", cov_pool_dim=4, cov_pool_norm="sqrt", cov_shrinkage=0.1)
    assert model.feat_dim == 10
    assert torch.isfinite(model(*_x(cfg))).all()


def test_spdnet_metric_learner(tiny_config):
    from arch_variants.metric_learners import build_metric_learner
    cfg = arch_keys(dict(tiny_config))
    cfg.update(metric_kind="proto", metric_backbone="spdnet", spd_dims=[6, 4], spd_reeig_eps=1e-4,
               cov_shrinkage=0.1, use_MOE=False)
    model = build_metric_learner(cfg)
    ep = _episode(cfg, n_way=3, n_q=2)
    logits, y, _ = model.episode_logits(ep["support"], ep["query"], "cpu")
    assert logits.shape == (6, 3) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, y).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ─────────────────────────────────────────────────────────────────────────────
# Classical methods and the analysis tools
# ─────────────────────────────────────────────────────────────────────────────
from test_classical import setup  # noqa: E402,F401  (fixture: cached fold-0 episodes on synthetic data)


@pytest.mark.parametrize("method", ["riemann_seg_mdm"])
def test_seg_mdm_beats_chance(setup, method):
    from episodes.episode_numpy import materialize
    from nonparametric import classical
    es, td = setup
    accs = [(classical.predict(method, m) == m.y_query).mean()
            for m in (materialize(ep, td) for ep in es.select(3, 1)[:6])]
    assert np.mean(accs) > 0.45


@pytest.mark.parametrize("method", ["riemann_rc_pool", "riemann_cluster_pool"])
def test_rc_pool_needs_state_and_runs(setup, method):
    from episodes.episode_numpy import materialize
    from nonparametric import classical
    es, td = setup
    m = materialize(es.select(3, 1)[0], td)
    with pytest.raises(ValueError, match="state"):
        classical.predict(method, m)
    train = sorted(td)          # synthetic data: only the fitting code is under test here (>= 4 users for 2 clusters)
    n_clusters = 1 if method == "riemann_rc_pool" else 2
    state = classical.fit_rc_pool(td, train, n_clusters=n_clusters, n_pca=8)
    p = classical.predict(method, m, state=state)
    assert p.shape == m.y_query.shape
    assert np.array_equal(p, classical.predict(method, m, state=state))
    wrong = classical.fit_rc_pool(td, train, n_clusters=3 - n_clusters, n_pca=8)
    with pytest.raises(ValueError):
        classical.predict(method, m, state=wrong)


def test_distance_bank_reproduces_classical_predictions(setup, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system" / "analysis"))
    import trial_distances as T
    from episodes.episode_numpy import materialize
    from nonparametric import classical
    es, td = setup
    u = es.users[0]
    flat = {f"{u}|{k}": v for k, v in T.user_distances(td[u]).items()}
    np.savez(tmp_path / "d.npz", **flat)
    bank = T.DistanceBank(tmp_path / "d.npz")
    pairs = {"mdm": "riemann_mdm", "seg_mdm": "riemann_seg_mdm", "knn_l1": "knn_l1",
             "td_knn": "td_knn", "dollar_b": "dollar_b"}
    eps = [ep for ep in es.select(3, 1) if ep.user == u][:3]
    assert eps
    for ep in eps:
        m = materialize(ep, td)
        for k, c in pairs.items():
            assert np.array_equal(bank.episode_distances(m, k).argmin(1), classical.predict(c, m)), k
        with pytest.raises(ValueError):
            m2 = materialize(es.select(3, 3)[0], td)
            bank.episode_distances(m2, "mdm")


def test_failure_analysis_and_hybrid_on_a_small_bank(setup, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system" / "analysis"))
    import trial_distances as T
    import failure_analysis as FA
    import hybrid_fusion as HF
    from episodes import episode_cache as ec
    es, td = setup
    # the fixture's synthetic dict covers fold 0's test users, which the 10-way cache cells need
    flat = {}
    for u in es.users:
        for k, v in T.user_distances(td[u], methods=("mdm", "td_knn")).items():
            flat[f"{u}|{k}"] = v
    np.savez(tmp_path / "d.npz", **flat)
    bank = T.DistanceBank(tmp_path / "d.npz")
    # fake model logits = MDM scores + noise, keyed like --dump-logits
    store = {}
    for ep in ec.load("test", 0).select(10, 1, max_per_user=2):
        ix = FA.EpisodeIndex(ep)
        s = -bank.episode_distances(ix, "mdm")
        store[f"{ep.id}|logits"] = s + np.random.default_rng(0).normal(0, s.std(), s.shape)
        store[f"{ep.id}|labels"] = ix.y_query
    np.savez(tmp_path / "nn_fold0.npz", **store)
    name, loaded = FA.load_neural(f"NN={tmp_path / 'nn_fold*.npz'}")
    q = FA.collect(bank, {name: loaded}, [0], max_per_user=2)
    for col in ("correct_mdm", "correct_td_knn", "correct_NN", "correct_mdm+NN", "rep_gap"):
        assert col in q
    comp = FA.complementarity(q, ["mdm", "td_knn", "NN"])
    assert all(c["oracle_or"] >= max(c["acc_a"], c["acc_b"]) - 1e-9 for c in comp)
    # hybrid: only the episodes that have logits
    acc = {}
    bad = dict(loaded); bad[next(k for k in bad if k.endswith("|labels"))] = np.zeros(1)
    with pytest.raises((ValueError, KeyError)):
        HF.per_user_acc(bank, bad, [0])
    assert HF.W_GRID[0] == 0.0 and HF.W_GRID[-1] == 1.0
    res = HF.louo({u: np.linspace(0.5, 0.9, len(HF.W_GRID)) for u in ("a", "b", "c")})
    assert res["louo_acc"] == pytest.approx(90.0) and set(res["w_chosen"]) == {1.0}
