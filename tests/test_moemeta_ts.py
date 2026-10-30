"""
tests/test_moemeta_ts.py
========================
MoEMeta-TS (arch_variants/moemeta_ts.py): the equations reduce as the paper says, every variant trains,
local adaptation lowers the support loss, evaluation works under no_grad, and the Q1 tokens are wired
through the registry and the HPO driver.
"""

from __future__ import annotations

import pytest
import torch

from test_arch_variants import _episode
from test_models_functional import arch_keys

MM = dict(metric_kind="moemeta", metric_backbone="cnnlstm", moemeta_variant="full", moemeta_loss="margin",
          moemeta_num_experts=4, moemeta_top_n=2, moemeta_expert_hidden=8, moemeta_margin=1.0,
          moemeta_inner_steps=1, moemeta_inner_lr=0.1, moemeta_first_order=False)


def _model(cfg, seed=0, **over):
    from arch_variants.metric_learners import build_metric_learner
    torch.manual_seed(seed)
    c = arch_keys(dict(cfg))
    c.update(MM, use_MOE=False, **over)
    return build_metric_learner(c), c


def test_builds_through_the_metric_dispatcher(tiny_config):
    from arch_variants.moemeta_ts import MoEMetaTS
    model, c = _model(tiny_config)
    assert isinstance(model, MoEMetaTS) and model.kind == "moemeta"
    assert len(model.experts) == 4 and model.gate is not None
    with pytest.raises(RuntimeError, match="episodic"):
        model(torch.zeros(1, c["emg_in_ch"], c["sequence_length"]), torch.zeros(1, c["imu_in_ch"], c["sequence_length"]))


@pytest.mark.parametrize("variant", ["full", "no_local", "mlp", "uniform"])
@pytest.mark.parametrize("loss", ["margin", "ce"])
def test_every_variant_gives_finite_logits_and_learns(tiny_config, variant, loss):
    from arch_variants.metric_learners import _episode_loss, metric_adapt_and_eval
    model, c = _model(tiny_config, moemeta_variant=variant, moemeta_loss=loss)
    ep = _episode(c, n_way=3, n_q=2)
    logits, y, _ = model.episode_logits(ep["support"], ep["query"], "cpu")
    assert logits.shape == (6, 3) and torch.isfinite(logits).all()
    assert (logits <= 0).all()                                   # logits are negative distances
    out = metric_adapt_and_eval(model, c, ep["support"], ep["query"])   # runs under no_grad
    assert 0.0 <= out["acc"] <= 1.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    first = None
    for _ in range(15):
        loss_v, _, _ = _episode_loss(model, ep, c, torch.nn.CrossEntropyLoss())
        first = float(loss_v.detach()) if first is None else first
        opt.zero_grad(); loss_v.backward(); opt.step()
    assert float(loss_v.detach()) < first
    # the outer loss reached the encoder (Phi is meta-learned end to end)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_variants_change_structure(tiny_config):
    full, _ = _model(tiny_config)
    mlp, _ = _model(tiny_config, moemeta_variant="mlp")
    uni, _ = _model(tiny_config, moemeta_variant="uniform")
    nola, _ = _model(tiny_config, moemeta_variant="no_local")
    assert len(mlp.experts) == 1 and mlp.gate is None and mlp.N == 1
    assert len(uni.experts) == 4 and uni.gate is None
    assert nola.inner_steps == 0 and full.inner_steps == 1
    with pytest.raises(ValueError, match="top_n"):
        _model(tiny_config, moemeta_top_n=9)
    with pytest.raises(ValueError, match="moemeta_variant"):
        _model(tiny_config, moemeta_variant="nope")


def test_gate_keeps_top_n_without_renormalising(tiny_config):
    model, c = _model(tiny_config)
    u, zs = torch.randn(model.feat_dim), torch.randn(5, model.feat_dim)
    model.relation_meta(u, zs)
    soft, hard = model.last_gate_soft, model.last_gate_hard
    assert soft.shape == hard.shape == (5, 4)
    assert ((hard > 0).sum(dim=1) == 2).all()                    # exactly N experts per input
    assert torch.allclose(hard[hard > 0], soft[hard > 0])        # kept weights are the softmax scores
    assert (hard.sum(dim=1) < 1.0 + 1e-6).all()


def test_identity_expert_at_one_shot_is_nearest_neighbour(tiny_config):
    """Eq 12 with f([u, z]) = z - u and no adaptation: d(q, c) = ||z_c - z_q|| (ANALYSIS_moemeta section 3)."""
    D = _model(tiny_config, moemeta_variant="mlp")[0].feat_dim
    # relu(x) - relu(-x) = x needs 2D hidden units
    model, c = _model(tiny_config, moemeta_variant="mlp", moemeta_inner_steps=0, moemeta_expert_hidden=2 * D)
    lin1, lin2 = model.experts[0][0], model.experts[0][2]
    with torch.no_grad():                                        # exact linear identity through the ReLU
        lin1.weight.zero_(); lin1.bias.zero_(); lin2.weight.zero_(); lin2.bias.zero_()
        for i in range(D):
            lin1.weight[i, D + i] = 1.0;  lin1.weight[i, i] = -1.0       # +(z - u)_i
            lin1.weight[D + i, D + i] = -1.0;  lin1.weight[D + i, i] = 1.0  # -(z - u)_i
            lin2.weight[i, i] = 1.0;  lin2.weight[i, D + i] = -1.0
    u, zs, zq = torch.randn(D), torch.randn(3, D), torch.randn(4, D)
    r = model.relation_meta(u, zs)
    assert torch.allclose(r, zs - u, atol=1e-6)
    R = r                                                       # 1-shot: R_c = r_c
    d = model._score(u, zq, R, torch.zeros(3, D))
    assert torch.allclose(d, torch.cdist(zq, zs), atol=1e-5)


def test_local_adaptation_lowers_the_support_loss(tiny_config):
    model, c = _model(tiny_config, moemeta_inner_steps=3, moemeta_inner_lr=0.05)
    model.eval()
    D = model.feat_dim
    torch.manual_seed(1)
    u, zs, y = torch.randn(D), torch.randn(3, D), torch.arange(3)
    with torch.no_grad():
        R0 = (torch.eye(3) @ model.relation_meta(u, zs))
        before = model._loss_from_dist(model._score(u, zs, R0, torch.zeros(3, D)), y)
        R1, p1 = model._adapt(u, zs, y, R0)
        after = model._loss_from_dist(model._score(u, zs, R1, p1), y)
    assert not R1.requires_grad and not p1.requires_grad
    assert float(after) <= float(before)
    assert (p1 != 0).any()                                      # the task vectors moved from zero


def test_first_order_and_second_order_both_backprop(tiny_config):
    for fo in (False, True):
        model, c = _model(tiny_config, moemeta_first_order=fo, moemeta_inner_steps=2)
        ep = _episode(c, n_way=3, n_q=1)
        model.train()
        logits, y, _ = model.episode_logits(ep["support"], ep["query"], "cpu")
        model.query_loss(logits, y).backward()
        assert all(p.grad is not None for p in model.experts.parameters())


def test_margin_loss_is_zero_when_separated_and_ce_matches_torch(tiny_config):
    import torch.nn.functional as F
    model, _ = _model(tiny_config)
    y = torch.tensor([0, 1])
    dist = torch.tensor([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0]])
    assert float(model._loss_from_dist(dist, y)) == 0.0          # margin 1 satisfied everywhere
    dist2 = torch.tensor([[1.0, 1.5, 5.0], [5.0, 0.0, 5.0]])
    assert abs(float(model._loss_from_dist(dist2, y)) - 0.5 / 4) < 1e-6   # one hinge of 0.5, over 4 wrong pairs
    model.loss_kind = "ce"
    assert torch.allclose(model._loss_from_dist(dist2, y), F.cross_entropy(-dist2, y))


# ── registry / HPO wiring ────────────────────────────────────────────────────
def test_q1_tokens_are_registered_as_metric_family():
    from arch_variants.registry import ARCH_VARIANTS, FAMILY_EVAL_DEFAULTS, variant_kind, uses_a4_widths, uses_moe
    toks = [t for t in ARCH_VARIANTS if t.startswith("Q1")]
    assert set(toks) == {"Q1", "Q1_nola", "Q1_mlp", "Q1_uni", "Q1_ce"}
    for t in toks:
        v = ARCH_VARIANTS[t]
        assert v["set"]["metric_kind"] == "moemeta" and "moemeta" in v["spaces"] and "metric" in v["spaces"]
        assert variant_kind(t) == "metric" and uses_a4_widths(t) and not uses_moe(t)
    assert ARCH_VARIANTS["Q1_ce"]["set"]["moemeta_loss"] == "ce"
    assert set(MM) - {"metric_kind", "metric_backbone", "moemeta_variant", "moemeta_loss"} == set(FAMILY_EVAL_DEFAULTS["moemeta"])


def test_eval_build_config_writes_moemeta_defaults(monkeypatch):
    import os, sys
    sys.path[:0] = [p for p in ("system/NOTS/paper/ablations/test_eval_files", "system/NOTS/paper/ablations")
                    if p not in sys.path]
    monkeypatch.setenv("HPO_CONFIG", "legacy")
    import arch_variant_eval as ave
    cfg = ave.build_config("Q1", smoke=True)
    assert cfg["metric_kind"] == "moemeta" and cfg["moemeta_num_experts"] == 32 and cfg["moemeta_top_n"] == 5
    assert cfg["arch_family"] == "metric" and cfg["use_MOE"] is False
    from arch_variants.registry import build_model_for_config
    cfg.update(cnn_base_filters=8, lstm_hidden=8, groupnorm_num_groups=4, moemeta_expert_hidden=8,
               moemeta_num_experts=3, moemeta_top_n=2, device=torch.device("cpu"))
    model = build_model_for_config(cfg)
    assert model.kind == "moemeta"
