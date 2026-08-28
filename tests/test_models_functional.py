"""
tests/test_models_functional.py
===============================
``torch.func.functional_call`` fidelity, for every model that MAML adapts.

WHY THIS IS THE HIGHEST-SEVERITY GUARD IN THE SUITE

The whole MAML++ inner loop is built on ``functional_call``: fast weights are
substituted by name (``FunctionalModel``, ``mamlpp.py:79``) rather than written
into the module. If a module ever reads a parameter through a path that
``functional_call`` does not intercept, the substitution is silently ignored --
the inner loop becomes a no-op for those weights, no exception is raised, and
the only symptom is a slightly worse number.

The concrete risk for the planned transformer work is
``nn.MultiheadAttention``: its fused fast path (``_native_multi_head_attention``)
reads ``self.in_proj_weight`` directly. It is disabled here because
``mamlpp_adapt`` forces ``model.train()`` (for the cuDNN RNN backward) and the
fast path requires ``not self.training`` -- but that is a *coincidence of two
unrelated constraints*, not a guarantee. A torch upgrade that relaxes either
would silently break every transformer inner-loop update.

Hence ``test_zeroing_a_param_changes_the_output``: it proves substitution is
actually observed, per parameter tensor, rather than assuming it.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from MAML.mamlpp import FunctionalModel, named_param_dict


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────
def _build(kind: str, cfg: dict, seed: int = 1234):
    # Seed construction so weight init -- and therefore which experts the gate
    # selects -- does not depend on test execution order.
    torch.manual_seed(seed)
    cfg = dict(cfg)
    if kind == "DeepCNNLSTM":
        from pretraining.pretrain_models import build_model

        cfg["use_MOE"] = False
        cfg["model_type"] = "DeepCNNLSTM"
        return build_model(cfg)
    if kind == "MetaCNNLSTM":
        from pretraining.pretrain_models import build_model

        cfg["use_MOE"] = False
        cfg["model_type"] = "MetaCNNLSTM"
        # FINDING: MetaCNNLSTM reads config['cnn_filters'], which
        # make_base_config() does NOT define (M0 is DeepCNNLSTM, which reads
        # cnn_base_filters). So MetaCNNLSTM is not buildable from the ablation
        # config as shipped -- it would KeyError at model construction. Supplied
        # here so the model is still covered by the functional_call guards.
        # See LIMITATIONS.md E8.
        cfg.setdefault("cnn_filters", cfg["cnn_base_filters"])
        return build_model(cfg)
    if kind == "TST":
        from pretraining.pretrain_models import build_model

        cfg["use_MOE"] = False
        cfg["model_type"] = "TST"
        cfg.setdefault("patch_len", 8)
        cfg.setdefault("d_model", 32)
        cfg.setdefault("n_heads", 4)
        cfg.setdefault("n_blocks", 2)
        return build_model(cfg)
    if kind == "DeepCNNLSTM_EncoderMOE":
        from MOE.MOE_encoder import build_MOE_model

        cfg["use_MOE"] = True
        cfg["model_type"] = "DeepCNNLSTM"
        cfg["MOE_placement"] = "encoder"
        return build_MOE_model(cfg)
    if kind == "DeepCNNLSTM_MiddleMOE":
        from MOE.MOE_encoder import build_MOE_model

        cfg["use_MOE"] = True
        cfg["model_type"] = "DeepCNNLSTM"
        cfg["MOE_placement"] = "middle"
        return build_MOE_model(cfg)
    raise AssertionError(f"unknown model kind {kind}")


ALL_MODELS = [
    "DeepCNNLSTM",
    "MetaCNNLSTM",
    "TST",
    "DeepCNNLSTM_EncoderMOE",
    "DeepCNNLSTM_MiddleMOE",
]

BATCH = 6


def _inputs(cfg):
    torch.manual_seed(7)
    return (
        torch.randn(BATCH, cfg["emg_in_ch"], cfg["sequence_length"]),
        torch.randn(BATCH, cfg["imu_in_ch"], cfg["sequence_length"]),
    )


def _forward(model, x_emg, x_imu):
    """Normalise the return: MoE models return (logits, routing_info)."""
    out = model(x_emg, x_imu)
    return out[0] if isinstance(out, tuple) else out


def _unselected_expert_prefixes(model, x_emg, x_imu) -> list[str]:
    """
    Parameter-name prefixes of experts with zero gate mass across the batch.

    Returns [] for non-MoE models. Used to exempt legitimately-inert experts
    from the per-tensor substitution check -- see the note in
    ``test_zeroing_a_param_changes_the_output``.
    """
    try:
        out = model(x_emg, x_imu, return_routing=True)
    except TypeError:
        return []
    if not (isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], dict)):
        return []
    gw = out[1].get("gate_weights")
    if gw is None:
        return []

    dead = (gw.sum(dim=0) == 0).nonzero(as_tuple=True)[0].tolist()
    if not dead:
        return []

    # Expert banks are ModuleLists; find their attribute names generically so
    # this keeps working for `expert_cnns`, `experts`, `emg_experts`, ...
    import torch.nn as nn

    banks = [
        name for name, mod in model.named_children()
        if isinstance(mod, nn.ModuleList) and len(mod) == gw.shape[1]
    ]
    return [f"{bank}.{e}." for bank in banks for e in dead]


# ─────────────────────────────────────────────────────────────────────────────
# Shape and construction
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ALL_MODELS)
def test_forward_shape(kind, tiny_config):
    model = _build(kind, tiny_config).eval()
    logits = _forward(model, *_inputs(tiny_config))
    assert logits.shape == (BATCH, tiny_config["n_way"])
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("kind", ALL_MODELS)
def test_named_param_dict_covers_every_trainable_parameter(kind, tiny_config):
    """
    Anything trainable but absent from ``named_param_dict`` is never adapted by
    the inner loop and never gets an LSLR entry (``PerParamPerStepLSLR`` builds
    its ParameterDict from this exact call). The LSLR lookup swallows misses via
    ``except (KeyError, AttributeError)`` and falls back to a flat alpha, so a
    gap here degrades silently.
    """
    model = _build(kind, tiny_config)
    from_dict = set(named_param_dict(model).keys())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == from_dict, (
        f"{kind}: not covered by named_param_dict: {sorted(trainable - from_dict)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# functional_call fidelity
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ALL_MODELS)
def test_functional_call_with_live_params_is_bitwise_identical(kind, tiny_config):
    """
    Substituting a model's OWN parameters must be a no-op, bit for bit.

    Any difference means ``functional_call`` is taking a different code path
    than the plain forward -- which would make the whole inner loop suspect.
    """
    model = _build(kind, tiny_config).eval()
    x_emg, x_imu = _inputs(tiny_config)

    torch.manual_seed(0)
    direct = _forward(model, x_emg, x_imu)

    params = OrderedDict(named_param_dict(model))
    torch.manual_seed(0)
    fm = FunctionalModel(model, params)
    out = fm(x_emg, x_imu)
    functional = out[0] if isinstance(out, tuple) else out

    assert torch.equal(direct, functional), (
        f"{kind}: functional_call with live params differs from the plain forward"
    )


@pytest.mark.parametrize("kind", ALL_MODELS)
def test_functional_call_observes_substituted_params(kind, tiny_config):
    """Scaling every weight must change the output -- substitution is observed."""
    model = _build(kind, tiny_config).eval()
    x_emg, x_imu = _inputs(tiny_config)

    base = _forward(model, x_emg, x_imu)

    perturbed = OrderedDict(
        (n, (p * 1.5).detach()) for n, p in named_param_dict(model).items()
    )
    out = FunctionalModel(model, perturbed)(x_emg, x_imu)
    got = out[0] if isinstance(out, tuple) else out

    assert not torch.allclose(base, got, atol=1e-6), (
        f"{kind}: perturbing every parameter left the output unchanged -- "
        f"functional_call substitution is NOT being observed"
    )


@pytest.mark.parametrize("kind", ALL_MODELS)
def test_zeroing_a_param_changes_the_output(kind, tiny_config):
    """
    PER-TENSOR substitution check -- the test that would catch an
    ``nn.MultiheadAttention`` fast-path bypass.

    For every weight tensor with more than one element, zero just that tensor
    and require the output to move. A tensor that can be zeroed with no effect
    is either dead capacity or, worse, being read through a path
    ``functional_call`` does not intercept.

    Genuinely inert tensors are exempted by name with a reason.
    """
    model = _build(kind, tiny_config).eval()
    x_emg, x_imu = _inputs(tiny_config)
    base = _forward(model, x_emg, x_imu)

    live = named_param_dict(model)
    inert: list[str] = []

    # Experts the gate gives ZERO mass to across this whole batch are
    # legitimately inert: with top_k < num_experts, some expert is unselected
    # for every sample, so its weights affect no output and receive no gradient.
    #
    # This is not a curiosity -- it is the reason the plan rules out TRUE sparse
    # dispatch under MAML. `torch.autograd.grad(..., allow_unused=True)` returns
    # None for such experts and `apply_update_repo_style` skips them, so they
    # silently stop adapting in the inner loop instead of raising.
    unselected_prefixes = _unselected_expert_prefixes(model, x_emg, x_imu)

    for name, p in live.items():
        if p.numel() <= 1:
            continue
        if any(name.startswith(pref) for pref in unselected_prefixes):
            continue
        # Biases can legitimately be zero-effect only if they are already zero.
        subbed = OrderedDict(
            (n, (torch.zeros_like(v) if n == name else v).detach())
            for n, v in live.items()
        )
        out = FunctionalModel(model, subbed)(x_emg, x_imu)
        got = out[0] if isinstance(out, tuple) else out
        if torch.allclose(base, got, atol=1e-7):
            if p.abs().sum().item() > 0:
                inert.append(name)

    assert not inert, (
        f"{kind}: zeroing these NON-ZERO parameters did not change the output, "
        f"so functional_call substitution is not reaching them: {inert}"
    )


@pytest.mark.parametrize("kind", ALL_MODELS)
def test_gradients_flow_through_functional_call(kind, tiny_config):
    """
    The inner loop needs ``torch.autograd.grad`` w.r.t. the substituted params.
    ``allow_unused=True`` is set on the real path, so a parameter that receives
    no gradient produces ``None`` and is silently skipped -- which is exactly how
    a broken substitution would hide. Assert most params DO get a gradient.
    """
    model = _build(kind, tiny_config).train()
    x_emg, x_imu = _inputs(tiny_config)
    labels = torch.randint(0, tiny_config["n_way"], (BATCH,))

    params = OrderedDict(
        (n, p.detach().clone().requires_grad_(True))
        for n, p in named_param_dict(model).items()
    )
    out = FunctionalModel(model, params)(x_emg, x_imu)
    logits = out[0] if isinstance(out, tuple) else out
    loss = torch.nn.functional.cross_entropy(logits, labels)

    grads = torch.autograd.grad(
        loss, list(params.values()), allow_unused=True, create_graph=False
    )
    n_none = sum(g is None for g in grads)
    n_zero = sum(
        g is not None and g.abs().sum().item() == 0.0 for g in grads
    )
    total = len(grads)
    assert n_none + n_zero < total * 0.5, (
        f"{kind}: {n_none} None + {n_zero} zero gradients out of {total} -- "
        f"more than half the parameters are not being trained"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TST-specific: the attention fast path and the pos_enc attribute
# ─────────────────────────────────────────────────────────────────────────────
def test_tst_attention_in_proj_weight_is_substitutable(tiny_config):
    """
    THE SPECIFIC RISK CALLED OUT IN THE PLAN.

    ``nn.MultiheadAttention``'s fused fast path reads ``self.in_proj_weight``
    directly and would ignore a ``functional_call`` substitution. Zeroing it must
    change the output. If this ever fails after a torch upgrade, every planned
    TST + MoE inner-loop update is a silent no-op.
    """
    model = _build("TST", tiny_config).eval()
    x_emg, x_imu = _inputs(tiny_config)
    base = _forward(model, x_emg, x_imu)

    live = named_param_dict(model)
    attn_keys = [k for k in live if "in_proj_weight" in k]
    assert attn_keys, "expected nn.MultiheadAttention in_proj_weight parameters in TST"

    for k in attn_keys:
        subbed = OrderedDict(
            (n, (torch.zeros_like(v) if n == k else v).detach())
            for n, v in live.items()
        )
        out = FunctionalModel(model, subbed)(x_emg, x_imu)
        got = out[0] if isinstance(out, tuple) else out
        assert not torch.allclose(base, got, atol=1e-7), (
            f"zeroing {k} did not change the TST output -- the attention fast "
            f"path is bypassing functional_call"
        )


def test_tst_pos_enc_is_not_in_state_dict(tiny_config):
    """
    ``TST.pos_enc`` is a plain tensor attribute, not a registered buffer, despite
    a comment suggesting otherwise. It is therefore absent from ``state_dict()``
    and not moved by ``.to(device)`` (handled by an explicit ``.to()`` at use).

    That is convenient for ``functional_call`` -- nothing to substitute -- so
    leave it alone. If it is ever "fixed" to a buffer, register it
    ``persistent=False`` or every existing checkpoint gains an unexpected key.
    """
    model = _build("TST", tiny_config)
    assert hasattr(model, "pos_enc")
    assert not any("pos_enc" in k for k in model.state_dict().keys()), (
        "pos_enc appeared in state_dict -- if this was deliberate, register it "
        "persistent=False and update LIMITATIONS.md"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MoE routing contract
# ─────────────────────────────────────────────────────────────────────────────
def test_moe_returns_routing_info_with_the_expected_contract(tiny_config):
    """
    ``MOE_analysis.RoutingCollector`` consumes per-sample ``(N, E)`` gate
    weights. Any new MoE variant (e.g. a token-level TST MoE) must reduce to
    that shape or every routing figure breaks.
    """
    model = _build("DeepCNNLSTM_EncoderMOE", tiny_config).eval()
    x_emg, x_imu = _inputs(tiny_config)

    out = model(x_emg, x_imu, return_routing=True)
    assert isinstance(out, tuple) and len(out) >= 2
    logits, routing = out[0], out[1]
    assert logits.shape == (BATCH, tiny_config["n_way"])

    assert isinstance(routing, dict)
    assert "gate_weights" in routing, f"routing keys: {sorted(routing)}"
    gw = routing["gate_weights"]
    E = tiny_config["num_experts"]
    assert gw.shape == (BATCH, E), f"expected (B, E)=({BATCH}, {E}), got {tuple(gw.shape)}"
    assert torch.allclose(gw.sum(-1), torch.ones(BATCH), atol=1e-5)
    assert ((gw > 0).sum(-1) == tiny_config["MOE_top_k"]).all()


def test_moe_forced_gate_weights_hook_survives_functional_call(tiny_config):
    """
    ``_forced_gate_weights`` is a PLAIN ATTRIBUTE read via
    ``getattr(self, "_forced_gate_weights", None)`` -- deliberately ABSENT until
    set, so ``hasattr`` is False on a fresh model. That is exactly why it
    survives ``functional_call``, which substitutes parameters and buffers but
    not arbitrary attributes.

    The same trick is what the planned modality-dropout switch and
    task-conditional routing vector will use, and portA depends on it working
    inside the MAML++ inner loop. So pin the whole contract:

      * a ``(E,)`` vector is broadcast to every sample in the batch;
      * it takes effect through ``functional_call``;
      * it survives SUBSTITUTED fast weights (i.e. it is live inside the inner
        loop, not just on a plain forward);
      * clearing it restores the original output exactly;
      * it does not block gradients.
    """
    from MOE.MOE_encoder import DeepCNNLSTM_EncoderMOE

    model = _build("DeepCNNLSTM_EncoderMOE", tiny_config).eval()
    assert isinstance(model, DeepCNNLSTM_EncoderMOE)
    x_emg, x_imu = _inputs(tiny_config)
    E = tiny_config["num_experts"]

    assert not hasattr(model, "_forced_gate_weights"), (
        "the hook must be absent by default -- it is read with getattr(..., None) "
        "so that it is inert and invisible unless deliberately set"
    )

    base = _forward(model, x_emg, x_imu)

    # (E,) vector, all mass on expert 0 -- the shape portA supplies.
    forced = torch.zeros(E)
    forced[0] = 1.0
    model._forced_gate_weights = forced

    # (a) effective through functional_call with LIVE params
    params = OrderedDict(named_param_dict(model))
    out = FunctionalModel(model, params)(x_emg, x_imu)
    got = out[0] if isinstance(out, tuple) else out
    assert not torch.allclose(base, got, atol=1e-6), (
        "forcing gate weights had no effect through functional_call"
    )

    # (b) still effective with SUBSTITUTED fast weights -- proves it is live
    #     inside the inner loop, where params are replaced every step
    fast = OrderedDict((n, (p * 1.1).detach()) for n, p in params.items())
    out_fast_forced = FunctionalModel(model, fast)(x_emg, x_imu)
    lf = out_fast_forced[0] if isinstance(out_fast_forced, tuple) else out_fast_forced

    model._forced_gate_weights = None
    out_fast_free = FunctionalModel(model, fast)(x_emg, x_imu)
    lu = out_fast_free[0] if isinstance(out_fast_free, tuple) else out_fast_free
    assert not torch.allclose(lf, lu, atol=1e-6), (
        "the hook stopped mattering once fast weights were substituted"
    )

    # (c) clearing restores exactly
    restored = _forward(model, x_emg, x_imu)
    assert torch.equal(base, restored), "clearing the hook did not restore the output"

    # (d) does not block gradients
    model._forced_gate_weights = forced
    grad_params = OrderedDict(
        (n, p.detach().clone().requires_grad_(True)) for n, p in params.items()
    )
    out = FunctionalModel(model, grad_params)(x_emg, x_imu)
    logits = out[0] if isinstance(out, tuple) else out
    loss = logits.square().mean()
    grads = torch.autograd.grad(loss, list(grad_params.values()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads), (
        "no gradients flowed while the gate override was active"
    )
    model._forced_gate_weights = None


def test_forced_gate_weights_rejects_wrong_expert_count(tiny_config):
    """A mismatched override must fail loudly, not broadcast into nonsense."""
    model = _build("DeepCNNLSTM_EncoderMOE", tiny_config).eval()
    x_emg, x_imu = _inputs(tiny_config)
    model._forced_gate_weights = torch.ones(tiny_config["num_experts"] + 3)
    with pytest.raises(AssertionError, match="experts but the gate"):
        _forward(model, x_emg, x_imu)
    model._forced_gate_weights = None
