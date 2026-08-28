"""
tests/test_moe_gating.py
========================
Gate and auxiliary-loss invariants. All data-free, all CPU.

Pins two findings from LIMITATIONS.md:

  * **C2** -- ``MOEGate`` applies top-k only ``if top_k < num_experts``, so an
    HPO that samples ``num_experts`` and ``MOE_top_k`` INDEPENDENTLY silently
    runs some cells dense and others sparse. In the A5 HPO space
    (``E in [4..40]``, ``top_k in [4..10]``) every ``E=4`` trial ran dense while
    ``E=40`` ran sparse, confounding expert count with routing density.

  * **A4** -- every shipped aux loss constrains BATCH MARGINALS ONLY. None of
    the three penalises a flat per-sample distribution, which is why measured
    routing entropy sits at 98.2% of its ceiling. The tests below make that
    structural fact explicit, so the eventual
    ``routing_mutual_information_loss`` has a baseline to be compared against.
"""

from __future__ import annotations

import math

import pytest
import torch

from MOE.MOE_encoder import (
    MOEGate,
    dense_MOE_aux_loss,
    topk_MOE_aux_loss,
    importance_loss,
)

B, IN_DIM = 32, 16


def _gate(num_experts, top_k=None, temperature=1.0, seed=0):
    torch.manual_seed(seed)
    return MOEGate(IN_DIM, num_experts, top_k=top_k, temperature=temperature)


def _r(seed=0):
    torch.manual_seed(seed + 500)
    return torch.randn(B, IN_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Basic gate invariants
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("E,k", [(4, 2), (8, 3), (22, 9), (40, 10)])
def test_gate_rows_are_distributions(E, k):
    hard, soft = _gate(E, top_k=k)(_r())
    assert hard.shape == soft.shape == (B, E)
    assert torch.allclose(hard.sum(-1), torch.ones(B), atol=1e-5)
    assert torch.allclose(soft.sum(-1), torch.ones(B), atol=1e-5)
    assert (hard >= 0).all() and (soft >= 0).all()


@pytest.mark.parametrize("E,k", [(4, 2), (8, 3), (22, 9), (40, 10)])
def test_topk_selects_exactly_k_experts_when_k_lt_E(E, k):
    hard, _ = _gate(E, top_k=k)(_r())
    nnz = (hard > 0).sum(-1)
    assert (nnz == k).all(), f"E={E}, k={k}: expected {k} active experts, got {nnz.unique().tolist()}"


def test_topk_none_is_dense_and_hard_equals_soft():
    hard, soft = _gate(22, top_k=None)(_r())
    assert torch.equal(hard, soft)
    assert ((hard > 0).sum(-1) == 22).all()


# ─────────────────────────────────────────────────────────────────────────────
# THE A5 HPO DEGENERACY  (LIMITATIONS.md C2)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("E,k", [(4, 4), (4, 9), (8, 8), (8, 10), (22, 22), (22, 30)])
def test_top_k_ge_num_experts_silently_runs_dense(E, k):
    """
    ``top_k >= num_experts`` is NOT an error and NOT clamped-with-a-warning --
    the gate silently returns dense routing.

    Consequence, and the reason this test exists: ``ablation_hpo.py`` samples
    ``num_experts in [4,8,...,40]`` and ``MOE_top_k in [4..10]`` independently,
    so every ``E=4`` trial ran DENSE while ``E=40`` ran sparse. The A5 HPO curve
    therefore confounds expert count with routing density. (The A5 *sweep
    script* is fine -- it derives ``top_k = round(E/3)``.)

    The fix is to derive top_k from a utilisation ratio, as ``M0_MOE_hpo.py``
    already does. When that lands, this test documents the old behaviour that
    the correction is relative to.
    """
    hard, soft = _gate(E, top_k=k)(_r())
    assert torch.equal(hard, soft), (
        f"E={E}, top_k={k}: expected silent dense fallback"
    )
    assert ((hard > 0).sum(-1) == E).all()


def test_top_k_exactly_one_is_argmax_routing():
    """k=1 collapses to hard argmax with weight 1.0 -- the sparse-routing limit."""
    hard, soft = _gate(8, top_k=1)(_r())
    assert ((hard > 0).sum(-1) == 1).all()
    assert torch.allclose(hard.max(-1).values, torch.ones(B), atol=1e-5)
    assert torch.equal(hard.argmax(-1), soft.argmax(-1))


# ─────────────────────────────────────────────────────────────────────────────
# Temperature  (LIMITATIONS.md A4, cause 1)
# ─────────────────────────────────────────────────────────────────────────────
def _mean_entropy(w, eps=1e-9):
    return -(w * (w + eps).log()).sum(-1).mean().item()


def test_temperature_above_one_flattens_and_below_one_sharpens():
    """
    THE SHIPPED CONFIG USES tau = 1.529, i.e. tau > 1, WHICH FLATTENS.

    ``MOEGate.forward`` computes ``logits / temperature``, so tau > 1 raises the
    per-sample entropy. The HPO selected a flattening temperature because the
    objective rewarded ensemble-averaging -- which, with 0 dead experts and
    measured entropy at 98.2% of its ceiling, is what the MoE is actually doing.
    Annealing tau below 1 is the cheapest intervention against it.
    """
    E, k = 22, 9
    r = _r()
    h_sharp, _ = _gate(E, top_k=k, temperature=0.4)(r)
    h_unit, _ = _gate(E, top_k=k, temperature=1.0)(r)
    h_flat, _ = _gate(E, top_k=k, temperature=1.529)(r)   # the shipped value

    e_sharp, e_unit, e_flat = map(_mean_entropy, (h_sharp, h_unit, h_flat))
    assert e_sharp < e_unit < e_flat, (
        f"expected entropy to increase with temperature, got "
        f"tau=0.4:{e_sharp:.4f} tau=1.0:{e_unit:.4f} tau=1.529:{e_flat:.4f}"
    )
    # The reachable ceiling under a top-k mask is log(k), not log(E).
    assert e_flat < math.log(k) + 1e-6


def test_entropy_ceiling_is_log_topk_not_log_num_experts():
    """
    Reporting normalisation matters for the paper.

    With top_k=9 of E=22, a uniform-over-selected distribution has entropy
    log 9 = 2.197, not log 22 = 3.091. The recorded H_mean of 2.158 is 98.2% of
    log(k) but only 69.8% of log(E) -- quoting the latter would badly understate
    how flat the routing is. Report both.
    """
    E, k = 22, 9
    hard, _ = _gate(E, top_k=k, temperature=50.0)(_r())   # ~uniform over top-k
    h = _mean_entropy(hard)
    assert h == pytest.approx(math.log(k), abs=1e-3)
    assert h < math.log(E) - 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Aux losses  (LIMITATIONS.md A4, cause 2)
# ─────────────────────────────────────────────────────────────────────────────
def test_dense_aux_loss_is_zero_at_uniform_and_positive_otherwise():
    E = 8
    uniform = torch.full((B, E), 1.0 / E)
    assert dense_MOE_aux_loss(uniform, coeff=1.0).item() == pytest.approx(0.0, abs=1e-6)

    collapsed = torch.zeros(B, E)
    collapsed[:, 0] = 1.0
    assert dense_MOE_aux_loss(collapsed, coeff=1.0).item() > 0.1


def test_switch_aux_loss_is_minimised_at_balanced_load():
    """Switch loss E * sum(f_i * P_i) is minimised when load is balanced."""
    E, k = 8, 2
    torch.manual_seed(0)

    soft_bal = torch.full((B, E), 1.0 / E)
    hard_bal = torch.zeros(B, E)
    for b in range(B):                       # rotate the active pair
        idx = [(b * k + j) % E for j in range(k)]
        hard_bal[b, idx] = 1.0 / k

    soft_col = torch.full((B, E), 0.01)
    soft_col[:, 0] = 1.0 - 0.01 * (E - 1)
    hard_col = torch.zeros(B, E)
    hard_col[:, :k] = 1.0 / k                # always the same k experts

    bal = topk_MOE_aux_loss(soft_bal, hard_bal, coeff=1.0).item()
    col = topk_MOE_aux_loss(soft_col, hard_col, coeff=1.0).item()
    assert bal < col, f"balanced load should score lower: {bal:.4f} vs {col:.4f}"


def test_switch_aux_loss_freezes_the_load_signal():
    """
    ``f`` (dispatch fraction) must be detached; only ``P`` carries gradient.
    The docstring calls this critical, so pin it: a gradient through the hard
    mask would be differentiating a top-k selection.
    """
    E, k = 8, 2
    soft = torch.full((B, E), 1.0 / E, requires_grad=True)
    hard = torch.zeros(B, E, requires_grad=True)
    hard.data[:, :k] = 1.0 / k

    topk_MOE_aux_loss(soft, hard, coeff=1.0).backward()
    assert soft.grad is not None and soft.grad.abs().sum() > 0
    assert hard.grad is None or hard.grad.abs().sum().item() == 0.0


def test_importance_loss_is_zero_at_balanced_importance():
    E = 8
    uniform = torch.full((B, E), 1.0 / E)
    assert importance_loss(uniform, coeff=1.0).item() == pytest.approx(0.0, abs=1e-6)

    skewed = torch.full((B, E), 0.01)
    skewed[:, 0] = 1.0 - 0.01 * (E - 1)
    assert importance_loss(skewed, coeff=1.0).item() > 0.0


@pytest.mark.parametrize(
    "loss_name", ["dense_MOE_aux_loss", "topk_MOE_aux_loss", "importance_loss"]
)
def test_no_shipped_aux_loss_penalises_flat_per_sample_routing(loss_name):
    """
    THE STRUCTURAL FINDING BEHIND LIMITATIONS.md A4.

    Construct two batches with the SAME per-expert batch marginal but opposite
    per-sample sharpness: (a) every sample uniform over all E experts;
    (b) every sample one-hot, cycling so the marginal is still uniform.

    All three shipped aux losses score these (near-)identically, because all
    three are functions of batch marginals only. Nothing in the objective
    prefers (b). That is why routing entropy sits at 98.2% of its ceiling, and
    why re-enabling ``MOE_importance_coeff`` -- a third BALANCE term -- cannot
    fix flatness. Say so in the paper rather than letting a reviewer find it.
    """
    E = 8
    flat = torch.full((B, E), 1.0 / E)

    sharp = torch.zeros(B, E)
    for b in range(B):
        sharp[b, b % E] = 1.0
    assert torch.allclose(sharp.mean(0), flat.mean(0), atol=1e-6), (
        "the two batches must share a marginal for this test to mean anything"
    )

    if loss_name == "topk_MOE_aux_loss":
        a = topk_MOE_aux_loss(flat, flat, coeff=1.0).item()
        b = topk_MOE_aux_loss(sharp, sharp, coeff=1.0).item()
        # Switch loss uses sum(f*P); one-hot f concentrates it, so it actually
        # PENALISES sharpness -- the wrong direction for specialisation.
        assert b >= a - 1e-6, (
            "Switch loss should not reward per-sample sharpness"
        )
    else:
        fn = {"dense_MOE_aux_loss": dense_MOE_aux_loss,
              "importance_loss": importance_loss}[loss_name]
        a = fn(flat, coeff=1.0).item()
        b = fn(sharp, coeff=1.0).item()
        assert a == pytest.approx(b, abs=1e-5), (
            f"{loss_name} distinguishes flat from sharp per-sample routing "
            f"({a:.6f} vs {b:.6f}); if this now fails, a sharpness term was "
            f"added and LIMITATIONS.md A4 needs updating"
        )


def test_routing_mutual_information_would_distinguish_them():
    """
    The gap the plan fills, computed inline so the target is unambiguous.

    I(x;e) = H(E[w]) - E[H(w)] decomposes exactly into the balance term the
    code already has and the sharpness term it lacks. On the two batches above
    it is 0 for the flat one and log(E) for the sharp one -- the discrimination
    every shipped loss misses.
    """
    E = 8
    eps = 1e-9

    def mi(w):
        marginal = w.mean(0)
        h_marg = -(marginal * (marginal + eps).log()).sum()
        h_cond = -(w * (w + eps).log()).sum(-1).mean()
        return (h_marg - h_cond).item()

    flat = torch.full((B, E), 1.0 / E)
    sharp = torch.zeros(B, E)
    for b in range(B):
        sharp[b, b % E] = 1.0

    assert mi(flat) == pytest.approx(0.0, abs=1e-5)
    assert mi(sharp) == pytest.approx(math.log(E), abs=1e-3)
