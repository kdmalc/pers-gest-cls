"""
The running-sum expert mixing (docs/SPEC_moe_compute.md section 1) is numerically the same
function as the old stack -> weight -> sum(dim=1), forward AND gradients, in float64.
"""
import pytest
import torch

import MOE.MOE_encoder as me
from test_models_functional import BATCH, _build


def _stack_version(calls, w_hard):
    """The old dense implementation (reference)."""
    outs = [c() for c in calls]
    stacked = torch.stack(outs, dim=1)
    w = w_hard.reshape(w_hard.shape[0], w_hard.shape[1], *([1] * (outs[0].dim() - 1)))
    return (stacked * w).sum(dim=1)


def _build_meta(cfg, placement):
    from MOE.MOE_encoder import build_MOE_model
    torch.manual_seed(1234)
    cfg = dict(cfg, use_MOE=True, model_type="MetaCNNLSTM", MOE_placement=placement)
    cfg.setdefault("cnn_filters", cfg["cnn_base_filters"])
    return build_MOE_model(cfg)


KINDS = ["DeepCNNLSTM_EncoderMOE", "DeepCNNLSTM_MiddleMOE", "Meta_Encoder", "Meta_Middle"]


def _make(kind, cfg):
    if kind == "Meta_Encoder":
        return _build_meta(cfg, "encoder")
    if kind == "Meta_Middle":
        return _build_meta(cfg, "middle")
    return _build(kind, cfg)


def _run(model, cfg):
    torch.manual_seed(7)
    x_emg = torch.randn(BATCH, cfg["emg_in_ch"], cfg["sequence_length"], dtype=torch.float64)
    x_imu = torch.randn(BATCH, cfg["imu_in_ch"], cfg["sequence_length"], dtype=torch.float64)
    model.zero_grad()
    out = model(x_emg, x_imu)
    logits = out[0] if isinstance(out, tuple) else out
    logits.pow(2).sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach().clone(), grads


@pytest.mark.parametrize("kind", KINDS)
def test_running_sum_equals_dense_stack_forward_and_grads(kind, tiny_config, monkeypatch):
    model = _make(kind, tiny_config).double().eval()
    new_logits, new_grads = _run(model, tiny_config)

    monkeypatch.setattr(me, "_weighted_expert_sum", _stack_version)
    old_logits, old_grads = _run(model, tiny_config)

    assert torch.allclose(new_logits, old_logits, atol=1e-10, rtol=1e-10)
    assert new_grads.keys() == old_grads.keys()
    for n in new_grads:
        assert torch.allclose(new_grads[n], old_grads[n], atol=1e-9, rtol=1e-8), n


def test_weighted_expert_sum_handles_zero_weights_and_dims():
    w = torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    outs3 = [torch.full((2, 4, 5), float(i + 1)) for i in range(3)]
    got = me._weighted_expert_sum([lambda o=o: o for o in outs3], w)
    ref = _stack_version([lambda o=o: o for o in outs3], w)
    assert torch.equal(got, ref) and got.shape == (2, 4, 5)
    outs2 = [torch.full((2, 7), float(i + 1)) for i in range(3)]
    assert torch.allclose(me._weighted_expert_sum([lambda o=o: o for o in outs2], w),
                          _stack_version([lambda o=o: o for o in outs2], w))
