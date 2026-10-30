"""
tests/test_eval_builds.py
=========================
The pieces built for the paper runs on 2026-10-30:
  * EMG-RAW: the learned strided stem (all three model families) and the raw3s crop;
  * the variant registry, shared by the HPO driver, the eval script and kfold_launcher.sh;
  * EMG_REPR actually reaching the HPO config, and the repr check on tuned overrides;
  * the screen's frozen-HP files;
  * the --seed tag;
  * arch_variant_eval.py end to end on synthetic data (MAML, metric and supervised tokens);
  * zero-shot Z1-Z4 helpers: transfer is detected when it exists, and chance when it does not;
  * the A16 leave-one-rep-out fine-tune.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from test_models_functional import arch_keys

REPO = Path(__file__).resolve().parents[1]
ABL = REPO / "system" / "NOTS" / "paper" / "ablations"


# ─────────────────────────────────────────────────────────────────────────────
# EMG-RAW stem
# ─────────────────────────────────────────────────────────────────────────────
def _stem_cfg(tiny_config, **over):
    c = arch_keys(dict(tiny_config))
    c.update(emg_stem="raw_strided", **over)
    return c


def _raw_inputs(cfg, B=4, L=6144):
    torch.manual_seed(3)
    return (torch.randn(B, cfg["emg_in_ch"], L).half(), torch.randn(B, cfg["imu_in_ch"], cfg["sequence_length"]))


def _families(cfg):
    from pretraining.pretrain_models import DeepCNNLSTM
    from MOE.MOE_encoder import build_MOE_model
    from arch_variants.flexnet import FlexNet
    torch.manual_seed(0)
    yield "DeepCNNLSTM", DeepCNNLSTM(dict(cfg, use_MOE=False, model_type="DeepCNNLSTM"))
    yield "EncoderMOE", build_MOE_model(dict(cfg, use_MOE=True, model_type="DeepCNNLSTM", MOE_placement="encoder"))
    yield "FlexNet_mid", FlexNet(dict(cfg, use_MOE=False, fusion_mode="mid"))


def test_raw_stem_forward_and_gradients_in_every_family(tiny_config):
    from MAML.mamlpp import FunctionalModel, named_param_dict
    cfg = _stem_cfg(tiny_config)
    x = _raw_inputs(cfg)
    for name, model in _families(cfg):
        assert model.emg_stem is not None, name
        out = model.eval()(*x)
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.shape == (4, cfg["n_way"]) and torch.isfinite(logits).all(), name
        # the stem is adapted through functional_call like everything else (MAML inner loop)
        params = OrderedDict((n, p.detach().clone().requires_grad_(True)) for n, p in named_param_dict(model).items())
        o = FunctionalModel(model.train(), params)(*x)
        loss = (o[0] if isinstance(o, tuple) else o).sum()
        stem = [n for n in params if n.startswith("emg_stem.")]
        grads = torch.autograd.grad(loss, [params[n] for n in stem], allow_unused=True)
        assert stem and all(g is not None and g.abs().sum() > 0 for g in grads), name


def test_no_stem_by_default_keeps_the_old_state_dict(tiny_config):
    from pretraining.pretrain_models import DeepCNNLSTM
    cfg = arch_keys(dict(tiny_config))
    m = DeepCNNLSTM(dict(cfg, use_MOE=False, model_type="DeepCNNLSTM", emg_stem="none"))
    assert m.emg_stem is None and not any(k.startswith("emg_stem") for k in m.state_dict())


def test_raw_stem_rejects_front_end_stride(tiny_config):
    from MOE.MOE_encoder import build_MOE_model
    with pytest.raises(ValueError, match="exclusive"):
        build_MOE_model(dict(_stem_cfg(tiny_config), use_MOE=True, model_type="DeepCNNLSTM",
                             MOE_placement="encoder", front_end_stride=4))


def test_raw3s_crop_is_centred_on_activity_and_zero_padded():
    import sys
    sys.path.insert(0, str(REPO / "system" / "universal_preprocessing" / "EMG_preprocessing"))
    from build_raw3s_tensor_dict import crop_window
    fs = 2048
    L = 4000                                   # shorter than the 6144 window
    x = np.full((L, 2), 0.01)                  # low background everywhere ...
    x[3000:3200] = 1.0                         # ... and a burst near the end
    w = crop_window(x, fs)
    assert w.shape == (6144, 2)
    burst = np.nonzero(w[:, 0] == 1.0)[0]
    assert abs(burst.mean() - 3072) < 150     # the burst sits near the middle of the window
    # centring on the burst (centroid ~3100) puts the window at trial samples 28..6172: the first ~28
    # trial samples fall outside, and everything after the trial end is zero padding
    start = int(np.nonzero(x[:, 0] == 1.0)[0].mean() + 0.5) - 3072
    assert 0 < start < 100
    n_in = int((w[:, 0] != 0).sum())                  # trial samples inside the window
    assert abs(n_in - (L - start)) <= 10             # (the background shifts the centroid slightly)
    assert (w[n_in:, 0] == 0).all() and (w[:n_in, 0] != 0).all()


def test_apply_emg_repr_sets_the_stem():
    from ablation_config import apply_emg_repr
    assert apply_emg_repr({}, "raw3s")["emg_stem"] == "raw_strided"
    assert apply_emg_repr({}, "fs64")["emg_stem"] == "none"
    assert apply_emg_repr({}, "fs64")["emg_in_ch"] == 112


# ─────────────────────────────────────────────────────────────────────────────
# registry, launcher, HPO config
# ─────────────────────────────────────────────────────────────────────────────
def test_launcher_variant_tokens_match_the_registry():
    from arch_variants.registry import ARCH_VARIANTS
    text = (ABL / "launchers" / "kfold_launcher.sh").read_text(encoding="utf-8")
    m = re.search(r'^ARCH_VARIANT_TOKENS="([^"]+)"', text, re.M)
    assert m and sorted(m.group(1).split()) == sorted(ARCH_VARIANTS)


@pytest.fixture(scope="module")
def hpo():
    os.environ.setdefault("HPO_DB_DIR", tempfile.mkdtemp(prefix="pgc-test-hpo-"))
    os.environ.setdefault("EMG_REPR", "fs64")
    import ablation_hpo
    return ablation_hpo


def _ask_config(hpo, token, monkeypatch, fixed=None, repr_="fs64", tmp=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setenv("EMG_REPR", repr_)
    if fixed is not None:
        monkeypatch.setenv("HPO_FIXED_PARAMS_JSON", str(fixed))
    else:
        monkeypatch.delenv("HPO_FIXED_PARAMS_JSON", raising=False)
    trial = optuna.create_study(direction="maximize").ask()
    return hpo.build_config_from_trial(trial, token, hpo.ABLATION_PROFILES[token])


def test_hpo_trains_on_the_requested_emg_repr(hpo, monkeypatch):
    assert _ask_config(hpo, "M0", monkeypatch, repr_="fs64")["emg_in_ch"] == 112
    c = _ask_config(hpo, "A4", monkeypatch, repr_="raw3s")
    assert c["emg_repr"] == "raw3s" and c["emg_stem"] == "raw_strided"


def test_hpo_registry_is_the_one_registry(hpo):
    from arch_variants import registry
    assert hpo.ARCH_VARIANTS is registry.ARCH_VARIANTS


def test_a2_hpo_uses_the_param_matched_widths(hpo, monkeypatch):
    from arch_variants.registry import A4_CNN_BASE_FILTERS, A4_LSTM_HIDDEN
    c = _ask_config(hpo, "A2", monkeypatch)
    assert (c["cnn_base_filters"], c["lstm_hidden"]) == (A4_CNN_BASE_FILTERS, A4_LSTM_HIDDEN)


@pytest.mark.parametrize("fname,tokens", [
    ("legacy_shared_maml.json", ["M0", "A4", "X2", "X3", "X4", "X5", "X6", "F1", "F2", "F3", "A4_ANIL", "M0_ANIL"]),
    ("legacy_inner_only.json", ["T0", "T2", "T4"]),
])
def test_screen_freeze_files_are_valid_for_their_tokens(hpo, monkeypatch, fname, tokens):
    path = ABL / "screen_fixed" / fname
    fixed = json.loads(path.read_text())
    for t in tokens:
        c = _ask_config(hpo, t, monkeypatch, fixed=path)
        assert c["hpo_fixed_params"] == fixed
        assert c["learning_rate"] == fixed["outer_lr"] if "outer_lr" in fixed else True


def test_tuned_override_from_another_repr_raises(monkeypatch, tmp_path):
    import ablation_config as ac
    cfgfile = tmp_path / "best.json"
    cfgfile.write_text(json.dumps({"M0": {"overrides": {"learning_rate": 1e-4},
                                          "provenance": {"study": "s", "trial": 0, "emg_repr": "env20"}}}))
    monkeypatch.setattr(ac, "HPO_CONFIG_PATH", cfgfile)
    c = ac.apply_emg_repr(ac.make_base_config("M0"), "fs64")
    with pytest.raises(ValueError, match="tuned on emg_repr"):
        ac.apply_hpo_config(c, "tuned")
    c = ac.apply_emg_repr(ac.make_base_config("M0"), "env20")
    assert ac.apply_hpo_config(c, "tuned")["learning_rate"] == 1e-4


def test_seed_tag():
    from ablation_config import kfold_tag
    assert kfold_tag(0) == kfold_tag(0, 42) == "fold00_summary"
    assert kfold_tag(3, 43) == "fold03_summary_seed43"
    assert kfold_tag(None, 44) == "summary_seed44"


# ─────────────────────────────────────────────────────────────────────────────
# arch_variant_eval.py end to end (synthetic data with the real fold-0 PIDs)
# ─────────────────────────────────────────────────────────────────────────────
from test_eval_functions import fold0_setup  # noqa: E402,F401  (fixture)


@pytest.mark.parametrize("token", ["X2", "P1_feat", "P3"])
def test_arch_variant_eval_one_fold(fold0_setup, monkeypatch, token):
    import arch_variant_eval as ave
    from ablation_config import apply_kfold_fold
    cfg0, path, f = fold0_setup
    monkeypatch.setattr(ave, "tensor_dict_file", lambda config: path)
    monkeypatch.setenv("EMG_REPR", "env20")
    config = ave.build_config(token, smoke=True)
    fold = apply_kfold_fold(config, 0)
    fold.update(device=torch.device("cpu"), emg_in_ch=cfg0["emg_in_ch"], imu_in_ch=cfg0["imu_in_ch"],
                cnn_base_filters=8, lstm_hidden=8, groupnorm_num_groups=4, num_experts=3,
                MOE_top_k=3 if fold["MOE_gate_mode"] == "uniform" else 2, top_k=3,
                train_PIDs=f["train"][:3], num_workers=0, num_epochs=1, episodes_per_epoch_train=2,
                meta_batchsize=2, maml_inner_steps=1, maml_inner_steps_eval=1, batch_size=16, ft_steps=2,
                MOE_ctx_out_dim=8, MOE_ctx_hidden_dim=8, train_reps=[1, 2, 3], val_reps=[4])
    res = ave.run_one_fold(token, "kfold7_fold0", 42, fold, cells=[(10, 1)])
    key = ave.MAIN_RESULT[ave.variant_kind(token)]
    assert 0.0 <= res[key]["mean_acc"] <= 1.0 and set(res[key]["per_user_acc"]) == set(f["test"])
    if token == "X2":
        assert res["routing_val"]["soft_entropy_norm_mean"] == pytest.approx(1.0)
    if token == "P3":
        assert "test_head_only" in res


def test_arch_variant_eval_refuses_overrides_that_change_the_token(fold0_setup):
    import arch_variant_eval as ave
    config = ave.build_config("F2", smoke=True)
    config["fusion_mode"] = "mid"
    with pytest.raises(ValueError, match="token defines"):
        ave._fold_consistency(config, "F2")


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _zs_dict(shared: bool, n_users=8, C=4, T=40, seed=0):
    """Reoriented-layout dict: (10 reps, C, T) per gesture. shared=True: every user performs gesture g
    the same way (class template shared); False: each user has their own random template per gesture."""
    rng = np.random.default_rng(seed)
    shared_t = rng.normal(size=(10, C, T))
    d = {}
    for u in range(n_users):
        own = rng.normal(size=(10, C, T))
        d[f"P{u:03d}"] = {g: {"emg": torch.tensor((shared_t[g] if shared else own[g])[None] * 2
                                                  + rng.normal(size=(10, C, T)) * 0.5, dtype=torch.float32),
                              "imu": torch.tensor(rng.normal(size=(10, 3, T)), dtype=torch.float32)}
                          for g in range(10)}
    return d


@pytest.mark.parametrize("shared,lo,hi", [(True, 0.8, 1.01), (False, 0.0, 0.25)])
def test_zero_shot_detects_transfer_only_when_it_exists(shared, lo, hi):
    from Z_zeroshot_crossuser import collect, nearest_centroid_cosine, summarise, z4_riemann
    d = _zs_dict(shared)
    pids = sorted(d)
    tr, te = collect(d, pids[:6]), collect(d, pids[6:])
    flat = lambda x: x["emg"].reshape(len(x["y"]), -1)
    acc_nc = summarise(nearest_centroid_cosine(flat(tr), tr["y"], flat(te)), te)["mean_acc"]
    assert lo <= acc_nc <= hi
    acc_z4 = summarise(z4_riemann(tr, te), te)
    assert set(acc_z4["per_user_acc"]) == set(pids[6:])
    if not shared:                                   # covariance carries no label info here either
        assert acc_z4["mean_acc"] <= 0.3
    s = summarise(np.zeros_like(te["y"]), te)
    assert s["per_gesture_acc"][0] == 1.0 and s["mean_acc"] == pytest.approx(0.1)
    assert np.asarray(s["confusion"]).sum() == len(te["y"])


# ─────────────────────────────────────────────────────────────────────────────
# A16 leave-one-rep-out
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_lo1o_runs_on_a_tiny_model(tiny_config):
    from A16_upper_bound_kfold import ft_config, lo1o_user
    from Z_zeroshot_crossuser import collect
    from pretraining.pretrain_models import DeepCNNLSTM
    d = _zs_dict(True, n_users=1, C=tiny_config["emg_in_ch"], T=tiny_config["sequence_length"])
    for g in d["P000"]:
        d["P000"][g]["imu"] = torch.randn(10, tiny_config["imu_in_ch"], tiny_config["sequence_length"])
    data = collect(d, ["P000"])
    cfg = dict(arch_keys(dict(tiny_config)), use_MOE=False, meta_learning=False, model_type="DeepCNNLSTM")
    torch.manual_seed(0)
    model = DeepCNNLSTM(cfg)
    acc = lo1o_user(model, ft_config(cfg, 1e-3, 5), data, "P000", [1, 2])
    assert 0.0 <= acc <= 1.0
