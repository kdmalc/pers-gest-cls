"""
tests/test_arch_variants.py
===========================
Behaviour of the architecture-exploration code (docs/SPEC_architecture_exploration.md):
the MoE variants inside M0's class (X2 uniform gate, X3 z-loss / sparse skip, X4 task routing),
FlexNet (fusion, transformers), the metric learners, ADANN, the routing diagnostics, and the HPO
wiring of every ARCH_VARIANTS token. functional_call fidelity of each model is covered by
test_models_functional.py (ARCH_VARIANT_KINDS).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from test_models_functional import arch_keys

B = 6


def _x(cfg, n=B, seed=7):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, cfg["emg_in_ch"], cfg["sequence_length"], generator=g),
            torch.randn(n, cfg["imu_in_ch"], cfg["sequence_length"], generator=g))


def _moe(cfg, seed=1234, **over):
    from MOE.MOE_encoder import build_MOE_model
    torch.manual_seed(seed)
    c = arch_keys(dict(cfg))
    c.update(use_MOE=True, model_type="DeepCNNLSTM", MOE_placement="encoder", **over)
    return build_MOE_model(c), c


def _episode(cfg, n_way=3, n_q=2, seed=0):
    """A hand-built episode with disjoint repetitions (support rep 0, query reps 1..n_q)."""
    g = torch.Generator().manual_seed(seed)
    C, Ci, T = cfg["emg_in_ch"], cfg["imu_in_ch"], cfg["sequence_length"]

    def part(n_per, rep0):
        y = torch.arange(n_way).repeat_interleave(n_per)
        n = len(y)
        return {"emg": torch.randn(n, C, T, generator=g) + y.view(-1, 1, 1).float(),
                "imu": torch.randn(n, Ci, T, generator=g), "labels": y,
                "global_labels": y.clone(), "demo": torch.zeros(n, 1),
                "rep_idx": torch.arange(n_per).repeat(n_way) + rep0}
    return {"support": part(1, 0), "query": part(n_q, 1), "user_id": "P_SYN", "episode_id": 0}


# ─────────────────────────────────────────────────────────────────────────────
# M0 class: defaults unchanged, X2 / X3 / X4 / T2
# ─────────────────────────────────────────────────────────────────────────────
def test_m0_without_new_keys_matches_m0_with_explicit_defaults(tiny_config):
    """Old checkpoints / configs (no exploration keys) build exactly the model the explicit defaults build."""
    from MOE.MOE_encoder import build_MOE_model
    old = dict(tiny_config, use_MOE=True, model_type="DeepCNNLSTM", MOE_placement="encoder")
    torch.manual_seed(1234)
    m_old = build_MOE_model(old).eval()
    m_new, _ = _moe(tiny_config)
    m_new.eval()
    assert list(m_old.state_dict()) == list(m_new.state_dict())
    for (k, a), (_, b) in zip(m_old.state_dict().items(), m_new.state_dict().items()):
        assert torch.equal(a, b), k
    x = _x(tiny_config)
    assert torch.equal(m_old(*x), m_new(*x))


def test_x2_uniform_gate_has_no_router_and_weights_1_over_E(tiny_config):
    m, c = _moe(tiny_config, MOE_gate_mode="uniform", MOE_top_k=tiny_config["num_experts"])
    assert not any(n.startswith(("router.", "gate.")) for n, _ in m.named_parameters())
    _, info = m.eval()(*_x(tiny_config), return_routing=True)
    E = c["num_experts"]
    assert torch.allclose(info["gate_weights"], torch.full((B, E), 1.0 / E))


def test_x2_uniform_gate_rejects_sparse_top_k(tiny_config):
    with pytest.raises(ValueError, match="dense"):
        _moe(tiny_config, MOE_gate_mode="uniform", MOE_top_k=1)


def test_x3_router_logits_feed_the_z_loss(tiny_config):
    from MAML.mamlpp import _compute_aux_loss
    m, c = _moe(tiny_config, MOE_top_k=1, MOE_router_z_coeff=1e-2)
    _, info = m.train()(*_x(tiny_config), return_routing=True)
    assert info["router_logits"].shape == (B, c["num_experts"])
    with_z = _compute_aux_loss(info, c, aux_coeff=0.0)
    assert float(with_z) > 0
    with_z.backward()
    assert m.gate.linear.weight.grad is not None and m.gate.linear.weight.grad.abs().sum() > 0
    c0 = dict(c, MOE_router_z_coeff=0.0)
    assert float(_compute_aux_loss(info, c0, aux_coeff=0.0)) == 0.0


def test_x3_skipping_unselected_experts_does_not_change_the_forward(tiny_config):
    m_skip, _ = _moe(tiny_config, MOE_top_k=1, MOE_skip_unselected_experts=True)
    m_full, _ = _moe(tiny_config, MOE_top_k=1, MOE_skip_unselected_experts=False)
    x = _x(tiny_config)
    assert torch.allclose(m_skip.eval()(*x), m_full.eval()(*x), atol=1e-6)


def test_x4_task_routing_needs_and_uses_the_support_set(tiny_config):
    from MAML.mamlpp import task_support
    m, _ = _moe(tiny_config, MOE_routing_signal="context_proj_task")
    m.eval()
    x = _x(tiny_config)
    with pytest.raises(RuntimeError, match="support set"):
        m(*x)
    s1 = {"emg": _x(tiny_config, 3, seed=1)[0], "imu": _x(tiny_config, 3, seed=1)[1]}
    s2 = {"emg": _x(tiny_config, 3, seed=2)[0] * 3, "imu": _x(tiny_config, 3, seed=2)[1]}
    with task_support(m, s1, "cpu"):
        out1 = m(*x)
    with task_support(m, s2, "cpu"):
        out2 = m(*x)
    assert m._task_support is None                       # cleared on exit
    assert not torch.allclose(out1, out2)                # the support set changes query routing


def test_x4_adapt_and_eval_runs_end_to_end(tiny_config):
    from MAML.mamlpp import mamlpp_adapt_and_eval
    m, c = _moe(tiny_config, MOE_routing_signal="context_proj_task")
    ep = _episode(c)
    out = mamlpp_adapt_and_eval(m, c, ep["support"], ep["query"])
    assert 0.0 <= out["acc"] <= 1.0


def test_t2_transformer_temporal_model(tiny_config):
    m, c = _moe(tiny_config, temporal_model="transformer")
    assert not hasattr(m, "lstm1") and m.feat_dim == c["tf_d_model"]
    assert m.eval()(*_x(tiny_config)).shape == (B, c["n_way"])


# ─────────────────────────────────────────────────────────────────────────────
# FlexNet
# ─────────────────────────────────────────────────────────────────────────────
def _flex(cfg, **over):
    from arch_variants.flexnet import FlexNet
    torch.manual_seed(1234)
    c = arch_keys(dict(cfg))
    c.update({"use_MOE": False, **over})
    return FlexNet(c), c


def test_flexnet_early_lstm_has_deepcnnlstm_parameter_count(tiny_config):
    """F0 in FlexNet is the A4 layout, so fusion variants are compared like for like."""
    from pretraining.pretrain_models import DeepCNNLSTM
    flex, c = _flex(tiny_config)
    ref = DeepCNNLSTM(dict(c, model_type="DeepCNNLSTM"))
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(flex) == n(ref)


def test_film_starts_at_identity(tiny_config):
    flex, _ = _flex(tiny_config, fusion_mode="film")
    xe, xi = _x(tiny_config)
    enc = flex.encoder
    assert torch.equal(enc(xe, xi), enc.emg_cnn(xe))


@pytest.mark.parametrize("mode", ["mid", "film", "xattn"])
def test_fusion_modes_use_both_modalities(tiny_config, mode):
    flex, _ = _flex(tiny_config, fusion_mode=mode)
    if mode == "film":
        torch.nn.init.normal_(flex.encoder.film.weight, 0, 0.1)
    flex.eval()
    xe, xi = _x(tiny_config)
    base = flex(xe, xi)
    assert not torch.allclose(base, flex(xe, xi * 0 + 1.0))       # IMU matters
    assert not torch.allclose(base, flex(xe * 0 + 1.0, xi))       # EMG matters


def test_fusion_without_imu_raises(tiny_config):
    with pytest.raises(ValueError, match="use_imu"):
        _flex(tiny_config, fusion_mode="mid", use_imu=False)


def test_t4_token_moe_routing_and_aux_loss(tiny_config):
    from MAML.mamlpp import _compute_aux_loss
    flex, c = _flex(tiny_config, fusion_mode="none", temporal_model="transformer", tf_moe_every=1,
                    use_MOE=True, MOE_top_k=1, MOE_router_z_coeff=1e-3)
    logits, info = flex.train()(*_x(tiny_config), return_routing=True)
    T, E, nb = c["sequence_length"], c["num_experts"], c["tf_blocks"]
    assert info["gate_weights"].shape == (B * T * nb, E)
    assert ((info["gate_weights"] > 0).sum(-1) == 1).all()
    assert float(_compute_aux_loss(info, c, aux_coeff=0.01)) > 0


def test_moe_ffn_skipping_matches_explicit_mixture():
    from arch_variants.attention import MoEFeedForward
    torch.manual_seed(0)
    ff = MoEFeedForward(8, 16, num_experts=4, top_k=1, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)
    y, info = ff(x)
    flat = x.reshape(10, 8)
    ref = sum(info["gate_weights"][:, e:e + 1] * ff.experts[e](flat) for e in range(4)).reshape(2, 5, 8)
    assert torch.allclose(y, ref, atol=1e-6)


def test_flexnet_use_moe_without_a_bank_raises(tiny_config):
    with pytest.raises(ValueError, match="no expert bank"):
        _flex(tiny_config, fusion_mode="none", temporal_model="transformer", use_MOE=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metric learners
# ─────────────────────────────────────────────────────────────────────────────
def _metric(cfg, kind, backbone="cnnlstm"):
    from arch_variants.metric_learners import build_metric_learner
    torch.manual_seed(0)
    c = arch_keys(dict(cfg))
    c.update(metric_kind=kind, metric_backbone=backbone, metric_relation_hidden=16,
             use_MOE=(backbone == "moe"), MOE_placement="encoder")
    return build_metric_learner(c), c


@pytest.mark.parametrize("kind", ["proto", "matching", "relation", "ridge", "feat"])
def test_metric_learner_logits_and_learning(tiny_config, kind):
    from arch_variants.metric_learners import _episode_loss, metric_adapt_and_eval
    model, c = _metric(tiny_config, kind)
    ep = _episode(c, n_way=3, n_q=2)
    logits, y, _ = model.episode_logits(ep["support"], ep["query"], "cpu")
    assert logits.shape == (6, 3) and torch.isfinite(logits).all()
    out = metric_adapt_and_eval(model, c, ep["support"], ep["query"])
    assert 0.0 <= out["acc"] <= 1.0
    # a few steps on one episode must lower its loss (the head / metric is trainable end to end)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    first = None
    for _ in range(15):
        loss, _, _ = _episode_loss(model, ep, c, crit)
        first = float(loss) if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first


def test_metric_learner_on_moe_backbone_adds_aux_loss(tiny_config):
    from arch_variants.metric_learners import _episode_loss
    model, c = _metric(tiny_config, "proto", backbone="moe")
    ep = _episode(c)
    _, _, routing = model.episode_logits(ep["support"], ep["query"], "cpu")
    assert routing is not None and routing["gate_weights"].shape[1] == c["num_experts"]
    loss, _, _ = _episode_loss(model, ep, c, torch.nn.CrossEntropyLoss())
    assert torch.isfinite(loss)


def test_metric_learner_has_no_plain_forward(tiny_config):
    model, c = _metric(tiny_config, "proto")
    with pytest.raises(RuntimeError, match="episodic"):
        model(*_x(c))


# ─────────────────────────────────────────────────────────────────────────────
# ADANN
# ─────────────────────────────────────────────────────────────────────────────
def test_grad_reverse_negates_and_scales():
    from arch_variants.adann import grad_reverse
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def test_adann_pretrain_runs_and_returns_the_classifier(tiny_config, reoriented_tensor_dict):
    from arch_variants.adann import adann_pretrain
    from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
    from pretraining.pretrain_models import build_model
    c = dict(tiny_config, meta_learning=False, use_MOE=False, adann_lambda=0.1, adann_disc_hidden=8,
             available_gesture_classes=list(range(10)), train_reps=[1, 2, 3], val_reps=[4],
             batch_size=8, num_epochs=1, earlystopping_patience=3, earlystopping_min_delta=0.0)
    train_dl, val_dl, _ = get_pretrain_dataloaders(c, tensor_dict=reoriented_tensor_dict)
    assert "users" in next(iter(train_dl))
    torch.manual_seed(0)
    model = build_model(c)
    out, hist = adann_pretrain(model, train_dl, val_dl, c)
    assert out is model and len(hist["val_acc"]) == 1 and len(hist["train_dom_acc"]) == 1


def test_flat_loader_has_no_users_without_adann(tiny_config, reoriented_tensor_dict):
    from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
    c = dict(tiny_config, available_gesture_classes=list(range(10)), train_reps=[1, 2], val_reps=[3],
             batch_size=4)
    train_dl, _, _ = get_pretrain_dataloaders(c, tensor_dict=reoriented_tensor_dict)
    assert "users" not in next(iter(train_dl))


# ─────────────────────────────────────────────────────────────────────────────
# Routing diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def test_routing_summary_uniform_is_pure_ensembling():
    from MOE.routing_metrics import routing_summary
    w = np.full((40, 8), 1 / 8)
    s = routing_summary(w, w, user_ids=[i % 4 for i in range(40)], gesture_ids=[i % 5 for i in range(40)])
    assert s["soft_entropy_norm_mean"] == pytest.approx(1.0)
    assert s["usage_entropy_norm"] == pytest.approx(1.0)
    assert s["mi_user"] == pytest.approx(0.0, abs=1e-9)
    assert s["dead_expert_frac"] == 0.0


def test_routing_summary_detects_user_specialisation():
    from MOE.routing_metrics import routing_summary
    users = np.repeat(np.arange(4), 25)
    w = np.eye(4)[users]                     # each user routed to its own expert
    s = routing_summary(w, w, user_ids=users, gesture_ids=np.tile(np.arange(5), 20))
    assert s["mi_user_norm"] == pytest.approx(1.0)
    assert s["mi_user_null"] < 0.2 * s["mi_user"]
    assert s["mi_gesture_within_user"] == pytest.approx(0.0, abs=1e-9)
    assert s["soft_entropy_norm_mean"] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# HPO wiring: every ARCH_VARIANTS token builds a config, a model and runs a forward
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def hpo_module():
    os.environ.setdefault("HPO_DB_DIR", tempfile.mkdtemp(prefix="pgc-test-hpo-"))
    os.environ.setdefault("EMG_REPR", "fs64")
    import ablation_hpo
    return ablation_hpo


def _variant_ids():
    # importing ablation_hpo at collection time would need HPO_DB_DIR; list the ids statically
    return ["X2", "X3", "X4", "X5", "X6", "F1", "F2", "F3", "T0", "T2", "T4",
            "P1", "P1_moe", "P1_match", "P1_rel", "P1_ridge", "P1_feat", "P3", "A4_ANIL", "M0_ANIL"]


def test_variant_id_list_matches_the_driver(hpo_module):
    assert sorted(_variant_ids()) == sorted(hpo_module.ARCH_VARIANTS)


@pytest.mark.parametrize("vid", _variant_ids())
def test_every_variant_builds_and_runs(hpo_module, vid):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    profile = hpo_module.ABLATION_PROFILES[vid]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=3))
    trial = study.ask()
    cfg = hpo_module.build_config_from_trial(trial, vid, profile)
    cfg["device"] = torch.device("cpu")
    # small CPU sizes; the variant's own structure is untouched
    cfg.update(cnn_base_filters=8, lstm_hidden=8, groupnorm_num_groups=4,
               num_experts=min(cfg.get("num_experts", 2), 3), tf_d_model=16, tf_heads=4,
               fusion_d_model=8, MOE_xattn_d_model=8)
    if cfg.get("MOE_top_k") is not None:
        cfg["MOE_top_k"] = cfg["top_k"] = (cfg["num_experts"] if cfg.get("MOE_gate_mode") == "uniform"
                                           else min(cfg["MOE_top_k"], cfg["num_experts"]))
    model = hpo_module.build_model(cfg)
    assert hpo_module.LAST_MODEL_N_PARAMS > 0
    x = (torch.randn(4, cfg["emg_in_ch"], cfg["sequence_length"]),
         torch.randn(4, cfg["imu_in_ch"], cfg["sequence_length"]))
    model.eval()
    if cfg["arch_family"] == "metric":
        ep = _episode(cfg, n_way=3, n_q=1)
        logits, _, _ = model.episode_logits(ep["support"], ep["query"], "cpu")
        assert logits.shape == (3, 3)
        return
    from MAML.mamlpp import task_support
    with task_support(model, {"emg": x[0], "imu": x[1]}, "cpu"):
        out = model(*x)
    logits = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(logits).all()
    if vid.endswith("ANIL"):
        assert cfg["maml_inner_param_include"] == ["head"]
    if vid == "P3":
        assert 0.01 <= cfg["adann_lambda"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Training loops end to end on cached (synthetic-data) episodes
# ─────────────────────────────────────────────────────────────────────────────
from test_eval_functions import fold0_setup  # noqa: E402,F401  (fixture)


def _loaders(fold0_setup, **over):
    from MAML.maml_data_pipeline import get_maml_dataloaders
    cfg, path, f = fold0_setup
    cfg = arch_keys(dict(cfg, num_workers=0, seed=1, num_epochs=1, episodes_per_epoch_train=4,
                         meta_batchsize=2, MOE_log_every=1, use_earlystopping=False,
                         earlystopping_patience=3, earlystopping_min_delta=0.0, **over))
    train_dl, val_dl = get_maml_dataloaders(cfg, tensor_dict_path=path)
    return cfg, train_dl, val_dl


@pytest.mark.parametrize("kind", ["proto", "feat"])
def test_metric_pretrain_one_epoch(fold0_setup, kind):
    from arch_variants.metric_learners import build_metric_learner, metric_pretrain
    cfg, train_dl, val_dl = _loaders(fold0_setup, use_MOE=False)
    cfg.update(metric_kind=kind, metric_backbone="cnnlstm", label_smooth=0.0)
    torch.manual_seed(0)
    model = build_metric_learner(cfg)
    seen = []
    _, hist = metric_pretrain(model, cfg, train_dl, episodic_val_loader=val_dl,
                              epoch_callback=lambda ep, acc: seen.append((ep, acc)))
    assert len(hist["val_acc_log"]) == 1 and seen and 0.0 <= seen[0][1] <= 1.0
    assert hist["best_state"] is not None


@pytest.mark.parametrize("variant", ["X4", "T4"])
def test_maml_pretrain_and_routing_diagnostics(fold0_setup, variant):
    """The MAML++ loop with task-conditioned routing (X4) and token-level MoE (T4), then routing metrics."""
    from MAML.mamlpp import mamlpp_pretrain
    from MOE.routing_metrics import collect_episode_routing
    cfg, train_dl, val_dl = _loaders(fold0_setup, use_MOE=True)
    if variant == "X4":
        model, cfg = _moe(cfg, MOE_routing_signal="context_proj_task")
    else:
        from arch_variants.flexnet import FlexNet
        cfg.update(fusion_mode="none", temporal_model="transformer", tf_moe_every=1, MOE_top_k=1,
                   MOE_router_z_coeff=1e-3, MOE_skip_unselected_experts=True)
        torch.manual_seed(0)
        model = FlexNet(cfg)
    _, hist = mamlpp_pretrain(model, cfg, train_dl, episodic_val_loader=val_dl)
    assert len(hist["val_acc_log"]) == 1
    rs = collect_episode_routing(model, cfg, val_dl, adapted=True)
    assert rs is not None and "mi_user" in rs and 0.0 <= rs["soft_entropy_norm_mean"] <= 1.0 + 1e-6


@pytest.mark.parametrize("vid", ["P1", "X2"])
def test_hpo_objective_end_to_end_records_timing(fold0_setup, hpo_module, monkeypatch, capsys, vid):
    """One real Optuna trial through ablation_hpo.objective on synthetic data (fold-0 PIDs, cached val
    episodes): dispatch, objective, routing diagnostics and the never-collapsing uniform gate (X2), and the per-trial timing record."""
    import optuna
    import ablation_config
    cfg0, path, f = fold0_setup
    hpo = hpo_module
    real_build = hpo.build_config_from_trial

    def small_build(trial, ablation_id, profile):
        c = real_build(trial, ablation_id, profile)
        c.update(device=torch.device("cpu"), emg_in_ch=cfg0["emg_in_ch"], imu_in_ch=cfg0["imu_in_ch"],
                 sequence_length=cfg0["sequence_length"], cnn_base_filters=8, lstm_hidden=8,
                 groupnorm_num_groups=4, num_epochs=1, episodes_per_epoch_train=2, meta_batchsize=2,
                 num_workers=0, n_way=3, num_classes=3, maml_inner_steps=1, maml_inner_steps_eval=1,
                 num_experts=min(c.get("num_experts", 2), 3), MOE_log_every=1)
        if c.get("MOE_top_k") is not None:
            c["MOE_top_k"] = c["top_k"] = (c["num_experts"] if c["MOE_gate_mode"] == "uniform"
                                           else min(c["MOE_top_k"], c["num_experts"]))
        return c

    def fold(config):
        config.update(train_PIDs=f["train"][:3], val_PIDs=f["val"], test_PIDs=[],
                      val_episode_source="val", fold_idx=0)
        return config

    monkeypatch.setattr(hpo, "build_config_from_trial", small_build)
    monkeypatch.setattr(hpo, "apply_fold_to_config", fold)
    monkeypatch.setattr(ablation_config, "tensor_dict_file", lambda config: path)
    monkeypatch.setattr(hpo, "RUN_DIR", __import__("pathlib").Path(tempfile.mkdtemp()))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: hpo.objective(t, vid, hpo.ABLATION_PROFILES[vid]), n_trials=1)
    t = study.trials[0]
    assert t.state.name == "COMPLETE", t.state
    assert 0.0 <= t.value <= 1.0
    assert t.user_attrs["state"] == "COMPLETE" and t.user_attrs["wall_time_s"] > 0
    assert t.user_attrs["n_params"] > 0
    out = capsys.readouterr().out
    assert "HPO_TRIAL_TIMING" in out
    if vid == "X2":
        assert "COLLAPSED" not in out and t.value > 0
        assert t.user_attrs["routing"]["soft_entropy_norm_mean"] == pytest.approx(1.0)


def test_fixed_params_are_frozen_and_validated(hpo_module, monkeypatch, tmp_path):
    """HPO_FIXED_PARAMS_JSON: frozen names are not sampled, out-of-range or unused names raise."""
    import json
    import optuna
    hpo = hpo_module
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    f = tmp_path / "fixed.json"

    def build(fixed, vid="X2"):
        f.write_text(json.dumps(fixed))
        monkeypatch.setenv("HPO_FIXED_PARAMS_JSON", str(f))
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        return hpo.build_config_from_trial(trial, vid, hpo.ABLATION_PROFILES[vid]), trial

    cfg, trial = build({"outer_lr": 1e-4, "maml_inner_steps": 7})
    assert cfg["learning_rate"] == 1e-4 and cfg["maml_inner_steps"] == 7
    assert "outer_lr" not in trial.params and "num_experts" in trial.params
    assert trial.user_attrs["fixed_params"] == {"outer_lr": 1e-4, "maml_inner_steps": 7}
    with pytest.raises(ValueError, match="outside"):
        build({"maml_inner_steps": 3})                      # not in {5, 7, 10, 15}
    with pytest.raises(SystemExit, match="not searched"):
        build({"ft_lr": 1e-3})                              # X2 has no fine-tune HPs
