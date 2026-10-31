"""
registry.py
===========
Side-effect-free definitions shared by the HPO driver (ablation_hpo.py) and the 7-fold evaluation
script (test_eval_files/arch_variant_eval.py), so the two can never disagree on what a token means:

  ARCH_DEFAULTS / arch_defaults()  the architecture-exploration config keys at their M0 / A4 values.
                                   make_base_config() writes them into EVERY config, so tuned
                                   overrides of any variant name existing keys.
  A4 widths                        the param-matched no-MoE widths the HPO uses for A4-based tokens.
  ARCH_VARIANTS                    one entry per exploration token (docs/specs/SPEC_architecture_exploration.md).
  variant_kind()                   how a token trains and is evaluated: "maml" | "metric" | "supervised".
"""

from __future__ import annotations

import copy

# Param-matched encoder dims for A4 (and every A4-based token). See ablation_hpo.py for how they are
# meant to be derived from M0's bank.
A4_CNN_BASE_FILTERS: int = 128
A4_LSTM_HIDDEN: int = 128

ARCH_DEFAULTS: dict = dict(
    arch_family="legacy",
    MOE_gate_mode="learned", MOE_routing_signal="context_proj", MOE_use_shared_expert=False,
    MOE_router_z_coeff=0.0, MOE_skip_unselected_experts=False,
    MOE_xattn_d_model=64, MOE_xattn_heads=4,
    temporal_model="lstm",
    fusion_mode="early", fusion_branch_width=0.71, fusion_d_model=64, fusion_heads=4,
    tf_d_model=128, tf_heads=4, tf_blocks=3, tf_ff_mult=2, tf_moe_every=0,
    maml_inner_param_include=None,
    # Riemannian variants (section 2R; docs/analysis/COVARIANCE_TUTORIAL.md). Defaults = M0 / A4 unchanged.
    feature_pool="mean", cov_pool_dim=16, cov_pool_norm="sqrt", cov_shrinkage=0.1,
    input_align="none", MOE_cov_expert_whiten=False,
    spd_dims=[32, 16], spd_reeig_eps=1e-4,
)


def arch_defaults(config: dict) -> dict:
    """Write every architecture-exploration key at its M0 / A4 value (explicit, never a silent .get)."""
    config.update(copy.deepcopy(ARCH_DEFAULTS))
    return config


# Each variant is its OWN Optuna study with the same budget (never a categorical arm inside one
# study). "base" = the profile whose HP groups it reuses (M0: MAML+MoE, A4: MAML no MoE at the
# param-matched widths, A2: supervised, A1: supervised + MoE); "family" = model builder (legacy =
# pretrain_models / MOE_encoder, flexnet = arch_variants.flexnet, metric = arch_variants.metric_learners);
# "set" = fixed config overrides; "spaces" = extra HP groups (ablation_hpo._apply_variant).
ARCH_VARIANTS: dict[str, dict] = {
    # ── MoE structure (same expert bank as M0; one change each) ──────────────
    "X2":  dict(base="M0", family="legacy", set={"MOE_gate_mode": "uniform"}, spaces=("moe_uniform",),
                note="M0 with a fixed uniform gate (1/E, no router): routing or just ensembling?"),
    "X3":  dict(base="M0", family="legacy", set={"MOE_skip_unselected_experts": True}, spaces=("moe_sparse",),
                note="Sparse top-1/2 over a small bank + load balance + router z-loss (ST-MoE recipe)"),
    "X4":  dict(base="M0", family="legacy", set={"MOE_routing_signal": "context_proj_task"}, spaces=(),
                note="Task-conditioned routing: [sample context, support-set mean context]"),
    "X5":  dict(base="M0", family="legacy", set={"MOE_use_shared_expert": True}, spaces=(),
                note="Shared always-on expert + routed experts (DeepSeek-MoE style)"),
    "X6":  dict(base="M0", family="legacy", set={"MOE_routing_signal": "xattn"}, spaces=("xattn_router",),
                note="Router input = EMG tokens cross-attending to IMU tokens"),
    # ── EMG-IMU fusion, screened on the no-MoE MAML backbone (reference: A4 = early concat) ─
    "F1":  dict(base="A4", family="flexnet", set={"fusion_mode": "mid"}, spaces=("fusion",),
                note="Mid fusion: per-modality CNNs, features concatenated before the LSTM"),
    "F2":  dict(base="A4", family="flexnet", set={"fusion_mode": "film"}, spaces=("fusion",),
                note="FiLM: IMU CNN produces per-channel, per-step gamma/beta for the EMG CNN"),
    "F3":  dict(base="A4", family="flexnet", set={"fusion_mode": "xattn"}, spaces=("fusion", "fusion_xattn"),
                note="Cross-attention fusion: EMG tokens <-> IMU tokens"),
    # ── Transformers ─────────────────────────────────────────────────────────
    "T0":  dict(base="A4", family="flexnet", set={"fusion_mode": "none", "temporal_model": "transformer"},
                spaces=("tf",), note="Single transformer on the 64 time steps (no CNN, no MoE), MAML++"),
    "T2":  dict(base="M0", family="legacy", set={"temporal_model": "transformer"}, spaces=("tf",),
                note="M0's CNN expert bank, LSTM replaced by a transformer"),
    "T4":  dict(base="M0", family="flexnet",
                set={"fusion_mode": "none", "temporal_model": "transformer", "tf_moe_every": 2,
                     "MOE_skip_unselected_experts": True},
                spaces=("tf", "moe_sparse"), note="Transformer with sparse MoE feed-forward layers (Switch-style)"),
    # ── Riemannian / covariance variants (section 2R) ────────────────────────
    "X7":  dict(base="M0", family="legacy", set={"MOE_routing_signal": "cov_tangent"}, spaces=(),
                note="Router input = log-Euclidean tangent vector of each modality's input covariance"),
    "X7T": dict(base="M0", family="legacy", set={"MOE_routing_signal": "cov_tangent_task"}, spaces=(),
                note="X7 + the support set's mean covariance vector (task-level routing, like X4)"),
    "X8":  dict(base="M0", family="legacy", set={"MOE_routing_signal": "cov_centroid"}, spaces=(),
                note="Riemannian experts: gate = distance to E centroids on the SPD manifold (k-means init on train users)"),
    "X8W": dict(base="M0", family="legacy",
                set={"MOE_routing_signal": "cov_centroid", "MOE_cov_expert_whiten": True}, spaces=(),
                note="X8 + each expert re-centres its input at its centroid (cluster-wise re-centring)"),
    "X9":  dict(base="M0", family="legacy", set={"feature_pool": "cov"}, spaces=("cov_pool",),
                note="M0 with second-order (covariance) pooling of the LSTM states instead of the mean"),
    "A4_cov": dict(base="A4", family="legacy", set={"feature_pool": "cov"}, spaces=("cov_pool",),
                   note="X9's control: covariance pooling without MoE"),
    "R1":  dict(base="M0", family="legacy", set={"input_align": "ea"}, spaces=(),
                note="M0 on inputs whitened by the episode support set's mean covariance (Euclidean alignment)"),
    "R1_A4": dict(base="A4", family="legacy", set={"input_align": "ea"}, spaces=(),
                  note="R1's control: Euclidean alignment without MoE"),
    "P1_cov": dict(base="A1", family="metric",
                   set={"metric_kind": "proto", "metric_backbone": "moe", "feature_pool": "cov"},
                   spaces=("metric", "cov_pool"), note="CovaMNet-style: ProtoNet on M0's expert bank with covariance pooling"),
    "P1_spd": dict(base="A4", family="metric", set={"metric_kind": "proto", "metric_backbone": "spdnet"},
                   spaces=("metric", "spd"), note="ProtoNet on an SPDNet (BiMap/ReEig/LogEig) per modality"),
    # ── Alternatives to MAML (episodic metric learners, no inner loop) ──────────
    "P1":       dict(base="A4", family="metric", set={"metric_kind": "proto", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="ProtoNet trained episodically (A4 encoder)"),
    "P1_moe":   dict(base="A1", family="metric", set={"metric_kind": "proto", "metric_backbone": "moe"},
                     spaces=("metric",), note="ProtoNet trained episodically on M0's expert-bank encoder"),
    "P1_match": dict(base="A4", family="metric", set={"metric_kind": "matching", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="Matching Network (no FCE)"),
    "P1_rel":   dict(base="A4", family="metric", set={"metric_kind": "relation", "metric_backbone": "cnnlstm"},
                     spaces=("metric", "metric_relation"), note="Relation Network (CE on relation scores)"),
    "P1_ridge": dict(base="A4", family="metric", set={"metric_kind": "ridge", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="R2-D2 closed-form ridge head"),
    "P1_feat":  dict(base="A4", family="metric", set={"metric_kind": "feat", "metric_backbone": "cnnlstm"},
                     spaces=("metric",), note="FEAT: set-to-set adapted prototypes"),
    # ── MoEMeta-TS (Wu & Yin, NeurIPS'25, ported; arch_variants/moemeta_ts.py). Reviewer ARaC's Q1. ─
    "Q1":      dict(base="A4", family="metric",
                    set={"metric_kind": "moemeta", "metric_backbone": "cnnlstm", "moemeta_variant": "full",
                         "moemeta_loss": "margin"},
                    spaces=("metric", "moemeta"), note="MoEMeta-TS: MoE relation meta + local adaptation, margin loss"),
    "Q1_nola": dict(base="A4", family="metric",
                    set={"metric_kind": "moemeta", "metric_backbone": "cnnlstm", "moemeta_variant": "no_local",
                         "moemeta_loss": "margin"},
                    spaces=("metric", "moemeta"), note="MoEMeta-TS without local adaptation (paper's w/o L.A)"),
    "Q1_mlp":  dict(base="A4", family="metric",
                    set={"metric_kind": "moemeta", "metric_backbone": "cnnlstm", "moemeta_variant": "mlp",
                         "moemeta_loss": "margin"},
                    spaces=("metric", "moemeta"), note="MoEMeta-TS with one MLP instead of the MoE (paper's w/o MoE, MetaR-like)"),
    "Q1_uni":  dict(base="A4", family="metric",
                    set={"metric_kind": "moemeta", "metric_backbone": "cnnlstm", "moemeta_variant": "uniform",
                         "moemeta_loss": "margin"},
                    spaces=("metric", "moemeta"), note="MoEMeta-TS with a uniform gate (does its router matter?)"),
    "Q1_ce":   dict(base="A4", family="metric",
                    set={"metric_kind": "moemeta", "metric_backbone": "cnnlstm", "moemeta_variant": "full",
                         "moemeta_loss": "ce"},
                    spaces=("metric", "moemeta"), note="MoEMeta-TS trained with cross-entropy on -distance (loss sensitivity)"),
    # ── Other learning approaches ──────────────────────────────────────────
    "P3":       dict(base="A2", family="legacy", set={"adann_disc_hidden": 128}, spaces=("adann",),
                     note="ADANN-style: supervised + gradient-reversal user head, then fine-tune"),
    "A4_ANIL":  dict(base="A4", family="legacy", set={"maml_inner_param_include": ["head"]}, spaces=(),
                     note="ANIL: inner loop adapts the head only (A4 backbone)"),
    "M0_ANIL":  dict(base="M0", family="legacy", set={"maml_inner_param_include": ["head"]}, spaces=(),
                     note="ANIL on M0 (MoEMeta Port B is close to this)"),
}


# Values of the family-specific keys that exist only for some tokens. The 7-fold evaluation writes
# them before the tuned overrides are applied (overrides may only name existing keys); in legacy /
# smoke mode these are the values used.
FAMILY_EVAL_DEFAULTS: dict = {
    "metric": dict(metric_relation_hidden=128, label_smooth=0.0, meta_batchsize=24),
    "adann": dict(adann_lambda=0.1, adann_disc_hidden=128),
    # the paper's values where it gives them (M=32, N=5, hidden 64, margin 1, one inner step); alpha is unstated
    "moemeta": dict(moemeta_num_experts=32, moemeta_top_n=5, moemeta_expert_hidden=64, moemeta_margin=1.0,
                    moemeta_inner_steps=1, moemeta_inner_lr=0.1, moemeta_first_order=False),
}


def variant_kind(token: str) -> str:
    v = ARCH_VARIANTS[token]
    if v["family"] == "metric":
        return "metric"
    if v["base"] in ("A1", "A2"):
        return "supervised"
    return "maml"


def uses_a4_widths(token: str) -> bool:
    """A4-based tokens use the param-matched no-MoE widths (T0 has no CNN, so widths are moot)."""
    return ARCH_VARIANTS[token]["base"] == "A4"


def uses_moe(token: str) -> bool:
    v = ARCH_VARIANTS[token]
    return v["base"] in ("M0", "A1")


def prepare_model_for_data(model, config: dict, log=print):
    """Data-dependent initialisation after building, from the config's TRAIN users only: the X8 / X8W
    covariance centroids (MOE_encoder.init_cov_centroids). A no-op for every other model."""
    if config.get("MOE_routing_signal") != "cov_centroid" or not config.get("use_MOE", False):
        return model
    import pickle
    from ablation_config import tensor_dict_file
    from MAML.maml_data_pipeline import reorient_tensor_dict
    from MOE.MOE_encoder import init_cov_centroids
    with open(tensor_dict_file(config), "rb") as f:
        td = reorient_tensor_dict(pickle.load(f), config)
    init_cov_centroids(model, td, config["train_PIDs"], seed=int(config.get("seed", 0)), log=log)
    return model


def build_model_for_config(config: dict):
    """The model a (HPO or eval) config describes, dispatched on arch_family and use_MOE. Not moved to a device."""
    family = config["arch_family"]
    if family == "flexnet":
        from arch_variants.flexnet import build_flexnet
        return build_flexnet(config)
    if family == "metric":
        from arch_variants.metric_learners import build_metric_learner
        return build_metric_learner(config)
    if family != "legacy":
        raise ValueError(f"unknown arch_family {family!r}")
    if config["use_MOE"]:
        from MOE.MOE_encoder import build_MOE_model
        return build_MOE_model(config)
    from pretraining.pretrain_models import build_model
    return build_model(config)
