"""
MOE_encoder.py
==============
Mixture-of-Experts wrappers for the CNN-LSTM EMG gesture recognition models.

Supports two MOE placements, controlled by config["MOE_placement"]:

  "encoder"
    Each expert IS a full CNN encoder.  A routing module reads the raw input
    to produce a routing vector r; the gating network uses r to decide how to
    weight the experts.  Weighted expert CNN outputs are summed and passed to
    the LSTM as usual.

      x  ──┬──► Routing module ──► r ──► Gate ──► weights w
           ├──► CNN Expert 1 ──► feat_1 ──┐
           ├──► CNN Expert 2 ──► feat_2 ──┤
           │         ···                  ├──► Σ wᵢ·featᵢ ──► LSTM ──► head
           └──► CNN Expert E ──► feat_E ──┘

  "middle"
    One shared CNN encoder (pretrained weights transfer here directly).
    The CNN feature map h is passed to E lightweight MLP experts and to a
    routing module.  The gating network weights the MLP outputs and the
    result replaces h before the LSTM.

      x  ──► Shared CNN ──┬──► Routing module ──► r ──► Gate ──► weights w
                          ├──► MLP Expert 1 ──► h_1 ──┐
                          ├──► MLP Expert 2 ──► h_2 ──┤
                          │         ···                ├──► Σ wᵢ·hᵢ ──► LSTM ──► head
                          └──► MLP Expert E ──► h_E ──┘

Design notes
────────────
* Routing signal is controlled by config["MOE_routing_signal"]:
    "linear_proj_input"      : GAP the raw input (or CNN features for middle),
                               then a single Linear → routing vector.  Cheapest.
                               Most similar to the standard Shazeer/Switch design.
    "context_proj"           : Lightweight CNN → GAP → two-layer MLP (original
                               default).  Produces a richer, more stable routing
                               signal at the cost of a few extra parameters.
    "context_proj_with_demo" : Same as "context_proj" but appends a learned
                               demographics embedding to r before the gate.
                               Only meaningful when demographics are provided.
                               Demographics inform routing but NOT expert compute.

* Soft routing (weighted sum) is used throughout — hard / top-k routing
  breaks higher-order gradients needed by MAML.  If you want sparsity add
  an auxiliary load-balancing loss (see aux loss functions below).

* Two aux losses are available and should be combined for top-k routing:
    topk_MOE_aux_loss   : Switch-style f_i * P_i load balance loss.
                          Penalises uneven dispatch counts across experts.
    importance_loss     : Shazeer-style CV² on per-expert soft weight sums.
                          Catches ordinal ranking dominance that f_i*P_i misses
                          (exactly the failure mode of a 1300x dispatch imbalance
                          with only 3x soft weight imbalance).

* Optional shared/universal expert (config["MOE_use_shared_expert"]=True):
    One always-on expert whose output is added to the weighted expert mixture.
    Handles "common" computation, freeing specialised experts to truly specialise.
    Ref: DeepSeek-V2 (2024) and related work.

* Both placements expose the same forward signature as MetaCNNLSTM /
  DeepCNNLSTM so they drop in as MAML init models with no changes to
  mamlpp.py.

* `return_routing` kwarg on forward() returns a routing_info dict:
    "gate_weights"      : (B, E) — weights_hard, actually applied to experts
    "gate_weights_soft" : (B, E) — pre-mask softmax probs (== gate_weights
                          for dense routing; differs only for top-k)

Config keys consumed (all optional with sensible defaults)
──────────────────────────────────────────────────────────
  MOE_placement          : "encoder" | "middle"   (required if use_MOE=True)
  MOE_routing_signal     : "linear_proj_input" | "context_proj" |
                           "context_proj_with_demo"   default: "context_proj"
  num_experts            : int, default 4
  MOE_expert_expand      : float, default 1.0  (width multiplier per expert CNN)
  MOE_ctx_hidden_dim     : int, default 64     (context projector hidden size)
  MOE_ctx_out_dim        : int, default 32     (routing vector dim r)
  MOE_dropout            : float, default 0.1
  MOE_gate_temperature   : float, default 1.0  (softmax temperature; >1 = flatter)
  MOE_mlp_hidden_mult    : float, default 1.0  (middle MOE: MLP hidden = CNN_out * mult)
  MOE_top_k              : int | None          (None = soft/dense routing; set for sparse)
  MOE_use_shared_expert  : bool, default False (add always-on universal expert)
  MOE_aux_coeff          : float, default 1e-2 (Switch-style load balance loss weight)
  MOE_importance_coeff   : float, default 1e-2 (importance loss weight) --> REMOVED FOR NOW!!!
  demo_embed_dim         : int, default 8      (demographics embedding output dim;
                                                used only when routing_signal ==
                                                "context_proj_with_demo")
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_lstm(lstm_module: nn.LSTM) -> None:
    """Orthogonal hidden-to-hidden, Xavier input-to-hidden, forget-gate=1."""
    for name, p in lstm_module.named_parameters():
        if "weight_hh" in name:
            nn.init.orthogonal_(p)
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)
            n = p.size(0)
            p.data[n // 4 : n // 2].fill_(1.0)


def _build_cnn_block(in_ch: int, base_filters: int, n_layers: int,
                     kernel: int, gn_groups: int, dropout: float,
                     width_mult: float = 1.0) -> Tuple[nn.Sequential, int]:
    """
    Build a standard CNN encoder block (same structure as DeepCNNLSTM).
    Returns (sequential, out_channels).
    """
    layers: List[nn.Module] = []
    curr_in = in_ch
    curr_out = max(gn_groups, int(base_filters * width_mult))
    # round curr_out to nearest multiple of gn_groups
    curr_out = max(gn_groups, (curr_out // gn_groups) * gn_groups)
    for _ in range(n_layers):
        layers += [
            nn.Conv1d(curr_in, curr_out, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.GroupNorm(gn_groups, curr_out),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        curr_in = curr_out
        curr_out = curr_out * 2
    return nn.Sequential(*layers), curr_in  # curr_in == last curr_out


# ─────────────────────────────────────────────────────────────────────────────
# Context Projector
# ─────────────────────────────────────────────────────────────────────────────

class ContextProjector(nn.Module):
    """
    Lightweight module that reads the raw input (or CNN features) and
    produces a compact routing vector r used by the gating network.

    For "encoder" placement: receives raw x (B, C, T) → CNN → GAP → MLP → r
    For "middle"   placement: receives CNN features h (B, C', T') → GAP → MLP → r

    Using a learned projector (rather than routing on raw x) is recommended
    because raw multi-channel time-series have very different scales and
    statistics across modalities; a tiny CNN extracts stable statistics before
    routing.
    """

    def __init__(self, in_ch: int, hidden_dim: int, out_dim: int,
                 use_conv: bool = True, conv_kernel: int = 5,
                 gn_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            # One conv layer to mix channels, then GAP
            conv_out = max(gn_groups, hidden_dim)
            conv_out = (conv_out // gn_groups) * gn_groups
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, conv_out, kernel_size=conv_kernel,
                          padding=conv_kernel // 2, bias=False),
                nn.GroupNorm(gn_groups, conv_out),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            mlp_in = conv_out
        else:
            # Features already extracted; just pool
            self.conv = None
            mlp_in = in_ch

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → r: (B, out_dim)"""
        if self.use_conv and self.conv is not None:
            x = self.conv(x)           # (B, conv_out, T)
        r = x.mean(dim=-1)             # GAP → (B, C')
        return self.mlp(r)             # (B, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Routing Module factory
# ─────────────────────────────────────────────────────────────────────────────

class LinearProjRouter(nn.Module):
    """
    Simplest possible router: GAP the input, then a single linear projection.
    Matches the standard Shazeer (2017) / Switch (2022) design.
    No extra parameters beyond the gate's own linear layer — the gate_in_dim
    IS the ctx_out_dim in this case, and this module produces it.

    in_ch   : channels of the input tensor (B, C, T)
    out_dim : routing vector dimension (fed into MOEGate)
    """
    def __init__(self, in_ch: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_ch, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                demo_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : (B, C, T) — raw input or CNN feature map
        demo_embed : ignored for this router type (no demo support)
        → r        : (B, out_dim)
        """
        r = x.mean(dim=-1)    # GAP → (B, C)
        return self.proj(r)   # (B, out_dim)


class ContextProjRouter(nn.Module):
    """
    Richer router: lightweight CNN → GAP → two-layer MLP.
    Original default design. Produces more stable routing signals for
    multi-channel time-series by mixing channels before pooling.

    Optionally appends a demographics embedding to r before returning,
    enabled by passing demo_embed (B, demo_embed_dim) at forward time.
    The demographics embedding is concatenated onto r, so gate_in_dim must
    be set to ctx_out_dim + demo_embed_dim when this is used.
    """
    def __init__(self, in_ch: int, hidden_dim: int, out_dim: int,
                 use_conv: bool = True, conv_kernel: int = 5,
                 gn_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.projector = ContextProjector(
            in_ch=in_ch, hidden_dim=hidden_dim, out_dim=out_dim,
            use_conv=use_conv, conv_kernel=conv_kernel,
            gn_groups=gn_groups, dropout=dropout,
        )

    def forward(self, x: torch.Tensor,
                demo_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : (B, C, T)
        demo_embed : (B, demo_embed_dim) or None — appended to r if provided
        → r        : (B, out_dim)  or  (B, out_dim + demo_embed_dim)
        """
        r = self.projector(x)
        if demo_embed is not None:
            r = torch.cat([r, demo_embed], dim=-1)
        return r


def build_routing_module(
    routing_signal: str,
    in_ch: int,
    ctx_hidden_dim: int,
    ctx_out_dim: int,
    use_conv_in_proj: bool,
    conv_kernel: int,
    gn_groups: int,
    dropout: float,
    demo_embed_dim: int,
) -> Tuple[nn.Module, int]:
    """
    Factory for routing modules.  Returns (module, gate_in_dim).

    gate_in_dim is what MOEGate.linear expects as input size — it differs
    from ctx_out_dim when demographics are appended.

    Args:
        routing_signal   : one of "linear_proj_input", "context_proj",
                           "context_proj_with_demo"
        in_ch            : input channels to the routing module
        ctx_hidden_dim   : hidden dim for context projector MLP
        ctx_out_dim      : output dim of routing vector (before demo concat)
        use_conv_in_proj : whether the context projector uses a CNN front-end
                           (True for encoder placement, False for middle)
        conv_kernel      : kernel size for the projector's CNN
        gn_groups        : GroupNorm groups
        dropout          : dropout rate
        demo_embed_dim   : size of demographics embedding; only used when
                           routing_signal == "context_proj_with_demo"

    Returns:
        router       : nn.Module with forward(x, demo_embed=None) -> r
        gate_in_dim  : int — the dimension MOEGate should expect as input
    """
    if routing_signal == "linear_proj_input":
        router = LinearProjRouter(
            in_ch=in_ch, out_dim=ctx_out_dim, dropout=dropout,
        )
        gate_in_dim = ctx_out_dim

    elif routing_signal == "context_proj":
        router = ContextProjRouter(
            in_ch=in_ch, hidden_dim=ctx_hidden_dim, out_dim=ctx_out_dim,
            use_conv=use_conv_in_proj, conv_kernel=conv_kernel,
            gn_groups=gn_groups, dropout=dropout,
        )
        gate_in_dim = ctx_out_dim

    elif routing_signal == "context_proj_with_demo":
        router = ContextProjRouter(
            in_ch=in_ch, hidden_dim=ctx_hidden_dim, out_dim=ctx_out_dim,
            use_conv=use_conv_in_proj, conv_kernel=conv_kernel,
            gn_groups=gn_groups, dropout=dropout,
        )
        # Demographics are concatenated ONTO r inside router.forward(),
        # so the gate sees ctx_out_dim + demo_embed_dim.
        gate_in_dim = ctx_out_dim + demo_embed_dim

    else:
        raise ValueError(
            f"Unknown MOE_routing_signal '{routing_signal}'. "
            "Must be one of: 'linear_proj_input', 'context_proj', "
            "'context_proj_with_demo'."
        )

    return router, gate_in_dim


# ─────────────────────────────────────────────────────────────────────────────
# Demographics Embedder (used only for context_proj_with_demo)
# ─────────────────────────────────────────────────────────────────────────────

class DemographicsEmbedder(nn.Module):
    """
    Tiny MLP that maps a raw demographics vector (B, demo_in_dim) to a
    compact embedding (B, demo_embed_dim) for concatenation onto the
    routing vector r.

    Demographics inform ROUTING only — they are never passed to expert CNNs.
    The intuition: demographics (age, sex, limb dominance, etc.) are
    sample-level invariants that may predict which expert is most appropriate
    (e.g., dominant vs non-dominant arm gesture patterns), but experts
    themselves should specialise in signal morphology, not demographics.
    """
    def __init__(self, demo_in_dim: int, demo_embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(demo_in_dim, demo_embed_dim * 2),
            nn.LayerNorm(demo_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(demo_embed_dim * 2, demo_embed_dim),
        )

    def forward(self, demo: torch.Tensor) -> torch.Tensor:
        """demo: (B, demo_in_dim) → (B, demo_embed_dim)"""
        return self.net(demo)


# ─────────────────────────────────────────────────────────────────────────────
# Gating Network
# ─────────────────────────────────────────────────────────────────────────────

class MOEGate(nn.Module):
    """
    Produces per-sample soft weights over E experts from routing vector r.

    top_k: if set, zeroes out all but the top-k weights and renormalises.
           Keeps gradients flowing through the top-k weights (straight-through
           for the mask).  Not recommended with MAML unless k/E ≥ 0.5.
    temperature: softmax temperature.  Values > 1 flatten the distribution
                 (all experts get similar weight); < 1 sharpens it.
                 Learnable temperature is possible but not the default.
    """

    def __init__(self, in_dim: int, num_experts: int,
                 top_k: Optional[int] = None,
                 temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.linear = nn.Linear(in_dim, num_experts)

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        r: (B, in_dim) → (weights_hard, weights_soft)

        weights_soft : (B, E) — raw softmax probabilities, always differentiable.
                       Used as P_i in the top-k aux loss and for dense routing.
        weights_hard : (B, E) — post-top-k masked & renormalised weights that are
                       actually applied to the experts.  Equal to weights_soft when
                       top_k is None (dense routing).

        Callers should apply weights_hard to the expert outputs and pass both
        tensors to the routing_info dict so the correct aux loss can be computed.
        """
        logits       = self.linear(r) / self.temperature   # (B, E)
        weights_soft = F.softmax(logits, dim=-1)            # (B, E)

        if self.top_k is not None and self.top_k < self.num_experts:
            _, topk_idx  = torch.topk(weights_soft, self.top_k, dim=-1)
            mask         = torch.zeros_like(weights_soft).scatter_(-1, topk_idx, 1.0)
            weights_hard = weights_soft * mask
            weights_hard = weights_hard / (weights_hard.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            weights_hard = weights_soft   # dense: identical

        return weights_hard, weights_soft


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary load-balancing losses
# ─────────────────────────────────────────────────────────────────────────────

def dense_MOE_aux_loss(gate_weights: torch.Tensor,
                       coeff: float = 1e-2,
                       eps: float = 1e-8) -> torch.Tensor:
    """
    Load-balancing loss for dense / soft routing.

    Minimises KL(avg_gate_weights || uniform), pushing the mean routing
    distribution toward equal utilisation of all experts.

    gate_weights : (B, E) — weights_hard (== weights_soft for dense routing)
    coeff        : loss scale factor; try 1e-2 → 1e-1 if collapse persists.

    Note: reduction='sum' is correct here because avg_usage is already a
    length-E vector (batch dimension already averaged out).
    """
    avg_usage = gate_weights.mean(dim=0)               # (E,)
    E         = avg_usage.numel()
    target    = torch.full_like(avg_usage, 1.0 / E)
    return coeff * F.kl_div((avg_usage + eps).log(), target, reduction='sum')


def topk_MOE_aux_loss(gate_weights_soft: torch.Tensor,
                      gate_weights_hard: torch.Tensor,
                      coeff: float = 1e-2) -> torch.Tensor:
    """
    Switch Transformer load-balancing loss for top-k routing.

    L = coeff * E * Σ_i  f_i * P_i
      P_i : mean soft probability before masking  — differentiable
      f_i : mean dispatch fraction after masking  — frozen (no gradient)

    gate_weights_soft : (B, E) — raw softmax probs, BEFORE top-k zeroing
    gate_weights_hard : (B, E) — weights AFTER top-k mask + renorm

    The .detach() on f is critical: f is a frozen load signal, not something
    we differentiate through.

    IMPORTANT: This loss alone is insufficient for preventing routing collapse.
    It penalises uneven dispatch counts but can miss ordinal ranking dominance —
    one expert can consistently rank first in top-k selection while maintaining
    near-uniform soft weights (3x soft imbalance, 1300x dispatch imbalance).
    Always pair with importance_loss() for top-k routing.
    """
    P = gate_weights_soft.mean(dim=0)                  # (E,) differentiable
    f = gate_weights_hard.detach().mean(dim=0)          # (E,) frozen
    E = P.numel()
    return coeff * E * (f * P).sum()


def importance_loss(gate_weights_soft: torch.Tensor,
                    coeff: float = 1e-2) -> torch.Tensor:
    """
    Shazeer et al. (2017) importance loss.

    Penalises variance in the per-expert SUM of soft gate weights across
    the batch, measured as the squared coefficient of variation (CV²).

    This catches ordinal ranking dominance that topk_MOE_aux_loss misses:
    if one expert consistently ranks near the top of the soft distribution,
    its importance (sum of soft weights) will be high even if the mean soft
    weight looks approximately uniform.  CV² penalises this directly.

    Concretely: in your observed failure mode (1300x dispatch imbalance but
    only 3x soft weight imbalance), the Switch loss sees P_i ≈ uniform and
    applies near-zero gradient to the router.  The importance loss sees
    importance_i = Σ_b w_soft[b,i] which IS highly non-uniform when one
    expert dominates the ordinal ranking, and penalises it.

    gate_weights_soft : (B, E) — raw softmax probs BEFORE top-k masking
    coeff             : loss scale factor

    Should be combined with topk_MOE_aux_loss() for top-k routing:
        aux = topk_MOE_aux_loss(w_soft, w_hard, coeff=cfg["MOE_aux_coeff"])
            + importance_loss(w_soft, coeff=cfg["MOE_importance_coeff"])
    """
    if coeff == 0.0:
        return 0.0
    importance = gate_weights_soft.sum(dim=0)           # (E,) total soft weight per expert
    mean_imp   = importance.mean()
    var_imp    = importance.var()
    cv_sq      = var_imp / (mean_imp ** 2 + 1e-8)       # squared coefficient of variation
    return coeff * cv_sq


# ─────────────────────────────────────────────────────────────────────────────
# Shared / Universal Expert
# ─────────────────────────────────────────────────────────────────────────────

class SharedExpert(nn.Module):
    """
    Always-on universal expert whose output is ADDED to the weighted mixture
    of specialised experts.  Ref: DeepSeek-V2 (2024) and related work.

    The intuition: one expert handles "common" computation that all inputs
    need (e.g., basic gesture dynamics), freeing the specialised experts to
    diverge more aggressively.  Because the shared expert is always active,
    it receives dense gradient signal and converges quickly; this also
    stabilises early training before the router has learned to specialise.

    For encoder placement: expert is a CNN (same structure as expert_cnns).
    For middle placement:  expert is an MLP (same structure as expert_mlps).

    The caller adds shared_out to the expert mixture:
        h = shared_expert(x) + expert_mixture     (encoder: feature maps)
        h = shared_expert(h_t) + expert_mixture   (middle: per-timestep feats)
    """

    def __init__(self, expert: nn.Module):
        """
        expert : a pre-built nn.Module (CNN Sequential or MLP Sequential)
                 matching the structure of the specialised experts.
                 Build it externally and pass it in so SharedExpert does not
                 need to know about architecture hyperparameters.
        """
        super().__init__()
        self.expert = expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder-level MOE: MetaCNNLSTM variant
# ─────────────────────────────────────────────────────────────────────────────

class MetaCNNLSTM_EncoderMOE(nn.Module):
    """
    MetaCNNLSTM with MOE at the encoder level.

    Architecture:
      - E expert CNN encoders (each identical to MetaCNNLSTM's single conv block)
      - Routing module reads raw x and outputs routing vector r
      - Gate produces soft weights over experts from r
      - Weighted sum of expert outputs (+ optional shared expert) → LSTM × 3 → head

    Pretrain note: The LSTMs and head can be initialised from a vanilla
    MetaCNNLSTM checkpoint by matching keys; the new expert CNNs and gate
    start from scratch (or can be seeded from the pretrained conv weights —
    see load_pretrained_backbone() below).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # ── Sizes ───────────────────────────────────────────────────────────
        C_emg    = config["emg_in_ch"]
        C_imu    = config.get("imu_in_ch", 72)
        use_imu  = config["use_imu"]
        n_filt   = config["cnn_filters"]
        k        = config["cnn_kernel"]
        gn_grps  = config["groupnorm_num_groups"]
        lstm_h   = config["lstm_hidden"]
        n_way    = config['n_way'] if config['meta_learning'] else config['pretrain_num_classes']
        drop     = config.get("dropout", 0.0)
        bidir    = config.get("bidirectional", False)
        head_type = config["head_type"]

        # MOE config
        E               = config["num_experts"]
        ctx_hidden      = config["MOE_ctx_hidden_dim"]
        ctx_out         = config["MOE_ctx_out_dim"]
        MOE_drop        = config["MOE_dropout"]
        gate_temp       = config["MOE_gate_temperature"]
        top_k           = config["MOE_top_k"]
        width_mult      = config["MOE_expert_expand"]
        routing_signal  = config.get("MOE_routing_signal", "context_proj")
        use_shared      = config.get("MOE_use_shared_expert", False)
        demo_embed_dim  = config.get("demo_embed_dim", 8)

        self.use_imu        = use_imu
        self.num_experts    = E
        self.routing_signal = routing_signal
        self.use_shared     = use_shared
        D = 2 if bidir else 1

        in_ch = C_emg + (C_imu if use_imu else 0)

        # ── Demographics embedder (only built when needed) ───────────────────
        if routing_signal == "context_proj_with_demo":
            demo_in_dim = config["demo_in_dim"]  # must be set in config
            self.demo_embedder = DemographicsEmbedder(
                demo_in_dim=demo_in_dim,
                demo_embed_dim=demo_embed_dim,
                dropout=MOE_drop,
            )
        else:
            self.demo_embedder = None

        # ── Routing module ───────────────────────────────────────────────────
        self.router, gate_in_dim = build_routing_module(
            routing_signal   = routing_signal,
            in_ch            = in_ch,
            ctx_hidden_dim   = ctx_hidden,
            ctx_out_dim      = ctx_out,
            use_conv_in_proj = True,   # encoder placement: raw x needs CNN mixing
            conv_kernel      = k,
            gn_groups        = gn_grps,
            dropout          = MOE_drop,
            demo_embed_dim   = demo_embed_dim,
        )

        # ── Expert CNN encoders ──────────────────────────────────────────────
        expert_cnns = []
        for _ in range(E):
            conv = nn.Sequential(
                nn.Conv1d(in_ch, int(n_filt * width_mult), kernel_size=k,
                          padding=k // 2, bias=False),
                nn.GroupNorm(gn_grps, int(n_filt * width_mult)),
                nn.ReLU(),
            )
            expert_cnns.append(conv)
        self.expert_cnns = nn.ModuleList(expert_cnns)
        cnn_out_ch = int(n_filt * width_mult)

        # ── Optional shared expert (same structure as a specialised expert) ──
        if use_shared:
            shared_cnn = nn.Sequential(
                nn.Conv1d(in_ch, cnn_out_ch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.GroupNorm(gn_grps, cnn_out_ch),
                nn.ReLU(),
            )
            self.shared_expert = SharedExpert(shared_cnn)
        else:
            self.shared_expert = None

        # ── Gating network ───────────────────────────────────────────────────
        self.gate = MOEGate(gate_in_dim, E, top_k=top_k, temperature=gate_temp)

        # ── LSTM layers ──────────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(cnn_out_ch, lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm3 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            _init_lstm(lstm)
        self.lstm_dropout = nn.Dropout(drop)

        self.feat_dim = lstm_h * D

        # ── Head ────────────────────────────────────────────────────────────
        if head_type == "linear":
            self.head = nn.Linear(self.feat_dim, n_way)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.GELU(), nn.Dropout(drop),
                nn.Linear(self.feat_dim // 2, n_way),
            )

    # ─────────────────────────────────────────────────────────────────────────
    def _get_routing_vector(self, x: torch.Tensor,
                            demographics: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute routing vector r, optionally incorporating demographics."""
        if self.routing_signal == "context_proj_with_demo":
            demo_embed = self.demo_embedder(demographics) \
                if (self.demo_embedder is not None and demographics is not None) \
                else None
            return self.router(x, demo_embed=demo_embed)
        return self.router(x)

    # ─────────────────────────────────────────────────────────────────────────
    def backbone(self, x_emg: torch.Tensor, x_imu=None, demographics=None,
                 return_routing: bool = False
                 ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns (final_feat, [layer1_feat, layer2_feat, layer3_feat]).
        If return_routing=True, appends gate_weights (B, E) as 4th element.
        """
        x = x_emg
        if self.use_imu and x_imu is not None:
            x = torch.cat([x, x_imu], dim=1)   # (B, C_emg+C_imu, T)

        # 1. Routing vector
        r = self._get_routing_vector(x, demographics)       # (B, gate_in_dim)

        # 2. Gating weights — always returns (hard, soft)
        w_hard, w_soft = self.gate(r)                       # each (B, E)

        # 3. Expert CNN forward passes
        expert_feats = [exp(x) for exp in self.expert_cnns] # each (B, cnn_out_ch, T)

        # 4. Weighted combination: Σ wᵢ · feat_i  (use w_hard for mixing)
        stacked = torch.stack(expert_feats, dim=1)           # (B, E, C, T)
        w_bcw   = w_hard.unsqueeze(-1).unsqueeze(-1)         # (B, E, 1, 1)
        h       = (stacked * w_bcw).sum(dim=1)               # (B, C, T)

        # 5. Add shared expert output if enabled
        if self.shared_expert is not None:
            h = h + self.shared_expert(x)                    # (B, C, T)

        # 6. Permute for LSTM: (B, T, C)
        h = h.permute(0, 2, 1)

        # 7. LSTM stack
        h1, _ = self.lstm1(h);  h1 = self.lstm_dropout(h1)
        h2, _ = self.lstm2(h1); h2 = self.lstm_dropout(h2)
        h3, _ = self.lstm3(h2)

        layer1_feat = h1.mean(dim=1)
        layer2_feat = h2.mean(dim=1)
        layer3_feat = h3.mean(dim=1)

        if return_routing:
            return layer3_feat, [layer1_feat, layer2_feat, layer3_feat], w_hard, w_soft
        return layer3_feat, [layer1_feat, layer2_feat, layer3_feat]

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x_emg: torch.Tensor, x_imu=None, demographics=None,
                return_routing: bool = False):
        result = self.backbone(x_emg, x_imu, demographics, return_routing=return_routing)
        if return_routing:
            feat, layers, w_hard, w_soft = result
            logits = self.head(feat)
            return logits, {"gate_weights": w_hard, "gate_weights_soft": w_soft}
        else:
            feat, _ = result
            return self.head(feat)

    def get_features(self, x_emg, x_imu=None, demographics=None):
        with torch.no_grad():
            return self.backbone(x_emg, x_imu, demographics)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder-level MOE: DeepCNNLSTM variant
# ─────────────────────────────────────────────────────────────────────────────

class DeepCNNLSTM_EncoderMOE(nn.Module):
    """
    DeepCNNLSTM with MOE at the encoder level.

    Architecture:
      - E expert CNN encoders (3-layer doubling CNN per expert)
      - Routing module reads raw x → r
      - Gating network weights experts from r
      - Weighted sum (+ optional shared expert) → LSTM × 3 → head

    The expert CNNs are the expensive part here.  With E=4 experts and a
    width_mult < 1, the total parameter count can be kept close to the
    baseline DeepCNNLSTM by making each expert CNN narrower.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # ── Sizes ───────────────────────────────────────────────────────────
        C_emg    = config["emg_in_ch"]
        C_imu    = config.get("imu_in_ch", 72)
        use_imu  = config["use_imu"]
        base_f   = config["cnn_base_filters"]
        n_layers = config["cnn_layers"]
        k        = config["cnn_kernel"]
        gn_grps  = config["groupnorm_num_groups"]
        lstm_h   = config["lstm_hidden"]
        n_way    = config['n_way'] if config['meta_learning'] else config['pretrain_num_classes']
        drop     = config["dropout"]
        bidir    = config["bidirectional"]
        head_type = config["head_type"]

        # MOE config
        E              = config["num_experts"]
        ctx_hidden     = config["MOE_ctx_hidden_dim"]
        ctx_out        = config["MOE_ctx_out_dim"]
        MOE_drop       = config["MOE_dropout"]
        gate_temp      = config["MOE_gate_temperature"]
        top_k          = config["MOE_top_k"]
        width_mult     = config["MOE_expert_expand"]
        routing_signal = config.get("MOE_routing_signal", "context_proj")
        use_shared     = config.get("MOE_use_shared_expert", False)
        demo_embed_dim = config.get("demo_embed_dim", 8)

        self.use_imu        = use_imu
        self.num_experts    = E
        self.routing_signal = routing_signal
        self.use_shared     = use_shared
        D = 2 if bidir else 1

        in_ch = C_emg + (C_imu if use_imu else 0)

        # ── Optional shared strided front-end ────────────────────────────────
        front_end_stride = config.get("front_end_stride", 0)
        assert isinstance(front_end_stride, int) and front_end_stride >= 0, \
            f"front_end_stride must be a non-negative int, got {front_end_stride}"
        if front_end_stride > 0:
            front_k = 2 * front_end_stride
            front_out_ch = max(gn_grps, in_ch)
            front_out_ch = (front_out_ch // gn_grps) * gn_grps
            self.front_end = nn.Sequential(
                nn.Conv1d(in_ch, front_out_ch, kernel_size=front_k,
                          stride=front_end_stride, padding=front_k // 2, bias=False),
                nn.GroupNorm(gn_grps, front_out_ch),
                nn.GELU(),
            )
            front_end_out_ch = front_out_ch
        else:
            self.front_end = None
            front_end_out_ch = in_ch
        self.front_end_stride = front_end_stride

        # ── Demographics embedder ────────────────────────────────────────────
        if routing_signal == "context_proj_with_demo":
            demo_in_dim = config["demo_in_dim"]
            self.demo_embedder = DemographicsEmbedder(
                demo_in_dim=demo_in_dim,
                demo_embed_dim=demo_embed_dim,
                dropout=MOE_drop,
            )
        else:
            self.demo_embedder = None

        # ── Routing module ───────────────────────────────────────────────────
        self.router, gate_in_dim = build_routing_module(
            routing_signal   = routing_signal,
            in_ch            = front_end_out_ch,
            ctx_hidden_dim   = ctx_hidden,
            ctx_out_dim      = ctx_out,
            use_conv_in_proj = True,
            conv_kernel      = k,
            gn_groups        = gn_grps,
            dropout          = MOE_drop,
            demo_embed_dim   = demo_embed_dim,
        )

        # ── Expert CNN encoders ──────────────────────────────────────────────
        expert_cnns = []
        for _ in range(E):
            cnn, cnn_out_ch = _build_cnn_block(
                in_ch=front_end_out_ch, base_filters=base_f, n_layers=n_layers,
                kernel=k, gn_groups=gn_grps, dropout=MOE_drop,
                width_mult=width_mult,
            )
            expert_cnns.append(cnn)
        self.expert_cnns = nn.ModuleList(expert_cnns)
        _, cnn_out_ch = _build_cnn_block(front_end_out_ch, base_f, n_layers, k, gn_grps, drop, width_mult)

        # ── Optional shared expert ───────────────────────────────────────────
        if use_shared:
            shared_cnn, _ = _build_cnn_block(
                in_ch=front_end_out_ch, base_filters=base_f, n_layers=n_layers,
                kernel=k, gn_groups=gn_grps, dropout=MOE_drop,
                width_mult=width_mult,
            )
            self.shared_expert = SharedExpert(shared_cnn)
        else:
            self.shared_expert = None

        # ── Gating ──────────────────────────────────────────────────────────
        self.gate = MOEGate(gate_in_dim, E, top_k=top_k, temperature=gate_temp)

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(cnn_out_ch, lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm3 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            _init_lstm(lstm)
        self.lstm_dropout = nn.Dropout(drop)
        self.feat_dim = lstm_h * D

        # ── Head ────────────────────────────────────────────────────────────
        if head_type == "linear":
            self.head = nn.Linear(self.feat_dim, n_way)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.GELU(), nn.Dropout(drop),
                nn.Linear(self.feat_dim // 2, n_way),
            )

    # ─────────────────────────────────────────────────────────────────────────
    def _get_routing_vector(self, x: torch.Tensor,
                            demographics: Optional[torch.Tensor]) -> torch.Tensor:
        if self.routing_signal == "context_proj_with_demo":
            demo_embed = self.demo_embedder(demographics) \
                if (self.demo_embedder is not None and demographics is not None) \
                else None
            return self.router(x, demo_embed=demo_embed)
        return self.router(x)

    # ─────────────────────────────────────────────────────────────────────────
    def backbone(self, x_emg, x_imu=None, demographics=None,
                 return_routing: bool = False):
        x = x_emg
        if self.use_imu and x_imu is not None:
            x = torch.cat([x, x_imu], dim=1)

        if self.front_end is not None:
            x = self.front_end(x)

        # 1. Routing vector
        r = self._get_routing_vector(x, demographics)
        w_hard, w_soft = self.gate(r)                              # each (B, E)

        # 2. Expert CNNs
        expert_feats = [exp(x) for exp in self.expert_cnns]       # each (B, C, T')
        stacked = torch.stack(expert_feats, dim=1)                 # (B, E, C, T')
        w_4d    = w_hard.unsqueeze(-1).unsqueeze(-1)               # (B, E, 1, 1)
        h       = (stacked * w_4d).sum(dim=1)                      # (B, C, T')

        # 3. Add shared expert if enabled
        if self.shared_expert is not None:
            h = h + self.shared_expert(x)

        # 4. LSTM
        h = h.permute(0, 2, 1)
        h1, _ = self.lstm1(h);  h1 = self.lstm_dropout(h1)
        h2, _ = self.lstm2(h1); h2 = self.lstm_dropout(h2)
        h3, _ = self.lstm3(h2)

        l1 = h1.mean(dim=1)
        l2 = h2.mean(dim=1)
        l3 = h3.mean(dim=1)

        if return_routing:
            return l3, [l1, l2, l3], w_hard, w_soft
        return l3, [l1, l2, l3]

    def forward(self, x_emg, x_imu=None, demographics=None,
                return_routing: bool = False):
        result = self.backbone(x_emg, x_imu, demographics, return_routing=return_routing)
        if return_routing:
            feat, _, w_hard, w_soft = result
            return self.head(feat), {"gate_weights": w_hard, "gate_weights_soft": w_soft}
        return self.head(result[0])

    def get_features(self, x_emg, x_imu=None, demographics=None):
        with torch.no_grad():
            return self.backbone(x_emg, x_imu, demographics)


# ─────────────────────────────────────────────────────────────────────────────
# Middle-level MOE: MetaCNNLSTM variant
# ─────────────────────────────────────────────────────────────────────────────

class MetaCNNLSTM_MiddleMOE(nn.Module):
    """
    MetaCNNLSTM with MOE at the CNN→LSTM bottleneck.

    Architecture:
      - Shared CNN (identical to MetaCNNLSTM's conv — easy pretrain transfer)
      - Routing module reads CNN features (not raw x) → routing vector r
      - E MLP experts transform CNN features h → adapted h_e
      - Weighted sum (+ optional shared expert) → LSTM × 3 → head

    This is the recommended starting point if you have a strong pretrained CNN.
    Only the MLP experts and gate are new; you can freeze the CNN and only
    adapt LSTMs + experts in early fine-tuning.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        C_emg    = config["emg_in_ch"]
        C_imu    = config.get("imu_in_ch", 72)
        use_imu  = config.get("use_imu", False)
        n_filt   = config["cnn_filters"]
        k        = config["cnn_kernel"]
        gn_grps  = config.get("groupnorm_num_groups", 8)
        lstm_h   = config["lstm_hidden"]
        n_way    = config['n_way'] if config['meta_learning'] else config['pretrain_num_classes']
        drop     = config.get("dropout", 0.0)
        bidir    = config.get("bidirectional", False)
        head_type = config.get("head_type", "linear")

        # MOE config
        E              = config.get("num_experts", 4)
        ctx_hidden     = config.get("MOE_ctx_hidden_dim", 64)
        ctx_out        = config.get("MOE_ctx_out_dim", 32)
        MOE_drop       = config.get("MOE_dropout", drop)
        gate_temp      = config.get("MOE_gate_temperature", 1.0)
        top_k          = config.get("MOE_top_k", None)
        mlp_mult       = config.get("MOE_mlp_hidden_mult", 1.0)
        routing_signal = config.get("MOE_routing_signal", "context_proj")
        use_shared     = config.get("MOE_use_shared_expert", False)
        demo_embed_dim = config.get("demo_embed_dim", 8)

        self.use_imu        = use_imu
        self.num_experts    = E
        self.routing_signal = routing_signal
        self.use_shared     = use_shared
        D = 2 if bidir else 1

        in_ch = C_emg + (C_imu if use_imu else 0)

        # ── Shared CNN ───────────────────────────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, n_filt, kernel_size=k, padding=k // 2, bias=False),
            nn.GroupNorm(gn_grps, n_filt),
            nn.ReLU(),
        )

        # ── Demographics embedder ────────────────────────────────────────────
        if routing_signal == "context_proj_with_demo":
            demo_in_dim = config["demo_in_dim"]
            self.demo_embedder = DemographicsEmbedder(
                demo_in_dim=demo_in_dim,
                demo_embed_dim=demo_embed_dim,
                dropout=MOE_drop,
            )
        else:
            self.demo_embedder = None

        # ── Routing module ───────────────────────────────────────────────────
        # Middle placement: router reads CNN features (already extracted),
        # so use_conv_in_proj=False (no additional CNN layer needed).
        self.router, gate_in_dim = build_routing_module(
            routing_signal   = routing_signal,
            in_ch            = n_filt,
            ctx_hidden_dim   = ctx_hidden,
            ctx_out_dim      = ctx_out,
            use_conv_in_proj = False,
            conv_kernel      = k,
            gn_groups        = gn_grps,
            dropout          = MOE_drop,
            demo_embed_dim   = demo_embed_dim,
        )

        # ── MLP Experts ──────────────────────────────────────────────────────
        hidden_ch = max(n_filt, int(n_filt * mlp_mult))
        expert_mlps = []
        for _ in range(E):
            expert_mlps.append(nn.Sequential(
                nn.Linear(n_filt, hidden_ch),
                nn.GELU(),
                nn.Dropout(MOE_drop),
                nn.Linear(hidden_ch, n_filt),
            ))
        self.expert_mlps = nn.ModuleList(expert_mlps)

        # ── Optional shared expert (MLP matching expert structure) ───────────
        if use_shared:
            shared_mlp = nn.Sequential(
                nn.Linear(n_filt, hidden_ch),
                nn.GELU(),
                nn.Dropout(MOE_drop),
                nn.Linear(hidden_ch, n_filt),
            )
            self.shared_expert = SharedExpert(shared_mlp)
        else:
            self.shared_expert = None

        # ── Gate ────────────────────────────────────────────────────────────
        self.gate = MOEGate(gate_in_dim, E, top_k=top_k, temperature=gate_temp)

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(n_filt,  lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.LSTM(lstm_h * D, lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm3 = nn.LSTM(lstm_h * D, lstm_h, batch_first=True, bidirectional=bidir)
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            _init_lstm(lstm)
        self.lstm_dropout = nn.Dropout(drop)
        self.feat_dim = lstm_h * D

        # ── Head ────────────────────────────────────────────────────────────
        if head_type == "linear":
            self.head = nn.Linear(self.feat_dim, n_way)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.GELU(), nn.Dropout(drop),
                nn.Linear(self.feat_dim // 2, n_way),
            )

    # ─────────────────────────────────────────────────────────────────────────
    def _get_routing_vector(self, x: torch.Tensor,
                            demographics: Optional[torch.Tensor]) -> torch.Tensor:
        if self.routing_signal == "context_proj_with_demo":
            demo_embed = self.demo_embedder(demographics) \
                if (self.demo_embedder is not None and demographics is not None) \
                else None
            return self.router(x, demo_embed=demo_embed)
        return self.router(x)

    # ─────────────────────────────────────────────────────────────────────────
    def backbone(self, x_emg, x_imu=None, demographics=None,
                 return_routing: bool = False):
        x = x_emg
        if self.use_imu and x_imu is not None:
            x = torch.cat([x, x_imu], dim=1)

        # Shared CNN: (B, C, T) → (B, n_filt, T)
        h = self.conv(x)

        # Routing
        r = self._get_routing_vector(h, demographics)  # (B, gate_in_dim)
        w_hard, w_soft = self.gate(r)                  # each (B, E)

        # MLP experts: operate on (B, T, n_filt)
        h_t = h.permute(0, 2, 1)
        expert_outs = [exp(h_t) for exp in self.expert_mlps]  # each (B, T, n_filt)
        stacked = torch.stack(expert_outs, dim=1)              # (B, E, T, n_filt)
        w_4d    = w_hard.unsqueeze(-1).unsqueeze(-1)           # (B, E, 1, 1)
        h_mix   = (stacked * w_4d).sum(dim=1)                  # (B, T, n_filt)

        # Add shared expert if enabled
        if self.shared_expert is not None:
            h_mix = h_mix + self.shared_expert(h_t)

        # LSTM stack (already in (B, T, C) format)
        h1, _ = self.lstm1(h_mix); h1 = self.lstm_dropout(h1)
        h2, _ = self.lstm2(h1);    h2 = self.lstm_dropout(h2)
        h3, _ = self.lstm3(h2)

        l1 = h1.mean(dim=1)
        l2 = h2.mean(dim=1)
        l3 = h3.mean(dim=1)

        if return_routing:
            return l3, [l1, l2, l3], w_hard, w_soft
        return l3, [l1, l2, l3]

    def forward(self, x_emg, x_imu=None, demographics=None,
                return_routing: bool = False):
        result = self.backbone(x_emg, x_imu, demographics, return_routing=return_routing)
        if return_routing:
            feat, _, w_hard, w_soft = result
            return self.head(feat), {"gate_weights": w_hard, "gate_weights_soft": w_soft}
        return self.head(result[0])

    def get_features(self, x_emg, x_imu=None, demographics=None):
        with torch.no_grad():
            return self.backbone(x_emg, x_imu, demographics)


# ─────────────────────────────────────────────────────────────────────────────
# Middle-level MOE: DeepCNNLSTM variant
# ─────────────────────────────────────────────────────────────────────────────

class DeepCNNLSTM_MiddleMOE(nn.Module):
    """
    DeepCNNLSTM with MOE at the CNN→LSTM bottleneck.

    This is the most transfer-friendly variant: the multi-layer CNN is shared
    and identical to baseline DeepCNNLSTM, so pretrained weights load directly.
    Only the E MLP experts + gate are new parameters.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        C_emg    = config["emg_in_ch"]
        C_imu    = config.get("imu_in_ch", 72)
        use_imu  = config.get("use_imu", False)
        base_f   = config["cnn_base_filters"]
        n_layers = config.get("cnn_layers", 3)
        k        = config["cnn_kernel"]
        gn_grps  = config["groupnorm_num_groups"]
        lstm_h   = config["lstm_hidden"]
        n_way    = config['n_way'] if config['meta_learning'] else config['pretrain_num_classes']
        drop     = config.get("dropout", 0.2)
        bidir    = config.get("bidirectional", True)
        head_type = config.get("head_type", "mlp")

        E              = config.get("num_experts", 4)
        ctx_hidden     = config.get("MOE_ctx_hidden_dim", 64)
        ctx_out        = config.get("MOE_ctx_out_dim", 32)
        MOE_drop       = config.get("MOE_dropout", drop)
        gate_temp      = config.get("MOE_gate_temperature", 1.0)
        top_k          = config.get("MOE_top_k", None)
        mlp_mult       = config.get("MOE_mlp_hidden_mult", 1.0)
        routing_signal = config.get("MOE_routing_signal", "context_proj")
        use_shared     = config.get("MOE_use_shared_expert", False)
        demo_embed_dim = config.get("demo_embed_dim", 8)

        self.use_imu        = use_imu
        self.num_experts    = E
        self.routing_signal = routing_signal
        self.use_shared     = use_shared
        D = 2 if bidir else 1

        in_ch = C_emg + (C_imu if use_imu else 0)

        # ── Shared CNN ───────────────────────────────────────────────────────
        cnn_layers: List[nn.Module] = []
        curr_in, curr_out = in_ch, base_f
        for _ in range(n_layers):
            cnn_layers += [
                nn.Conv1d(curr_in, curr_out, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.GroupNorm(num_groups=gn_grps, num_channels=curr_out),
                nn.GELU(),
            ]
            curr_in = curr_out
            curr_out = curr_out * 2
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_ch = curr_in

        # ── Demographics embedder ────────────────────────────────────────────
        if routing_signal == "context_proj_with_demo":
            demo_in_dim = config["demo_in_dim"]
            self.demo_embedder = DemographicsEmbedder(
                demo_in_dim=demo_in_dim,
                demo_embed_dim=demo_embed_dim,
                dropout=MOE_drop,
            )
        else:
            self.demo_embedder = None

        # ── Routing module ───────────────────────────────────────────────────
        self.router, gate_in_dim = build_routing_module(
            routing_signal   = routing_signal,
            in_ch            = cnn_out_ch,
            ctx_hidden_dim   = ctx_hidden,
            ctx_out_dim      = ctx_out,
            use_conv_in_proj = False,   # middle: CNN already extracted features
            conv_kernel      = k,
            gn_groups        = gn_grps,
            dropout          = MOE_drop,
            demo_embed_dim   = demo_embed_dim,
        )

        # ── MLP Experts ──────────────────────────────────────────────────────
        hidden_ch = max(cnn_out_ch, int(cnn_out_ch * mlp_mult))
        expert_mlps = []
        for _ in range(E):
            expert_mlps.append(nn.Sequential(
                nn.Linear(cnn_out_ch, hidden_ch),
                nn.GELU(),
                nn.Dropout(MOE_drop),
                nn.Linear(hidden_ch, cnn_out_ch),
            ))
        self.expert_mlps = nn.ModuleList(expert_mlps)

        # ── Optional shared expert ───────────────────────────────────────────
        if use_shared:
            shared_mlp = nn.Sequential(
                nn.Linear(cnn_out_ch, hidden_ch),
                nn.GELU(),
                nn.Dropout(MOE_drop),
                nn.Linear(hidden_ch, cnn_out_ch),
            )
            self.shared_expert = SharedExpert(shared_mlp)
        else:
            self.shared_expert = None

        # ── Gate ────────────────────────────────────────────────────────────
        self.gate = MOEGate(gate_in_dim, E, top_k=top_k, temperature=gate_temp)

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(cnn_out_ch, lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm3 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            _init_lstm(lstm)
        self.lstm_dropout = nn.Dropout(drop)
        self.feat_dim = lstm_h * D

        # ── Head ────────────────────────────────────────────────────────────
        if head_type == "linear":
            self.head = nn.Linear(self.feat_dim, n_way)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.GELU(), nn.Dropout(drop),
                nn.Linear(self.feat_dim // 2, n_way),
            )

    # ─────────────────────────────────────────────────────────────────────────
    def _get_routing_vector(self, x: torch.Tensor,
                            demographics: Optional[torch.Tensor]) -> torch.Tensor:
        if self.routing_signal == "context_proj_with_demo":
            demo_embed = self.demo_embedder(demographics) \
                if (self.demo_embedder is not None and demographics is not None) \
                else None
            return self.router(x, demo_embed=demo_embed)
        return self.router(x)

    # ─────────────────────────────────────────────────────────────────────────
    def backbone(self, x_emg, x_imu=None, demographics=None,
                 return_routing: bool = False):
        x = x_emg
        if self.use_imu and x_imu is not None:
            x = torch.cat([x, x_imu], dim=1)

        h = self.cnn(x)                         # (B, cnn_out_ch, T')

        r = self._get_routing_vector(h, demographics)
        w_hard, w_soft = self.gate(r)           # each (B, E)

        h_t = h.permute(0, 2, 1)               # (B, T', cnn_out_ch)
        expert_outs = [exp(h_t) for exp in self.expert_mlps]
        stacked = torch.stack(expert_outs, dim=1)   # (B, E, T', cnn_out_ch)
        w_4d    = w_hard.unsqueeze(-1).unsqueeze(-1)
        h_mix   = (stacked * w_4d).sum(dim=1)       # (B, T', cnn_out_ch)

        # Add shared expert if enabled
        if self.shared_expert is not None:
            h_mix = h_mix + self.shared_expert(h_t)

        h1, _ = self.lstm1(h_mix); h1 = self.lstm_dropout(h1)
        h2, _ = self.lstm2(h1);    h2 = self.lstm_dropout(h2)
        h3, _ = self.lstm3(h2)

        l1 = h1.mean(dim=1)
        l2 = h2.mean(dim=1)
        l3 = h3.mean(dim=1)

        if return_routing:
            return l3, [l1, l2, l3], w_hard, w_soft
        return l3, [l1, l2, l3]

    def forward(self, x_emg, x_imu=None, demographics=None,
                return_routing: bool = False):
        result = self.backbone(x_emg, x_imu, demographics, return_routing=return_routing)
        if return_routing:
            feat, _, w_hard, w_soft = result
            return self.head(feat), {"gate_weights": w_hard, "gate_weights_soft": w_soft}
        return self.head(result[0])

    def get_features(self, x_emg, x_imu=None, demographics=None):
        with torch.no_grad():
            return self.backbone(x_emg, x_imu, demographics)


# ─────────────────────────────────────────────────────────────────────────────
# ContrastiveGestureEncoder with MOE support
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveEncoderMOE(nn.Module):
    """
    ContrastiveGestureEncoder with MOE support (both placements).

    Since the contrastive encoder uses a custom CNN structure (separate EMG /
    IMU CNNs + FiLM), we keep that structure and inject MOE at either:
      "encoder": each expert is a full (emg_cnn, imu_cnn) pair
      "middle":  shared (emg_cnn, imu_cnn), MOE MLP experts at the fusion point

    Forward: (B, C_emg, T), (B, C_imu, T) → (B, embedding_dim) L2-normalised.
    The signature is compatible with ContrastiveGestureEncoder.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.use_imu       = config.get("use_imu", True)
        self.MOE_placement = config.get("MOE_placement", "middle")

        E              = config.get("num_experts", 4)
        ctx_hidden     = config.get("MOE_ctx_hidden_dim", 64)
        ctx_out        = config.get("MOE_ctx_out_dim", 32)
        MOE_drop       = config.get("MOE_dropout", config.get("dropout", 0.1))
        gate_temp      = config.get("MOE_gate_temperature", 1.0)
        top_k          = config.get("MOE_top_k", None)
        mlp_mult       = config.get("MOE_mlp_hidden_mult", 1.0)
        width_mult     = config.get("MOE_expert_expand", 1.0)
        routing_signal = config.get("MOE_routing_signal", "context_proj")
        use_shared     = config.get("MOE_use_shared_expert", False)
        demo_embed_dim = config.get("demo_embed_dim", 8)

        self.num_experts    = E
        self.routing_signal = routing_signal
        self.use_shared     = use_shared

        emg_in_ch  = config["emg_in_ch"]
        imu_in_ch  = config.get("imu_in_ch", 72)
        emg_base   = config["emg_base_cnn_filters"]
        imu_base   = config["imu_base_cnn_filters"]
        emg_layers = config["emg_cnn_layers"]
        imu_layers = config["imu_cnn_layers"]
        k          = config["cnn_kernel_size"]
        gn_grps    = config["groupnorm_num_groups"]
        drop       = config.get("dropout", 0.1)

        emg_out_ch  = emg_base * (2 ** (emg_layers - 1))
        imu_out_ch  = imu_base * (2 ** (imu_layers - 1)) if self.use_imu else 0
        cnn_out_dim = emg_out_ch + imu_out_ch

        arch = config.get("arch_mode", "cnn_attn")
        self.arch = arch

        # ── Demographics embedder ────────────────────────────────────────────
        if routing_signal == "context_proj_with_demo":
            demo_in_dim = config["demo_in_dim"]
            self.demo_embedder = DemographicsEmbedder(
                demo_in_dim=demo_in_dim,
                demo_embed_dim=demo_embed_dim,
                dropout=MOE_drop,
            )
        else:
            self.demo_embedder = None

        # ── Build expert CNNs or shared CNN depending on placement ───────────
        if self.MOE_placement == "encoder":
            self.emg_experts = nn.ModuleList([
                self._build_cnn(emg_in_ch, emg_base, emg_layers, k, 1, gn_grps, drop, width_mult)
                for _ in range(E)
            ])
            self.imu_experts = nn.ModuleList([
                self._build_cnn(imu_in_ch, imu_base, imu_layers, k, 1, gn_grps, drop, width_mult)
                for _ in range(E)
            ]) if self.use_imu else None

            in_ch_ctx = emg_in_ch + (imu_in_ch if self.use_imu else 0)
            self.router, gate_in_dim = build_routing_module(
                routing_signal   = routing_signal,
                in_ch            = in_ch_ctx,
                ctx_hidden_dim   = ctx_hidden,
                ctx_out_dim      = ctx_out,
                use_conv_in_proj = True,
                conv_kernel      = k,
                gn_groups        = gn_grps,
                dropout          = MOE_drop,
                demo_embed_dim   = demo_embed_dim,
            )

            scaled_emg = int(emg_out_ch * width_mult)
            scaled_imu = int(imu_out_ch * width_mult) if self.use_imu else 0
            fused_dim  = scaled_emg + scaled_imu

            # Shared expert for encoder placement
            if use_shared:
                shared_emg = self._build_cnn(emg_in_ch, emg_base, emg_layers, k, 1, gn_grps, drop, width_mult)
                if self.use_imu:
                    shared_imu = self._build_cnn(imu_in_ch, imu_base, imu_layers, k, 1, gn_grps, drop, width_mult)
                    self._shared_emg_cnn = shared_emg
                    self._shared_imu_cnn = shared_imu
                    # Wrap as a tuple-returning module via a small lambda — we handle
                    # this case explicitly in _encode_signals instead.
                    self.shared_expert = None   # handled manually below
                    self._use_split_shared = True
                else:
                    self.shared_expert = SharedExpert(shared_emg)
                    self._use_split_shared = False
                # Register split shared CNNs so optimizer sees them
                if use_shared and self.use_imu:
                    self.shared_emg_expert = SharedExpert(shared_emg)
                    self.shared_imu_expert = SharedExpert(shared_imu)
            else:
                self.shared_expert = None

        else:  # "middle"
            self.emg_encoder = self._build_cnn(emg_in_ch, emg_base, emg_layers, k, 1, gn_grps, drop)
            self.imu_encoder = self._build_cnn(imu_in_ch, imu_base, imu_layers, k, 1, gn_grps, drop) if self.use_imu else None

            self.router, gate_in_dim = build_routing_module(
                routing_signal   = routing_signal,
                in_ch            = cnn_out_dim,
                ctx_hidden_dim   = ctx_hidden,
                ctx_out_dim      = ctx_out,
                use_conv_in_proj = False,
                conv_kernel      = k,
                gn_groups        = gn_grps,
                dropout          = MOE_drop,
                demo_embed_dim   = demo_embed_dim,
            )

            hidden_ch = max(cnn_out_dim, int(cnn_out_dim * mlp_mult))
            self.expert_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cnn_out_dim, hidden_ch),
                    nn.GELU(),
                    nn.Dropout(MOE_drop),
                    nn.Linear(hidden_ch, cnn_out_dim),
                ) for _ in range(E)
            ])

            if use_shared:
                shared_mlp = nn.Sequential(
                    nn.Linear(cnn_out_dim, hidden_ch),
                    nn.GELU(),
                    nn.Dropout(MOE_drop),
                    nn.Linear(hidden_ch, cnn_out_dim),
                )
                self.shared_expert = SharedExpert(shared_mlp)
            else:
                self.shared_expert = None

            fused_dim = cnn_out_dim

        # ── Gating ──────────────────────────────────────────────────────────
        self.gate = MOEGate(gate_in_dim, E, top_k=top_k, temperature=gate_temp)

        # ── Temporal module ──────────────────────────────────────────────────
        if arch == "cnn_lstm":
            lstm_h      = config["lstm_hidden"]
            lstm_layers = config.get("lstm_layers", 2)
            self.temporal = nn.LSTM(
                input_size=fused_dim,
                hidden_size=lstm_h,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=drop if lstm_layers > 1 else 0,
            )
            backbone_dim = lstm_h * 2
            self.use_gap = config.get("use_GlobalAvgPooling", True)
            self.attn_pool = None
        elif arch == "cnn_attn":
            try:
                from system.pretraining.contrastive_net.contrastive_encoder import AttentionPool1d
            except ImportError:
                from pretraining.contrastive_net.contrastive_encoder import AttentionPool1d
            self.temporal = None
            self.attn_pool = AttentionPool1d(
                in_dim=fused_dim,
                num_heads=config.get("attn_pool_heads", 4),
            )
            backbone_dim = fused_dim
        else:
            raise ValueError(f"Unknown arch_mode: {arch}")

        self.backbone_dim = backbone_dim

        # ── Projection Head ──────────────────────────────────────────────────
        try:
            from system.pretraining.contrastive_net.contrastive_encoder import ProjectionHead
        except ImportError:
            from pretraining.contrastive_net.contrastive_encoder import ProjectionHead
        self.proj_head = ProjectionHead(
            in_dim=backbone_dim,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config.get("proj_hidden_dim", 256),
        )
        self.embedding_dim = config["embedding_dim"]

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_cnn(in_ch, base_filters, num_layers, kernel_size, stride,
                   gn_groups, dropout, width_mult=1.0):
        layers: List[nn.Module] = []
        c_in  = in_ch
        c_out = max(gn_groups, int(base_filters * width_mult))
        c_out = (c_out // gn_groups) * gn_groups
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2),
                nn.GroupNorm(gn_groups, c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c_in  = c_out
            c_out = c_out * 2
        return nn.Sequential(*layers)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_routing_vector(self, x: torch.Tensor,
                            demographics: Optional[torch.Tensor]) -> torch.Tensor:
        if self.routing_signal == "context_proj_with_demo":
            demo_embed = self.demo_embedder(demographics) \
                if (self.demo_embedder is not None and demographics is not None) \
                else None
            return self.router(x, demo_embed=demo_embed)
        return self.router(x)

    # ─────────────────────────────────────────────────────────────────────────
    def _encode_signals(self, x_emg, x_imu=None, demographics=None,
                        return_routing: bool = False):
        if self.MOE_placement == "encoder":
            raw = x_emg if not (self.use_imu and x_imu is not None) else \
                  torch.cat([x_emg, x_imu], dim=1)
            r = self._get_routing_vector(raw, demographics)
            w_hard, w_soft = self.gate(r)

            emg_feats = [exp(x_emg) for exp in self.emg_experts]
            if self.use_imu and x_imu is not None and self.imu_experts is not None:
                imu_feats = [exp(x_imu) for exp in self.imu_experts]
                expert_combined = [torch.cat([e, i], dim=1)
                                   for e, i in zip(emg_feats, imu_feats)]
            else:
                expert_combined = emg_feats

            stacked  = torch.stack(expert_combined, dim=1)   # (B, E, C, T')
            w_4d     = w_hard.unsqueeze(-1).unsqueeze(-1)
            combined = (stacked * w_4d).sum(dim=1)           # (B, C, T')

            # Shared expert for encoder placement (split EMG/IMU case)
            if self.use_shared:
                if self.use_imu and x_imu is not None and hasattr(self, 'shared_emg_expert'):
                    shared_out = torch.cat([
                        self.shared_emg_expert(x_emg),
                        self.shared_imu_expert(x_imu),
                    ], dim=1)
                    combined = combined + shared_out
                elif self.shared_expert is not None:
                    combined = combined + self.shared_expert(x_emg)

        else:  # "middle"
            e = self.emg_encoder(x_emg)
            i = self.imu_encoder(x_imu) if (self.imu_encoder is not None and x_imu is not None) else None
            combined = torch.cat([e, i], dim=1) if i is not None else e

            r = self._get_routing_vector(combined, demographics)
            w_hard, w_soft = self.gate(r)

            h_t = combined.permute(0, 2, 1)
            expert_outs = [exp(h_t) for exp in self.expert_mlps]
            stacked  = torch.stack(expert_outs, dim=1)
            w_4d     = w_hard.unsqueeze(-1).unsqueeze(-1)
            h_mix    = (stacked * w_4d).sum(dim=1)

            if self.shared_expert is not None:
                h_mix = h_mix + self.shared_expert(h_t)

            combined = h_mix.permute(0, 2, 1)

        # Temporal module
        if self.temporal is not None:
            x_t = combined.permute(0, 2, 1)
            out, (hn, _) = self.temporal(x_t)
            if self.use_gap:
                feat = out.mean(dim=1)
            else:
                feat = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            feat = self.attn_pool(combined)

        if return_routing:
            return feat, w_hard, w_soft
        return feat, None, None

    # ─────────────────────────────────────────────────────────────────────────
    def encode(self, x_emg, x_imu=None, demographics=None):
        """Backbone only (no projection head). Returns (B, backbone_dim)."""
        feat, _, _ = self._encode_signals(x_emg, x_imu, demographics)
        return feat

    def forward(self, x_emg, x_imu=None, demographics=None,
                return_routing: bool = False):
        """Returns L2-normalised embedding (B, embedding_dim)."""
        feat, w_hard, w_soft = self._encode_signals(x_emg, x_imu, demographics,
                                                    return_routing=return_routing)
        z = self.proj_head(feat)
        if return_routing:
            return z, {"gate_weights": w_hard, "gate_weights_soft": w_soft}
        return z

    @torch.no_grad()
    def get_prototypes(self, support_emg, support_labels, support_imu=None, support_demo=None):
        import torch.nn.functional as F
        was_training = self.training
        self.eval()
        z = self.forward(support_emg, support_imu)
        prototypes = {}
        for cls in support_labels.unique():
            mask = support_labels == cls
            proto = z[mask].mean(dim=0)
            prototypes[cls.item()] = F.normalize(proto, dim=0)
        if was_training:
            self.train()
        return prototypes

    @torch.no_grad()
    def predict(self, query_emg, prototypes, query_imu=None, query_demo=None):
        import torch.nn.functional as F
        was_training = self.training
        self.eval()
        z_q = self.forward(query_emg, query_imu)
        labels = sorted(prototypes.keys())
        proto_mat = torch.stack([prototypes[l] for l in labels], dim=0).to(z_q.device)
        sims = z_q @ proto_mat.T
        pred_idx = sims.argmax(dim=1)
        pred_labels = torch.tensor([labels[i] for i in pred_idx.tolist()],
                                   dtype=torch.long, device=z_q.device)
        if was_training:
            self.train()
        return pred_labels


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build_MOE_model
# ─────────────────────────────────────────────────────────────────────────────

def build_MOE_model(config: dict) -> nn.Module:
    """
    Instantiate the correct MOE model.

    config["model_type"]      : "MetaCNNLSTM" | "DeepCNNLSTM" | "ContrastiveNet"
    config["MOE_placement"]   : "encoder" | "middle"
    config["MOE_routing_signal"]: "linear_proj_input" | "context_proj" |
                                  "context_proj_with_demo"

    Example config fragment:
        config["use_MOE"]                = True
        config["MOE_placement"]          = "middle"
        config["MOE_routing_signal"]     = "context_proj"
        config["num_experts"]            = 4
        config["MOE_ctx_hidden_dim"]     = 64
        config["MOE_ctx_out_dim"]        = 32
        config["MOE_gate_temperature"]   = 1.0
        config["MOE_top_k"]              = None       # dense routing
        config["MOE_use_shared_expert"]  = False
        config["MOE_expert_expand"]      = 1.0
        config["MOE_mlp_hidden_mult"]    = 1.0
        config["MOE_aux_coeff"]          = 1e-2       # Switch load balance loss
        config["MOE_importance_coeff"]   = 1e-2       # Shazeer importance loss
        # Only needed for "context_proj_with_demo":
        config["demo_in_dim"]            = 5
        config["demo_embed_dim"]         = 8
    """
    model_type = config["model_type"]
    placement  = config.get("MOE_placement", "middle")

    if model_type == "MetaCNNLSTM":
        if placement == "encoder":
            return MetaCNNLSTM_EncoderMOE(config)
        else:
            return MetaCNNLSTM_MiddleMOE(config)

    elif model_type == "DeepCNNLSTM":
        if placement == "encoder":
            return DeepCNNLSTM_EncoderMOE(config)
        else:
            return DeepCNNLSTM_MiddleMOE(config)

    elif model_type == "ContrastiveNet":
        return ContrastiveEncoderMOE(config)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from MetaCNNLSTM, DeepCNNLSTM, ContrastiveNet."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained weight transfer utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained_into_MOE(MOE_model: nn.Module,
                             pretrained_state_dict: dict,
                             placement: str,
                             seed_experts: bool = True,
                             verbose: bool = True) -> nn.Module:
    """
    Load a pretrained (non-MOE) backbone into an MOE model.

    For "middle" placement:
      - Loads `conv` / `cnn` weights from the pretrained dict directly.
        (The key names match since the CNN module is structurally identical.)
      - LSTM and head weights are also loaded if present.
      - Expert MLP weights are left random (new parameters).

    For "encoder" placement:
      - LSTM and head weights are loaded.
      - Optionally seeds all expert CNNs from the pretrained conv weights
        (seed_experts=True).  This gives a warm start where each expert begins
        identical to the pretrained CNN and then diverges during training.
        seed_experts=False leaves experts random.

    Args:
        MOE_model:            An instantiated MOE model (from build_MOE_model).
        pretrained_state_dict: state_dict from torch.load(checkpoint)["model_state_dict"].
        placement:            "encoder" | "middle".
        seed_experts:         If True (encoder MOE only), copy pretrained CNN
                              weights into each expert CNN.
        verbose:              Print load summary.

    Returns:
        MOE_model with loaded weights (in-place).
    """
    MOE_dict  = MOE_model.state_dict()
    loaded_keys: List[str] = []
    skipped_keys: List[str] = []

    if placement == "middle":
        # Keys to load: everything EXCEPT expert_mlps, gate, router, demo_embedder
        skip_prefixes = ("expert_mlps.", "gate.", "router.", "demo_embedder.",
                         "shared_expert.")
        for k, v in pretrained_state_dict.items():
            if any(k.startswith(p) for p in skip_prefixes):
                skipped_keys.append(k)
                continue
            if k in MOE_dict and MOE_dict[k].shape == v.shape:
                MOE_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

    elif placement == "encoder":
        # Load LSTMs and head; optionally seed expert CNNs
        skip_prefixes = ("conv.", "cnn.")
        for k, v in pretrained_state_dict.items():
            if any(k.startswith(p) for p in skip_prefixes):
                if seed_experts:
                    for e_idx in range(getattr(MOE_model, "num_experts", 0)):
                        new_key = k.replace("conv.", f"expert_cnns.{e_idx}.", 1)
                        new_key = new_key.replace("cnn.",  f"expert_cnns.{e_idx}.", 1)
                        if new_key in MOE_dict and MOE_dict[new_key].shape == v.shape:
                            MOE_dict[new_key] = v
                            loaded_keys.append(new_key)
                continue
            if k in MOE_dict and MOE_dict[k].shape == v.shape:
                MOE_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

    MOE_model.load_state_dict(MOE_dict)

    if verbose:
        print(f"[load_pretrained_into_MOE] Loaded {len(loaded_keys)} keys, "
              f"skipped {len(skipped_keys)} keys.")
        if skipped_keys:
            print(f"  Skipped: {skipped_keys[:8]}{'...' if len(skipped_keys) > 8 else ''}")

    return MOE_model