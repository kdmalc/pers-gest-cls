"""
attention.py
============
Attention building blocks that are safe under torch.func.functional_call (the MAML++ inner loop).

Why not nn.MultiheadAttention: its fused fast path reads self.in_proj_weight directly, bypassing
functional_call's parameter substitution, so the inner loop would silently stop adapting the
attention weights (tests/test_models_functional.py, SPEC_moe_compute.md section 3). Everything here
is plain nn.Linear / nn.LayerNorm plus explicit math, so every parameter is read through a module
call that functional_call intercepts. The attention is written out (softmax(QK^T / sqrt(d)) V)
rather than F.scaled_dot_product_attention so second-order MAML (double backward) also works;
at 64 tokens the cost difference is negligible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positions(T: int, d: int, device, dtype) -> torch.Tensor:
    """(1, T, d) fixed sinusoidal position code; computed on the fly (no buffer, nothing to adapt)."""
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0).to(dtype)


class MultiHeadAttention(nn.Module):
    """Multi-head attention; self-attention when kv is None, cross-attention otherwise."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, d_kv: int | None = None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        d_kv = d_model if d_kv is None else d_kv
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        # no key bias: q.(k + b) adds the same q.b to every key's score, which softmax cancels, so a
        # key bias is a dead parameter (tests/test_models_functional.py flags it as inert)
        self.k = nn.Linear(d_kv, d_model, bias=False)
        self.v = nn.Linear(d_kv, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, Tq, d_model); kv: (B, Tk, d_kv) or None -> (B, Tq, d_model)."""
        kv = x if kv is None else kv
        B, Tq, _ = x.shape
        Tk = kv.shape[1]
        q = self.q(x).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)    # (B, h, Tq, dh)
        k = self.k(kv).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(kv).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)               # (B, h, Tq, Tk)
        att = self.attn_drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, Tq, self.n_heads * self.d_head)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class MoEFeedForward(nn.Module):
    """
    Token-level sparse MoE feed-forward (Switch / ST-MoE style; T4).

    Each token is routed by a linear router to its top-k experts (small MLPs). Experts are run on
    all tokens and mixed by a running weighted sum, the same scheme as MOE_encoder._weighted_expert_sum;
    an expert that no token in the batch selected is skipped (its gradient is exactly zero either way).

    Returns (y, routing) with routing = {"gate_weights": (N, E) post-top-k, "gate_weights_soft": (N, E),
    "router_logits": (N, E)}, N = B*T tokens. mamlpp._compute_aux_loss consumes these unchanged.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int, dropout: float,
                 temperature: float = 1.0):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff, dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        logits = self.router(flat) / self.temperature
        soft = logits.softmax(dim=-1)
        if self.top_k < self.num_experts:
            _, idx = torch.topk(soft, self.top_k, dim=-1)
            mask = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
            hard = soft * mask
            hard = hard / (hard.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            hard = soft
        y = None
        for e, expert in enumerate(self.experts):
            w = hard[:, e:e + 1]
            if self.top_k < self.num_experts and not bool((w > 0).any()):
                continue
            term = expert(flat) * w
            y = term if y is None else y + term
        return y.reshape(B, T, D), {"gate_weights": hard, "gate_weights_soft": soft, "router_logits": logits}


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block; the feed-forward is dense or a token-level MoE."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 moe_experts: int = 0, moe_top_k: int = 1, moe_temperature: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.is_moe = moe_experts > 0
        if self.is_moe:
            self.ff = MoEFeedForward(d_model, d_ff, moe_experts, moe_top_k, dropout, moe_temperature)
        else:
            self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """Returns (x, routing_or_None)."""
        x = x + self.drop(self.attn(self.norm1(x)))
        if self.is_moe:
            y, routing = self.ff(self.norm2(x))
            return x + y, routing
        return x + self.ff(self.norm2(x)), None


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention: x attends to ctx (F3 fusion, X6 routing). Residual on x."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, d_ctx: int | None = None):
        super().__init__()
        d_ctx = d_model if d_ctx is None else d_ctx
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_ctx)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, d_kv=d_ctx)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.attn(self.norm_x(x), self.norm_ctx(ctx)))


class TemporalTransformer(nn.Module):
    """
    Drop-in replacement for the 3-layer LSTM stack (T2, T0, T4).

    (B, C_in, T) -> input projection to d_model -> + sinusoidal positions -> n_blocks blocks -> final LN.
    backbone-style output: (feat (B, d_model) mean over time, per-block mean features, routing list).
    Blocks with index in moe_layers use a MoE feed-forward (T4).
    """

    def __init__(self, in_ch: int, d_model: int, n_heads: int, n_blocks: int, d_ff: int, dropout: float,
                 moe_layers: tuple = (), moe_experts: int = 0, moe_top_k: int = 1,
                 moe_temperature: float = 1.0):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout,
                             moe_experts=(moe_experts if i in moe_layers else 0),
                             moe_top_k=moe_top_k, moe_temperature=moe_temperature)
            for i in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, h: torch.Tensor):
        x = self.in_proj(h.permute(0, 2, 1))                               # (B, T, d)
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)
        feats, routings = [], []
        for blk in self.blocks:
            x, r = blk(x)
            feats.append(x.mean(dim=1))
            if r is not None:
                routings.append(r)
        x = self.norm(x)
        return x.mean(dim=1), feats, routings


def merge_routing(routings: list) -> dict | None:
    """Concatenate per-layer token routing dicts along the token axis (aux loss is a mean over rows)."""
    if not routings:
        return None
    return {k: torch.cat([r[k] for r in routings], dim=0) for k in routings[0]}
