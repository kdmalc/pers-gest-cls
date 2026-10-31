"""
moemeta_ts.py
=============
MoEMeta-TS: a time-series port of MoEMeta (Wu & Yin, "Mixture-of-Experts Meta Learning for Few-Shot
Relational Learning", NeurIPS 2025, arXiv 2510.23013). Reviewer ARaC asked for this comparison; the
rebuttal's Port B was ANIL, not MoEMeta (docs/analysis/ANALYSIS_moemeta.md section 4).

The port maps a knowledge-graph triplet (head h, relation r, tail t) onto an episode as

    h  = the user centroid  u  (mean of the support embeddings: everything we know about a new person)
    r  = the gesture class  c
    t  = one window embedding z

so a support window (z_i, y_i) is the triplet (u, y_i, z_i). Then, following the paper's equations:

  relation meta (Eq 5-8)   r_i = (1/N) sum_j g_ij f_j([u, z_i]);  g = top-N of softmax(Gate([u, z_i]))
                           R_c = mean of r_i over the support windows of class c
  local adaptation (9-15)  task vectors p_h, p_r, p_t (zero-initialised, see below) and the R_c are updated
                           by `moemeta_inner_steps` gradient steps on the support loss; the encoder,
                           experts and gate (Phi) stay frozen at test time
  projections (9-11)       h'_c = u + (p_h . R_c) R_c,  R'_c = R_c + (p_r . R_c) R_c,  t'_c = z + (p_t . R_c) R_c
  score (12)               d(z, c) = || h'_c + R'_c - t'_c ||_2    (smaller = better)
  loss (13, 16)            sum of max(0, d(z, y) + margin - d(z, c')) over the wrong classes c'

Substitutions, all forced by the change of domain (list them in the paper):
  * entity embeddings: the paper uses TransE-pretrained entities plus a neighbour attention; we use the
    same CNN-LSTM encoder family as every other method, trained end to end in the outer loop
  * the negative in the margin loss corrupts the class (what we rank) instead of the tail entity (what
    they rank); the ranked slot is corrupted in both cases
  * the task vectors start at zero rather than random: the projections are then exactly the identity
    before adaptation, which makes the no-local-adaptation variant and the first inner step well defined
  * one relation meta per class rather than one per task, because an episode has N classes to rank

With one identity expert at 1-shot the score is || z_i - z_q ||, i.e. 1-nearest-neighbour, which is why
MoEMeta-TS is a credible competitor given kNN's 72.9% on fs64 (ANALYSIS_moemeta section 3).

  config["moemeta_variant"]     "full" | "no_local" (paper's w/o L.A) | "mlp" (w/o MoE: one expert, as
                                MetaR) | "uniform" (every expert at 1/M: is the gate doing anything?)
  config["moemeta_num_experts"] M (paper 32)       config["moemeta_top_n"]        N (paper 5)
  config["moemeta_expert_hidden"] (paper 64)       config["moemeta_margin"]       gamma (paper 1)
  config["moemeta_inner_steps"] (paper's Algorithm 1 shows 1)   config["moemeta_inner_lr"]  alpha (unstated)
  config["moemeta_loss"]        "margin" (paper) | "ce" (sensitivity check: cross-entropy on -d)
  config["moemeta_first_order"] False: differentiate through the inner step (the paper does not say)

The class is an EpisodicMetricLearner (same episode_logits / embed interface), so metric_pretrain,
metric_adapt_and_eval, the HPO objective and arch_variant_eval all run it unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_variants.metric_learners import EpisodicMetricLearner, _build_encoder

MOEMETA_VARIANTS = ("full", "no_local", "mlp", "uniform")
MOEMETA_LOSSES = ("margin", "ce")


class MoEMetaTS(EpisodicMetricLearner):
    def __init__(self, config: dict):
        nn.Module.__init__(self)                     # not super(): the parent validates metric_kind
        self.kind = "moemeta"
        variant = config["moemeta_variant"]
        if variant not in MOEMETA_VARIANTS:
            raise ValueError(f"moemeta_variant must be one of {MOEMETA_VARIANTS}, got {variant!r}")
        loss_kind = config["moemeta_loss"]
        if loss_kind not in MOEMETA_LOSSES:
            raise ValueError(f"moemeta_loss must be one of {MOEMETA_LOSSES}, got {loss_kind!r}")
        self.variant, self.loss_kind = variant, loss_kind
        self.encoder = _build_encoder(config)
        D = self.encoder.feat_dim
        self.feat_dim = D

        M, N = int(config["moemeta_num_experts"]), int(config["moemeta_top_n"])
        if variant == "mlp":
            M = N = 1
        if not 1 <= N <= M:
            raise ValueError(f"moemeta_top_n must be in [1, num_experts], got N={N}, M={M}")
        self.M, self.N = M, N
        H = int(config["moemeta_expert_hidden"])
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * D, H), nn.ReLU(), nn.Linear(H, D)) for _ in range(M)])
        # the paper's gate is a two-layer ReLU MLP; "uniform" and "mlp" have no gate
        self.gate = (nn.Sequential(nn.Linear(2 * D, H), nn.ReLU(), nn.Linear(H, M))
                     if (M > 1 and variant != "uniform") else None)

        self.inner_steps = 0 if variant == "no_local" else int(config["moemeta_inner_steps"])
        self.inner_lr = float(config["moemeta_inner_lr"])
        self.margin = float(config["moemeta_margin"])
        self.first_order = bool(config["moemeta_first_order"])
        self.last_gate_soft = None                   # (Ns, M) of the last episode, for routing analysis
        self.last_gate_hard = None

    # ── MoE relation meta (Eq 5-7) ───────────────────────────────────────────
    def relation_meta(self, u: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        """u (D), zs (Ns, D) -> r_i (Ns, D)."""
        x = torch.cat([u.unsqueeze(0).expand_as(zs), zs], dim=-1)          # (Ns, 2D)
        outs = torch.stack([e(x) for e in self.experts], dim=1)            # (Ns, M, D)
        if self.gate is None:
            soft = outs.new_full((zs.shape[0], self.M), 1.0 / self.M)
            self.last_gate_soft = self.last_gate_hard = soft.detach()
            return outs.mean(dim=1)                                        # N = M: (1/M) sum_j f_j
        soft = self.gate(x).softmax(dim=-1)                                # (Ns, M)
        top = soft.topk(self.N, dim=-1).indices
        hard = torch.zeros_like(soft).scatter(1, top, soft.gather(1, top))  # top-N kept, no renormalisation
        self.last_gate_soft, self.last_gate_hard = soft.detach(), hard.detach()
        return (hard.unsqueeze(-1) * outs).sum(dim=1) / self.N             # Eq 5, with its 1/N

    # ── projections and score (Eq 9-12) ──────────────────────────────────────
    def _score(self, u, z, R, p) -> torch.Tensor:
        """u (D), z (B, D), R (n, D), p (3, D) -> distance (B, n)."""
        coef = p @ R.t()                                                   # (3, n): p_x . R_c
        h = u.unsqueeze(0) + coef[0].unsqueeze(-1) * R                      # (n, D)
        Rp = R + coef[1].unsqueeze(-1) * R                                  # (n, D)
        t = z.unsqueeze(1) + (coef[2].unsqueeze(-1) * R).unsqueeze(0)       # (B, n, D)
        return (h + Rp).unsqueeze(0).sub(t).norm(dim=-1)                    # (B, n)

    def _loss_from_dist(self, dist: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "ce":
            return F.cross_entropy(-dist, y)
        pos = dist.gather(1, y.unsqueeze(1))                               # (B, 1)
        wrong = torch.ones_like(dist).scatter(1, y.unsqueeze(1), 0.0)
        hinge = F.relu(pos + self.margin - dist) * wrong
        return hinge.sum() / wrong.sum().clamp_min(1.0)

    def query_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Outer-loop loss on the query logits (= -distance). Called by metric_learners._episode_loss."""
        return self._loss_from_dist(-logits, y)

    # ── local adaptation (Eq 13-15) ──────────────────────────────────────────
    def _adapt(self, u, zs, y_s, R):
        p = zs.new_zeros(3, R.shape[1])
        if self.inner_steps == 0:
            return R, p
        # Outer training differentiates through the inner steps (unless first_order). At evaluation the
        # outer context is no_grad, so the inner loop builds its own local graph on detached leaves.
        keep_graph = self.training and torch.is_grad_enabled()
        with torch.enable_grad():
            if not keep_graph:
                R = R.detach().requires_grad_(True)
            p = p.requires_grad_(True)
            for _ in range(self.inner_steps):
                loss = self._loss_from_dist(self._score(u, zs, R, p), y_s)
                create = keep_graph and not self.first_order
                gR, gp = torch.autograd.grad(loss, [R, p], create_graph=create)
                R, p = R - self.inner_lr * gR, p - self.inner_lr * gp
                if not keep_graph:
                    R, p = R.detach().requires_grad_(True), p.detach().requires_grad_(True)
        if not keep_graph:
            R, p = R.detach(), p.detach()
        return R, p

    # ── episode ──────────────────────────────────────────────────────────────
    def episode_logits(self, support: dict, query: dict, device):
        def _get(b, k):
            v = b.get(k)
            return v.to(device) if v is not None else None
        s_emg, s_imu, s_y = _get(support, "emg"), _get(support, "imu"), support["labels"].to(device).long()
        q_emg, q_imu, q_y = _get(query, "emg"), _get(query, "imu"), query["labels"].to(device).long()
        n_s = s_emg.shape[0]
        x_imu = torch.cat([s_imu, q_imu]) if s_imu is not None else None
        z, routing = self.embed(torch.cat([s_emg, q_emg]), x_imu)
        zs, zq = z[:n_s], z[n_s:]
        n_way = int(s_y.max().item()) + 1
        onehot = F.one_hot(s_y, n_way).float()                             # (Ns, n)

        u = zs.mean(dim=0)                                                 # the user, from the support only
        r = self.relation_meta(u, zs)                                      # (Ns, D)
        R = (onehot.t() @ r) / onehot.sum(dim=0, keepdim=True).t()         # (n, D): Eq 8 per class
        R, p = self._adapt(u, zs, s_y, R)
        logits = -self._score(u, zq, R, p)                                 # argmax = nearest class
        return logits, q_y, routing


def build_moemeta_ts(config: dict) -> MoEMetaTS:
    return MoEMetaTS(config)
