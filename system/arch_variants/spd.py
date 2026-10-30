"""
spd.py
======
Covariance / SPD-matrix building blocks for the Riemannian variants (docs/analysis/COVARIANCE_TUTORIAL.md,
docs/specs/SPEC_architecture_exploration.md section 2R). Pure torch, batched, functional_call-safe
(no parameters outside nn.Modules, no in-place ops on inputs).

  shrunk_cov(x, rho)            (B, C, T) -> (B, C, C) channel covariance, shrunk towards (tr/C)·I
  sym_fn(S, fn)                 f(S) for symmetric S through eigh, eigenvalues clamped >= eps
  sqrtm_ns(S, iters)            matrix square root by Newton-Schulz (iSQRT-COV; matmuls only)
  triu_vec(M)                   upper triangle with sqrt(2) on off-diagonals (so ||vec|| = ||M||_F)
  untriu_vec(v, C)              inverse of triu_vec
  log_euclid_vec(x, rho)        triu_vec(log(shrunk_cov(x))): the log-Euclidean tangent vector at I
  block_log_vec(x, blocks, rho) the same per channel block (EMG, IMU), concatenated
  CovPool                       second-order pooling layer: Linear(H -> p), covariance over time,
                                sqrt or log normalisation, triu -> (B, p(p+1)/2)
  support_whitener(xs, rho)     R^{-1/2} with R = mean support covariance (Euclidean alignment, He & Wu 2020)

Raw-input covariances (routers, alignment) are computed under no_grad: they depend on the data only,
so no gradient ever flows through eigh there. CovPool does backpropagate through its normalisation;
"sqrt" (Newton-Schulz) needs only matmuls, "log" goes through eigh.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EIG_EPS = 1e-6


def shrunk_cov(x: torch.Tensor, rho: float) -> torch.Tensor:
    """(B, C, T) -> (B, C, C). Centred over time; (1 - rho)·S + rho·(tr S / C)·I. rho in (0, 1]."""
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"shrinkage rho must be in (0, 1], got {rho}")
    B, C, T = x.shape
    if T < 2:
        raise ValueError(f"need at least 2 time steps for a covariance, got T={T}")
    xc = x - x.mean(dim=2, keepdim=True)
    S = xc @ xc.transpose(1, 2) / (T - 1)
    mu = S.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(EIG_EPS)          # tr / C
    eye = torch.eye(C, device=x.device, dtype=x.dtype)
    return (1.0 - rho) * S + rho * mu[:, None, None] * eye


def sym_fn(S: torch.Tensor, fn) -> torch.Tensor:
    """f(S) = V f(L) V^T for symmetric (B, C, C) S; eigenvalues clamped at EIG_EPS."""
    S = 0.5 * (S + S.transpose(-1, -2))
    w, V = torch.linalg.eigh(S)
    return (V * fn(w.clamp_min(EIG_EPS)).unsqueeze(-2)) @ V.transpose(-1, -2)


def sym_log(S):
    return sym_fn(S, torch.log)


def sym_invsqrt(S):
    return sym_fn(S, torch.rsqrt)


def sqrtm_ns(S: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Matrix square root of SPD (B, C, C) by coupled Newton-Schulz on S / tr(S) (Li et al. CVPR 2018)."""
    B, C, _ = S.shape
    tr = S.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(EIG_EPS)[:, None, None]
    A = S / tr
    eye = torch.eye(C, device=S.device, dtype=S.dtype).expand(B, C, C)
    Y, Z = A, eye
    for _ in range(iters):
        Tm = 0.5 * (3.0 * eye - Z @ Y)
        Y, Z = Y @ Tm, Tm @ Z
    return Y * tr.sqrt()


def _triu_idx(C: int, device):
    return torch.triu_indices(C, C, device=device)


def triu_vec(M: torch.Tensor) -> torch.Tensor:
    """(B, C, C) symmetric -> (B, C(C+1)/2); off-diagonals times sqrt(2) so the vector norm is the Frobenius norm."""
    C = M.shape[-1]
    i, j = _triu_idx(C, M.device)
    w = torch.where(i == j, torch.ones((), device=M.device, dtype=M.dtype),
                    torch.full((), math.sqrt(2.0), device=M.device, dtype=M.dtype))
    return M[..., i, j] * w


def untriu_vec(v: torch.Tensor, C: int) -> torch.Tensor:
    """Inverse of triu_vec: (B, C(C+1)/2) -> (B, C, C) symmetric."""
    if v.shape[-1] != C * (C + 1) // 2:
        raise ValueError(f"vector length {v.shape[-1]} is not C(C+1)/2 for C={C}")
    i, j = _triu_idx(C, v.device)
    w = torch.where(i == j, torch.ones((), device=v.device, dtype=v.dtype),
                    torch.full((), 1.0 / math.sqrt(2.0), device=v.device, dtype=v.dtype))
    vals = v * w
    M = v.new_zeros(v.shape[:-1] + (C, C))
    M[..., i, j] = vals
    M[..., j, i] = vals
    return M


def tangent_dim(C: int) -> int:
    return C * (C + 1) // 2


def log_euclid_vec(x: torch.Tensor, rho: float) -> torch.Tensor:
    """(B, C, T) -> (B, C(C+1)/2): log-Euclidean tangent vector of the shrunk covariance."""
    return triu_vec(sym_log(shrunk_cov(x, rho)))


def block_log_vec(blocks, rho: float) -> torch.Tensor:
    """List of (B, C_b, T) channel blocks (e.g. [EMG, IMU]) -> concatenated per-block tangent vectors."""
    return torch.cat([log_euclid_vec(b.double(), rho) for b in blocks], dim=-1)


def support_whitener(xs: torch.Tensor, rho: float) -> torch.Tensor:
    """Support set (N, C, T) -> (C, C) R^{-1/2}, R = arithmetic mean of the shrunk support covariances
    (Euclidean alignment, He & Wu 2020). Data-only: call under no_grad."""
    R = shrunk_cov(xs.double(), rho).mean(dim=0, keepdim=True)
    return sym_invsqrt(R)[0]


class CovPool(nn.Module):
    """Second-order pooling of a feature sequence (B, T, H) -> (B, p(p+1)/2).

    A linear map H -> p (the network learns WHAT to take the covariance of), the shrunk covariance over
    time, then a matrix normalisation: "sqrt" (Newton-Schulz, iSQRT-COV) or "log" (eigh, log-Euclidean),
    and the upper-triangle vector. MPN-COV / iSQRT-COV (Li et al. ICCV 2017, CVPR 2018), CovaMNet
    (Li et al. AAAI 2019) for the few-shot use."""

    NORMS = ("sqrt", "log")

    def __init__(self, in_dim: int, p: int, norm: str = "sqrt", rho: float = 0.1, ns_iters: int = 5):
        super().__init__()
        if norm not in self.NORMS:
            raise ValueError(f"cov_pool_norm must be one of {self.NORMS}, got {norm!r}")
        if p < 2:
            raise ValueError(f"cov_pool_dim must be >= 2, got {p}")
        self.proj = nn.Linear(in_dim, p, bias=False)       # a bias would be removed by the centring
        self.norm, self.rho, self.ns_iters, self.p = norm, float(rho), int(ns_iters), int(p)
        self.out_dim = tangent_dim(p)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.proj(h).transpose(1, 2)                         # (B, p, T)
        S = shrunk_cov(z, self.rho)
        M = sqrtm_ns(S, self.ns_iters) if self.norm == "sqrt" else sym_log(S)
        return triu_vec(M)


def feature_pool_dim(config: dict, H: int) -> int:
    """Feature size after the model's pooling: H for "mean", p(p+1)/2 for "cov"."""
    mode = config.get("feature_pool", "mean")
    if mode == "mean":
        return H
    if mode == "cov":
        return tangent_dim(int(config["cov_pool_dim"]))
    raise ValueError(f"feature_pool must be 'mean' or 'cov', got {mode!r}")


def build_feature_pool(config: dict, H: int):
    """None for "mean" (the models keep their own mean over time), a CovPool for "cov"."""
    mode = config.get("feature_pool", "mean")
    if mode == "mean":
        return None
    if mode == "cov":
        return CovPool(H, int(config["cov_pool_dim"]), norm=config["cov_pool_norm"],
                       rho=float(config["cov_shrinkage"]))
    raise ValueError(f"feature_pool must be 'mean' or 'cov', got {mode!r}")


# ── Euclidean alignment of the inputs by the episode's support set ──────────────────────────
INPUT_ALIGN_MODES = ("none", "ea")


def check_input_align(config: dict) -> str:
    mode = config.get("input_align", "none")
    if mode not in INPUT_ALIGN_MODES:
        raise ValueError(f"input_align must be one of {INPUT_ALIGN_MODES}, got {mode!r}")
    if mode == "ea" and config.get("emg_stem") not in (None, "none"):
        raise ValueError("input_align='ea' works on the windowed inputs; it is not defined for the raw EMG stem")
    return mode


def ea_align(model, x_emg, x_imu):
    """Whiten EMG and IMU (separately) by the episode support set's mean covariance, R^{-1/2} x.
    The support comes from mamlpp.task_support (model._task_support). Raises if it is missing."""
    sup = model._task_support
    if sup is None:
        raise RuntimeError("input_align='ea' needs the episode's support set: wrap the forward in "
                           "mamlpp.task_support(model, support_batch, device)")
    s_emg, s_imu = sup
    rho = float(model.config["cov_shrinkage"])
    with torch.no_grad():
        We = support_whitener(s_emg, rho).to(x_emg.dtype)
        Wi = support_whitener(s_imu, rho).to(x_imu.dtype) if (x_imu is not None and s_imu is not None) else None
    x_emg = torch.einsum("ij,bjt->bit", We, x_emg)
    if Wi is not None:
        x_imu = torch.einsum("ij,bjt->bit", Wi, x_imu)
    return x_emg, x_imu


# ── SPDNet (Huang & Van Gool, AAAI 2017) as a metric-learner backbone (P1_spd) ──────────────
class BiMap(nn.Module):
    """S -> W^T S W with W (C_in, C_out) column-orthonormal (QR of an unconstrained matrix)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        if c_out > c_in:
            raise ValueError(f"BiMap must reduce the dimension: {c_in} -> {c_out}")
        A = torch.empty(c_in, c_out)
        nn.init.orthogonal_(A)
        self.A = nn.Parameter(A)

    def forward(self, S):
        W, _ = torch.linalg.qr(self.A)
        return W.transpose(0, 1) @ S @ W


class ReEig(nn.Module):
    """Eigenvalue rectification max(lambda, eps): the SPDNet non-linearity."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = float(eps)

    def forward(self, S):
        return sym_fn(S, lambda w: w.clamp_min(self.eps))


class SPDNetEncoder(nn.Module):
    """Per modality: shrunk covariance -> [BiMap -> ReEig] x len(spd_dims) -> LogEig -> triu vector;
    modalities concatenated. Same backbone() contract as the CNN-LSTM encoders."""

    def __init__(self, config: dict):
        super().__init__()
        dims = [int(d) for d in config["spd_dims"]]
        if not dims:
            raise ValueError("spd_dims must list at least one BiMap output size")
        self.rho = float(config["cov_shrinkage"])
        self.use_imu = bool(config["use_imu"])
        chans = [config["emg_in_ch"]] + ([config["imu_in_ch"]] if self.use_imu else [])
        self.towers = nn.ModuleList()
        for C in chans:
            layers, c = [], C
            for d in dims:
                d = min(d, c)
                layers += [BiMap(c, d), ReEig(config["spd_reeig_eps"])]
                c = d
            self.towers.append(nn.Sequential(*layers))
        self.feat_dim = 0
        for C in chans:
            c = C
            for d in dims:
                c = min(d, c)
            self.feat_dim += tangent_dim(c)
        self.needs_task_support = False

    def backbone(self, x_emg, x_imu=None, demographics=None):
        xs = [x_emg] + ([x_imu] if (self.use_imu and x_imu is not None) else [])
        if len(xs) != len(self.towers):
            raise ValueError(f"SPDNetEncoder built for {len(self.towers)} modalities, got {len(xs)}")
        feats = [triu_vec(sym_log(tower(shrunk_cov(x.float(), self.rho)))) for tower, x in zip(self.towers, xs)]
        f = torch.cat(feats, dim=-1)
        return f, [f]
