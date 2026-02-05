"""
Multimodal MoE for EMG + IMU (+ optional demographics conditioning)
-------------------------------------------------------------------

Design goals
- Separate encoders per modality (EMG, IMU) with late fusion.
- Drop-in compatible with your existing training loop & WithUserOverride wrapper:
  forward(x_emg, x_imu=None, user_ids=None, user_embed_override=None, demographics=None, return_aux=False)
- Gate can use features only, user only, user-aware, or FiLM variants (code provided).
- Demographics conditioning module supports FiLM and concat; runtime default = concat.
- Experts remain simple MLP (works well with CosineHead). Hooks included where to add LSTM/attention if desired.
- Missing-modality robustness via ModDrop (randomly dropping IMU at train time).

Shapes
- x_emg: (B, C_emg, T_emg)  e.g., C_emg=16, T_emg≈200 (or whatever you use)
- x_imu: (B, C_imu, T_imu)  e.g., C_imu=6,  T_imu≈200; resample/align upstream
- demographics: (B, D_demo) already one-hot/normalized; if categorical raw IDs are used, embed upstream or extend DemographicsEncoder

Where to add attention (recommended next steps)
- Cross-modal attention: after encoders, before fusion. See TODO tags: [ATTN-CROSS].
- Temporal self-attention inside encoders: replace final TCN block with a small Transformer block. See TODO tags: [ATTN-TEMP].
- Expert temporal head (LSTM/Transformer) for sequence-level classification if you keep sequences. See TODO tags: [TEMP-HEAD].

SSL hooks (not implemented here)
- Add projection heads (MLP) to EMGEncoder/IMUEncoder and expose h_seq for contrastive/self-supervised losses (TS2Vec/CPC/BYOL-TS).

"""
from typing import Optional, Tuple

#from collections import defaultdict
#import random
import pandas as pd
#import pickle
#from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset#, IterableDataset

# -------------------- Utility Blocks --------------------

class TemporalProject1x1(nn.Module):
    """
    Per-time-step channel fusion. Concatenate per-modality sequences along channels,
    then 1x1 conv back to emb_dim. Keeps time length T' intact.
    Input:  sequences like (B, D_i, T)
    Output: (B, D, T)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
    def forward(self, *seqs):
        z = torch.cat(seqs, dim=1)     # (B, sum D_i, T)
        z = self.proj(z)               # (B, out_ch, T)
        return self.norm(z)


class LSTMEncoder(nn.Module):
    """
    LSTM over fused per-step features.
    Input:  z_seq (B, D, T)
    Output: (seq_out: (B, T, H), h_pool: (B, H))
    """
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1,
                 bidirectional: bool = False, pool_mode: str = "last", pdrop: float = 0.0):
        super().__init__()
        assert pool_mode in ("last", "mean")
        self.rnn = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        self.drop = nn.Dropout(pdrop)
        self.out_dim = hidden * (2 if bidirectional else 1)
        self.pool_mode = pool_mode

    def forward(self, z_seq_bdt: torch.Tensor):
        x = z_seq_bdt.transpose(1, 2)  # (B, T, D)
        y, _ = self.rnn(x)             # (B, T, H)
        y = self.drop(y)
        h_pool = y[:, -1, :] if self.pool_mode == "last" else y.mean(dim=1)
        return y, h_pool
    

class ConvBlock1D(nn.Module):
    """Depthwise-separable style 1D conv block with residual, LN, GELU.
    Input: (B, C, T) -> Output: (B, D, T') (stride may downsample)
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, stride: int = 1, dilation: int = 1, pdrop: float = 0.0):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, dilation=dilation, groups=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(pdrop)
        self.res_match = None
        if in_ch != out_ch or stride != 1:
            self.res_match = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        if self.res_match is not None:
            x = self.res_match(x)
        return y + x

class TemporalPool(nn.Module):
    """Temporal pooling to produce a fixed-size embedding from a sequence. Options: 'avg', 'max', 'avgmax'."""
    def __init__(self, mode: str = 'avg'):
        super().__init__()
        #print(f"TemporalPool: pool mode: {mode}, type(mode): {type(mode)}")
        assert mode in ['avg', 'max', 'avgmax']
        self.mode = mode
    def forward(self, x):  # x: (B, D, T)
        if self.mode == 'avg':
            return x.mean(dim=-1)
        elif self.mode == 'max':
            return x.max(dim=-1).values
        else:  # avgmax
            return torch.cat([x.mean(dim=-1), x.max(dim=-1).values], dim=-1)

# -------------------- Heads --------------------

class CosineHead(nn.Module):
    def __init__(self, emb_dim, num_classes, init_tau=10.0, learnable_tau=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.02)
        tau = torch.tensor(float(init_tau))
        if learnable_tau:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)
    def forward(self, h):              # h: (B, D)
        h = F.normalize(h, dim=-1)
        W = F.normalize(self.W, dim=-1)
        return self.tau * (h @ W.t())  # (B, C)

class Expert(nn.Module):
    """Simple MLP expert. Optionally swap fc2 with CosineHead externally via helper.
    NOTE: If you decide to use a temporal head later, see [TEMP-HEAD] markers below.
    """
    def __init__(self, emb_dim=64, num_classes=10, pdrop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, num_classes)
        self.drop = nn.Dropout(pdrop)
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, h):  # h: (B, D)
        z = F.gelu(self.fc1(h))
        z = self.norm(z)
        z = self.drop(z)
        return self.fc2(z)

# -------------------- Gates --------------------

class UserAwareGate(nn.Module):
    def __init__(self, emb_dim=64, user_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.lin = nn.Linear(emb_dim + user_dim, num_experts)
    def forward(self, h, u):  # h:(B,D) u:(B,U)
        g = self.lin(torch.cat([h, u], dim=-1))  # (B,E)
        w = F.softmax(g, dim=-1)
        E = w.size(-1)
        k = self.top_k
        # Only sparsify when 1 <= k < E
        if (k is not None) and (k > 0) and (k < E):
            topk = torch.topk(w, self.top_k, dim=-1)
            mask = torch.zeros_like(w).scatter(-1, topk.indices, 1.0)
            w = (w * mask)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        # else: do dense gating and just return w
        return w

class FeatureOnlyGate(nn.Module):
    def __init__(self, emb_dim=64, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.lin = nn.Linear(emb_dim, num_experts)
    def forward(self, h, u=None):
        g = self.lin(h)
        w = F.softmax(g, dim=-1)
        E = w.size(-1)
        k = self.top_k
        # Only sparsify when 1 <= k < E
        if (k is not None) and (k > 0) and (k < E):
            idx = torch.topk(w, self.top_k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, idx, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-9)
        # else: do dense gating and just return w
        return w

class UserOnlyGate(nn.Module):
    def __init__(self, user_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.lin = nn.Linear(user_dim, num_experts)
    def forward(self, h, u):
        w = F.softmax(self.lin(u), dim=-1)
        E = w.size(-1)
        k = self.top_k
        # Only sparsify when 1 <= k < E
        if (k is not None) and (k > 0) and (k < E):
            idx = torch.topk(w, self.top_k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, idx, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-9)
        # else: do dense gating and just return w
        return w

class FiLMGate(nn.Module):
    def __init__(self, emb_dim=64, user_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gamma = nn.Linear(user_dim, emb_dim)
        self.beta = nn.Linear(user_dim, emb_dim)
        self.lin = nn.Linear(emb_dim, num_experts)
    def forward(self, h, u):
        h_t = h * (1 + self.gamma(u)) + self.beta(u)  # affine mod
        w = F.softmax(self.lin(h_t), dim=-1)
        E = w.size(-1)
        k = self.top_k
        # Only sparsify when 1 <= k < E
        if (k is not None) and (k > 0) and (k < E):
            idx = torch.topk(w, self.top_k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, idx, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-9)
        # else: do dense gating and just return w
        return w

# -------------------- Modality Encoders --------------------

class EMGEncoderTCN(nn.Module):
    """Compact temporal CNN (TCN-like) for EMG.
    Exposes both sequence features (h_seq) and pooled embedding (h).
    [ATTN-TEMP] Replace last block with a TransformerEncoder for temporal attention.
    """
    def __init__(self, config):
        super().__init__()
        in_ch = config['emg_in_ch']
        emb_dim = config['emb_dim']
        base_ch = emb_dim
        base_ch_scaling = config['emg_CNN_capacity_scaling']
        pdrop = config["pdrop"]
        pool_mode = config["pool_mode"]

        self.block1 = ConvBlock1D(in_ch, base_ch, k=7, stride=2, pdrop=pdrop)
        self.block2 = ConvBlock1D(base_ch, base_ch_scaling*base_ch, k=5, stride=config['emg_stride2'], pdrop=pdrop)
        self.block3 = ConvBlock1D(base_ch_scaling*base_ch, emb_dim, k=3, stride=1, pdrop=pdrop)
        self.temporal_norm = nn.GroupNorm(num_groups=config["groupnorm_num_groups"], num_channels=emb_dim)
        self.pool = TemporalPool(mode=pool_mode)
        self.out_dim = emb_dim if pool_mode != 'avgmax' else emb_dim * 2
    def forward(self, x):  # (B, C_emg, T)
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.temporal_norm(z)
        h = self.pool(z)  # (B, D)
        return z, h  # z: (B, D, T'), h: (B, D)

class IMUEncoderTCN(nn.Module):
    """Compact temporal CNN for IMU (accel/gyro).
    Mirrors EMGEncoderTCN to keep representation scales similar.
    """
    def __init__(self, config):
        super().__init__()
        in_ch = config['imu_in_ch']
        emb_dim = config['emb_dim']
        base_ch = emb_dim
        base_ch_scaling = config['imu_CNN_capacity_scaling']
        pdrop = config["pdrop"]
        pool_mode = config["pool_mode"]

        self.block1 = ConvBlock1D(in_ch, base_ch, k=7, stride=2, pdrop=pdrop)
        self.block2 = ConvBlock1D(base_ch, base_ch_scaling*base_ch, k=5, stride=config['imu_stride2'], pdrop=pdrop)
        self.block3 = ConvBlock1D(base_ch_scaling*base_ch, emb_dim, k=3, stride=1, pdrop=pdrop)
        self.temporal_norm = nn.GroupNorm(num_groups=config["groupnorm_num_groups"], num_channels=emb_dim)
        self.pool = TemporalPool(mode=pool_mode)
        self.out_dim = emb_dim if pool_mode != 'avgmax' else emb_dim * 2
    def forward(self, x):  # (B, C_imu, T)
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.temporal_norm(z)
        h = self.pool(z)
        return z, h

# -------------------- Fusion & Conditioning --------------------

class LateFusion(nn.Module):
    """Concatenate pooled modality embeddings then project to common dim.
    [ATTN-CROSS] If you add cross-attention, do it before this projector.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
    def forward(self, *embs):
        h = torch.cat(embs, dim=-1)
        return self.proj(h)

class DemographicsEncoder(nn.Module):
    """Tiny MLP to embed demographics (already numeric). If you have categorical IDs,
    you can add Embedding layers upstream. Output dim = demo_emb_dim.
    """
    def __init__(self, in_dim: int, demo_emb_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, max(16, demo_emb_dim)),
            nn.GELU(),
            nn.Linear(max(16, demo_emb_dim), demo_emb_dim),
        )
        self.out_dim = demo_emb_dim
    def forward(self, d):
        return self.net(d)

class FiLMConditioner(nn.Module):
    """Feature-wise linear modulation from a conditioning vector (e.g., demographics).
    We keep this available, but runtime default will use simple concatenation in the gate.
    """
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)
    def forward(self, h, c):
        return h * (1 + self.gamma(c)) + self.beta(c)

# -------------------- MoE Classifier (Multimodal) --------------------

def make_gate(config, emb_dim_override: Optional[int] = None):
    emb_dim = emb_dim_override if emb_dim_override is not None else config["emb_dim"]
    user_dim = config["user_emb_dim"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]
    gtype = config["gate_type"]

    if gtype == "feature_only":
        return FeatureOnlyGate(emb_dim=emb_dim, num_experts=num_experts, top_k=top_k)
    elif gtype == "user_only":
        return UserOnlyGate(user_dim=user_dim, num_experts=num_experts, top_k=top_k)
    elif gtype == "film_gate":
        return FiLMGate(emb_dim=emb_dim, user_dim=user_dim, num_experts=num_experts, top_k=top_k)
    elif gtype == "user_aware":
        return UserAwareGate(emb_dim=emb_dim, user_dim=user_dim, num_experts=num_experts, top_k=top_k)
    else:
        raise ValueError("Unrecognized gate type!")


class MultiModalMoEClassifier(nn.Module):
    """Multimodal MoE with separate encoders and late fusion.

    - notes: [TEMP-HEAD] where to place an LSTM/Transformer head if you decide to keep sequences end-to-end.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mix_demo_u_alpha = config["mix_demo_u_alpha"]
        D = config["emb_dim"]
        pdrop = config["pdrop"]

        # ---------- Encoders (unchanged) ----------
        self.emg_enc = EMGEncoderTCN(config)
        self.imu_enc = IMUEncoderTCN(config)

        # Late fusion for pooled vectors (used by TCN-only path)
        fused_in_dim = self.emg_enc.out_dim + self.imu_enc.out_dim
        self.fusion = LateFusion(in_dim=fused_in_dim, out_dim=D)

        # ---------- Temporal backbone toggle ----------
        self.temporal_backbone = config["temporal_backbone"]
        if self.temporal_backbone == "lstm":
            # Per-step fuser: (B, D_e, T') + (B, D_i, T') -> (B, D, T')
            #in_ch = self.emg_enc.out_dim + self.imu_enc.out_dim 
            # ^ Was initialized with the pooled dims (self.emg_enc.out_dim + self.imu_enc.out_dim), but it actually receives sequence maps
            # Each encoder's *sequence* output channel = emb_dim (not affected by pool_mode)
            seq_ch_emg = config["emb_dim"]
            seq_ch_imu = config["emb_dim"]
            in_ch = seq_ch_emg + seq_ch_imu
            self.temporal_fuser = TemporalProject1x1(in_ch=in_ch, out_ch=D)

            # LSTM over fused per-step features
            self.lstm_enc = LSTMEncoder(
                in_dim=D,
                hidden=config["lstm_hidden"],
                num_layers=config["lstm_layers"],
                bidirectional=config["lstm_bidirectional"],
                pool_mode=config["temporal_pool_mode"],
                pdrop=pdrop,
            )
            backbone_out_dim = self.lstm_enc.out_dim
        else:
            self.temporal_fuser = None
            self.lstm_enc = None
            backbone_out_dim = D

        # ---------- Demographics encoder & conditioning (unchanged) ----------
        self.demo_encoder = None
        self.demo_conditioning = config["demo_conditioning"]  # 'concat' | 'film'
        demo_in_dim = config["demo_in_dim"]
        demo_emb_dim = config["demo_emb_dim"]
        if demo_in_dim is not None:
            self.demo_encoder = DemographicsEncoder(demo_in_dim, demo_emb_dim)
        self.film_cond = FiLMConditioner(feat_dim=backbone_out_dim if self.temporal_backbone=="lstm" else D,
                                        cond_dim=demo_emb_dim) if demo_in_dim is not None else None
        self.demo_concat_proj = None
        if demo_in_dim is not None and self.demo_conditioning == "concat":
            if self.temporal_backbone.upper() == "LSTM":
                in_dim  = backbone_out_dim + demo_emb_dim
                out_dim = backbone_out_dim
            else:
                in_dim  = D + demo_emb_dim
                out_dim = D
            self.demo_concat_proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
            )

        # ---------- Optional widening before experts ----------
        self.pre_expert_proj = None
        if config["expert_bigger"]:
            mult = int(config["expert_bigger_mult"])
            widened = backbone_out_dim * mult
            self.pre_expert_proj = nn.Sequential(
                nn.LayerNorm(backbone_out_dim),
                nn.Linear(backbone_out_dim, widened),
                nn.GELU(),
                nn.LayerNorm(widened),
            )
            expert_in_dim = widened
        else:
            expert_in_dim = backbone_out_dim

        # ---------- Experts (MoE head) ----------
        num_experts = config["num_experts"]
        self.experts = nn.ModuleList([
            Expert(emb_dim=expert_in_dim, num_classes=config["num_classes"], pdrop=pdrop)
            for _ in range(num_experts)
        ])

        # ---------- Gate ----------
        # after temporal_backbone branch
        backbone_out_dim = self.lstm_enc.out_dim if self.temporal_backbone == "lstm" else D
        # the gate should use the PRE-WIDENED dimension (before pre_expert_proj)
        self.gate_in_dim = backbone_out_dim
        # build the gate with the correct emb_dim
        self.gate = make_gate(config, emb_dim_override=self.gate_in_dim)

        # ---------- User embeddings ----------
        self.use_user_table = config["use_user_table"]
        self.user_emb_dim = config["user_emb_dim"]
        if self.use_user_table:
            self.user_table = nn.Embedding(config["num_total_users"], self.user_emb_dim)
        else:
            self.user_table = None

        ###################################################################################
        # TODO: Another spot where MAML might be using u... figure this out
        if self.demo_encoder is not None:
            self.demo_to_u = nn.Sequential(
                nn.LayerNorm(demo_emb_dim),
                nn.Linear(demo_emb_dim, self.user_emb_dim),
            )
        else:
            self.demo_to_u = None
        ###################################################################################

        # ---------- Misc ----------
        self.expert_keys = nn.Parameter(torch.randn(num_experts, D) * 0.1)
        self.mixture_mode = config["mixture_mode"]
        self.moddrop_p = config["moddrop_p"]

        # Optional: keep a backbone list (if you use it elsewhere)
        bb = nn.ModuleList([self.emg_enc, self.imu_enc, self.fusion])
        if self.temporal_fuser is not None: bb.append(self.temporal_fuser)
        if self.lstm_enc is not None: bb.append(self.lstm_enc)
        if self.demo_encoder is not None: bb.append(self.demo_encoder)
        if self.demo_conditioning == "film" and self.film_cond is not None: bb.append(self.film_cond)
        elif self.demo_conditioning == "concat" and self.demo_concat_proj is not None: bb.append(self.demo_concat_proj)
        if self.pre_expert_proj is not None: bb.append(self.pre_expert_proj)
        self.backbone = bb


    # -------------------- Forward --------------------
    def forward(
        self,
        x_emg: torch.Tensor,
        x_imu: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None,
        user_embed_override: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        B = x_emg.size(0)
        d_emb = None  # will be set if demographics provided

        # ---------- ModDrop ----------
        if self.training and (self.moddrop_p > 0) and (x_imu is not None):
            drop_mask = (torch.rand(B, device=x_emg.device) < self.moddrop_p).float().view(B, 1, 1)
            x_imu = x_imu * (1.0 - drop_mask)

        # ---------- Encoders ----------
        emg_seq, emg_h = self.emg_enc(x_emg)     # emg_seq: (B, D_e, T'), emg_h: (B, D_e or 2D_e)
        if x_imu is not None:
            imu_seq, imu_h = self.imu_enc(x_imu) # imu_seq: (B, D_i, T'), imu_h: (B, D_i or 2D_i)
        else:
            imu_seq = torch.zeros(B, self.imu_enc.out_dim, emg_seq.size(-1), device=x_emg.device)
            imu_h   = torch.zeros(B, self.imu_enc.out_dim, device=x_emg.device)

        # ---------- Branch: TCN-only vs LSTM path ----------
        if self.temporal_backbone.upper() == "NONE":
            # Pooled late fusion (your current path)
            fused_h = self.fusion(emg_h, imu_h)  # (B, D)

            # Demographics conditioning (vector)
            if self.demo_encoder is not None and demographics is not None:
                d_emb = self.demo_encoder(demographics)  # (B, demo_emb_dim)
                if self.demo_conditioning == "film" and self.film_cond is not None:
                    fused_h = self.film_cond(fused_h, d_emb)
                elif self.demo_conditioning == "concat" and self.demo_concat_proj is not None:
                    fused_h = self.demo_concat_proj(torch.cat([fused_h, d_emb], dim=-1))

            core_embed = fused_h  # (B, D)
        elif self.temporal_backbone.upper() == "LSTM":
            # Per-step fusion → LSTM → pooled state
            z_seq = self.temporal_fuser(emg_seq, imu_seq)   # (B, D, T')
            _, h_pool = self.lstm_enc(z_seq)                # (B, H)

            # Demographics conditioning (vector)
            if self.demo_encoder is not None and demographics is not None:
                d_emb = self.demo_encoder(demographics)     # (B, demo_emb_dim)
                if self.demo_conditioning == "film" and self.film_cond is not None:
                    h_pool = self.film_cond(h_pool, d_emb)
                elif self.demo_conditioning == "concat" and self.demo_concat_proj is not None:
                    h_pool = self.demo_concat_proj(torch.cat([h_pool, d_emb], dim=-1))

            core_embed = h_pool  # (B, backbone_out_dim)
        else:
            raise ValueError("temporal_backbone {self.temporal_backbone} not recognized! Needs to be either NONE or LSTM")

        # ---------- Optional widening before experts ----------
        if self.pre_expert_proj is not None:
            core_embed = self.pre_expert_proj(core_embed)   # (B, expert_in_dim)

        ################################################################################################
        # TODO: Does MAML use u???
        # ---------- Build user embedding u ----------
        if self.config['u_user_and_demos'] == 'demo':
            if d_emb is None:
                raise ValueError("Demographics required but not provided for u_user_and_demos='demo'.")
            u = self.demo_to_u(d_emb)
        elif self.config['u_user_and_demos'] == 'mix':
            if d_emb is None:
                raise ValueError("Demographics required but not provided for u_user_and_demos='mix'.")
            if user_embed_override is not None:
                u_from_table = user_embed_override
            elif (self.user_table is not None) and (user_ids is not None):
                u_from_table = self.user_table(user_ids)
            else:
                raise ValueError("u_user_and_demos='mix' but user embedding missing.")
            u = u_from_table + self.mix_demo_u_alpha * self.demo_to_u(d_emb)
        elif self.config['u_user_and_demos'] == 'u_user' and user_embed_override is not None:
            u = user_embed_override
        elif self.config['u_user_and_demos'] == 'u_user' and (self.user_table is not None) and (user_ids is not None):
            u = self.user_table(user_ids)
        else:
            raise ValueError("User embedding missing (check u_user_and_demos & inputs).")
        ################################################################################################

        # ---------- Gate + Experts ----------
        # (A) Gate over the pre-widened core embedding
        core_for_gate = core_embed  # (B, self.gate_in_dim)
        w = self.gate(core_for_gate, u)  # (B, E)
        # (B) Optional widening is for the experts ONLY
        core_for_experts = self.pre_expert_proj(core_embed) if self.pre_expert_proj is not None else core_embed
        # (C) Experts consume core_for_experts
        logits_per_exp = torch.stack([exp(core_for_experts) for exp in self.experts], dim=1)  # (B, E, C)
        #logits_per_exp = torch.stack([exp(core_embed) for exp in self.experts], dim=1)  # (B, E, C)

        # ---------- Mixture ----------
        if self.mixture_mode == "probs":
            probs_per_exp = torch.softmax(logits_per_exp, dim=-1)
            out = (w.unsqueeze(-1) * probs_per_exp).sum(dim=1)
        elif self.mixture_mode == "logprobs":
            log_probs_per_exp = torch.log_softmax(logits_per_exp, dim=-1)
            log_w = torch.log(w.clamp_min(1e-9)).unsqueeze(-1)
            out = torch.logsumexp(log_w + log_probs_per_exp, dim=1)
        else:  # 'logits'
            out = (w.unsqueeze(-1) * logits_per_exp).sum(dim=1)

        aux = {}
        if return_aux:
            aux["gate_usage"] = w.mean(dim=0)      # (E,)
            aux["fused_h"] = core_embed.detach()   # post-conditioning, pre-MoE rep

        return out, aux

    def swap_expert_head_to_cosine(self, init_tau=10.0, learnable_tau=True):
        # self.experts is a ModuleList of Expert blocks
        for i, exp in enumerate(self.experts):
            in_dim = exp.fc1.out_features  # <-- ground truth width
            exp.fc2 = CosineHead(emb_dim=in_dim,
                                num_classes=self.config["num_classes"],
                                init_tau=init_tau,
                                learnable_tau=learnable_tau)
            
    # Added this for Simple Clustering approach but never pursued it further
    ## Can probably remove this, this is only relevant to old clustering approaches
    @torch.no_grad()
    def encode_batch(self, batch, return_parts: bool = False):
        """
        Extracts the penultimate embedding used by the gate (i.e., post-fusion and
        optional demographics conditioning, but pre-expert / pre-gate widening).
        Returns: core_embed  (B, D_gate) where D_gate == self.gate_in_dim.
        If return_parts=True, also returns a dict of intermediate tensors for debugging.
        """
        # --- Pull inputs from batch with flexible key names ---
        def _get_any(d, *keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None

        x_emg = _get_any(batch, "x_emg", "emg", "x")
        x_imu = _get_any(batch, "x_imu", "imu")
        demographics = _get_any(batch, "demographics", "demo", "d")

        if x_emg is None:
            raise ValueError("encode_batch: expected EMG tensor under keys ['x_emg','emg','x'].")

        device = next(self.parameters()).device
        x_emg = x_emg.to(device)
        B = x_emg.size(0)

        # ---------- Encoders ----------
        emg_seq, emg_h = self.emg_enc(x_emg)  # emg_h: (B, D_e or 2D_e) depending on pool_mode

        if x_imu is not None:
            x_imu = x_imu.to(device)
            imu_seq, imu_h = self.imu_enc(x_imu)
        else:
            # keep shapes consistent if IMU missing
            imu_seq = None
            imu_h = torch.zeros(B, self.imu_enc.out_dim, device=device)

        d_emb = None
        parts = {}

        # ---------- Temporal backbone branch ----------
        if str(self.temporal_backbone).upper() == "NONE":
            # pooled late fusion
            fused_h = self.fusion(emg_h, imu_h)  # (B, D)

            # demographics conditioning
            if (self.demo_encoder is not None) and (demographics is not None):
                d_emb = self.demo_encoder(demographics.to(device))  # (B, demo_emb_dim)
                if self.demo_conditioning == "film" and self.film_cond is not None:
                    fused_h = self.film_cond(fused_h, d_emb)
                elif self.demo_conditioning == "concat" and self.demo_concat_proj is not None:
                    fused_h = self.demo_concat_proj(torch.cat([fused_h, d_emb], dim=-1))

            core_embed = fused_h  # (B, D) == gate_in_dim in this branch

        elif str(self.temporal_backbone).upper() == "LSTM":
            # per-step fusion -> LSTM -> pooled state
            if imu_seq is None:
                # build a zero sequence with same T' as EMG path
                imu_seq = torch.zeros(B, self.imu_enc.out_dim, emg_seq.size(-1), device=device)
            z_seq = self.temporal_fuser(emg_seq, imu_seq)  # (B, D, T')
            _, h_pool = self.lstm_enc(z_seq)               # (B, backbone_out_dim)

            if (self.demo_encoder is not None) and (demographics is not None):
                d_emb = self.demo_encoder(demographics.to(device))
                if self.demo_conditioning == "film" and self.film_cond is not None:
                    h_pool = self.film_cond(h_pool, d_emb)
                elif self.demo_conditioning == "concat" and self.demo_concat_proj is not None:
                    h_pool = self.demo_concat_proj(torch.cat([h_pool, d_emb], dim=-1))

            core_embed = h_pool  # (B, backbone_out_dim) == gate_in_dim in this branch

        else:
            raise ValueError(f"encode_batch: unknown temporal_backbone={self.temporal_backbone}")

        # ---------- DO NOT apply pre_expert_proj (we want the gate input space) ----------
        # core_embed here matches self.gate_in_dim and is what you should cluster.

        if return_parts:
            parts.update({
                "emg_h": emg_h, "imu_h": imu_h,
                "d_emb": d_emb, "core_embed": core_embed
            })
            return core_embed, parts
        return core_embed


# -------------------- Notes on LSTMs / Attention --------------------
"""
Should experts be larger or add an MLP/LSTM after experts?
- Start simple: keep Expert as small MLP (as above). If you widen the encoder/fused dim (emb_dim), consider setting expert_bigger=True.
- If you want sequence-aware classification, expose emg_seq/imu_seq and add a [TEMP-HEAD] temporal head:
    class TemporalHead(nn.Module):
        def __init__(self, in_dim, hidden=128, num_layers=1, num_classes=10):
            super().__init__()
            self.rnn = nn.LSTM(in_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.cls = nn.Linear(hidden*2, num_classes)
        def forward(self, seq):  # (B, T, D)
            y, _ = self.rnn(seq)
            return self.cls(y.mean(dim=1))
- Where to hook it: replace `fused_h` with a fused sequence representation and feed to TemporalHead.

Should I investigate LSTMs?
- Only after you establish a clear win with EMG+IMU late fusion. LSTMs help when precise temporal order matters and windows are long. For short EMG windows (~300ms), TCNs often suffice. A tiny Transformer encoder layer (2 heads) at [ATTN-TEMP] is another strong option.

Where exactly to add attention?
- [ATTN-CROSS] insert a CrossAttention block consuming (emg_seq, imu_seq) and return updated sequences => pool => fused_h.
- [ATTN-TEMP] replace EMGEncoderTCN.block3 with TransformerEncoderLayer(d_model=emb_dim, nhead=2, dim_feedforward=2*emb_dim) on z^T.
"""

#########################################################################################################################

# -------------------- Data utilities (DataFrame -> Dataset -> DataLoader) --------------------

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---- TwoDFSequenceDataset ----------------------------------------------------
class TwoDFSequenceDataset(Dataset):
    """
    Unified dataset that can be backed by:
      (A) DataFrames (your original behavior), OR
      (B) Prebuilt tensors.

    Emits one sample dict with:
      emg:   (C_emg, T)  float32
      imu:   (C_imu, T)  float32 or None
      demo:  (D_demo,)   float32
      label: ()          int64   (class index)
      PIDs:  ()          int64   (numeric user index for embeddings)
    """
    # --------- Constructors for clarity (optional to use) ----------
    @classmethod
    def from_dataframes(cls, time_df, demo_df, **kwargs):
        return cls(time_df=time_df, demo_df=demo_df, **kwargs)

    @classmethod
    def from_tensors(cls, emg_t, labels_t, pids_t, *, imu_t=None, demo_t=None):
        # Note: window_len/emg_cols/etc. are irrelevant for tensor mode
        return cls(
            time_df=None,
            demo_df=None,
            emg_t=emg_t,
            imu_t=imu_t,
            demo_t=demo_t,
            labels_t=labels_t,
            pids_t=pids_t,
        )

    def __init__(
        self,
        # ---- DF mode (A) args, unchanged ----
        time_df: pd.DataFrame | None = None,
        demo_df: pd.DataFrame | None = None,
        *,
        window_len: int = 64,
        emg_cols=None,
        imu_cols=None,
        demo_cols=None,
        label_col: str = "Enc_Gesture_ID",
        user_id_col: str = "Enc_PID",

        # ---- Tensor mode (B) args (all tensors) ----
        emg_t: torch.Tensor | None = None,     # (N, C_emg, T)
        imu_t: torch.Tensor | None = None,     # (N, C_imu, T) or None
        demo_t: torch.Tensor | None = None,    # (N, D_demo) or None
        labels_t: torch.Tensor | None = None,  # (N,)
        pids_t: torch.Tensor | None = None,    # (N,)
    ):
        super().__init__()

        # Detect mode
        tensor_mode = emg_t is not None or labels_t is not None or pids_t is not None
        df_mode = (time_df is not None) and (demo_df is not None)

        if tensor_mode and df_mode:
            raise ValueError("Provide EITHER DataFrames (time_df & demo_df) OR tensor args (emg_t, labels_t, pids_t), not both.")
        if not tensor_mode and not df_mode:
            raise ValueError("You must provide DataFrames (DF mode) OR tensors (tensor mode).")

        self._mode = "tensor" if tensor_mode else "df"

        if self._mode == "df":
            # --------- Original DF-backed behavior (unchanged) ---------
            self.time_df = time_df.reset_index(drop=True)
            self.demo_df = demo_df.copy()

            if user_id_col not in self.demo_df.columns:
                raise KeyError(f"'{user_id_col}' must be a column in demo_df.")
            self.demo_df = self.demo_df.set_index(user_id_col, drop=False)

            self.window_len = int(window_len)
            self.emg_cols = list(emg_cols)
            self.imu_cols = list(imu_cols) if imu_cols is not None and len(imu_cols) > 0 else None
            self.demo_cols = list(demo_cols) if demo_cols is not None and len(demo_cols) > 0 else None
            self.label_col = label_col
            self.user_id_col = user_id_col

            unique_ids = pd.Index(self.demo_df.index.unique())
            self.pid_to_index = {pid: i for i, pid in enumerate(unique_ids)}
            self.index_to_pid = {i: pid for pid, i in self.pid_to_index.items()}

            n = len(self.time_df)
            if n < self.window_len:
                raise ValueError(f"time_df has {n} rows, smaller than window_len={self.window_len}.")
            self.n_full = (n // self.window_len) * self.window_len
            self.starts = np.arange(0, self.n_full, self.window_len, dtype=int)

        else:
            # --------- Tensor-backed behavior ---------
            # Basic presence & shape checks
            if emg_t is None or labels_t is None or pids_t is None:
                raise ValueError("Tensor mode requires emg_t, labels_t, and pids_t (imu_t/demo_t are optional).")

            if not torch.is_tensor(emg_t):    emg_t = torch.as_tensor(emg_t, dtype=torch.float32)
            if imu_t is not None and not torch.is_tensor(imu_t):   imu_t = torch.as_tensor(imu_t, dtype=torch.float32)
            if demo_t is not None and not torch.is_tensor(demo_t): demo_t = torch.as_tensor(demo_t, dtype=torch.float32)
            if not torch.is_tensor(labels_t): labels_t = torch.as_tensor(labels_t, dtype=torch.long)
            if not torch.is_tensor(pids_t):   pids_t   = torch.as_tensor(pids_t,   dtype=torch.long)

            if emg_t.ndim != 3:
                raise ValueError(f"emg_t must be (N,C,T); got {tuple(emg_t.shape)}")
            if imu_t is not None and imu_t.ndim != 3:
                raise ValueError(f"imu_t must be (N,C,T); got {tuple(imu_t.shape)}")
            if demo_t is not None and demo_t.ndim not in (1, 2):
                raise ValueError(f"demo_t must be (N,D) or (N,); got {tuple(demo_t.shape)}")

            if demo_t is not None and demo_t.ndim == 1:
                demo_t = demo_t.unsqueeze(1)  # (N,) -> (N,1)

            N = emg_t.shape[0]
            if labels_t.shape[0] != N: raise ValueError("labels_t length must match emg_t batch size")
            if pids_t.shape[0]   != N: raise ValueError("pids_t length must match emg_t batch size")
            if imu_t  is not None and imu_t.shape[0]  != N: raise ValueError("imu_t length must match emg_t batch size")
            if demo_t is not None and demo_t.shape[0] != N: raise ValueError("demo_t length must match emg_t batch size")

            # Store tensors
            self._emg_t    = emg_t.contiguous()
            self._imu_t    = None if imu_t is None else imu_t.contiguous()
            self._demo_t   = None if demo_t is None else demo_t.contiguous()
            self._labels_t = labels_t.contiguous()
            self._pids_t   = pids_t.contiguous()

    def __len__(self):
        if self._mode == "df":
            return len(self.starts)
        else:
            return self._emg_t.shape[0]

    def __getitem__(self, idx):
        if self._mode == "df":
            s = self.starts[idx]
            e = s + self.window_len
            block = self.time_df.iloc[s:e]

            # ---- EMG (C, T)
            emg_np = block[self.emg_cols].to_numpy(dtype=np.float32, copy=False)  # (T, C_emg)
            emg = torch.from_numpy(emg_np).T.contiguous()  # -> (C_emg, T)

            # ---- IMU (C, T) or None
            imu = None
            if self.imu_cols is not None:
                imu_np = block[self.imu_cols].to_numpy(dtype=np.float32, copy=False)  # (T, C_imu)
                imu = torch.from_numpy(imu_np).T.contiguous()  # -> (C_imu, T)

            # ---- Label (int64)
            label_val = block.iloc[0][self.label_col]
            if not np.issubdtype(np.asarray(label_val).dtype, np.integer):
                raise ValueError(
                    f"Label '{self.label_col}' must be integer-coded (got {label_val!r}). "
                    "Map classes to integers before using the dataset."
                )
            label = torch.tensor(int(label_val), dtype=torch.long)

            # ---- PID -> numeric index
            pid_val = block.iloc[0][self.user_id_col]
            if pid_val not in self.pid_to_index:
                raise KeyError(f"User {pid_val!r} not found in demo_df index.")
            PIDs = torch.tensor(self.pid_to_index[pid_val], dtype=torch.long)

            # ---- Demo (D,)
            demo_row = self.demo_df.loc[pid_val]
            if isinstance(demo_row, pd.DataFrame):
                demo_row = demo_row.iloc[0]  # if duplicates per PID, take the first
            if self.demo_cols is None:
                demo_vals = demo_row.drop(labels=[self.user_id_col], errors="ignore").to_numpy()
            else:
                demo_vals = demo_row[self.demo_cols].to_numpy()
            demo = torch.as_tensor(np.asarray(demo_vals, dtype=np.float32).reshape(-1), dtype=torch.float32)

            return {"emg": emg, "imu": imu, "demo": demo, "label": label, "PIDs": PIDs}

        else:
            # Tensor-backed path
            emg   = self._emg_t[idx]
            imu   = None if self._imu_t is None else self._imu_t[idx]
            demo  = None if self._demo_t is None else self._demo_t[idx]
            label = self._labels_t[idx]
            pids  = self._pids_t[idx]
            return {"emg": emg, "imu": imu, "demo": demo, "label": label, "PIDs": pids}


# ---- Collate (unimodal or multimodal) ---------------------------------------
def default_mm_collate_fixed(batch):
    """
    Stacks TwoDFSequenceDataset samples into model-ready tensors.

    Returns:
      emg:   (B, C_emg, T)
      imu:   (B, C_imu, T) or None
      demo:  (B, D_demo)
      label: (B,)          int64
      PIDs:  (B,)          int64
    """
    # EMG (C,T) -> (B,C,T)
    emg = torch.stack([b["emg"] for b in batch], dim=0)
    if emg.dim() != 3:
        raise ValueError(f"EMG must be 3D, got {tuple(emg.shape)}")

    # IMU optional
    imu = None
    if all(("imu" in b) and (b["imu"] is not None) for b in batch):
        imu = torch.stack([b["imu"] for b in batch], dim=0)
        if imu.dim() != 3:
            raise ValueError(f"IMU must be 3D, got {tuple(imu.shape)}")

    # Demo (B, D)
    demo = torch.stack([b["demo"] for b in batch], dim=0).float()

    # Labels / PIDs: robust to ints / numpy scalars / 0-D tensors
    label = torch.as_tensor([int(b["label"]) for b in batch], dtype=torch.long)
    PIDs  = torch.as_tensor([int(b["PIDs"])  for b in batch], dtype=torch.long)

    return {"emg": emg, "imu": imu, "demo": demo, "label": label, "PIDs": PIDs}


# ---- Dataloader builder ------------------------------------------------------
def build_dataloader_from_two_dfs(
    time_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    *,  # TODO: I hate that it uses *. Remove this if possible
    sample_keys=None,   # unused
    emg_cols=None,
    imu_cols=None,
    demo_cols=None,
    label_col="Enc_Gesture_ID",
    user_id_col="Enc_PID",
    window_len=64,
    batch_size=64, # TODO: I dont like that its using this with a default value, also idk what this is doing in terms of a meta-batch...
    shuffle=True,
    num_workers=0, # This ought to be pulled from the config...
    collate_fn=None,
):
    if emg_cols is None:
        raise ValueError("emg_cols must be provided (list of EMG column names).")
    if collate_fn is None:
        collate_fn = default_mm_collate_fixed

    ds = TwoDFSequenceDataset(
        time_df=time_df,
        demo_df=demo_df,
        window_len=window_len,
        emg_cols=emg_cols,
        imu_cols=imu_cols,
        demo_cols=demo_cols,
        label_col=label_col,
        user_id_col=user_id_col,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, dl


def ensure_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype) if dtype else x.clone().detach()
    return torch.tensor(x, dtype=dtype)


def _reshape_2d_to_3d_if_needed(x, channels, length):
    # x: (N, C*L) -> (N, C, L)
    if x.ndim == 2:
        return x.view(-1, channels, length)
    return x

# This does get used in mutlimodal_data_processing...
def make_MOE_tensor_dataset(*args, reshape_2d_to_3d=True, participant_ids=None):
    """
    Overloaded behavior:

    1) Legacy (single-modal):
       make_MOE_tensor_dataset(features, labels, config, reshape_2d_to_3d=True, participant_ids=None)
       -> returns TensorDataset(features, labels[, participant_ids])

    2) Multimodal:
       make_MOE_tensor_dataset(emg, imu, demo, labels, config, reshape_2d_to_3d=True, participant_ids=None)
       -> returns Dataset yielding dict with keys: emg, imu, demo, label, PIDs

    Config keys used:
      - Legacy:
          'num_channels', 'sequence_length'
      - Multimodal (if reshape needed or validation desired):
          Prefer 'emg_num_channels', 'emg_sequence_length'
          Prefer 'imu_num_channels', 'imu_sequence_length'
          Fallback to 'num_channels'/'sequence_length' if *_specific not provided.
    """
    if len(args) == 3:
        # ----- Legacy branch -----
        features, labels, config = args
        features = ensure_tensor(features, dtype=torch.float32)
        labels   = ensure_tensor(labels,   dtype=torch.long)

        if features.ndim == 2 and reshape_2d_to_3d:
            num_channels    = config["num_channels"]
            sequence_length = config["sequence_length"]
            if num_channels is None or sequence_length is None:
                raise ValueError("Legacy reshape requires config['num_channels'] and config['sequence_length'].")
            features = _reshape_2d_to_3d_if_needed(features, num_channels, sequence_length)

        assert features.ndim == 3, (
            f"Expected 3D tensor (N, C, L), got {features.ndim}D with shape {features.shape}"
        )
        if "num_channels" in config:
            assert features.shape[1] == config["num_channels"], "Channel mismatch vs config['num_channels']"
        if "sequence_length" in config:
            assert features.shape[2] == config["sequence_length"], "Length mismatch vs config['sequence_length']"

        if participant_ids is not None:
            # Accept strings like 'P10' and coerce
            if isinstance(participant_ids, (list, tuple)) and len(participant_ids) > 0 and isinstance(participant_ids[0], str):
                try:
                    participant_ids = [int(pid.lstrip('Pp')) for pid in participant_ids]
                except Exception as e:
                    raise ValueError(f"Failed to convert participant_ids to integers: {e}")

            participant_ids = ensure_tensor(participant_ids, dtype=torch.long)
            assert participant_ids.shape[0] == features.shape[0], (
                f"participant_ids length {participant_ids.shape[0]} does not match number of samples {features.shape[0]}"
            )
            return TensorDataset(features, labels, participant_ids)

        return TensorDataset(features, labels)

    elif len(args) == 5:
        # ----- Multimodal branch -----
        emg, imu, demo, labels, config = args

        # Tensors & dtypes
        emg    = ensure_tensor(emg,    dtype=torch.float32) if emg is not None else None
        imu    = ensure_tensor(imu,    dtype=torch.float32) if imu is not None else None
        demo   = ensure_tensor(demo,   dtype=torch.float32) if demo is not None else None
        labels = ensure_tensor(labels, dtype=torch.long)

        # Optional reshape for time-series (2D -> 3D)
        if emg is not None and emg.ndim == 2 and reshape_2d_to_3d:
            C = config.get("emg_in_ch", config["num_channels"])
            L = config.get("emg_sequence_length", config["sequence_length"])
            if C is None or L is None:
                raise ValueError("EMG reshape requires emg_num_channels/emg_sequence_length (or num_channels/sequence_length).")
            emg = _reshape_2d_to_3d_if_needed(emg, C, L)

        if imu is not None and imu.ndim == 2 and reshape_2d_to_3d:
            C = config.get("imu_in_ch", config["num_channels"])
            L = config.get("imu_sequence_length", config["sequence_length"])
            if C is None or L is None:
                raise ValueError("IMU reshape requires imu_num_channels/imu_sequence_length (or num_channels/sequence_length).")
            imu = _reshape_2d_to_3d_if_needed(imu, C, L)

        # Basic validation
        assert emg is not None, "EMG tensor cannot be None in multimodal branch."
        assert emg.ndim == 3, f"EMG must be 3D (N, C, L). Got {emg.ndim}D with shape {emg.shape}"
        # TODO: This is super convoluted and needs to be cleaned up. I know what is in my config.
        if "emg_num_channels" in config:
            assert emg.shape[1] == config["emg_num_channels"], "EMG channel mismatch vs config['emg_num_channels']"
        elif "num_channels" in config:
            # fallback validation if global single-mod keys are used
            assert emg.shape[1] == config["num_channels"], "EMG channel mismatch vs config['num_channels']"
        if "emg_sequence_length" in config:
            assert emg.shape[2] == config["emg_sequence_length"], "EMG length mismatch vs config['emg_sequence_length']"
        elif "sequence_length" in config:
            assert emg.shape[2] == config["sequence_length"], "EMG length mismatch vs config['sequence_length']"
        #
        if imu is not None:
            assert imu.ndim == 3, f"IMU must be 3D (N, C, L). Got {imu.ndim}D with shape {imu.shape}"
            if "imu_num_channels" in config:
                assert imu.shape[1] == config["imu_num_channels"], "IMU channel mismatch vs config['imu_num_channels']"
            if "imu_sequence_length" in config:
                assert imu.shape[2] == config["imu_sequence_length"], "IMU length mismatch vs config['imu_sequence_length']"

        if demo is not None:
            # Allow (N, D) or (N,) -> promote to (N, D)
            if demo.ndim == 1:
                demo = demo.unsqueeze(1)
            assert demo.ndim == 2, f"Demo must be 2D (N, D). Got {demo.ndim}D with shape {demo.shape}"

        # Participant IDs
        if participant_ids is not None:
            if isinstance(participant_ids, (list, tuple)) and len(participant_ids) > 0 and isinstance(participant_ids[0], str):
                try:
                    participant_ids = [int(pid.lstrip('Pp')) for pid in participant_ids]
                except Exception as e:
                    raise ValueError(f"Failed to convert participant_ids to integers: {e}")
            pids_t = ensure_tensor(participant_ids, dtype=torch.long)
        else:
            # I dont think this branch is ever called? So this actually isnt a problem?
            raise ValueError("PIDs not provided. Raising an error, if we don't need PIDs then figure out what we should do here")
            # It sets the -1 placeholder. Should be setting it to PID
            ## I would have to pass PID in tho is all
            # Provide -1 placeholder so the key always exists
            pids_t = torch.full((emg.shape[0],), -1, dtype=torch.long)
            
        # Length checks for demo/imu vs emg will be enforced in dataset ctor
        return _MultiModalTensorDataset(emg, imu, demo, labels, pids_t)

    else:
        raise TypeError(
            "make_MOE_tensor_dataset expected either 3 args (features, labels, config) "
            "or 5 args (emg, imu, demo, labels, config)."
        )


# TODO: This is used below. Would rather use my existing dataset class for everything... or just 1 of these... simplify things...
## Oh this is the private version of mine... ... is taht really necessary?
class _MultiModalTensorDataset(Dataset):
    """
    Returns dict samples:
      {
        'emg':  (C_emg, T) float32,
        'imu':  (C_imu, T) float32 or None,
        'demo': (D_demo,)  float32 or None,
        'label': ()        int64,
        'PIDs':  ()        int64  (=-1 if not provided)
      }
    """
    def __init__(self, emg, imu, demo, labels, pids):
        self.emg   = emg
        self.imu   = imu
        self.demo  = demo
        self.label = labels
        self.pids  = pids

        n = self.emg.shape[0]
        assert self.label.shape[0] == n, "labels length must match emg batch size"
        if self.imu is not None:
            assert self.imu.shape[0] == n, "imu length must match emg batch size"
        if self.demo is not None:
            assert self.demo.shape[0] == n, "demo length must match emg batch size"
        assert self.pids.shape[0] == n, "PIDs length must match emg batch size"

    def __len__(self):
        return self.emg.shape[0]

    def __getitem__(self, idx):
        return {
            'emg':   self.emg[idx],
            'imu':   None if self.imu is None else self.imu[idx],
            'demo':  None if self.demo is None else self.demo[idx],
            'label': self.label[idx],
            'PIDs':  self.pids[idx],
        }