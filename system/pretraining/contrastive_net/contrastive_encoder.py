"""
contrastive_encoder.py

Encoder backbone + projection head for contrastive gesture recognition.

Two backbone modes (set via config['arch_mode']):
  - 'cnn_lstm' : CNN → BiLSTM → pool → MLP projection head
  - 'cnn_attn' : CNN → learned attention pool → MLP projection head

The final output is always L2-normalized embeddings in R^D,
suitable for cosine similarity / SupCon loss.

At inference: prototype = mean(embed(support_samples))  (already L2-normed)
              predict   = argmax cosine_sim(embed(query), prototypes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BUILDING BLOCKS (reused from / consistent with MOE_CNN_LSTM)
# ============================================================

class DemographicsEncoder(nn.Module):
    """Encodes a demographics vector into a fixed-dim embedding."""
    def __init__(self, in_dim: int, demo_emb_dim: int = 16):
        super().__init__()
        hidden = max(16, demo_emb_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, demo_emb_dim),
        )

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return self.net(d)


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation.
    Modulates CNN feature maps using a conditioning vector (e.g. demographics).
    h: (B, C, T) → scaled/shifted by gamma, beta derived from cond_dim vector.
    """
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta  = nn.Linear(cond_dim, feat_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # h: (B, C, T),  c: (B, cond_dim)
        gamma = self.gamma(c).unsqueeze(-1)   # (B, C, 1)
        beta = self.beta(c).unsqueeze(-1)    # (B, C, 1)
        return h * (1 + gamma) + beta


class AttentionPool1d(nn.Module):
    """
    Learnable attention pooling over a (B, C, T) feature map → (B, C).
    Computes a scalar attention weight per time-step, softmax, weighted sum.
    Faster than LSTM and naturally permutation-equivariant.
    """
    def __init__(self, in_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = max(1, in_dim // num_heads)

        self.query = nn.Parameter(torch.randn(1, num_heads, head_dim))
        self.key_proj = nn.Linear(in_dim, num_heads * head_dim)
        self.val_proj = nn.Linear(in_dim, in_dim)
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → permute to (B, T, C)
        B, C, T = x.shape
        x_t = x.permute(0, 2, 1)                         # (B, T, C)

        k = self.key_proj(x_t)                            # (B, T, H*Hd)
        k = k.view(B, T, self.num_heads, self.head_dim)   # (B, T, H, Hd)
        k = k.permute(0, 2, 1, 3)                         # (B, H, T, Hd)

        q = self.query.expand(B, -1, -1)                  # (B, H, Hd)
        q = q.unsqueeze(2)                                 # (B, H, 1, Hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, H, 1, T)
        attn = F.softmax(attn, dim=-1)                     # (B, H, 1, T)

        v = self.val_proj(x_t)                            # (B, T, C)
        # Weighted sum: average over heads, sum over time
        attn_avg = attn.mean(dim=1).squeeze(1)            # (B, T)
        pooled = (attn_avg.unsqueeze(-1) * v).sum(dim=1) # (B, C)
        return pooled


class ProjectionHead(nn.Module):
    """
    MLP projection head: backbone_dim → proj_hidden_dim → embedding_dim
    Output is L2-normalized (unit hypersphere).

    If proj_hidden_dim is None, uses a single linear layer.
    This is the 'g(·)' in SimCLR/SupCon — only used during training.
    At inference, you can optionally use the backbone features directly
    (before projection) for prototyping, but using projection is fine too.
    """
    def __init__(self, in_dim: int, embedding_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, embedding_dim),
            )
        else:
            self.net = nn.Linear(in_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ============================================================
# MAIN ENCODER
# ============================================================

class ContrastiveGestureEncoder(nn.Module):
    """
    Full encoder: raw signal → L2-normalized embedding.

    Forward signature:
        emg  : (B, C_emg, T)
        imu  : (B, C_imu, T) | None
        demo : (B, demo_in_dim) | None

    Returns:
        z    : (B, embedding_dim) — L2 normalized, on unit sphere
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # ---- CNN Encoders ----
        self.emg_encoder = self._build_cnn(
            config['emg_in_ch'],
            config['emg_base_cnn_filters'],
            config['emg_cnn_layers'],
            config['cnn_kernel_size'],
            config['emg_stride'],
            config['groupnorm_num_groups'],
        )
        emg_out_ch = config['emg_base_cnn_filters'] * (2 ** (config['emg_cnn_layers'] - 1))

        imu_out_ch = 0
        self.imu_encoder = None
        if config.get('use_imu', False):
            self.imu_encoder = self._build_cnn(
                config['imu_in_ch'],
                config['imu_base_cnn_filters'],
                config['imu_cnn_layers'],
                config['cnn_kernel_size'],
                config['imu_stride'],
                config['groupnorm_num_groups'],
            )
            imu_out_ch = config['imu_base_cnn_filters'] * (2 ** (config['imu_cnn_layers'] - 1))

        cnn_out_dim = emg_out_ch + imu_out_ch

        # ---- Demographics + FiLM ----
        self.demo_encoder = None
        self.film_emg = None
        self.film_imu = None
        if config.get('use_demographics', False):
            self.demo_encoder = DemographicsEncoder(config['demo_in_dim'], config['demo_emb_dim'])
            if config.get('use_film_x_demo', False):
                self.film_emg = FiLMConditioner(emg_out_ch, config['demo_emb_dim'])
                if self.imu_encoder is not None:
                    self.film_imu = FiLMConditioner(imu_out_ch, config['demo_emb_dim'])

        # ---- Temporal module ----
        arch = config.get('arch_mode', 'cnn_attn')

        if arch == 'cnn_lstm':
            self.temporal = nn.LSTM(
                input_size=cnn_out_dim,
                hidden_size=config['lstm_hidden'],
                num_layers=config['lstm_layers'],
                batch_first=True,
                bidirectional=True,
                dropout=config['dropout'] if config['lstm_layers'] > 1 else 0,
            )
            backbone_dim = config['lstm_hidden'] * 2
            self.use_gap = config.get('use_GlobalAvgPooling', True)
            self.attn_pool = None

        elif arch == 'cnn_attn':
            self.temporal = None
            self.attn_pool = AttentionPool1d(
                in_dim=cnn_out_dim,
                num_heads=config.get('attn_pool_heads', 4),
            )
            backbone_dim = cnn_out_dim
        else:
            raise ValueError(f"Unknown arch_mode: '{arch}'. Choose 'cnn_lstm' or 'cnn_attn'.")

        # ---- Projection Head ----
        self.proj_head = ProjectionHead(
            in_dim=backbone_dim,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config.get('proj_hidden_dim', 256),
        )

        self.backbone_dim = backbone_dim
        self.embedding_dim = config['embedding_dim']

    # ----------------------------------------------------------
    def _build_cnn(self, in_ch, base_filters, num_layers, kernel_size, stride, gn_groups):
        layers = []
        c_in, c_out = in_ch, base_filters
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2),
                nn.GroupNorm(gn_groups, c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config['dropout']),
            ]
            c_in, c_out = c_out, c_out * 2
        return nn.Sequential(*layers)

    # ----------------------------------------------------------
    def _encode_signals(self, emg: torch.Tensor,
                        imu: torch.Tensor = None,
                        d_emb: torch.Tensor = None) -> torch.Tensor:
        """CNN → (optional FiLM) → concat → temporal → (B, backbone_dim)"""

        # 1. CNN
        e = self.emg_encoder(emg)   # (B, emg_out_ch, T')

        i = None
        if self.imu_encoder is not None and imu is not None:
            i = self.imu_encoder(imu)

        # 2. FiLM conditioning
        if d_emb is not None and self.film_emg is not None:
            if d_emb.size(0) == 1 and e.size(0) > 1:
                d_emb_exp = d_emb.expand(e.size(0), -1)
            else:
                d_emb_exp = d_emb
            e = self.film_emg(e, d_emb_exp)
            if i is not None and self.film_imu is not None:
                i = self.film_imu(i, d_emb_exp)

        # 3. Fusion
        combined = torch.cat([e, i], dim=1) if i is not None else e  # (B, C, T')

        # 4. Temporal
        if self.temporal is not None:
            # LSTM path
            x = combined.permute(0, 2, 1)                 # (B, T', C)
            out, (hn, _) = self.temporal(x)
            if self.use_gap:
                feat = torch.mean(out, dim=1)              # (B, lstm_hidden*2)
            else:
                feat = torch.cat([hn[-2], hn[-1]], dim=1) # (B, lstm_hidden*2)
        else:
            # Attention pool path
            feat = self.attn_pool(combined)                # (B, C)

        return feat

    # ----------------------------------------------------------
    def encode(self, emg: torch.Tensor,
               imu: torch.Tensor = None,
               demo: torch.Tensor = None) -> torch.Tensor:
        """
        Backbone only (before projection head, DOES NOT APPLY THE PROJECTION HEAD).
        Useful for prototype-based inference (can also use forward() — both work).
        Returns (B, backbone_dim), NOT normalized.
        """
        d_emb = self.demo_encoder(demo) if (self.demo_encoder and demo is not None) else None
        return self._encode_signals(emg, imu, d_emb)

    # ----------------------------------------------------------
    def forward(self, emg: torch.Tensor,
                imu: torch.Tensor = None,
                demo: torch.Tensor = None) -> torch.Tensor:
        """
        Full forward pass: signal → backbone → projection head → L2-norm.
        DOES APPLY THE PROJECTION HEAD
        Returns z: (B, embedding_dim), unit-norm vectors on the hypersphere.
        """
        d_emb = self.demo_encoder(demo) if (self.demo_encoder and demo is not None) else None
        feat = self._encode_signals(emg, imu, d_emb)
        return self.proj_head(feat)

    # ----------------------------------------------------------
    @torch.no_grad()
    def get_prototypes(self, support_emg: torch.Tensor,
                       support_labels: torch.Tensor,
                       support_imu: torch.Tensor = None,
                       support_demo: torch.Tensor = None) -> dict:
        """
        Compute class prototypes from a support set.

        Args:
            support_emg    : (N, C, T)
            support_labels : (N,)  integer class labels
            support_imu    : (N, C_imu, T) | None
            support_demo   : (N, demo_dim) | None  (or (1, demo_dim) broadcast)

        Returns:
            dict: {label (int) → prototype (embedding_dim,)}
            All prototypes are L2-normalized (mean of normalized embeddings).
        """
        was_training = self.training
        self.eval()

        z = self.forward(support_emg, support_imu, support_demo)  # (N, D) already normed

        prototypes = {}
        for cls in support_labels.unique():
            mask = support_labels == cls
            proto = z[mask].mean(dim=0)
            prototypes[cls.item()] = F.normalize(proto, dim=0)

        if was_training:
            self.train()
        return prototypes

    # ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, query_emg: torch.Tensor,
                prototypes: dict,
                query_imu: torch.Tensor = None,
                query_demo: torch.Tensor = None) -> torch.Tensor:
        """
        Nearest-prototype cosine prediction.

        Args:
            query_emg   : (B, C, T)
            prototypes  : dict from get_prototypes()

        Returns:
            pred_labels : (B,) integer tensor of predicted class labels
        """
        was_training = self.training
        self.eval()

        z_q = self.forward(query_emg, query_imu, query_demo)  # (B, D)

        # TODO: What is the connection to k-shot here? Does this still run in 1-shot?
        # $B k-shot... is a bit separate. $B did 1-NN no matter the shot...
        # TODO: I dont love that this is inferring labels...
        # Since this is only the prediction function we don't get/know the labels yet
        ## It is just pulling the labels from the prototypes to assign our query samples...
        labels  = sorted(prototypes.keys())
        proto_mat = torch.stack([prototypes[l] for l in labels], dim=0)  # (K, D)
        proto_mat = proto_mat.to(z_q.device)

        # cosine sim: z_q already normed, proto_mat already normed
        sims = z_q @ proto_mat.T             # (B, K)
        pred_idx = sims.argmax(dim=1)        # (B,)
        pred_labels = torch.tensor([labels[i] for i in pred_idx.tolist()],
                                   dtype=torch.long, device=z_q.device)

        if was_training:
            self.train()
        return pred_labels
