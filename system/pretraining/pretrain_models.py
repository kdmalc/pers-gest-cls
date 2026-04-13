"""
pretrain_models.py
==================
Three model architectures for supervised pretraining of EMG gesture classification.
All are designed to eventually transfer their backbone to MAML fine-tuning.

Architecture overview:
  1. MetaCNNLSTM   - Meta/Bengio-style: 1 Conv + 3 LSTM + linear head (minimal nonlinearity)
  2. DeepCNNLSTM   - NinaPro-style: 3 Conv + 3 LSTM + MLP head (our slight tweak from original 4+2)
  3. TST            - Time Series Transformer: patch embedding + N encoder blocks + CLS token

Input convention (all models):
  x_emg: (B, C_emg, T)  e.g. (B, 16, 64)
  x_imu: (B, C_imu, T)  e.g. (B, 72, 64)  [optional]

Backbone vs. head separation is explicit in every model so MAML can easily
replace the head with its own linear layer at meta-train time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    """Single linear classification layer — the 'linear readout' from the Meta paper."""
    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    """Two-layer MLP with GELU. Slightly more expressive than linear."""
    def __init__(self, feat_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MetaCNNLSTM
# ─────────────────────────────────────────────────────────────────────────────

class MetaCNNLSTM(nn.Module):
    """
    Closely mirrors the architecture described in Bengio/Meta paper:
      - 1 temporal Conv1d layer (bandpass-like filters over time)
      - 3 LSTM layers (stacked, bidirectional optional)
      - Linear readout (the paper explicitly removes nonlinearity in the readout)

    Key design choices to match the paper:
      - GroupNorm rather than BatchNorm (works correctly with batch_size=1 at meta-test)
      - NO nonlinearity between LSTM → linear head
      - Intermediate representations from each LSTM layer are easily extractable
        (used for the tSNE/PCA visualization in Fig.4f-h)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        C_emg    = config['emg_in_ch']        # 16
        C_imu    = config.get('imu_in_ch', 72)
        use_imu  = config.get('use_imu', False)
        n_filt   = config['cnn_filters']       # e.g. 64
        k        = config['cnn_kernel']        # e.g. 3
        gn_grps  = config.get('groupnorm_num_groups', 8)
        lstm_h   = config['lstm_hidden']       # e.g. 128
        n_way    = config['n_way'] if config['meta_learning'] else config['pretrain_num_classes']
        drop     = config.get('dropout', 0.0)
        bidir    = config.get('bidirectional', False)
        head_type = config.get('head_type', 'linear')

        # ── Conv layer ──────────────────────────────────────────────────────
        in_ch = C_emg + (C_imu if use_imu else 0)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, n_filt, kernel_size=k, padding=k // 2, bias=False),
            nn.GroupNorm(gn_grps, n_filt),
            nn.ReLU(),
            #nn.Dropout(drop),  # I dont think conv layers need dropout...
        )

        # ── 3 × LSTM layers ─────────────────────────────────────────────────
        # Keep them as separate modules so we can extract intermediate reps
        D = 2 if bidir else 1
        self.lstm1 = nn.LSTM(n_filt,  lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.LSTM(lstm_h * D, lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm3 = nn.LSTM(lstm_h * D, lstm_h, batch_first=True, bidirectional=bidir)

        def _init_lstm(lstm_module):
            for name, p in lstm_module.named_parameters():
                if 'weight_hh' in name:
                    # Orthogonal init keeps gradient norms stable through time
                    nn.init.orthogonal_(p)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
                    # Set forget gate bias to 1 — critical for gradient flow
                    n = p.size(0)
                    p.data[n//4 : n//2].fill_(1.0)

        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            _init_lstm(lstm)

        self.dropout = nn.Dropout(drop)
        self.feat_dim = lstm_h * D  # dimension that enters the head

        # ── Head ────────────────────────────────────────────────────────────
        if head_type == 'linear':
            self.head = LinearHead(self.feat_dim, n_way)
        else:
            self.head = MLPHead(self.feat_dim, self.feat_dim // 2, n_way, drop)

    # ── backbone returns (final_feat, [layer1_feat, layer2_feat, layer3_feat]) ──
    def backbone(self, x_emg: torch.Tensor, x_imu=None, demographics=None):
        """
        Returns the final feature vector AND intermediate LSTM representations
        (needed for Fig.4-style tSNE visualization).

        x_emg: (B, C_emg, T)
        Returns:
          feat:   (B, feat_dim)  final representation
          layers: list of (B, feat_dim) — one per LSTM layer
        """
        x = x_emg
        if self.config.get('use_imu', False) and x_imu is not None:
            x = torch.cat([x, x_imu], dim=1)   # (B, C_emg+C_imu, T)

        # Conv: (B, C, T) → (B, n_filt, T) → permute → (B, T, n_filt)
        h = self.conv(x).permute(0, 2, 1)

        # LSTM layers — collect last hidden state from each
        h1, (hn1, _) = self.lstm1(h)
        h1 = self.dropout(h1)
        # Concat fwd+bwd hidden for bidir
        layer1_feat = self._pool(h1)

        h2, (hn2, _) = self.lstm2(h1)
        h2 = self.dropout(h2)
        layer2_feat = self._pool(h2)

        h3, (hn3, _) = self.lstm3(h2)
        #h3 = self.dropout(h3)  # Final LSTM layer probably doesnt need dropout either
        layer3_feat = self._pool(h3)

        return layer3_feat, [layer1_feat, layer2_feat, layer3_feat]

    def _pool(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool over time. Returns (B, hidden).
        Alternative: use last hidden state — both are fine.
        Mean-pool tends to be more stable for shorter sequences.
        """
        return lstm_out.mean(dim=1)

    def forward(self, x_emg: torch.Tensor, x_imu=None, demographics=None):
        feat, _ = self.backbone(x_emg, x_imu, demographics=demographics)
        return self.head(feat)

    def get_features(self, x_emg: torch.Tensor, x_imu=None, demographics=None):
        with torch.no_grad():
            feat, layers = self.backbone(x_emg, x_imu, demographics=demographics)
        return feat, layers


# ─────────────────────────────────────────────────────────────────────────────
# 2. DeepCNNLSTM
# ─────────────────────────────────────────────────────────────────────────────

class DeepCNNLSTM(nn.Module):
    """
    Inspired by NinaPro / Côté-Allard et al. deep CNN-LSTM for sEMG.
    Original: 4 Conv + 2 LSTM. Here: 3 Conv + 3 LSTM (slightly more temporal depth).

    Key differences from MetaCNNLSTM:
      - Multiple conv layers with doubling filters (acts as a feature pyramid)
      - GELU activations throughout (smoother gradients)
      - GroupNorm inside of CNN
      - MLP head with one hidden layer
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        C_emg     = config['emg_in_ch']
        C_imu     = config.get('imu_in_ch', 72)
        use_imu   = config.get('use_imu', False)
        base_f    = config['cnn_base_filters']    # e.g. 32
        n_layers  = config.get('cnn_layers', 3)   # 3 conv layers
        k         = config['cnn_kernel']
        gn_groups = config['groupnorm_num_groups']
        lstm_h    = config['lstm_hidden']
        n_way     = config['n_way'] if config['meta_learning'] else config['pretrain_num_classes']
        drop      = config.get('dropout', 0.2)
        bidir     = config.get('bidirectional', True)
        head_type = config.get('head_type', 'mlp')

        in_ch = C_emg + (C_imu if use_imu else 0)

        # ── CNN Encoder (3 layers, doubling filters) ─────────────────────────
        cnn_layers = []
        curr_in, curr_out = in_ch, base_f
        for i in range(n_layers):
            cnn_layers += [
                nn.Conv1d(curr_in, curr_out, kernel_size=k, padding=k // 2, bias=False),
                #nn.BatchNorm1d(curr_out),  
                nn.GroupNorm(num_groups=gn_groups, num_channels=curr_out),
                nn.GELU(),
                #nn.Dropout(drop),  # I dont think convs need dropout...
            ]
            curr_in = curr_out
            curr_out = curr_out * 2
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_dim = curr_in  # after the last doubling

        # ── 3 × LSTM layers ─────────────────────────────────────────────────
        D = 2 if bidir else 1
        self.lstm1 = nn.LSTM(cnn_out_dim, lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.LSTM(lstm_h * D, lstm_h,  batch_first=True, bidirectional=bidir)
        self.lstm3 = nn.LSTM(lstm_h * D, lstm_h,  batch_first=True, bidirectional=bidir)

        def _init_lstm(lstm_module):
            for name, p in lstm_module.named_parameters():
                if 'weight_hh' in name:
                    # Orthogonal init keeps gradient norms stable through time
                    nn.init.orthogonal_(p)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
                    # Set forget gate bias to 1 — critical for gradient flow
                    n = p.size(0)
                    p.data[n//4 : n//2].fill_(1.0)

        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            _init_lstm(lstm)

        self.dropout = nn.Dropout(drop)
        self.feat_dim = lstm_h * D

        # ── Head ────────────────────────────────────────────────────────────
        if head_type == 'linear':
            self.head = LinearHead(self.feat_dim, n_way)
        else:
            self.head = MLPHead(self.feat_dim, self.feat_dim // 2, n_way, drop)

    def backbone(self, x_emg: torch.Tensor, x_imu=None, demographics=None):
        if self.config.get('use_imu', False) and x_imu is not None:
            assert x_emg.shape[0] == x_imu.shape[0], \
                f"Batch size mismatch: EMG {x_emg.shape} vs IMU {x_imu.shape}"
            assert x_emg.shape[2] == x_imu.shape[2], \
                f"Sequence length mismatch: EMG {x_emg.shape} vs IMU {x_imu.shape}. " \
                f"Check strides/padding in data pipeline."
            x = torch.cat([x_emg, x_imu], dim=1)
        else:
            x = x_emg

        h = self.cnn(x).permute(0, 2, 1)   # (B, T, C)

        h1, _ = self.lstm1(h);   h1 = self.dropout(h1)
        h2, _ = self.lstm2(h1);  h2 = self.dropout(h2)
        h3, _ = self.lstm3(h2);  #h3 = self.dropout(h3)

        layer1_feat = h1.mean(dim=1)
        layer2_feat = h2.mean(dim=1)
        layer3_feat = h3.mean(dim=1)

        return layer3_feat, [layer1_feat, layer2_feat, layer3_feat]

    def forward(self, x_emg: torch.Tensor, x_imu=None, demographics=None):
        feat, _ = self.backbone(x_emg, x_imu, demographics=demographics)
        return self.head(feat)

    def get_features(self, x_emg, x_imu=None, demographics=None):
        with torch.no_grad():
            return self.backbone(x_emg, x_imu, demographics=demographics)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time Series Transformer (TST)
# ─────────────────────────────────────────────────────────────────────────────
# Following: Zerveas et al. "A Transformer-based Framework for Multivariate
# Time Series Representation Learning" (KDD 2021).
#
# Key design choices for YOUR data (small dataset):
#   - Patch-based tokenization (reduces sequence length T=64 → T/P tokens)
#   - Learnable CLS token (pooling via CLS avoids aggregation ablations)
#   - Pre-LN (layer norm before attention) — more stable for small data
#   - Relative positional bias rather than absolute PE (optional, off by default)
#   - d_model scaled DOWN (64-128 instead of 512+) — your dataset is tiny
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Projects each non-overlapping time patch to d_model.
    input: (B, C, T) → output: (B, n_patches, d_model)
    """
    def __init__(self, in_channels: int, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_len, stride=patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        x = self.proj(x)           # (B, d_model, n_patches)
        x = x.permute(0, 2, 1)    # (B, n_patches, d_model)
        return self.dropout(x)


class TSTEncoderBlock(nn.Module):
    """
    Single Pre-LN Transformer encoder block.
    Pre-LN (norm before sublayer) is strongly preferred for small data —
    it eliminates the need for learning rate warmup and is more stable.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-LN self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Pre-LN FFN
        x = x + self.ff(self.norm2(x))
        return x


class TST(nn.Module):
    """
    Time Series Transformer for EMG gesture classification.

    Design follows Zerveas et al. (KDD 2021) with adaptations for small data:
      1. PatchEmbedding instead of per-timestep projection (reduces seq len, improves inductive bias)
      2. CLS token for classification pooling
      3. Absolute sinusoidal positional encoding (optional; learnable is fine too)
      4. Scaled down d_model (default 64) for the dataset size
      5. 3-4 encoder blocks (deep enough to be expressive, shallow enough not to overfit)

    The backbone() method returns intermediate block outputs for probing.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        C_emg     = config['emg_in_ch']
        C_imu     = config.get('imu_in_ch', 72)
        use_imu   = config.get('use_imu', False)
        patch_len = config.get('patch_len', 8)     # T=64 → 8 patches
        d_model   = config['d_model']              # e.g. 64 or 128
        n_heads   = config['n_heads']              # must divide d_model
        d_ff      = config.get('d_ff', d_model * 4)
        n_blocks  = config.get('n_blocks', 4)      # 3-6 per paper
        drop      = config.get('dropout', 0.2)
        n_way     = config['n_way'] if config['meta_learning'] else config['pretrain_num_classes']
        head_type = config.get('head_type', 'mlp')

        in_ch = C_emg + (C_imu if use_imu else 0)

        # ── Patch embedding ──────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(in_ch, patch_len, d_model, drop)
        T = config.get('seq_len', 64)
        n_patches = T // patch_len

        # ── CLS token + positional encoding ─────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Sinusoidal PE for sequence positions (CLS gets position 0)
        self.pos_enc = self._make_sinusoidal_pe(n_patches + 1, d_model)
        self.pos_drop = nn.Dropout(drop)

        # ── Transformer blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TSTEncoderBlock(d_model, n_heads, d_ff, drop)
            for _ in range(n_blocks)
        ])
        self.norm_final = nn.LayerNorm(d_model)

        self.feat_dim = d_model
        self.n_blocks = n_blocks

        # ── Head ────────────────────────────────────────────────────────────
        if head_type == 'linear':
            self.head = LinearHead(d_model, n_way)
        else:
            self.head = MLPHead(d_model, d_model // 2, n_way, drop)

        self._init_weights()

    @staticmethod
    def _make_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def backbone(self, x_emg: torch.Tensor, x_imu=None, demographics=None):
        """
        Returns (cls_feat, [block_i_cls_feat, ...])
        cls_feat: (B, d_model) — CLS token output from final block
        block_feats: list of (B, d_model) from each block (for visualization/probing)

        demographics is not used at all...
        """
        x = x_emg
        if self.config.get('use_imu', False) and x_imu is not None:
            x = torch.cat([x, x_imu], dim=1)

        # Patch embed: (B, C, T) → (B, n_patches, d_model)
        patches = self.patch_embed(x)
        B = patches.size(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)   # (B, 1+n_patches, d_model)

        # Positional encoding (register as non-parameter buffer so it moves with .to(device))
        pe = self.pos_enc.to(tokens.device)
        tokens = self.pos_drop(tokens + pe)

        # Pass through blocks, collect CLS rep after each
        block_feats = []
        for block in self.blocks:
            tokens = block(tokens)
            block_feats.append(tokens[:, 0])   # CLS token

        # Final LayerNorm on CLS
        cls_feat = self.norm_final(tokens[:, 0])
        return cls_feat, block_feats

    def forward(self, x_emg: torch.Tensor, x_imu=None, demographics=None):
        feat, _ = self.backbone(x_emg, x_imu, demographics=demographics)
        #print("--- FEATURE STATS ---")
        #print(f"feat mean={feat.mean().item():.4f} std={feat.std().item():.4f} norm={feat.norm().item():.4f}")
        return self.head(feat)

    def get_features(self, x_emg, x_imu=None, demographics=None):
        with torch.no_grad():
            return self.backbone(x_emg, x_imu, demographics=demographics)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Instantiate a model by name.
    config['model_type'] ∈ {'MetaCNNLSTM', 'DeepCNNLSTM', 'TST'}
    """
    name = config['model_type']
    if name == 'MetaCNNLSTM':
        return MetaCNNLSTM(config)
    elif name == 'DeepCNNLSTM':
        return DeepCNNLSTM(config)
    elif name == 'TST':
        return TST(config)
    else:
        raise ValueError(f"Unknown model_type: '{name}'. Choose from MetaCNNLSTM, DeepCNNLSTM, TST.")
