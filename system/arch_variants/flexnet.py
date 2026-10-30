"""
flexnet.py
==========
FlexNet = [fusion encoder | E fusion encoders + router] -> temporal model -> head.

One class covers the fusion screen and the transformer backbones of
docs/specs/SPEC_architecture_exploration.md:

  fusion_mode   (config["fusion_mode"])
    "early"  F0: concat EMG+IMU channels, one CNN (same layout as DeepCNNLSTM / A4)
    "mid"    F1: one CNN per modality, features concatenated before the temporal model
    "film"   F2: EMG CNN modulated per channel and time step by gamma, beta computed from an IMU CNN
                 (FiLM; the modulation starts at identity because the gamma/beta conv is zero-initialised)
    "xattn"  F3: per-modality CNNs, then EMG tokens cross-attend to IMU tokens and vice versa
    "none"   no CNN: raw concatenated channels go straight to the temporal model (T0, T4)

  temporal_model (config["temporal_model"])
    "lstm"         3 x (bi)LSTM, as M0 / A4
    "transformer"  arch_variants.attention.TemporalTransformer (T0; T4 when tf_moe_every > 0)

  use_MOE (config["use_MOE"]) with an encoder fusion mode: E copies of the fusion encoder are the
  experts, routed exactly like M0 (context projector on the concatenated raw input, softmax gate,
  top-k). This is how the winning fusion is carried into the MoE family.

M0 itself (X1) and its gate / routing variants (X2-X6) and T2 stay in MOE_encoder.DeepCNNLSTM_EncoderMOE
so that every X comparison changes one thing only; FlexNet is for the new families.

Every config key FlexNet reads is required (config[...]): the HPO driver and tests set them; nothing
falls back silently.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arch_variants.attention import CrossAttentionBlock, TemporalTransformer, merge_routing


def _width(base: int, mult: float, gn: int) -> int:
    """Channel width rounded down to a multiple of the GroupNorm group count (min one group)."""
    return max(gn, (int(round(base * mult)) // gn) * gn)


def _cnn(in_ch: int, base: int, n_layers: int, k: int, gn: int):
    """Doubling Conv-GN-GELU stack, the DeepCNNLSTM layout (no conv dropout). Returns (module, out_ch)."""
    layers, c_in, c_out = [], in_ch, base
    for _ in range(n_layers):
        layers += [nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2, bias=False),
                   nn.GroupNorm(gn, c_out), nn.GELU()]
        c_in, c_out = c_out, c_out * 2
    return nn.Sequential(*layers), c_in


class FusionEncoder(nn.Module):
    """(x_emg (B, Ce, T), x_imu (B, Ci, T)) -> (B, out_ch, T)."""

    def __init__(self, config: dict):
        super().__init__()
        mode = config["fusion_mode"]
        C_emg, C_imu = config["emg_in_ch"], config["imu_in_ch"]
        base, n_layers = config["cnn_base_filters"], config["cnn_layers"]
        k, gn = config["cnn_kernel"], config["groupnorm_num_groups"]
        if mode != "early" and not config["use_imu"]:
            raise ValueError(f"fusion_mode={mode!r} needs use_imu=True")
        self.mode = mode
        if mode == "early":
            self.cnn, self.out_ch = _cnn(C_emg + (C_imu if config["use_imu"] else 0), base, n_layers, k, gn)
            self.use_imu = config["use_imu"]
        elif mode in ("mid", "xattn"):
            w = config["fusion_branch_width"]
            self.emg_cnn, ce = _cnn(C_emg, _width(base, w, gn), n_layers, k, gn)
            self.imu_cnn, ci = _cnn(C_imu, _width(base, w, gn), n_layers, k, gn)
            if mode == "mid":
                self.out_ch = ce + ci
            else:
                d = config["fusion_d_model"]
                self.emg_tok = nn.Linear(ce, d)
                self.imu_tok = nn.Linear(ci, d)
                self.emg_from_imu = CrossAttentionBlock(d, config["fusion_heads"], config["dropout"])
                self.imu_from_emg = CrossAttentionBlock(d, config["fusion_heads"], config["dropout"])
                self.out_ch = 2 * d
        elif mode == "film":
            self.emg_cnn, ce = _cnn(C_emg, base, n_layers, k, gn)
            self.imu_cnn, ci = _cnn(C_imu, _width(base, config["fusion_branch_width"], gn), n_layers, k, gn)
            self.film = nn.Conv1d(ci, 2 * ce, kernel_size=1)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.out_ch = ce
        else:
            raise ValueError(f"FusionEncoder: unknown fusion_mode {mode!r} (early|mid|film|xattn)")

    def forward(self, x_emg, x_imu):
        if self.mode == "early":
            x = torch.cat([x_emg, x_imu], dim=1) if (self.use_imu and x_imu is not None) else x_emg
            return self.cnn(x)
        if x_imu is None:
            raise ValueError(f"fusion_mode={self.mode!r} got no IMU input")
        he, hi = self.emg_cnn(x_emg), self.imu_cnn(x_imu)
        if self.mode == "mid":
            return torch.cat([he, hi], dim=1)
        if self.mode == "film":
            gamma, beta = self.film(hi).chunk(2, dim=1)
            return he * (1.0 + gamma) + beta
        e = self.emg_tok(he.permute(0, 2, 1))
        i = self.imu_tok(hi.permute(0, 2, 1))
        e2, i2 = self.emg_from_imu(e, i), self.imu_from_emg(i, e)
        return torch.cat([e2, i2], dim=-1).permute(0, 2, 1)


class FlexNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        C_emg, C_imu = config["emg_in_ch"], config["imu_in_ch"]
        self.use_imu = config["use_imu"]
        drop = config["dropout"]
        n_way = config["n_way"] if config["meta_learning"] else config["pretrain_num_classes"]
        self.fusion_mode = config["fusion_mode"]
        self.temporal_model = config["temporal_model"]
        self.use_moe_encoder = bool(config["use_MOE"]) and self.fusion_mode != "none"
        self.needs_task_support = False
        raw_ch = C_emg + (C_imu if self.use_imu else 0)

        # ── encoder (single, or an expert bank of fusion encoders) ───────────
        if self.fusion_mode == "none":
            self.encoder, enc_out = None, raw_ch
        elif not self.use_moe_encoder:
            self.encoder = FusionEncoder(config)
            enc_out = self.encoder.out_ch
        else:
            from MOE.MOE_encoder import MOEGate, build_routing_module
            E = config["num_experts"]
            self.num_experts = E
            self.experts = nn.ModuleList([FusionEncoder(config) for _ in range(E)])
            enc_out = self.experts[0].out_ch
            self.router, gate_in = build_routing_module(
                routing_signal="context_proj", in_ch=raw_ch,
                ctx_hidden_dim=config["MOE_ctx_hidden_dim"], ctx_out_dim=config["MOE_ctx_out_dim"],
                use_conv_in_proj=True, conv_kernel=config["cnn_kernel"],
                gn_groups=config["groupnorm_num_groups"], dropout=config["MOE_dropout"], demo_embed_dim=0)
            self.gate = MOEGate(gate_in, E, top_k=config["MOE_top_k"],
                                temperature=config["MOE_gate_temperature"])
            self.skip_unselected = bool(config["MOE_skip_unselected_experts"])

        # ── temporal model ───────────────────────────────────────────────────
        if self.temporal_model == "lstm":
            lstm_h, bidir = config["lstm_hidden"], config["bidirectional"]
            D = 2 if bidir else 1
            self.lstms = nn.ModuleList([
                nn.LSTM(enc_out if i == 0 else lstm_h * D, lstm_h, batch_first=True, bidirectional=bidir)
                for i in range(3)])
            from MOE.MOE_encoder import _init_lstm
            for l in self.lstms:
                _init_lstm(l)
            self.lstm_dropout = nn.Dropout(drop)
            self.feat_dim = lstm_h * D
        elif self.temporal_model == "transformer":
            d = config["tf_d_model"]
            every = config["tf_moe_every"]          # T4: a MoE feed-forward in every `every`-th block
            moe_layers = tuple(i for i in range(config["tf_blocks"]) if every > 0 and i % every == every - 1)
            if moe_layers and self.use_moe_encoder:
                raise ValueError("token-level MoE (tf_moe_every > 0) and an encoder expert bank are exclusive")
            self.token_moe = bool(moe_layers)
            self.temporal = TemporalTransformer(
                enc_out, d_model=d, n_heads=config["tf_heads"], n_blocks=config["tf_blocks"],
                d_ff=int(config["tf_ff_mult"] * d), dropout=drop, moe_layers=moe_layers,
                moe_experts=(config["num_experts"] if moe_layers else 0),
                moe_top_k=(config["MOE_top_k"] if moe_layers else 1),
                moe_temperature=(config["MOE_gate_temperature"] if moe_layers else 1.0))
            self.feat_dim = d
        else:
            raise ValueError(f"temporal_model must be 'lstm' or 'transformer', got {self.temporal_model!r}")
        if bool(config["use_MOE"]) and not (self.use_moe_encoder or getattr(self, "token_moe", False)):
            raise ValueError("use_MOE=True but FlexNet has no expert bank: use an encoder fusion mode, "
                             "or fusion_mode='none' with temporal_model='transformer' and tf_moe_every > 0")

        if config["head_type"] == "linear":
            self.head = nn.Linear(self.feat_dim, n_way)
        else:
            self.head = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 2), nn.GELU(),
                                      nn.Dropout(drop), nn.Linear(self.feat_dim // 2, n_way))

    def _encode(self, x_emg, x_imu):
        """-> (h (B, C, T), routing dict or None)."""
        if self.fusion_mode == "none":
            x = torch.cat([x_emg, x_imu], dim=1) if (self.use_imu and x_imu is not None) else x_emg
            return x, None
        if not self.use_moe_encoder:
            return self.encoder(x_emg, x_imu), None
        from MOE.MOE_encoder import _weighted_expert_sum
        x = torch.cat([x_emg, x_imu], dim=1) if (self.use_imu and x_imu is not None) else x_emg
        w_hard, w_soft, logits = self.gate.forward_with_logits(self.router(x))
        h = _weighted_expert_sum([lambda ex=ex: ex(x_emg, x_imu) for ex in self.experts], w_hard,
                                 skip_unselected=self.skip_unselected)
        return h, {"gate_weights": w_hard, "gate_weights_soft": w_soft, "router_logits": logits}

    def _backbone(self, x_emg, x_imu=None):
        h, routing = self._encode(x_emg, x_imu)
        if self.temporal_model == "lstm":
            x = h.permute(0, 2, 1)
            feats = []
            for i, lstm in enumerate(self.lstms):
                x, _ = lstm(x)
                if i < 2:
                    x = self.lstm_dropout(x)
                feats.append(x.mean(dim=1))
            return feats[-1], feats, routing
        feat, feats, token_routings = self.temporal(h)
        if token_routings:
            routing = merge_routing(token_routings)
        return feat, feats, routing

    def backbone(self, x_emg, x_imu=None, demographics=None):
        feat, feats, _ = self._backbone(x_emg, x_imu)
        return feat, feats

    def forward(self, x_emg, x_imu=None, demographics=None, return_routing: bool = False):
        feat, _, routing = self._backbone(x_emg, x_imu)
        logits = self.head(feat)
        if return_routing and routing is not None:
            return logits, routing
        return logits

    def get_features(self, x_emg, x_imu=None, demographics=None):
        with torch.no_grad():
            return self.backbone(x_emg, x_imu, demographics)


def build_flexnet(config: dict) -> FlexNet:
    return FlexNet(config)
