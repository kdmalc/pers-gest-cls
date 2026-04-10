"""
A9_emg_imu_modality_encoding.py
================================
Ablation A9: EMG+IMU Modality Encoding Ablation

⚠️  IMPORTANT ARCHITECTURE NOTE — READ BEFORE RUNNING ⚠️
=========================================================
After inspecting MOE_encoder.py, the M0 (DeepCNNLSTM_EncoderMOE) model ALREADY
performs naive channel-dimension concatenation of EMG and IMU before all K expert
CNNs. Specifically, every expert CNN sees `x = cat([x_emg, x_imu], dim=1)` with
shape (B, C_emg + C_imu, T) = (B, 88, T). The context projector also operates on
this concatenated input.

So what the spec calls "naive concatenation" is precisely what M0 does. The natural
ablation is therefore to test SEPARATE modality encoders — i.e., have each expert run
a dedicated EMG CNN and a dedicated IMU CNN, fuse their outputs (e.g. by summation or
concatenation before the LSTM), rather than feeding a single concatenated tensor into
one CNN per expert.

This is actually a stronger ablation: it tests whether the implicit joint encoding
(single CNN on cat([emg, imu])) of M0 matches the (likely more principled) approach
of learning modality-specific features before combining.

Please CONFIRM with your PI which direction this ablation should go:

  Option A (spec as written — but this IS M0):
    "Concat then route" — a single CNN per expert sees cat([emg, imu]).
    This is identical to M0. Would need a prior separate-encoder M0 to compare against.

  Option B (the informative ablation):
    "Route then concat" — each expert has a separate EMG CNN and IMU CNN;
    outputs are concatenated before the LSTM. The gate routes based on
    cat([emg, imu]) (unchanged context projector).
    This is implemented below and is the comparison the spec seems to intend.

  Option C (EMG-only baseline):
    Drop IMU entirely (use_imu=False). Tests multimodal vs unimodal.
    Simplest to implement since it's just a config flag.

This file implements Option B (separate per-modality expert CNNs) and Option C
(EMG-only) as the two most informative contrasts against M0.
Discuss with your PI and comment out the one you don't want.

Training : Episodic
Evaluation: Episodic (1-shot 3-way)
"""

import os, sys, copy, json, argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config,
    set_seeds, FIXED_SEED, NUM_FINAL_SEEDS,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    RUN_DIR,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import mamlpp_pretrain

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Separate-modality MoE encoder (Option B)
# =============================================================================
# This wraps DeepCNNLSTM_EncoderMOE logic but gives each expert a dedicated
# EMG CNN and a dedicated IMU CNN. The LSTM and head are unchanged.
#
# Forward signature is identical to M0 so mamlpp.py works with zero changes.

class SeparateModalityEncoderMOE(nn.Module):
    """
    Each of the E experts has its own (emg_cnn, imu_cnn) pair.
    Expert outputs are concatenated before the LSTM.

    The context projector still operates on cat([emg, imu]) so routing signal
    is multimodal — only the expert encoding is separate.

    in_ch_emg : C_emg (16)
    in_ch_imu : C_imu (72)
    cnn_out_ch after separate CNNs: emg_out + imu_out
    """

    def __init__(self, config: dict):
        super().__init__()
        from MOE.MOE_encoder import (
            ContextProjector, MOEGate, _build_cnn_block, _init_lstm,
        )

        self.config = config

        C_emg    = config["emg_in_ch"]       # 16
        C_imu    = config.get("imu_in_ch", 72)  # 72
        base_f   = config["cnn_base_filters"]
        n_layers = config.get("cnn_layers", 3)
        k        = config["cnn_kernel"]
        gn_grps  = config["groupnorm_num_groups"]
        lstm_h   = config["lstm_hidden"]
        n_way    = config["n_way"]
        drop     = config.get("dropout", 0.2)
        bidir    = config.get("bidirectional", True)
        head_type = config.get("head_type", "mlp")

        E          = config.get("num_experts", 32)
        ctx_hidden = config.get("MOE_ctx_hidden_dim", 32)
        ctx_out    = config.get("MOE_ctx_out_dim", 32)
        MOE_drop   = config.get("MOE_dropout", 0.05)
        gate_temp  = config.get("MOE_gate_temperature", 0.65)
        top_k      = config.get("MOE_top_k", 9)
        width_mult = config.get("MOE_expert_expand", 1.0)

        self.use_imu     = True   # A9 is always multimodal
        self.num_experts = E
        D = 2 if bidir else 1

        in_ch_joint = C_emg + C_imu   # for context projector

        # ── Context projector (joint, unchanged) ──────────────────────────────
        self.ctx_proj = ContextProjector(
            in_ch=in_ch_joint, hidden_dim=ctx_hidden, out_dim=ctx_out,
            use_conv=True, conv_kernel=k, gn_groups=gn_grps, dropout=MOE_drop,
        )

        # ── Per-expert separate EMG and IMU CNNs ──────────────────────────────
        # Half the base_filters for each modality so total expert width ≈ M0
        emg_base = max(gn_grps, base_f // 2)
        imu_base = max(gn_grps, base_f // 2)
        # Round to multiple of gn_grps
        emg_base = (emg_base // gn_grps) * gn_grps
        imu_base = (imu_base // gn_grps) * gn_grps

        self.emg_experts = nn.ModuleList()
        self.imu_experts = nn.ModuleList()
        for _ in range(E):
            emg_cnn, emg_out_ch = _build_cnn_block(
                C_emg, emg_base, n_layers, k, gn_grps, MOE_drop, width_mult
            )
            imu_cnn, imu_out_ch = _build_cnn_block(
                C_imu, imu_base, n_layers, k, gn_grps, MOE_drop, width_mult
            )
            self.emg_experts.append(emg_cnn)
            self.imu_experts.append(imu_cnn)

        # Recompute out channels (same for all experts)
        _, emg_out_ch = _build_cnn_block(C_emg, emg_base, n_layers, k, gn_grps, drop, width_mult)
        _, imu_out_ch = _build_cnn_block(C_imu, imu_base, n_layers, k, gn_grps, drop, width_mult)
        cnn_out_ch = emg_out_ch + imu_out_ch

        # ── Gate ──────────────────────────────────────────────────────────────
        self.gate = MOEGate(ctx_out, E, top_k=top_k, temperature=gate_temp)

        # ── LSTM ──────────────────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(cnn_out_ch, lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        self.lstm3 = nn.LSTM(lstm_h * D,  lstm_h, batch_first=True, bidirectional=bidir)
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            _init_lstm(lstm)
        self.lstm_dropout = nn.Dropout(drop)
        self.feat_dim = lstm_h * D

        # ── Head ──────────────────────────────────────────────────────────────
        if head_type == "linear":
            self.head = nn.Linear(self.feat_dim, n_way)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.GELU(), nn.Dropout(drop),
                nn.Linear(self.feat_dim // 2, n_way),
            )

    def backbone(self, x_emg, x_imu=None, demographics=None,
                 return_routing: bool = False):
        if x_imu is None:
            raise ValueError(
                "A9 SeparateModalityEncoderMOE requires x_imu — "
                "got None. Check your dataloader config (use_imu must be True)."
            )

        x_joint = torch.cat([x_emg, x_imu], dim=1)   # (B, C_emg+C_imu, T)

        # Routing from joint signal (unchanged vs M0)
        r = self.ctx_proj(x_joint)
        w_hard, w_soft = self.gate(r)                 # (B, E)

        # Per-expert SEPARATE encoding of each modality, then cat
        expert_feats = []
        for e_emg, e_imu in zip(self.emg_experts, self.imu_experts):
            feat_emg = e_emg(x_emg)   # (B, emg_out_ch, T)
            feat_imu = e_imu(x_imu)   # (B, imu_out_ch, T)
            expert_feats.append(torch.cat([feat_emg, feat_imu], dim=1))  # (B, total_ch, T)

        stacked = torch.stack(expert_feats, dim=1)    # (B, E, C, T)
        w_4d    = w_hard.unsqueeze(-1).unsqueeze(-1)
        h       = (stacked * w_4d).sum(dim=1)         # (B, C, T)

        h = h.permute(0, 2, 1)   # (B, T, C)
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
        feat, _ = self.backbone(x_emg, x_imu, demographics)
        return feat


# =============================================================================
# Config builders
# =============================================================================

def build_config_a9_separate() -> dict:
    """Option B: separate per-modality expert CNNs (the informative ablation)."""
    config = make_base_config(ablation_id="A9_separate")
    config["use_imu"] = True   # must be True — this is the multimodal ablation
    # All other MoE/MAML params unchanged from M0
    return config


def build_config_a9_emg_only() -> dict:
    """Option C: EMG-only (drop IMU entirely). Simplest modality ablation."""
    config = make_base_config(ablation_id="A9_emg_only")
    config["use_imu"]  = False
    config["use_MOE"]  = True
    return config


# =============================================================================
# Training loop
# =============================================================================

def run_one_seed(ablation_id: str, seed: int, config: dict,
                 model_builder) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed

    model = model_builder(config)
    model.to(config["device"])
    n_params = count_parameters(model)
    print(f"\n[{ablation_id} | seed={seed}] Parameters: {n_params:,}")

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[{ablation_id} | seed={seed}] Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "seed":             seed,
            "model_state_dict": train_history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
        },
        config,
        tag=f"seed{seed}_best",
    )

    trained_model.load_state_dict(train_history["best_state"])

    test_results = run_episodic_test_eval(
        trained_model, config, tensor_dict_path, config["test_PIDs"]
    )
    print(f"[{ablation_id} | seed={seed}] Test: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")

    return {
        "seed":         seed,
        "best_val_acc": float(best_val_acc),
        "test_results": test_results,
        "n_params":     n_params,
    }


def run_variant(ablation_id: str, config: dict, model_builder, description: str):
    print(f"\n{ablation_id} CONFIG:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    all_seed_results = []
    for seed_idx in range(NUM_FINAL_SEEDS):
        actual_seed = FIXED_SEED + seed_idx
        print(f"\n{'='*70}")
        print(f"[{ablation_id}] Seed {seed_idx+1}/{NUM_FINAL_SEEDS}  (seed={actual_seed})")
        print(f"{'='*70}")
        result = run_one_seed(ablation_id, actual_seed, config, model_builder)
        all_seed_results.append(result)

    test_accs = [r["test_results"]["mean_acc"] for r in all_seed_results]
    summary = {
        "ablation_id":    ablation_id,
        "description":    description,
        "n_params":       all_seed_results[0]["n_params"],
        "seed_results":   all_seed_results,
        "mean_test_acc":  float(np.mean(test_accs)),
        "std_test_acc":   float(np.std(test_accs)),
        "num_seeds":      NUM_FINAL_SEEDS,
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag="summary")

    print(f"\n{'='*70}")
    print(f"[{ablation_id}] FINAL: {summary['mean_test_acc']*100:.2f}% "
          f"± {summary['std_test_acc']*100:.2f}%")
    print(f"{'='*70}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["separate", "emg_only", "both"],
        default="separate",
        help=(
            "separate = Option B (separate per-modality expert CNNs vs M0's joint encoding). "
            "emg_only = Option C (drop IMU entirely). "
            "Confirm which is intended with your PI before running."
        ),
    )
    args = parser.parse_args()

    if args.variant in ("separate", "both"):
        config = build_config_a9_separate()
        run_variant(
            "A9_separate", config,
            model_builder=lambda cfg: SeparateModalityEncoderMOE(cfg),
            description=(
                "A9: Separate modality expert CNNs (each expert has dedicated "
                "EMG CNN + IMU CNN, fused before LSTM). Compare vs M0's joint "
                "single-CNN per expert."
            ),
        )

    if args.variant in ("emg_only", "both"):
        from MOE.MOE_encoder import build_MOE_model
        config = build_config_a9_emg_only()
        run_variant(
            "A9_emg_only", config,
            model_builder=lambda cfg: build_MOE_model(cfg),
            description="A9: EMG-only (no IMU). Unimodal vs multimodal comparison.",
        )


if __name__ == "__main__":
    main()
