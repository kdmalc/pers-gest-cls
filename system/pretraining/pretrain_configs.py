# pretrain_configs.py
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Terminology reminder (keep these distinct everywhere in the codebase):
#
#   gesture_class / class_label  : int, 0-indexed,  0 … (n_classes-1)  ← what we predict
#   trial_num / rep_num          : int, 1-indexed,  1 … 10              ← one recording
#
#   "train_reps" / "val_reps"  are TRIAL/REP NUMBERS (1-indexed), NOT class labels.
# ─────────────────────────────────────────────────────────────────────────────

PRETRAIN_CONFIG = {
    # ── Dataset / task ────────────────────────────────────────────────────────
    "n_way":    10,   # number of gesture classes --> NOTE: THIS IS SUPERVISED PRETRAINING, not few-shot learning! Keep this as 10 here!

    # ── Participant split ──────────────────────────────────────────────────────
    "train_PIDs": [
        "P102","P114","P119","P005","P107","P126","P132","P112",
        "P103","P125","P127","P010","P128","P111","P118",
        "P124","P110","P116","P108","P104","P122","P131","P106","P115"
    ],
    "val_PIDs":  ["P011","P006","P105","P109"],
    "test_PIDs": ["P008","P004","P123","P121"],

    # ── Trial/repetition split (1-indexed, range 1…10) ───────────────────────
    "all_rep_indices": list(range(1, 11)),
    "train_reps":      list(range(1, 9)),
    "val_reps":        [9, 10],

    # ── Gesture classes (0-indexed) ───────────────────────────────────────────
    "available_gesture_classes": list(range(0, 10)),

    # ── Modality & Dataloader ──────────────────────────────────────────────────
    "use_imu":     True,
    "batch_size":  64,
    "num_workers": 4,

    # ── Augmentation (train only) ─────────────────────────────────────────────
    "augment":       False,
    "aug_noise_std": 0.05,
    "aug_max_shift": 4,
    "aug_ch_drop":   0.05,

    # ── Training ──────────────────────────────────────────────────────────────
    "num_epochs":          50,
    "learning_rate":       1e-3,
    "weight_decay":        1e-4,
    "dropout":             0.1,
    "label_smooth":        0.0,
    "warmup_epochs":       5,
    "use_scheduler":       True,
    "use_early_stopping":  True,
    "es_patience":         8,
    "es_min_delta":        1e-4,
    "grad_clip":           5.0,
    "use_amp":             False,

    # ── Misc ──────────────────────────────────────────────────────────────────
    "seed":   42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

MODEL_CONFIGS = {
    "MetaCNNLSTM": {
        "model_type":    "MetaCNNLSTM",
        "emg_in_ch":     16,
        "imu_in_ch":     72,
        "use_imu":       PRETRAIN_CONFIG["use_imu"],
        "seq_len":       64,
        "cnn_filters":   32,
        "cnn_kernel":    5,
        "groupnorm_num_groups":     8,
        "lstm_hidden":   32,
        "bidirectional": False,
        "dropout":       PRETRAIN_CONFIG["dropout"],
        "head_type":     "linear",
        "n_way":         PRETRAIN_CONFIG["n_way"],
    },
    "DeepCNNLSTM": {
        "model_type":       "DeepCNNLSTM",
        "emg_in_ch":        16,
        "imu_in_ch":        72,
        "use_imu":          PRETRAIN_CONFIG["use_imu"],
        "seq_len":          64,
        "cnn_base_filters": 32,
        "cnn_layers":       3,
        "cnn_kernel":       5,
        "groupnorm_num_groups":        8,
        "lstm_hidden":      64,
        "bidirectional":    True,
        "dropout":          PRETRAIN_CONFIG["dropout"],
        "head_type":        "mlp",
        "n_way":            PRETRAIN_CONFIG["n_way"],
    },
    "TST": {
        "model_type": "TST",
        "emg_in_ch":  16,
        "imu_in_ch":  72,
        "use_imu":    PRETRAIN_CONFIG["use_imu"],
        "seq_len":    64,
        "patch_len":  8,
        "d_model":    64,
        "n_heads":    4,
        "n_blocks":   3,
        "dropout":    PRETRAIN_CONFIG["dropout"],
        "head_type":  "mlp",
        "n_way":      PRETRAIN_CONFIG["n_way"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MoE Configuration (added to MODEL_CONFIGS at runtime — see build_model_with_moe)
#
# Two placement modes:
#   "encoder" : Each expert IS a full CNN encoder.
#               Context projector reads raw x → routing vector r.
#               Gate weights E expert CNNs; weighted sum fed to LSTM.
#               Good for: exploring whether experts specialise by signal type/user.
#               Cost: E × CNN parameters.  Use moe_expert_expand < 1.0 to offset.
#
#   "middle"  : Shared CNN (identical to baseline — easy pretrain weight transfer).
#               Context projector reads CNN features → r.
#               Gate weights E lightweight MLP experts.
#               Good for: starting from a strong pretrained CNN.
#               Cost: E × small MLP parameters (much cheaper than "encoder").
#
# Recommended starting point: "middle" with 4 experts, moe_top_k=None (dense routing).
# Dense routing is fully differentiable and works well with MAML.
#
# Load-balancing loss: add moe_aux_loss(gate_weights, coeff) to your training loss.
# The coefficient moe_aux_coeff=1e-2 is a good starting point.
# ─────────────────────────────────────────────────────────────────────────────

MOE_CONFIG_DEFAULTS = {
    # ── Placement ─────────────────────────────────────────────────────────────
    "use_MOE":              True,          # Set True to activate MoE
    "MOE_placement":        "encoder",       # "encoder" | "middle"

    # ── Expert count and routing ──────────────────────────────────────────────
    "num_experts":          4,              # Number of expert modules
    "MOE_top_k":            None,           # None=dense (recommended), int=sparse top-k

    # ── Context projector (produces routing signal r) ─────────────────────────
    "MOE_ctx_hidden_dim":   64,             # Context projector hidden layer size
    "MOE_ctx_out_dim":      32,             # Routing vector dimension

    # ── Gate ──────────────────────────────────────────────────────────────────
    "MOE_gate_temperature": 1.0,            # Softmax temp: >1 flatter, <1 sharper

    # ── Expert width (encoder placement only) ─────────────────────────────────
    "MOE_expert_expand":    0.75,           # Each expert CNN is this fraction of baseline width
    #                                       # 1.0 = same width; 0.5 = half; saves params

    # ── Expert MLP width (middle placement only) ──────────────────────────────
    "MOE_mlp_hidden_mult":  1.0,            # MLP hidden = CNN_out_ch * this

    # ── Dropout inside MoE modules ────────────────────────────────────────────
    "MOE_dropout":          0.1,

    # ── Auxiliary load-balancing loss ─────────────────────────────────────────
    "MOE_aux_coeff":        1e-2,           # Scale for Switch Transformer aux loss
    #                                       # Set to 0 to disable
}

# Example: MetaCNNLSTM with middle MoE, 4 experts
MOE_META_MIDDLE = {
    **MODEL_CONFIGS["MetaCNNLSTM"],
    **MOE_CONFIG_DEFAULTS,
    "use_MOE":       True,
    "MOE_placement": "middle",
    "num_experts":   4,
}

# Example: DeepCNNLSTM with encoder MoE, 4 narrower experts
MOE_DEEP_ENCODER = {
    **MODEL_CONFIGS["DeepCNNLSTM"],
    **MOE_CONFIG_DEFAULTS,
    "use_MOE":              True,
    "MOE_placement":        "encoder",
    "num_experts":          4,
    "MOE_expert_expand":    0.75,   # 4 experts × 0.75 width ≈ 3× baseline params
}


HPO_CONFIG = {
    "n_trials":                  50,
    "lr_range":                  [1e-4, 5e-3],
    "wd_range":                  [1e-5, 1e-2],
    "dropout_range":             [0.0, 0.4],
    "label_smooth_range":        [0.0, 0.2],
    "batch_size_choices":        [32, 64, 128],
    "lstm_hidden_choices":       [64, 128, 256],
    "cnn_filters_choices":       [32, 64, 128],
    "cnn_base_filters_choices":  [32, 64],
    "bidirectional_choices":     [True, False],
    "head_type_choices":         ["linear", "mlp"],
    # MoE-specific HPO ranges (used when use_moe=True)
    "MOE_num_experts_choices":   [2, 4, 6, 8],
    "MOE_placement_choices":     ["middle", "encoder"],
    "MOE_ctx_out_dim_choices":   [16, 32, 64],
    "MOE_gate_temp_range":       [0.5, 2.0],
    "MOE_expert_expand_range":   [0.5, 1.5],
    "MOE_aux_coeff_range":       [1e-3, 1e-1],
}