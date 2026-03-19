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
    "n_way":    10,   # number of gesture classes

    # ── Participant split ──────────────────────────────────────────────────────
    #####################################################################################################################
    "train_PIDs": [
        "P102","P114","P119","P005","P107","P126","P132","P112",
    # Using a smaller train set so it goes faster. Ig this might make the task easier? Not sure tbh
        "P103","P125","P127","P010","P128","P111","P118",
        "P124","P110","P116","P108","P104","P122","P131","P106","P115"
    ],
    # TODO: Should val and test match train PIDs? Doing in-distribution vs cross-user out-of-distribution...
    #"val_PIDs": ["P102","P114","P119","P005","P107","P126","P132","P112"],
    "val_PIDs":  ["P011","P006","P105","P109"],
    "test_PIDs": ["P008","P004","P123","P121"],  
    # TODO: OOD probe should (for now) test linear probing on the val participants
    #####################################################################################################################

    # ── Trial/repetition split (1-indexed, range 1…10) ───────────────────────
    # "train_reps" and "val_reps" select which of the 10 TRIALS are used.
    # These are NOT class labels.
    "all_rep_indices": list(range(1, 11)),   # all available trial numbers: [1..10]
    "train_reps":      list(range(1, 9)),    # trials 1-8  → training
    "val_reps":        [9, 10],             # trials 9-10 → validation

    # ── Gesture classes (0-indexed) ───────────────────────────────────────────
    "available_gesture_classes": list(range(0, 10)),  # class labels 0…9

    # ── Modality & Dataloader ──────────────────────────────────────────────────
    "use_imu":     True,
    "batch_size":  64,
    "num_workers": 4,

    # ── Augmentation (train only; all aug_ keys match BASE_CONFIG) ────────────
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
    "es_patience":         12,
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
        "gn_groups":     8,
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
        "gn_groups":        8,
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
}
