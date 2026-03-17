# config.py
import torch

PRETRAIN_CONFIG = {
    # ── Meta-Learning / Dataset Params ──
    "n_classes": 10,
    "n_way": 10,
    "k_shot": 1,
    
    # ── User Split ──
    "train_PIDs": [
        "P102","P114","P119","P005","P107","P126","P132","P112",
        "P103","P125","P127","P010","P128","P111","P118",
        "P124","P110","P116","P108","P104","P122","P131","P106","P115"
    ],
    "val_PIDs": ["P011","P006","P105","P109"],
    "test_PIDs": ["P008","P004","P123","P121"],

    # ── Repetition Split (Assuming 1-indexed) ──
    "train_reps": [1, 2, 3, 4, 5, 6, 7, 8],
    "val_reps": [9, 10],

    # ── Modality & Dataloader ──
    "use_imu": True,
    "batch_size": 64,
    "num_workers": 4,

    # ── Augmentation ──
    "augment": False,
    "noise_std": 0.05,
    "max_shift": 4,
    "ch_drop_prob": 0.05,

    # ── Training ──
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "label_smooth": 0.0,
    "warmup_epochs": 5,
    "use_scheduler": True,
    "use_early_stopping": True,
    "es_patience": 12,
    "es_min_delta": 1e-4,
    "grad_clip": 5.0,
    # TODO: Idk if this should be True or not... currently it is...
    "use_amp": False,  #torch.cuda.is_available(),
}

MODEL_CONFIGS = {
    "MetaCNNLSTM": {
        "model_type": "MetaCNNLSTM",
        "emg_in_ch": 16,
        "imu_in_ch": 72,
        "use_imu": PRETRAIN_CONFIG["use_imu"],
        "seq_len": 64,
        "cnn_filters": 32,
        "cnn_kernel": 5,
        "gn_groups": 8,
        "lstm_hidden": 32,
        "bidirectional": False,
        "dropout": PRETRAIN_CONFIG["dropout"],
        "head_type": "linear",
        "n_way": PRETRAIN_CONFIG["n_way"],
    },
    "DeepCNNLSTM": {
        "model_type": "DeepCNNLSTM",
        "emg_in_ch": 16,
        "imu_in_ch": 72,
        "use_imu": PRETRAIN_CONFIG["use_imu"],
        "seq_len": 64,
        "cnn_base_filters": 32,
        "cnn_layers": 3,
        "cnn_kernel": 5,
        "gn_groups": 8,
        "lstm_hidden": 64,
        "bidirectional": True,
        "dropout": PRETRAIN_CONFIG["dropout"],
        "head_type": "mlp",
        "n_way": PRETRAIN_CONFIG["n_way"],
    },
    "TST": {
        "model_type": "TST",
        "emg_in_ch": 16,
        "imu_in_ch": 72,
        "use_imu": PRETRAIN_CONFIG["use_imu"],
        "seq_len": 64,
        "patch_len": 8,
        "d_model": 64,
        "n_heads": 4,
        "n_blocks": 3,
        "dropout": PRETRAIN_CONFIG["dropout"],
        "head_type": "mlp",
        "n_way": PRETRAIN_CONFIG["n_way"],
    }
}

HPO_CONFIG = {
    "n_trials": 50,
    "lr_range": [1e-4, 5e-3],
    "wd_range": [1e-5, 1e-2],
    "dropout_range": [0.0, 0.4],
    "label_smooth_range": [0.0, 0.2],
    "batch_size_choices": [32, 64, 128],
    "lstm_hidden_choices": [64, 128, 256],
    "cnn_filters_choices": [32, 64, 128],
    "cnn_base_filters_choices": [32, 64],
    "bidirectional_choices": [True, False],
    "head_type_choices": ["linear", "mlp"]
}

# EXAMPLE:
#"learning_rate": trial.suggest_float("learning_rate", *HPO_CONFIG["lr_range"], log=True),

##############################################################
# OLD CONFIG FILES
##############################################################

# ─────────────────────────────────────────────────────────────────────────────
# Default configs (good starting point before HPO)
# ─────────────────────────────────────────────────────────────────────────────

OLD_DEFAULT_CONFIGS = {
    "MetaCNNLSTM": {
        "model_type":     "MetaCNNLSTM",
        "emg_in_ch":      16,
        "imu_in_ch":      72,
        "use_imu":        False,
        "seq_len":        64,
        "cnn_filters":    32,
        "cnn_kernel":     5,  # NOTE: I have no idea what this should be for us, since we do not have putative MUAPs...
        "gn_groups":      8,
        "lstm_hidden":    32,
        "bidirectional":  False,     # Meta paper uses unidirectional
        "dropout":        0.0,
        "head_type":      "linear",  # matches Meta paper
        "n_way":          10,
    },
    "DeepCNNLSTM": {
        "model_type":     "DeepCNNLSTM",
        "emg_in_ch":      16,
        "imu_in_ch":      72,
        "use_imu":        False,
        "seq_len":        64,
        "cnn_base_filters": 32,
        "cnn_layers":     3,
        "cnn_kernel":     5,
        "gn_groups":      8,
        "lstm_hidden":    64,
        "bidirectional":  True,
        "dropout":        0.0,
        "head_type":      "mlp",
        "n_way":          10,
    },
    "TST": {
        "model_type":     "TST",
        "emg_in_ch":      16,
        "imu_in_ch":      72,
        "use_imu":        False,
        "seq_len":        64,
        "patch_len":      8,         # 64/8 = 8 patches
        "d_model":        64,        # small for your dataset size
        "n_heads":        4,
        "d_ff":           256,
        "n_blocks":       4,
        "dropout":        0.2,
        "head_type":      "mlp",
        "n_way":          10,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Default config reference
# ─────────────────────────────────────────────────────────────────────────────

OLD_PRETRAIN_DEFAULT_CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    # NOTE: For supervised learning: train and val PIDs SHOULD MATCH! The val set is the val gesture split, NOT a user split!!
    "train_PIDs":   ["P102","P114"],        
            #["P102","P114","P119","P005","P107","P126","P132","P112",
            #"P103","P125","P127","P010","P128","P111","P118",
            #"P124","P110","P116","P108","P104","P122","P131","P106","P115"],
    "val_PIDs":     ["P102","P114"],        
            #["P011","P006","P105","P109"],
    "train_reps":  [1, 2, 3, 4, 5, 6, 7, 8],   # These are REPITIONS NOT CLASSES! So we should do a split here
    "val_reps":[9, 10],
    "use_imu":              False,
    "batch_size":           64,
    "num_workers":          1,
    # ── Augmentation ──────────────────────────────────────────────────────
    "augment":              False,
    "noise_std":            0.05,  # TODO: Is this way too big?? What are our EMG magnitudes?
    "max_shift":            4,
    "ch_drop_prob":         0.05,
    # ── Training ──────────────────────────────────────────────────────────
    "num_epochs":           50,
    "learning_rate":        1e-3,
    "optimizer":            "adamw",
    "weight_decay":         1e-4,
    "label_smooth":         0.0,
    "grad_clip":            10.0,
    "warmup_epochs":        5,
    "use_scheduler":        True,
    "use_amp":              False,
    # ── Early stopping ────────────────────────────────────────────────────
    "use_early_stopping":   True,
    "es_patience":          12,
    "es_min_delta":         1e-4,
    # ── Device ────────────────────────────────────────────────────────────
    "device":               "cuda",
}

