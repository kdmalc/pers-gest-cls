"""
contrastive_config.py

Centralized configuration for the SupCon / Siamese contrastive gesture recognition system.
All hyperparameters live here — no magic numbers scattered across files.

Data facts (from MAML HPO):
  - 32 total users, 24 train / 4 val / 4 test (4-fold CV)
  - 16 EMG channels, 72 IMU channels, seq_len=64
  - 10 gesture classes (labels 0-9)
  - 12-dim demographics vector

Training philosophy:
  - SupCon loss on flat batches (no episodic structure)
  - Encoder f_θ: (B, C, T) → (B, D) → L2-norm → (B, D)
  - Inference: 1-shot prototyping via nearest cosine neighbor

Toggle ARCH_MODE between:
  - 'cnn_lstm'     : CNN feature extraction → BiLSTM temporal → pool → proj head
  - 'cnn_attn'     : CNN feature extraction → Attention pool → proj head (faster, recommended)

Toggle LOSS_MODE between:
  - 'supcon'       : Supervised Contrastive (recommended — rich within-batch positives)
  - 'siamese'      : Classic pairwise Siamese with cosine margin loss (for ablation)
"""

import torch
from pathlib import Path
import os

# ============================================================
# PATHS
# ============================================================
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR  = Path(os.environ.get("RUN_DIR",  "./runs")).resolve()

CONTRASTIVE_CONFIG = {

    # ----------------------------------------------------------
    # PATHS
    # ----------------------------------------------------------
    "code_dir":             str(CODE_DIR),
    "data_dir":             str(DATA_DIR),
    "run_dir":              str(RUN_DIR),
    "tensor_dict_path":     str(CODE_DIR / "dataset" / "segfilt_rts_tensor_dict.pkl"),
    "user_split_json_filepath": str(CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"),
    "checkpoint_dir":       str(RUN_DIR / "contrastive_checkpoints"),

    # ----------------------------------------------------------
    # ARCHITECTURE TOGGLE
    # ----------------------------------------------------------
    "arch_mode":            "cnn_attn",   # RECOMMENDED start --> Other option is... cnn_lstm?

    # ----------------------------------------------------------
    # LOSS TOGGLE
    # ----------------------------------------------------------
    "loss_mode":            "supcon",     # RECOMMENDED start --> Try NT-Xent loss... less human bias (no labels)

    # ----------------------------------------------------------
    # DATA / MODALITY
    # ----------------------------------------------------------
    # TODO: Is IMU support even?
    "use_imu":              False,        # Start EMG-only; ablate IMU later
    "use_demographics":     True,         # FiLM conditioning on demo vector
    "use_film_x_demo":      True,

    "emg_in_ch":            16,
    "imu_in_ch":            72,
    "demo_in_dim":          12,
    "sequence_length":      64,
    "num_classes":          10,           # 10-way gesture set
    "gesture_labels":       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # 0-indexed to match new tensor_dict

    # ----------------------------------------------------------
    # USER & TRIAL SPLITS
    # ----------------------------------------------------------
    "num_total_users":      32,

    # Should val and test match train PIDs? Doing in-distribution vs cross-user out-of-distribution...
    # For now I think we should set them to train...
    ## It actually does very very well if its just an intra-subject split, it gets 100% train and 80% val lol
    "train_PIDs":         [
        "P102","P114","P119","P005","P107","P126","P132","P112",
        "P103","P125","P127","P010","P128","P111","P118",
        "P124","P110","P116","P108","P104","P122","P131","P106","P115"
        ],
    #"val_PIDs":           ["P102","P114","P119","P005","P107","P126","P132","P112"],
    "val_PIDs":           ["P011","P006","P105","P109"],
    "test_PIDs":          ["P008","P004","P123","P121"],  

    #"train_PIDs":           [],           # Populated by apply_fold_to_config()
    #"val_PIDs":             [],
    #"test_PIDs":            [],

    # 1-indexed trial numbers to withhold/train on.
    "train_reps":           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # INTRA: [1, 2, 3, 4, 5, 6, 7, 8],
    # Cross-user episodic eval usually requires all 10 reps to do 1-shot 9-query:
    "val_reps":             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # INTRA: [9, 10],

    # ----------------------------------------------------------
    # CNN ENCODER
    # ----------------------------------------------------------
    "emg_base_cnn_filters": 64,           # First layer width; doubles each layer
    "emg_cnn_layers":       3,
    "imu_base_cnn_filters": 32,           # Why is this different than EMG...
    "imu_cnn_layers":       2,
    "cnn_kernel_size":      5,
    "emg_stride":           1,
    "imu_stride":           1,
    "groupnorm_num_groups": 8,            # GroupNorm groups (must divide filter count)
    "dropout":              0.1,

    # ----------------------------------------------------------
    # TEMPORAL PROCESSING  (only used if arch_mode == 'cnn_lstm')
    # ----------------------------------------------------------
    "use_lstm":             True,
    "lstm_hidden":          128,
    "lstm_layers":          2,
    # TODO: Is GAP used AFTER CNN BEFORE LSTM, or just AFTER LSTM? If its after it doesnt matter
    "use_GlobalAvgPooling": True,         # True=GAP over LSTM outputs; False=concat last hidden

    # ----------------------------------------------------------
    # ATTENTION POOLING  (only used if arch_mode == 'cnn_attn')
    # ----------------------------------------------------------
    "attn_pool_heads":      4,

    # ----------------------------------------------------------
    # DEMOGRAPHICS ENCODER
    # ----------------------------------------------------------
    "demo_emb_dim":         16,

    # ----------------------------------------------------------
    # PROJECTION HEAD  (maps backbone features → contrastive embedding)
    # ----------------------------------------------------------
    "embedding_dim":        128,
    "proj_hidden_dim":      256,          # None → single linear layer

    # ----------------------------------------------------------
    # SUPCON LOSS  (loss_mode == 'supcon')
    # ----------------------------------------------------------
    "supcon_temperature":   0.07,
    "hard_negative_mining": False,        # Start False; ablate on
    "label_hierarchy":      False,        # 4-level: (user,gest) > (user,diff) > (diff,gest) > (diff,diff)

    # ----------------------------------------------------------
    # SIAMESE LOSS  (loss_mode == 'siamese')
    # ----------------------------------------------------------
    "cosine_margin":        0.4,
    "pos_weight":           1.0,

    # ----------------------------------------------------------
    # DATALOADER / BATCH CONSTRUCTION
    # ----------------------------------------------------------
    "batch_construction":   "balanced",   # 'balanced' (recommended) or 'random'
    "samples_per_class":    6,            # M samples per gesture per batch  
    "classes_per_batch":    10,           # How many gesture classes to include per batch
    "num_workers":          8,

    # Validation: 1-shot prototyping accuracy (mimics test-time protocol exactly)
    "val_support_shots":    1,            # k-shot for prototype construction
    "val_query_per_class":  9,            # How many query samples to evaluate per class
    "num_val_episodes":     20,           

    # ----------------------------------------------------------
    # OPTIMIZATION
    # ----------------------------------------------------------
    "optimizer":            "adamw",
    "learning_rate":        1e-3,
    "weight_decay":         1e-4,
    "num_epochs":           100,
    "lr_scheduler":         "cosine",     # 'cosine', 'reduce_on_plateau', or None
    "lr_warmup_epochs":     5,
    "lr_min":               1e-6,         # Cosine annealing minimum

    "use_earlystopping":    True,
    "earlystopping_patience": 12,
    "earlystopping_min_delta": 0.002,

    # ----------------------------------------------------------
    # LINEAR PROBE EVALUATION
    # ----------------------------------------------------------
    # Linear probe: freeze backbone, fit nn.Linear(backbone_dim -> num_classes)
    # with CE loss on training embeddings, eval on val embeddings.
    # Run every epochs_between_linprob epochs and always on the final epoch.
    # None is logged for skipped epochs so log lists stay epoch-index aligned.
    "epochs_between_linprob": 5,          # How often to run the linear probe
    "linprob_epochs":         50,         # How many CE epochs to fit the linear layer
    "linprob_lr":             1e-2,       # LR for the linear probe Adam optimizer

    # ----------------------------------------------------------
    # MISC
    # ----------------------------------------------------------
    "device":               "cuda" if torch.cuda.is_available() else "cpu",
    "seed":                 42,
    "verbose":              True,
    "grad_clip":            5.0,          # Max gradient norm; None to disable
    "log_interval":         100,          # Steps between training log prints
}


def apply_fold_to_config(config: dict, all_splits: list, fold_idx: int) -> dict:
    """Mutates config in-place with the correct user split for this fold."""
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    config["num_train_users"] = len(config["train_PIDs"])
    config["num_val_users"]   = len(config["val_PIDs"])
    config["num_test_users"]  = len(config["test_PIDs"])
    return config


def get_config() -> dict:
    """Returns a fresh copy of the config. Always use this rather than importing the dict directly."""
    import copy
    return copy.deepcopy(CONTRASTIVE_CONFIG)