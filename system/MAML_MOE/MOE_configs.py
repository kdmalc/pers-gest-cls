import copy
from datetime import datetime
import torch
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
NUM_CHANNELS = 16  # Note that ELEC573Net is hardcoded in, doesn't use this...
NUM_TRAIN_GESTURES = 8  # For pretrained models
NUM_FT_GESTURES = 1  # Oneshot finetuning


# ===================== MODEL CONFIG ===========================
basicMOE_config = {
    # ----- Identity / IO -----
    "model_str": "MoEClassifier",          # class name
    "backbone_type": "TinyBackbone",
    "feature_engr": "moments",             # keep if your loaders rely on this
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # PRETRAINING
    "num_epochs": 300,
    "batch_size": 64,
    "learning_rate": 7e-05,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "lr_scheduler_patience": 7,
    "lr_scheduler_factor": 0.2,
    "earlystopping_patience": 10,
    "earlystopping_min_delta": 0.005,
    "num_train_gesture_trials": NUM_TRAIN_GESTURES,
    "num_pretrain_users": 24,

    # ----- Data shape (maps to TinyBackbone) -----
    "num_channels": 16,                    # -> TinyBackbone.in_ch
    "sequence_length": 5,                  # -> TinyBackbone.seq_len
    "time_steps": 1,                       # legacy (kept for compatibility)

    # ----- Embeddings / dims -----
    "emb_dim": 64,                    # -> TinyBackbone.emb_dim and Expert/keys
    "user_emb_dim": 16,                  # -> user embedding dim

    # ----- Task -----
    "num_classes": 10,

    # ----- MoE layout -----
    "num_experts": 6,            # -> number of Expert heads
    "top_k": 2,                        # -> gating sparsity (None for dense)
    "use_user_table": True,      # -> whether to learn user embedding table
    "gate_type": "user_aware",             # ("user_aware" | "feature_only" | "user_only" | "FiLM" | "bilinear" )
    "gate_requires_u_user": True,          # TODO: MANUALLY MAINTAIN THIS! Only False if using feature_only!!!
    "use_u_init_warm_start": True,          
    "gate_dense_before_topk": True,        # softmax then mask (matches code)

    # ----- Expert head (maps to Expert) -----
    "expert_hidden": 64,              # Expert.fc1 hidden = emb_dim
    "expert_dropout": 0.10,           # -> Expert.drop p
    "expert_norm": "layernorm",            # fixed in code; keeping for clarity

    # ----- Prototype / keys -----
    "expert_keys_init_std": 0.1,           # -> nn.Parameter N(0, std^2)

    # ----- Regularization / losses -----
    "label_smooth": 0.05,
    "gate_balance_coef": 0.05,

    # ----- Training / logging -----
    "seed": 17,
    "print_every": 50,
    "user_split_json_filepath": "C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\fixed_user_splits\\24_8_user_splits_RS17.json",
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\MOE\\{timestamp}_MOE",
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\models\\MOE\\{timestamp}_MOE",

    # ----- FINETUNING / PEFT / adaptation -----
    "finetune_strategy": "full",  # full, experts_only, experts_plus_gate --> linear_probing is missing the linear probe...
    "use_dropout_during_peft": False,
    "ft_learning_rate": 0.001,
    "ft_batch_size": 10,
    "num_ft_epochs": 100,
    "ft_weight_decay": 1e-3,
    "ft_lr_scheduler_patience": 7,
    "ft_lr_scheduler_factor": 0.25,
    "ft_earlystopping_patience": 10,
    "ft_earlystopping_min_delta": 0.003,
    "num_ft_gesture_trials": NUM_FT_GESTURES,
    "num_testft_users": 8,
    "save_ft_models": False,
    "reset_ft_layers": False,
    "alt_or_seq_MOE_user_emb_ft": "sequential", 

    # MISC
    "cluster_iter_str": 'Iter18',  # Not used currently... idk how I would/should combine MOE with clustering... probably doing MOE in place of clustering
    "use_earlystopping": True,
    "verbose": False,
    "log_each_pid_results": False, 
    'timestamp': timestamp

    # ===== Legacy CNN/LSTM keys kept for reference (NOT IMPLEMENTED) =====
    # "use_batch_norm": False,             # NOT IMPLEMENTED in TinyBackbone/Expert ---> For FTing I think the batches are too small to use this
    # "conv_layers": [[32, 3, 1]],         # NOT IMPLEMENTED
    # "fc_layers": [256],                  # NOT IMPLEMENTED (backbone uses fixed MLP)
    # "fc_dropout": 0.3,                   # NOT IMPLEMENTED
    # "cnn_dropout": 0.0,                  # NOT IMPLEMENTED
    # "dense_cnnlstm_dropout": 0.1,        # NOT IMPLEMENTED
    # "use_dense_cnn_lstm": True,          # NOT IMPLEMENTED
    # "use_layerwise_maxpool": False,      # NOT IMPLEMENTED
    # "pooling_layers": [False],           # NOT IMPLEMENTED
    # "lstm_num_layers": 0,                # NOT IMPLEMENTED
    # "lstm_hidden_size": 0,               # NOT IMPLEMENTED
    # "lstm_dropout": 0.0,                 # NOT IMPLEMENTED
    # "padding": 0,                        # NOT USED
}

# HPO EMG Moments Only:
# Best trial: Value: 0.7833333333333333
emg_moments_only_MOE_config = {
    "multimodal": False, 
    "emb_dim": 64, 
    "user_emb_dim": 16, 
    "num_experts": 8, 
    "top_k": 3, 
    "gate_type": "user_only", 
    "head_type": "cosine", 
    "init_tau": 5.237961277700698, 
    "expert_dropout": 0.2992372749347322, 
    "label_smooth": 0.12408221619309728, 
    "gate_balance_coef": 0.09764902136021002, 
    "finetune_strategy": "experts_only", 
    "use_dropout_during_peft": False,  
    "alt_or_seq_MOE_user_emb_ft": "alternating", 

    #"pre_lr": 8.041599118997163e-05, 
    #"pre_wd": 2.713989100551137e-06, 
    #"pre_opt": "adamw", 
    #"pre_sched_factor": 0.1, 
    #"pre_sched_pat": 8, 
    #"pre_es_pat": 9, 
    #"pre_es_delta": 0.004944018565207135, 
    "learning_rate": 8.041599118997163e-05, 
    "weight_decay": 2.713989100551137e-06, 
    "optimizer": "adamw", 
    "lr_scheduler_factor": 0.1, 
    "lr_scheduler_patience": 8, 
    "earlystopping_patience": 9, 
    "earlystopping_min_delta": 0.004944018565207135, 

    #"ft_lr": 0.0012733525226279476, 
    #"ft_wd": 1.0729603706519642e-05, 
    #"ft_sched_factor": 0.1, 
    #"ft_sched_pat": 4, 
    #"ft_es_pat": 9, 
    #"ft_es_delta": 0.007732449511096917,
    "ft_learning_rate": 0.0012733525226279476, 
    "ft_weight_decay": 1.0729603706519642e-05, 
    "ft_lr_scheduler_factor": 0.1, 
    "ft_lr_scheduler_patience": 4, 
    "ft_earlystopping_patience": 9, 
    "ft_earlystopping_min_delta": 0.007732449511096917,

    # Not included in that Optuna run but used in other parts of my code
    ## Maybe I don't need these if the Optuna ran fine
    ## How was this running without these???
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
    "cluster_iter_str": None,
    "feature_engr": "None",
    "time_steps": 1,
    "sequence_length": 64,
    "num_train_gesture_trials": 9, 
    "num_ft_gesture_trials": 1,
    "num_pretrain_users": 24, 
    "num_testft_users": 4,  # TODO: Is this true? Did I do 4 or 5 or 6?
    "padding": 0, 
    "batch_size": 64,
    "use_batch_norm": False,
    "timestamp": timestamp,
    "fc_dropout": 0.3,
    "num_epochs": 100,
    "num_classes": 10,
    "ft_batch_size": 10,
    "num_ft_epochs": 50,
    "use_earlystopping": True,
    "reset_ft_layers": False, 
    "verbose": False,
    "num_total_users": 32, 

    # There's also nothing about validation data splits usage here...
    # This is sort of hardcoded? I am not sure where/how it is using testing dataloaders... I guess I dont supply it any val loaders but that isn't a config thing
}