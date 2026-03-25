N_TRIALS = 1
FIXED_SEED = 42

import os
import argparse
import copy, json, time
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend
import random
from pathlib import Path

# Allow numpy scalars to be loaded in weights_only mode
#torch.serialization.add_safe_globals([np.core.multiarray.scalar])
#torch.serialization.add_safe_globals([np.scalar])
# Didnt work. Just suppress the warning instead:
import warnings
# Suppress the specific warning about weights_only=False
warnings.filterwarnings(
    "ignore", 
    message=".*weights_only=False.*", 
    category=UserWarning
)
# Sometimes it is cast as a FutureWarning depending on the torch version
warnings.filterwarnings(
    "ignore", 
    message=".*weights_only=False.*", 
    category=FutureWarning
)

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# env -> Path objects
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR = Path(os.environ.get("RUN_DIR", "./")).resolve()
print(f"CODE_DIR: {CODE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RUN_DIR: {RUN_DIR}")

# === SAVING (to SCRATCH) ===
results_save_dir = RUN_DIR
models_save_dir  = RUN_DIR
results_save_dir.mkdir(parents=True, exist_ok=True)
models_save_dir.mkdir(parents=True, exist_ok=True)

# === LOADING (from SCRATCH data bucket) ===
user_split_json_filepath = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"
def apply_fold_to_config(config, all_splits, fold_idx):
    """Mutates config in-place to set train/val/test PIDs for the given fold."""
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    #config["num_pretrain_users"] = len(config["train_PIDs"])
    #config["num_testft_users"] = len(config["val_PIDs"])

from system.MAML_MOE.mamlpp import *
from system.MAML_MOE.maml_data_pipeline import get_maml_dataloaders
from system.MAML_MOE.shared_maml import *
from system.MAML_MOE.MOE_CNN_LSTM import *

from system.pretraining.pretrain_models import build_model
from system.pretraining.contrastive_net.contrastive_encoder import ContrastiveGestureEncoder

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

#############################################################

def inject_model_config(config: dict, model_type: str):
    """
    Injects the exact architecture parameters used during pretraining.
    These MUST match pretrain_configs.py or the weights will fail to load.
    """
    config["model_type"] = model_type
    config["sequence_length"] = 64
    config['emg_in_ch'] = 16
    config['imu_in_ch'] = 72
    config['demo_in_dim'] = 12
    config["use_imu"] = True 

    # TODO: I dont think I can call trial to HPO these values?
    ## Architecture values cannot be HPOd (need to match the pretrained model)
    ## Other things (learning rate and such) can be HPOd... not totally sure if NONE of these values need to be HPOd...

    if model_type == "MetaCNNLSTM":
        config.update({
            # emg_base_cnn_filters, imu_base_cnn_filters
            # cnn_kernel_size, groupnorm_num_groups
            "cnn_filters": 32, "emg_cnn_layers": 1, "imu_cnn_layers": 1,
            "cnn_kernel": 5, "gn_groups": 8,
            "lstm_hidden": 32, "lstm_layers": 1, "bidirectional": False,
            "head_type": 'linear',
        })
    elif model_type == "DeepCNNLSTM":
        config.update({
            #"emg_base_cnn_filters": 32, "emg_cnn_layers": 3,
            #"imu_base_cnn_filters": 32, "imu_cnn_layers": 3,\
            "cnn_base_filters": 32, "cnn_layers": 3,
            #"cnn_kernel_size": 5, "groupnorm_num_groups": 8,
            "cnn_kernel": 5, "gn_groups": 8,
            "lstm_hidden": 64, "lstm_layers": 3, "bidirectional": True,
            'head_type': 'mlp',
        })
    elif model_type == "TST":
        config.update({
            "patch_len": 8, "d_model": 64, "n_heads": 4, "n_blocks": 3,
        })
    elif model_type == "ContrastiveNet":
        config.update({"arch_mode": "cnn_attn",})
        config.update({
            #"proj_hidden": 128, "proj_out": 64,  # --> Not sure what proj_out is...
        
            "train_reps":           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # INTRA: [1, 2, 3, 4, 5, 6, 7, 8],
            "val_reps":             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # INTRA: [9, 10],
            # CNN ENCODER
            "emg_base_cnn_filters": 64,           # First layer width; doubles each layer
            "emg_cnn_layers":       3,
            "imu_base_cnn_filters": 32,           # Why is this different than EMG...
            "imu_cnn_layers":       2,
            "cnn_kernel_size":      5,
            "emg_stride":           1,
            "imu_stride":           1,
            "groupnorm_num_groups": 8,            # GroupNorm groups (must divide filter count)
            # TEMPORAL PROCESSING  (only used if arch_mode == 'cnn_lstm')
            "use_lstm":             config['arch_mode'] == 'cnn_lstm',  # TODO: So what happens when this is true but arch_mode is set to attn...
            "lstm_hidden":          128,
            "lstm_layers":          2,
            # In Contrastive: GAP used AFTER LSTM
            "use_GlobalAvgPooling": True,         # True=GAP over LSTM outputs; False=concat last hidden
            # ATTENTION POOLING  (only used if arch_mode == 'cnn_attn')
            "attn_pool_heads":      4,
            # PROJECTION HEAD  (maps backbone features → contrastive embedding)
            "embedding_dim":        128,
            "proj_hidden_dim":      256,          # None → single linear layer

            # SUPCON LOSS SPECIFC (loss_mode == 'supcon') --> No effect / not called with MAML??
            #"supcon_temperature":   0.07,         # NOTE: This isn't changed (is it even used?) during MAML training right? This is SupConLoss (pretraining) specific right?
            #"hard_negative_mining": False,        # Start False; ablate on
            #"label_hierarchy":      False,        # 4-level: (user,gest) > (user,diff) > (diff,gest) > (diff,diff)
            # SIAMESE LOSS  (loss_mode == 'siamese')
            #"cosine_margin":        0.4,
            #"pos_weight":           1.0,
            # DATALOADER / BATCH CONSTRUCTION
            #"batch_construction":   "balanced",   # 'balanced' (recommended) or 'random'
            #"samples_per_class":    6,            # M samples per gesture per batch  
            #"classes_per_batch":    10,           # How many gesture classes to include per batch
            # Validation: 1-shot prototyping accuracy (mimics test-time protocol exactly)
            #"val_support_shots":    1,            # k-shot for prototype construction
            #"val_query_per_class":  9,            # How many query samples to evaluate per class

            "num_val_episodes":     20,           
            # OPTIMIZATION
            "lr_scheduler":         "cosine",     # 'cosine', 'reduce_on_plateau', or None
            "lr_warmup_epochs":     5,
            "lr_min":               1e-6,         # Cosine annealing minimum
            # LINEAR PROBE EVALUATION
            #"epochs_between_linprob": 5,          # How often to run the linear probe
            #"linprob_epochs":         50,         # How many CE epochs to fit the linear layer
            #"linprob_lr":             1e-2,       # LR for the linear probe Adam optimizer
            # MISC
            "grad_clip":            5.0,          # Max gradient norm; None to disable
            "log_interval":         100,          # Steps between training log prints
        })
    #elif model_type == "MOECNNLSTM":
    else:
        print("Falling back to old MOE dynamic config (this may not be supported...)")

        # TODO: This is the original network... I dont think it is really support right now...
        # CNN Width & Depth
        config["emg_base_cnn_filters"] = 64
        config["imu_base_cnn_filters"] = 64
        config["emg_cnn_layers"] = 2
        config["imu_cnn_layers"] = 2
        config["cnn_kernel_size"] = 3
        # LSTM
        config["use_lstm"] = True 
        config["lstm_hidden"] = 128
        config["lstm_layers"] = 2
    
    return config

# ===================== OPTUNA TUNING SCRIPT =====================
# NOTE: This now takes model_type as a required input...
def build_model_from_trial(trial, model_type, base_config=None):
    config = copy.deepcopy(base_config) if base_config else {}

    # 1. Inject Architecture Constants
    config = inject_model_config(config, model_type)

    # === Task Setup ===
    # NOTE: Running 1-shot 3-way for now. Final will be 1-shot 10-way... (harder...)
    config["n_way"] = 3  
    config["k_shot"] = 1  
    config["q_query"] = 9
    config["num_classes"] = 10

    config["feature_engr"] = "None"

    config["NOTS"] = True
    if config["NOTS"]==False:
        #config["emg_imu_pkl_full_path"] = 'C:\\Users\\kdmen\\Box\\Yamagami Lab\\Data\\Meta_Gesture_Project\\filtered_datasets\\metadata_IMU_EMG_allgestures_allusers.pkl'
        config["pwmd_xlsx_filepath"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\Biosignal gesture questionnaire for participants with disabilities.xlsx"
        config["pwoutmd_xlsx_filepath"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\Biosignal gesture questionnaire for participants without disabilities.xlsx"
        config["dfs_save_path"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\meta-learning-sup-que-ds\\"
        config["dfs_load_path"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\meta-learning-sup-que-ds\\"
        config["user_split_json_filepath"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\system\\fixed_user_splits\\4kfcv_splits_shared_test.json"
        config["results_save_dir"] = f"C:\\Users\\kdmen\\Repos\\pers-gest-cls\\system\\results\\local_{timestamp}"
        config["models_save_dir"] = f"C:\\Users\\kdmen\\Repos\\pers-gest-cls\\system\\models\\local_{timestamp}"
    elif config["NOTS"]==True:
        ## SAVING
        config["user_split_json_filepath"] = user_split_json_filepath
        config["results_save_dir"] = results_save_dir
        config["models_save_dir"] = models_save_dir
        ## Mutlimodal LOADING
        config["emg_imu_pkl_full_path"] = f"{CODE_DIR}//dataset//filtered_datasets//metadata_IMU_EMG_allgestures_allusers.pkl" 
        
        config["pwmd_xlsx_filepath"] = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants with disabilities.xlsx"
        config["pwoutmd_xlsx_filepath"] = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants without disabilities.xlsx"
        
        config["dfs_save_path"] = f"{CODE_DIR}/dataset//"
        config["dfs_load_path"] = f"{CODE_DIR}/dataset/meta-learning-sup-que-ds//"

    # DEBUG
    config["track_gradient_alignment"] = False
    config["verbose"] = False
    config['gradient_clip_max_norm'] = 10.0  # Allegedly CFinn uses 5-10
    config['num_eval_episodes'] = 10
    config['debug_one_user_only'] = False
    config['debug_one_episode'] = False
    config['debug_five_episodes'] = False
    if config['debug_one_episode']:
        config["meta_batchsize"] = 1
    elif config['debug_five_episodes']:
        config["meta_batchsize"] = 5
    else:
        config["meta_batchsize"] = trial.suggest_categorical("meta_batchsize", [4, 8, 16, 24, 32])  # Meta learning batch size, ie number of episodes per batch (this is handled via looping NOT in the dataloaders since sizes may not match bewteen episodes)

    # === MAML Core Hyperparameters ===
    config["maml_inner_steps"] = trial.suggest_categorical("maml_inner_steps", [2, 3, 5, 7])
    config["maml_inner_steps_eval"] = trial.suggest_categorical("maml_inner_steps_eval", [10, 15, 20, 30])
    
    config["maml_alpha_init"] = trial.suggest_float("maml_alpha_init", 1e-4, 1e-1, log=True)
    config["maml_alpha_init_eval"] = trial.suggest_float("maml_alpha_init_eval", 1e-4, 1e-2, log=True)
    config["learning_rate"] = trial.suggest_float("outer_lr", 1e-5, 1e-2, log=True)
    config["weight_decay"] = trial.suggest_float("wd", 1e-6, 1e-4, log=True)

    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["use_batch_norm"] = False
    config["groupnorm_num_groups"] = trial.suggest_categorical("groupnorm_num_groups", [4, 8])
    config["dropout"] = 0.1 

    config['emg_stride'] = 1  
    config['imu_stride'] = 1  
    config["padding"] = 0 

    # TODO: Is this GAP after the CNN or after the LSTM...
    config["use_GlobalAvgPooling"] = trial.suggest_categorical("use_GlobalAvgPooling", [True, False])

    # === Finetuning / Transfer Learning Strategy ===
    config["finetuning_approach"] = trial.suggest_categorical("finetuning_approach", ["full"])  #, "anil", "frozen_backbone"])

    config["use_pretrained"] = True 
    config["best_or_last_pretr"] = trial.suggest_categorical("best_or_last_pretr", ["best", "last"])

    # === Multimodal & Conditioning (Keeping these if you still use FiLM/Demo heads) ===
    # NOTE: Turning demographics off (wasnt used in pretraining)
    config["multimodal"] = True  # TODO: I dont know if this gets used at all anymore...
    config["use_imu"] = True 
    config["use_demographics"] = False
    config["use_film_x_demo"] = False  #trial.suggest_categorical("use_film_x_demo", [True, False])
    config["FILM_on_context_or_demo"] = 'context' 
    config["context_emb_dim"] = trial.suggest_categorical("context_emb_dim", [8, 16, 32, 64])
    #config["demo_emb_dim"] = trial.suggest_categorical("demo_emb_dim", [8, 16, 32, 64])
    config["context_pool_type"] = trial.suggest_categorical("context_pool_type", ['mean', 'attn'])  

    # === MoE (Mixture of Experts) ===
    # Set use_MOE to trial.suggest_categorical if you want Optuna to decide
    config["use_MOE"] = False 
    if config["use_MOE"]:
        config["num_experts"] = trial.suggest_int("num_experts", 4, 8)
        config["top_k"] = trial.suggest_int("top_k", 2, 4)
        config["gate_type"] = "context_feature_demo"
        config["expert_architecture"] = "MLP"

    config["use_label_shuf_meta_aug"] = False  # TODO: This probably should be turned back on?
    config["num_epochs"] = 50 
    config["episodes_per_epoch_train"] = trial.suggest_categorical("episodes_per_epoch_train", [250, 500])
    config["label_smooth"] = 0.0

    config["num_total_users"] = 32  # TODO: Not sure if this is still used

    config["maml_gesture_classes"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # NOTE: THIS IS GESTURE CLASS
    config["target_trial_indices"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # NOTE: THIS IS GESTURE TRIAL/REPETITION NUM

    # Pretraining optim
    config["optimizer"]          = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
    config["use_earlystopping"] = True
    config["lr_scheduler_factor"]= 0.1  #trial.suggest_categorical("pre_sched_factor", [0.1, 0.2])
    config["lr_scheduler_patience"]= 6  #trial.suggest_int("pre_sched_pat", 4, 10)
    config["earlystopping_patience"]= 8 #trial.suggest_int("pre_es_pat", 6, 14)
    config["earlystopping_min_delta"]= 0.005 #trial.suggest_float("pre_es_delta", 0.001, 0.01)

    # ADDING MAML SPECIFIC
    config["meta_learning"] = True
    config["num_workers"] = 8  # This is the dataloader, something about how many processes the CPU can use (more is faster generally)

    # MAML++
    # MULTI STEP LOSS
    config["use_maml_msl"] = trial.suggest_categorical("use_maml_msl", [True, False, "hybrid"])                              # MSL (multi-step loss) on
    if config["use_maml_msl"] == "hybrid":
        config["maml_msl_num_epochs"] = trial.suggest_int("maml_msl_num_epochs", 1, 40)  # Also note that currently the max num_epochs is 40 (plus we use ES so may not even hit this)
    # Theoretically this should be even be used, but just in case...
    elif config["use_maml_msl"] == True:
        config["maml_msl_num_epochs"] = 1000000  # Arbitrarily large to never trigger and turn MSL off
    elif config["use_maml_msl"] == False:
        config["maml_msl_num_epochs"] = 0
    # OPTIMIZATION ORDER
    # NOTE: CuDNN doesnt support second order. So either just use first order OR wrap a context manager around EVERY forward pass that disables CuDNN flags.
    config["maml_opt_order"] = "first"  #trial.suggest_categorical("maml_opt_order", ["first", "second", "hybrid"])                         # enables second-order when DOA switches on
    if config["maml_opt_order"] == "hybrid":
        config["maml_first_order_to_second_order_epoch"] = trial.suggest_int("maml_first_order_to_second_order_epoch", 5, 40)      # DOA threshold (epochs <= this are first-order)
    # Theoretically this should be even be used, but just in case...
    elif config["maml_opt_order"] == "first":
        config["maml_first_order_to_second_order_epoch"] = 1000000  # Arbitrarily large to never trigger and switch to second
    elif config["maml_opt_order"] == "second":
        config["maml_first_order_to_second_order_epoch"] = 0  # Do second the whole time
    # LSLR
    config["maml_use_lslr"] = True
    # MISC  
    config["enable_inner_loop_optimizable_bn_params"] = False  # by default, do NOT adapt BN in inner loop --> I should not be using BN at all AFAIK
    config["use_cosine_outer_lr"] = False                       # This is cosine-based lr annealing... is this in addition to my lr scheduler....
    config["use_lslr_at_eval"] = False                         # set True if you want to use learned per-parameter step sizes at eval

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    if model_type in ["MetaCNNLSTM", "DeepCNNLSTM", "TST"]:
        model = build_model(config)
    elif model_type == "MOE":
        model = MultimodalCNNLSTMMOE(config)
    elif model_type == "ContrastiveNet": 
        model = ContrastiveGestureEncoder(config)
    else:
        raise ValueError(f"model_type {model_type} unknown!")
    
    # =========================================================================
    # ################### PRETRAINED WEIGHT LOADING ######################### #
    # =========================================================================
    if config["use_pretrained"]:
        print(f"--> Loading pretrained weights for {model_type}...")

        if config["NOTS"]: # Linux uses forward slashes
            pretrain_path = r"/projects/my13/kai/meta-pers-gest/pers-gest-cls/pretrain_outputs/checkpoints/"
        else: # Windows uses back slashes
            pretrain_path = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\pretrain_outputs\\checkpoints\\"
        

        if model_type == "MetaCNNLSTM":
            load_path = f"{pretrain_path}MetaCNNLSTM_03232026_170503_{config['best_or_last_pretr']}.pt"
        elif model_type == "DeepCNNLSTM":
            load_path = f"{pretrain_path}DeepCNNLSTM_03232026_165043_{config['best_or_last_pretr']}.pt"
        elif model_type == "TST":
            load_path = f"{pretrain_path}TST_03232026_163527_{config['best_or_last_pretr']}.pt"
        elif model_type == "MOE":
            load_path = None  # There is no pretrained MOE-CNN-LSTM model!!
        elif model_type == "ContrastiveNet": 
            load_path = f"{pretrain_path}ContrastiveNet_{config['arch_mode'][-4:]}_20260325_1558_{config['best_or_last_pretr']}.pt"
        else:
            raise ValueError("Unknown model_type!")
        
        try:
            # Load the full checkpoint
            checkpoint = torch.load(load_path, map_location=config["device"], weights_only=False)
            # Extract just the weights
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                # Fallback in case some files are just the weights
                state_dict = checkpoint

            # Filter out the classification/projection head so we only load the backbone
            filtered_dict = {k: v for k, v in state_dict.items() if "head" not in k and "projector" not in k}
            
            model_dict = model.state_dict()
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"--> Successfully loaded backbone weights from: {load_path}")
            
        except FileNotFoundError:
            print(f"\n#################################################################")
            print(f"WARNING: Pretrained weight file not found at {load_path}. ")
            print(f"Using random initialization instead!")
            print(f"#################################################################\n")
    # =========================================================================
    # ####################################################################### #
    # =========================================================================

    model.to(config["device"])
    return model, config


#############################################################
# ---------- Load splits once ----------
with open(user_split_json_filepath, "r") as f:
    ALL_SPLITS = json.load(f)
NUM_FOLDS = 2  

BASE_CONFIG = {} 

def objective(trial, model_type):
    """Optuna objective wrapped to accept model_type."""
    fold_mean_accs = []
    all_fold_user_accs = []      
    pretrain_val_accs = []       

    for fold_idx in range(NUM_FOLDS):
        fold_start_time = time.time()

        print("=" * 80)
        print(f"[Trial {trial.number}] Starting fold {fold_idx + 1}/{NUM_FOLDS} for model {model_type}")
        print("=" * 80)

        # ---- Build model + config for this trial/fold ----
        model, config = build_model_from_trial(trial, model_type, base_config=BASE_CONFIG)

        if config["device"].type == "cpu":
            print("HPO is happening on the CPU! Probably ought to switch to GPU!")

        apply_fold_to_config(config, ALL_SPLITS, fold_idx)

        # ---- Data Loading ----
        tensor_dict_path = os.path.join(config["dfs_load_path"], f"segfilt_rts_tensor_dict.pkl")
        episodic_train_loader, episodic_val_loader = get_maml_dataloaders(
            config,
            tensor_dict_path=tensor_dict_path,
        )

        # ---- MAML Pretrain ----
        ## This is different from actual pretraining. This is "pretraining" in the sense that novel users get their own adaptation phase...
        pretrained_model, pretrain_res_dict = mamlpp_pretrain(
            model,
            config,
            episodic_train_loader,
            episodic_val_loader=episodic_val_loader,
        )
        best_val_acc = pretrain_res_dict["best_val_acc"]
        best_state   = pretrain_res_dict["best_state"]
        pretrain_val_accs.append(float(best_val_acc))

        print(f"[Trial {trial.number} | Fold {fold_idx}] Pretraining done. Best val acc = {best_val_acc:.4f}")

        model_filename = f"trial_{trial.number}_fold_{fold_idx}_best.pt"
        save_path = os.path.join(models_save_dir, model_filename)
        torch.save({
            'trial_num': trial.number,
            'fold_idx': fold_idx,
            'model_state_dict': best_state,
            'config': config,
            'best_val_acc': best_val_acc, 
            'train_loss_log': pretrain_res_dict["train_loss_log"], 
            'train_acc_log': pretrain_res_dict["train_acc_log"],
            'val_loss_log': pretrain_res_dict["val_loss_log"],
            'val_acc_log': pretrain_res_dict["val_acc_log"]
        }, save_path)
        print(f"Model permanently saved to {save_path}")

        # --------- Finetuning / Adaptation per Novel user ---------
        model.load_state_dict(best_state)
        user_metrics = defaultdict(list)
        
        for batch in episodic_val_loader:
            user_id = batch['user_id']
            support_set = batch['support']
            query_set = batch['query']
            val_metrics = mamlpp_adapt_and_eval(model, config, support_set, query_set)
            user_metrics[user_id].append(val_metrics["acc"])

        all_user_means = []
        for user_id, accs in user_metrics.items():
            m_acc = np.mean(accs)
            all_user_means.append(float(m_acc))

            print(f"User {user_id} | Acc: {m_acc*100:.2f}% ± {m_acc*100:.2f}% (over {len(accs)} episodes)")
        # Calculate summary across users
        # These are still ratios (e.g., 0.10)
        mean_acc_ratio = np.mean(all_user_means)
        std_acc_ratio = np.std(all_user_means)
        # Create a clean list of percentages for the summary print
        user_acc_percentages = [round(a * 100, 2) for a in all_user_means]
        # --- END TIMER & PRINT ---
        fold_duration = time.time() - fold_start_time
        print(f"[Trial {trial.number} | Fold {fold_idx}] User accs (%): {user_acc_percentages}")
        # Multiply by 100 only ONCE here for display
        print(f"[Trial {trial.number} | Fold {fold_idx}] Mean acc: {mean_acc_ratio*100:.2f}% ± {std_acc_ratio*100:.2f}%")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Finished in {fold_duration:.2f} seconds.")

        fold_mean_accs.append(mean_acc_ratio)
        all_fold_user_accs.append(all_user_means)

    clean_fold_accs = [float(f) for f in fold_mean_accs]
    overall_mean_acc = float(np.nanmean(clean_fold_accs))

    trial.set_user_attr("fold_mean_accs", fold_mean_accs)
    trial.set_user_attr("fold_user_accs", all_fold_user_accs)
    trial.set_user_attr("mean_pretrain_val_acc", float(np.nanmean(pretrain_val_accs)))

    return overall_mean_acc


def run_study(study_name, storage_path, model_type, n_trials=1):
    sleep_time = random.uniform(0, 10)
    print(f"Staggering start: sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)

    lock_obj = JournalFileBackend(storage_path)
    storage = JournalStorage(lock_obj)

    time.sleep(random.uniform(0, 10))

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True, 
    )

    # Wrap the objective to pass the model_type
    study.optimize(lambda trial: objective(trial, model_type), n_trials=n_trials, gc_after_trial=True)
    
    return study

if __name__ == "__main__":
    # --- Parse Command Line Args ---
    parser = argparse.ArgumentParser(description="Run MAML++ HPO for a specific model architecture.")
    parser.add_argument("--model_type", type=str, default="TST", 
                        choices=["MetaCNNLSTM", "DeepCNNLSTM", "TST", "ContrastiveNet", "MOE"],
                        help="Which model architecture to optimize hyperparameters for.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    db_dir = "/scratch/my13/kai/meta-pers-gest/optuna_dbs"
    os.makedirs(db_dir, exist_ok=True)

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    
    # Create a unique database and study name per model architecture!
    study_name = f"mamlpp_pretr_{args.model_type}_2fcv_hpo"
    journal_path = os.path.join(db_dir, f"{study_name}.log")

    print(f"Starting HPO Study: {study_name}")
    print(f"Journal Path: {journal_path}")

    run_study(
        study_name=study_name,
        storage_path=journal_path,
        model_type=args.model_type,
        n_trials=N_TRIALS, 
    )