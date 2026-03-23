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
import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend
import random
from pathlib import Path

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
    config["num_pretrain_users"] = len(config["train_PIDs"])
    config["num_testft_users"] = len(config["val_PIDs"])

from system.MAML_MOE.mamlpp import *
from system.MAML_MOE.maml_data_pipeline import get_maml_dataloaders
from system.MAML_MOE.shared_maml import *
from system.MAML_MOE.MOE_CNN_LSTM import *

# Import new models here
## TODO: is there overlap / overwriting between these .py's? Hopefully not...
from system.pretraining.pretrain_models import *
from system.pretraining.contrastive_net.contrastive_encoder import *

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

    if model_type == "MetaCNNLSTM":
        config.update({
            "emg_base_cnn_filters": 32, "emg_cnn_layers": 1,
            "imu_base_cnn_filters": 32, "imu_cnn_layers": 1,
            "cnn_kernel_size": 5, "groupnorm_num_groups": 8,
            "lstm_hidden": 32, "lstm_layers": 1, "bidirectional": True,
        })
    elif model_type == "DeepCNNLSTM":
        config.update({
            "emg_base_cnn_filters": 32, "emg_cnn_layers": 3,
            "imu_base_cnn_filters": 32, "imu_cnn_layers": 3,
            "cnn_kernel_size": 5, "groupnorm_num_groups": 8,
            "lstm_hidden": 64, "lstm_layers": 3, "bidirectional": True,
        })
    elif model_type == "TST":
        config.update({
            "patch_len": 8, "d_model": 64, "n_heads": 4, "n_blocks": 3,
        })
    elif model_type == "ContrastiveNet":
        config.update({
            "arch_mode": "cnn_attn",
            "proj_hidden": 128, "proj_out": 64,
        })
    else:
        print("Falling back to old MOE dynamic config (this may not be supported...)")
    
    return config

# ===================== OPTUNA TUNING SCRIPT =====================
def build_model_from_trial(trial, model_type, base_config=None):
    config = copy.deepcopy(base_config) if base_config else {}

    # 1. Inject Architecture Constants
    config = inject_model_config(config, model_type)

    # === Task Setup ===
    config["n_way"] = 3  
    config["k_shot"] = 5  
    config["q_query"] = 5
    config["num_classes"] = 10
    config["meta_batchsize"] = trial.suggest_categorical("meta_batchsize", [16, 32]) 

    # === MAML Core Hyperparameters ===
    config["maml_inner_steps"] = trial.suggest_categorical("maml_inner_steps", [2, 3, 5, 7])
    config["maml_inner_steps_eval"] = trial.suggest_categorical("maml_inner_steps_eval", [10, 15, 20, 30])
    config["maml_opt_order"] = "first"  # TODO: Re-toggle this?
    config["maml_use_lslr"] = False  # TODO: this was giving us negative learning rates earlier... try turning it on again?
    
    config["maml_alpha_init"] = trial.suggest_float("maml_alpha_init", 1e-4, 1e-1, log=True)
    config["maml_alpha_init_eval"] = trial.suggest_float("maml_alpha_init_eval", 1e-4, 1e-2, log=True)
    config["learning_rate"] = trial.suggest_float("outer_lr", 1e-5, 1e-2, log=True)
    config["weight_decay"] = trial.suggest_float("wd", 1e-6, 1e-4, log=True)

    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)  # TODO: Not totally sure where this appears across ALL the different models...

    # === Finetuning / Transfer Learning Strategy ===
    config["finetuning_approach"] = trial.suggest_categorical("finetuning_approach", ["full"])  #, "anil", "frozen_backbone"])
    config["use_pretrained"] = True # Hardcoded to true based on your goal

    # === Multimodal & Conditioning (Keeping these if you still use FiLM/Demo heads) ===
    config["use_demographics"] = True
    config["use_film_x_demo"] = trial.suggest_categorical("use_film_x_demo", [True, False])
    config["FILM_on_context_or_demo"] = 'context' 
    config["context_emb_dim"] = trial.suggest_categorical("context_emb_dim", [8, 16, 32, 64])
    config["demo_emb_dim"] = trial.suggest_categorical("demo_emb_dim", [8, 16, 32, 64])
    config["context_pool_type"] = trial.suggest_categorical("context_pool_type", ['mean', 'attn'])  

    config["use_MOE"] = False 
    config["use_label_shuf_meta_aug"] = False  # TODO: This probably should be turned back on?
    config["num_epochs"] = 50 
    config["episodes_per_epoch_train"] = trial.suggest_categorical("episodes_per_epoch_train", [250, 500])
    config["earlystopping_patience"] = 8 
    config["label_smooth"] = 0.0

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    # TODO: Swap this to your actual model factories from pretrain_models.py
    # if model_type in ["MetaCNNLSTM", "DeepCNNLSTM", "TST"]:
    #     model = build_model(config)
    # else: 
    #     model = ContrastiveEncoder(config)
    
    # Placeholder for now so the code doesn't break
    model = MultimodalCNNLSTMMOE(config)
    
    # =========================================================================
    # ####################################################################### #
    # ################### PRETRAINED WEIGHT LOADING ######################### #
    # ####################################################################### #
    # =========================================================================
    if config["use_pretrained"]:
        print(f"--> Loading pretrained weights for {model_type}...")

        local_pretrain_path = r"C:\Users\kdmen\Repos\pers-gest-cls\pretrain_outputs\checkpoints"
        cluster_pretrain_path = r"\projects\my13\kai\meta-pers-gest\pers-gest-cls\pretrain_outputs\checkpoints"
        
        # #######################################################
        # REPLACE THESE DUMMY PATHS WITH YOUR ACTUAL FILE PATHS
        # #######################################################
        weight_paths = {
            "MetaCNNLSTM": f"{cluster_pretrain_path}\MetaCNNLSTM_best.pt",
            "DeepCNNLSTM": f"{cluster_pretrain_path}\DeepCNNLSTM_best.pt",
            "TST": f"{cluster_pretrain_path}\TST_best.pt",
            "ContrastiveNet": f"{cluster_pretrain_path}\ContrastiveNet_best_best.pt",
            "MOE": None # THERE IS NO PRETRAINED WEIGHTS FOR THIS ONE!
        }
        
        load_path = weight_paths.get(model_type, "")
        
        try:
            state_dict = torch.load(load_path, map_location=config["device"])
            # Filter out the classification/projection head so we only load the backbone
            filtered_dict = {k: v for k, v in state_dict.items() if "head" not in k and "projector" not in k}
            
            model_dict = model.state_dict()
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"--> Successfully loaded backbone weights from: {load_path}")
            
        except FileNotFoundError:
            print(f"#################################################################")
            print(f"WARNING: Pretrained weight file not found at {load_path}. ")
            print(f"Using random initialization instead!")
            print(f"#################################################################")
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
        tensor_dict_path = os.path.join(config.get("dfs_load_path", "./"), f"segfilt_rts_tensor_dict.pkl")
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
            
        mean_acc_ratio = np.mean(all_user_means)
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
    parser.add_argument("--model_type", type=str, default="DeepCNNLSTM", 
                        choices=["MetaCNNLSTM", "DeepCNNLSTM", "TST", "ContrastiveNet", "MOE"],
                        help="Which model architecture to optimize hyperparameters for.")
    args = parser.parse_args()

    db_dir = "/scratch/my13/kai/meta-pers-gest/optuna_dbs"
    os.makedirs(db_dir, exist_ok=True)

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    
    # Create a unique database and study name per model architecture!
    study_name = f"mamlpp_{args.model_type}_2fcv_hpo"
    journal_path = os.path.join(db_dir, f"{study_name}.log")

    print(f"Starting HPO Study: {study_name}")
    print(f"Journal Path: {journal_path}")

    run_study(
        study_name=study_name,
        storage_path=journal_path,
        model_type=args.model_type,
        n_trials=N_TRIALS, 
    )