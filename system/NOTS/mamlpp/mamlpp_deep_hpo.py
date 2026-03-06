N_TRIALS = 1  # <5 min per epoch --> 5*50 = 250min = 4 hours
FIXED_SEED = 42

import os
import copy, json, time
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend
from pathlib import Path
import random

# Standardize Paths
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR = Path(os.environ.get("RUN_DIR", "./")).resolve()
results_save_dir = RUN_DIR
models_save_dir  = RUN_DIR

from system.MAML_MOE.mamlpp import *
from maml_data_pipeline import get_maml_dataloaders
from system.MAML_MOE.shared_maml import *
from system.MAML_MOE.MOE_CNN_LSTM import *

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
user_split_json_filepath = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"

def apply_fold_to_config(config, all_splits, fold_idx):
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    config["num_pretrain_users"] = len(config["train_PIDs"])
    config["num_testft_users"] = len(config["val_PIDs"])

def build_model_from_trial(trial, base_config=None):
    config = copy.deepcopy(base_config) if base_config else {}

    # === STATIC HYPERPARAMETERS (Reducing Search Space) ===
    config["n_way"] = 3  # TODO: Should I HPO 3way or 10way... 10way wasnt really learning at all... probably check both ig...
    config["k_shot"] = 1
    config["q_query"] = 9
    config["meta_batchsize"] = 32  # Stable baseline for meta-gradients
    config["maml_inner_steps"] = 3 # Moderate steps for training
    config["maml_inner_steps_eval"] = 10 # More steps for evaluation
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["sequence_length"] = 64
    config["use_batch_norm"] = False
    config["dropout"] = 0.1 # Fixed low dropout to prevent underfitting
    config["gate_type"] = "context_feature_demo" # Use all info for MoE
    config["num_experts"] = 4
    config["top_k"] = 2
    config["expert_architecture"] = "MLP"
    config["mixture_mode"] = 'logits'
    config["num_epochs"] = 50 
    config["maml_opt_order"] = "first" 

    config["maml_use_lslr"] = True # Learned Step Size is crucial for bigger models
    if config["maml_use_lslr"] == True:
        # These are only getting used as inits so it doesn't really matter I dont think...
        # Alpha Init (Inner Loop) - higher capacity models might need larger initial steps
        config["maml_alpha_init"] = trial.suggest_float("maml_alpha_init", 0.0001, 0.1, log=True)
        config["maml_alpha_init_eval"] = trial.suggest_float("maml_alpha_init_eval", 0.0001, 0.01, log=True)
    elif config["maml_use_lslr"] == False:
        config["maml_alpha_init"] = trial.suggest_float("maml_alpha_init", 0.0001, 0.1, log=True)
        config["maml_alpha_init_eval"] = trial.suggest_float("maml_alpha_init_eval", 0.0001, 0.01, log=True)
    config["enable_inner_loop_optimizable_bn_params"] = False  # by default, do NOT adapt BN in inner loop --> I should not be using BN at all AFAIK
    # Eval
    config["use_cosine_outer_lr"] = False                       # This is cosine-based lr annealing... is this in addition to my lr scheduler....
    config["use_lslr_at_eval"] = False

    config["use_maml_msl"] = False
    if config["use_maml_msl"] == "hybrid":
        config["maml_msl_num_epochs"] = trial.suggest_int("maml_msl_num_epochs", 1, 40)
    elif config["use_maml_msl"] == True:
        config["maml_msl_num_epochs"] = 1000000  # Arbitrarily large to never trigger and turn MSL off
    elif config["use_maml_msl"] == False:
        config["maml_msl_num_epochs"] = 0
    
    # === DYNAMIC HYPERPARAMETERS (Focusing on Model Size) ===

    config["groupnorm_num_groups"] = trial.suggest_categorical("groupnorm_num_groups", [4, 8])
    config["use_film_x_demo"] = trial.suggest_categorical("use_film_x_demo", [True, False])
    config["use_imu"] = True 
    config["use_demographics"] = True  # Is it worth testing turning this off?
    config["context_emb_dim"] = trial.suggest_categorical("context_emb_dim", [4, 8, 12, 16, 32, 64])
    config["context_pool_type"] = trial.suggest_categorical("context_pool_type", ['mean', 'attn'])  
    config["demo_emb_dim"] = trial.suggest_categorical("demo_emb_dim", [4, 8, 16, 32, 64])
    
    # Increase CNN capacity (Width and Depth)
    # We move from 16-128 to 64-512 to give the model more "memory" for features
    config["emg_base_cnn_filters"] = trial.suggest_categorical("emg_width", [64, 128, 256, 512])
    config["imu_base_cnn_filters"] = trial.suggest_categorical("imu_width", [64, 128, 256, 512])
    # Depth: 3 to 6 layers. 
    # (Note: Beyond 6 usually requires ResNet connections to train in MAML)
    config["emg_cnn_layers"] = trial.suggest_int("emg_depth", 3, 6)
    config["imu_cnn_layers"] = trial.suggest_int("imu_depth", 3, 6)
    config["cnn_kernel_size"] = trial.suggest_categorical("cnn_kernel", [3, 5])
    config["use_GlobalAvgPooling"] = trial.suggest_categorical("use_GlobalAvgPooling", [True, False])
    config['emg_stride'] = 1  
    config['imu_stride'] = 1 

    # LSTM Capacity
    config["use_lstm"] = True 
    config["lstm_hidden"] = trial.suggest_categorical("lstm_hidden", [64, 128, 256, 512])
    config["lstm_layers"] = trial.suggest_int("lstm_layers", 1, 3)

    # Optimization (Bigger models need tailored LR and Weight Decay)
    config["learning_rate"] = trial.suggest_float("outer_lr", 1e-5, 1e-2, log=True)
    config["weight_decay"] = trial.suggest_float("wd", 1e-6, 1e-4, log=True)

    # Path setups (Remain static)
    config["NOTS"] = True
    config["user_split_json_filepath"] = user_split_json_filepath
    config["results_save_dir"] = results_save_dir
    config["models_save_dir"] = models_save_dir
    config["dfs_load_path"] = f"{CODE_DIR}/dataset/meta-learning-sup-que-ds//"

    # Things it dropped
    config["optimizer"]          = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
    # TODO: Should I even bother turning ES on? My ES is pretty generous...
    config["lr_scheduler_factor"]= 0.1  
    config["lr_scheduler_patience"]= 6  
    config["earlystopping_patience"]= 8 
    config["earlystopping_min_delta"]= 0.005 
    config["meta_learning"] = True
    config["episodes_per_epoch_train"] = trial.suggest_categorical("episodes_per_epoch_train", [250, 500, 1000])  # TODO: I have no idea what this should be... this is the max on the number of tasks per EPOCH. So this limits training, if the iterable is way too  (obvi true)
    config["num_workers"] = 8
    config["label_smooth"] = 0.0
    config["multimodal"] = True
    config['emg_in_ch'] = 16
    config['imu_in_ch'] = 72
    config['demo_in_dim'] = 12
    config["num_classes"] = 10
    config["return_aux"] = True

    model = MultimodalCNNLSTMMOE(config)
    model.to(config["device"])
    return model, config


#############################################################
# ---------- Load splits once ----------
with open(user_split_json_filepath, "r") as f:
    ALL_SPLITS = json.load(f)
NUM_FOLDS = 2  #len(ALL_SPLITS) --> Overwriting to just be 1 for HPO, otherwise runs legit cannot finish
## Im going to change it to 2 so that we dont just overfit to the fixed val set...
## I dont think 2 folds can finish in 24 hours... increase the time to 24 hours ig...
# If you want to store per-fold metrics:
fold_mean_accs = []
fold_user_accs = []  # list of lists (per-fold user accs)
# --------- base config ----------
BASE_CONFIG = {}  #emg_moments_only_MOE_config

def objective(trial):
    """
    Optuna objective:
    - For each fold in ALL_SPLITS:
        * build model/config from this trial's hyperparams
        * set train/val/test PIDs for this fold
        * run pretraining (MAML++)
        * finetune per user and compute accuracy
    - Return mean accuracy across folds.
    """

    fold_mean_accs = []
    all_fold_user_accs = []      # list of lists, per-fold user accuracies
    pretrain_val_accs = []       # best pretrain val acc per fold (for logging/debug)

    # I am going to switch to just using fold 0 for computational reasons.
    ## After HPO, we should do our 4fcv on the top 3 or so trials
    ## The reasoning for not doing kfcv during HPO is;
    ## if a set of HPs perform better on one fold, they should still perform better (than a different set of HPs) on a different fold even if the absolute perforamnce changes 
    ## (ie the relative performance of HPs should not change based on the fold)
    for fold_idx in range(NUM_FOLDS):
        fold_start_time = time.time()

        print("=" * 80)
        print(f"[Trial {trial.number}] Starting fold {fold_idx + 1}/{NUM_FOLDS}")
        print("=" * 80)

        # ---- Build model + config for this trial/fold ----
        # Uses your new builder that takes `trial` and sets hyperparams via trial.suggest_*
        model, config = build_model_from_trial(trial, base_config=BASE_CONFIG)

        print(f"CONFIG[GATE_TYPE]: {config['gate_type']}")

        if config["device"] == "cpu":
            print("HPO is happening on the CPU! Probably ought to switch to GPU!")

        # ---- Apply the fold-specific user IDs (from JSON) ----
        apply_fold_to_config(config, ALL_SPLITS, fold_idx)

        # ---- Pretraining data & training ----
        #episodic_train_loader, episodic_val_loader, episodic_test_loader = load_multimodal_data_loaders(
        #    config,
        #    load_existing_dfs=True,
        #)
        # ---- New Optimized Data Loading ----
        # Define the path to the .pkl file we just created/verified
        tensor_dict_path = os.path.join(config["dfs_load_path"], f"maml_tensor_dict.pkl")
        # This returns two standard PyTorch DataLoaders
        # It uses config['train_PIDs'] and config['val_PIDs'] internally to split the data
        episodic_train_loader, episodic_val_loader = get_maml_dataloaders(
            config,
            tensor_dict_path=tensor_dict_path,
        )

        # Do the meta "pretraining" (is this just the meta-train phase?)
        ## mamlpp_pretrain returns: train_loss_log, train_acc_log, val_loss_log, val_acc_log, model, best_state, best_val_acc,
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
            'config': config, # Useful to save the config used to build the model!
            'best_val_acc': best_val_acc, 
            'train_loss_log': pretrain_res_dict["train_loss_log"], 
            'train_acc_log': pretrain_res_dict["train_acc_log"],
            'val_loss_log': pretrain_res_dict["val_loss_log"],
            'val_acc_log': pretrain_res_dict["val_acc_log"]
        }, save_path)
        print(f"Model permanently saved to {save_path}")

        # If you want to evaluate using the best pretrain weights:
        model.load_state_dict(best_state)

        # --------- Finetuning / Adaptation per Novel user ---------
        user_metrics = defaultdict(list)
        # The val_loader now iterates through eval_episodes per user
        for batch in episodic_val_loader:
            user_id = batch['user_id']
            support_set = batch['support']
            query_set = batch['query']
            val_metrics = mamlpp_adapt_and_eval(model, config, support_set, query_set)
            user_metrics[user_id].append(val_metrics["acc"])

        # Calculate and print grouped metrics
        print("\n--- Final User-Specific Evaluation ---")
        all_user_means = []
        for user_id, accs in user_metrics.items():
            # Keep as ratios (0.0 to 1.0) internally
            m_acc = np.mean(accs)
            s_acc = np.std(accs)
            # Convert to standard Python float to avoid np.float64 wrapping in the list
            all_user_means.append(float(m_acc))
            print(f"User {user_id} | Acc: {m_acc*100:.2f}% ± {s_acc*100:.2f}% (over {len(accs)} episodes)")
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

        # (Optional) you could report per-fold for pruning:
        # trial.report(mean_acc, step=fold_idx)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    # Aggregate across folds for Optuna
    # Final k-fold print
    clean_fold_accs = [float(f) for f in fold_mean_accs]
    overall_mean_acc = float(np.nanmean(clean_fold_accs))
    print(f"[Trial {trial.number}] Fold mean accs: {clean_fold_accs}")
    print(f"[Trial {trial.number}] Overall k-fold mean acc: {np.mean(clean_fold_accs)*100:.2f}%")

    # ---- Log ancillary info for analysis ----
    ## NOTE: This is logged as part of Optuna? How/where do I access this?... From the db??
    trial.set_user_attr("fold_mean_accs", fold_mean_accs)
    trial.set_user_attr("fold_user_accs", all_fold_user_accs)
    trial.set_user_attr("pretrain_val_accs_per_fold", pretrain_val_accs)
    trial.set_user_attr("mean_pretrain_val_acc", float(np.nanmean(pretrain_val_accs)))

    # This is what Optuna optimizes
    return overall_mean_acc


def run_study(study_name, storage_path, n_trials=1):

    # Generate a random sleep offset
    # This prevents the "Thundering Herd" problem on HPC filesystems
    sleep_time = random.uniform(0, 10)
    print(f"Staggering start: sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)

    # 1. Use JournalStorage to avoid SQLite locking issues on HPC filesystems
    lock_obj = JournalFileBackend(storage_path)
    storage = JournalStorage(lock_obj)

    # 2. Add a small random sleep to prevent "thundering herd" race conditions
    # when multiple jobs start at the exact same millisecond.
    time.sleep(random.uniform(0, 10))

    # TODO: Not sure how startup trials works since I am only using 1 per job...
    ## Plus I dont have intermediate values turned on right now (should probably use fold 1 performance?)
    #sampler = TPESampler(n_startup_trials=12, multivariate=True, group=True)
    #pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        #sampler=sampler,
        #pruner=pruner,
        storage=storage,
        load_if_exists=True, # Critical for parallel workers
    )

    # 3. Setting n_trials=1 ensures this job does one set of HPs and then finishes.
    # This allows Slurm to manage the queue effectively.
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    
    return study

if __name__ == "__main__":
    # Ensure the directory for the journal exists
    db_dir = "/scratch/my13/kai/meta-pers-gest/optuna_dbs"
    os.makedirs(db_dir, exist_ok=True)

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    
    # The journal file is just a log of operations (no complex SQL locking)
    journal_path = os.path.join(db_dir, "mamlpp_CNNLSTMMLP_deep_1s3w_2fcv_hpo.log")

    run_study(
        study_name="mamlpp_CNNLSTMMLP_deep_1s3w_2fcv_hpo",
        storage_path=journal_path,
        n_trials=N_TRIALS, # Each Slurm worker does one trial (N_TRIALS=1)
    )

