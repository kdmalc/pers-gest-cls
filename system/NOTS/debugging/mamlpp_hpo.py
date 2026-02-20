N_TRIALS = 1
FIXED_SEED = 42

import os
code_dir = os.environ["CODE_DIR"]
data_dir = os.environ["DATA_DIR"]
run_dir  = os.environ["RUN_DIR"]

import copy, json, time#, joblib, sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import optuna
#from optuna.samplers import TPESampler
#from optuna.pruners import MedianPruner
#from optuna.storages import JournalStorage, JournalFileBackend
from optuna.storages.journal import JournalStorage, JournalFileBackend
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random

from pathlib import Path
# env -> Path objects
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
RUN_DIR = Path(os.environ.get("RUN_DIR", "./")).resolve()
print(f"CODE_DIR: {CODE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RUN_DIR: {RUN_DIR}")
# TODO: Update these for the appropriate saving. Don't save into the git repo on PROJECTS
#######################################################
# === SAVING (to SCRATCH) ===
#results_save_dir = RUN_DIR.parent / "runs" / f"{timestamp}_MOE"        # /scratch/my13/kai/runs/<timestamp>_MOE
#models_save_dir  = RUN_DIR.parent / "models" / "MOE" / f"{timestamp}_MOE"
results_save_dir = RUN_DIR
models_save_dir  = RUN_DIR
# make sure they exist
results_save_dir.mkdir(parents=True, exist_ok=True)
models_save_dir.mkdir(parents=True, exist_ok=True)

# === LOADING (from SCRATCH data bucket) ===
# adjust these to where you staged each file under $DATA_DIR
## These are now inputs to the config directly... --> These are actually the NOTS specific versions...
###########################
user_split_json_filepath = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"
def apply_fold_to_config(config, all_splits, fold_idx):
    """Mutates config in-place to set train/val/test PIDs for the given fold."""
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    config["num_pretrain_users"] = len(config["train_PIDs"])
    config["num_testft_users"] = len(config["val_PIDs"])
###########################

#from system.MAML_MOE.multimodal_data_processing import *  # Needed for load_multimodal_dataloaders()
#from system.MAML_MOE.maml_multimodal_dataloaders import *
from system.MAML_MOE.mamlpp import *
from maml_data_pipeline import get_maml_dataloaders, maml_mm_collate
from system.MAML_MOE.shared_maml import *
from system.MAML_MOE.MOE_CNN_LSTM import *

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

#############################################################

# ===================== OPTUNA TUNING SCRIPT =====================
# ====== (3) --- Objective: build model, pretrain, finetune, return metric ======
def build_model_from_trial(trial, base_config=None):
    if base_config is not None:
        config = copy.deepcopy(base_config) 
    else:
        config = dict()
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    config["feature_engr"] = "None"
    config["time_steps"] = 1  # TODO: Idk if this is used... its not called by the new model...
    config["sequence_length"] = 64  # TODO: Idk if this is used... its not called by the new model...
    config["num_train_gesture_trials"] = 9
    config["num_ft_gesture_trials"] = 1
    config["padding"] = 0 
    config["use_batch_norm"] = False  # NEVER USE! Our batches are small!
    config["timestamp"] = timestamp
    config["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.25])
    config["num_classes"] = 10
    config["use_earlystopping"] = True
    config["verbose"] = False
    config["num_total_users"] = 32
    config["train_gesture_range"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["valtest_gesture_range"] = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    config["NOTS"] = True
    if config["NOTS"]==False:
        #config["emg_imu_pkl_full_path"] = 'C:\\Users\\kdmen\\Box\\Yamagami Lab\\Data\\Meta_Gesture_Project\\filtered_datasets\\metadata_IMU_EMG_allgestures_allusers.pkl'
        config["pwmd_xlsx_filepath"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\Biosignal gesture questionnaire for participants with disabilities.xlsx"
        config["pwoutmd_xlsx_filepath"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\Biosignal gesture questionnaire for participants without disabilities.xlsx"
        config["dfs_save_path"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\meta-learning-sup-que-ds\\"
        config["dfs_load_path"] = "C:\\Users\\kdmen\\Repos\\pers-gest-cls\\dataset\\meta-learning-sup-que-ds\\"
        config["saved_df_timestamp"] = '20250917_1217'
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

    # ----- Model layout hyperparams -----
    #config["user_emb_dim"]  = trial.suggest_int("user_emb_dim", 12, 48)  # TODO: This is the size u? I dont think this is used with the new model? There must be a different var that contorls this...
    config["num_experts"]   = trial.suggest_int("num_experts", 2, 10)
    config["use_shared_expert"]   = trial.suggest_categorical("use_shared_expert", [True, False])
    config["expert_architecture"]   = trial.suggest_categorical("expert_architecture", ["MLP", "linear", "cosine"])
    # TODO: Should I add init_tau back in? Or is it not worth HPOing over... note that this is not synced right now
    #if config["expert_architecture"] == "cosine":
    #    config["init_tau"] = 5.0  #trial.suggest_float("init_tau", 5.0, 30.0)
    config["top_k"]         = trial.suggest_categorical("top_k", [None, 1, 2, 3])  # None means all/equal voting --> TODO: Does None still mean that...
    config["gate_type"]     = trial.suggest_categorical("gate_type", ["context_only", "feature_only", "demographic_only", "context_feature", "context_feature_demo"])
    #config["gate_dense_before_topk"] = True 
    #config["pool_mode"] = trial.suggest_categorical("pool_mode", ['avg', 'max', 'avgmax'])  # --> I dont remember what this was or where it was...
    config["mixture_mode"] = 'logits'  # 'logits' | 'probs' | 'logprobs' --> I didnt even add the other options
    config["return_aux"] = True  # Have the MOE layer return which expert got which sample. I am not 100% sure this is working correctly lol

    # NEW MULTIMODAL
    # NOTE: The latent dimensionality must be divisible by num_groups or it will break!!
    config["groupnorm_num_groups"] = trial.suggest_categorical("groupnorm_num_groups", [4, 8])
    config["emg_base_cnn_filters"]       = trial.suggest_categorical("emg_emb_dim", [16, 32, 64, 96, 128, 256])
    config["imu_base_cnn_filters"]       = trial.suggest_categorical("imu_emb_dim", [16, 32, 64, 96, 128, 256])
    # Idk if my model will still work if I increase the stride... nmight need to get the shapes to match....
    ## Hmm I wonder if the strides need to be the same actually so the feature maps have the same seq lens... not sure...
    config['emg_stride'] = 1  #trial.suggest_int("emg_stride2", 1, 2)
    config['imu_stride'] = 1  #trial.suggest_int("imu_stride2", 1, 2)
    config["cnn_kernel_size"] = trial.suggest_categorical("cnn_kernel_size", [3, 5, 7])
    config["imu_cnn_layers"] = trial.suggest_categorical("imu_cnn_layers", [1, 2, 3])
    config["emg_cnn_layers"] = trial.suggest_categorical("emg_cnn_layers", [1, 2, 3])
    config["use_film_x_demo"] = trial.suggest_categorical("use_film_x_demo", [True, False])
    config["use_imu"] = True 
    config["use_demographics"] = True  # Is it worth testing turning this off?
    config["context_emb_dim"] = trial.suggest_categorical("context_emb_dim", [4, 8, 12, 16])
    config["context_pool_type"] = trial.suggest_categorical("context_pool_type", ['mean', 'attn'])  
    config["demo_emb_dim"] = trial.suggest_categorical("demo_emb_dim", [4, 8, 16])
    #config["demo_conditioning"] = trial.suggest_categorical("demo_conditioning", ['concat', 'film'])

    config["multimodal"] = True
    config['emg_in_ch'] = 16
    config['imu_in_ch'] = 72
    config['demo_in_dim'] = 12
    config['num_epochs'] = 40  

    #config['log_each_pid_results'] = False --> Not used rn
    # This is used for the support query dfs that get loaded and then merged... that code should maybe get refactored but I dont think it is affecting MAML perf...
    #config['saved_df_timestamp'] = '20250917_1217'  

    config["use_lstm"] = trial.suggest_categorical("use_lstm", [True, False])
    if config["use_lstm"]:
        config["lstm_hidden"] = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
        config["lstm_layers"] = trial.suggest_categorical("lstm_layers", [1, 2, 3])
        #config["lstm_bidirectional"] = True  # Not implemented, hardcoded as True by default
        config["use_GlobalAvgPooling"] = trial.suggest_categorical("use_GlobalAvgPooling", [True, False])
    # ---- MoE placement (you can leave this as-is; we keep MoE at the head) ----
    #config["moe_placement"] = "head"        # ("head" recommended; others optional/unused here) --> This architecture is hardcoded in but ought to test other placements probably...

    # MOE Hyperparams
    config["label_smooth"]       = 0.1  #trial.suggest_float("label_smooth", 0.0, 0.15)
    # TODO: These are not used at all atm
    #config["expert_dropout"]     = 0.25  #trial.suggest_float("expert_dropout", 0.0, 0.40)
    #config["gate_balance_coef"]  = 0.1  #trial.suggest_float("gate_balance_coef", 0.0, 0.15)

    # Pretraining optim
    config["weight_decay"]       = trial.suggest_float("pre_wd", 1e-6, 1e-3, log=True)  # TODO: Is weight_decay used right now lol
    config["optimizer"]          = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
    config["lr_scheduler_factor"]= 0.1  #trial.suggest_categorical("pre_sched_factor", [0.1, 0.2])
    config["lr_scheduler_patience"]= 6  #trial.suggest_int("pre_sched_pat", 4, 10)
    config["earlystopping_patience"]= 8 #trial.suggest_int("pre_es_pat", 6, 14)
    config["earlystopping_min_delta"]= 0.005 #trial.suggest_float("pre_es_delta", 0.001, 0.01)

    # ADDING MAML SPECIFIC
    config["meta_learning"] = True
    config["n_way"] = 10
    config["k_shot"] = 1
    config["q_query"] = 9  # TODO: Does this need to be 9? If it set it lower does that just make it faster? Does that impact the model? Slightly noiser eval??
    # ^^ Is this actually being used as 9... or is the full complement being used...
    # TODO: Do the below eps/batch and eps/epoch need to be multiple of each other?
    config["meta_batchsize"] = trial.suggest_categorical("meta_batchsize", [4, 8, 16, 32, 64])  # Meta learning batch size, ie number of episodes per batch (this is handled via looping NOT in the dataloaders since sizes may not match bewteen episodes)
    config["episodes_per_epoch_train"] = trial.suggest_categorical("episodes_per_epoch_train", [250, 500, 1000])  # TODO: I have no idea what this should be... this is the max on the number of tasks per EPOCH. So this limits training, if the iterable is way too  (obvi true)
    config["num_workers"] = 8  # This is the dataloader, something about how many processes the CPU can use (more is faster generally)
    # Core MAML++
    config["maml_inner_steps"] = trial.suggest_int("maml_inner_steps", 1, 3)
    
    # First epochs are first order, then switches to second, if using hybrid
    config["maml_opt_order"] = trial.suggest_categorical("maml_opt_order", ["first", "second", "hybrid"])                         # enables second-order when DOA switches on
    if config["maml_opt_order"] == "hybrid":
        config["maml_first_order_to_second_order_epoch"] = trial.suggest_int("maml_first_order_to_second_order_epoch", 1, 40)      # DOA threshold (epochs <= this are first-order)
    # Theoretically this should be even be used, but just in case...
    elif config["maml_opt_order"] == "first":
        config["maml_first_order_to_second_order_epoch"] = 1000000  # Arbitrarily large to never trigger and switch to second
    elif config["maml_opt_order"] == "second":
        config["maml_first_order_to_second_order_epoch"] = 0  # Do second the whole time
    
    # use MSL during first N epochs; after that, final-step loss only
    ## First epochs are MSL, then turns it off
    config["use_maml_msl"] = trial.suggest_categorical("use_maml_msl", [True, False, "hybrid"])                              # MSL (multi-step loss) on
    if config["use_maml_msl"] == "hybrid":
        config["maml_msl_num_epochs"] = trial.suggest_int("maml_msl_num_epochs", 1, 40)  # Also note that currently the max num_epochs is 40 (plus we use ES so may not even hit this)
    # Theoretically this should be even be used, but just in case...
    elif config["use_maml_msl"] == True:
        config["maml_msl_num_epochs"] = 1000000  # Arbitrarily large to never trigger and turn MSL off
    elif config["use_maml_msl"] == False:
        config["maml_msl_num_epochs"] = 0
    
    # If LSLR is used, then alpha is only used as a fallback/init (and then we learn the learning rate). I am not sure if beta gets used at all. I think it does?
    config["maml_use_lslr"] = trial.suggest_categorical("maml_use_lslr", [True, False])                             # learn per-parameter, per-step inner LRs
    # AKA Beta. This is the outer loop learning rate (the one that actually gets used directly in the optimizer)
    config["learning_rate"]      = trial.suggest_float("pre_lr", 0.00001, 0.05)
    # Alpha is the inner loop learning rate
    ## I remember that in PerFedAvg they said beta was around 0.5 or something (IIRC)
    if config["maml_use_lslr"] == True:
        # These are only getting used as inits so it doesn't really matter I dont think...
        config["maml_alpha_init"] = 1E-3                            # fallback α (also eval α if LSLR not used at eval)
        config["maml_alpha_init_eval"] = 1E-3
    elif config["maml_use_lslr"] == False:
        config["maml_alpha_init"] = trial.suggest_float("maml_alpha_init", 0.00001, 0.05)
        config["maml_alpha_init_eval"] = trial.suggest_float("maml_alpha_init_eval", 0.00001, 0.05)
    config["enable_inner_loop_optimizable_bn_params"] = False  # by default, do NOT adapt BN in inner loop --> I should not be using BN at all AFAIK
    # Eval
    # At eval this is just the inner loop with no outer, so no MSL and no Hessian. This should be much quicker. 5-10 is common here
    config["maml_inner_steps_eval"] = trial.suggest_categorical("maml_inner_steps_eval", [3, 5, 10, 15, 20, 30])
    config["use_cosine_outer_lr"] = False                       # This is cosine-based lr annealing... is this in addition to my lr scheduler....
    config["use_lslr_at_eval"] = False                         # set True if you want to use learned per-parameter step sizes at eval

    # ----- Build model -----
    model = MultimodalCNNLSTMMOE(config)
    device = config["device"]

    model.to(device)
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
        #tensor_dict_path = os.path.join(config["dfs_load_path"], f"{config['saved_df_timestamp']}_tensor_dict.pkl")
        tensor_dict_path = os.path.join(config["dfs_load_path"], f"maml_tensor_dict.pkl")

        # This returns two standard PyTorch DataLoaders
        # It uses config['train_PIDs'] and config['val_PIDs'] internally to split the data
        episodic_train_loader, episodic_val_loader = get_maml_dataloaders(
            config,
            tensor_dict_path=tensor_dict_path,
            collate_fn=maml_mm_collate
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
        ## NOTE: Allegedly this is the same as just calling meta_evaluate()
        ## In the Outer Loop (Meta-Training), small batches are noisy. But in the Meta-Test phase, there is no "batch size" because there is no outer update.
        ## You are just processing episodes one by one. Increasing the "batch size" here would just be a trick to make it run faster on your GPU by parallelizing users; it wouldn't change the accuracy at all.
        #user_loaders = make_user_loaders_from_dataloaders(
        #    episodic_val_loader,
        #    episodic_test_loader,
        #    config,
        #)
        #val_dls = user_loaders[0]  # val_dls is a dictionary of dataloaders, keys are user_ids and values are the user-specific dataloaders
        #test_dls = user_loaders[1]
        #for user_id, user_val_dl in val_dls.items():
        #    if user_val_dl is None:
        #        raise ValueError("user_val_dl is None, preventing maml_finetune_and_eval...")
        #
        #    val_metrics = meta_evaluate(model, user_val_dl, config, mamlpp_adapt_and_eval)
        #    final_user_val_loss, final_user_val_acc = val_metrics["loss"], val_metrics["acc"]
        #    user_accs.append(final_user_val_acc)
        #    print(f"Final user{user_id} loss: {final_user_val_acc*100:.2f}%")
        user_accs = []
        # The new val_loader iterates through each validation user exactly once
        for batch in episodic_val_loader:
            user_id = batch['user_id']
            support_set = batch['support']
            query_set = batch['query']

            # Call your evaluation function using the support and query sets directly
            # We use mamlpp_adapt_and_eval which handles the inner-loop update
            val_metrics = mamlpp_adapt_and_eval(model, config, support_set, query_set)
            
            user_acc = val_metrics["acc"]
            user_accs.append(user_acc)
            print(f"User {user_id} | Validation Acc: {user_acc*100:.2f}%")

        mean_acc = float(np.mean(user_accs)) if len(user_accs) > 0 else float("nan")

        # --- END TIMER & PRINT ---
        fold_duration = time.time() - fold_start_time
        print(f"[Trial {trial.number} | Fold {fold_idx}] User accs: {user_accs}")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Mean acc: {mean_acc*100:.2f}%")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Finished in {fold_duration:.2f} seconds.")

        fold_mean_accs.append(mean_acc)
        all_fold_user_accs.append(user_accs)

        # (Optional) you could report per-fold for pruning:
        # trial.report(mean_acc, step=fold_idx)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    # Aggregate across folds for Optuna
    overall_mean_acc = float(np.nanmean(fold_mean_accs))
    print(f"[Trial {trial.number}] Fold mean accs: {fold_mean_accs}")
    print(f"[Trial {trial.number}] Overall k-fold mean acc: {overall_mean_acc*100:.2f}%")

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
    sleep_time = random.uniform(0, 60)
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
    journal_path = os.path.join(db_dir, "mamlpp_CNNLSTMMLP_2fcv_hpo.log")

    run_study(
        study_name="mamlpp_CNNLSTMMLP_2fcv_hpo",
        storage_path=journal_path,
        n_trials=N_TRIALS, # Each Slurm worker does one trial (N_TRIALS=1)
    )

