N_TRIALS = 2

import os
code_dir = os.environ["CODE_DIR"]
data_dir = os.environ["DATA_DIR"]
run_dir  = os.environ["RUN_DIR"]

import sys, copy, json, time, joblib

import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
user_split_json_filepath = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_reduced.json"
def apply_fold_to_config(config, all_splits, fold_idx):
    """Mutates config in-place to set train/val/test PIDs for the given fold."""
    split = all_splits[fold_idx]
    config["train_PIDs"] = split["train"]
    config["val_PIDs"]   = split["val"]
    config["test_PIDs"]  = split["test"]
    config["num_pretrain_users"] = len(config["train_PIDs"])
    config["num_testft_users"] = len(config["val_PIDs"])
###########################

#code_dir.April_25.MOE. --> Said not to use this... code_dir isn't an actual package...
# Make sure I don't have files named the same thing...
from system.MAML_MOE.MOE_multimodal_model_classes import *
from system.MAML_MOE.MOE_quick_cls_heads import *
from system.MAML_MOE.MOE_training import *
from system.MAML_MOE.MOE_configs import *
from system.MAML_MOE.multimodal_data_processing import *  # Needed for load_multimodal_dataloaders()
from system.MAML_MOE.mamlpp import *
from system.MAML_MOE.maml_multimodal_dataloaders import *

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

# TODO: Not sure if this is needed on NOTS... or how this should ideally resolve itself...
# TODO: These dont even exist in the new repo...
#######################################################
# Add the parent directory folder to the system path
#sys.path.append(os.path.abspath(os.path.join('..')))
#print(f"CWD after sys path append: {os.getcwd()}")
#from April_25.configs.hyperparam_tuned_configs import *
#from April_25.utils.DNN_FT_funcs import *
#from April_25.utils.gesture_dataset_classes import *
#from April_25.utils.global_seed import set_seed
#set_seed()
#######################################################

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
    #config["cluster_iter_str"] = None
    config["feature_engr"] = "None"
    config["time_steps"] = 1
    config["sequence_length"] = 64
    config["num_train_gesture_trials"] = 9
    config["num_ft_gesture_trials"] = 1
    config["padding"] = 0 
    config["use_batch_norm"] = False
    config["timestamp"] = timestamp
    config["fc_dropout"] = 0.0
    config["num_classes"] = 10
    config["use_earlystopping"] = True
    config["reset_ft_layers"] = False 
    config["verbose"] = False
    config["num_total_users"] = 32
    config["train_gesture_range"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["valtest_gesture_range"] = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    config["NOTS"] = True
    if config["NOTS"]==False:
        # TODO: These are not updated yet...
        raise ValueError("You are running the NOTS=False (non-cluster) version!")

        # Presumably this will never be used since this .py file should only be called by a slurm file?
        ## SAVING
        config["user_split_json_filepath"] = "C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\fixed_user_splits\\24_8_user_splits_RS17.json"
        config["results_save_dir"] = f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\MOE\\{timestamp}_MOE"
        config["models_save_dir"] = f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\models\\MOE\\{timestamp}_MOE"
        ## Mutlimodal LOADING
        config["emg_imu_pkl_full_path"] = 'C:\\Users\\kdmen\\Box\\Yamagami Lab\\Data\\Meta_Gesture_Project\\filtered_datasets\\metadata_IMU_EMG_allgestures_allusers.pkl'
        config["pwmd_xlsx_filepath"] = "C:\\Users\\kdmen\\Repos\\fl-gestures\\Biosignal gesture questionnaire for participants with disabilities.xlsx"
        config["pwoutmd_xlsx_filepath"] = "C:\\Users\\kdmen\\Repos\\fl-gestures\\Biosignal gesture questionnaire for participants without disabilities.xlsx" 
        config["dfs_save_path"] = "C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\MOE\\full_datasplit_dfs\\"
        config["dfs_load_path"] = "C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\MOE\\full_datasplit_dfs\\Initial_Multimodal\\"
    elif config["NOTS"]==True:
        ## SAVING
        config["user_split_json_filepath"] = user_split_json_filepath
        config["results_save_dir"] = results_save_dir
        config["models_save_dir"] = models_save_dir
        ## Mutlimodal LOADING
        # TODO God fucking damn it I have no idea where the fuck this is
        #/scratch/my13/kai/meta-pers-gest/data/filtered_datasets
        config["emg_imu_pkl_full_path"] = f"{CODE_DIR}//dataset//filtered_datasets//metadata_IMU_EMG_allgestures_allusers.pkl" 
        
        config["pwmd_xlsx_filepath"] = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants with disabilities.xlsx"
        config["pwoutmd_xlsx_filepath"] = f"{CODE_DIR}//dataset//Biosignal gesture questionnaire for participants without disabilities.xlsx"
        
        # TODO I dont fucking know where this shit is either
        # TODO I dont even know what this is or what it is supposed to be................
        config["dfs_save_path"] = f"{CODE_DIR}/dataset//"
        config["dfs_load_path"] = f"{CODE_DIR}/dataset/meta-learning-sup-que-ds//"

    # ----- Model layout hyperparams -----
    config["user_emb_dim"]  = trial.suggest_int("user_emb_dim", 12, 48)
    config["num_experts"]   = trial.suggest_int("num_experts", 2, 10)
    config["top_k"]         = trial.suggest_categorical("top_k", [None, 1, 2, 3])

    # Gate choice
    config["gate_type"]     = trial.suggest_categorical("gate_type", ["user_aware", "feature_only", "user_only", "film_gate"])  #, "bilinear"
    config["gate_requires_u_user"] = False if config["gate_type"] == "feature_only" else True
    config["use_u_init_warm_start"] = True #trial.suggest_categorical("use_u_init_warm_start", [True, False])
    # ^ False is broken right now because WithUserOverride doesn't accept None as the init vector
    #if config["gate_type"]=="bilinear":
    #    # Using min here will def break this in Optuna right? ...
    #    config["rank"] = trial.suggest_int("bilinear_rank", 4, min(config["emb_dim"], 16))

    # Head choice
    config["head_type"]     = trial.suggest_categorical("head_type", ["linear", "cosine"])
    if config["head_type"] == "cosine":
        config["init_tau"] = 5.0  #trial.suggest_float("init_tau", 5.0, 30.0)

    # Dropout / regularizers
    config["expert_dropout"]     = 0.25  #trial.suggest_float("expert_dropout", 0.0, 0.40)
    config["label_smooth"]       = 0.1  #trial.suggest_float("label_smooth", 0.0, 0.15)
    config["gate_balance_coef"]  = 0.1  #trial.suggest_float("gate_balance_coef", 0.0, 0.15)

    # Pretraining optim
    config["learning_rate"]      = trial.suggest_float("pre_lr", 5e-6, 5e-4, log=True)
    config["weight_decay"]       = trial.suggest_float("pre_wd", 1e-6, 3e-3, log=True)
    config["optimizer"]          = "adamw"  #trial.suggest_categorical("pre_opt", ["adamw", "adam", "sgd"])
    config["lr_scheduler_factor"]= 0.1  #trial.suggest_categorical("pre_sched_factor", [0.1, 0.2])
    config["lr_scheduler_patience"]= 6  #trial.suggest_int("pre_sched_pat", 4, 10)
    config["earlystopping_patience"]= 8 #trial.suggest_int("pre_es_pat", 6, 14)
    config["earlystopping_min_delta"]= 0.005 #trial.suggest_float("pre_es_delta", 0.001, 0.01)

    # Finetuning regime
    config["finetune_strategy"]  = 'adaptation' #trial.suggest_categorical("finetune_strategy", ["experts_only", "experts_plus_gate", "full"])  #"linear_probing", 
    config["use_dropout_during_peft"] = False  #trial.suggest_categorical("use_dropout_during_peft", [False, True])
    config["ft_learning_rate"]   = trial.suggest_float("ft_lr", 1e-4, 5e-2, log=True)
    config["ft_weight_decay"]    = trial.suggest_float("ft_wd", 1e-6, 5e-3, log=True)
    config["ft_lr_scheduler_factor"]= 0.1  #trial.suggest_categorical("ft_sched_factor", [0.1, 0.25, 0.5])
    config["ft_lr_scheduler_patience"]= 4  #trial.suggest_int("ft_sched_pat", 4, 10)
    config["ft_earlystopping_patience"]= 10  #trial.suggest_int("ft_es_pat", 6, 14)
    config["ft_earlystopping_min_delta"]= 0.008  #trial.suggest_float("ft_es_delta", 0.0005, 0.01)

    # TODO: Surely this isn't used with MAML? We don't do PEFT... so wtf is the user table doing then.......
    #config["alt_or_seq_MOE_user_emb_ft"]= trial.suggest_categorical("alt_or_seq_MOE_user_emb_ft", ["sequential", "alternating"])

    # Batch sizes (keep pretrain stable; you can expose if needed)
    ## TODO: Confirm this has no effect... for MAML it should be fully controlled by num episodes or something??
    config["batch_size"] = 128  #trial.suggest_categorical("pre_bs", [32, 64, 128, 256, 512, 1024])
    config["ft_batch_size"] = 10  #trial.suggest_categorical("ft_bs", [1, 2, 8, 10])

    # User table usage (important for novel users) --> I think this needs to stay True, if False there's no backup method to learn user embeddings rn...
    # TODO: I have literally no idea how this works for a new user. I dont remember anymore
    config["use_user_table"]     = True  #trial.suggest_categorical("use_user_table", [True, False])

    # NEW MULTIMODAL
    # NOTE: GroupNorm uses 8 groups currently, could raise/lower that, but emb_dim must be divisible by num_groups or it will break!!
    config["groupnorm_num_groups"] = trial.suggest_categorical("groupnorm_num_groups", [4, 6, 8, 12])
    #config["emg_emb_dim"]       = trial.suggest_categorical("emg_emb_dim", [72, 96, 120, 192, 216, 288, 360])
    #config["imu_emb_dim"]       = trial.suggest_categorical("imu_emb_dim", [72, 96, 120, 192, 216, 288, 360])
    # Actually I'm gonna keep these the same. Simplifies the network. If IMU >> EMG in the emb dim, it might just overfit to IMU
    config["emb_dim"]       = trial.suggest_categorical("emb_dim", [72, 96, 120, 192, 216, 288, 360])

    # It is probably only worth trying strides of 221 and 211. My data is already downsampled to 64 so no reason to use higher stride idt
    ## Hmm I wonder if the strides need to be the same actually so the feature maps have the same seq lens... not sure...
    config['emg_stride2'] = trial.suggest_int("emg_stride2", 1, 2)
    config['imu_stride2'] = trial.suggest_int("imu_stride2", 1, 2)

    # Eh I'll just scale by 2...
    #config['emg_CNN_capacity'] = trial.suggest_categorical("emg_CNN_capacity", [72, 96, 120, 192, 216, 288, 360])
    #config['imu_CNN_capacity'] = trial.suggest_categorical("imu_CNN_capacity", [72, 96, 120, 192, 216, 288, 360])
    # Setting them equal for now since for this model the emg and imu are the same length
    config['emg_CNN_capacity_scaling'] = trial.suggest_categorical("emg_CNN_capacity_scaling", [1, 2, 3])
    config['imu_CNN_capacity_scaling'] = config['emg_CNN_capacity_scaling']  #trial.suggest_categorical("imu_CNN_capacity_scaling", [1, 2, 3])

    config["multimodal"] = True
    config['emg_in_ch'] = 16
    config['imu_in_ch'] = 72
    config['demo_in_dim'] = 12
    config['num_epochs'] = 35
    config['num_ft_epochs'] = 15

    # NEW FIELDS!
    config["mix_demo_u_alpha"] = 0.5

    # ADDING MAML SPECIFIC
    # TODO: How do all of these interact???
    config["meta_learning"] = True
    config["n_way"] = 10
    config["k_shot"] = 1
    config["q_query"] = 9  # TODO: Does this need to be 9? If it set it lower does that just make it faster? Does that impact the model? Slightly noiser eval??
    # TODO: Do the below eps/batch and eps/epoch need to be multiple of each other?
    config["episodes_per_batch_train"] = trial.suggest_categorical("episodes_per_batch_train", [50, 100, 250])  # Meta learning batch size
    config["episodes_per_epoch_train"] = trial.suggest_categorical("episodes_per_epoch_train", [250, 500, 1000])  # TODO: I have no idea what this should be... this is the max on the number of tasks per EPOCH. So this limits training, if the iterable is way too  (obvi true)
    config["num_workers"] = 8  # This is the dataloader, something about how many processes the CPU can use (more is faster generally)
    # Core MAML++
    config["maml_inner_steps"] = trial.suggest_int("maml_inner_steps", 1, 3)
    # TODO: Are the first and second order plus MSL not... like almost the same thing? I guess with no MSL there is literally no inner loop??
    config["maml_second_order"] = trial.suggest_categorical("maml_second_order", [True, False])                         # enables second-order when DOA switches on
    config["maml_first_order_to_second_order_epoch"] = trial.suggest_categorical("maml_first_order_to_second_order_epoch", [10, 30, 60, 100])      # DOA threshold (epochs <= this are first-order)
    config["maml_use_msl"] = trial.suggest_categorical("maml_use_msl", [True, False])                              # MSL (multi-step loss) on
    config["maml_msl_num_epochs"] = trial.suggest_categorical("maml_msl_num_epochs", [10, 30, 60, 100])                         # use MSL during first N epochs; after that, final-step loss only
    config["maml_use_lslr"] = trial.suggest_categorical("maml_use_lslr", [True, False])                             # learn per-parameter, per-step inner LRs
    # TODO: Is this maml_alpha_init being used as a learning rate?
    ## I remember that in PerFedAvg they said beta was around 0.5 or something (IIRC)
    ## Yes this is being used as a learning rate
    ## Gotta sort this out with the other one, idek if the other one is being used anymore...
    config["maml_alpha_init"] = 1E-3                            # fallback α (also eval α if LSLR not used at eval)
    config["enable_inner_loop_optimizable_bn_params"] = False  # by default, do NOT adapt BN in inner loop
    # Eval
    config["maml_inner_steps_eval"] = trial.suggest_int("maml_inner_steps_eval", 1, 3)
    config["maml_alpha_init_eval"] = 1E-3
    config["use_cosine_outer_lr"] = False                       # This is cosine-based lr annealing... is this in addition to my lr scheduler....
    config["use_lslr_at_eval"] = False                         # set True if you want to use learned per-parameter step sizes at eval

    # OPTUNA
    config["pool_mode"] = trial.suggest_categorical("pool_mode", ['avg', 'max', 'avgmax']) 
    config["pdrop"] = 0.1  # TODO: No idea what this is...
    config["mixture_mode"] = 'logits'  # 'logits' | 'probs' | 'logprobs' --> I don't think this is implemented AFAIK
    config["use_user_table"] = True  # TODO: I think this won't even run when False? Passing in None for users...
    config["moddrop_p"] = 0.15  # TODO: No idea what this is --? "(probability to drop IMU at train time)"
    config["demo_emb_dim"] = 16
    config["demo_conditioning"] = trial.suggest_categorical("demo_conditioning", ['concat', 'film'])
    config["expert_bigger"] = False  # (if True, widen Expert hidden)
    config["expert_bigger_mult"] = 2
    config["u_user_and_demos"] = trial.suggest_categorical("u_user_and_demos", ["demo", "mix", "u_user"])  # (ie table and u_user_overwriting, ie the default version) 

    # Thse should be in there but don't seem to be printed? Idk...
    config["use_u_init_warm_start"] = True
    config["gate_dense_before_topk"] = True 
    config["gate_requires_u_user"] = False if config["gate_type"] == "feature_only" else True
    config['log_each_pid_results'] = False
    config['saved_df_timestamp'] = '20250917_1217'  

    # NEW FOR LSTM VERSION! I AM NOT USING THE LSTM VERSION IN THIS NB!
    # ---- backbone toggle ----
    config["temporal_backbone"] = "none"     # "none" (current TCN-only) | "lstm"
    # ---- LSTM settings (used when temporal_backbone == "lstm") ----
    #config["lstm_hidden"] = 128
    #config["lstm_layers"] = 2
    #config["lstm_bidirectional"] = False
    #config["temporal_pool_mode"] = "last"    # "last" | "mean"   (pool *after* LSTM)
    # ---- MoE placement (you can leave this as-is; we keep MoE at the head) ----
    #config["moe_placement"] = "head"        # ("head" recommended; others optional/unused here)

    config["use_supportquery_for_ft"] = True

    # ----- Build model -----
    model = MultiModalMoEClassifier(config)
    device = config["device"]

    # Tweak Expert’s dropout inline (uses Expert.drop)
    for exp in model.experts:
        if isinstance(exp.drop, nn.Dropout):
            exp.drop.p = config["expert_dropout"]

    # Swap head if cosine
    if config["head_type"] == "cosine":
        #swap_expert_head_to_cosine(model, emb_dim=config["emb_dim"], num_classes=config["num_classes"], init_tau=config["init_tau"])
        model.swap_expert_head_to_cosine(init_tau=config["init_tau"])  # Default values: init_tau=10.0, learnable_tau=True)

    model.to(device)
    return model, config


#############################################################
# ---------- Load splits once ----------
with open(user_split_json_filepath, "r") as f:
    ALL_SPLITS = json.load(f)
NUM_FOLDS = len(ALL_SPLITS)
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

    #for fold_idx in range(NUM_FOLDS):
    # I am going to switch to just using fold 0 for computational reasons.
    ## After HPO, we should do our 4fcv on the top 3 or so trials
    ## The reasoning for not doing kfcv during HPO is;
    ## if a set of HPs perform better on one fold, they should still perform better (than a different set of HPs) on a different fold even if the absolute perforamnce changes 
    ## (ie the relative performance of HPs should not change based on the fold)
    for fold_idx in range(1):
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
        episodic_train_loader, episodic_val_loader, episodic_test_loader = load_multimodal_data_loaders(
            config,
            load_existing_dfs=True,
        )

        # Do the meta pretraining
        pretrained_model, pretrain_res_dict = MAMLpp_pretrain(
            model,
            config,
            episodic_train_loader,
            episodic_val_loader=episodic_val_loader,
        )
        best_val_acc = pretrain_res_dict["best_val_acc"]
        best_state   = pretrain_res_dict["best_state"]
        pretrain_val_accs.append(float(best_val_acc))

        print(f"[Trial {trial.number} | Fold {fold_idx}] Pretraining done. Best val acc = {best_val_acc:.4f}")

        # If you want to evaluate using the best pretrain weights:
        model.load_state_dict(best_state)

        # --------- Finetuning / Adaptation per user ---------
        user_loaders = make_user_loaders_from_dataloaders(
            episodic_val_loader,
            episodic_test_loader,
            config,
        )

        user_accs = []

        for pid, (user_val_epi_dl, user_test_epi_dl) in user_loaders.items():
            # For FixedOneShotPerUserIterable, each dl will usually yield exactly 1 episode

            if user_test_epi_dl is None:
                continue

            for episode in user_test_epi_dl:
                support_batch = episode["support"]
                query_batch   = episode["query"]

                result = mamlpp_finetune_and_eval(
                    model=model,
                    config=config,
                    support_batch=support_batch,
                    query_batch=query_batch,
                    # use_lslr_at_eval=False,
                )

                user_accs.append(result["acc"])
                # optionally store result["adapted_params"] per pid

        mean_acc = float(np.mean(user_accs)) if len(user_accs) > 0 else float("nan")
        print(f"[Trial {trial.number} | Fold {fold_idx}] User accs: {user_accs}")
        print(f"[Trial {trial.number} | Fold {fold_idx}] Mean acc: {mean_acc*100:.2f}%")

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
    trial.set_user_attr("fold_mean_accs", fold_mean_accs)
    trial.set_user_attr("fold_user_accs", all_fold_user_accs)
    trial.set_user_attr("pretrain_val_accs_per_fold", pretrain_val_accs)
    trial.set_user_attr("mean_pretrain_val_acc", float(np.nanmean(pretrain_val_accs)))

    # This is what Optuna optimizes
    return overall_mean_acc


def run_study(study_name="maml_mmoe_ft_HPO", storage=None, n_trials=2):

    sampler = TPESampler(n_startup_trials=12, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=8, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,          # e.g., "sqlite:///optuna_moe.db"
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Best trial:")
    bt = study.best_trial
    print("  value (k-fold mean finetune acc):", bt.value)
    print("  params:")
    for k, v in bt.params.items():
        print(f"    {k}: {v}")
    print("  mean_pretrain_val_acc:", bt.user_attrs.get("mean_pretrain_val_acc"))
    print("  fold_mean_accs:", bt.user_attrs.get("fold_mean_accs"))

    return study


if __name__ == "__main__":
    # NOTE: This is where the SQL db is set!
    db_path = "/scratch/my13/kai/meta-pers-gest/optuna_dbs/maml_moe_hpo.db"
    storage_url = f"sqlite:///{db_path}"

    study_res = run_study(
        study_name="maml_mmoe_4kfcv_hpo",
        storage=storage_url,
        n_trials=N_TRIALS,
    )

