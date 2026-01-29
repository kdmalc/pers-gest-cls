import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import time
import json
import os

from global_seed import set_seed
set_seed()

from DNN_FT_funcs import *
from revamped_model_classes import *

import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)


def create_optuna_study(expdef_df, model_str, my_feature_engr, num_trials=1, ft_method=None):
    NUM_PRESAVED_KFOLDS = 4
    def objective(trial, expdef_df, model_str, my_feature_engr, use_kfcv=True, num_presaved_kfolds=5, ft_method=None):

        # Set feature engineering and related params
        # I don't think these vary depending on if DMN or DCNN are used
        if my_feature_engr=="None":
            num_input_channels = 16
            seq_len = 64
            if model_str == "CTRLNet":
                use_layerwise_maxpool = False
            else:
                # This is the only one that has the seq len depth to actually use layerwise maxpooling
                use_layerwise_maxpool = trial.suggest_categorical('use_layerwise_maxpool', [True, False])
        elif my_feature_engr=="FS":
            num_input_channels = 184
            # I think this is basically immutable, otherwise not sure how I would get the correlation into seq length...
            seq_len = 1

            use_layerwise_maxpool = trial.suggest_categorical('use_layerwise_maxpool', [True, False])
            # Overwrite to False
            use_layerwise_maxpool = False  # It downsamples too much...
        elif my_feature_engr=="moments":
            #seq_len = trial.suggest_categorical(f'seq_len', [1, 5])
            seq_len = 1
            if seq_len==1:
                num_input_channels = 80
            elif seq_len==5:
                num_input_channels = 16
            else:
                raise ValueError("moments sequence length not recognized")
            
            # Overwrite to False since it downsamples too much
            use_layerwise_maxpool = trial.suggest_categorical('use_layerwise_maxpool', [False])
        #print(f"{my_feature_engr}: nc{num_input_channels} sl{seq_len}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Architecture Parameters
        if seq_len == 1:
            # If the sequence length is 1, kernel size must also be 1
            adj_ks = 1
        elif seq_len > 1:
            #max_ks = 7 if seq_len >= 7 else seq_len
            #min_ks = 1 #if seq_len < 3 else 3
            # Ensure kernel sizes are odd by constraining to odd values only
            # Suggest integers that are odd and within the min_ks to max_ks range
            ## So this filters for odd numbers only, sure
            adj_ks = trial.suggest_int(f'kernel_size', 1, 8) #[ks for ks in range(min_ks, max_ks - 1, 2) if ks % 2 != 0])
        cnn_layers_lists = [
            [4, 8, 16, 16, 32, 32],
            [8, 16, 16, 32, 32, 64], 
            [16, 32, 32, 64, 64, 128],
            [32, 64, 128, 256]
        ]
        conv_layers = []
        if model_str == "CTRLNet":
            n_conv_layers = 1
        else:
            n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        for i in range(n_conv_layers):
            # Dynamically choose valid options for the current layer
            #layer_options = [x for x in cnn_layers_lists[i] if x >= (conv_layers[-1][0] if conv_layers else 0)]
            # CANNOT DYNAMICALLY CHOOSE WITH OPTUNA, choices must remain fixed for all studies
            if model_str == "CTRLNet":
                out_channels = trial.suggest_int(f'out_channels_{i}', 4, 256, log=True)
                kernel_size = adj_ks
                stride = trial.suggest_int(f'stride_{i}', 1, 4)
            else:
                layer_options = cnn_layers_lists[i]
                # Suggest out_channels from the valid options
                out_channels = trial.suggest_categorical(f'out_channels_{i}', layer_options)
                # TODO: kernel_size isn't saved if it isn't suggested (seq_len=1 case)...
                kernel_size = adj_ks
                stride = trial.suggest_int(f'stride_{i}', 1, 2)
            conv_layers.append((out_channels, kernel_size, stride))

        lr_scheduler_patience = trial.suggest_categorical('lr_scheduler_patience', [5, 7])
        # NOTE: Commented out version was dynamically set since lr_scheduler_patience changes. Can't use that in optnua...
        earlystopping_patience = trial.suggest_categorical('earlystopping_patience', [7, 10])
        #                                           [x for x in [lr_scheduler_patience+2, lr_scheduler_patience*2, lr_scheduler_patience*2+1, lr_scheduler_patience*3]])
        ft_lr_scheduler_patience = trial.suggest_categorical('ft_lr_scheduler_patience', [7, 10])
        ft_earlystopping_patience = trial.suggest_categorical('ft_earlystopping_patience', [7, 10])
        #                                           [x for x in [ft_lr_scheduler_patience+2, ft_lr_scheduler_patience*2, ft_lr_scheduler_patience*2+1, ft_lr_scheduler_patience*3]])
        
        # Hyperparameters
        if model_str == "CTRLNet":
            config = {
                "model_str": model_str,
                "num_channels": num_input_channels,
                "sequence_length": seq_len, 
                "time_steps": 1,   # I don't think this is used with the 2D gesture classes...
                "num_classes": 10,
                "num_epochs": 80,  # Note that earlystopping should be on so we shouldn't actually reach this
                "num_ft_epochs": 50,  # Also has ES
                "num_train_gesture_trials": 8, 
                "num_ft_gesture_trials": 1,
                "num_pretrain_users": 24,  # Does this need to be updated? Why is this even passed in through the model config?
                "num_testft_users": 8,  # Does this need to be updated? Why is this even passed in through the model config?
                "conv_layers": conv_layers,
                "lstm_num_layers": trial.suggest_int('lstm_num_layers', 1, 3),  # TODO: Is there even an LSTM with CTRLNet? Or is it just fixed...
                "feature_engr": my_feature_engr, 
                #"use_layerwise_maxpool": False,  # NOT IMPLEMENTED FOR CTRLNet
                #"maxpool_kernel_size": 1,  # NOT IMPLEMENTED FOR CTRLNet
                "padding": trial.suggest_int('padding', 0, 2),  # Used in the conv layers
                #"use_batch_norm": False,  # NOT IMPLEMENTED FOR CTRLNet
                "batch_size": trial.suggest_int('batch_size', 1, 64),
                "learning_rate": trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True),
                "optimizer": trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
                "weight_decay": trial.suggest_float('weight_decay', 0.0, 1e-3),
                "fc_dropout": trial.suggest_float('fc_dropout', 0.1, 0.8), 
                #"cnn_dropout": 0.0,  #trial.suggest_float('cnn_dropout', 0.0, 0.2),  # NOT IMPLEMENTED FOR CTRLNet
                "ft_learning_rate": trial.suggest_float('ft_learning_rate', 1e-6, 1e-1, log=True),
                "ft_weight_decay": trial.suggest_float('weight_decay', 0.0, 1e-3),
                "ft_batch_size": trial.suggest_int('ft_batch_size', 1, 10),
                "use_earlystopping": True,
                "lr_scheduler_patience": lr_scheduler_patience, 
                "lr_scheduler_factor": trial.suggest_float('lr_scheduler_factor', 0.0, 0.5), 
                "earlystopping_patience": earlystopping_patience,
                "earlystopping_min_delta": trial.suggest_float('earlystopping_min_delta', 1e-5, 1e-1),  
                "ft_lr_scheduler_patience": ft_lr_scheduler_patience,
                "ft_lr_scheduler_factor": trial.suggest_float('ft_lr_scheduler_factor', 0.0, 0.5), 
                "ft_earlystopping_patience": ft_earlystopping_patience,
                "ft_earlystopping_min_delta": trial.suggest_float('ft_earlystopping_min_delta', 1e-5, 1e-1), 
                # NOTE: jupyter notebooks cwd is April_25, .py is already within April_25
                "user_split_json_filepath": "fixed_user_splits\\4kfcv_splits_RS17.json", # TODO: Is kfcv necessary? It is the only one implemented rn...
                "finetune_strategy": trial.suggest_categorical('finetune_strategy', ["full", "freeze_cnn", "freeze_all_but_final_dense"]),  
                "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning\\{timestamp}",
                "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\hyperparam_tuning\\{timestamp}", 
                "perf_log_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs", 
                "timestamp": timestamp,
                "verbose": False,
                "log_each_pid_results": False, 
                "save_ft_models": False,
                "reset_ft_layers": False
            }
        else:
            use_maxpool = trial.suggest_categorical('use_maxpool', [True, False])

            config = {
                "model_str": model_str,
                "num_channels": num_input_channels,
                "sequence_length": seq_len, 
                "time_steps": 1,   # I don't think this is used with the 2D gesture classes...
                "num_classes": 10,
                "num_epochs": 50,  # Note that earlystopping should be on so we shouldn't actually reach this
                "num_ft_epochs": 30,  # Also has ES
                "num_train_gesture_trials": 8, 
                "num_ft_gesture_trials": 1,
                "num_pretrain_users": 24, 
                "num_testft_users": 8,
                "conv_layers": conv_layers,  # This is variable but set above!
                "feature_engr": my_feature_engr, 
                "use_layerwise_maxpool": use_layerwise_maxpool,  # Conditional on whether or not to use maxpool after each layer. If false then not used at all if global isn't turned on...
                # TODO: If it doesn't suggest, then it doesn't save (in the case of else 1)
                "maxpool_kernel_size": trial.suggest_int('maxpool_kernel_size', 2, 4) if use_maxpool else 1,  # ^ This controls the maxpool kernel_size in DCNN and DMN
                "padding": trial.suggest_int('padding', 0, 2),  # Used in the conv layers
                "use_batch_norm": False,  # Assuming that the batches are too small for this to be helpful, I can manually flip this later
                "batch_size": trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64]),
                "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                "optimizer": trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
                "weight_decay": trial.suggest_float('weight_decay', 0.0, 1e-3),
                "fc_dropout": trial.suggest_float('fc_dropout', 0.1, 0.8), 
                "cnn_dropout": 0.0,  #trial.suggest_float('cnn_dropout', 0.0, 0.2),
                #######################################################################
                # Added, specifically for the LSTM
                "lstm_dropout": trial.suggest_float('lstm_dropout', 0.0, 0.8),
                "lstm_num_layers": trial.suggest_int('lstm_num_layers', 0, 4),
                "use_dense_cnn_lstm": trial.suggest_categorical('use_dense_cnn_lstm', [True, False]),
                "lstm_hidden_size": trial.suggest_categorical('lstm_hidden_size', [4, 8, 16, 32, 64]),
                #######################################################################
                "ft_learning_rate": trial.suggest_float('ft_learning_rate', 1e-6, 1e-2, log=True),
                "ft_weight_decay": trial.suggest_float('weight_decay', 0.0, 1e-3),
                "ft_batch_size": trial.suggest_categorical('ft_batch_size', [2, 5, 10]),
                "use_earlystopping": True,
                "lr_scheduler_patience": lr_scheduler_patience, 
                "lr_scheduler_factor": trial.suggest_float('lr_scheduler_factor', 0.0, 0.5), 
                "earlystopping_patience": earlystopping_patience,
                "earlystopping_min_delta": trial.suggest_float('earlystopping_min_delta', 1e-5, 1e-2),  
                "ft_lr_scheduler_patience": ft_lr_scheduler_patience,
                "ft_lr_scheduler_factor": trial.suggest_float('ft_lr_scheduler_factor', 0.0, 0.5), 
                "ft_earlystopping_patience": ft_earlystopping_patience,
                "ft_earlystopping_min_delta": trial.suggest_float('ft_earlystopping_min_delta', 1e-5, 1e-2), 
                # NOTE: jupyter notebooks cwd is April_25, .py is already within April_25
                "user_split_json_filepath": "fixed_user_splits\\4kfcv_splits_RS17.json",  # This used to be 5kfcv, but 5 doesnt go into 32 evenly
                "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning\\{timestamp}",
                "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\hyperparam_tuning\\{timestamp}", 
                "perf_log_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs", 
                "timestamp": timestamp,
                "verbose": False,
                "log_each_pid_results": False, 
                "save_ft_models": False,
                "reset_ft_layers": trial.suggest_categorical('reset_ft_layers', [True, False]),
            }

            if ft_method is None:
                config["finetune_strategy"] = trial.suggest_categorical('finetune_strategy', ["full", "freeze_cnn", "linear_probing", "progressive_unfreeze"])
            else:
                config["finetune_strategy"] = ft_method

            # Can I remove this?
            #####################################################
            #fc_layers = []
            ## First FC layer is always required
            #fc1_size = trial.suggest_categorical('fc1_size', [16, 32, 64, 128, 256])
            #fc_layers.append(fc1_size)
            ## Default values
            #fc2_size = 0
            #fc3_size = 0
            ## Conditionally add second FC layer
            #use_fc2 = trial.suggest_categorical('use_fc2', [True, False])
            #if use_fc2:
            #    fc2_size = trial.suggest_categorical('fc2_size', [16, 32, 64])
            #    fc_layers.append(fc2_size)
            ## Conditionally add third FC layer
            #use_fc3 = trial.suggest_categorical('use_fc3', [True, False]) if use_fc2 else False
            #if use_fc3:
            #    fc3_size = trial.suggest_categorical('fc3_size', [12, 16, 24, 32])
            #    fc_layers.append(fc3_size)
            ## Log parameters to Optuna with default values
            #trial.set_user_attr("fc2_size", fc2_size)
            #trial.set_user_attr("fc3_size", fc3_size)
            # Add to config
            #config["fc_layers"] = fc_layers
            #config["fc2_size"] = fc2_size
            #config["fc3_size"] = fc3_size
            #if use_fc2 == False:
            #    config["use_fc3"] = use_fc3
            #####################################################
            num_fc_layers = trial.suggest_int('num_fc_layers', 1, 3)
            fc_layers = []
            fc_size_spaces = [
                [16, 32, 64, 128, 256],
                [16, 32, 64, 128],
                [12, 16, 24, 32]
            ]
            for i in range(num_fc_layers):
                fc_layers.append(trial.suggest_categorical(f'fc{i+1}_size', fc_size_spaces[i]))
            config["num_fc_layers"] = num_fc_layers
            config["fc_layers"] = fc_layers

        if config["finetune_strategy"]=="progressive_unfreeze":
            config["progressive_unfreezing_schedule"] = trial.suggest_categorical('progressive_unfreezing_schedule', [1, 2 , 3, 5, 7])

        # MODEL TRAINING AND EVALUATION PIPELINE
        if use_kfcv:
            split_results = []
            for fold_idx in range(num_presaved_kfolds):
                fold_datasplit, all_participants, test_participants = make_data_split(expdef_df, config, split_index=fold_idx, return_participants=True)

                # Train the model
                training_results = main_training_pipeline(
                    fold_datasplit, all_participants=all_participants, test_participants=test_participants,
                    config=config
                )
                pretrained_model = training_results["model"]

                # Evaluate the configuration on the current data split
                user_accuracies = evaluate_configuration_on_ft(fold_datasplit, pretrained_model, config, timestamp)
                avg_accuracy = sum(user_accuracies) / len(user_accuracies)
                split_results.append({"avg_accuracy": avg_accuracy, "user_accuracies": user_accuracies})

            # Aggregate results across data splits
            overall_avg_accuracy = sum(split_result["avg_accuracy"] for split_result in split_results) / len(split_results)
            #overall_user_accuracies = [acc for split_result in split_results for acc in split_result["user_accuracies"]]

            return overall_avg_accuracy
        else:
            raise ValueError("Only kfcv is supported currently")

    # === Study Setup ===
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(n_warmup_steps=5)  # Automatically stops unpromising trials
    )
    study.optimize(lambda trial: objective(trial, expdef_df, model_str, my_feature_engr, num_presaved_kfolds=NUM_PRESAVED_KFOLDS, ft_method=ft_method), n_trials=num_trials)
    return study