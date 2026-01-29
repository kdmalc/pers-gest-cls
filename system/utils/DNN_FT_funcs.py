import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.model_selection import ParameterSampler
import pandas as pd
from datetime import datetime
import pickle, random, re, copy, time, sys, os, json
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from utils.revamped_model_classes import *

current_directory = os.getcwd()
print(f"DNN_FT_funcs.py: The current working directory is: {current_directory}")
if current_directory != "c:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\MOE":
    print(f"DNN_FT_funcs.py: {current_directory} != c:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\MOE")
    # This should be the .py version...
    from ..MOE.MOE_model_classes import WithUserOverride
    from ..MOE.MOE_training import peft_user_emb_vec, uservec_from_support_mean
else:
    from MOE_model_classes import WithUserOverride
    from MOE_training import peft_user_emb_vec, uservec_from_support_mean


def convert_participant_ids(participant_ids):
    """
    Normalize participant IDs to ints.
    Accepts list/np.ndarray of strings like 'P123'/'P10X' or ints/np.ints.
    Returns: list[int]
    """
    if participant_ids is None:
        print("participant_ids is None, returning an empty list")
        return []
    # Support numpy arrays
    if isinstance(participant_ids, np.ndarray):
        participant_ids = participant_ids.tolist()
    if len(participant_ids) == 0:
        return []

    out = []
    for pid in participant_ids:
        # Already int-like
        if isinstance(pid, (int, np.integer)):
            out.append(int(pid))
            continue
        # String-like -> extract the first group of digits
        if isinstance(pid, str):
            m = re.search(r'(\d+)', pid)
            if not m:
                raise ValueError(f"Could not extract numeric portion from participant_id='{pid}'")
            out.append(int(m.group(1)))
            continue
        # Fallback
        raise TypeError(f"Unsupported participant_id type: {type(pid)} for value {pid}")
    return out


def load_data_and_make_dataloaders(config):
    """
    Builds BOTH:
      - pretrain-only user-id namespace (for pretraining user tables)
      - global user-id namespace (stable across all splits used in this run)

    Encodes:
      - pretrain splits with pretrain_id2idx
      - novel FT/eval with global_id2idx (so indices never 'reset')

    Returns:
      train_loader, val_loader, ft_loader, novel_test_loader, meta_dict
      where meta_dict contains both mappings for reproducibility.
    """

    # -----------------------
    # Load splits
    # -----------------------
    expdef_df = load_expdef_gestures(feateng_method=config["feature_engr"])
    data_splits = make_data_split(expdef_df, config, split_index=None)
    # keys include: (the last two may be empty // are not fully supported yet...)
    #   'pretrain_dict', 'pretrain_subject_test_dict',
    #   'novel_trainFT_dict', 'novel_subject_test_dict',
    #   'novel_val_trainFT_dict', 'novel_subject_val_test_dict'

    # Helper to safely get IDs from a split
    def get_ids(split_key):
        split = data_splits.get(split_key)
        if not split:
            print(f"Split {split_key} was empty!")
            return np.array([], dtype=np.int64)
        return np.asarray(convert_participant_ids(split.get('participant_ids', [])), dtype=np.int64)

    # -----------------------
    # Collect IDs per split
    # -----------------------
    train_ids_raw        = get_ids('pretrain_dict')
    val_ids_raw          = get_ids('pretrain_subject_test_dict')

    novel_ft_ids_raw       = get_ids('novel_trainFT_dict')
    novel_val_ft_ids_raw   = get_ids('novel_val_trainFT_dict')
    novel_test_ids_raw     = get_ids('novel_subject_test_dict')
    novel_val_test_ids_raw = get_ids('novel_subject_val_test_dict')

    # -----------------------
    # Build namespaces
    # -----------------------
    pretrain_ids = np.unique(np.concatenate([train_ids_raw, val_ids_raw])).astype(np.int64)
    global_ids   = np.unique(np.concatenate([
        pretrain_ids,
        novel_ft_ids_raw, novel_val_ft_ids_raw,
        novel_test_ids_raw, novel_val_test_ids_raw
    ])).astype(np.int64)

    pretrain_ids = np.sort(pretrain_ids)
    global_ids   = np.sort(global_ids)

    pretrain_id2idx = {int(u): i for i, u in enumerate(pretrain_ids)}
    pretrain_idx2id = pretrain_ids.tolist()

    global_id2idx = {int(u): i for i, u in enumerate(global_ids)}
    global_idx2id = global_ids.tolist()

    #print(f"[map] pretrain users: {len(pretrain_idx2id)} -> [0..{len(pretrain_idx2id)-1}]")
    #print(f"[map] global users:   {len(global_idx2id)} -> [0..{len(global_idx2id)-1}]")

    # -----------------------
    # Encode with appropriate namespace
    # -----------------------
    # Pretraining with pretrain namespace
    enc_train_ids = [pretrain_id2idx[int(u)] for u in train_ids_raw]
    enc_val_ids   = [pretrain_id2idx[int(u)] for u in val_ids_raw]

    # Novel FT / eval with global namespace (stable across splits; no reset)
    enc_novel_ft_ids       = [global_id2idx[int(u)] for u in novel_ft_ids_raw]
    enc_novel_test_ids     = [global_id2idx[int(u)] for u in novel_test_ids_raw]
    #enc_novel_val_ft_ids   = [global_id2idx[int(u)] for u in novel_val_ft_ids_raw]     # optional
    #enc_novel_val_test_ids = [global_id2idx[int(u)] for u in novel_val_test_ids_raw]   # optional

    # -----------------------
    # Build datasets / loaders
    # -----------------------
    # Pretrain train
    train_dataset = make_tensor_dataset(
        data_splits['pretrain_dict']['feature'],
        data_splits['pretrain_dict']['labels'],
        config,
        participant_ids=enc_train_ids
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    # Pretrain val (intra-subject test); don't shuffle
    val_dataset = make_tensor_dataset(
        data_splits['pretrain_subject_test_dict']['feature'],
        data_splits['pretrain_subject_test_dict']['labels'],
        config,
        participant_ids=enc_val_ids
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    # Novel one-shot FT (may be empty depending on split)
    if len(novel_ft_ids_raw) > 0 and data_splits.get('novel_trainFT_dict'):
        ft_dataset = make_tensor_dataset(
            data_splits['novel_trainFT_dict']['feature'],
            data_splits['novel_trainFT_dict']['labels'],
            config,
            participant_ids=enc_novel_ft_ids
        )
        ft_loader = DataLoader(
            ft_dataset,
            batch_size=config["ft_batch_size"],
            shuffle=True
        )
    else:
        ft_loader = None

    # Novel final test (don’t shuffle)
    if len(novel_test_ids_raw) > 0 and data_splits.get('novel_subject_test_dict'):
        novel_test_dataset = make_tensor_dataset(
            data_splits['novel_subject_test_dict']['feature'],
            data_splits['novel_subject_test_dict']['labels'],
            config,
            participant_ids=enc_novel_test_ids
        )
        novel_test_loader = DataLoader(
            novel_test_dataset,
            batch_size=config["batch_size"],
            shuffle=False
        )
    else:
        novel_test_loader = None

    # Optional: also prepare novel_val_* loaders if you use them
    # (kept here as references; enable if needed)
    # if len(novel_val_ft_ids_raw) > 0 and data_splits.get('novel_val_trainFT_dict'):
    #     novel_val_ft_dataset = make_tensor_dataset(
    #         data_splits['novel_val_trainFT_dict']['feature'],
    #         data_splits['novel_val_trainFT_dict']['labels'],
    #         config,
    #         participant_ids=enc_novel_val_ft_ids
    #     )
    #     novel_val_ft_loader = DataLoader(
    #         novel_val_ft_dataset,
    #         batch_size=config.get("ft_batch_size", 16),
    #         shuffle=False
    #     )
    # else:
    #     novel_val_ft_loader = None

    # if len(novel_val_test_ids_raw) > 0 and data_splits.get('novel_subject_val_test_dict'):
    #     novel_val_test_dataset = make_tensor_dataset(
    #         data_splits['novel_subject_val_test_dict']['feature'],
    #         data_splits['novel_subject_val_test_dict']['labels'],
    #         config,
    #         participant_ids=enc_novel_val_test_ids
    #     )
    #     novel_val_test_loader = DataLoader(
    #         novel_val_test_dataset,
    #         batch_size=config.get("batch_size", 64),
    #         shuffle=False
    #     )
    # else:
    #     novel_val_test_loader = None

    meta = {
        "pretrain_id2idx": pretrain_id2idx,
        "pretrain_idx2id": pretrain_idx2id,
        "global_id2idx": global_id2idx,
        "global_idx2id": global_idx2id,
    }

    return train_loader, val_loader, ft_loader, novel_test_loader, meta

def save_results(results, save_dir, timestamp):
    """Save the results to a JSON file, sorted by overall average accuracy."""
    # Sort results by overall average accuracy in descending order
    sorted_results = sorted(results, key=lambda x: x["overall_avg_accuracy"], reverse=True)

    # Add a note about the best configuration
    if sorted_results:
        best_config = sorted_results[0]["config"]
        best_accuracy = sorted_results[0]["overall_avg_accuracy"]
        sorted_results.insert(0, {
            "note": f"Best configuration: {best_config} with overall average accuracy: {best_accuracy:.4f}"
        })

    # Save to JSON file
    results_path = os.path.join(save_dir, f'{timestamp}_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Ensure the directory exists
    with open(results_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    #print(f"Results saved to: {results_path}")


def group_data_by_participant(features, labels, pids):
    """Helper function to group features and labels by participant ID."""
    user_data = defaultdict(lambda: ([], []))  # Tuple of lists: (features, labels)
    for feature, label, pid in zip(features, labels, pids):
        user_data[pid][0].append(feature)
        user_data[pid][1].append(label)
    return user_data


def evaluate_configuration_on_ft(datasplit, pretrained_model, config, model_str):
    """Evaluate a configuration on a given data split."""
    user_accuracies = []

    #########################################################
    # Here this code is called once and then we iterate through all the FT users below so this is fine

    # Validate and then call function
    train_dict_key = 'novel_val_trainFT_dict'
    test_dict_key = 'novel_subject_val_test_dict'
    ft_features = validate_feature_data(datasplit.get(train_dict_key, {}).get('feature'), f"{train_dict_key}.feature")
    novel_features = validate_feature_data(datasplit.get(test_dict_key, {}).get('feature'), f"{test_dict_key}.feature")

    # Only continue if features exist
    if ft_features is not None and novel_features is not None:
        pass
    else:
        train_dict_key = 'novel_trainFT_dict'
        test_dict_key = 'novel_subject_test_dict'
        ft_features = validate_feature_data(datasplit.get(train_dict_key, {}).get('feature'), f"{train_dict_key}.feature")
        novel_features = validate_feature_data(datasplit.get(test_dict_key, {}).get('feature'), f"{test_dict_key}.feature")

        if ft_features is not None and novel_features is not None:
            pass
        else:
            raise ValueError("No data found/available")
    
    ft_labels = datasplit.get(train_dict_key, {}).get('labels', [])
    ft_pids = datasplit.get(train_dict_key, {}).get('participant_ids', [])
    novel_labels = datasplit.get(test_dict_key, {}).get('labels', [])
    novel_pids = datasplit.get(test_dict_key, {}).get('participant_ids', [])

    ## Novel subject training data (probably one-shot)
    ft_user_data = group_data_by_participant(ft_features, ft_labels, ft_pids)
    ## Novel subject testing data (the remaining gestures)
    novel_user_data = group_data_by_participant(novel_features, novel_labels, novel_pids)
    #########################################################

    # Iterate through each unique participant ID
    ## The same users are in ft and cross datasets
    for pid in ft_user_data.keys() & novel_user_data.keys():  # Only common participant IDs ... these should overlap 100%...
        if config["verbose"]:
            print(f"Fine-tuning on user {pid}")

        # Prepare datasets
        ft_features, ft_labels = ft_user_data[pid]
        ft_features = torch.tensor(np.array(ft_features), dtype=torch.float32)
        #assert ft_features.ndim == 3, f"Expected 3D tensor (batch, channels, sequence), got {ft_features.ndim}D with shape {ft_features.shape}"
        #assert ft_features.shape[1] == config["num_channels"], f"Expected {config['num_channels']} channels, got {ft_features.shape[1]}"
        #assert ft_features.shape[2] == config["sequence_length"], f"Expected sequence length {config['sequence_length']}, got {ft_features.shape[2]}"
        ft_labels = torch.tensor(ft_labels, dtype=torch.long)
        ft_train_dataset = make_tensor_dataset(ft_features, ft_labels, config)

        novel_features, novel_labels = novel_user_data[pid]
        novel_features = torch.tensor(np.array(novel_features), dtype=torch.float32)
        #assert novel_features.ndim == 3, f"Expected 3D tensor (batch, channels, sequence), got {novel_features.ndim}D with shape {novel_features.shape}"
        #assert novel_features.shape[1] == config["num_channels"], f"Expected {config['num_channels']} channels, got {novel_features.shape[1]}"
        #assert novel_features.shape[2] == config["sequence_length"], f"Expected sequence length {config['sequence_length']}, got {novel_features.shape[2]}"
        novel_labels = torch.tensor(novel_labels, dtype=torch.long)
        novel_test_dataset = make_tensor_dataset(novel_features, novel_labels, config)

        # Create subject-specific dataloaders
        fine_tune_loader = DataLoader(ft_train_dataset, batch_size=config['ft_batch_size'], shuffle=True)
        novel_test_loader = DataLoader(novel_test_dataset, batch_size=config['ft_batch_size'], shuffle=False)

        # Fine-tune and evaluate the model
        ft_res = fine_tune_model(
            pretrained_model, fine_tune_loader, config, config["timestamp"], 
            test_loader=novel_test_loader, pid=pid)
        finetuned_model = ft_res['finetuned_model']
        metrics = evaluate_model(finetuned_model, novel_test_loader)
        user_accuracies.append(metrics["accuracy"])
        if config["save_ft_models"]:
            save_model(finetuned_model, model_str, config["models_save_dir"], f"{pid}_pretrainedFT", verbose=config["verbose"])
    # So this just returns the final accuracy for each user? No logs? I think that is fine...
    ## At least for optuna tuning purposes
    return user_accuracies


def validate_feature_data(feature_data, data_name):
    # TODO: This prints that it is empty and returns None...
    ## Fine for validation data (that is not set up yet), but what does this mean for test data??
    if feature_data is None:
        print(f"Key '{data_name}' does not exist.")
        return None
    elif isinstance(feature_data, (list, dict, str)) and not feature_data:
        #print(f"Feature data '{data_name}' is empty.")
        return None
    elif hasattr(feature_data, "size") and feature_data.size == 0:  # NumPy check
        #print(f"Feature data '{data_name}' is an empty NumPy array.")
        return None
    elif hasattr(feature_data, "numel") and feature_data.numel() == 0:  # PyTorch check
        #print(f"Feature data '{data_name}' is an empty tensor.")
        return None
    return feature_data  # Return valid data


def finetune_in_nb(config, datasplit, model, ft_user_pid=None):
    print("finetune_in_nb called!")  # I don't think this func gets used at all...

    #########################################################
    # TODO: Does this code get repeated verbatim for every single user and then discards the rest...
    ## Yah this could be precomputed (outside of finetune_in_nb) and then passed in...

    # Validate and then call function
    train_dict_key = 'novel_val_trainFT_dict'
    test_dict_key = 'novel_subject_val_test_dict'
    ft_features = validate_feature_data(datasplit.get(train_dict_key, {}).get('feature'), f"{train_dict_key}.feature")
    novel_features = validate_feature_data(datasplit.get(test_dict_key, {}).get('feature'), f"{test_dict_key}.feature")

    # Only continue if features exist, else switch from val to test (ie val isn't defined for this split)
    if ft_features is not None and novel_features is not None:
        pass
    else:
        print("Fixing empty val set")
        train_dict_key = 'novel_trainFT_dict'
        test_dict_key = 'novel_subject_test_dict'
        ft_features = validate_feature_data(datasplit.get(train_dict_key, {}).get('feature'), f"{train_dict_key}.feature")
        novel_features = validate_feature_data(datasplit.get(test_dict_key, {}).get('feature'), f"{test_dict_key}.feature")

        if ft_features is not None and novel_features is not None:
            pass
        else:
            raise ValueError("No data found/available")
    
    ft_labels = datasplit.get(train_dict_key, {}).get('labels', [])
    ft_pids = datasplit.get(train_dict_key, {}).get('participant_ids', [])
    novel_labels = datasplit.get(test_dict_key, {}).get('labels', [])
    novel_pids = datasplit.get(test_dict_key, {}).get('participant_ids', [])

    if ft_user_pid is None:
        ft_user_pid = novel_pids[0]
        print(f"PID was None! Choosing PID {ft_user_pid}")

    ft_user_data = group_data_by_participant(ft_features, ft_labels, ft_pids)
    novel_user_data = group_data_by_participant(novel_features, novel_labels, novel_pids)
    #########################################################
    
    ft_features, ft_labels = ft_user_data[ft_user_pid]
    # For some reason, ft_features is a array or arrays. So I turn it into a single numpy array which is much faster
    ft_features = torch.tensor(np.array(ft_features), dtype=torch.float32)
    #assert ft_features.ndim == 3, f"Expected 3D tensor (batch, channels, sequence), got {ft_features.ndim}D with shape {ft_features.shape}"
    #assert ft_features.shape[1] == config["num_channels"], f"Expected {config['num_channels']} channels, got {ft_features.shape[1]}"
    #assert ft_features.shape[2] == config["sequence_length"], f"Expected sequence length {config['sequence_length']}, got {ft_features.shape[2]}"
    ft_labels = torch.tensor(ft_labels, dtype=torch.long)
    # TODO: This line is breaking for some reason. ft_features empty?
    ft_train_dataset = make_tensor_dataset(ft_features, ft_labels, config)

    novel_features, novel_labels = novel_user_data[ft_user_pid]
    novel_features = torch.tensor(novel_features, dtype=torch.float32)
    #assert novel_features.ndim == 3, f"Expected 3D tensor (batch, channels, sequence), got {novel_features.ndim}D with shape {novel_features.shape}"
    #assert novel_features.shape[1] == config["num_channels"], f"Expected {config['num_channels']} channels, got {novel_features.shape[1]}"
    #assert novel_features.shape[2] == config["sequence_length"], f"Expected sequence length {config['sequence_length']}, got {novel_features.shape[2]}"
    novel_labels = torch.tensor(novel_labels, dtype=torch.long)
    novel_test_dataset = make_tensor_dataset(novel_features, novel_labels, config)

    # Create subject-specific dataloaders
    fine_tune_loader = DataLoader(ft_train_dataset, batch_size=config['ft_batch_size'], shuffle=True)
    novel_test_loader = DataLoader(novel_test_dataset, batch_size=config['ft_batch_size'], shuffle=False)

    ft_res = fine_tune_model(
                copy.deepcopy(model), fine_tune_loader, config, config["timestamp"], 
                test_loader=novel_test_loader, pid=ft_user_pid)
    finetuned_model = ft_res['finetuned_model']
    metrics = evaluate_model(finetuned_model, novel_test_loader)
    print(f"Acc: {(metrics['accuracy']*100):.2f}%")
    return metrics, finetuned_model


def verify_layer_freeze(model, print_layer_counts=True, print_layer_names=True):
    unfrozen_layers_count = 0
    frozen_layers_count = 0
    unfrozen_layers_lst = []
    frozen_layers_lst = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            unfrozen_layers_count += 1
            unfrozen_layers_lst.append(name)
        elif param.requires_grad==False:
            frozen_layers_count += 1
            frozen_layers_lst.append(name)
    total_layers_count = len(unfrozen_layers_lst) + len(frozen_layers_lst)
    
    if print_layer_counts==True and print_layer_names==False:
        print(f"Total layers: {total_layers_count}, unfrozen layers: {frozen_layers_count}, frozen layers: {frozen_layers_count}")
        print()
    elif print_layer_counts==False and print_layer_names==True:
        print(f"Unfrozen layers: \n{unfrozen_layers_lst}\n frozen layers: \n{frozen_layers_lst}")
        print()
    elif print_layer_counts==True and print_layer_names==True:
        print(f"Total layers: {total_layers_count}, unfrozen layers: {frozen_layers_count}, frozen layers: {frozen_layers_count}")
        print(f"Unfrozen layers: \n{unfrozen_layers_lst}\n frozen layers: \n{frozen_layers_lst}")
        print()


def save_model(model, model_str, save_dir, model_scenario_str, verbose=True, timestamp=None, save_with_timestamp=True):
    """Save the model with a timestamp."""
    if save_with_timestamp:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        full_path = os.path.join(save_dir, f'{timestamp}_{model_scenario_str}_{model_str}_model.pth')
    else:
        full_path = os.path.join(save_dir, f'{model_scenario_str}_{model_str}_model.pth')
    if verbose:
        print("Full Path:", full_path)
    torch.save(model.state_dict(), full_path)


# main function for hyperparameter tuning the finetuned models
def hyperparam_tuning_for_ft(model_str, expdef_df, hyperparameter_space, architecture_space, metadata_config,
                             num_configs_to_test=20, num_datasplits_to_test=2, num_train_trials=8, num_ft_trials=3, 
                             guaranteed_configs_to_test_lst=None, seed=100):
    
    print("Creating directories")
    # Results
    os.makedirs(metadata_config["results_save_dir"][0])
    print(f'Directory {metadata_config["results_save_dir"][0]} created successfully!')
    # Models
    os.makedirs(metadata_config["models_save_dir"][0]) 
    print(f'Directory {metadata_config["models_save_dir"][0]} created successfully!') 

    # Generate all possible configurations
    ## Does this like shuffle or is this deterministic?
    print("Combining configs")
    # Grid Search (generates full space, so O(N))
    #configs = list(ParameterGrid({**hyperparameter_space, **architecture_space, **metadata_config}))
    # Random search variant (samples so O(1))
    #configs = list(ParameterSampler({**hyperparameter_space, **architecture_space, **metadata_config}, n_iter=num_configs_to_test))
    configs = list(ParameterSampler(
        {**hyperparameter_space, **architecture_space, **metadata_config},
        n_iter=num_configs_to_test,
        random_state=seed
    ))
    # Shuffle the configurations
    random.shuffle(configs)
    configs = configs[:num_configs_to_test]  # Limit the number of configurations to test
    if guaranteed_configs_to_test_lst is not None:
        configs.extend(guaranteed_configs_to_test_lst)

    # This creates the (multiple) train/test splits
    print("Creating datasplits")
    data_splits_lst = []
    for datasplit in range(num_datasplits_to_test):
        all_participants = expdef_df['Participant'].unique()
        # Shuffle the participants for train/test user split --> UNIQUE
        random.shuffle(all_participants)
        test_participants = all_participants[24:]  # 24 train / 8 test
        data_splits_lst.append(prepare_data(
            expdef_df, 'feature', 'Gesture_Encoded', 
            all_participants, test_participants, 
            training_trials_per_gesture=num_train_trials, 
            finetuning_trials_per_gesture=num_ft_trials,
        ))

    results = []
    for config_idx, config in enumerate(configs):
        print(f"Testing config {config_idx + 1}/{len(configs)}:\n{config}")
        start_time = time.time()

        split_results = []
        for datasplit in data_splits_lst:
            # Here, each config+datasplit pair gets its own timestamp... not sure what is optimal
            ## Dont want anything to overwrite mainly
            ## TODO: Augment saving to create and save to folders (folders are timestamps?)
            ## I did that for results, but maybe should make the folder include the model name and scenario...
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Train the model
            training_results = main_training_pipeline(
                datasplit, all_participants=all_participants, test_participants=test_participants,
                config=config
            )
            pretrained_model = training_results["model"]

            # Evaluate the configuration on the current data split
            user_accuracies = evaluate_configuration_on_ft(datasplit, pretrained_model, config, model_str, timestamp)
            avg_accuracy = sum(user_accuracies) / len(user_accuracies)
            split_results.append({"avg_accuracy": avg_accuracy, "user_accuracies": user_accuracies})

        # Aggregate results across data splits
        overall_avg_accuracy = sum(split_result["avg_accuracy"] for split_result in split_results) / len(split_results)
        overall_user_accuracies = [acc for split_result in split_results for acc in split_result["user_accuracies"]]
        results.append({
            "config": config,
            "overall_avg_accuracy": overall_avg_accuracy,
            "overall_user_accuracies": overall_user_accuracies,
            "split_results": split_results
        })
        print(f"Overall accuracies: {overall_avg_accuracy:.4f}")
        total_time = time.time() - start_time
        print(f"Completed in {total_time:.2f}s\n")

    # Save the results
    ## This is the aggregated and sorted JSON file. This always needs to be saved
    save_results(results, metadata_config["results_save_dir"][0], metadata_config["timestamp"][0]) 

    return results


def load_expdef_gestures(feateng_method, laptop_data_save_path='C:\\Users\\\kdmen\\Box\\Yamagami Lab\\Data\\Meta_Gesture_Project', 
                         noFE_filename='\\noFE_windowed_segraw_allEMG.pkl',
                         moments_filename='\\moments_segraw_allEMG.csv', 
                         tdfs_filename='\\tdfs_segraw_allEMG.csv',
                         custom_path_bool=False):
    #print(f"Loading in data, with {feateng_method} feature engineering!")

    if custom_path_bool:
        with open(laptop_data_save_path, 'rb') as file:
            raw_expdef_data_df = pickle.load(file)
        # Rename only if "windowed_ts_data" is present
        if "windowed_ts_data" in raw_expdef_data_df.columns:
            expdef_df = raw_expdef_data_df.rename(columns={"windowed_ts_data": "feature"})
        else:
            expdef_df = raw_expdef_data_df  # Ensures expdef_df still gets assigned
    elif feateng_method=="None":
        with open(laptop_data_save_path+noFE_filename, 'rb') as file:
            raw_expdef_data_df = pickle.load(file)  # (204800, 19)
        expdef_df = raw_expdef_data_df.rename(columns={"windowed_ts_data": "feature"})
    elif feateng_method=="moments":
        moments_expdef_df = pd.read_csv(laptop_data_save_path+moments_filename).reset_index(drop=True)
        # Update so that all feature columns (for each row) are combined into a list or something and then treated as one column...
        first_three_columns = ['Participant', 'Gesture_ID', 'Gesture_Num']
        # Create a new dataframe with the first three columns
        expdef_df = moments_expdef_df[first_three_columns].copy()
        # Add the 'feature' column by concatenating the remaining columns
        expdef_df['feature'] = moments_expdef_df.drop(columns=first_three_columns).apply(lambda row: np.array(row.tolist()), axis=1)
    elif feateng_method=="FS":
        tdfs_expdef_df = pd.read_csv(laptop_data_save_path+tdfs_filename).reset_index(drop=True)
        # Update so that all feature columns (for each row) are combined into a list or something and then treated as one column...
        first_three_columns = ['Participant', 'Gesture_ID', 'Gesture_Num']
        # Create a new dataframe with the first three columns
        expdef_df = tdfs_expdef_df[first_three_columns].copy()
        # Add the 'feature' column by concatenating the remaining columns
        expdef_df['feature'] = tdfs_expdef_df.drop(columns=first_three_columns).apply(lambda row: np.array(row.tolist()), axis=1)
    else:
        raise ValueError(f"feateng_method {feateng_method} not recognized")
    
    #convert Gesture_ID to numerical with new Gesture_Encoded column
    label_encoder = LabelEncoder()
    expdef_df['Gesture_Encoded'] = label_encoder.fit_transform(expdef_df['Gesture_ID'])
    label_encoder2 = LabelEncoder()
    expdef_df['Cluster_ID'] = label_encoder2.fit_transform(expdef_df['Participant'])

    return expdef_df


def set_optimizer(model, lr, use_weight_decay, weight_decay, optimizer_name):
    """Configure optimizer with optional weight decay."""

    if optimizer_name.upper() == "ADAM":
        if use_weight_decay:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.upper() == "ADAMW":
        if use_weight_decay:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name.upper() == "SGD":
        if use_weight_decay:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Only ADAM, ADAMW, and SGD are supported right now")

    return optimizer


def make_data_split(
    expdef_df, config, 
    participants_lst=None, split_index=None, return_participants=False, 
    num_train_tpg=None, num_ft_tpg=None, use_only_these_encoded_gestures=None, 
    meta_learning_version=False
):
    """Loads data for either a single predefined user split or a specific train/test split based on the given index."""

    if participants_lst is None:
        # Load user splits
        try:
            with open(config["user_split_json_filepath"], "r") as f:
                splits = json.load(f)
        except FileNotFoundError:
            #updated_user_split_filepath = config["user_split_json_filepath"].replace('April_25\\', '')
            with open(config["user_split_json_filepath"], "r") as f:
                splits = json.load(f)

        # Handle both single fixed split and k-fold splits
        if split_index is None:
            try:
                all_participants = np.unique(list(splits["all_users"]))
                val_participants = splits["val_users"]
                test_participants = splits["test_users"]
            except:
                all_participants = np.unique(list(splits["train"])+list(splits["val"])+list(splits["test"]))
                val_participants = splits["val"]
                test_participants = splits["test"]
        else:  # kfold I think? Idk what split_index represents tho...
            if split_index < 0 or split_index >= len(splits):
                raise IndexError(f"Invalid split_index {split_index}. Must be between 0 and {len(splits)-1}.")
            all_participants = np.unique(list(splits[split_index]["train"]) + list(splits[split_index]["val"]))
            test_participants = splits[split_index]["val"]
    else:
        all_participants = participants_lst[0]
        val_participants = participants_lst[1]
        test_participants = participants_lst[2]

    # Prepare data (pass down meta_learning_version)
    data_splits = prepare_data(
        expdef_df, 'feature', 'Gesture_Encoded', 
        all_participants, test_participants, 
        training_trials_per_gesture=config["num_train_gesture_trials"] if num_train_tpg is None else num_train_tpg, 
        finetuning_trials_per_gesture=config["num_ft_gesture_trials"] if num_ft_tpg is None else num_ft_tpg, 
        use_only_these_encoded_gestures=use_only_these_encoded_gestures,
        meta_learning_version=meta_learning_version
    )

    if return_participants:
        return data_splits, all_participants, test_participants
    else:
        return data_splits


def prepare_data(
    df, feature_column, target_column, participants, test_participants, 
    training_trials_per_gesture=8, finetuning_trials_per_gesture=1, 
    use_only_these_encoded_gestures=None, meta_learning_version=False
):
    """
    Prepare data for training and testing across participants.
    If meta_learning_version=True, return dicts 'meta_train' and 'meta_test' per spec.
    """
    train_participants = [p for p in participants if p not in test_participants]

    # Initialize all the splits
    pretrain_data = {'feature': [], 'labels': [], 'participant_ids': []}
    pretrain_subject_test_data = {'feature': [], 'labels': [], 'participant_ids': []}
    novel_trainFT_data = {'feature': [], 'labels': [], 'participant_ids': []}
    novel_subject_test_data = {'feature': [], 'labels': [], 'participant_ids': []}
    novel_val_trainFT_data = {'feature': [], 'labels': [], 'participant_ids': []}
    novel_subject_val_test_data = {'feature': [], 'labels': [], 'participant_ids': []}
    
    for participant in participants:
        participant_data = df[df['Participant'] == participant]
        if use_only_these_encoded_gestures is not None:
            participant_data = participant_data[participant_data[target_column].isin(use_only_these_encoded_gestures)]
        gesture_groups = participant_data.groupby(target_column)
        
        if participant in train_participants:
            training_dict = pretrain_data
            testing_dict = pretrain_subject_test_data
            max_trials = training_trials_per_gesture
        elif participant in test_participants:
            training_dict = novel_trainFT_data
            testing_dict = novel_subject_test_data
            max_trials = finetuning_trials_per_gesture
        else:
            training_dict = novel_val_trainFT_data
            testing_dict = novel_subject_val_test_data
            max_trials = finetuning_trials_per_gesture
            
        for gesture, group in gesture_groups:
            group_features = np.array([x.flatten() for x in group[feature_column]])
            group_labels = group[target_column].values
            indices = np.random.choice(
                len(group_features), 
                size=min(max_trials, len(group_features)), 
                replace=False)
            train_features = group_features[indices]
            train_labels = group_labels[indices]
            training_dict['feature'].extend(train_features)
            training_dict['labels'].extend(train_labels)
            training_dict['participant_ids'].extend([participant] * len(train_labels))
            all_indices = np.arange(len(group_features)) 
            test_indices = np.setdiff1d(all_indices, indices)
            test_features = group_features[test_indices]
            test_labels = group_labels[test_indices]
            testing_dict['feature'].extend(test_features)
            testing_dict['labels'].extend(test_labels)
            testing_dict['participant_ids'].extend([participant] * len(test_labels))
    
    # Helper to stack/concat dictionaries (features and labels)
    def stack_dicts(dicts):
        features = np.concatenate([d['feature'] for d in dicts if len(d['feature']) > 0], axis=0) if any(len(d['feature']) > 0 for d in dicts) else np.array([])
        labels = np.concatenate([d['labels'] for d in dicts if len(d['labels']) > 0], axis=0) if any(len(d['labels']) > 0 for d in dicts) else np.array([])
        participant_ids = sum([d['participant_ids'] for d in dicts if len(d['participant_ids']) > 0], []) if any(len(d['participant_ids']) > 0 for d in dicts) else []
        return {'feature': features, 'labels': labels, 'participant_ids': participant_ids}
    
    if meta_learning_version:
        meta_train = stack_dicts([pretrain_data, pretrain_subject_test_data])
        meta_test = stack_dicts([
            novel_trainFT_data, novel_subject_test_data, 
            novel_val_trainFT_data, novel_subject_val_test_data
        ])
        return {
            'meta_train': meta_train,
            'meta_test': meta_test
        }
    else:
        return {
            'pretrain_dict': { 
                'feature': np.array(pretrain_data['feature']),
                'labels': np.array(pretrain_data['labels']),
                'participant_ids': pretrain_data['participant_ids']
            },
            'pretrain_subject_test_dict': { 
                'feature': np.array(pretrain_subject_test_data['feature']),
                'labels': np.array(pretrain_subject_test_data['labels']),
                'participant_ids': pretrain_subject_test_data['participant_ids']
            },
            'novel_trainFT_dict': { 
                'feature': np.array(novel_trainFT_data['feature']),
                'labels': np.array(novel_trainFT_data['labels']),
                'participant_ids': novel_trainFT_data['participant_ids']
            },
            'novel_subject_test_dict': { 
                'feature': np.array(novel_subject_test_data['feature']),
                'labels': np.array(novel_subject_test_data['labels']),
                'participant_ids': novel_subject_test_data['participant_ids']
            },
            'novel_val_trainFT_dict': { 
                'feature': np.array(novel_val_trainFT_data['feature']),
                'labels': np.array(novel_val_trainFT_data['labels']),
                'participant_ids': novel_val_trainFT_data['participant_ids']
            },
            'novel_subject_val_test_dict': { 
                'feature': np.array(novel_subject_val_test_data['feature']),
                'labels': np.array(novel_subject_val_test_data['labels']),
                'participant_ids': novel_subject_val_test_data['participant_ids']
            }
        }


def handle_batches(batch_idx, batch, num_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Add some config[verbose] or something here for data aug version
    ## Don't use config[verbose] verbatim bc it probably is used elsewhere in places idc about...
    #if num_batches is not None and (batch_idx==0 or batch_idx%500==0):
    #    print(f"Starting batch {batch_idx}/{num_batches}!")

    # Unpack depending on if batch is a tuple or list
    if len(batch) == 4:
        #features, (now encoded) gesture_names, participant_ids, gesture_nums = batch
        batch_features, batch_labels, _, _ = batch
        batch_features = batch_features.reshape(-1, 16, 5)

        # If labels are numpy or wrong type, convert to torch.long and move to device
        if not torch.is_tensor(batch_labels):
            batch_labels = torch.tensor(batch_labels, dtype=torch.LongTensor)
        else:
            if batch_labels.dtype != torch.long:
                batch_labels = batch_labels.long()
    elif len(batch) == 2:
        batch_features, batch_labels = batch
    else:
        raise ValueError(f"Expected either 2 or 4 elements from dataset, got {len(batch)}")
    
    batch_features = batch_features.to(device)
    batch_labels = batch_labels.to(device)
    #print("Batch set up complete!")
    return batch_features, batch_labels


def train_model(model, train_loader, optimizer, criterion=None, moe_version=False, config=None):
    """Train the model for one epoch. DOES NOT FULLY TRAIN THE MODEL"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    total_loss = 0
    correct_predictions = 0
    num_batches = len(train_loader)
    total_samples = 0

    #print("Forward pass complete!")
    if criterion is None and moe_version is True:
        # keep label smoothing if you want
        criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    elif criterion is None and moe_version is False:
        # keep label smoothing if you want
        criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(train_loader):
        batch_features, batch_labels = handle_batches(batch_idx, batch, num_batches=num_batches)

        optimizer.zero_grad()
        outputs = model(batch_features)

        # MOE Version returns logits and aux (gate usage log), so have to handle that specifically
        # --- accept tuple/dict/anything and extract logits tensor ---
        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif isinstance(outputs, dict):
            # common keys people use; fall back to first tensor value
            logits = outputs.get("logits", next(v for v in outputs.values() if isinstance(v, torch.Tensor)))
        else:
            logits = outputs

        loss = criterion(logits, batch_labels)
        #print("Loss complete!")
        loss.backward()
        #print("Backprop complete!")
        optimizer.step()
        #print("Optimization complete!")
        
        total_loss += loss.item()
            
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': correct_predictions / total_samples}


def evaluate_model(model, dataloader, criterion=nn.CrossEntropyLoss()):
    """Evaluate the model and return detailed performance metrics"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO: This ought to be part of config...
    model.to(device)
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_features, batch_labels = handle_batches(batch_idx, batch, num_batches=num_batches)

            #print("EVAL Batch features shape:", batch_features.shape)
            #print("EVAL Batch labels shape:", batch_labels.shape)
            
            outputs = model(batch_features)
            # MOE Version returns logits and aux (gate usage log), so have to handle that specifically
            # --- accept tuple/dict/anything and extract logits tensor ---
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                # common keys people use; fall back to first tensor value
                logits = outputs.get("logits", next(v for v in outputs.values() if isinstance(v, torch.Tensor)))
            else:
                logits = outputs

            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': correct_predictions / total_samples,
    }


def gesture_performance_by_participant(predictions, true_labels, all_unique_participants, 
                                       all_shuffled_participant_ids, unique_gestures):
    """
    Calculate performance metrics for each participant and gesture
    
    Returns:
    - Dictionary with performance for each participant and gesture
    """
    performance = {}
    
    for participant in all_unique_participants:
        participant_mask = np.array(all_shuffled_participant_ids) == participant
        participant_preds = np.array(predictions)[participant_mask]
        participant_true = np.array(true_labels)[participant_mask]
        
        participant_performance = {}
        for gesture in unique_gestures:
            gesture_mask = participant_true == gesture
            if np.sum(gesture_mask) > 0:
                gesture_preds = participant_preds[gesture_mask]
                gesture_true = participant_true[gesture_mask]
                participant_performance[gesture] = np.mean(gesture_preds == gesture_true)
        
        performance[participant] = participant_performance
    
    return performance


def initialize_weights_xavier(model):
    """
    Apply Xavier initialization to the weights of the model.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def main_training_pipeline(data_splits, all_participants, test_participants, config, 
                           train_intra_cross_loaders=None, save_loss_here=False, set_init_model_weights=False, single_participant=False, scenario=""):
    """
    Main training pipeline with comprehensive performance tracking
    
    Args:
    - data_splits: dictionary (NOT DATALOADERS) with keys according to scenario and train/test
    - all_participants: List of all (unique) participants
    - test_participants: List of participants to hold out
    - config: dictionary of configuration values for model architecture and hyperparams
    - train_intra_cross_loaders: ... not sure what this is, it seems to be left to the default None
    Returns:
    - Trained model, performance metrics
    """

    bs = config["batch_size"]
    ## Could be used in train_model and eval_model but I am content using nn.crossentropy by default
    if scenario.upper()=="LOCAL":
        # Local should actually used the ft_learning_rate bc it ahs the same num samples, otherwise is using the pretraining lr which is tiny
        lr = config["ft_learning_rate"]
        max_epochs = config["num_ft_epochs"]  # Local should train an equivalent amount to the other FT methods!
    else:
        lr = config["learning_rate"]
        max_epochs = config["num_epochs"]
    weight_decay = config["weight_decay"]
    #sequence_length = config["sequence_length"]
    #time_steps = config["time_steps"]

    # Not used right not (we are only using CPU anyways...)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_intra_cross_loaders is not None:  # This is used in the NB full study...
        if single_participant:
            # Shuffling doesnt matter if there's only one pid
            pass
        else:
            print("POSSIBLY PROBLEMATIC BRANCH: pids not aligned with shuffled dataloaders!")

        train_loader = train_intra_cross_loaders[0]
        intra_test_loader = train_intra_cross_loaders[1]
        cross_test_loader = train_intra_cross_loaders[2]
        #if len(train_intra_cross_loaders)==4:
        #    ft_loader = train_intra_cross_loaders[3]  # Not used at all...

        # TODO: train_pids is not shuffled like the dataloader and thus will be off...
        # TODO: This exlucdes the case where we are training and testing on the same subjects...
        train_pids = [pid for pid in all_participants if pid not in test_participants]
        # This ought to be changed as well... this should be the unshuffled version (which it currently is)
        #intra_pids = train_pids
        #cross_pids = test_participants
    else:
        # Shuffle features, labels, and participant IDs together
        ## Otherwise train_pids will not map correctly to the data in the dataloader that gets shuffled!
        train_features, train_labels, train_pids = shuffle(
            data_splits['pretrain_dict']['feature'], 
            data_splits['pretrain_dict']['labels'], 
            data_splits['pretrain_dict']['participant_ids'], 
            random_state=17  # Use a fixed seed for reproducibility
            # TODO: Get this random state to sync with global_seed (hardcoded to match rn...)
        )

        train_features = torch.tensor(train_features, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        train_dataset = make_tensor_dataset(train_features, train_labels, config)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)  # No shuffling here, already shuffled in the above!
        
        # INTRA SUBJECT (480)
        intra_test_features = data_splits['pretrain_subject_test_dict']['feature']
        intra_test_labels = data_splits['pretrain_subject_test_dict']['labels']
        intra_test_features = torch.tensor(intra_test_features, dtype=torch.float32)
        intra_test_labels = torch.tensor(intra_test_labels, dtype=torch.long)
        intra_test_dataset = make_tensor_dataset(intra_test_features, intra_test_labels, config)
        # Shuffle doesn't matter for testing
        intra_test_loader = DataLoader(intra_test_dataset, batch_size=bs, shuffle=False) #, drop_last=True)

        # CROSS (NOVEL USER) SUBJECT (560)
        cross_test_features = data_splits['novel_subject_test_dict']['feature']
        cross_test_labels = data_splits['novel_subject_test_dict']['labels']
        cross_test_features = torch.tensor(cross_test_features, dtype=torch.float32)
        cross_test_labels = torch.tensor(cross_test_labels, dtype=torch.long)
        cross_test_dataset = make_tensor_dataset(cross_test_features, cross_test_labels, config)     
        # Shuffle doesn't matter for testing
        cross_test_loader = DataLoader(cross_test_dataset, batch_size=bs, shuffle=False) #, drop_last=True)

    # Select model
    model = select_model(config['model_str'], config)
    initialize_weights_xavier(model)
    #print(model)
    if set_init_model_weights:
        # Apply Xavier weight initialization
        initialize_weights_xavier(model)
    # Loss and optimizer
    optimizer = set_optimizer(model, lr=lr, use_weight_decay=weight_decay>0, weight_decay=weight_decay, optimizer_name=config["optimizer"])
    if config['use_earlystopping']:
        early_stopping = SmoothedEarlyStopping(patience=config["earlystopping_patience"], min_delta=config["earlystopping_min_delta"])
    if config["lr_scheduler_factor"]>0.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config["lr_scheduler_patience"], factor=config["lr_scheduler_factor"])

    # Training
    train_loss_log = []
    intra_test_loss_log = []
    cross_test_loss_log = []
    train_acc_log = []
    intra_test_acc_log = []
    cross_test_acc_log = []

    if save_loss_here: 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_file = open(f"{config['results_save_dir']}\\{timestamp}_{config['model_str']}_pretrained_training_log.txt", "w")

    epoch = 0
    done = False
    while not done and epoch < max_epochs:
        epoch += 1

        train_metrics = train_model(model, train_loader, optimizer)
        #train_metrics = evaluate_model(model, train_loader)
        train_loss_log.append(train_metrics['loss'])
        train_acc_log.append(train_metrics['accuracy'])

        #train_metrics2 = evaluate_model(model, train_loader)
        
        # Validation
        intra_test_metrics = evaluate_model(model, intra_test_loader)
        intra_test_loss_log.append(intra_test_metrics['loss'])
        intra_test_acc_log.append(intra_test_metrics['accuracy'])
        cross_test_metrics = evaluate_model(model, cross_test_loader)
        cross_test_loss_log.append(cross_test_metrics['loss'])
        cross_test_acc_log.append(cross_test_metrics['accuracy'])

        # Anneal the learning rate (advance scheduler) if applicable
        #if config["lr_scheduler_gamma"]<1.0:
        #    scheduler.step() 
        if config["lr_scheduler_factor"]>0.0:
            # Reduce LR if validation loss plateaus
            # Get current LR(s) before stepping
            #prev_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            scheduler.step(intra_test_loss_log[-1])
            #new_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            # Compare before and after
            #for i, (prev_lr, new_lr) in enumerate(zip(prev_lrs, new_lrs)):
            #    if new_lr < prev_lr:
            #        print(f"Learning rate reduced for param group {i} on epoch {epoch}: {prev_lr:.5e} → {new_lr:.5e}")
        # Early stopping check
        if config['use_earlystopping'] and early_stopping(intra_test_loss_log[-1]):
            print(f"MainTrn: Early stopping reached after {epoch} epochs")
            done = True

        # Log metrics to the console and the text file
        log_message = (
            f"Epoch {epoch}/{max_epochs}, "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Intra Testing Loss: {intra_test_loss_log[-1]:.4f}, "
            f"Cross Testing Loss: {cross_test_loss_log[-1]:.4f}\n")
            #f"{earlystopping_status}\n")
        if config["verbose"]:
            print(log_message, end="")  # Print to console
        if save_loss_here:
            log_file.write(log_message)  # Write to file
    if save_loss_here:
        # Close the log file
        log_file.close()
    
    # Evaluation --> One-off final results! Gets used in gesture_performance_by_participant
    train_results = evaluate_model(model, train_loader)
    intra_test_results = evaluate_model(model, intra_test_loader)
    cross_test_results = evaluate_model(model, cross_test_loader)

    return {
        'model': model,
        'train_accuracy': train_results['accuracy'],
        'intra_test_accuracy': intra_test_results['accuracy'],
        'cross_test_accuracy': cross_test_results['accuracy'], 
        # Train/test curves
        'train_loss_log': train_loss_log,
        'intra_test_loss_log': intra_test_loss_log,
        'cross_test_loss_log': cross_test_loss_log,
        'train_acc_log': train_acc_log,
        'intra_test_acc_log': intra_test_acc_log,
        'cross_test_acc_log': cross_test_acc_log
    }


def process_split(data_splits, split_key, label_encoder):
        features_df = pd.DataFrame(data_splits[split_key]['feature'])
        features_df['feature'] = features_df.apply(lambda row: row.tolist(), axis=1)
        features_df = features_df[['feature']]
        df = pd.concat([
            features_df, 
            pd.Series(data_splits[split_key]['labels'], name='Gesture_Encoded'), 
            pd.Series(data_splits[split_key]['participant_ids'], name='participant_ids')
        ], axis=1)
        df['Cluster_ID'] = label_encoder.transform(df['participant_ids'])
        return df


def print_trainable_layers(model):
    print("=== Trainable (Unfrozen) Parameters ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  [UNFROZEN] {name} | shape: {tuple(param.shape)}")
    print("=== Frozen Parameters ===")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  [FROZEN]   {name} | shape: {tuple(param.shape)}")


def assert_some_params_frozen(model):
    """Raise AssertionError if all parameters are unfrozen (i.e., all require_grad=True)."""
    all_trainable = all(param.requires_grad for param in model.parameters())
    if all_trainable:
        raise AssertionError("No layers are frozen! All model parameters are trainable.")
    

def assert_only_last_linear_trainable(model):
    """
    Assert that only the last nn.Linear in the model has requires_grad=True,
    and all other parameters are frozen.
    """
    last_linear = None
    last_linear_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            last_linear_name = name

    if last_linear is None:
        raise AssertionError("No nn.Linear found in model.")

    # Now check that only last_linear is trainable
    problems = []
    for name, param in model.named_parameters():
        # Check if param belongs to last_linear
        is_last_linear_param = name.startswith(last_linear_name)
        if is_last_linear_param:
            if not param.requires_grad:
                problems.append(f"Param {name} in last nn.Linear should be trainable but is frozen.")
        else:
            if param.requires_grad:
                problems.append(f"Param {name} is not in last nn.Linear but is trainable (should be frozen).")
    if problems:
        raise AssertionError("Layer freezing error:\n" + "\n".join(problems))


def fine_tune_model(finetuned_model, fine_tune_loader, config, timestamp, test_loader=None, pid=None, num_epochs=None):  #use_earlystopping=None,
    """    
    Args:
    - finetuned_model: Model to fine-tune (note that is shouldn't be finetuned yet)
    - fine_tune_loader: Dataloader of finetuned data and labels
    - config: The dictionary with config/hyper params
    Returns:
    - Dictionary including the finetuned model, and various performance logs
    """

    finetune_strategy = config["finetune_strategy"]
    #print(f"Beginning {finetune_strategy} finetuning!")
    '''
    - finetune_strategy: The fine-tuning method to use. Options:
        - "full": Train/update the entire model, this is basically just another training run
        - "freeze_cnn": Freeze CNN, train the rest of the network (CNN as feature extractor)
        - "freeze_all_but_final_dense"/"linear_probing": Only train/update/finetune the final dense layer
        - "progressive_unfreeze": Start with everything but final layer frozen, progressively unfreeze one layer at a time.
    '''

    if pid is None:
        pid = ""
    else:
        pid = pid + "_"

    # TODO: If we have early stopping on then this still finishes early right (doesnt run to specified number of epochs)?
    if num_epochs is None:
        max_epochs = config["num_ft_epochs"]
    else:
        max_epochs = num_epochs

    finetuned_model.train()  # Model is in training mode

    # Apply fine-tuning strategy
    if finetune_strategy == "full":
        # We don't need to do any layer freezing
        ## Eg we will just start retraining the entire model
        pass
    elif finetune_strategy == "freeze_cnn":
        for name, module in finetuned_model.named_modules():
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv2d):
                for param in module.parameters():
                    param.requires_grad = False
            else:
                if config["reset_ft_layers"]:
                    # Extract the module that owns this parameter
                    layer_name = name.rsplit(".", 1)[0]  # Get the parent module's name
                    layer = dict(finetuned_model.named_modules()).get(layer_name, None)
                    # Reset only if the layer supports reset_parameters()
                    if layer and hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()  # Reset to random initialization
        assert_some_params_frozen(finetuned_model)
    elif finetune_strategy in ["freeze_all_but_final_dense", "linear_probing"]:  # These are the same
        # Freeze everything
        for param in finetuned_model.parameters():
            param.requires_grad = False

        # Find last nn.Linear in model (could be nested or in Sequential)
        last_linear = None
        last_linear_name = None
        for name, module in finetuned_model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                last_linear_name = name

        if last_linear is None:
            raise RuntimeError("No nn.Linear found in model! Cannot perform linear probing.")

        # Optionally reset the last linear layer's weights
        if config.get("reset_ft_layers", False):
            if hasattr(last_linear, 'reset_parameters'):
                last_linear.reset_parameters()
            #else:
            #    print(f"[WARN] Last linear layer ({last_linear_name}) does not support reset_parameters()")

        # Unfreeze parameters of last linear layer
        for param in last_linear.parameters():
            param.requires_grad = True

        #print(f"[INFO] Linear probing: Unfroze and reset (if requested) last linear layer '{last_linear_name}'.")

        assert_only_last_linear_trainable(finetuned_model)
    elif finetune_strategy == "progressive_unfreeze":
        # Freeze everything first
        for param in finetuned_model.parameters():
            param.requires_grad = False
        
        # Identify trainable layers in reverse order (last layers first)
        trainable_layers = []
        # Final dense layers:
        if hasattr(finetuned_model, 'fc_layers'):
            trainable_layers.append(finetuned_model.fc_layers)
        else:
            # Fallback: Use fc2 if fc_layers doesn't exist
            if hasattr(finetuned_model, 'fc2'):
                trainable_layers.append(finetuned_model.fc2)
            if hasattr(finetuned_model, 'fc1'):
                trainable_layers.append(finetuned_model.fc1)
            if hasattr(finetuned_model, 'fc'):
                trainable_layers.append(finetuned_model.fc)
        # LSTM (if present)
        if hasattr(finetuned_model, 'lstm'):
            trainable_layers.append(finetuned_model.lstm)
        # CNN 
        trainable_layers.append(finetuned_model.conv_layers)
        # Unfreeze the very last layer so it can FT at start
        ## Index 0 but we saved in reserve order remember
        for param in trainable_layers[0].parameters():
                param.requires_grad = True
        # Gradually unfreeze starting from the last layer
        current_unfreeze_step = 1  # Start on 1 since we already unfroze one layer
        total_steps = len(trainable_layers)

        assert_some_params_frozen(finetuned_model)
    else:
        raise ValueError(f"finetune_strategy ({finetune_strategy}) not recognized!")
    
    #print_trainable_layers(finetuned_model)

    # Loss and optimizer (with different learning rate for fine-tuning)
    optimizer = set_optimizer(finetuned_model, lr=config["ft_learning_rate"], use_weight_decay=config["ft_weight_decay"] > 0, weight_decay=config["ft_weight_decay"], optimizer_name=config["optimizer"])
    if config["use_earlystopping"]==True:
        early_stopping = SmoothedEarlyStopping(patience=config["ft_earlystopping_patience"], min_delta=config["ft_earlystopping_min_delta"])
    if config["ft_lr_scheduler_factor"]>0.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config["ft_lr_scheduler_patience"], factor=config["ft_lr_scheduler_factor"])#, verbose=False)

    # Fine-tuning training loop
    ## Not tracking cross for finetuning (who cares)
    train_loss_log = []
    intra_test_loss_log = []
    #cross_test_loss_log = []
    train_acc_log = []
    intra_test_acc_log = []
    #cross_test_acc_log = []
    epoch = 0
    done = False
    # Initialize a generator for reproducible shuffling (that can be different for each epoch!)
    #dl_shuffler_generator = torch.Generator()
    # Open a text file for logging
    if config["log_each_pid_results"]:
        log_file = open(f"{timestamp}_{pid}ft_log.txt", "w")
    while not done and epoch < max_epochs:
        epoch += 1
        #dl_shuffler_generator.manual_seed(epoch)
        # Reinitialize DataLoader with the new shuffle
        #fine_tune_loader = DataLoader(ft_dataset, batch_size=config["batch_size"], shuffle=True, generator=dl_shuffler_generator)

        # Progressive unfreezing logic
        if (finetune_strategy == "progressive_unfreeze" and
            epoch % config["progressive_unfreezing_schedule"] == 0 and 
            current_unfreeze_step < total_steps):
            # Unfreeze the next layer group
            for param in trainable_layers[current_unfreeze_step].parameters():
                param.requires_grad = True
            current_unfreeze_step += 1

        # FOR MOE: Handle model's that require a user-gate-embedding
        if config.get("gate_requires_u_user", None) is True:
            # Here I ... either need to warm start and do no PEFT (probably not great) or do warm start and PEFT
            if config.get("use_u_init_warm_start", None) is True:
                u_init = uservec_from_support_mean(finetuned_model, fine_tune_loader, config)
            else:
                u_init = None
            u_user, moe_logs = peft_user_emb_vec(finetuned_model, fine_tune_loader, config, u_init=u_init)
            finetuned_model = WithUserOverride(finetuned_model, u_user)

        train_metrics = train_model(finetuned_model, fine_tune_loader, optimizer)
        #train_loss_log.append(train_loss)
        #train_metrics = evaluate_model(finetuned_model, fine_tune_loader)
        train_loss_log.append(train_metrics['loss'])
        train_acc_log.append(train_metrics['accuracy'])
        # Validation
        if test_loader is not None:
            intra_test_metrics = evaluate_model(finetuned_model, test_loader)
            intra_test_loss_log.append(intra_test_metrics['loss'])
            intra_test_acc_log.append(intra_test_metrics['accuracy'])

        # NOTE: You don't have to call / check the loss every time and compare against patience, both these do that internally (allegedly)
        if config["ft_lr_scheduler_factor"]>0.0:
            # Reduce LR if validation loss plateaus
            scheduler.step(intra_test_loss_log[-1])
        # Early stopping check
        if config["use_earlystopping"]==True and early_stopping(intra_test_loss_log[-1]):
            print(f"FT {pid}: Early stopping reached after {epoch} epochs")
            done = True

    # Log metrics to the console and the text file, AFTER the while loop has finished
    log_message = (
        f"Participant ID {pid[:-1]}, "  # [] in order to drop the "_" 
        f"Epoch {epoch}/{max_epochs}, "
        # TODO: Why are these losses and not accuracies...
        f"FT Train Loss: {train_loss_log[-1]:.4f}, "
        f"Novel Intra Subject Testing Loss: {intra_test_loss_log[-1]:.4f}\n")
        #f"{earlystopping_status}\n")
    if config["verbose"]:
            print(log_message, end="")  # Print to console
    if config["log_each_pid_results"]:
        log_file.write(log_message)  # Write to file
        # Close the log file
        log_file.close()

    if test_loader is not None:
        final_intra_test_acc = evaluate_model(finetuned_model, test_loader)['accuracy']
    else:
        final_intra_test_acc = []

    return {
            'finetuned_model': finetuned_model,
        #    # These are the final accuracies on the respective datasets
            'train_accuracy': evaluate_model(finetuned_model, fine_tune_loader)['accuracy'],
            'intra_test_accuracy': final_intra_test_acc,
        #    'cross_test_accuracy': cross_test_results['accuracy'], 
            'train_loss_log': train_loss_log,
            'train_acc_log': train_acc_log, 
            'intra_test_loss_log': intra_test_loss_log, 
            'intra_test_acc_log': intra_test_acc_log
        }


def log_performance(results, config, base_filename='model_performance_and_config'):
    """
    Comprehensive logging of model performance
    
    Args:
    - results: Dictionary containing training and testing performance
    - log_dir: Directory to save log files
    - base_filename: Base name for log files
    
    Returns:
    - Path to the created log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(config["results_save_dir"], exist_ok=True)
    
    # Generate unique filename with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    timestamp = config["timestamp"]
    log_filename = f"{timestamp}_{base_filename}.txt"
    log_path = os.path.join(config["results_save_dir"], log_filename)
    
    # Capture console output and log to file
    class Logger:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, "w")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Redirect stdout to both console and file
    sys.stdout = Logger(log_path)
    
    try:
        if config is not None:
            for key, value in config.items():
                # Handle list values separately for better formatting
                if isinstance(value, list):
                    print(f"{key}: {', '.join(map(str, value))}")
                else:
                    print(f"{key}: {value}")
            print()

        # Perform visualization and logging
        print("Model Performance Analysis")
        print("=" * 30)
        
        # Training Set Performance
        print("\n--- Training Set Performance ---")
        for participant, gesture_performance in results['train_performance'].items():
            if participant in results['train_performance'].keys():
                continue
            print(f"\nParticipant {participant}:")
            participant_accuracies = []
            for gesture, accuracy in sorted(gesture_performance.items()):
                print(f"  Gesture {gesture}: {accuracy:.2%}")
                participant_accuracies.append(accuracy)
            print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
        # Training Set Summary
        train_accuracies = [acc for participant in results['train_performance'].values() for acc in participant.values()]
        print("\nTraining Set Summary:")
        print(f"Mean Accuracy: {np.mean(train_accuracies):.2%}")
        print(f"Accuracy Standard Deviation: {np.std(train_accuracies):.2%}")
        print(f"Minimum Accuracy: {np.min(train_accuracies):.2%}")
        print(f"Maximum Accuracy: {np.max(train_accuracies):.2%}")

        ######################################################################
        
        # Repeat for Testing Set
        print("\n--- INTRA Testing Set Performance ---")
        for participant, gesture_performance in results['intra_test_performance'].items():
            print(f"\nParticipant {participant}:")
            participant_accuracies = []
            for gesture, accuracy in sorted(gesture_performance.items()):
                print(f"  Gesture {gesture}: {accuracy:.2%}")
                participant_accuracies.append(accuracy)
            print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
        # Testing Set Summary
        test_accuracies = [acc for participant in results['intra_test_performance'].values() for acc in participant.values()]
        print("\nIntra Testing Set Summary:")
        print(f"Mean Accuracy: {np.mean(test_accuracies):.2%}")
        print(f"Accuracy Standard Deviation: {np.std(test_accuracies):.2%}")
        print(f"Minimum Accuracy: {np.min(test_accuracies):.2%}")
        print(f"Maximum Accuracy: {np.max(test_accuracies):.2%}")

        ######################################################################
        # Idrc about this right now

        # Repeat for Testing Set
        #print("\n--- CROSS Testing Set Performance ---")
        #for participant, gesture_performance in results['cross_test_performance'].items():
        #    print(f"\nParticipant {participant}:")
        #    participant_accuracies = []
        #    for gesture, accuracy in sorted(gesture_performance.items()):
        #        print(f"  Gesture {gesture}: {accuracy:.2%}")
        #        participant_accuracies.append(accuracy)
        #    print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
        ## Testing Set Summary
        #test_accuracies = [acc for participant in results['cross_test_performance'].values() for acc in participant.values()]
        #print("\nCross Testing Set Summary:")
        #print(f"Mean Accuracy: {np.mean(test_accuracies):.2%}")
        #print(f"Accuracy Standard Deviation: {np.std(test_accuracies):.2%}")
        #print(f"Minimum Accuracy: {np.min(test_accuracies):.2%}")
        #print(f"Maximum Accuracy: {np.max(test_accuracies):.2%}")

        ######################################################################
        
        # Overall Model Performance
        print("\nOverall Model Performance:")
        print(f"Training Accuracy: {results['train_accuracy']:.2%}")
        print(f"Intra Testing Accuracy: {results['intra_test_accuracy']:.2%}")
        #print(f"Cross Testing Accuracy: {results['cross_test_accuracy']:.2%}")
    
    finally:
        # Restore stdout
        sys.stdout = sys.stdout.terminal
    
    #return log_path

