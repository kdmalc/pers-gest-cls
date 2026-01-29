import pandas as pd
import numpy as np
import copy
from torch.utils.data import DataLoader
import torch

from utils.agglo_model_clust import *
from utils.DNN_FT_funcs import *
from utils.AMC_extras import *


def train_DNN_cluster_model(train_df, test_df, cluster_ids, config, 
                                   cluster_column='Cluster_ID', feature_column='feature', 
                                   target_column='Gesture_Encoded'):

    max_epochs = config["num_epochs"]
    bs = config["batch_size"]
    lr = config["learning_rate"]

    clus_model_dict = {}
    cluster_logs_dict = {}
    for cluster in cluster_ids:
        #print(f"Processing cluster {cluster}")
        cluster_logs_dict[cluster] = {}

        # Filter data for the current cluster
        cluster_train_data = train_df[train_df[cluster_column] == cluster]
        if cluster_train_data.empty:
            # Could be for test or once it gets merged? Idk
            raise ValueError("Why is this cluster empty")
        cluster_test_data = test_df[test_df[cluster_column] == cluster]
        if cluster_test_data.empty:
            # Could be for test or once it gets merged? Idk
            raise ValueError("Why is this cluster empty")

        X_train = np.array([x for x in cluster_train_data[feature_column]])
        y_train = np.array(cluster_train_data[target_column])
        X_val = np.array([x for x in cluster_test_data[feature_column]])
        y_val = np.array(cluster_test_data[target_column])

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("One of these are empty...")

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        # Create DataLoaders
        train_loader = DataLoader(make_tensor_dataset(X_train_tensor, y_train_tensor, config), batch_size=bs, shuffle=True)
        val_loader = DataLoader(make_tensor_dataset(X_val_tensor, y_val_tensor, config), batch_size=bs, shuffle=False)

        # Initialize model and optimizer
        fold_model = select_model(config["model_str"], config)

        optimizer = set_optimizer(fold_model, lr=lr, 
                                    use_weight_decay=config["weight_decay"]>0,
                                    weight_decay=config["weight_decay"], 
                                    optimizer_name=config["optimizer"])

        # Training loop with early stopping
        if config["use_earlystopping"]:
            earlystopping = EarlyStopping()
        epoch = 0
        done = False
        train_loss_log = []
        val_loss_log = []
        while not done and epoch < max_epochs:
            epoch += 1
            train_loss = train_model(fold_model, train_loader, optimizer)
            train_res = evaluate_model(fold_model, train_loader)
            val_res = evaluate_model(fold_model, val_loader)
            train_loss_log.append(train_res['loss'])
            val_loss_log.append(val_res['loss'])

            if config["use_earlystopping"]==True and earlystopping(fold_model, val_res['loss']):
                done = True

            if config["verbose"]:
                print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_res['loss']:.4f}")

        clus_model_dict[cluster] = copy.deepcopy(fold_model)
        cluster_logs_dict[cluster]["train_loss_log"] = copy.deepcopy(train_loss_log)
        cluster_logs_dict[cluster]["val_loss_log"] = copy.deepcopy(val_loss_log)

    return clus_model_dict, cluster_logs_dict


def DNN_agglo_merge_procedure(data_dfs_dict, config):
    print("DNN_agglo_merge_procedure started!")
    
    train_df = data_dfs_dict['pretrain_df']
    intra_test_df = data_dfs_dict['pretrain_subject_test_df']
    
    # Initialize model tracking structures
    clus_model_dict = {}  # Persists across iterations
    previous_clusters = set()
    nested_clus_model_dict = {}

    # Data structures for logging
    merge_log = []
    unique_clusters_log = []
    intra_cluster_performance = {}
    cross_cluster_performance = {}

    # Load predefined user splits from JSON
    #with open(config["user_split_json_filepath"], "r") as f:
    #    splits = json.load(f)
    # Determine if splits are a single dict or list of splits
    #predefined_splits = [splits] if isinstance(splits, dict) else splits
    #n_splits = len(predefined_splits)  # Override n_splits with actual count
    n_splits = 1  # Only 1 split is allowed right now
    # Do I need this? As long as I am using split_index (ie not trying to run all kfcv's at once), I think this should be fine actually
    #assert("kfcv" not in config["user_split_json_filepath"])
    if n_splits != 1:
        raise ValueError("kfcv not supported here yet...")
    #for i in range(n_splits):  # If we have kfcv this will loop over all our splits (I dont think I should be kfcv'ing here tho)
    iterations = 0
    all_clus_logs_dict = {}
    while len(train_df['Cluster_ID'].unique()) > 1:
        print(f"Iter {iterations}: {len(train_df['Cluster_ID'].unique())} Clusters Remaining")
        current_clusters = sorted(train_df['Cluster_ID'].unique())
        current_cluster_set = set(current_clusters)
        unique_clusters_log.append(current_clusters)

        # Identify new clusters that need training
        new_clusters = current_cluster_set - previous_clusters
        print(f"New clusters to train: {new_clusters}")

        # Train only new clusters
        if new_clusters:
            new_models, cluster_logs_dict = train_DNN_cluster_model(
                train_df, intra_test_df, list(new_clusters), config
            )
            clus_model_dict.update(new_models)
        all_clus_logs_dict[iterations] =copy.deepcopy(cluster_logs_dict)

        # Remove models for merged clusters (no longer exist)
        clus_model_dict = {k: v for k, v in clus_model_dict.items() if k in current_cluster_set}

        # Log current state of models
        nested_clus_model_dict[f"Iter{iterations}"] = copy.deepcopy(clus_model_dict)

        # Test all current models on current clusters
        current_models = [clus_model_dict[clus] for clus in current_clusters]
        sym_acc_arr = test_models_on_clusters(intra_test_df, current_models, current_clusters, config, pytorch_bool=True)

        for idx, cluster_id in enumerate(current_clusters):
            cross_acc_sum = 0
            cross_acc_count = 0

            for idx2, cluster_id2 in enumerate(current_clusters):
                if cluster_id not in intra_cluster_performance:
                    intra_cluster_performance[cluster_id] = []  # Initialize list

                if idx == idx2:  # Diagonal, so intra-cluster
                    # Ensure the logic assumption holds
                    if cluster_id != cluster_id2:
                        raise ValueError("This code isn't working as expected...")
                    intra_cluster_performance[cluster_id].append((iterations, sym_acc_arr[idx, idx2]))
                else:  # Non-diagonal, so cross-cluster
                    cross_acc_sum += sym_acc_arr[idx, idx2]
                    cross_acc_count += 1

            # Calculate average cross-cluster accuracy
            if cross_acc_count > 0:
                avg_cross_acc = cross_acc_sum / cross_acc_count
            else:
                avg_cross_acc = None  # Handle the case where no cross-cluster pairs exist
            # Append the average cross-cluster accuracy to all relevant clusters
            if cluster_id not in cross_cluster_performance:
                cross_cluster_performance[cluster_id] = []  # Initialize list
            cross_cluster_performance[cluster_id].append((iterations, avg_cross_acc))

        # Find and merge clusters
        masked_diag_array = sym_acc_arr.copy()
        np.fill_diagonal(masked_diag_array, 0.0)
        similarity_score = np.max(masked_diag_array)
        max_index = np.unravel_index(np.argmax(masked_diag_array), masked_diag_array.shape)
        
        row_idx_to_merge = max_index[0]
        col_idx_to_merge = max_index[1]
        # Get actual cluster IDs to merge
        row_cluster_to_merge = current_clusters[row_idx_to_merge]
        col_cluster_to_merge = current_clusters[col_idx_to_merge]
        # Create a new cluster ID for the merged cluster
        new_cluster_id = max(current_clusters) + 1
        #print(f"MERGE: {row_cluster_to_merge, col_cluster_to_merge} @ {similarity_score*100:.2f}. New cluster: {new_cluster_id}")
        # Log the merge
        merge_log.append((iterations, row_cluster_to_merge, col_cluster_to_merge, similarity_score, new_cluster_id))
        # Update the DataFrame with the new merged cluster
        #userdef_df.loc[userdef_df['Cluster_ID'].isin([row_cluster_to_merge, col_cluster_to_merge]), 'Cluster_ID'] = new_cluster_id
        train_df.loc[train_df['Cluster_ID'].isin([row_cluster_to_merge, col_cluster_to_merge]), 'Cluster_ID'] = new_cluster_id
        intra_test_df.loc[intra_test_df['Cluster_ID'].isin([row_cluster_to_merge, col_cluster_to_merge]), 'Cluster_ID'] = new_cluster_id
        
        # Remove merged clusters from tracking (mark end with None)
        intra_cluster_performance[row_cluster_to_merge].append((iterations, None))
        intra_cluster_performance[col_cluster_to_merge].append((iterations, None))
        cross_cluster_performance[row_cluster_to_merge].append((iterations, None))
        cross_cluster_performance[col_cluster_to_merge].append((iterations, None))

        # Update cluster tracking
        previous_clusters = current_cluster_set
        iterations += 1

    return merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict, all_clus_logs_dict


def run_cluster_assignment(nested_clus_model_dict, ft_loader, config):
    """
    Assigns a participant to the best-performing cluster model.

    Handles both single-level and double-level nested_clus_model_dict.
    If cluster_iter_str is None or "All", tests all available cluster models.
    """

    clus_model_res_dict = {}  # { (iter_str, clus_id): accuracy }
    cluster_iter_str = config["cluster_iter_str"]
    
    # Detect if dict is double nested (iteration -> cluster_id -> model)
    double_nested = all(isinstance(v, dict) for v in nested_clus_model_dict.values())

    if double_nested:
        if cluster_iter_str is None or cluster_iter_str.upper() == "ALL":
            # All models from all iterations
            for iter_key, cluster_dict in nested_clus_model_dict.items():
                for clus_id, model in cluster_dict.items():
                    acc = evaluate_model(model, ft_loader)["accuracy"]
                    clus_model_res_dict[(iter_key, clus_id)] = acc
        else:
            # Only models from the specified iteration
            for clus_id, model in nested_clus_model_dict[cluster_iter_str].items():
                acc = evaluate_model(model, ft_loader)["accuracy"]
                clus_model_res_dict[(cluster_iter_str, clus_id)] = acc
    else:
        # Single-level: treat cluster_iter_str as irrelevant (ie we are running the ALL case)
        for clus_id, model in nested_clus_model_dict.items():
            acc = evaluate_model(model, ft_loader)["accuracy"]
            clus_model_res_dict[("NA", clus_id)] = acc

    # Find the key with the highest accuracy
    max_key = max(clus_model_res_dict, key=clus_model_res_dict.get)
    max_value = clus_model_res_dict[max_key]

    # Count ties (excluding the chosen one)
    tie_count = sum(1 for v in clus_model_res_dict.values() if v == max_value) - 1

    iter_str_assigned, clus_id_assigned = max_key

    #if config.get('verbose', False):
    print(f"Cluster {clus_id_assigned} (iter={iter_str_assigned}) had the highest accuracy ({max_value:.4f}), "
              f"Ties: {tie_count}")
    #print("Full cluster assignment results dict:")
    #print(clus_model_res_dict)

    # Store info if needed
    cluster_asgnmt_info_dict = {
        "iter_str": iter_str_assigned,
        "cluster_id": clus_id_assigned,
        "accuracy": max_value,
        "tie_count": tie_count
    }

    # Evaluate chosen cluster model on intra_test_loader
    if double_nested:
        model = nested_clus_model_dict[iter_str_assigned][clus_id_assigned]
    else:
        model = nested_clus_model_dict[clus_id_assigned]

    # Don't need to do this within the function
    #pretrained_clus_res = evaluate_model(model, intra_test_loader)
    #novel_pid_res_dict[pid]["pretrained_cluster_intra_test_acc"] = pretrained_clus_res["accuracy"]

    return model, cluster_asgnmt_info_dict