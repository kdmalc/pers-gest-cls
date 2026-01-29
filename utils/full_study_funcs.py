from torch.utils.data import DataLoader
import numpy as np

from utils.agglo_model_clust import *
from utils.DNN_FT_funcs import *
from utils.DNN_AMC_funcs import *
from utils.revamped_model_classes import *
from utils.viz_utils.quick_analysis_plots import plot_model_acc_boxplots


def generate_final_plot(config, correct_data_dfs_dict, correct_data_splits, correct_expdef_df, 
                        include_cluster_models=False, include_local_models=True, include_FT_random=True, 
                        cluster_models_filepath=None, title=None, save_fig=False):
    
    MODEL_STR = config['model_str']
    test_participants = list(np.unique(correct_data_splits['novel_subject_test_dict']['participant_ids']))
    all_participants = np.unique(correct_data_splits['pretrain_dict']['participant_ids'] + correct_data_splits['pretrain_subject_test_dict']['participant_ids'] + test_participants)

    # Train base global model from scratch (it should only take a few mins...)
    ## Honestly probably should save and load...
    results = main_training_pipeline(
        correct_data_splits, 
        all_participants=all_participants, 
        test_participants=test_participants,
        config=config)
    pretrained_generic_model = copy.deepcopy(results["model"])

    one_trial_data_splits = make_data_split(correct_expdef_df, config, num_train_tpg=1, num_ft_tpg=1)

    if include_cluster_models:
        if cluster_models_filepath is None:
            merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict, all_clus_logs_dict = DNN_agglo_merge_procedure(copy.deepcopy(correct_data_dfs_dict), config)

            print(f'{config["results_save_dir"]}')
            # Create log directory if it doesn't exist
            os.makedirs(config["results_save_dir"], exist_ok=True)
            with open(f'{config["results_save_dir"]}\\{MODEL_STR}_agglo_merge_res.pkl', 'wb') as f:
                pickle.dump(merge_log, f)
                pickle.dump(intra_cluster_performance, f)
                pickle.dump(cross_cluster_performance, f)
                pickle.dump(nested_clus_model_dict, f)
            print("Clustering results saved successfully!")
        else:
            with open(cluster_models_filepath, 'rb') as f:
                _ = pickle.load(f)
                _ = pickle.load(f)
                _ = pickle.load(f)
                nested_clus_model_dict = pickle.load(f)
            print("Clustering results loaded successfully.")

        data_dict_1_1 = full_comparison_run(one_trial_data_splits, one_trial_data_splits, config, copy.deepcopy(pretrained_generic_model),
                            copy.deepcopy(nested_clus_model_dict), run_local=include_local_models, run_clusters=include_cluster_models, run_FT_random=include_FT_random)
    else:
        data_dict_1_1 = full_comparison_run(one_trial_data_splits, one_trial_data_splits, config, copy.deepcopy(pretrained_generic_model),
                            None, run_local=include_local_models, run_clusters=include_cluster_models, run_FT_random=include_FT_random)

    if title is None:
        #fig_title = f"{MODEL_STR} (Moments) One-shot Novel User Accuracy"
        fig_title = title
    else:
        fig_title = title
    # Ordered full list of entries
    full_data_keys = [
        ('global_acc_data', True),
        ('pretrained_cluster_acc_data', include_cluster_models),
        ('local_acc_data', include_local_models),
        ('ft_random_acc_data', include_FT_random),
        ('ft_global_acc_data', True),
        ('ft_cluster_acc_data', include_cluster_models)
    ]
    full_labels = [
        ('Generic Global', True),
        ('Pretrained Cluster', include_cluster_models),
        ('Local', include_local_models),
        ('Fine-Tuned Random', include_FT_random),
        ('Fine-Tuned Global', True),
        ('Fine-Tuned Cluster', include_cluster_models)
    ]
    # Filter based on toggles
    my_data_keys = [key for key, include in full_data_keys if include]
    my_labels = [label for label, include in full_labels if include]
    plot_model_acc_boxplots(data_dict_1_1, my_title=fig_title, save_fig=save_fig, plot_save_name=f"Final_{MODEL_STR}_Acc_1TA_1TT", 
                                data_keys=my_data_keys, labels=my_labels)

    return pretrained_generic_model


def average_res_dicts(dicts_lst, client_final_losses_suffix_str="acc_data"):
    avg_dict = {}
    for key in dicts_lst[0].keys():
        if not key.endswith(client_final_losses_suffix_str):
            continue  # Skip keys that don't match

        # Collect the values for this key across all dicts
        values = [d[key] for d in dicts_lst]

        # Ensure all values are arrays and have the same shape
        try:
            arrays = [np.array(v, dtype=float) for v in values]
            shapes = [a.shape for a in arrays]
            if len(set(shapes)) > 1:
                raise ValueError(f"Inconsistent shapes for key '{key}': {shapes}")
            avg_dict[key] = np.mean(arrays, axis=0)
        except Exception as e:
            print(f"Skipping key '{key}' due to error: {e}")
            continue

    return avg_dict


def full_comparison_run(finetuning_datasplits, cluster_assgnmt_data_splits, config, pretrained_global_model,
                        nested_clus_model_dict, run_local=True, run_clusters=True, run_FT_random=False):
    
    if finetuning_datasplits != cluster_assgnmt_data_splits:
        print("Unique cluster_assgnmt_data_splits passed in. Not currently supported in the code!")
    
    os.makedirs(config['results_save_dir'], exist_ok=True)

    novel_participant_ft_data = finetuning_datasplits['novel_trainFT_dict']
    novel_participant_test_data = finetuning_datasplits['novel_subject_test_dict']
    novel_pids = np.unique(finetuning_datasplits['novel_trainFT_dict']['participant_ids'])
    #novel_pid_clus_asgn_data = cluster_assgnmt_data_splits['novel_trainFT_dict']
    
    novel_pid_res_dict = {}
    for pid_count, pid in enumerate(novel_pids):
        print(pid)
        #print(f"PID {pid}, {pid_count+1}/{len(novel_pids)}")
        novel_pid_res_dict[pid] = {}

        # No shuffling is necessary here since the training is done on the novel user level
        ## We are already restricted to just one user, so there's no pids to shuffle

        ############## Novel Participant Finetuning Dataset ##############
        novel_ft_indices = [i for i, datasplit_pid in enumerate(novel_participant_ft_data['participant_ids']) if datasplit_pid == pid]
        ft_features_np = np.array([novel_participant_ft_data['feature'][i] for i in novel_ft_indices])
        ft_features = torch.tensor(ft_features_np, dtype=torch.float32)
        ft_labels = torch.tensor([novel_participant_ft_data['labels'][i] for i in novel_ft_indices], dtype=torch.long)
        #assert ft_features.ndim == 3, f"Expected 3D tensor (batch, channels, sequence), got {ft_features.ndim}D with shape {ft_features.shape}"
        #assert ft_features.shape[1] == config["num_channels"]
        #assert ft_features.shape[2] == config["sequence_length"]
        ft_dataset = make_tensor_dataset(ft_features, ft_labels, config)
        ft_loader = DataLoader(ft_dataset, batch_size=config["batch_size"], shuffle=True)

        ############## Novel Participant Intra Testing Dataset ##############
        novel_test_indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid == pid]
        intra_test_features_np = np.array([novel_participant_test_data['feature'][i] for i in novel_test_indices])
        intra_test_features = torch.tensor(intra_test_features_np, dtype=torch.float32)
        intra_test_labels = torch.tensor([novel_participant_test_data['labels'][i] for i in novel_test_indices], dtype=torch.long)
        intra_test_dataset = make_tensor_dataset(intra_test_features, intra_test_labels, config)
        intra_test_loader = DataLoader(intra_test_dataset, batch_size=config["batch_size"], shuffle=False)

        ############## Novel Participant Cross Testing Dataset ##############
        # This code is testing on all the other novel participants... I don't think we care about that right now
        ## Idc but this will allow us to check cross perf. No real reason to remove...
        cross_indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid != pid]        
        cross_test_features_np = np.array([novel_participant_test_data['feature'][i] for i in cross_indices])
        cross_test_features = torch.tensor(cross_test_features_np, dtype=torch.float32)
        cross_test_labels = torch.tensor([novel_participant_test_data['labels'][i] for i in cross_indices], dtype=torch.long)
        cross_test_dataset = make_tensor_dataset(cross_test_features, cross_test_labels, config)
        cross_test_loader = DataLoader(cross_test_dataset, batch_size=config["batch_size"], shuffle=False)

        subject_specific_cross_pids = list(set([novel_participant_test_data['participant_ids'][i] for i in cross_indices]))

        # 1) Train a local model for the current NOVEL subject
        if run_local:
            print("Running Local")
            local_res = main_training_pipeline(data_splits=None, all_participants=novel_pids, test_participants=subject_specific_cross_pids, config=config, 
                            train_intra_cross_loaders=[ft_loader, intra_test_loader, cross_test_loader], single_participant=True, scenario="Local")
            novel_pid_res_dict[pid]["local_train_acc"] = local_res["train_accuracy"]
            novel_pid_res_dict[pid]["local_intra_test_acc"] = local_res["intra_test_accuracy"]
            novel_pid_res_dict[pid]["local_train_loss_log"] = local_res["train_loss_log"]
            novel_pid_res_dict[pid]["local_intra_test_loss_log"] = local_res["intra_test_loss_log"]
            novel_pid_res_dict[pid]["local_train_acc_log"] = local_res["train_acc_log"]
            novel_pid_res_dict[pid]["local_intra_test_acc_log"] = local_res["intra_test_acc_log"]

        # 2) Test the full pretrained global model on the current NOVEL subject
        global_global_res = evaluate_model(pretrained_global_model, intra_test_loader)
        novel_pid_res_dict[pid]["global_intra_test_acc"] = global_global_res["accuracy"]

        # 3) Test finetuned pretrained global model on the current NOVEL subject
        print("Fintuning Global")
        ft_global_res_dict = fine_tune_model(
            copy.deepcopy(pretrained_global_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        novel_pid_res_dict[pid]["ft_global_train_acc"] = ft_global_res_dict["train_accuracy"]
        novel_pid_res_dict[pid]["ft_global_intra_test_acc"] = ft_global_res_dict["intra_test_accuracy"]
        novel_pid_res_dict[pid]["ft_global_train_loss_log"] = ft_global_res_dict["train_loss_log"]
        novel_pid_res_dict[pid]["ft_global_intra_test_loss_log"] = ft_global_res_dict["intra_test_loss_log"]
        novel_pid_res_dict[pid]["ft_global_train_acc_log"] = ft_global_res_dict["train_acc_log"]
        novel_pid_res_dict[pid]["ft_global_intra_test_acc_log"] = ft_global_res_dict["intra_test_acc_log"]

        # 3.5) Test finetuned random model on current NOVEL subject!
        if run_FT_random:
            print(f"Finetuning Random")
            random_model = select_model(config['model_str'], config)
            initialize_weights_xavier(random_model)
            ft_random_res_dict = fine_tune_model(
                copy.deepcopy(random_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
            novel_pid_res_dict[pid]["ft_random_train_acc"] = ft_random_res_dict["train_accuracy"]
            novel_pid_res_dict[pid]["ft_random_intra_test_acc"] = ft_random_res_dict["intra_test_accuracy"]
            novel_pid_res_dict[pid]["ft_random_train_loss_log"] = ft_random_res_dict["train_loss_log"]
            novel_pid_res_dict[pid]["ft_random_intra_test_loss_log"] = ft_random_res_dict["intra_test_loss_log"]
            novel_pid_res_dict[pid]["ft_random_train_acc_log"] = ft_random_res_dict["train_acc_log"]
            novel_pid_res_dict[pid]["ft_random_intra_test_acc_log"] = ft_random_res_dict["intra_test_acc_log"]

        if run_clusters:
            print(f"Running Clusters")
            # 4) CLUSTER MODEL: Have the pretrained model from the best cluster do inference
            assigned_cluster_model, cluster_asgnmt_info_dict = run_cluster_assignment(nested_clus_model_dict, ft_loader, config)
            # Have the pretrained model from the best cluster do inference
            pretrained_clus_res = evaluate_model(assigned_cluster_model, intra_test_loader)
            novel_pid_res_dict[pid]["pretrained_cluster_intra_test_acc"] = pretrained_clus_res["accuracy"]

            # 5) FT the pretrained cluster model on the current NOVEL subject
            ft_clus_res_dict = fine_tune_model(
                copy.deepcopy(assigned_cluster_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
            novel_pid_res_dict[pid]["ft_cluster_train_acc"] = ft_clus_res_dict["train_accuracy"]
            novel_pid_res_dict[pid]["ft_cluster_intra_test_acc"] = ft_clus_res_dict["intra_test_accuracy"]
            novel_pid_res_dict[pid]["ft_cluster_train_loss_log"] = ft_clus_res_dict["train_loss_log"]
            novel_pid_res_dict[pid]["ft_cluster_intra_test_loss_log"] = ft_clus_res_dict["intra_test_loss_log"]
            novel_pid_res_dict[pid]["ft_cluster_train_acc_log"] = ft_clus_res_dict["train_acc_log"]
            novel_pid_res_dict[pid]["ft_cluster_intra_test_acc_log"] = ft_clus_res_dict["intra_test_acc_log"]

        print()

    data_dict = {
        'local_acc_data': [],
        'local_train_acc_data': [],
        'local_train_loss_log': [],
        'local_intra_test_loss_log': [],
        'local_train_acc_log': [],
        'local_intra_test_acc_log': [],
        #
        'global_acc_data': [],
        #
        'ft_global_acc_data': [],
        'ft_global_train_acc_data': [],
        'ft_global_train_loss_log': [],
        'ft_global_intra_test_loss_log': [],
        'ft_global_train_acc_log': [],
        'ft_global_intra_test_acc_log': [],
        #
        'pretrained_cluster_acc_data': [],
        #
        'ft_cluster_acc_data': [],
        'ft_cluster_train_acc_data': [],
        'ft_cluster_train_loss_log': [], 
        'ft_cluster_intra_test_loss_log': [], 
        'ft_cluster_train_acc_log': [], 
        'ft_cluster_intra_test_acc_log': [],
        #
        'ft_random_acc_data': [],  #ft_random_intra_test_acc
        'ft_random_train_acc_data': [], #ft_random_train_acc
        'ft_random_train_loss_log': [],
        'ft_random_intra_test_loss_log': [],
        'ft_random_train_acc_log': [],
        'ft_random_intra_test_acc_log': []
    }

    for pid, res in novel_pid_res_dict.items():
        if run_local:
            data_dict['local_acc_data'].append(res['local_intra_test_acc'])
            data_dict['local_train_acc_data'].append(res['local_train_acc'])
            data_dict['local_train_loss_log'].append(res["local_train_loss_log"])
            data_dict['local_intra_test_loss_log'].append(res["local_intra_test_loss_log"])
            data_dict['local_train_acc_log'].append(res["local_train_acc_log"])
            data_dict['local_intra_test_acc_log'].append(res["local_intra_test_acc_log"])

        data_dict['global_acc_data'].append(res['global_intra_test_acc'])

        data_dict['ft_global_acc_data'].append(res['ft_global_intra_test_acc'])
        data_dict['ft_global_train_acc_data'].append(res['ft_global_train_acc'])
        data_dict['ft_global_train_loss_log'].append(res["ft_global_train_loss_log"])
        data_dict['ft_global_intra_test_loss_log'].append(res["ft_global_intra_test_loss_log"])
        data_dict['ft_global_train_acc_log'].append(res["ft_global_train_acc_log"])
        data_dict['ft_global_intra_test_acc_log'].append(res["ft_global_intra_test_acc_log"])

        if run_FT_random:
            data_dict["ft_random_train_acc_data"].append(res["ft_random_train_acc"])
            data_dict["ft_random_acc_data"].append(res["ft_random_intra_test_acc"])
            data_dict["ft_random_train_loss_log"].append(res["ft_random_train_loss_log"])
            data_dict["ft_random_intra_test_loss_log"].append(res["ft_random_intra_test_loss_log"])
            data_dict["ft_random_train_acc_log"].append(res["ft_random_train_acc_log"])
            data_dict["ft_random_intra_test_acc_log"].append(res["ft_random_intra_test_acc_log"])

        if run_clusters:
            data_dict['pretrained_cluster_acc_data'].append(res['pretrained_cluster_intra_test_acc'])

            data_dict['ft_cluster_acc_data'].append(res['ft_cluster_intra_test_acc'])
            data_dict['ft_cluster_train_acc_data'].append(res['ft_cluster_train_acc'])
            data_dict['ft_cluster_train_loss_log'].append(res["ft_cluster_train_loss_log"])
            data_dict['ft_cluster_intra_test_loss_log'].append(res["ft_cluster_intra_test_loss_log"])
            data_dict['ft_cluster_train_acc_log'].append(res["ft_cluster_train_acc_log"])
            data_dict['ft_cluster_intra_test_acc_log'].append(res["ft_cluster_intra_test_acc_log"])

    return data_dict


# FOR LOCAL
def group_data_by_pid(features, labels, pids):
    """Group features and labels by unique participant IDs."""
    pids_npy = np.array(pids)
    unique_pids = np.unique(pids)
    pid_data = {}
    for pid in unique_pids:
        mask = (pids_npy == pid)
        pid_features = features[mask]
        pid_labels = labels[mask]
        pid_data[pid] = (pid_features, pid_labels)
    return pid_data


def prepare_data_for_local_models(data_splits, config, pretrain_or_novel="novel"):

    bs = config["batch_size"]
    sequence_length = config["sequence_length"]
    time_steps = config["time_steps"]

    training_data_key = 'pretrain_dict' if pretrain_or_novel=="pretrain" else 'novel_trainFT_dict'
    testing_data_key = 'pretrain_subject_test_dict' if pretrain_or_novel=="pretrain" else 'novel_subject_test_dict'
    print(f"training_data_key: {training_data_key}")
    print(f"testing_data_key: {testing_data_key}")

    # Group data from each split by participant ID
    train_groups = group_data_by_pid(
        data_splits[training_data_key]['feature'],
        data_splits[training_data_key]['labels'],
        data_splits[training_data_key]['participant_ids']
    )
    intra_groups = group_data_by_pid(
        data_splits[testing_data_key]['feature'],
        data_splits[testing_data_key]['labels'],
        data_splits[testing_data_key]['participant_ids']
    )

    # Could assert that training_data_key pids equals testing_data_key pids? ...

    # Get all unique participant IDs across all splits
    all_pids = set(data_splits['pretrain_dict']['participant_ids']).union(set(data_splits['novel_trainFT_dict']['participant_ids']))
    
    if pretrain_or_novel == "pretrain":
        # If we're running this on pretrain users, then we can use the novel users for cross subject testing
        cross_features = torch.tensor(data_splits['novel_subject_test_dict']['feature'], dtype=torch.float32)
        cross_labels = torch.tensor(data_splits['novel_subject_test_dict']['labels'], dtype=torch.long)
    else:
        # If we're already running this on novel users, then we can use the pretrained users for LOCAL cross subject testing
        cross_features = torch.tensor(data_splits['pretrain_subject_test_dict']['feature'], dtype=torch.float32)
        cross_labels = torch.tensor(data_splits['pretrain_subject_test_dict']['labels'], dtype=torch.long)
    assert cross_features.ndim == 3
    assert cross_features.shape[1] == config["num_channels"]
    assert cross_features.shape[2] == config["sequence_length"]
    cross_dataset = make_tensor_dataset(cross_features, cross_labels, config)
    cross_loader = DataLoader(cross_dataset, batch_size=bs, shuffle=False)

    user_dict = {}
    for pid in all_pids:
        train_loader, intra_loader = None, None
        
        # Create train loader if participant exists in training split
        if pid in train_groups:
            features, labels = train_groups[pid]
            train_dataset = make_tensor_dataset(features, labels, config) 
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        
            # Create intra-subject test loader
            features, labels = intra_groups[pid]
            intra_dataset = make_tensor_dataset(features, labels, config) 
            intra_loader = DataLoader(intra_dataset, batch_size=bs, shuffle=False)
        
        # USE THE SAME CROSS LOADER FOR ALL USERS. NO SUBJECT SPECIFC CROSS SUBJECT (BY DEFINITION)
        user_dict[pid] = (train_loader, intra_loader, cross_loader)
    return user_dict


def finetuning_comparison_run(finetuning_datasplits, config, pretrained_global_model, nested_clus_model_dict, 
                              use_cluster_models=True, dataloaders_zip_lst=None,
                              test_untrained_global=False, test_FT_untrained_global=False, untrained_global_model=None):
    
    #os.makedirs(config['results_save_dir'], exist_ok=True)

    if dataloaders_zip_lst is not None:
        novel_pids = [tpl[0] for tpl in dataloaders_zip_lst]
    else:
        novel_participant_ft_data = finetuning_datasplits['novel_trainFT_dict']
        # novel "cross subject" is the same as novel intra (but needs to be separated according to PID first...)
        novel_participant_test_data = finetuning_datasplits['novel_subject_test_dict']
        novel_pids = np.unique(finetuning_datasplits['novel_trainFT_dict']['participant_ids'])
        #novel_pid_clus_asgn_data = cluster_assgnmt_data_splits['novel_trainFT_dict']

    if (test_untrained_global==True or test_FT_untrained_global==True) and untrained_global_model is None:
        raise ValueError("untrained_global_model is None! Pass in an untrained model (call select_model())")

    novel_pid_res_dict = {}
    for pid_count, pid in enumerate(novel_pids):
        print(f"PID {pid}, {pid_count+1}/{len(novel_pids)}")
        novel_pid_res_dict[pid] = {}

        if dataloaders_zip_lst is not None:
            ft_loader = dataloaders_zip_lst[pid_count][1]
            intra_test_loader = dataloaders_zip_lst[pid_count][2]
        else:
            # Create the testloader by segmenting out this specific pid
            # Filter based on CURRENT participant ID: 
            ############## Novel Participant Finetuning Dataset ##############
            novel_ft_indices = [i for i, datasplit_pid in enumerate(novel_participant_ft_data['participant_ids']) if datasplit_pid == pid]
            ft_dataset = make_tensor_dataset([novel_participant_ft_data['feature'][i] for i in novel_ft_indices], [novel_participant_ft_data['labels'][i] for i in novel_ft_indices], config) 
            ft_loader = DataLoader(ft_dataset, batch_size=config["batch_size"], shuffle=True)
            ############## Novel Participant Intra Testing Dataset ##############
            novel_test_indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid == pid]
            intra_test_dataset = make_tensor_dataset([novel_participant_test_data['feature'][i] for i in novel_test_indices], [novel_participant_test_data['labels'][i] for i in novel_test_indices], config) 
            intra_test_loader = DataLoader(intra_test_dataset, batch_size=config["batch_size"], shuffle=True)
            ############## Cluster Assignment Dataset ##############
            ## This will just use ft_loader. No reason to have them separated really. 
            ## Removing this will remove functionality where num cluster assgnt != num ft trials
            #indices = [i for i, datasplit_pid in enumerate(novel_pid_clus_asgn_data['participant_ids']) if datasplit_pid == pid]
            #clust_asgn_dataset = GestureDataset([novel_pid_clus_asgn_data['feature'][i] for i in indices], [novel_pid_clus_asgn_data['labels'][i] for i in indices])
            #clust_asgn_loader = DataLoader(clust_asgn_dataset, batch_size=config["batch_size"], shuffle=True)
            ############## Novel Participant Cross Testing Dataset ##############
            # This code is testing on all the other novel participants... I don't think we care about that right now
            ## Idc but this will allow us to check cross perf. No real reason to remove...
            #indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid != pid]
            #cross_test_dataset = GestureDataset([novel_participant_test_data['feature'][i] for i in indices], [novel_participant_test_data['labels'][i] for i in indices])
            #cross_test_loader = DataLoader(cross_test_dataset, batch_size=config["batch_size"], shuffle=True)

        # 0) Untrained model
        ## a) Generic untrained
        if test_untrained_global:
            novel_pid_res_dict[pid]["untrained_global_intra_test_acc"] = evaluate_model(untrained_global_model, intra_test_loader)["accuracy"]
        ## b) Finetuned untrained
        if test_FT_untrained_global:
            ft_untrained_global_res_dict = fine_tune_model(
                copy.deepcopy(untrained_global_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
            novel_pid_res_dict[pid]["ft_untrained_global_train_acc"] = ft_untrained_global_res_dict["train_accuracy"]
            novel_pid_res_dict[pid]["ft_untrained_global_intra_test_acc"] = ft_untrained_global_res_dict["intra_test_accuracy"]
            novel_pid_res_dict[pid]["ft_untrained_global_train_loss_log"] = ft_untrained_global_res_dict["train_loss_log"]
            novel_pid_res_dict[pid]["ft_untrained_global_intra_test_loss_log"] = ft_untrained_global_res_dict["intra_test_loss_log"]
            novel_pid_res_dict[pid]["ft_untrained_global_train_acc_log"] = ft_untrained_global_res_dict["train_acc_log"]
            novel_pid_res_dict[pid]["ft_untrained_global_intra_test_acc_log"] = ft_untrained_global_res_dict["intra_test_acc_log"]

        # 1) Local (not run/tracked here)

        # 2) Test the full pretrained (global) model
        novel_pid_res_dict[pid]["global_intra_test_acc"] = evaluate_model(pretrained_global_model, intra_test_loader)["accuracy"]

        # 3) Test finetuned pretrained (global) model
        ft_global_res_dict = fine_tune_model(
            copy.deepcopy(pretrained_global_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        novel_pid_res_dict[pid]["ft_global_train_acc"] = ft_global_res_dict["train_accuracy"]
        novel_pid_res_dict[pid]["ft_global_intra_test_acc"] = ft_global_res_dict["intra_test_accuracy"]
        novel_pid_res_dict[pid]["ft_global_train_loss_log"] = ft_global_res_dict["train_loss_log"]
        novel_pid_res_dict[pid]["ft_global_intra_test_loss_log"] = ft_global_res_dict["intra_test_loss_log"]
        novel_pid_res_dict[pid]["ft_global_train_acc_log"] = ft_global_res_dict["train_acc_log"]
        novel_pid_res_dict[pid]["ft_global_intra_test_acc_log"] = ft_global_res_dict["intra_test_acc_log"]

        if use_cluster_models:
            # 4) CLUSTER MODEL: Have the pretrained model from the best cluster do inference
            assigned_cluster_model, cluster_asgnmt_info_dict = run_cluster_assignment(nested_clus_model_dict, ft_loader, config)
            # Have the pretrained model from the best cluster do inference
            pretrained_clus_res = evaluate_model(assigned_cluster_model, intra_test_loader)
            novel_pid_res_dict[pid]["pretrained_cluster_intra_test_acc"] = pretrained_clus_res["accuracy"]

            # 5) FT the pretrained cluster model on the participant
            ft_clus_res_dict = fine_tune_model(
                copy.deepcopy(assigned_cluster_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
            #ft_clus_res = evaluate_model(ft_clus_res_dict["finetuned_model"], intra_test_loader)
            novel_pid_res_dict[pid]["ft_cluster_train_acc"] = ft_clus_res_dict["train_accuracy"]
            novel_pid_res_dict[pid]["ft_cluster_intra_test_acc"] = ft_clus_res_dict["intra_test_accuracy"]
            novel_pid_res_dict[pid]["ft_cluster_train_loss_log"] = ft_clus_res_dict["train_loss_log"]
            novel_pid_res_dict[pid]["ft_cluster_intra_test_loss_log"] = ft_clus_res_dict["intra_test_loss_log"]
            novel_pid_res_dict[pid]["ft_cluster_train_acc_log"] = ft_clus_res_dict["train_acc_log"]
            novel_pid_res_dict[pid]["ft_cluster_intra_test_acc_log"] = ft_clus_res_dict["intra_test_acc_log"]

    data_dict = {
        'local_acc_data': [],
        'local_train_acc_data': [],
        'local_train_loss_log': [],
        'local_intra_test_loss_log': [],
        'local_train_acc_log': [],
        'local_intra_test_acc_log': [],
        'global_acc_data': [],
        'ft_global_acc_data': [],
        'ft_global_train_acc_data': [],
        'ft_global_train_loss_log': [],
        'ft_global_intra_test_loss_log': [],
        'ft_global_train_acc_log': [],
        'ft_global_intra_test_acc_log': [],
        'pretrained_cluster_acc_data': [],
        'ft_cluster_acc_data': [],
        'ft_cluster_train_acc_data': [],
        'ft_cluster_train_loss_log': [], 
        'ft_cluster_intra_test_loss_log': [], 
        'ft_cluster_train_acc_log': [], 
        'ft_cluster_intra_test_acc_log': [],

        "untrained_global_intra_test_acc": [], 
        "ft_untrained_global_train_acc": [], 
        "ft_untrained_global_intra_test_acc": [], 
        "ft_untrained_global_train_loss_log": [], 
        "ft_untrained_global_intra_test_loss_log": [], 
        "ft_untrained_global_train_acc_log": [], 
        "ft_untrained_global_intra_test_acc_log": []
    }

    for pid, res in novel_pid_res_dict.items():
        data_dict['global_acc_data'].append(res['global_intra_test_acc'])

        data_dict['ft_global_acc_data'].append(res['ft_global_intra_test_acc'])
        data_dict['ft_global_train_acc_data'].append(res['ft_global_train_acc'])
        data_dict['ft_global_train_loss_log'].append(res["ft_global_train_loss_log"])
        data_dict['ft_global_intra_test_loss_log'].append(res["ft_global_intra_test_loss_log"])
        data_dict['ft_global_train_acc_log'].append(res["ft_global_train_acc_log"])
        data_dict['ft_global_intra_test_acc_log'].append(res["ft_global_intra_test_acc_log"])

        if use_cluster_models:
            data_dict['pretrained_cluster_acc_data'].append(res['pretrained_cluster_intra_test_acc'])

            data_dict['ft_cluster_acc_data'].append(res['ft_cluster_intra_test_acc'])
            data_dict['ft_cluster_train_acc_data'].append(res['ft_cluster_train_acc'])
            data_dict['ft_cluster_train_loss_log'].append(res["ft_cluster_train_loss_log"])
            data_dict['ft_cluster_intra_test_loss_log'].append(res["ft_cluster_intra_test_loss_log"])
            data_dict['ft_cluster_train_acc_log'].append(res["ft_cluster_train_acc_log"])
            data_dict['ft_cluster_intra_test_acc_log'].append(res["ft_cluster_intra_test_acc_log"])

        if test_untrained_global:
            data_dict['untrained_global_intra_test_acc'].append(res['untrained_global_intra_test_acc'])
        if test_FT_untrained_global:
            data_dict['ft_untrained_global_train_acc'].append(res['ft_untrained_global_train_acc'])
            data_dict['ft_untrained_global_intra_test_acc'].append(res['ft_untrained_global_intra_test_acc'])
            data_dict['ft_untrained_global_train_loss_log'].append(res["ft_untrained_global_train_loss_log"])
            data_dict['ft_untrained_global_intra_test_loss_log'].append(res["ft_untrained_global_intra_test_loss_log"])
            data_dict['ft_untrained_global_train_acc_log'].append(res["ft_untrained_global_train_acc_log"])
            data_dict['ft_untrained_global_intra_test_acc_log'].append(res["ft_untrained_global_intra_test_acc_log"])

    return data_dict


def create_shared_trial_data_splits(expdef_df, all_participants, test_participants, num_monte_carlo_runs=1, num_train_gesture_trials=8, num_ft_gesture_trials=1):
    trial_data_splits_lst = [0]*num_monte_carlo_runs
    for i in range(num_monte_carlo_runs):
        # Prepare data
        trial_data_splits = prepare_data(
            expdef_df, 'feature', 'Gesture_Encoded', 
            all_participants, test_participants, 
            training_trials_per_gesture=num_train_gesture_trials, finetuning_trials_per_gesture=num_ft_gesture_trials,
        )
        trial_data_splits_lst[i] = trial_data_splits
    return trial_data_splits_lst


def finetuning_run(trial_data_splits_lst, config, pretrained_global_model, nested_clus_model_dict):
    """This function de facto does MonteCarlo averaging, and the number of elements (separate datasplits) in trial_data_splits_lst determines how many repetitions there are. """
    num_monte_carlo_runs = len(trial_data_splits_lst)
    print(f"finetuning_run(): num_monte_carlo_runs={num_monte_carlo_runs}")  # I think this should always be 1...
    num_ft_users = config["num_testft_users"]
    
    lst_of_res_dicts = [0] * num_monte_carlo_runs
    train_test_logs_list = [0] * num_monte_carlo_runs
    log_keys = ['local_train_loss_log', 'local_intra_test_loss_log', 'local_train_acc_log', 'local_intra_test_acc_log',
        'ft_global_train_loss_log', 'ft_global_intra_test_loss_log', 'ft_global_train_acc_log', 'ft_global_intra_test_acc_log',
        'ft_cluster_train_loss_log', 'ft_cluster_intra_test_loss_log', 'ft_cluster_train_acc_log', 'ft_cluster_intra_test_acc_log', 
        "ft_untrained_global_train_loss_log", "ft_untrained_global_intra_test_loss_log", "ft_untrained_global_train_acc_log", "ft_untrained_global_intra_test_acc_log"]
    for i in range(num_monte_carlo_runs):
        res_dict = finetuning_comparison_run(trial_data_splits_lst[i], config, pretrained_global_model,
                                              nested_clus_model_dict)
        lst_of_res_dicts[i] = res_dict
        # Logs are saved in res_dict directly, but not currently passed into data_dict except for the below:
        train_test_logs_list[i] = {key: res_dict[key] for key in log_keys}  # Save logs without averaging
    
    data_dict = {}    
    data_dict['global_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)
    data_dict['ft_global_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)
    data_dict['pretrained_cluster_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)
    data_dict['ft_cluster_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)

    for idx, res_dict in enumerate(lst_of_res_dicts): # This should match num_monte_carlo_runs
        data_dict['global_acc_data'] += np.array(res_dict['global_acc_data'])
        data_dict['ft_global_acc_data'] += np.array(res_dict['ft_global_acc_data'])
        data_dict['pretrained_cluster_acc_data'] += np.array(res_dict['pretrained_cluster_acc_data'])
        data_dict['ft_cluster_acc_data'] += np.array(res_dict['ft_cluster_acc_data'])

    data_dict['global_acc_data'] /= num_monte_carlo_runs
    data_dict['ft_global_acc_data'] /= num_monte_carlo_runs
    data_dict['pretrained_cluster_acc_data'] /= num_monte_carlo_runs
    data_dict['ft_cluster_acc_data'] /= num_monte_carlo_runs

    # Return both averaged results and raw logs
    data_dict['train_test_logs_list'] = train_test_logs_list
    return data_dict
