import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from full_study_funcs import * 

import os  
cwd = os.getcwd()
print("Current Working Directory: ", cwd)
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


def make_dataloaders(h5_path, indices_dict, batch_size=128, num_workers=0, shuffle=True):
    # Apparently my laptop has 20 physical CPU cores, and 28 Logical Processors (virtual cores)

    # You can customize shuffle per split if needed
    loaders = {}
    for split_name, idx_list in indices_dict.items():
        ds = H5SegmentDataset(h5_path, idx_list)
        loaders[split_name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle if 'train' in split_name else False, num_workers=num_workers
        )
    return loaders


def generate_final_plot_dataloaders(config, dataloaders_dict, 
                        include_local_models=True, include_FT_random=True, title=None, save_fig=False):
    
    MODEL_STR = config['model_str']
    pretrain_loader = dataloaders_dict["pretrain"]
    pretrain_intra_test_loader = dataloaders_dict["pretrain_subject_test"]
    pretrain_cross_test_loader = dataloaders_dict["novel_subject_test"]
    ft_loader = dataloaders_dict["novel_trainFT"]
    ft_test_loader = dataloaders_dict["novel_subject_test"]

    # Train base global model from scratch (it should only take a few mins...)
    ## Honestly probably should save and load...
    ## TODO: Is this supposed to be single participant? Surely not?
    results = main_training_pipeline(data_splits=None, all_participants=config["pretrain_participants"]+config["novel_participants"], test_participants=config["novel_participants"], config=config, 
                            train_intra_cross_loaders=[pretrain_loader, pretrain_intra_test_loader, pretrain_cross_test_loader], single_participant=True)
    pretrained_generic_model = copy.deepcopy(results["model"])

    #data_dict_1_1 = full_comparison_run(one_trial_data_splits, one_trial_data_splits, config, copy.deepcopy(pretrained_generic_model),
    #                    None, cluster_iter_str='Iter18', run_local=include_local_models, run_FT_random=include_FT_random)
    data_dict_1_1 = full_comparison_run_dataloaders(ft_loader, ft_test_loader, ft_test_loader, config, pretrained_generic_model,
                        run_local=include_local_models, run_FT_random=include_FT_random)

    if title is None:
        fig_title = f"{MODEL_STR} (Moments) One-shot Novel User Accuracy"
    else:
        fig_title = title
    # Ordered full list of entries
    full_data_keys = [
        ('global_acc_data', True),
        ('local_acc_data', include_local_models),
        ('ft_random_acc_data', include_FT_random),
        ('ft_global_acc_data', True),
    ]
    full_labels = [
        ('Generic Global', True),
        ('Local', include_local_models),
        ('Fine-Tuned Random', include_FT_random),
        ('Fine-Tuned Global', True),
    ]
    # Filter based on toggles
    my_data_keys = [key for key, include in full_data_keys if include]
    my_labels = [label for label, include in full_labels if include]
    plot_model_acc_boxplots(data_dict_1_1, my_title=fig_title, save_fig=save_fig, plot_save_name=f"Final_{MODEL_STR}_Acc_1TA_1TT", 
                                data_keys=my_data_keys, labels=my_labels)

    return pretrained_generic_model


def create_split_indices(
    h5_path, 
    train_participants=None, 
    finetune_participants=None, 
    batch_size=100_000
):
    """
    Args:
        h5_path: path to your .h5 file
        train_participants: list of str
        finetune_participants: list of str
        batch_size: batch size for reading meta
    Returns:
        dict with index lists for all four splits,
        plus individual fields per finetune participant: "{PID}_novel_trainFT" and "{PID}_novel_subject_test"
    """

    indices = {
        'pretrain': [],
        'pretrain_subject_test': [],
        'novel_trainFT': [],
        'novel_subject_test': []
    }
    
    # Also create per-PID entries for all finetune participants
    finetune_participants = finetune_participants or []
    for pid in finetune_participants:
        indices[f"{pid}_novel_trainFT"] = []
        indices[f"{pid}_novel_subject_test"] = []

    train_set = set(train_participants or [])
    finetune_set = set(finetune_participants or [])
    
    with h5py.File(h5_path, 'r') as f:
        N = f['participant'].shape[0]
        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)
            pids = [p.decode() for p in f['participant'][start:stop]]
            gestures = f['gesture_num'][start:stop]
            for i, (pid, gnum) in enumerate(zip(pids, gestures)):
                idx = start + i
                # Group 1: Train participants
                if pid in train_set:
                    if 1 <= gnum <= 9:
                        indices['pretrain'].append(idx)
                    elif gnum == 10:
                        indices['pretrain_subject_test'].append(idx)
                # Group 2: Finetune participants (all combined)
                if pid in finetune_set:
                    if gnum == 1:
                        indices['novel_trainFT'].append(idx)
                        indices[f"{pid}_novel_trainFT"].append(idx)  # per-user field
                    elif 2 <= gnum <= 10:
                        indices['novel_subject_test'].append(idx)
                        indices[f"{pid}_novel_subject_test"].append(idx)  # per-user field
    return indices


class H5SegmentDataset(Dataset):
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = indices

        # There is probably a better way to do this, this is specific (and fine) for $B
        hardcoded_gesture_strings = ['close', 'delete', 'duplicate', 'move', 'open', 'pan', 'rotate', 'select-single', 'zoom-in', 'zoom-out']
        self.gesture_str_to_label_map = {name: i for i, name in enumerate(hardcoded_gesture_strings)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        with h5py.File(self.h5_path, 'r') as f:
            X = f['features'][real_idx]        # shape: [window_size, num_channels]
            gesture_str = f['gesture_id'][real_idx].decode()
            y = self.gesture_str_to_label_map[gesture_str]
            pid = f['participant'][real_idx].decode()
            gesture_num = f['gesture_num'][real_idx]
        # Convert to torch tensor
        X = torch.from_numpy(X).float()
        return X, y, pid, gesture_num


def full_comparison_run_dataloaders(ft_loader, intra_test_loader, cross_test_loader, config, pretrained_global_model,
                        run_local=True, run_FT_random=False):
    
    os.makedirs(config['results_save_dir'], exist_ok=True)

    novel_pids = config["novel_participants"] 
    #novel_pid_clus_asgn_data = cluster_assgnmt_data_splits['novel_trainFT_dict']
    
    novel_pid_res_dict = {}
    for pid_count, pid in enumerate(novel_pids):
        #print(f"PID {pid}, {pid_count+1}/{len(novel_pids)}")
        novel_pid_res_dict[pid] = {}

        # I have no idea what this is supposed to be...
        ## Presumably trying to define cross to not include the current pid... but that's not what it looks like...
        ## Maybe the part where cross_indices had that and that is just removed here
        # Why was I testing on Cross for Local...
        ## Is that the reason why Local and FT'd differ in performance so much?
        ## Granted, if Local was getting 40% acc on Cross that is pretty good...
        #subject_specific_cross_pids = list(set([novel_participant_test_data['participant_ids'][i] for i in cross_indices]))

        # 1) Train a local model for the current NOVEL subject
        if run_local:
            local_res = main_training_pipeline(data_splits=None, all_participants=novel_pids, test_participants=[pid], config=config, 
                            train_intra_cross_loaders=[ft_loader, intra_test_loader, cross_test_loader], single_participant=True)
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
        ft_global_res_dict = fine_tune_model(
            copy.deepcopy(pretrained_global_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        novel_pid_res_dict[pid]["ft_global_train_acc"] = ft_global_res_dict["train_accuracy"]
        novel_pid_res_dict[pid]["ft_global_intra_test_acc"] = ft_global_res_dict["intra_test_accuracy"]
        novel_pid_res_dict[pid]["ft_global_train_loss_log"] = ft_global_res_dict["train_loss_log"]
        novel_pid_res_dict[pid]["ft_global_intra_test_loss_log"] = ft_global_res_dict["intra_test_loss_log"]
        novel_pid_res_dict[pid]["ft_global_train_acc_log"] = ft_global_res_dict["train_acc_log"]
        novel_pid_res_dict[pid]["ft_global_intra_test_acc_log"] = ft_global_res_dict["intra_test_acc_log"]

        # 3.5) Test finetuned random model on current NOVEL subject!
        random_model = select_model(config['model_str'], config)
        if run_FT_random:
            ft_random_res_dict = fine_tune_model(
                copy.deepcopy(random_model), ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
            novel_pid_res_dict[pid]["ft_random_train_acc"] = ft_random_res_dict["train_accuracy"]
            novel_pid_res_dict[pid]["ft_random_intra_test_acc"] = ft_random_res_dict["intra_test_accuracy"]
            novel_pid_res_dict[pid]["ft_random_train_loss_log"] = ft_random_res_dict["train_loss_log"]
            novel_pid_res_dict[pid]["ft_random_intra_test_loss_log"] = ft_random_res_dict["intra_test_loss_log"]
            novel_pid_res_dict[pid]["ft_random_train_acc_log"] = ft_random_res_dict["train_acc_log"]
            novel_pid_res_dict[pid]["ft_random_intra_test_acc_log"] = ft_random_res_dict["intra_test_acc_log"]

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

    return data_dict