from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from moments_engr import *
#from configs.hyperparam_tuned_configs import *
from DNN_FT_funcs import *
from viz_utils.quick_analysis_plots import *
from full_study_funcs import * 
from tsaug import TimeWarp, Drift, AddNoise
import random

class TsaugTransform:
    def __init__(
        self,
        mirror_prob=0.3,
        timewarp_params={"n_speed_change": 3, "max_speed_ratio": 2},
        drift_params={"max_drift": (0.1, 0.3)},
        noise_params={"scale": 0.01},
        timewarp_prob=0.3,
        drift_prob=0.3,
        noise_prob=0.3, 
        num_channels=16
    ):
        self.timewarp_params = timewarp_params
        self.drift_params = drift_params
        self.noise_params = noise_params

        self.timewarp_prob = timewarp_prob
        self.drift_prob = drift_prob
        self.noise_prob = noise_prob
        self.mirror_prob = mirror_prob

        self.num_channels = num_channels

    def mirror_left_right(self, x_np):
        """
        Custom augmentation: permute left-right symmetric channels.
        Channel indices are 0-based, so 1→5 means index 0↔4, 2→6 means 1↔5, etc.
        """
        channel_map = {
            0: 4, 1: 5, 2: 6, 3: 7,
            8: 12, 9: 13, 10: 14, 11: 15
        }
        x_mirrored = x_np.copy()
        for i, j in channel_map.items():
            x_mirrored[:, [i, j]] = x_mirrored[:, [j, i]]
        return x_mirrored

    def __call__(self, x):
        was_tensor = isinstance(x, torch.Tensor)
        x_np = x.detach().cpu().numpy() if was_tensor else x

        transpose_back = False
        if self.num_channels is not None:
            if x_np.shape[1] == self.num_channels:
                pass
            elif x_np.shape[0] == self.num_channels:
                x_np = x_np.T
                transpose_back = True
            else:
                raise ValueError(f"Expected one dimension to match num_channels={self.num_channels}, but got shape {x_np.shape}")

        #print(f"[Before] shape: {x_np.shape}")  # DEBUG

        x_aug = x_np.copy()
        applied = []  # This is just a list containing the strings of all the transformations applied for the given sample
        if random.random() < self.timewarp_prob:
            #print("Applying timewarp!")
            x_aug = TimeWarp(**self.timewarp_params).augment(x_aug)
            applied.append("timewarp")
        if random.random() < self.drift_prob:
            #print("Applying drift!")
            x_aug = Drift(**self.drift_params).augment(x_aug)
            applied.append("drift")
        if random.random() < self.noise_prob:
            #print("Applying noise!")
            x_aug = AddNoise(**self.noise_params).augment(x_aug)
            applied.append("noise")
        if random.random() < self.mirror_prob:
            #print("Applying mirror!")
            x_aug = self.mirror_left_right(x_aug)
            applied.append("mirror")

        #print(f"[After ] shape: {x_aug.shape} | applied: {applied}")  # DEBUG
        #print()

        if transpose_back:
            x_aug = x_aug.T

        #return (torch.tensor(x_aug, dtype=torch.float32) if was_tensor else x_aug), applied
        # If you want to debug which transforms were applied, just print(applied) or log it: don’t return it from the function.
        return torch.tensor(x_aug, dtype=torch.float32) if was_tensor else x_aug


class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, label_list, transform=None):
        self.data = data_list
        self.labels = label_list
        self.transform = transform

    def __getitem__(self, idx):
        x = self.data[idx]  # Can be torch.Tensor or np.ndarray
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def extract_dfs_from_pkl(config, dataset_df_pickle_filename='\\noFE_windowed_segraw_allEMG.pkl'):

    none_expdef_df = load_expdef_gestures(feateng_method="None", noFE_filename=dataset_df_pickle_filename)  
    none_data_splits = make_data_split(none_expdef_df, config, split_index=None)

    # Prepare participant and label encoder
    test_participants = list(np.unique(none_data_splits['novel_subject_test_dict']['participant_ids']))
    all_participants = np.unique(
        none_data_splits['pretrain_dict']['participant_ids'] +
        none_data_splits['pretrain_subject_test_dict']['participant_ids'] +
        test_participants
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(all_participants)

    # Process the splits
    train_df = process_split(none_data_splits, 'pretrain_dict', label_encoder)
    intra_test_df = process_split(none_data_splits, 'pretrain_subject_test_dict', label_encoder)
    cross_test_df = process_split(none_data_splits, 'novel_subject_test_dict', label_encoder)
    # This doesn't get used anywhere here AFAIK
    #none_data_dfs_dict = {
    #    'pretrain_df': train_df,
    #    'pretrain_subject_test_df': intra_test_df
    #}

    return train_df, intra_test_df, cross_test_df, all_participants, test_participants



def create_datasets_and_dataloaders(train_df, intra_df, cross_df, tsaug_config, config, feature_column="feature", target_column="Gesture_Encoded", apply_transform=True, only_return_dataset=False):
    # --- Extract features and labels ---
    X_train = np.array([x for x in train_df[feature_column]])
    y_train = np.array(train_df[target_column])
    X_intra = np.array([x for x in intra_df[feature_column]])
    y_intra = np.array(intra_df[target_column])
    X_cross = np.array([x for x in cross_df[feature_column]])
    y_cross = np.array(cross_df[target_column])

    if len(X_train) == 0 or len(X_intra) == 0 or len(X_cross) == 0:
        raise ValueError("Training or validation set is empty")
    # Reshape to [B, C, T] if flat
    if X_train.ndim == 2:
        num_channels = tsaug_config['num_channels']
        sequence_length = X_train.shape[1] // num_channels
        X_train = X_train.reshape(-1, num_channels, sequence_length)
        X_intra = X_intra.reshape(-1, num_channels, sequence_length)
        X_cross = X_cross.reshape(-1, num_channels, sequence_length)

    # --- Create EMGDatasets and DataLoaders ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_intra_tensor = torch.tensor(X_intra, dtype=torch.float32)
    y_intra_tensor = torch.tensor(y_intra, dtype=torch.long)
    X_cross_tensor = torch.tensor(X_cross, dtype=torch.float32)
    y_cross_tensor = torch.tensor(y_cross, dtype=torch.long)

    if apply_transform:
        # --- Set up augmentation ---
        transform = TsaugTransform(
            mirror_prob=tsaug_config['mirror_prob'],
            timewarp_prob=tsaug_config['timewarp_prob'],
            drift_prob=tsaug_config['drift_prob'],
            noise_prob=tsaug_config['noise_prob'],
            timewarp_params=tsaug_config['timewarp_params'],
            drift_params=tsaug_config['drift_params'],
            noise_params=tsaug_config['noise_params'],
            num_channels=tsaug_config['num_channels']
        )

        train_dataset = EMGDataset(X_train_tensor, y_train_tensor, transform=transform)
    else:
        train_dataset = EMGDataset(X_train_tensor, y_train_tensor, transform=None)  # Applying transform is False!
    intra_dataset = EMGDataset(X_intra_tensor, y_intra_tensor, transform=None)  # No aug for val
    cross_dataset = EMGDataset(X_cross_tensor, y_cross_tensor, transform=None)  # No aug for val

    if only_return_dataset:
        return train_dataset, intra_dataset, cross_dataset

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    intra_loader = DataLoader(intra_dataset, batch_size=config["batch_size"], shuffle=False)
    cross_loader = DataLoader(cross_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_loader, intra_loader, cross_loader
