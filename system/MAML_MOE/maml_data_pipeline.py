import os
import torch
import random
import pickle
from torch.utils.data import Dataset, DataLoader

# TODO: Does this even get called... not yet...
def maml_mm_collate(batch):
    episode = batch[0]
    
    def stack_samples(sample_list):
        if not sample_list: return None
        
        # Pull EMG
        emg = torch.stack([s["emg"] for s in sample_list], dim=0).permute(0, 2, 1)
        
        # Pull IMU (Conditional based on existence/config)
        imu = None
        if "imu" in sample_list[0] and sample_list[0]["imu"] is not None:
            imu = torch.stack([s["imu"] for s in sample_list], dim=0).permute(0, 2, 1)

        demo = torch.stack([s["demo"] for s in sample_list], dim=0).float()
        labels = torch.as_tensor([s["label"] for s in sample_list], dtype=torch.long)

        return {
            "emg": emg, 
            "imu": imu, 
            "demo": demo, 
            "labels": labels
        }

    return {
        "support": stack_samples(episode["support"]),
        "query": stack_samples(episode["query"]),
        "user_id": episode.get("user_id")
    }

# ==========================================
# THE META-LEARNING DATASET
# ==========================================
class MetaGestureDataset(Dataset):
    """
    A unified Map-style Dataset for both Training (random episodes) 
    and Val/Test (deterministic user-specific episodes).
    """
    def __init__(self, tensor_dict, target_pids, target_gestures, n_way=10, k_shot=1, q_query=9, 
                 episodes_per_epoch=1000, is_train=True, seed=42):
        self.data = {pid: tensor_dict[pid] for pid in target_pids}
        self.pids = list(self.data.keys())
        self.target_gestures = target_gestures
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.is_train = is_train
        self.rng = random.Random(seed)

    def __len__(self):
        # Train: Arbitrary number of episodes. Val/Test: Exactly one episode per user.
        return self.episodes_per_epoch if self.is_train else len(self.pids)

    def __getitem__(self, idx):
        """
        Returns a single MAML episode (Support set and Query set).
        Each sample in the sets now contains 'emg' and 'imu' separately.
        """
        # -----------------------------------------------------------
        # 1. IDENTIFY USER AND CLASSES (N-WAY)
        # -----------------------------------------------------------
        if self.is_train:
            # Training: Randomly pick a user and then pick N random gestures they have performed
            user_id = self.rng.choice(self.pids)
            available_gestures = [g for g in self.target_gestures if g in self.data[user_id]]
            classes = self.rng.sample(available_gestures, self.n_way)
        else:
            # Evaluation: Deterministic. Each idx maps to a specific user to ensure 
            # consistent reporting across HPO trials.
            user_id = self.pids[idx]
            classes = sorted([g for g in self.target_gestures if g in self.data[user_id]])[:self.n_way]

        support_samples, query_samples = [], []
        label_map = {c: i for i, c in enumerate(classes)}

        # -----------------------------------------------------------
        # 2. LOOP THROUGH CLASSES TO BUILD SUPPORT/QUERY SAMPLES
        # -----------------------------------------------------------
        for global_c in classes:
            local_label = label_map[global_c]
            
            # Access the 3D Tensors: (Num_Trials, Time, Channels)
            user_emg_data = self.data[user_id][global_c]['emg'] 
            user_imu_data = self.data[user_id][global_c]['imu']
            user_demo = self.data[user_id][global_c]['demo']
            
            # Determine available trials (usually 10)
            total_trials = user_emg_data.shape[0]
            indices = list(range(total_trials))
            
            # Shuffle trials during training so the 'Support' trial isn't always the same one
            if self.is_train:
                self.rng.shuffle(indices)
            
            # -----------------------------------------------------------
            # 3. SPLIT DISJOINT INDICES (K-SHOT vs Q-QUERY)
            # -----------------------------------------------------------
            # Support indices: The first K trials
            sup_idx = indices[:self.k_shot]
            
            # Query indices: The next Q trials. 
            # This ensures support and query never overlap.
            # If is_train is false, we use all remaining trials for a more robust evaluation.
            if self.is_train:
                # Slice from K to K+Q (e.g., 1:1+5)
                qry_idx = indices[self.k_shot : self.k_shot + self.q_query]
            else:
                # Use everything else for evaluation
                qry_idx = indices[self.k_shot:]

            # -----------------------------------------------------------
            # 4. MATERIALIZE SAMPLE DICTIONARIES
            # -----------------------------------------------------------
            # Build Support List for this class
            for i in sup_idx:
                support_samples.append({
                    'emg': user_emg_data[i], # Shape: (64, 16)
                    'imu': user_imu_data[i], # Shape: (64, 72)
                    'demo': user_demo,
                    'label': torch.tensor(local_label, dtype=torch.long),
                    'global_class': global_c
                })
                
            # Build Query List for this class
            for i in qry_idx:
                query_samples.append({
                    'emg': user_emg_data[i],
                    'imu': user_imu_data[i],
                    'demo': user_demo,
                    'label': torch.tensor(local_label, dtype=torch.long),
                    'global_class': global_c
                })

        # -----------------------------------------------------------
        # 5. FINAL SHUFFLE AND RETURN
        # -----------------------------------------------------------
        # Shuffle the samples within the episode so the model doesn't see 
        # all 'Class 0' samples followed by all 'Class 1' samples.
        if self.is_train:
            self.rng.shuffle(support_samples)
            self.rng.shuffle(query_samples)

        return {
            'support': support_samples, 
            'query': query_samples,
            'user_id': user_id,
            'label_map': label_map
        }


# ==========================================
# 3. DATALOADER BUILDER
# ==========================================
def get_maml_dataloaders(config, tensor_dict_path, collate_fn):
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)
        
    num_workers = int(config.get('num_workers', 4))
    
    # Train Loader (Randomly sampling all users)
    train_ds = MetaGestureDataset(
        tensor_dict, 
        target_pids=config["train_PIDs"], 
        target_gestures=config["train_gesture_range"] + [10], # Replaces the merge
        n_way=config['n_way'], k_shot=config['k_shot'], q_query=config['q_query'],
        episodes_per_epoch=config['episodes_per_epoch_train'],
        is_train=True
    )
    
    # By default, PyTorch DataLoaders will fetch individual items from the Dataset and batch them.
    # Because our __getitem__ returns a FULL EPISODE, we set batch_size=1 so it yields 1 episode at a time.
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    # Val Loader (Deterministic, one episode per user)
    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids=config["val_PIDs"],
        target_gestures=[1] + config["valtest_gesture_range"], # 1 is support, rest are query
        n_way=config['n_way'], k_shot=1, q_query=None, # q_query=None grabs all remaining for eval
        is_train=False
    )
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dl, val_dl