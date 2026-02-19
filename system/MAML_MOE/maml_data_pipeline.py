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
        # Map-style datasets cleanly handle num_workers without duplicating episodes.
        
        if self.is_train:
            # Pick a random user
            user_id = self.rng.choice(self.pids)
            # Pick random gestures available for this user
            available_gestures = [g for g in self.target_gestures if g in self.data[user_id]]
            classes = self.rng.sample(available_gestures, self.n_way)
        else:
            # Deterministic evaluation: idx directly corresponds to a specific user
            user_id = self.pids[idx]
            # Always evaluate on the exact same fixed support/query gestures
            classes = sorted([g for g in self.target_gestures if g in self.data[user_id]])[:self.n_way]

        support_samples, query_samples = [], []
        label_map = {c: i for i, c in enumerate(classes)}

        for global_c in classes:
            local_label = label_map[global_c]
            user_class_data = self.data[user_id][global_c]['timeseries'] # Tensor of shape (num_trials, feature_dim)
            user_demo = self.data[user_id][global_c]['demo']
            
            total_trials = user_class_data.shape[0]
            indices = list(range(total_trials))
            
            if self.is_train:
                self.rng.shuffle(indices)
            # If eval, indices remain sorted for deterministic 1-shot selection
            
            sup_idx = indices[:self.k_shot]
            qry_idx = indices[self.k_shot : self.k_shot + self.q_query] if self.is_train else indices[self.k_shot:]

            # Build support list
            for i in sup_idx:
                support_samples.append({
                    'timeseries': user_class_data[i],
                    'demo': user_demo,
                    'label': torch.tensor(local_label, dtype=torch.long),
                    'global_class': global_c
                })
                
            # Build query list
            for i in qry_idx:
                query_samples.append({
                    'timeseries': user_class_data[i],
                    'demo': user_demo,
                    'label': torch.tensor(local_label, dtype=torch.long),
                    'global_class': global_c
                })

        # Shuffle tasks during training to prevent the model from memorizing class orders
        if self.is_train:
            self.rng.shuffle(support_samples)
            self.rng.shuffle(query_samples)

        # TODO: We are not returning global_class here FYI...
        return {
            'support': support_samples, # You can apply your custom collate_fn to this in the DataLoader
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