import os
import torch
import random
import pickle
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. COLLATE FUNCTION (Gradient Accumulation)
# ==========================================
def maml_mm_collate(batch):
    # Since batch_size=1, batch is a list of exactly 1 episode.
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
        "user_id": episode.get("user_id"),
        "label_map": episode.get("label_map")
    }

# ==========================================
# 2. THE META-LEARNING DATASET
# ==========================================
class MetaGestureDataset(Dataset):
    """
    A unified Map-style Dataset for both Training (random episodes on the fly) 
    and Val/Test (deterministic pre-generated episodes).
    """
    def __init__(self, tensor_dict, target_pids, target_gestures, n_way=10, k_shot=1, q_query=9, 
                 episodes_per_epoch=1000, is_train=True, seed=42, eval_episodes=10):
        self.data = {pid: tensor_dict[pid] for pid in target_pids}
        self.pids = list(self.data.keys())
        self.target_gestures = target_gestures
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.is_train = is_train
        self.eval_episodes = eval_episodes 
        
        # Cache for deterministic validation
        self.val_episodes_cache = []
        if not self.is_train:
            self._precompute_val_episodes(seed)

    def _precompute_val_episodes(self, seed):
        """Pre-generates validation episodes so they remain exactly the same every epoch."""
        val_rng = random.Random(seed) # Isolated deterministic RNG
        
        for user_id in self.pids:
            available_gestures = [g for g in self.target_gestures if g in self.data[user_id]]
            
            for _ in range(self.eval_episodes):
                # 1. Randomly sample and order classes to ensure randomized label permutation
                classes = val_rng.sample(available_gestures, min(self.n_way, len(available_gestures)))
                label_map = {c: i for i, c in enumerate(classes)}
                
                episode = self._build_episode(user_id, classes, label_map, val_rng, is_train=False)
                self.val_episodes_cache.append(episode)

    def _build_episode(self, user_id, classes, label_map, rng_instance, is_train):
        """Helper to construct the support and query sets for a given set of classes."""
        support_samples, query_samples = [], []
        
        for global_c in classes:
            local_label = label_map[global_c]
            
            user_emg_data = self.data[user_id][global_c]['emg'] 
            user_imu_data = self.data[user_id][global_c]['imu']
            user_demo = self.data[user_id][global_c]['demo']

            total_trials = user_emg_data.shape[0]
            indices = list(range(total_trials))
            
            # Shuffle indices to randomize which trials become support vs query
            rng_instance.shuffle(indices)
            
            # Split Disjoint Indices
            sup_idx = indices[:self.k_shot]
            if is_train and self.q_query is not None:
                qry_idx = indices[self.k_shot : self.k_shot + self.q_query]
            else:
                # If evaluating or q_query is None, use all remaining trials for query
                qry_idx = indices[self.k_shot:]

            # Materialize Dictionaries
            for i in sup_idx:
                support_samples.append({
                    'emg': user_emg_data[i],
                    'imu': user_imu_data[i] if user_imu_data is not None else None,
                    'demo': user_demo,
                    'label': local_label,
                    'global_class': global_c
                })
                
            for i in qry_idx:
                query_samples.append({
                    'emg': user_emg_data[i],
                    'imu': user_imu_data[i] if user_imu_data is not None else None,
                    'demo': user_demo,
                    'label': local_label,
                    'global_class': global_c
                })

        # Shuffle the final lists so the model doesn't see sequential labels
        rng_instance.shuffle(support_samples)
        rng_instance.shuffle(query_samples)

        return {
            'support': support_samples, 
            'query': query_samples,
            'user_id': user_id,
            'label_map': label_map
        }

    def __len__(self):
        if self.is_train:
            return self.episodes_per_epoch 
        else:
            return len(self.val_episodes_cache)

    def __getitem__(self, idx):
        if self.is_train:
            # Generate on the fly using standard random (safe for DataLoader workers)
            user_id = random.choice(self.pids)
            available_gestures = [g for g in self.target_gestures if g in self.data[user_id]]
            
            classes = random.sample(available_gestures, min(self.n_way, len(available_gestures)))
            label_map = {c: i for i, c in enumerate(classes)}
            
            return self._build_episode(user_id, classes, label_map, random, is_train=True)
        else:
            # Return pre-computed deterministic episode
            return self.val_episodes_cache[idx]


# ==========================================
# 3. DATALOADER BUILDER
# ==========================================
def get_maml_dataloaders(config, tensor_dict_path):
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)
        
    num_workers = int(config.get('num_workers', 4))
    
    # Train Loader (Randomly sampling all users on the fly)
    train_ds = MetaGestureDataset(
        tensor_dict, 
        target_pids=config["train_PIDs"], 
        target_gestures=config["train_gesture_range"] + [10],
        n_way=config['n_way'], 
        k_shot=config['k_shot'], 
        q_query=config['q_query'],
        episodes_per_epoch=config['episodes_per_epoch_train'],
        is_train=True
    )
    
    # batch_size=1 to yield 1 episode dictionary at a time for Gradient Accumulation
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=maml_mm_collate)

    # Val Loader (Deterministic, predefined episodes per user)
    val_ds = MetaGestureDataset(
        tensor_dict,
        target_pids=config["val_PIDs"],
        target_gestures=[1] + config["valtest_gesture_range"], 
        n_way=config['n_way'], 
        k_shot=config["k_shot"], 
        q_query=config.get("q_query", None), # Use None to grab all remaining for eval if desired
        eval_episodes=config.get('eval_episodes', 10), # Toggleable validation episodes
        is_train=False,
        seed=config.get('seed', 42)
    )
    
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=maml_mm_collate)

    return train_dl, val_dl