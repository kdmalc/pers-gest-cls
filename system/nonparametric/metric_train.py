import copy
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from pathlib import Path

import sys
import os
# Adds the 'system' directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MAML_MOE.maml_data_pipeline import get_maml_dataloaders
from metric_models import NeuralSubspaceClassifier, CovarianceEmbeddingNet, CrossAttentionRelationNet

def prepare_episode_tensors(batch, device):
    """
    Extracts support and query sets from the dataloader batch and 
    concatenates EMG and IMU into a single (Batch, 88, 64) tensor.
    """
    sup_dict = batch['support']
    qry_dict = batch['query']
    
    # Concatenate EMG (16 channels) and IMU (72 channels) -> 88 channels
    if sup_dict['imu'] is not None:
        sup_x = torch.cat([sup_dict['emg'], sup_dict['imu']], dim=1).to(device)
        qry_x = torch.cat([qry_dict['emg'], qry_dict['imu']], dim=1).to(device)
    else:
        sup_x = sup_dict['emg'].to(device)
        qry_x = qry_dict['emg'].to(device)
        
    sup_y = sup_dict['labels'].to(device)
    qry_y = qry_dict['labels'].to(device)
    
    return sup_x, sup_y, qry_x, qry_y

def compute_episodic_logits(model, model_type, sup_x, sup_y, qry_x, n_way):
    """
    Computes logits for the query set based on the support set.
    Handles K-shot by averaging the distance/similarity to the K support samples per class.
    """
    N_q = qry_x.shape[0]
    logits = torch.zeros(N_q, n_way, device=qry_x.device)
    
    if model_type == "cov_embedding":
        # Approach 2: Prototypical Network on Covariance Space
        cov_s = model.extract_cov_upper_tri(sup_x)
        z_s = model.encoder(cov_s)
        
        # Calculate class prototypes (average support embeddings per class)
        prototypes = []
        for c in range(n_way):
            prototypes.append(z_s[sup_y == c].mean(dim=0))
        prototypes = torch.stack(prototypes) # (n_way, embed_dim)
        
        cov_q = model.extract_cov_upper_tri(qry_x)
        z_q = model.encoder(cov_q)
        
        # Distance between queries and prototypes
        dists = torch.cdist(z_q, prototypes, p=2) # L2 distance: (N_q, n_way)
        logits = -dists # Negative distance so argmax finds the closest class
        
    elif model_type in ["neural_subspace", "cross_attention"]:
        # Approaches 1 & 3: Pairwise comparison
        # We compare each query against each support sample, then average the scores per class
        for c in range(n_way):
            sup_c = sup_x[sup_y == c] # (K_shot, C, T)
            K = sup_c.shape[0]
            class_scores = torch.zeros(N_q, K, device=qry_x.device)
            
            for k in range(K):
                # Expand the single support sample to match the query batch size
                s_k = sup_c[k:k+1].expand(N_q, -1, -1) 
                
                if model_type == "neural_subspace":
                    dist_k = model(s_k, qry_x) # L1 distance
                    class_scores[:, k] = -dist_k # Negative distance
                else: # cross_attention
                    sim_k = model(s_k, qry_x) # Similarity score
                    class_scores[:, k] = sim_k
                    
            # Average the scores over the K support samples for this class
            logits[:, c] = class_scores.mean(dim=1)
            
    return logits

def train_metric_model(model, model_type, config, train_dl, val_dl):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    best_val_acc = 0.0
    best_state = None
    
    print(f"--- Starting Training: {model_type} ---")
    for epoch in range(config["num_epochs"]):
        model.train()
        train_losses, train_accs = [], []
        
        for batch in train_dl:
            sup_x, sup_y, qry_x, qry_y = prepare_episode_tensors(batch, config["device"])
            
            optimizer.zero_grad()
            logits = compute_episodic_logits(model, model_type, sup_x, sup_y, qry_x, config["n_way"])
            
            loss = F.cross_entropy(logits, qry_y)
            loss.backward()
            optimizer.step()
            
            preds = logits.argmax(dim=1)
            acc = (preds == qry_y).float().mean().item()
            
            train_losses.append(loss.item())
            train_accs.append(acc)
            
        scheduler.step()
        
        # --- Validation Phase ---
        model.eval()
        val_accs = []
        user_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_dl:
                sup_x, sup_y, qry_x, qry_y = prepare_episode_tensors(batch, config["device"])
                logits = compute_episodic_logits(model, model_type, sup_x, sup_y, qry_x, config["n_way"])
                
                preds = logits.argmax(dim=1)
                acc = (preds == qry_y).float().mean().item()
                val_accs.append(acc)
                user_metrics[batch['user_id']].append(acc)
                
        epoch_val_acc = np.mean(val_accs)
        print(f"Epoch {epoch+1:02d} | Train Loss: {np.mean(train_losses):.4f} | Train Acc: {np.mean(train_accs)*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}%")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = copy.deepcopy(model.state_dict())
            
    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # Display per-user evaluation using the best model
    model.load_state_dict(best_state)
    print("\n--- Final User-Specific Evaluation (Best Model) ---")
    for user_id, accs in user_metrics.items():
        print(f"User {user_id} | Acc: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        
    return best_val_acc, best_state

if __name__ == "__main__":
    CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
    
    # 1. Configuration (Adapted from your mamlpp_deep_hpo.py)
    config = {
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "n_way": 10,  # 10-way user-defined gestures
        "k_shot": 1,  # 1-shot (or 5-shot)
        "q_query": 5,
        "episodes_per_epoch_train": 250,
        "num_epochs": 40,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 4,
        "emg_in_ch": 16,
        "imu_in_ch": 72,
        "maml_gesture_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "use_label_shuf_meta_aug": True,
        "dfs_load_path": f"{CODE_DIR}/dataset/meta-learning-sup-que-ds//"
    }
    
    # Load user splits
    user_split_json_filepath = CODE_DIR / "system" / "fixed_user_splits" / "4kfcv_splits_shared_test.json"
    with open(user_split_json_filepath, "r") as f:
        splits = json.load(f)
        
    # Use Fold 0 for demonstration
    config["train_PIDs"] = splits[0]["train"]
    config["val_PIDs"]   = splits[0]["val"]

    # 2. Initialize DataLoaders
    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path)

    in_channels = config["emg_in_ch"] + config["imu_in_ch"] # 88 channels

    # 3. Select which model to train
    # Options: "neural_subspace", "cov_embedding", "cross_attention"
    MODEL_TYPE = "cov_embedding" 
    
    if MODEL_TYPE == "neural_subspace":
        model = NeuralSubspaceClassifier(in_channels=in_channels).to(config["device"])
    elif MODEL_TYPE == "cov_embedding":
        model = CovarianceEmbeddingNet(in_channels=in_channels).to(config["device"])
    elif MODEL_TYPE == "cross_attention":
        model = CrossAttentionRelationNet(in_channels=in_channels).to(config["device"])

    # 4. Train and Evaluate
    best_acc, best_weights = train_metric_model(model, MODEL_TYPE, config, train_dl, val_dl)