import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.gesture_dataset_classes import *


def _ensure_CT(x, cfg):
    x = np.asarray(x)
    C = int(cfg["num_channels"])
    T = int(cfg["sequence_length"])
    CT = C * T
    if x.shape == (C, T):
        return x
    if x.shape == (T, C):
        return x.T
    if x.ndim == 1 and x.size == CT:
        return x.reshape(C, T)
    raise ValueError(
        f"Bad feature shape {x.shape}; expected (C,T)=({C},{T}) "
        f"or ({T},{C}) or flat length {CT}. "
        f"Likely a mistaken reshape using num_classes={cfg.get('num_classes')}."
    )


def test_models_on_clusters(
    test_df,
    trained_clus_models_ds,
    cluster_ids,
    config,
    cluster_column='Cluster_ID',
    feature_column='feature',
    target_column='Gesture_Encoded',
    pytorch_bool=False,
    criterion=nn.CrossEntropyLoss()
):
    """
    Test trained models for each cluster on a pre-split test set.

    Returns:
    - acc_matrix (ndarray): Accuracy matrix where (i, j) is accuracy of the i-th
      cluster's model on the j-th cluster's data.
    """
    num_clusters = len(cluster_ids)
    acc_matrix = np.zeros((num_clusters, num_clusters), dtype=float)

    # Choose device once
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # If using PyTorch models, move them to device and set eval()
    if pytorch_bool:
        if isinstance(trained_clus_models_ds, dict):
            for k, model in trained_clus_models_ds.items():
                trained_clus_models_ds[k] = model.to(device).eval()
        else:
            for i, model in enumerate(trained_clus_models_ds):
                trained_clus_models_ds[i] = model.to(device).eval()

    for i, clus_id in enumerate(cluster_ids):
        if config.get("verbose", False):
            print(f"Testing model for Cluster ID {clus_id} ({i+1}/{num_clusters})")

        # Get the model for this row (either by cluster id key or by index)
        model = trained_clus_models_ds[clus_id] if isinstance(trained_clus_models_ds, dict) else trained_clus_models_ds[i]

        for j, clus_id_test in enumerate(cluster_ids):
            clus_testset = test_df[test_df[cluster_column] == clus_id_test]

            if clus_testset.empty:
                if config.get("verbose", False):
                    print(f"No test data for Cluster ID {clus_id_test}. Skipping.")
                acc_matrix[i, j] = np.nan
                continue

            # Labels (numpy for both branches; converted to torch only if pytorch_bool)
            y_np = np.asarray(clus_testset[target_column])

            if pytorch_bool:
                # Build tensors on CPU first; move to device in the loop
                # X expected shape: [N, C, T] for 1D convs
                X_np = np.stack([_ensure_CT(x, config) for x in clus_testset[feature_column].to_list()])
                y_np = np.asarray(clus_testset[target_column])

                X = torch.as_tensor(X_np, dtype=torch.float32)
                y = torch.as_tensor(y_np, dtype=torch.long)

                assert X.shape[1] == config["num_channels"] and X.shape[2] == config["sequence_length"], \
                    f"Got X.shape={tuple(X.shape)}; expected (N,{config['num_channels']},{config['sequence_length']})"

                # DataLoader (no shuffle for eval)
                loader = DataLoader(
                    TensorDataset(X, y),
                    batch_size=int(config.get("batch_size", 64)),
                    shuffle=False,
                    num_workers=int(config.get("num_workers", 0)),
                    #pin_memory=bool(config.get("pin_memory", True)) and (device.startswith("cuda"))
                )

                total_correct = 0
                total_samples = 0

                # Ensure eval and on device (safe even if already done above)
                model.eval()
                model.to(device)

                with torch.no_grad():
                    for Xb, yb in loader:
                        Xb = Xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        # Optional sanity check:
                        # assert next(model.parameters()).device == Xb.device
                        logits = model(Xb)
                        preds = logits.argmax(dim=1)
                        total_correct += (preds == yb).sum().item()
                        total_samples += yb.size(0)

                acc_matrix[i, j] = (total_correct / total_samples) if total_samples > 0 else np.nan

            else:
                # Non-PyTorch branch (e.g., sklearn): flatten features per sample
                X_flat = np.stack([np.asarray(x).reshape(-1) for x in clus_testset[feature_column].to_list()])
                acc_matrix[i, j] = accuracy_score(y_np, model.predict(X_flat))

            if config.get("verbose", False):
                print(f"Model for Cluster {clus_id} on Cluster {clus_id_test}: "
                      f"Accuracy={acc_matrix[i, j]:.4f}")

    return acc_matrix