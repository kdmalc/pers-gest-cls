import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl


def extract_latent_features(
    model,
    user_df,
    n_users=8,
    n_reps_per_gesture=1,   # Only uses gestures where gesture_num <= n_reps_per_gesture
):
    """
    Extracts latent features from the model for a subset of users in user_df.
    Only the first n_reps_per_gesture gesture_num's (starting from 1) for each (Numeric_PIDs, Gesture_Encoded) pair are used.
    Returns:
        features: np.ndarray [n_samples, embedding_dim]
        meta_df:  pd.DataFrame with columns ['Gesture_Encoded', 'Numeric_PIDs', 'source', 'gesture_num']
    """
    # Select users
    unique_pids = user_df['Numeric_PIDs'].unique()
    selected_pids = unique_pids[-n_users:]   # last n_users
    sub_df = user_df[user_df['Numeric_PIDs'].isin(selected_pids)].reset_index(drop=True)

    # Filter: gesture_num should be <= n_reps_per_gesture
    filtered_df = sub_df[sub_df['gesture_num'] <= n_reps_per_gesture].reset_index(drop=True)

    # Ensure model is in eval mode and on correct device
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare lists for features and metadata
    features_list = []
    gesture_list = []
    subject_list = []
    source_list = []
    gesture_num_list = []

    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        feat = row["feature"]
        # Convert feature to tensor
        if isinstance(feat, np.ndarray):
            x = torch.from_numpy(feat).float()
        elif isinstance(feat, list):
            x = torch.from_numpy(np.array(feat)).float()
        elif torch.is_tensor(feat):
            x = feat.float()
        else:
            raise ValueError("feature must be numpy array, list, or torch tensor")
        # Reshape if necessary (your logic here, adjust as needed)
        if x.shape[0] == 80:
            x = x.view(16, 5)
        x = x.unsqueeze(0).to(device)  # Shape: [1, channels, seq_len]
        # Extract latent features
        with torch.no_grad():
            z = model.forward_features(x)
        features_list.append(z.cpu().numpy().squeeze())
        gesture_list.append(row["Gesture_Encoded"])
        subject_list.append(row["Numeric_PIDs"])
        source_list.append(row["source"])
        gesture_num_list.append(row["gesture_num"])

    # Build features array and metadata DataFrame
    features = np.stack(features_list)  # shape: [n_samples, embedding_dim]
    meta_df = pd.DataFrame({
        "Gesture_Encoded": gesture_list,
        "Numeric_PIDs": subject_list,
        "source": source_list,
        "gesture_num": gesture_num_list,
    }).reset_index(drop=True)

    return features, meta_df


def train_dimreduc_transforms(train_features, test_features=None, use_pca=True, use_tsne=True, use_umap=True, apply_train=False):
    pca2d, tsne2d, umap2d = None, None, None
    
    if apply_train:
        # Fit on train, apply to both train and test
        if use_pca:
            pca = PCA(n_components=2)
            pca2d_train = pca.fit_transform(train_features)
            pca2d_test = pca.transform(test_features) if test_features is not None else None
            pca2d = np.concatenate([pca2d_train, pca2d_test], axis=0) if pca2d_test is not None else pca2d_train

        if use_tsne:
            print("tSNE cannot do out-of-sample transformations! Skipping tSNE.")
            tsne2d = None

        if use_umap:
            reducer = umap.UMAP(n_components=2, random_state=42)
            umap2d_train = reducer.fit_transform(train_features)
            umap2d_test = reducer.transform(test_features) if test_features is not None else None
            umap2d = np.concatenate([umap2d_train, umap2d_test], axis=0) if umap2d_test is not None else umap2d_train

        # This is all the latent data, the trained on latent and the applied tested data, concatenated
        return pca2d, tsne2d, umap2d

    else:
        # This should cover both ALL and None

        # Fit and transform whatever is passed in
        features = train_features if test_features is None else np.concatenate([train_features, test_features], axis=0)
        pca2d = PCA(n_components=2).fit_transform(features) if use_pca else None
        tsne2d = TSNE(n_components=2, random_state=42).fit_transform(features) if use_tsne else None
        umap2d = umap.UMAP(n_components=2, random_state=42).fit_transform(features) if use_umap else None
        return pca2d, tsne2d, umap2d


def create_latent_rep_fig(
    user_df, 
    pca2d=None, tsne2d=None, umap2d=None, 
    ft_user_to_highlight=None,
    my_title=None
):
    """
    Plots PCA/tSNE/UMAP 2D projections, differentiating train and test with marker shape.
    If any embedding is None, an empty subplot is drawn.
    user_df must include: 'Numeric_PIDs', 'Gesture_Encoded', and 'source' columns.
    """

    # Collect embeddings and their names
    methods = [("PCA", pca2d), ("t-SNE", tsne2d), ("UMAP", umap2d)]
    n_rows = len(methods)
    n_cols = 2  # Color by subject (col 0), gesture (col 1)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(13, 4.5 * n_rows))
    plt.subplots_adjust(hspace=0.35, wspace=0.2)
    if n_rows == 1:
        axs = np.expand_dims(axs, 0)

    # Plot settings
    marker_map = {"Train": "o", "Test": "^"}  # Test is now a star!
    if marker_map["Test"] == "*":
        test_size = 200
    else:
        test_size = 70
    marker_size_map = {"Train": 70, "Test": test_size}  # Make stars bigger
    alpha_dict = {"Train": 0.9, "Test": 0.6}
    label_names = ["Numeric_PIDs", "Gesture_Encoded"]
    colormaps = ["tab20", "tab10"]  # tab20 for subjects (at least 20 colors), tab10 for gestures since there's only 10
    label_titles = ["Subject", "Gesture"]

    user_df['PID_color'] = LabelEncoder().fit_transform(user_df['Numeric_PIDs'])
    user_df['Gesture_color'] = LabelEncoder().fit_transform(user_df['Gesture_Encoded'])

    # before your loops
    pid_codes = user_df['PID_color']
    pid_norm  = mpl.colors.Normalize(vmin=pid_codes.min(), vmax=pid_codes.max())

    for row, (name, emb) in enumerate(methods):  # This loops through PCA, tSNE, and UMAP (3)
        for col, (label, cmap, ltitle) in enumerate(zip(label_names, colormaps, label_titles)):  # This goes through subject then gesture (2)
            ax = axs[row, col]
            if emb is None:
                ax.axis('off')
                ax.text(0.5, 0.5, f"No {name} embedding available",
                        ha='center', va='center', fontsize=14, color='gray')
                continue

            # Decide which color column to use
            if label == "Numeric_PIDs":
                colors = user_df['PID_color']
            else:
                colors = user_df['Gesture_color']

            for source, marker in marker_map.items():
                mask = user_df["source"] == source
                
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=colors[mask],
                    cmap=cmap,
                    norm=pid_norm,       # <<< force a common scale, otherwise matplotlib automatically will rescale
                    marker=marker,
                    alpha=alpha_dict[source],
                    edgecolor='k',
                    s=marker_size_map[source],
                    label=source
                )

            # Highlight FT user after plotting other points
            if ft_user_to_highlight is not None:
                try:
                    ft_int = int(str(ft_user_to_highlight).replace("P", ""))
                    ft_mask = (user_df["Numeric_PIDs"].astype(int) == ft_int)
                    if ft_mask.any():
                        ax.scatter(
                            emb[ft_mask, 0], emb[ft_mask, 1],
                            c='none', edgecolor='yellow', linewidth=2.5,
                            s=150, marker=marker_map["Test"], label='Finetuned User'
                        )
                except Exception:
                    pass

            ax.set_title(f"{name} (colored by {ltitle})", fontsize=13)
            ax.set_xlabel("Dim 1", fontsize=11)
            ax.set_ylabel("Dim 2", fontsize=11)

            # Legend
            handles, labels_ = ax.get_legend_handles_labels()
            # Remove duplicate legend entries
            from collections import OrderedDict
            legend_dict = OrderedDict(zip(labels_, handles))
            ax.legend(legend_dict.values(), legend_dict.keys(), loc='best', fontsize=10, frameon=True)

    if my_title is not None:
        fig.suptitle(my_title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def full_latent_rep_pipeline(
    model, 
    user_df, 
    ft_user_to_highlight=None,
    use_pca=True, use_tsne=True, use_umap=True,
    n_train_users=8, n_test_users=8, 
    combined_method=None,
    n_reps_per_gesture=1
):
    if combined_method is None:
        # We are running on either train or test, idk which, it shouldn't matter
        # Only extract one and the other is False...
        features, meta = extract_latent_features(model, user_df, n_users=n_train_users, n_reps_per_gesture=n_reps_per_gesture)
        # Just fit on all features passed in (for backward compatibility)
        pca2d, tsne2d, umap2d = train_dimreduc_transforms(features, use_pca=use_pca, use_tsne=use_tsne, use_umap=use_umap)
        create_latent_rep_fig(meta, pca2d=pca2d, tsne2d=tsne2d, umap2d=umap2d,
                              ft_user_to_highlight=ft_user_to_highlight)
    else:
        # Slice to get train and test dfs
        train_df = user_df[user_df["source"]=="Train"]
        test_df = user_df[user_df["source"]=="Test"]

        # Get train/test features and metadata (ensure your feature extractor supports this)
        train_features, train_meta = extract_latent_features(model, train_df, n_users=n_train_users, n_reps_per_gesture=n_reps_per_gesture)
        test_features, test_meta = extract_latent_features(model, test_df, n_users=n_test_users, n_reps_per_gesture=n_reps_per_gesture)

        if combined_method.upper()=="ALL":
            # Fit transform on all data, then plot all
            all_features = np.concatenate([train_features, test_features], axis=0)
            all_meta = pd.concat([train_meta, test_meta], ignore_index=True)
            pca2d, tsne2d, umap2d = train_dimreduc_transforms(all_features, use_pca=use_pca, use_tsne=use_tsne, use_umap=use_umap)
            create_latent_rep_fig(all_meta, pca2d=pca2d, tsne2d=tsne2d, umap2d=umap2d,
                                ft_user_to_highlight=ft_user_to_highlight)
        elif combined_method.upper()=="APPLY_TRAIN":
            # Fit transform on the train data, then apply to test
            #all_features = np.concatenate([train_features, test_features], axis=0)
            all_meta = pd.concat([train_meta, test_meta], ignore_index=True)
            #print("tSNE cannot be applied to new data! Overwriting tSNE to False!")
            pca2d, tsne2d, umap2d = train_dimreduc_transforms(train_features=train_features, test_features=test_features, use_pca=use_pca, use_tsne=use_tsne, use_umap=use_umap, apply_train=True)
            create_latent_rep_fig(all_meta, pca2d=pca2d, umap2d=umap2d,
                                ft_user_to_highlight=ft_user_to_highlight)

