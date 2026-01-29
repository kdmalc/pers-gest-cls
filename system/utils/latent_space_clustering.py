# === Clustering on latent space from a pretrained model with forward_features ===

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd

###############################################################
### EXTRACTING SELECTED CLUSTERS ###
###############################################################

from typing import Dict, Tuple, Callable, Optional, Any
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------
# 0) Utilities: seeds, metrics, assigners, sorting
# ---------------------------------------------------------------------

def set_global_seed(seed: int = 17):
    """
    Make sklearn/np operations deterministic. (For PyTorch determinism,
    call your existing set_seed() before training your DNNs.)
    """
    np.random.seed(seed)
    # sklearn uses numpy's RNG internally, so this is sufficient on the sklearn side.

def safe_scores(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute silhouette, Calinski–Harabasz, Davies–Bouldin safely.
    Returns (silhouette, ch, db) with np.nan when undefined.
    """
    uniq = np.unique(labels)
    if len(uniq) <= 1 or len(uniq) >= len(labels):
        return (np.nan, np.nan, np.nan)
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan
    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan
    return sil, ch, db

def build_out_of_sample_assigner(Xtr: np.ndarray, labels: np.ndarray, model_obj: Any) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function f(X_new) -> cluster_labels for new data.
    Uses model.predict if available; otherwise falls back to nearest centroid.
    """
    if hasattr(model_obj, "predict"):
        return lambda X_new: model_obj.predict(X_new)

    # Fallback: nearest centroid in the standardized space
    centroids = []
    order = []
    for c in np.unique(labels):
        centroids.append(Xtr[labels == c].mean(axis=0))
        order.append(int(c))
    centroids = np.vstack(centroids)

    def _assign(X_new: np.ndarray) -> np.ndarray:
        d = cdist(X_new, centroids, metric="euclidean")
        nearest = np.argmin(d, axis=1)
        # map column index back to actual cluster id in `order`
        return np.array([order[j] for j in nearest], dtype=int)

    return _assign

def rank_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank by: higher silhouette, higher CH, lower DB.
    """
    _df = df.copy()
    _df["sil_rank"] = (-_df["Silhouette_Score"]).rank(method="min")
    _df["ch_rank"]  = (-_df["CH_Index"]).rank(method="min")
    _df["db_rank"]  = (_df["DB_Index"]).rank(method="min")
    _df["rank_sum"] = _df[["sil_rank","ch_rank","db_rank"]].sum(axis=1)
    _df = _df.sort_values(["rank_sum","Silhouette_Score","CH_Index","DB_Index"], ascending=[True, False, False, True])
    return _df.drop(columns=["sil_rank","ch_rank","db_rank","rank_sum"])

# ---------------------------------------------------------------------
# 1) Clustering on fixed configs + labels + partition plan
# ---------------------------------------------------------------------

def cluster_selected_and_score(
    X_train: np.ndarray,
    random_state: int = 42,
    configs: Tuple[Tuple[str, int], ...] = (
        ("GaussianMixture", 6),
        ("KMeans", 8),
        ("MiniBatchKMeans", 3),
        ("Birch", 3),
    ),
    deterministic: bool = True,
):
    """
    Returns:
      results_df : DataFrame with clustering metrics
      fitted     : (algo,k) -> fitted sklearn model
      labels_dict: (algo,k) -> np.array of labels for each row in X_train
      partition  : (algo,k) -> {cluster_id: idx_array}
      meta       : (algo,k) -> metadata dict
      scaler     : StandardScaler fitted on X_train
    """
    if deterministic:
        set_global_seed(random_state)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)

    rows = []
    fitted: Dict[Tuple[str,int], Any] = {}
    labels_dict: Dict[Tuple[str,int], np.ndarray] = {}
    partition: Dict[Tuple[str,int], Dict[int, np.ndarray]] = {}
    meta: Dict[Tuple[str,int], Dict[str, Any]] = {}

    for algo_name, k in configs:
        key = (algo_name, k)
        try:
            if algo_name == "GaussianMixture":
                model_obj = GaussianMixture(
                    n_components=k, covariance_type="full",
                    random_state=random_state, n_init=5
                )
                model_obj.fit(Xtr)
                labels = model_obj.predict(Xtr)
                elbow_val = float(model_obj.bic(Xtr))
                soft_resp = model_obj.predict_proba(Xtr)

            elif algo_name == "KMeans":
                model_obj = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                labels = model_obj.fit_predict(Xtr)
                elbow_val = float(model_obj.inertia_)
                soft_resp = None

            elif algo_name == "MiniBatchKMeans":
                model_obj = MiniBatchKMeans(
                    n_clusters=k, random_state=random_state, n_init=10,
                    batch_size=1024, max_no_improvement=20
                )
                labels = model_obj.fit_predict(Xtr)
                elbow_val = float(model_obj.inertia_)
                soft_resp = None

            elif algo_name == "Birch":
                model_obj = Birch(n_clusters=k, threshold=0.5, branching_factor=50)
                labels = model_obj.fit_predict(Xtr)
                elbow_val = np.nan
                soft_resp = None

            else:
                raise ValueError(f"Unsupported algorithm: {algo_name}")

            # Metrics (silhouette / CH / DB)
            sil, ch, db = safe_scores(Xtr, labels)

            # Store outputs
            fitted[key] = model_obj
            labels_dict[key] = labels
            part = {int(c): np.where(labels == c)[0] for c in np.unique(labels)}
            partition[key] = part

            sizes = {int(c): int(len(idx)) for c, idx in part.items()}
            md = {
                "algo": algo_name, "k": k, "elbow": elbow_val,
                "silhouette": float(sil) if sil is not None else np.nan,
                "calinski_harabasz": float(ch) if ch is not None else np.nan,
                "davies_bouldin": float(db) if db is not None else np.nan,
                "cluster_sizes": sizes,
            }
            if soft_resp is not None:
                md["soft_resp_shape"] = list(soft_resp.shape)
            meta[key] = md

            rows.append({
                "Clustering_Algorithm": algo_name,
                "Num_Clusters": k,
                "Silhouette_Score": sil,
                "CH_Index": ch,
                "DB_Index": db
            })

        except Exception as e:
            rows.append({
                "Clustering_Algorithm": algo_name,
                "Num_Clusters": k,
                "Silhouette_Score": np.nan,
                "CH_Index": np.nan,
                "DB_Index": np.nan
            })
            print(f"[WARN] {algo_name}@{k} failed: {e}")

    results_df = pd.DataFrame(rows)
    results_df = rank_and_sort(results_df)
    return results_df, fitted, labels_dict, partition, meta, scaler

# ---------------------------------------------------------------------
# 2) Helpers for (optional) intra-test assignment & feature extraction
# ---------------------------------------------------------------------

# Extracts data from a PD DATAFRAME! For some reason only returns 80 instead of 96...
## This DOES NOT extract latent features... this just converts from df to npy
def extract_feature_matrix_from_df(df: pd.DataFrame, feature_column: str = "feature") -> np.ndarray:
    """
    Expects df[feature_column] to hold per-row array-likes with consistent length.
    Returns stacked 2D array [N, D].
    """
    if feature_column not in df.columns:
        raise KeyError(f"'{feature_column}' not found in DataFrame.")
    # Safely stack; if entries are tensors, convert to numpy first
    vals = df[feature_column].to_numpy()
    X = np.vstack([np.asarray(v) for v in vals])
    return X

@torch.no_grad()
def embed_df_to_latents(
    df: pd.DataFrame,
    model,
    feature_column: str = "feature",
    device=None,
    batch_size: int = 256,
    num_channels: int = 16,        # <— will pass from config
    expect_layout: str = "BCT",    # model expects [B, C, T]
):
    """
    Convert df[feature_column] -> torch batch -> model.forward_features -> np latents.

    Handles inputs that are:
      - flat vectors per row: [D] where D == num_channels * T (we reshape to [C,T])
      - 2D signals per row: [C, T] or [T, C]
      - already batched chunks that stack to [B, D], [B, C, T], or [B, T, C]

    Returns
      latents: [N, D_latent] as np.ndarray
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    feats = []
    vals = df[feature_column].to_numpy()  # preserves row order

    for i in range(0, len(vals), batch_size):
        chunk = vals[i:i + batch_size]
        # Convert each row to np.array and ensure at least 1D
        rows = [np.asarray(v) for v in chunk]

        # Stack naively; we'll fix layout below
        x = np.stack(rows, axis=0)  # candidate shapes: [B, D] | [B, C, T] | [B, T, C]

        # ---- Make x into [B, C, T] ----
        if x.ndim == 2:
            # [B, D] flat — infer T from D and num_channels
            B, D = x.shape
            if D % num_channels != 0:
                raise ValueError(
                    f"embed_df_to_latents: cannot reshape flat input [B={B}, D={D}] "
                    f"to [B, C={num_channels}, T]. D must be a multiple of num_channels."
                )
            T = D // num_channels
            x = x.reshape(B, num_channels, T)  # [B, C, T]
        elif x.ndim == 3:
            B, A, B_or_T = x.shape
            # If already [B, C, T], great.
            if A == num_channels:
                pass
            # If [B, T, C], transpose
            elif B_or_T == num_channels:
                x = np.transpose(x, (0, 2, 1))  # [B, C, T]
            else:
                # Maybe each row was [C, T] and stacking kept that; already [B, C, T] then.
                # If neither dim equals num_channels, we can't infer.
                raise ValueError(
                    f"embed_df_to_latents: 3D input has shape {x.shape}, but "
                    f"neither dim 1 nor 2 equals num_channels={num_channels}. "
                    f"Please ensure each row is [C,T] or [T,C] with C={num_channels}."
                )
        else:
            raise ValueError(
                f"embed_df_to_latents: expected 2D or 3D stacked array, got {x.ndim}D with shape {x.shape}."
            )

        # To torch
        x = torch.as_tensor(x, dtype=torch.float32, device=device)  # [B, C, T]

        # Encode
        z = model.forward_features(x)   # -> [B, D_latent] or (z, aux)
        if isinstance(z, (tuple, list)):
            z = z[0]
        feats.append(z.detach().cpu().numpy())

    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------
# 3) Train a DNN per cluster for each (algo,k)
# ---------------------------------------------------------------------
def train_models_per_cluster_from_labels(
    data_dfs_dict: Dict[str, pd.DataFrame],
    fitted_models: Dict[Tuple[str,int], Any],          # (algo,k) -> fitted clusterer (KMeans/GMM/Birch/MBKMeans)
    labels_dict: Dict[Tuple[str,int], np.ndarray],     # (algo,k) -> train labels computed on X_train_latent
    scaler: StandardScaler,                            # fitted on X_train_latent
    encoder_model,                                     # your feature extractor with .forward_features
    device=None,
    config: Optional[Dict[str, Any]] = None,
    feature_column: str = "feature",
    cluster_colname: str = "Cluster_ID",
    skip_tiny_clusters_below: int = 3,
    # hooks you already have:
    train_DNN_cluster_model: Optional[Callable] = None,
    test_models_on_clusters: Optional[Callable] = None,
) -> Tuple[list, dict, dict, dict, dict]:
    """
    For each (algo,k):
      - Attaches known train Cluster_IDs from labels_dict (latent-space clustering).
      - Embeds intra_test_df -> latent, scales with the SAME scaler, predicts clusters with the SAME model.
      - Trains one DNN per cluster and evaluates.

    scaler MUST be fit on X_train_latent; do NOT use it on raw inputs.
    """
    assert train_DNN_cluster_model is not None, "Please pass train_DNN_cluster_model."
    assert test_models_on_clusters is not None, "Please pass test_models_on_clusters."

    base_train_df = data_dfs_dict['pretrain_df']
    base_intra_test_df = data_dfs_dict['pretrain_subject_test_df']

    merge_log = []  # compatibility
    intra_cluster_performance = {}
    cross_cluster_performance = {}
    nested_clus_model_dict = {}
    all_clus_logs_dict = {}

    for (algo, k), train_labels in labels_dict.items():
        algo_key = f"{algo}@{k}"
        print(f"\n=== Training per-cluster DNNs for {algo_key} (latent-space) ===")

        if len(train_labels) != len(base_train_df):
            raise ValueError(
                f"Label length mismatch for {algo_key}: labels={len(train_labels)} vs train_df={len(base_train_df)}"
            )

        # 1) Attach train labels (already computed in latent space)
        train_df = base_train_df.copy()
        train_df[cluster_colname] = train_labels

        # Filter tiny clusters
        cluster_ids_all = sorted(np.unique(train_labels).astype(int))
        cluster_ids = [cid for cid in cluster_ids_all if int((train_labels == cid).sum()) >= skip_tiny_clusters_below]
        for cid in cluster_ids_all:
            n = int((train_labels == cid).sum())
            if n < skip_tiny_clusters_below:
                print(f"[INFO] {algo_key}: skipping tiny cluster {cid} (n={n} < {skip_tiny_clusters_below})")
        if not cluster_ids:
            print(f"[WARN] {algo_key}: no clusters left to train after tiny-cluster filtering.")
            continue

        # 2) Embed intra/validation to latent, then assign clusters IN LATENT SPACE
        intra_test_df = base_intra_test_df.copy()

        X_intra_latent = embed_df_to_latents(
            intra_test_df, model=encoder_model, feature_column=feature_column, device=device, batch_size=256
        )
        X_intra_latent_std = scaler.transform(X_intra_latent)  # scale latents only

        clusterer = fitted_models[(algo, k)]
        intra_labels = clusterer.predict(X_intra_latent_std)   # works for KMeans, MBKMeans, GMM, Birch
        intra_test_df[cluster_colname] = intra_labels

        # Optional: build a validation partition by cluster if your trainer prefers it
        # val_partition = {int(c): np.where(intra_labels == c)[0] for c in np.unique(intra_labels)}

        # 3) Train a DNN per cluster (your hook should read Cluster_ID to split)
        new_models, cluster_logs_dict = train_DNN_cluster_model(
            train_df, intra_test_df, cluster_ids, config
        )
        nested_clus_model_dict[algo_key] = new_models
        all_clus_logs_dict[algo_key] = cluster_logs_dict

        # 4) Evaluate: your test function can evaluate a symmetric acc matrix by cluster
        trained_models_in_order = [new_models[cid] for cid in cluster_ids]
        sym_acc_arr = test_models_on_clusters(
            intra_test_df, trained_models_in_order, cluster_ids, config, pytorch_bool=True
        )

        # 5) Record metrics
        intra_cluster_performance[algo_key] = {}
        cross_cluster_performance[algo_key] = {}
        if sym_acc_arr is not None and sym_acc_arr.size:
            for i, cid in enumerate(cluster_ids):
                intra_val = float(sym_acc_arr[i, i])
                cross_vals = [sym_acc_arr[i, j] for j in range(len(cluster_ids)) if j != i]
                cross_avg = float(np.mean(cross_vals)) if len(cross_vals) else np.nan
                intra_cluster_performance[algo_key][cid] = [(0, intra_val)]
                cross_cluster_performance[algo_key][cid] = [(0, cross_avg)]
        else:
            for cid in cluster_ids:
                intra_cluster_performance[algo_key][cid] = [(0, np.nan)]
                cross_cluster_performance[algo_key][cid] = [(0, np.nan)]

    return merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict, all_clus_logs_dict

###############################################################
### EXPLORATION ###
###############################################################

# ----------------------------
# Feature extraction utilities
# ----------------------------
@torch.no_grad()
def extract_latent_features(model, dataloader, device=None):
    """Returns (features, indices) where features is [N, D].
       Assumes dataloader yields (x, *_) and model.forward_features(x)->[B, D].
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    feats = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        x = batch[0].to(device)
        z = model.forward_features(x)
        if isinstance(z, (tuple, list)):  # just in case your model returns (z, aux)
            z = z[0]
        feats.append(z.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

# ---------------------------------
# Clustering + evaluation utilities
# ---------------------------------
def safe_scores(X, labels):
    """Compute clustering metrics, guarding edge cases."""
    # invalid if only 1 cluster or cluster size/pathologies
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X):
        return np.nan, np.nan, np.nan
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan
    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan
    return sil, ch, db

def make_algorithms(random_state=42):
    """Factories for algorithms. Each value is a function taking n_clusters and returning an estimator."""
    algs = {
        "KMeans": lambda k: KMeans(n_clusters=k, random_state=random_state, n_init="auto"),
        "MiniBatchKMeans": lambda k: MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1024, n_init="auto"),
        "GaussianMixture": lambda k: GaussianMixture(n_components=k, covariance_type="full", random_state=random_state),
        "Agglomerative": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
        #"Spectral": lambda k: SpectralClustering(n_clusters=k, assign_labels="kmeans", random_state=random_state),
        "Birch": lambda k: Birch(n_clusters=k, threshold=0.5)
    }
    return algs

def algorithm_supports_predict(model_obj):
    """Native out-of-sample prediction availability."""
    return hasattr(model_obj, "predict")

def build_out_of_sample_assigner(X_train, labels_train, algo_model):
    """Return a callable f(X) -> labels for test data.
       - Use native .predict when available (KMeans, MBKMeans, GMM, Birch).
       - Otherwise, fall back to kNN in latent space trained on cluster labels.
    """
    if algorithm_supports_predict(algo_model):
        return lambda X: algo_model.predict(X)
    # Fallback: kNN on latent features with cluster labels from train
    knn = KNeighborsClassifier(n_neighbors=min(10, max(1, len(np.unique(labels_train)))))
    knn.fit(X_train, labels_train)
    return lambda X: knn.predict(X)

def plot_elbow_curves(elbow_store):
    """Make one plot per algorithm. For each, plot the recorded curve across k.
       Note: The 'elbow' quantity differs by algorithm:
         - KMeans/MiniBatchKMeans: inertia (WCSS) ↓ is better
         - GaussianMixture: BIC ↓ is better (we plot BIC)
         - Others: we use -Silhouette (↓ better) as a proxy 'elbow' curve
    """
    for algo_name, series in elbow_store.items():
        ks = sorted(series.keys())
        ys = [series[k] for k in ks]
        plt.figure()
        plt.plot(ks, ys, marker="o")
        plt.title(f"Elbow / Model-Selection Curve: {algo_name}")
        plt.xlabel("k (num clusters)")
        # Dynamic y-label based on algo
        if algo_name in ("KMeans", "MiniBatchKMeans"):
            plt.ylabel("Inertia (lower is better)")
        elif algo_name == "GaussianMixture":
            plt.ylabel("BIC (lower is better)")
        else:
            plt.ylabel("-Silhouette (lower is better)")
        plt.show()

def rank_and_sort(df):
    """Create a combined rank to sort 'best' first.
       - silhouette: higher better
       - CH: higher better
       - DB: lower better
       Combine normalized ranks.
    """
    df = df.copy()
    # Replace inf/nan safely
    for col in ["Silhouette_Score", "CH_Index", "DB_Index"]:
        df[col] = df[col].astype(float)

    # Ranks: smaller is better for DB; invert others
    df["rank_sil"] = (-df["Silhouette_Score"]).rank(method="average")  # negative so higher sil gets smaller rank
    df["rank_ch"]  = (-df["CH_Index"]).rank(method="average")
    df["rank_db"]  = (df["DB_Index"]).rank(method="average")  # lower DB is better
    df["Combined_Rank"] = df[["rank_sil", "rank_ch", "rank_db"]].mean(axis=1)
    df = df.sort_values(by=["Combined_Rank", "Silhouette_Score"], ascending=[True, False]).reset_index(drop=True)
    return df.drop(columns=["rank_sil", "rank_ch", "rank_db"])

def cluster_and_score(X_train, X_test=None, k_range=range(2, 16), random_state=42):
    """Main routine:
       - Standardizes features
       - Runs multiple clustering algorithms across k
       - Records metrics + 'elbow' curves
       - Builds out-of-sample assigners for each (algo, k)
       Returns:
         results_df, assigners
         where assigners[(algo_name, k)] is a callable mapping new X->labels
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test) if X_test is not None else None

    alg_factories = make_algorithms(random_state=random_state)
    rows = []
    elbow_store = defaultdict(dict)   # algo_name -> {k: score_for_curve}
    assigners = {}                    # (algo, k) -> callable for test mapping
    fitted_models = {}                # optional: store fitted model objects

    for algo_name, factory in alg_factories.items():
        for k in k_range:
            try:
                model_obj = factory(k)
                # Fit + labels
                if hasattr(model_obj, "fit_predict"):
                    labels = model_obj.fit_predict(Xtr)
                else:
                    model_obj.fit(Xtr)
                    # Some models (e.g., GMM) need predict on train to get labels
                    labels = model_obj.predict(Xtr)

                # Metrics
                sil, ch, db = safe_scores(Xtr, labels)

                # Elbow curve quantity by algorithm
                if algo_name in ("KMeans", "MiniBatchKMeans"):
                    elbow_val = float(model_obj.inertia_)
                elif algo_name == "GaussianMixture":
                    elbow_val = float(model_obj.bic(Xtr))
                else:
                    # proxy: use -silhouette to emulate an elbow-like curve
                    elbow_val = float(-sil) if np.isfinite(sil) else np.nan

                elbow_store[algo_name][k] = elbow_val

                # Build assigner for test use
                assigner = build_out_of_sample_assigner(Xtr, labels, model_obj)
                assigners[(algo_name, k)] = (scaler, assigner)  # keep scaler + assigner to handle new data
                fitted_models[(algo_name, k)] = model_obj

                rows.append({
                    "Clustering_Algorithm": algo_name,
                    "Num_Clusters": k,
                    "Silhouette_Score": sil,
                    "CH_Index": ch,
                    "DB_Index": db
                })

            except Exception as e:
                # Gracefully handle failures (e.g., Spectral failing on certain k)
                rows.append({
                    "Clustering_Algorithm": algo_name,
                    "Num_Clusters": k,
                    "Silhouette_Score": np.nan,
                    "CH_Index": np.nan,
                    "DB_Index": np.nan
                })
                # Still mark elbow as NaN for completeness
                elbow_store[algo_name][k] = np.nan

    results_df = pd.DataFrame(rows)
    results_df = rank_and_sort(results_df)

    # Plot elbow curves
    plot_elbow_curves(elbow_store)

    # Optionally: assign clusters to test set for the top result
    if Xte is not None:
        best_algo = results_df.iloc[0]["Clustering_Algorithm"]
        best_k = int(results_df.iloc[0]["Num_Clusters"])
        best_scaler, best_assigner = assigners[(best_algo, best_k)]
        test_labels = best_assigner(best_scaler.transform(X_test))
        print(f"\nTop config: {best_algo} (k={best_k}). Assigned {len(test_labels)} test samples to clusters.")
    else:
        test_labels = None

    return results_df, assigners, fitted_models

