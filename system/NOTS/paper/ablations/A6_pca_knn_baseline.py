"""
A6_pca_knn_baseline.py
=======================
Ablation A6: Prior Work — PCA + KNN (Non-DL Anchor)

TODO: Reproduce the subject-specific PCA + KNN baseline from prior work.
      This script is a STUB. Before running:

      1. Identify the exact PCA + KNN implementation from the cited prior work.
         Key details to match:
           - Which features go into PCA? (raw EMG? windowed RMS? time-domain features?)
           - How many PCA components are used?
           - What is the KNN neighbour count (k)?
           - Is PCA fit per-subject or across subjects?

      2. This is NOT a neural network. No GPU needed. Run on CPU.

      3. Evaluation must use THE SAME episodic episodes as all other ablations
         (same random seed, same test_PIDs) for a fair comparison.

      4. See the spec note: "Reproduce exactly as described in the cited prior work."

Requirements:
  pip install scikit-learn

Architecture decision you need to make with your PI:
  The spec says "per-sample PCA" — clarify whether this means:
    (a) PCA fit on the support set of each episode, then applied to the query, OR
    (b) PCA fit per-subject globally (as a feature extractor), then 1-NN on episodes.
  Option (a) mirrors the MAML evaluation protocol most faithfully.
"""

import os, sys, json
import numpy as np

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))

from ablation_config import (
    make_base_config, FIXED_SEED, NUM_FINAL_SEEDS, NUM_TEST_EPISODES,
    save_results, TEST_PIDS, RUN_DIR,
)

raise NotImplementedError(
    "A6 is a stub. See the docstring at the top of this file for implementation notes. "
    "You need to: (1) clarify the prior work's PCA/KNN spec with your PI, "
    "(2) implement the feature extraction pipeline, "
    "(3) wrap it in the episodic eval loop below."
)


# =============================================================================
# IMPLEMENTATION SKETCH (fill in once you have the prior work details)
# =============================================================================

# from sklearn.decomposition import PCA
# from sklearn.neighbors import KNeighborsClassifier
# import pickle

# PCA_N_COMPONENTS = ???   # match prior work
# KNN_K            = 1     # 1-NN for 1-shot evaluation

# def extract_features(emg_tensor):
#     """
#     Convert raw EMG to the feature representation used in prior work.
#     emg_tensor: (B, C, T)
#     Returns: (B, feat_dim) numpy array
#     """
#     # TODO: implement feature extraction (e.g. RMS per channel, etc.)
#     raise NotImplementedError

# def run_episode(support_emg, support_labels, query_emg, query_labels,
#                 n_components=PCA_N_COMPONENTS):
#     """Run one episodic PCA+KNN trial."""
#     X_sup = extract_features(support_emg)
#     X_qry = extract_features(query_emg)
#     y_sup = support_labels.numpy()
#     y_qry = query_labels.numpy()
#
#     pca = PCA(n_components=min(n_components, X_sup.shape[0] - 1, X_sup.shape[1]))
#     pca.fit(X_sup)
#     X_sup_pca = pca.transform(X_sup)
#     X_qry_pca = pca.transform(X_qry)
#
#     knn = KNeighborsClassifier(n_neighbors=KNN_K)
#     knn.fit(X_sup_pca, y_sup)
#     acc = knn.score(X_qry_pca, y_qry)
#     return acc

# def main():
#     config = make_base_config(ablation_id="A6")
#
#     # Load tensor_dict
#     tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
#     ...
#
#     # Run episodic eval (must use same episodes as other ablations — fix seed!)
#     # Use the MetaGestureDataset with seed=FIXED_SEED so episodes are identical.
#     ...

if __name__ == "__main__":
    main()
