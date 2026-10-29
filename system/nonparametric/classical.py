"""
Classical / non-parametric few-shot baselines on cached episodes (docs/SOTA_COMPARISONS.md, Tier 1).

Every method maps a MaterializedEpisode (support + query arrays, local labels) to predicted local
labels for the query samples. They use ONLY the support set (no query labels, no other users; the one
exception is td_lda_pool, which also uses the fold's TRAIN users, never val/test);
they are deterministic given the episode. Inputs are the per-trial `$B`-normalised tensors
(reps, C, T) from the shared tensor dict, EMG and IMU kept separate so each method can fuse them
the way it was designed to.

Methods
  dollar_b        faithful `$B`: per-template PCA (50 PCs of the template's own channel covariance),
                  project the query into each template's space, L1 distance, 1-NN
                  (run_$B.ipynb::exact_dollar_B_classify, early-fused 88 channels).
  dollar_b_user   one PCA per episode (fit on all support time steps), L1 1-NN (the notebook's
                  single_user_encoder variant).
  knn_l1          1-NN with L1 distance on the raw flattened (EMG+IMU) tensors.
  proto_euclid    nearest class-mean, Euclidean, raw flattened tensors.
  riemann_mdm     per-modality channel covariance (OAS), nearest class Riemannian mean, distances
                  summed over modalities.
  riemann_ts      per-modality covariance (OAS) -> tangent space at the support mean -> concat ->
                  standardise -> logistic regression.
  minirocket      MiniRocket features (aeon) fitted on the support set, RidgeClassifierCV (LOO alpha).
  td_knn          1-NN (Euclidean) on trial-level summary features (mean and std over time of every
                  EMG and IMU channel). With the `td64` EMG dict these are Hudgins MAV/WL/ZC/SSC.
  td_lda          shrinkage LDA on the same features; support-only, so at K=1 (zero within-class
                  scatter) it reduces to nearest class mean (shrinkage = 1).
  td_lda_pool     LDA whose pooled within-class covariance is estimated from the fold's TRAIN users
                  (`fit_whitener`), then nearest class mean in the whitened space; the strongest
                  fair LDA at K=1. Needs `whitener=` (A6 runner builds it per fold).
"""
import numpy as np

METHODS = ("dollar_b", "dollar_b_user", "knn_l1", "proto_euclid", "riemann_mdm", "riemann_ts",
           "minirocket", "td_knn", "td_lda", "td_lda_pool")
NEEDS_FIT = ("td_lda_pool",)                         # extra fitted state beyond the episode
STATELESS = tuple(m for m in METHODS if m not in NEEDS_FIT)


# ─────────────────────────────────────────────────────────────────────────────
def _fuse(emg, imu):
    if imu is None:
        raise ValueError("these baselines expect EMG+IMU; imu is None")
    return np.concatenate([emg, imu], axis=1)          # (n, C_emg + C_imu, T)


def _pca_basis(x_ct, n_pc):
    """x_ct: (C, T). Returns mean (C,1) and U (C, n_pc): top eigenvectors of the channel covariance."""
    mean = x_ct.mean(axis=1, keepdims=True)
    xc = x_ct - mean
    cov = xc @ xc.T / (x_ct.shape[1] - 1)
    w, v = np.linalg.eigh(cov)
    return mean, v[:, -n_pc:][:, ::-1]


def dollar_b(ep, n_components: int = 50):
    sup, qry = _fuse(ep.emg_support, ep.imu_support), _fuse(ep.emg_query, ep.imu_query)
    n_pc = min(n_components, sup.shape[1] - 1)
    dists = np.zeros((qry.shape[0], sup.shape[0]))
    for i in range(sup.shape[0]):
        mean, U = _pca_basis(sup[i], n_pc)
        proj_t = (U.T @ (sup[i] - mean)).ravel()
        proj_q = np.einsum("cp,qct->qpt", U, qry - mean[None]).reshape(qry.shape[0], -1)
        dists[:, i] = np.abs(proj_q - proj_t[None]).sum(axis=1)
    return ep.y_support[dists.argmin(axis=1)]


def dollar_b_user(ep, n_components: int = 50):
    sup, qry = _fuse(ep.emg_support, ep.imu_support), _fuse(ep.emg_query, ep.imu_query)
    n_pc = min(n_components, sup.shape[1] - 1)
    flat = sup.transpose(1, 0, 2).reshape(sup.shape[1], -1)
    mean, U = _pca_basis(flat, n_pc)
    ps = np.einsum("cp,sct->spt", U, sup - mean[None]).reshape(sup.shape[0], -1)
    pq = np.einsum("cp,qct->qpt", U, qry - mean[None]).reshape(qry.shape[0], -1)
    d = np.abs(pq[:, None, :] - ps[None, :, :]).sum(axis=2)
    return ep.y_support[d.argmin(axis=1)]


def knn_l1(ep):
    s = _fuse(ep.emg_support, ep.imu_support).reshape(len(ep.y_support), -1)
    q = _fuse(ep.emg_query, ep.imu_query).reshape(len(ep.y_query), -1)
    d = np.abs(q[:, None, :] - s[None, :, :]).sum(axis=2)
    return ep.y_support[d.argmin(axis=1)]


def proto_euclid(ep):
    s = _fuse(ep.emg_support, ep.imu_support).reshape(len(ep.y_support), -1)
    q = _fuse(ep.emg_query, ep.imu_query).reshape(len(ep.y_query), -1)
    labels = np.arange(ep.n_way)
    protos = np.stack([s[ep.y_support == c].mean(axis=0) for c in labels])
    d = ((q[:, None, :] - protos[None]) ** 2).sum(axis=2)
    return labels[d.argmin(axis=1)]


def _covs(x, estimator="oas"):
    from pyriemann.estimation import Covariances
    return Covariances(estimator=estimator).fit_transform(x.astype(np.float64))


def riemann_mdm(ep, estimator: str = "oas", metric: str = "riemann"):
    from pyriemann.classification import MDM
    dist = None
    for s, q in ((ep.emg_support, ep.emg_query), (ep.imu_support, ep.imu_query)):
        mdm = MDM(metric=metric).fit(_covs(s, estimator), ep.y_support)
        d = mdm.transform(_covs(q, estimator))             # (n_query, n_way) distances
        dist = d if dist is None else dist + d
    return mdm.classes_[dist.argmin(axis=1)]


def riemann_ts(ep, estimator: str = "oas", C: float = 1.0):
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    fs, fq = [], []
    for s, q in ((ep.emg_support, ep.emg_query), (ep.imu_support, ep.imu_query)):
        ts = TangentSpace(metric="riemann").fit(_covs(s, estimator))
        fs.append(ts.transform(_covs(s, estimator)))
        fq.append(ts.transform(_covs(q, estimator)))
    xs, xq = np.concatenate(fs, axis=1), np.concatenate(fq, axis=1)
    sc = StandardScaler().fit(xs)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(xs), ep.y_support)
    return clf.predict(sc.transform(xq))


def minirocket(ep, n_kernels: int = 5000, seed: int = 0):
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV
    xs = _fuse(ep.emg_support, ep.imu_support).astype(np.float32)
    xq = _fuse(ep.emg_query, ep.imu_query).astype(np.float32)
    tf = MiniRocket(n_kernels=n_kernels, random_state=seed).fit(xs)
    fs, fq = tf.transform(xs), tf.transform(xq)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(fs, ep.y_support)
    return clf.predict(fq)


def summary_features(emg, imu):
    """(n, Ce, T), (n, Ci, T) -> (n, 2 * (Ce + Ci)): mean and std over time of every channel."""
    x = np.concatenate([emg, imu], axis=1).astype(np.float64)
    return np.concatenate([x.mean(axis=2), x.std(axis=2)], axis=1)


def td_knn(ep):
    s = summary_features(ep.emg_support, ep.imu_support) if ep.imu_support is not None else None
    if s is None:
        raise ValueError("these baselines expect EMG+IMU; imu is None")
    q = summary_features(ep.emg_query, ep.imu_query)
    d = ((q[:, None, :] - s[None, :, :]) ** 2).sum(axis=2)
    return ep.y_support[d.argmin(axis=1)]


def td_lda(ep):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if ep.imu_support is None:
        raise ValueError("these baselines expect EMG+IMU; imu is None")
    xs = summary_features(ep.emg_support, ep.imu_support)
    xq = summary_features(ep.emg_query, ep.imu_query)
    if ep.k_shot == 1:
        # within-class scatter is exactly zero, so full shrinkage (identity covariance, equal priors)
        # is a nearest class mean; sklearn's LDA refuses n_samples == n_classes
        d = ((xq[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2)
        return ep.y_support[d.argmin(axis=1)]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                     priors=np.full(ep.n_way, 1.0 / ep.n_way))
    return clf.fit(xs, ep.y_support).predict(xq)


def fit_whitener(tensor_dict, train_pids):
    """Pooled within-(user, class) covariance of the summary features over TRAIN users only, as a
    whitening matrix W (features @ W has ~identity within-class covariance). Ledoit-Wolf shrinkage.
    tensor_dict must be reoriented to (reps, C, T)."""
    from sklearn.covariance import LedoitWolf
    centred = []
    for pid in sorted(train_pids):
        for slot in tensor_dict[pid].values():
            f = summary_features(slot["emg"].numpy(), slot["imu"].numpy())
            centred.append(f - f.mean(axis=0, keepdims=True))
    cov = LedoitWolf().fit(np.concatenate(centred)).covariance_
    w, v = np.linalg.eigh(cov)
    return (v / np.sqrt(np.maximum(w, 1e-12))) @ v.T


def td_lda_pool(ep, whitener=None):
    if whitener is None:
        raise ValueError("td_lda_pool needs whitener= (classical.fit_whitener on the fold's train users)")
    if ep.imu_support is None:
        raise ValueError("these baselines expect EMG+IMU; imu is None")
    xs = summary_features(ep.emg_support, ep.imu_support) @ whitener
    xq = summary_features(ep.emg_query, ep.imu_query) @ whitener
    labels = np.arange(ep.n_way)
    protos = np.stack([xs[ep.y_support == c].mean(axis=0) for c in labels])
    d = ((xq[:, None, :] - protos[None]) ** 2).sum(axis=2)
    return labels[d.argmin(axis=1)]


_IMPL = {"dollar_b": dollar_b, "dollar_b_user": dollar_b_user, "knn_l1": knn_l1,
         "proto_euclid": proto_euclid, "riemann_mdm": riemann_mdm, "riemann_ts": riemann_ts,
         "minirocket": minirocket, "td_knn": td_knn, "td_lda": td_lda, "td_lda_pool": td_lda_pool}
assert set(_IMPL) == set(METHODS)


def predict(method: str, mep, **kwargs):
    """Predicted local labels (n_query,) for a MaterializedEpisode. Uses support only."""
    if method not in _IMPL:
        raise KeyError(f"unknown method {method!r}; choose from {METHODS}")
    pred = np.asarray(_IMPL[method](mep, **kwargs))
    if pred.shape != mep.y_query.shape:
        raise AssertionError(f"{method}: prediction shape {pred.shape} != {mep.y_query.shape}")
    return pred
