"""
Classical / non-parametric few-shot baselines on cached episodes (docs/SOTA_COMPARISONS.md, Tier 1).

Every method maps a MaterializedEpisode (support + query arrays, local labels) to predicted local
labels for the query samples. They use ONLY the support set (no query labels, no other users);
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
"""
import numpy as np

METHODS = ("dollar_b", "dollar_b_user", "knn_l1", "proto_euclid", "riemann_mdm", "riemann_ts",
           "minirocket")


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


_IMPL = {"dollar_b": dollar_b, "dollar_b_user": dollar_b_user, "knn_l1": knn_l1,
         "proto_euclid": proto_euclid, "riemann_mdm": riemann_mdm, "riemann_ts": riemann_ts,
         "minirocket": minirocket}
assert set(_IMPL) == set(METHODS)


def predict(method: str, mep, **kwargs):
    """Predicted local labels (n_query,) for a MaterializedEpisode. Uses support only."""
    if method not in _IMPL:
        raise KeyError(f"unknown method {method!r}; choose from {METHODS}")
    pred = np.asarray(_IMPL[method](mep, **kwargs))
    if pred.shape != mep.y_query.shape:
        raise AssertionError(f"{method}: prediction shape {pred.shape} != {mep.y_query.shape}")
    return pred
