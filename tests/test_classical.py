"""Classical baselines on cached episodes: sane accuracy on separable synthetic data, no label
leakage from the query set, determinism, error handling."""
import copy

import numpy as np
import pytest

from episodes import episode_cache as ec
from episodes.episode_numpy import materialize
from nonparametric import classical

pytest.importorskip("pyriemann")
pytest.importorskip("aeon")


def _td(tiny_config, es):
    from MAML.maml_data_pipeline import reorient_tensor_dict
    from synthetic import make_synthetic_tensor_dict
    payload = make_synthetic_tensor_dict(n_pids=len(es.users), n_classes=10, n_trials=10, seed=0)
    old = sorted(payload["data"].keys())
    payload["data"] = {new: payload["data"][o] for o, new in zip(old, es.users)}
    return reorient_tensor_dict(payload, tiny_config)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from tiny_config import make_tiny_config
    from synthetic import pid_names
    es = ec.load("test", 0)
    cfg = make_tiny_config(pid_names(6))
    return es, _td(cfg, es)


@pytest.mark.parametrize("method", classical.METHODS)
def test_method_beats_chance_on_separable_data_and_is_deterministic(setup, method):
    es, td = setup
    eps = es.select(3, 3)[:6]
    accs = []
    for ep in eps:
        mep = materialize(ep, td)
        p1 = classical.predict(method, mep)
        p2 = classical.predict(method, mep)
        assert np.array_equal(p1, p2)
        assert p1.shape == mep.y_query.shape and set(p1.tolist()) <= set(range(3))
        accs.append((p1 == mep.y_query).mean())
    assert np.mean(accs) > 0.55, f"{method}: {np.mean(accs):.2f} (chance 0.33)"


@pytest.mark.parametrize("method", classical.METHODS)
def test_predictions_do_not_depend_on_query_labels_or_ids(setup, method):
    es, td = setup
    mep = materialize(es.select(3, 1)[0], td)
    base = classical.predict(method, mep)
    other = copy.copy(mep)
    other.y_query = (mep.y_query + 1) % 3          # corrupt the query labels
    other.rep_query = mep.rep_query[::-1].copy()
    other.cls_query = mep.cls_query[::-1].copy()
    assert np.array_equal(classical.predict(method, other), base)


def test_unknown_method_and_missing_imu_raise(setup):
    es, td = setup
    mep = materialize(es.select(3, 1)[0], td)
    with pytest.raises(KeyError):
        classical.predict("nope", mep)
    no_imu = copy.copy(mep)
    no_imu.imu_support = None
    with pytest.raises(ValueError, match="imu is None"):
        classical.predict("knn_l1", no_imu)


def test_dollar_b_matches_torch_reference_from_notebook(setup):
    """Re-implementation check against a direct transcription of exact_dollar_B_classify."""
    import torch
    es, td = setup
    mep = materialize(es.select(3, 1)[0], td)
    sup = torch.from_numpy(np.concatenate([mep.emg_support, mep.imu_support], 1)).double()
    qry = torch.from_numpy(np.concatenate([mep.emg_query, mep.imu_query], 1)).double()
    n_pc = min(50, sup.shape[1] - 1)
    dist = torch.zeros(qry.shape[0], sup.shape[0], dtype=torch.double)
    for i in range(sup.shape[0]):
        t = sup[i]
        mean_c = t.mean(dim=1, keepdim=True)
        xc = t - mean_c
        cov = (xc @ xc.t()) / (t.shape[1] - 1)
        _, vecs = torch.linalg.eigh(cov)
        U = vecs[:, -n_pc:].flip(dims=[1])
        pt = (U.t() @ xc).flatten()
        for q in range(qry.shape[0]):
            dist[q, i] = ((U.t() @ (qry[q] - mean_c)).flatten() - pt).abs().sum()
    ref = torch.from_numpy(mep.y_support)[dist.argmin(dim=1)].numpy()
    assert np.array_equal(classical.predict("dollar_b", mep), ref)
