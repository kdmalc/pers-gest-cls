"""
The cluster-facing evaluation functions end to end on cached episodes (no real data):
run_episodic_test_eval, run_supervised_test_eval, and get_maml_dataloaders' cached val loader.
Synthetic tensors are re-keyed to the REAL PIDs of fold 0. Episode lists are truncated to a few
episodes per cell so the test is fast; everything else is the production code path.
"""
import pickle

import pytest
import torch

import ablation_config as ac
from episodes import episode_cache as ec
from synthetic import make_synthetic_tensor_dict
from test_models_functional import _build


@pytest.fixture
def fold0_setup(tiny_config, tmp_path, monkeypatch):
    f = ac.kfold_split.get_fold(0)
    pids = f["test"] + f["val"] + f["train"][:3]
    payload = make_synthetic_tensor_dict(n_pids=len(pids), n_classes=10, n_trials=10, seed=0)
    old = sorted(payload["data"].keys())
    payload["data"] = {new: payload["data"][o] for o, new in zip(old, pids)}
    path = tmp_path / "td.pkl"
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)

    cfg = dict(tiny_config, fold_idx=0, train_PIDs=f["train"][:3], val_PIDs=f["val"],
               test_PIDs=f["test"], test_procedure="kfold7", n_way=3, k_shot=1,
               val_episode_source="val")

    # keep it fast: 3 episodes per user per cell, no worker processes
    real_select = ec.EpisodeSet.select

    def short_select(self, n_way, k_shot, users=None):
        eps = real_select(self, n_way, k_shot, users)
        keep, seen = [], {}
        for e in eps:
            seen[e.user] = seen.get(e.user, 0) + 1
            if seen[e.user] <= 3:
                keep.append(e)
        return keep

    monkeypatch.setattr(ec.EpisodeSet, "select", short_select)
    real_dl = torch.utils.data.DataLoader
    monkeypatch.setattr(torch.utils.data, "DataLoader",
                        lambda *a, **k: real_dl(*a, **{**k, "num_workers": 0}))
    return cfg, str(path), f


def _check_result_block(res, f):
    assert set(res["per_user_acc"]) == set(f["test"])
    assert all(0.0 <= a <= 1.0 for a in res["per_user_acc"].values())
    assert res["n_episodes"] == 12                       # 4 users x 3 episodes
    assert res["episode_source"]["sha256"] == ec.load_manifest()["files"]["test_fold0.json.gz"]
    assert len(res["per_episode"]) == 12
    for r in res["per_episode"]:
        assert r["id"].startswith("test-f0-") and r["n_query"] == 3 * 9
        assert 0 <= r["n_correct"] <= r["n_query"]


def test_run_episodic_test_eval_uses_cache_and_reports_contract(fold0_setup):
    cfg, path, f = fold0_setup
    model = _build("DeepCNNLSTM", cfg)
    res = ac.run_episodic_test_eval(model, cfg, path, f["test"])
    _check_result_block(res, f)
    assert list(res["per_cell"]) == ["n3k1"]


def test_run_episodic_test_eval_second_cell_changes_query_size(fold0_setup):
    cfg, path, f = fold0_setup
    model = _build("DeepCNNLSTM", cfg)
    res = ac.run_episodic_test_eval(model, cfg, path, f["test"], cells=[(3, 1), (3, 5)])
    assert set(res["per_cell"]) == {"n3k1", "n3k5"}
    q = {(r["k_shot"]): r["n_query"] for r in res["per_episode"]}
    assert q == {1: 3 * 9, 5: 3 * 5}


def test_episodic_eval_rejects_head_size_mismatch_and_wrong_users(fold0_setup):
    cfg, path, f = fold0_setup
    model = _build("DeepCNNLSTM", cfg)
    with pytest.raises(AssertionError, match="trained head size"):
        ac.run_episodic_test_eval(model, cfg, path, f["test"], cells=[(10, 1)])
    with pytest.raises(AssertionError, match="cached test users"):
        ac.run_episodic_test_eval(model, cfg, path, f["test"][:3])
    leaky = dict(cfg, train_PIDs=cfg["train_PIDs"] + [f["test"][0]])
    with pytest.raises(AssertionError, match="also in train/val"):
        ac.run_episodic_test_eval(model, leaky, path, f["test"])


def test_run_supervised_test_eval_uses_cache(fold0_setup):
    cfg, path, f = fold0_setup
    cfg = dict(cfg, meta_learning=False, ft_steps=2, ft_lr=1e-2, ft_optimizer="sgd",
               ft_weight_decay=0.0, ft_label_smooth=0.0)
    model = _build("DeepCNNLSTM", cfg)
    res = ac.run_supervised_test_eval(model, cfg, path, f["test"], ft_mode="head_only")
    _check_result_block(res, f)
    assert res["ft_mode"] == "head_only"


def test_cached_val_loader_in_get_maml_dataloaders(fold0_setup):
    from MAML.maml_data_pipeline import get_maml_dataloaders
    cfg, path, f = fold0_setup
    cfg = dict(cfg, num_workers=0, seed=1)
    train_dl, val_dl = get_maml_dataloaders(cfg, tensor_dict_path=path)
    users = set()
    for batch in val_dl:
        assert batch["episode_id"].startswith("val-f0-")
        users.add(batch["user_id"])
    assert users == set(f["val"])
    bad = dict(cfg, val_PIDs=f["val"][:2])
    with pytest.raises(AssertionError, match="episode-file users"):
        get_maml_dataloaders(bad, tensor_dict_path=path)
