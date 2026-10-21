"""
End-to-end guards for the kfold7 evaluation path (co-advisor audit, 2026-10-21):

  * a fold's train/val/test/dev users are pairwise disjoint,
  * every episode's support and query come from DISJOINT repetitions, checked via the rep ids
    that travel with the samples, on the same adapt-and-eval function reported numbers use,
  * labels are randomly permuted per episode at eval (no flag exists),
  * missing users raise instead of being silently dropped.

Synthetic data is re-keyed to the REAL PIDs of a real fold so the protocol, not just the code,
is exercised. (Identical-episode-file-hash checks across methods wait for the shared episode
cache.)
"""
import copy

import pytest
import torch

import kfold_split_loader  # noqa: F401  (tests/ helper below registers the module)
from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate, reorient_tensor_dict
from MAML.mamlpp import assert_support_query_rep_disjoint, mamlpp_adapt_and_eval
from synthetic import make_synthetic_tensor_dict
from test_models_functional import _build

ks = kfold_split_loader.ks


def _fold_payload(fold_idx):
    """Synthetic tensors keyed by the real train/val/test PIDs of one fold (first 3 of train)."""
    f = ks.get_fold(fold_idx)
    pids = f["test"] + f["val"][:1] + f["train"][:2]
    payload = make_synthetic_tensor_dict(n_pids=len(pids), n_classes=10, n_trials=10, seed=0)
    old = sorted(payload["data"].keys())
    payload["data"] = {new: payload["data"][o] for o, new in zip(old, pids)}
    return payload, f, pids


def test_fold_users_disjoint_and_exclude_dev():
    for k in range(ks.N_FOLDS):
        f = ks.get_fold(k)
        ks.assert_fold_disjoint(f["train"], f["val"], f["test"], ks.hpo_val_pids())


def test_assert_fold_disjoint_catches_dev_leak_and_overlap():
    f = ks.get_fold(0)
    dev = ks.hpo_val_pids()
    with pytest.raises(AssertionError):
        ks.assert_fold_disjoint(f["train"] + [dev[0]], f["val"], f["test"], dev)
    with pytest.raises(AssertionError):
        ks.assert_fold_disjoint(f["train"], f["val"], f["test"] + [f["train"][0]], dev)


@pytest.mark.parametrize("k_shot", [1, 3, 5])
def test_real_fold_episodes_have_disjoint_reps_and_permuted_labels(tiny_config, k_shot):
    payload, f, pids = _fold_payload(0)
    cfg = dict(tiny_config, k_shot=k_shot, n_way=10)
    td = reorient_tensor_dict(payload, cfg)
    ds = MetaGestureDataset(
        td, target_pids=f["test"], target_gesture_classes=cfg["maml_gesture_classes"],
        target_trial_reps=cfg["target_trial_reps"], n_way=10, k_shot=k_shot,
        q_query=cfg["q_query"], is_train=False, seed=7, num_eval_episodes=6,
    )
    maps = set()
    for i in range(len(ds)):
        ep = maml_mm_collate([ds[i]])
        assert ep["user_id"] in f["test"]
        assert_support_query_rep_disjoint(ep["support"], ep["query"])
        assert ep["support"]["rep_idx"].numel() == 10 * k_shot
        assert ep["query"]["rep_idx"].numel() == 10 * (10 - k_shot)
        maps.add(tuple(sorted(ds[i]["label_map"].items())))
    assert len(maps) > 1, "labels must be permuted per episode at eval"


def test_adapt_and_eval_refuses_overlapping_reps(tiny_config):
    payload, f, pids = _fold_payload(0)
    td = reorient_tensor_dict(payload, tiny_config)
    ds = MetaGestureDataset(
        td, target_pids=f["test"][:1], target_gesture_classes=tiny_config["maml_gesture_classes"],
        target_trial_reps=tiny_config["target_trial_reps"], n_way=tiny_config["n_way"],
        k_shot=1, q_query=tiny_config["q_query"], is_train=False, seed=1, num_eval_episodes=1,
    )
    ep = maml_mm_collate([ds[0]])
    model = _build("DeepCNNLSTM", tiny_config)

    ok = mamlpp_adapt_and_eval(model, tiny_config, ep["support"], ep["query"])
    assert 0.0 <= ok["acc"] <= 1.0

    bad_query = copy.deepcopy(ep["query"])
    bad_query["rep_idx"] = ep["support"]["rep_idx"].repeat(
        (bad_query["rep_idx"].numel() // ep["support"]["rep_idx"].numel()) + 1
    )[: bad_query["rep_idx"].numel()]
    bad_query["global_labels"] = ep["support"]["global_labels"].repeat(
        (bad_query["global_labels"].numel() // ep["support"]["global_labels"].numel()) + 1
    )[: bad_query["global_labels"].numel()]
    with pytest.raises(AssertionError, match="share repetitions"):
        mamlpp_adapt_and_eval(model, tiny_config, ep["support"], bad_query)


def test_missing_user_raises_instead_of_being_dropped(tiny_config):
    payload, f, pids = _fold_payload(0)
    td = reorient_tensor_dict(payload, tiny_config)
    with pytest.raises(KeyError, match="not in tensor_dict"):
        MetaGestureDataset(
            td, target_pids=f["test"] + ["P999"],
            target_gesture_classes=tiny_config["maml_gesture_classes"],
            target_trial_reps=tiny_config["target_trial_reps"], n_way=3, k_shot=1,
            q_query=tiny_config["q_query"], is_train=False, seed=1, num_eval_episodes=1,
        )
