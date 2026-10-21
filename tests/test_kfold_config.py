"""
kfold7 wiring in ablation_config (28/4 outer split + 7 folds of 20/4/4).

Needs torch (imports ablation_config); the split JSON itself is covered torch-free in
test_splits.py.
"""
import pytest

import ablation_config as ac


def test_default_protocol_is_kfold7():
    cfg = ac.make_base_config("T")
    assert cfg["test_procedure"] == "kfold7"
    assert len(cfg["all_PIDs"]) == 28
    assert cfg["fold_idx"] == 0
    assert set(cfg["hpo_val_PIDs"]) == {"P005", "P131", "P116", "P109"}


@pytest.mark.parametrize("k", range(7))
def test_apply_kfold_fold_disjoint_and_sized(k):
    cfg = ac.apply_kfold_fold(ac.make_base_config("T"), k)
    tr, va, te = cfg["train_PIDs"], cfg["val_PIDs"], cfg["test_PIDs"]
    assert (len(tr), len(va), len(te)) == (20, 4, 4)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert not (set(tr + va + te) & set(cfg["hpo_val_PIDs"]))
    assert cfg["fold_idx"] == k and cfg["test_procedure"] == "kfold7"


def test_apply_kfold_fold_does_not_mutate_input():
    base = ac.make_base_config("T")
    before = list(base["test_PIDs"])
    ac.apply_kfold_fold(base, 3)
    assert base["test_PIDs"] == before


def test_hpo_split_uses_only_dev_users_for_val_and_no_test():
    s = ac.get_hpo_split()
    assert len(s["train"]) == 20 and len(s["val"]) == 4 and s["test"] == []
    assert not set(s["train"]) & set(s["val"])


def test_resolve_fold_ids_and_run_kfold():
    assert ac.resolve_fold_ids(None) == list(range(7))
    assert ac.resolve_fold_ids(4) == [4]
    with pytest.raises(ValueError):
        ac.resolve_fold_ids(7)
    seen = []

    def fake(fold_id, seed, config):
        seen.append((fold_id, tuple(config["test_PIDs"])))
        return {"fold_id": fold_id}

    out = ac.run_kfold(ac.make_base_config("T"), None, fake)
    assert len(out) == 7 and len({t for _, t in seen}) == 7
    assert sorted(p for _, t in seen for p in t) == sorted(ac.KFOLD_ALL_TEST_PIDS)


def test_protocol_check_accepts_kfold7():
    cfg = ac.apply_kfold_fold(ac.make_base_config("T"), 0)
    rec = ac.assert_protocol_consistent(cfg, n_folds=7)
    assert rec["consistent"] and rec["declared_test_procedure"] == "kfold7"
