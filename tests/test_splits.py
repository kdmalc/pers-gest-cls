"""
Invariants of the 28/4 + 7-fold (20/4/4) split. Stdlib only; runnable without
torch via:  pytest tests/test_splits.py --noconftest
"""
import importlib.util
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "system" / "fixed_user_splits"
_spec = importlib.util.spec_from_file_location("kfold_split", _DIR / "kfold_split.py")
ks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ks)


def test_fold_sizes_and_disjointness():
    for k in range(ks.N_FOLDS):
        f = ks.get_fold(k)
        assert (len(f["train"]), len(f["val"]), len(f["test"])) == (20, 4, 4)
        s = [set(f["train"]), set(f["val"]), set(f["test"])]
        assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2])
        assert not (set(f["train"] + f["val"] + f["test"]) & set(ks.hpo_val_pids()))


def test_every_user_tested_once_and_hpo_users_never_tested():
    tests = ks.all_test_pids()
    assert len(tests) == 28 and len(set(tests)) == 28
    assert not set(tests) & set(ks.hpo_val_pids())
    assert len(ks.hpo_val_pids()) == 4
    groups = ks.load_split()["groups"]
    assert set(tests) | set(ks.hpo_val_pids()) == {p for g in groups.values() for p in g}


def test_val_is_other_folds_test_block():
    folds = ks.load_split()["folds"]
    for k, f in enumerate(folds):
        assert f["val"] == folds[(k + 1) % ks.N_FOLDS]["test"]


def test_stratification_balanced():
    d = ks.load_split()
    g_of = {p: g for g, ps in d["groups"].items() for p in ps}
    for f in d["folds"]:
        counts = {}
        for p in f["test"]:
            counts[g_of[p]] = counts.get(g_of[p], 0) + 1
        # largest stratum (Disabled Non-Kapanji, 14 of 28) must give exactly 2 per fold
        assert counts.get("Disabled Non-Kapanji", 0) == 2
