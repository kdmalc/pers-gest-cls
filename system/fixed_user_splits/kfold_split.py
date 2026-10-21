"""
Loader for kfold28_hpo4_split.json (28/4 outer split, 7 folds of 20/4/4).

Stdlib only, so it can be imported without torch. All eval scripts should get
their train/val/test PIDs from here rather than hard-coding PID lists.
"""
import json
from pathlib import Path

SPLIT_PATH = Path(__file__).resolve().parent / "kfold28_hpo4_split.json"
N_FOLDS = 7
PROTOCOL_NAME = "kfold28_hpo4_7fold"


def load_split(path=SPLIT_PATH):
    return json.loads(Path(path).read_text())


def get_fold(fold_idx, path=SPLIT_PATH):
    """Return dict(train, val, test) for fold_idx in [0, 7)."""
    folds = load_split(path)["folds"]
    if not 0 <= fold_idx < len(folds):
        raise IndexError(f"fold_idx {fold_idx} out of range [0, {len(folds)})")
    f = folds[fold_idx]
    return {"train": list(f["train"]), "val": list(f["val"]), "test": list(f["test"])}


def assert_fold_disjoint(train, val, test, dev):
    """train/val/test of one fold must be pairwise disjoint and exclude every dev user."""
    sets = {"train": set(train), "val": set(val), "test": set(test)}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), f"{a}/{b} overlap: {sorted(sets[a] & sets[b])}"
        assert not (sets[a] & set(dev)), f"{a} contains dev users: {sorted(sets[a] & set(dev))}"


def hpo_val_pids(path=SPLIT_PATH):
    """The 4 users used only for HPO validation; never in any fold's test set."""
    return list(load_split(path)["hpo_val"])


def all_test_pids(path=SPLIT_PATH):
    """All 28 evaluated users (each is a test user in exactly one fold)."""
    return [p for f in load_split(path)["folds"] for p in f["test"]]
