"""
Shared episode cache (docs/specs/SPEC_episode_cache.md). Stdlib only.

Episodes are generated ONCE from a seed, with no access to data, written to disk, hashed, and
read by every method. An episode stores ids only:

    user, n_way, k_shot, classes (label order = label permutation),
    support[i] = rep indices (0-based tensor rows) for classes[i]

Query reps are derived: every rep of classes[i] not in support[i].
No fallbacks: every inconsistency raises.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
PROTOCOL = "kfold7"
BASE_SEED = 20261021
N_REPS = 10
N_CLASSES = 10
N_WAYS = (3, 5, 10)
K_SHOTS = (1, 3, 5)
CELLS = tuple((n, k) for n in N_WAYS for k in K_SHOTS)
# episodes per user per cell
N_PER_CELL = {"test": 200, "val": 50, "dev": 100}

EPISODE_DIR = Path(__file__).resolve().parent / "kfold7"
MANIFEST_NAME = "MANIFEST.json"

_ks_path = Path(__file__).resolve().parents[1] / "fixed_user_splits" / "kfold_split.py"
_spec = importlib.util.spec_from_file_location("kfold_split", _ks_path)
kfold_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kfold_split)


# ─────────────────────────────────────────────────────────────────────────────
# Hashing / io
# ─────────────────────────────────────────────────────────────────────────────
def canonical_sha256(obj) -> str:
    """sha256 of the canonical JSON of obj (gzip bytes are NOT hashed: not deterministic)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write(obj: dict, path) -> str:
    """Write obj as gzip JSON (mtime=0 so bytes are stable too). Returns the canonical hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(blob)
    return canonical_sha256(obj)


def _read_obj(path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
def file_name(split: str, fold) -> str:
    if split == "dev":
        if fold is not None:
            raise ValueError("dev split has no fold")
        return "dev.json.gz"
    if split not in ("test", "val"):
        raise ValueError(f"unknown split {split!r}")
    if fold is None:
        raise ValueError(f"{split} split needs a fold")
    return f"{split}_fold{fold}.json.gz"


def _stream_seed(base_seed: int, split: str, fold, user: str, n: int, k: int) -> int:
    key = f"{base_seed}|{split}|{fold}|{user}|n{n}k{k}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def users_for(split: str, fold) -> list:
    if split == "dev":
        return sorted(kfold_split.hpo_val_pids())
    return sorted(kfold_split.get_fold(fold)[split])


def generate(split: str, fold, base_seed: int = BASE_SEED, cells=CELLS,
             n_per_cell: int = None, users=None) -> dict:
    """Build one episode-file object. Depends only on its arguments (no data, no globals)."""
    n_per_cell = N_PER_CELL[split] if n_per_cell is None else n_per_cell
    users = users_for(split, fold) if users is None else sorted(users)
    fold_tag = "x" if split == "dev" else f"f{fold}"

    episodes = []
    for user in users:
        for n, k in cells:
            rng = random.Random(_stream_seed(base_seed, split, fold, user, n, k))
            for i in range(n_per_cell):
                classes = rng.sample(range(N_CLASSES), n)
                support = [sorted(rng.sample(range(N_REPS), k)) for _ in classes]
                episodes.append({
                    "id": f"{split}-{fold_tag}-{user}-n{n}k{k}-{i:04d}",
                    "user": user, "n_way": n, "k_shot": k,
                    "classes": classes, "support": support,
                })
    obj = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "split": split,
        "fold": fold, "base_seed": base_seed, "n_reps": N_REPS, "n_classes": N_CLASSES,
        "cells": [list(c) for c in cells], "users": list(users), "episodes": episodes,
    }
    validate(obj)
    return obj


def validate(obj: dict) -> None:
    """Structural asserts (generation and load). Raises AssertionError on any violation."""
    assert obj["schema_version"] == SCHEMA_VERSION, obj["schema_version"]
    assert obj["protocol"] == PROTOCOL, obj["protocol"]
    split, fold = obj["split"], obj["fold"]
    assert split in ("test", "val", "dev"), split
    assert (fold is None) == (split == "dev"), (split, fold)
    ids = [e["id"] for e in obj["episodes"]]
    assert len(ids) == len(set(ids)), "episode ids are not unique"
    users = set(obj["users"])
    cells = {tuple(c) for c in obj["cells"]}
    for e in obj["episodes"]:
        assert e["user"] in users, e["id"]
        assert (e["n_way"], e["k_shot"]) in cells, e["id"]
        assert len(e["classes"]) == e["n_way"] == len(set(e["classes"])), e["id"]
        assert all(0 <= c < obj["n_classes"] for c in e["classes"]), e["id"]
        assert len(e["support"]) == e["n_way"], e["id"]
        for reps in e["support"]:
            assert len(reps) == e["k_shot"] == len(set(reps)), e["id"]
            assert all(0 <= r < obj["n_reps"] for r in reps), e["id"]
    # users must come from the right place and not leak across the split structure
    if split == "dev":
        assert users == set(kfold_split.hpo_val_pids()), "dev users must be the HPO-val users"
        all_fold_users = set(kfold_split.all_test_pids())
        assert not users & all_fold_users, "dev users appear in a fold"
    else:
        f = kfold_split.get_fold(fold)
        kfold_split.assert_fold_disjoint(f["train"], f["val"], f["test"],
                                         kfold_split.hpo_val_pids())
        assert users == set(f[split]), f"{split} users do not match fold {fold}"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Episode:
    id: str
    user: str
    n_way: int
    k_shot: int
    classes: tuple
    support: tuple  # support[i] = tuple of rep indices for classes[i]
    n_reps: int

    def query(self, i: int) -> tuple:
        sup = set(self.support[i])
        return tuple(r for r in range(self.n_reps) if r not in sup)

    @property
    def n_query_per_class(self) -> int:
        return self.n_reps - self.k_shot


class EpisodeSet:
    def __init__(self, obj: dict, sha256: str, path):
        self.obj, self.sha256, self.path = obj, sha256, str(path)
        self.split, self.fold = obj["split"], obj["fold"]
        self.users = tuple(obj["users"])
        self.cells = tuple(tuple(c) for c in obj["cells"])
        self.episodes = tuple(
            Episode(e["id"], e["user"], e["n_way"], e["k_shot"], tuple(e["classes"]),
                    tuple(tuple(s) for s in e["support"]), obj["n_reps"])
            for e in obj["episodes"])

    def select(self, n_way: int, k_shot: int, users=None, max_per_user=None) -> list:
        if (n_way, k_shot) not in self.cells:
            raise KeyError(f"cell (n={n_way}, k={k_shot}) not in this file: {self.cells}")
        if users is not None:
            unknown = set(users) - set(self.users)
            if unknown:
                raise KeyError(f"users not in this episode file: {sorted(unknown)}")
        out = [e for e in self.episodes if e.n_way == n_way and e.k_shot == k_shot
               and (users is None or e.user in users)]
        if not out:
            raise ValueError(f"no episodes for cell (n={n_way}, k={k_shot}), users={users}")
        if max_per_user is not None:
            # first m episodes per user in file order (a deterministic prefix of each user's stream)
            if max_per_user < 1:
                raise ValueError("max_per_user must be >= 1")
            seen, kept = defaultdict(int), []
            for e in out:
                seen[e.user] += 1
                if seen[e.user] <= max_per_user:
                    kept.append(e)
            out = kept
        return out

    def source_block(self) -> dict:
        """The `episode_source` entry every results JSON must carry."""
        try:
            rel = str(Path(self.path).resolve().relative_to(Path(__file__).resolve().parents[2]))
        except ValueError:
            rel = self.path
        return {"file": rel.replace("\\", "/"), "sha256": self.sha256}


def load_manifest(directory=EPISODE_DIR) -> dict:
    p = Path(directory) / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run system/episodes/make_episodes.py")
    return json.loads(p.read_text())


def load(split: str, fold=None, directory=EPISODE_DIR) -> EpisodeSet:
    """Load one episode file, verify its hash against the manifest and its split/fold."""
    directory = Path(directory)
    name = file_name(split, fold)
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run system/episodes/make_episodes.py")
    obj = _read_obj(path)
    sha = canonical_sha256(obj)
    manifest = load_manifest(directory)
    if name not in manifest["files"]:
        raise KeyError(f"{name} not listed in {MANIFEST_NAME}")
    if manifest["files"][name] != sha:
        raise ValueError(f"{name}: hash {sha} != manifest {manifest['files'][name]}")
    assert obj["split"] == split and obj["fold"] == fold, (obj["split"], obj["fold"])
    validate(obj)
    return EpisodeSet(obj, sha, path)


def materialize_check(ep: Episode, tensor_dict: dict) -> None:
    """Every (user, class) touched has exactly ep.n_reps reps (no silent data assumptions)."""
    if ep.user not in tensor_dict:
        raise KeyError(f"user {ep.user} not in tensor_dict")
    for c in ep.classes:
        if c not in tensor_dict[ep.user]:
            raise KeyError(f"user {ep.user} has no class {c}")
        n = tensor_dict[ep.user][c]["emg"].shape[0]
        if n != ep.n_reps:
            raise ValueError(f"{ep.user} class {c} has {n} reps, expected {ep.n_reps}")
