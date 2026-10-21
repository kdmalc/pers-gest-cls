"""
Shared episode cache (docs/SPEC_episode_cache.md): determinism, structure, independence,
label permutation, tamper detection, and the torch / numpy adapters on real fold PIDs.
"""
import copy
import gzip
import json
import shutil

import pytest

from episodes import episode_cache as ec

TARGETS = [("dev", None)] + [(s, k) for k in range(7) for s in ("test", "val")]


# ─────────────────────────────────────────────────────────────────────────────
# Generation and files
# ─────────────────────────────────────────────────────────────────────────────
def test_committed_files_regenerate_to_manifest_hashes():
    """Catches RNG drift across Python versions: the committed hashes are the contract."""
    manifest = ec.load_manifest()["files"]
    assert set(manifest) == {ec.file_name(s, k) for s, k in TARGETS}
    for split, fold in TARGETS:
        assert ec.canonical_sha256(ec.generate(split, fold)) == manifest[ec.file_name(split, fold)]


@pytest.mark.parametrize("split,fold", TARGETS)
def test_load_verifies_and_matches_fold_users(split, fold):
    es = ec.load(split, fold)
    assert es.split == split and es.fold == fold
    assert set(es.users) == (set(ec.kfold_split.hpo_val_pids()) if split == "dev"
                             else set(ec.kfold_split.get_fold(fold)[split]))
    per_user_cell = ec.N_PER_CELL[split]
    assert len(es.episodes) == len(es.users) * len(ec.CELLS) * per_user_cell


def test_load_rejects_tampered_file(tmp_path):
    for p in ec.EPISODE_DIR.iterdir():
        shutil.copy(p, tmp_path / p.name)
    path = tmp_path / "test_fold0.json.gz"
    with gzip.open(path, "rb") as f:
        obj = json.loads(f.read())
    obj["episodes"][0]["support"][0] = [(obj["episodes"][0]["support"][0][0] + 1) % 10]
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="manifest"):
        ec.load("test", 0, directory=tmp_path)


def test_load_missing_file_raises(tmp_path):
    shutil.copy(ec.EPISODE_DIR / ec.MANIFEST_NAME, tmp_path / ec.MANIFEST_NAME)
    with pytest.raises(FileNotFoundError):
        ec.load("test", 0, directory=tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Independence of streams
# ─────────────────────────────────────────────────────────────────────────────
def test_more_episodes_per_cell_extends_without_changing_existing():
    a = ec.generate("test", 0, n_per_cell=5)["episodes"]
    b = ec.generate("test", 0, n_per_cell=10)["episodes"]
    by_id = {e["id"]: e for e in b}
    for e in a:
        assert by_id[e["id"]] == e


def test_adding_a_cell_does_not_change_other_cells():
    one = ec.generate("test", 1, cells=[(3, 1)], n_per_cell=7)["episodes"]
    full = {e["id"]: e for e in ec.generate("test", 1, n_per_cell=7)["episodes"]}
    for e in one:
        assert full[e["id"]] == e


def test_different_base_seed_changes_episodes():
    a = ec.generate("val", 2, n_per_cell=3)["episodes"]
    b = ec.generate("val", 2, base_seed=ec.BASE_SEED + 1, n_per_cell=3)["episodes"]
    assert [e["classes"] for e in a] != [e["classes"] for e in b]


# ─────────────────────────────────────────────────────────────────────────────
# Structure
# ─────────────────────────────────────────────────────────────────────────────
def test_validate_rejects_structural_violations():
    base = ec.generate("test", 0, n_per_cell=2, cells=[(3, 3)])
    ec.validate(base)
    for mutate in (
        lambda o: o["episodes"][0]["support"][0].__setitem__(0, o["episodes"][0]["support"][0][1]),
        lambda o: o["episodes"][0].__setitem__("classes", [1, 1, 2]),
        lambda o: o["episodes"][1].__setitem__("id", o["episodes"][0]["id"]),
        lambda o: o["episodes"][0]["support"][0].__setitem__(0, 10),
        lambda o: o.__setitem__("users", o["users"][:-1] + ["P999"]),
    ):
        bad = copy.deepcopy(base)
        mutate(bad)
        with pytest.raises((AssertionError, KeyError)):
            ec.validate(bad)


def test_dev_users_never_in_any_fold_file():
    dev = set(ec.kfold_split.hpo_val_pids())
    for split, fold in TARGETS:
        if split != "dev":
            assert not set(ec.load(split, fold).users) & dev


def test_every_user_is_a_test_user_in_exactly_one_file():
    seen = []
    for k in range(7):
        seen += list(ec.load("test", k).users)
    assert len(seen) == 28 and len(set(seen)) == 28


def test_query_is_the_complement_and_q_equals_10_minus_k():
    es = ec.load("test", 0)
    for ep in es.episodes[::97]:
        for i in range(ep.n_way):
            q = ep.query(i)
            assert set(q) | set(ep.support[i]) == set(range(10))
            assert not set(q) & set(ep.support[i])
            assert len(q) == 10 - ep.k_shot == ep.n_query_per_class


def test_labels_are_permuted_not_identity_and_cover_all_slots():
    es = ec.load("test", 0)
    user = es.users[0]
    eps = es.select(10, 1, users=[user])
    assert len(eps) == 200
    identity = tuple(range(10))
    n_identity = sum(e.classes == identity for e in eps)
    assert n_identity / len(eps) < 0.01
    for slot in range(10):
        assert {e.classes[slot] for e in eps} == set(range(10)), f"slot {slot} misses a class"


def test_select_unknown_cell_or_user_raises():
    es = ec.load("val", 0)
    with pytest.raises(KeyError):
        es.select(4, 1)
    with pytest.raises(KeyError):
        es.select(3, 1, users=["P999"])


# ─────────────────────────────────────────────────────────────────────────────
# Adapters on real fold PIDs (synthetic tensors re-keyed to the fold's users)
# ─────────────────────────────────────────────────────────────────────────────
def _td_for(es, tiny_config):
    from MAML.maml_data_pipeline import reorient_tensor_dict
    from synthetic import make_synthetic_tensor_dict

    payload = make_synthetic_tensor_dict(n_pids=len(es.users), n_classes=10, n_trials=10, seed=0)
    old = sorted(payload["data"].keys())
    payload["data"] = {new: payload["data"][o] for o, new in zip(old, es.users)}
    return reorient_tensor_dict(payload, tiny_config)


@pytest.mark.parametrize("n,k", [(3, 1), (10, 1), (10, 5)])
def test_torch_adapter_matches_metagesturedataset_contract(tiny_config, n, k):
    from MAML.maml_data_pipeline import maml_mm_collate
    from MAML.mamlpp import assert_support_query_rep_disjoint
    from episodes.episode_torch import CachedEpisodeDataset

    es = ec.load("test", 0)
    td = _td_for(es, tiny_config)
    eps = es.select(n, k)[:6]
    ds = CachedEpisodeDataset(td, eps, "both")
    for i, ep in enumerate(eps):
        batch = maml_mm_collate([ds[i]])
        assert batch["episode_id"] == ep.id and batch["user_id"] == ep.user
        assert batch["support"]["labels"].numel() == n * k
        assert batch["query"]["labels"].numel() == n * (10 - k)
        assert_support_query_rep_disjoint(batch["support"], batch["query"])
        # local label i <-> classes[i] (the permutation)
        for lab, g in zip(batch["support"]["labels"].tolist(), batch["support"]["global_labels"].tolist()):
            assert ep.classes[lab] == g
    assert ds.episode_shape_log[0]["q_per_class"] == 10 - k


def test_torch_adapter_runs_through_real_adapt_and_eval(tiny_config):
    from MAML.maml_data_pipeline import maml_mm_collate
    from MAML.mamlpp import mamlpp_adapt_and_eval
    from episodes.episode_torch import CachedEpisodeDataset
    from test_models_functional import _build

    es = ec.load("test", 0)
    td = _td_for(es, tiny_config)
    ds = CachedEpisodeDataset(td, es.select(3, 1)[:2], "both")
    model = _build("DeepCNNLSTM", tiny_config)
    for i in range(2):
        ep = maml_mm_collate([ds[i]])
        out = mamlpp_adapt_and_eval(model, tiny_config, ep["support"], ep["query"])
        assert 0.0 <= out["acc"] <= 1.0


def test_torch_adapter_rejects_wrong_rep_count(tiny_config):
    from episodes.episode_torch import CachedEpisodeDataset

    es = ec.load("test", 0)
    td = _td_for(es, tiny_config)
    ep = es.select(3, 1)[0]
    td[ep.user][ep.classes[0]]["emg"] = td[ep.user][ep.classes[0]]["emg"][:9]
    with pytest.raises(ValueError, match="expected 10"):
        CachedEpisodeDataset(td, [ep], "both")


def test_numpy_adapter_shapes_and_disjointness(tiny_config):
    from episodes.episode_numpy import materialize

    es = ec.load("test", 3)
    td = _td_for(es, tiny_config)
    ep = es.select(5, 3)[0]
    m = materialize(ep, td)
    assert m.emg_support.shape[0] == 5 * 3 and m.emg_query.shape[0] == 5 * 7
    assert m.imu_support.shape[0] == 15
    pairs_s = set(zip(m.cls_support.tolist(), m.rep_support.tolist()))
    pairs_q = set(zip(m.cls_query.tolist(), m.rep_query.tolist()))
    assert not pairs_s & pairs_q
    assert sorted(set(m.y_support.tolist())) == list(range(5))


def test_outcomes_to_per_episode_validates_counts():
    from episodes.episode_torch import outcomes_to_per_episode

    eps = ec.load("val", 0).select(3, 1)[:3]
    rows = outcomes_to_per_episode(eps, [27, 0, 5])
    assert rows[0] == {"id": eps[0].id, "user": eps[0].user, "n_way": 3, "k_shot": 1,
                       "n_query": 27, "n_correct": 27}
    with pytest.raises(ValueError):
        outcomes_to_per_episode(eps, [28, 0, 5])
    with pytest.raises(ValueError):
        outcomes_to_per_episode(eps, [1, 2])


def test_source_block_has_hash_and_relative_path():
    blk = ec.load("test", 0).source_block()
    assert blk["file"].endswith("system/episodes/kfold7/test_fold0.json.gz")
    assert blk["sha256"] == ec.load_manifest()["files"]["test_fold0.json.gz"]


def test_select_max_per_user_is_a_prefix_per_user():
    es = ec.load("dev", None)
    full = es.select(10, 1)
    part = es.select(10, 1, max_per_user=25)
    assert len(part) == 25 * len(es.users) and len(full) == 100 * len(es.users)
    for u in es.users:
        assert [e.id for e in part if e.user == u] == [e.id for e in full if e.user == u][:25]
    with pytest.raises(ValueError):
        es.select(10, 1, max_per_user=0)
