"""
tests/test_episodes.py
======================
Episode-sampler invariants. All data-free.

These encode the manual verifications recorded in
``rebuttal/REBUTTAL_CODE_FINDINGS.md`` §2 and §7 -- which were done in a scratch
environment that was never committed -- so they become regressions rather than
folklore. See LIMITATIONS.md B2, B3, B7, B12.
"""

from __future__ import annotations

import pytest
import torch

from MAML.maml_data_pipeline import MetaGestureDataset, reorient_tensor_dict
from synthetic import (
    make_synthetic_tensor_dict,
    hash_tensor_dict,
    IMU_N_SENSORS,
    IMU_AXES_PER_SENSOR,
)


def _eval_dataset(td, cfg, **kw):
    """Eval-mode dataset with pre-computed episodes (the reported-number path)."""
    params = dict(
        target_pids=cfg["train_PIDs"],
        target_gesture_classes=cfg["maml_gesture_classes"],
        target_trial_reps=cfg["target_trial_reps"],
        n_way=cfg["n_way"],
        k_shot=cfg["k_shot"],
        q_query=cfg["q_query"],
        episodes_per_epoch=cfg["episodes_per_epoch_train"],
        is_train=False,
        seed=cfg["seed"],
        num_eval_episodes=cfg["num_eval_episodes"],
        use_label_shuf_meta_aug=False,
    )
    params.update(kw)
    return MetaGestureDataset(td, **params)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture correctness (guards the guard)
# ─────────────────────────────────────────────────────────────────────────────
def test_fixture_rejects_ambiguous_shapes():
    """
    T must differ from both channel counts.

    ``reorient_tensor_dict`` decides orientation via
    ``emg.shape[-1] == config['emg_in_ch']``. A fixture with ``T == C_emg``
    matches in both orientations, so the idempotency guard is defeated and the
    tensor flips on every call -- silently, and only in tests.
    """
    with pytest.raises(ValueError, match="Ambiguous fixture shape"):
        make_synthetic_tensor_dict(T=16, C_emg=16, C_imu=72, n_pids=1, n_classes=1)

    with pytest.raises(ValueError, match="Ambiguous fixture shape"):
        make_synthetic_tensor_dict(T=72, C_emg=16, C_imu=72, n_pids=1, n_classes=1)


def test_fixture_is_on_disk_orientation(synthetic_payload):
    """The factory returns (trials, T, C), matching the real pickle."""
    pid = sorted(synthetic_payload["data"].keys())[0]
    entry = synthetic_payload["data"][pid][0]
    assert entry["emg"].shape == (10, 64, 16)
    assert entry["imu"].shape == (10, 64, 72)
    assert entry["rep_indices"] == list(range(1, 11)), "rep_indices must be 1-INDEXED"
    assert entry["gest_ID"] == 0
    assert "_reoriented" not in synthetic_payload, (
        "the real pickle has no _reoriented key; reorient_tensor_dict uses "
        ".get(..., False), so the fixture must not pre-set it"
    )


def test_imu_layout_is_sensor_major(synthetic_payload):
    """
    The fixture's 72 IMU channels are 12 sensors x 6 axes, gyro = 2 x accel.

    This is the structure the SO(3) rotation augmentation will assume. It is NOT
    yet verified against the real data -- see LIMITATIONS.md E7, which blocks
    rotation augmentation until the 72 real column names are dumped from the
    cluster pickle.
    """
    pid = sorted(synthetic_payload["data"].keys())[0]
    imu = synthetic_payload["data"][pid][0]["imu"]     # (trials, T, C)
    assert imu.shape[-1] == IMU_N_SENSORS * IMU_AXES_PER_SENSOR

    trial = imu[0].transpose(0, 1)                      # (C, T)
    for s in range(IMU_N_SENSORS):
        a0 = 6 * s
        accel = trial[a0 + 0 : a0 + 3]
        gyro = trial[a0 + 3 : a0 + 6]
        # Additive noise at 0.02 on each triad, so gyro - 2*accel has
        # sd = 0.02*sqrt(1 + 4) ~ 0.045; allow a comfortable ~7 sigma.
        assert torch.allclose(gyro, 2.0 * accel, atol=0.32), (
            f"sensor {s}: gyro/accel relation broken in the fixture"
        )


# ─────────────────────────────────────────────────────────────────────────────
# reorient_tensor_dict
# ─────────────────────────────────────────────────────────────────────────────
def test_reorient_flips_then_is_idempotent(fresh_payload, tiny_config):
    pid = sorted(fresh_payload["data"].keys())[0]
    assert fresh_payload["data"][pid][0]["emg"].shape == (10, 64, 16)

    td = reorient_tensor_dict(fresh_payload, tiny_config)
    assert td[pid][0]["emg"].shape == (10, 16, 64)
    assert td[pid][0]["imu"].shape == (10, 72, 64)
    assert fresh_payload["_reoriented"] is True

    first = td[pid][0]["emg"].clone()
    td2 = reorient_tensor_dict(fresh_payload, tiny_config)
    assert torch.equal(td2[pid][0]["emg"], first), "reorient must be idempotent"
    assert td2[pid][0]["emg"].is_contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Support / query construction  (LIMITATIONS B2)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("k_shot", [1, 3, 5])
def test_support_query_disjoint(reoriented_tensor_dict, tiny_config, k_shot):
    """
    Support and query are disjoint slices of one shuffled trial list, so overlap
    is structurally impossible. Verified here by comparing the actual tensors,
    which is what ``assert_no_support_query_leakage`` does on the real path.
    """
    ds = _eval_dataset(reoriented_tensor_dict, tiny_config, k_shot=k_shot)

    for i in range(len(ds)):
        ep = ds[i]
        sup = torch.stack([s["emg"] for s in ep["support"]])
        qry = torch.stack([q["emg"] for q in ep["query"]])
        for s in range(sup.shape[0]):
            for q in range(qry.shape[0]):
                assert not torch.equal(sup[s], qry[q]), (
                    f"episode {i}: support[{s}] is identical to query[{q}]"
                )


@pytest.mark.parametrize("k_shot,expected_q", [(1, 9), (3, 7), (5, 5)])
def test_realised_q_is_ten_minus_k_not_configured_q(
    reoriented_tensor_dict, tiny_config, k_shot, expected_q
):
    """
    THE DISCLOSURE IN LIMITATIONS.md B2.

    The eval path ignores ``q_query`` (default ``q_query_eval_mode="all_remaining"``)
    and assigns every non-support rep to the query set. With 10 reps that is
    Q = 9 / 7 / 5 at K = 1 / 3 / 5, NOT the configured 9. The paper must state
    realised Q measured per episode, not taken from the config.

    ``q_query`` is set to 4 by the tiny config precisely so that a passing test
    cannot be a coincidence of the configured value.
    """
    assert tiny_config["q_query"] == 4, "this test needs q_query != any expected Q"

    ds = _eval_dataset(reoriented_tensor_dict, tiny_config, k_shot=k_shot)
    _ = [ds[i] for i in range(len(ds))]

    assert ds.episode_shape_log, "episode_shape_log should be populated at eval"
    for rec in ds.episode_shape_log:
        assert rec["q_per_class"] == pytest.approx(expected_q), (
            f"K={k_shot}: realised Q should be {expected_q}, got {rec['q_per_class']}"
        )
        assert rec["n_support"] == k_shot * rec["n_classes_realised"]


@pytest.mark.parametrize("k_shot,expected_q", [(1, 4), (3, 4), (5, 4)])
def test_q_query_eval_mode_fixed_caps_at_configured_q(
    reoriented_tensor_dict, tiny_config, k_shot, expected_q
):
    """``q_query_eval_mode="fixed"`` honours the configured Q (a non-default path)."""
    ds = _eval_dataset(
        reoriented_tensor_dict, tiny_config, k_shot=k_shot, q_query_eval_mode="fixed"
    )
    _ = [ds[i] for i in range(len(ds))]
    for rec in ds.episode_shape_log:
        assert rec["q_per_class"] == pytest.approx(expected_q)


# ─────────────────────────────────────────────────────────────────────────────
# N-way degeneracy  (LIMITATIONS B3)
# ─────────────────────────────────────────────────────────────────────────────
def test_n10_has_exactly_one_class_set_and_label_map(reoriented_tensor_dict, tiny_config):
    """
    IRREDUCIBLE: with exactly 10 gestures, N=10 admits C(10,10) = 1 class set,
    and with label-shuffle off at eval, one identity label map. Only the
    support/query rep assignment varies between episodes.

    Anyone proposing "sample more classes" for the 10-way condition has not
    counted. The fix space is the rep axis (exhaustible) and the label→head-unit
    permutation axis (samplable).
    """
    ds = _eval_dataset(reoriented_tensor_dict, tiny_config, n_way=10, num_eval_episodes=8)
    maps = {tuple(sorted(ds[i]["label_map"].items())) for i in range(len(ds))}
    assert len(maps) == 1, f"N=10 should give exactly one label map, got {len(maps)}"
    assert set(next(iter(maps))) == {(c, c) for c in range(10)}, (
        "with label-shuffle off, the N=10 map should be the identity"
    )


def test_n3_has_many_distinct_label_maps(reoriented_tensor_dict, tiny_config):
    """Contrast with N=10: at N=3 the class subsets and orderings vary freely."""
    ds = _eval_dataset(reoriented_tensor_dict, tiny_config, n_way=3, num_eval_episodes=20)
    maps = {tuple(sorted(ds[i]["label_map"].items())) for i in range(len(ds))}
    assert len(maps) > 5, f"N=3 should give many distinct label maps, got {len(maps)}"


# ─────────────────────────────────────────────────────────────────────────────
# Determinism and pairing  (LIMITATIONS B7)
# ─────────────────────────────────────────────────────────────────────────────
def test_eval_episodes_are_byte_identical_across_dataset_objects(
    reoriented_tensor_dict, tiny_config
):
    """
    THE PAIRING GUARANTEE, made a test rather than an inference.

    ``_precompute_val_episodes`` uses a private ``random.Random(seed)``, so for a
    fixed (pid order, n_way, k_shot, reps, num_episodes) the entire episode
    stream is identical across models, ablations and training seeds. Every
    cross-model comparison is therefore PAIRED at the episode level -- which the
    paper does not currently claim or exploit.
    """
    a = _eval_dataset(reoriented_tensor_dict, tiny_config)
    b = _eval_dataset(reoriented_tensor_dict, tiny_config)
    assert len(a) == len(b)

    for i in range(len(a)):
        ea, eb = a[i], b[i]
        assert ea["user_id"] == eb["user_id"]
        assert ea["label_map"] == eb["label_map"]
        for sa, sb in zip(ea["support"], eb["support"]):
            assert torch.equal(sa["emg"], sb["emg"])
            assert sa["label"] == sb["label"]
        for qa, qb in zip(ea["query"], eb["query"]):
            assert torch.equal(qa["emg"], qb["emg"])


def test_eval_episodes_depend_on_pid_order(reoriented_tensor_dict, tiny_config):
    """
    Documents the fragility behind the pairing guarantee.

    The single private RNG is threaded sequentially across users, so permuting
    ``target_pids`` changes the episode stream. This is also the mechanism behind
    the 88.46 / 87.58 / 90.68 same-config spread on the training path
    (LIMITATIONS B1). Pinning it here means a future ``canonical_pid_order``
    change has a baseline to be measured against.
    """
    fwd = tiny_config["train_PIDs"]
    rev = list(reversed(fwd))
    assert len(fwd) > 1

    a = _eval_dataset(reoriented_tensor_dict, tiny_config, target_pids=fwd)
    b = _eval_dataset(reoriented_tensor_dict, tiny_config, target_pids=rev)

    users_a = [a[i]["user_id"] for i in range(len(a))]
    users_b = [b[i]["user_id"] for i in range(len(b))]
    assert users_a != users_b, (
        "expected PID-order sensitivity; if this now passes trivially, the "
        "sampler changed and LIMITATIONS.md B1 needs updating"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Modality masking  (LIMITATIONS A1)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "mask,emg_zero,imu_zero",
    [("both", False, False), ("emg_only", False, True), ("imu_only", True, False)],
)
def test_modality_mask_zeroes_correct_block(
    reoriented_tensor_dict, tiny_config, mask, emg_zero, imu_zero
):
    """
    Ports A13's ``verify_masking`` pre-flight into a test.

    Channels are zero-filled, not removed, so input width and parameter count
    are unchanged -- at the cost that a masked unimodal run is NOT identical to a
    purpose-built unimodal model. That confound is why the plan replaces A13 with
    properly-trained unimodal models (A22).
    """
    ds = _eval_dataset(reoriented_tensor_dict, tiny_config, modality_mask=mask)
    ep = ds[0]

    for sample in list(ep["support"]) + list(ep["query"]):
        assert sample["emg"].shape == (16, 64), "masking must preserve channel count"
        assert sample["imu"].shape == (72, 64)
        assert (sample["emg"].abs().sum().item() == 0.0) is emg_zero
        assert (sample["imu"].abs().sum().item() == 0.0) is imu_zero


def test_modality_mask_rejects_unknown_value(reoriented_tensor_dict, tiny_config):
    with pytest.raises(AssertionError):
        _eval_dataset(reoriented_tensor_dict, tiny_config, modality_mask="emg")


@pytest.mark.parametrize("mask", ["both", "emg_only", "imu_only"])
def test_masking_does_not_mutate_shared_tensor_dict(
    fresh_payload, tiny_config, mask
):
    """
    THE BUG CLASS THIS WHOLE FILE EXISTS FOR.

    ``_build_episode`` hands out VIEWS into the shared ``tensor_dict``
    (``emg_all[idx]``), so an in-place transform corrupts every later episode --
    and every later *model*, since the dict is shared. ``_apply_modality_mask``
    uses ``zeros_like`` precisely to avoid this, and the augmentation pipeline
    must follow the same rule (one clone at pipeline entry).
    """
    td = reorient_tensor_dict(fresh_payload, tiny_config)
    before = hash_tensor_dict(td)

    ds = _eval_dataset(td, tiny_config, modality_mask=mask)
    _ = [ds[i] for i in range(len(ds))]

    assert hash_tensor_dict(td) == before, (
        f"modality_mask={mask} mutated the shared tensor_dict"
    )


# ─────────────────────────────────────────────────────────────────────────────
# strict_n_way  (LIMITATIONS B5 -- the dropped-class path)
# ─────────────────────────────────────────────────────────────────────────────
def test_out_of_range_rep_num_raises_loudly(fresh_payload, tiny_config):
    """
    With ``target_trial_reps`` set (the shipped default is [1..10]),
    ``_available_trial_indices_for_class`` bounds-checks after converting
    1-indexed rep numbers to 0-indexed positions and raises IndexError rather
    than silently dropping data. Pinned because it fires BEFORE the class-drop
    warning below, which is why that warning is hard to reach in practice.
    """
    td = reorient_tensor_dict(fresh_payload, tiny_config)
    pid = tiny_config["train_PIDs"][0]
    g = sorted(td[pid].keys())[0]
    td[pid][g]["emg"] = td[pid][g]["emg"][:1]
    td[pid][g]["imu"] = td[pid][g]["imu"][:1]
    td[pid][g]["rep_indices"] = [1]

    # Raised eagerly, during construction -- the dataset validates rep numbers
    # while pre-computing eval episodes rather than deferring to __getitem__.
    with pytest.raises(IndexError, match="out of range"):
        ds = _eval_dataset(td, tiny_config, target_pids=[pid], n_way=10, k_shot=3)
        _ = ds[0]


def test_class_drop_warns_by_default_and_raises_when_strict(fresh_payload, tiny_config):
    """
    A class with fewer than k_shot+1 usable trials is DROPPED, producing an
    episode with fewer than n_way classes and a non-contiguous label map --
    invisible downstream. Default is a RuntimeWarning; ``strict_n_way=True``
    raises.

    Reaching this branch requires ``target_trial_reps=None``; with the shipped
    [1..10] the bounds check above fires first. A dropped class is also what
    makes the model-selection and reporting estimators disagree
    (LIMITATIONS.md B5).
    """
    td = reorient_tensor_dict(fresh_payload, tiny_config)
    pid = tiny_config["train_PIDs"][0]

    # Starve 9 of 10 classes so an episode cannot fill n_way=10.
    for g in sorted(td[pid].keys())[:9]:
        td[pid][g]["emg"] = td[pid][g]["emg"][:1]
        td[pid][g]["imu"] = td[pid][g]["imu"][:1]
        td[pid][g]["rep_indices"] = [1]

    kw = dict(
        target_pids=[pid], n_way=10, k_shot=3,
        num_eval_episodes=2, target_trial_reps=None,
    )

    with pytest.warns(RuntimeWarning, match="is being dropped"):
        ds = _eval_dataset(td, tiny_config, **kw)
        ep = ds[0]
    # The surviving class count is what the caller silently gets.
    assert len(ep["label_map"]) == 10, "class set is chosen before the drop"
    assert len({s["label"] for s in ep["support"]}) == 1, (
        "only the one un-starved class should survive into the episode"
    )

    with pytest.raises(ValueError, match="is being dropped"):
        ds = _eval_dataset(td, tiny_config, strict_n_way=True, **kw)
        _ = ds[0]
