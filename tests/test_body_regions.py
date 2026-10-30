"""Sensor map (Kai's numbering, docs/specs/SPEC_body_regions.md) and the Gate 0 headroom arithmetic."""
import numpy as np

from arch_variants import body_regions as br
from nonparametric import region_headroom as rh


def test_regions_partition_every_channel():
    assert sorted(c for r in br.REGIONS for c in r.emg) == list(range(16))
    assert sorted(s for r in br.REGIONS for s in r.imu) == list(range(12))


def test_two_lead_pods_follow_kais_numbering():
    # 1-based EMG channels 9..16 are leads 9a 9b 10a 10b 11a 11b 12a 12b
    assert br.EMG_LEADS[8:] == ("9a", "9b", "10a", "10b", "11a", "11b", "12a", "12b")
    assert br.EMG_LEADS[:8] == tuple(str(i) for i in range(1, 9))
    by = {r.name: r for r in br.REGIONS}
    assert by["R_forearm"].emg == (8, 9) and by["R_hand"].emg == (10, 11)
    assert by["L_forearm"].emg == (12, 13) and by["L_hand"].emg == (14, 15)
    assert by["R_hand"].imu == () and by["L_hand"].imu == ()


def test_side_of_each_region_matches_its_muscles():
    for r in br.REGIONS:
        assert all(br.EMG_MUSCLES[c].startswith(r.side) for c in r.emg)


def test_tensor_indices():
    assert br.emg_tensor_indices([0], "env20") == [0]
    assert br.emg_tensor_indices([1], "fs64") == list(range(7, 14))
    assert br.imu_tensor_indices([2]) == list(range(12, 18))


def _arrays(seed=0, n=3):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, 112, 64)), r.normal(size=(n, 72, 64))


def test_channel_distances_sum_to_balanced_l1():
    ea, ia = _arrays(0, 4)
    eb, ib = _arrays(1, 3)
    d = rh.channel_distances(ea, ia, eb, ib).sum(axis=2)
    block = lambda x: x.reshape(len(x), -1) / np.sqrt(x.shape[1] * x.shape[2])
    a = np.hstack([block(ea), block(ia)])
    b = np.hstack([block(eb), block(ib)])
    assert np.allclose(d, np.abs(a[:, None] - b[None]).sum(axis=2), atol=1e-3)


def test_groups_cover_all_channels():
    for g in (rh.anatomical_groups(), rh.random_groups(np.random.default_rng(3))):
        assert sorted(c for x in g for c in x) == list(range(28))
    shape = sorted((sum(c < 16 for c in x), sum(c >= 16 for c in x)) for x in rh.anatomical_groups())
    rnd = sorted((sum(c < 16 for c in x), sum(c >= 16 for c in x))
                 for x in rh.random_groups(np.random.default_rng(3)))
    assert shape == rnd


def test_last_subset_is_every_region():
    m = rh.subset_masks(8)
    assert m.shape == (255, 8) and m[-1].all() and m.sum(axis=1).min() == 1


def test_subset_accuracy_on_informative_and_noise_regions():
    # region 0 identifies the class; region 1 is misleading
    n_ep, nq, ns = 5, 4, 4
    y_q = np.tile(np.arange(4), (n_ep, 1))
    y_s = np.tile(np.arange(4), (n_ep, 1))
    good = np.ones((n_ep, nq, ns)) - np.eye(4)
    bad = np.ones((n_ep, nq, ns)) - np.roll(np.eye(4), 1, axis=1)
    d = np.stack([good, bad], axis=-1)
    acc = rh.subset_accuracy(d, y_s, y_q, rh.subset_masks(2))
    assert np.allclose(acc[0], 1.0) and np.allclose(acc[1], 0.0)


def test_cv_oracle_scores_held_out_episodes():
    # subset 0 wins on odd episodes only, so choosing it on them scores 0 on the others
    acc = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    assert rh.cv_oracle(acc) == 0.0


def test_support_selector_keeps_the_high_spread_region():
    n_ep, nq, ns = 2, 4, 4
    y_q = np.tile(np.arange(4), (n_ep, 1))
    y_s = np.tile(np.arange(4), (n_ep, 1))
    good = np.ones((n_ep, nq, ns)) - np.eye(4)
    bad = np.ones((n_ep, nq, ns)) - np.roll(np.eye(4), 1, axis=1)
    d = np.stack([good, bad], axis=-1)
    spreads = np.array([[5.0, 1.0]] * n_ep)
    out = rh.support_selector(d, spreads, np.ones(2), y_s, y_q, 1)
    assert np.allclose(out, 1.0)
