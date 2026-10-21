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


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter mode (legacy / frozen / tuned)
# ─────────────────────────────────────────────────────────────────────────────
def _write_hpo(tmp_path):
    import json
    p = tmp_path / "best.json"
    prov = {"study": "s", "trial": 7}
    p.write_text(json.dumps({
        "M0": {"overrides": {"learning_rate": 0.123, "num_experts": 8, "maml_inner_steps_eval": 25,
                             "maml_alpha_init_eval": 0.02, "use_MOE": False},
               "provenance": prov},
        "A2": {"overrides": {"learning_rate": 0.456, "ft_steps": 100, "ft_lr": 0.01},
               "provenance": prov}}))
    return p


def test_apply_hpo_config_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "HPO_CONFIG_PATH", _write_hpo(tmp_path))
    base = ac.make_base_config("A2")
    base["meta_learning"] = False
    base.update(ft_steps=10, ft_lr=0.005)      # A1/A2 build_config adds these before the fold is applied
    leg = ac.apply_hpo_config(dict(base), "legacy")
    assert leg["learning_rate"] == base["learning_rate"] and leg["hpo_config"] == "legacy"

    tuned = ac.apply_hpo_config(dict(base), "tuned")           # A2's own result
    assert tuned["learning_rate"] == 0.456 and tuned["ft_steps"] == 100

    frozen = ac.apply_hpo_config(dict(base), "frozen")         # M0's result, structure untouched
    assert frozen["learning_rate"] == 0.123 and frozen["num_experts"] == 8
    assert frozen["use_MOE"] == base["use_MOE"]                # structural key not overridden
    assert (frozen["ft_steps"], frozen["ft_lr"]) == (25, 0.02)  # supervised budget re-mirrored

    m0_like = dict(ac.make_base_config("A13_both"))            # no own entry -> M0's
    assert ac.apply_hpo_config(m0_like, "tuned")["learning_rate"] == 0.123


def test_apply_hpo_config_rejects_bad_inputs(tmp_path, monkeypatch):
    import json
    with pytest.raises(ValueError):
        ac.apply_hpo_config(ac.make_base_config("M0"), "nope")
    monkeypatch.setattr(ac, "HPO_CONFIG_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        ac.apply_hpo_config(ac.make_base_config("M0"), "tuned")
    p = tmp_path / "typo.json"
    p.write_text(json.dumps({"M0": {"overrides": {"learnin_rate": 1}, "provenance": {"study": "s", "trial": 1}}}))
    monkeypatch.setattr(ac, "HPO_CONFIG_PATH", p)
    with pytest.raises(KeyError, match="not a config key"):
        ac.apply_hpo_config(ac.make_base_config("M0"), "tuned")


def test_apply_kfold_fold_requires_hpo_config_env(monkeypatch):
    monkeypatch.delenv("HPO_CONFIG")
    with pytest.raises(KeyError, match="HPO_CONFIG"):
        ac.apply_kfold_fold(ac.make_base_config("T"), 0)


def test_emg_repr_switches_dict_file_and_channels(monkeypatch):
    base = ac.make_base_config("T")
    assert base["emg_repr"] == "env20" and base["emg_in_ch"] == 16
    assert ac.tensor_dict_file(base).endswith("segfilt_rts_tensor_dict.pkl")
    fs = ac.apply_emg_repr(dict(base), "fs64")
    assert fs["emg_in_ch"] == 112 and ac.tensor_dict_file(fs).endswith("segfilt_fs64_tensor_dict.pkl")
    with pytest.raises(ValueError):
        ac.apply_emg_repr(dict(base), "raw")
    monkeypatch.setenv("EMG_REPR", "fs64")
    assert ac.apply_kfold_fold(base, 0)["emg_in_ch"] == 112
