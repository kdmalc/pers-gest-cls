"""
tests/test_config_defaults.py
=============================
The mechanical guarantee behind "every published number stays reproducible".

``ablation_config.make_base_config()`` IS the M0 config -- its values are Trial
89 of ``ablation_M0_1s3w_hpo_v1``, hardcoded with an audit trail. Every new
feature in the rebuild lands as a default-off key. Two things must therefore be
impossible:

  1. a new key silently defaulting to ON, which would move a published number;
  2. ``tests/tiny_config.py`` drifting away from the real config, which would
     make every other test prove nothing.

The golden file is regenerated deliberately, never automatically::

    python tests/regen_golden.py     # then read the diff before committing

Also pinned here: the documented KeyError trap. A13, A15 and portB each lost
cluster jobs to ``KeyError: 'seed'`` AFTER their pre-flight checks printed PASS,
because a key was read but never declared.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).with_name("golden_base_config_keys.json")

# Keys added by the rebuild that MUST default to a no-op. Each entry is
# (key, required_default). Grows as the plan lands; a key here that is absent
# from make_base_config() is reported as "not wired yet" rather than failing, so
# this list can be written ahead of the implementation.
DEFAULT_OFF_KEYS: list[tuple[str, object]] = [
    # Fusion / normalisation (plan 0.5)
    ("fusion_mode", "early_concat"),
    ("input_norm_mode", "none"),
    ("MOE_expert_groups", None),
    ("MOE_aux_coeff_group", 0.0),
    # Prototype head (plan 0.4)
    ("head_type", "mlp"),
    ("proto_head_init", False),
    ("proto_head_detach_init", False),
    ("proto_head_residual", False),
    # Modality dropout + aux heads (plan 0.6)
    ("modality_dropout_p", 0.0),
    ("modality_aux_coeff", 0.0),
    # Augmentation (plan 0.7)
    ("maml_augment", False),
    ("augment_support_at_eval", False),
    # Routing specialisation (plan 0.9)
    ("MOE_routing_mi_coeff", 0.0),
    ("MOE_entropy_coeff", 0.0),
    ("MOE_expert_dropout", 0.0),
    ("MOE_gate_temperature_learnable", False),
    ("MOE_gate_temp_anneal_to", None),
    ("MOE_task_routing_mode", "none"),
    # Hybrid nonparametric branch (plan 0.10)
    ("hybrid_np_enable", False),
    # Protocol / determinism (plan 0.2, 1c)
    ("canonical_pid_order", False),
    ("episodic_train_rng", "global"),
    ("eval_episode_design", "random"),
    ("eval_label_perm_mode", "identity"),
    ("model_select_metric", "micro_query"),
]


def _kind(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, dict):
        return "dict"
    if v is None:
        return "none"
    return type(v).__name__


@pytest.fixture(scope="module")
def base_config() -> dict:
    from ablation_config import make_base_config

    return make_base_config("M0")


def test_golden_file_exists():
    assert GOLDEN.exists(), (
        f"{GOLDEN.name} is missing. Regenerate with `python tests/regen_golden.py` "
        f"and read the diff before committing."
    )


def test_base_config_keys_match_golden(base_config):
    """
    No key added, removed, or type-changed without a deliberate golden update.

    This is the single test that makes "published numbers reproduce" mechanical
    rather than aspirational.
    """
    golden = json.loads(GOLDEN.read_text())["types"]
    actual = {k: _kind(v) for k, v in base_config.items()}

    added = sorted(set(actual) - set(golden))
    removed = sorted(set(golden) - set(actual))
    changed = sorted(
        f"{k}: {golden[k]} -> {actual[k]}"
        for k in set(golden) & set(actual)
        if golden[k] != actual[k]
    )

    assert not (added or removed or changed), (
        "make_base_config() drifted from the golden baseline.\n"
        f"  ADDED   : {added}\n"
        f"  REMOVED : {removed}\n"
        f"  RETYPED : {changed}\n"
        "If intended, run `python tests/regen_golden.py`, read the diff, and "
        "confirm every added key is default-off (see test_new_keys_are_default_off)."
    )


def test_new_keys_are_default_off(base_config):
    """
    Every rebuild key present in the config must carry its no-op default.

    Keys not yet wired are reported, not failed, so this list can be written
    ahead of the implementation and turns green as the plan lands.
    """
    wrong = []
    not_wired = []
    for key, required in DEFAULT_OFF_KEYS:
        if key not in base_config:
            not_wired.append(key)
            continue
        if base_config[key] != required:
            wrong.append(f"{key}={base_config[key]!r}, must default to {required!r}")

    assert not wrong, (
        "these keys are wired but do NOT default to a no-op, so the default "
        "path has changed and published numbers may have moved:\n  "
        + "\n  ".join(wrong)
    )
    if not_wired:
        print(f"\n[not yet wired, {len(not_wired)}] " + ", ".join(sorted(not_wired)))


def test_tiny_config_is_subset_of_base(base_config, tiny_config):
    """
    ``tests/tiny_config.py`` must stay a type-compatible subset of the real
    config, or every test built on it proves nothing about the real pipeline.

    Exemptions are explicit and each has a reason.
    """
    exempt = {
        "ablation_id",   # tiny uses "TEST"
        "device",        # tiny forces cpu
        "seq_len",       # the alias this plan ADDS to make_base_config (P4)
    }
    # Keys whose type is genuinely a union in the real config, so a type
    # mismatch against one branch is not drift.
    #   use_maml_msl: "hybrid" | False  (str | bool)
    union_typed = {"use_maml_msl"}

    missing, mistyped = [], []
    for k, v in tiny_config.items():
        if k in exempt:
            continue
        if k not in base_config:
            missing.append(k)
        elif _kind(v) != _kind(base_config[k]) and k not in union_typed:
            # int/float are interchangeable for numeric hyperparameters.
            if {_kind(v), _kind(base_config[k])} != {"int", "float"}:
                mistyped.append(f"{k}: tiny={_kind(v)} base={_kind(base_config[k])}")

    assert not missing, (
        "tiny_config declares keys that make_base_config() does not -- either a "
        f"typo or a rename that needs propagating: {sorted(missing)}"
    )
    assert not mistyped, f"type drift between tiny and base config: {mistyped}"


def test_tiny_config_covers_every_key_the_sampler_reads(tiny_config):
    """
    ``get_maml_dataloaders`` reads these with ``config[...]`` (not ``.get``), so a
    missing one is a KeyError at job start -- the exact failure that cost A13,
    A15 and portB cluster jobs after their pre-flight printed PASS.
    """
    required = [
        "train_PIDs", "val_PIDs", "maml_gesture_classes", "target_trial_reps",
        "n_way", "k_shot", "q_query", "episodes_per_epoch_train",
        "num_eval_episodes", "seed", "num_workers", "use_label_shuf_meta_aug",
        "debug_one_episode", "debug_five_episodes", "debug_one_user_only",
    ]
    missing = [k for k in required if k not in tiny_config]
    assert not missing, f"tiny_config is missing sampler keys: {missing}"


def test_base_config_still_has_the_published_m0_values(base_config):
    """
    Spot-check the load-bearing Trial 89 values. If any of these move, the
    published M0 number is no longer reproducible from this tree and every
    comparison in the paper is against a different model.
    """
    expected = {
        "num_experts": 22,
        "MOE_top_k": 9,
        "maml_inner_steps": 10,
        "maml_inner_steps_eval": 10,
        "n_way": 3,
        "k_shot": 1,
        "q_query": 9,
        "cnn_base_filters": 64,
        "lstm_hidden": 64,
        "groupnorm_num_groups": 8,
        "num_epochs": 23,
        "episodes_per_epoch_train": 500,
        "meta_batchsize": 24,
        "seed": 42,
    }
    wrong = {
        k: (base_config.get(k), v) for k, v in expected.items()
        if base_config.get(k) != v
    }
    assert not wrong, f"published M0 hyperparameters changed (got, expected): {wrong}"
