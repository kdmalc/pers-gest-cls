"""
tests/tiny_config.py
====================
A small, fast, CPU-only config for tests.

Why not ``make_base_config()``: that import has side effects (prints, mkdir,
reads the split JSON) and its M0 values -- 22 experts, 500 episodes/epoch, 10
inner steps -- make CPU tests slow. So tests use a hand-built subset.

The obvious risk is drift: a test config that no longer resembles the real one
proves nothing. ``test_config_defaults.py::test_tiny_config_is_subset_of_base``
closes that by asserting every key here exists in ``make_base_config()`` with a
compatible type, so a rename in one place fails CI rather than silently
diverging.

One trap encoded below: ``_build_cnn_block`` floors each layer's output width to
a multiple of ``groupnorm_num_groups``::

    curr_out = max(gn_groups, (curr_out // gn_groups) * gn_groups)

so ``cnn_base_filters=8`` with ``groupnorm_num_groups=8`` gives width 8, not the
doubling you expect. We use gn=4 with base=8 and assert the realised widths in
``test_fusion.py`` rather than assuming them.
"""

from __future__ import annotations

from synthetic import (
    DEFAULT_C_EMG,
    DEFAULT_C_IMU,
    DEFAULT_DEMO_DIM,
    DEFAULT_T,
)

FIXED_SEED = 42


def make_tiny_config(pids: list[str], **overrides) -> dict:
    """
    Build the tiny config.

    Args:
        pids: synthetic PID list. Split 4 train / 1 val / 1 test at n_pids=6.
        **overrides: applied last, so a test can flip one key inline.
    """
    n = len(pids)
    assert n >= 3, f"need >=3 synthetic PIDs to form a train/val/test split, got {n}"
    n_val = max(1, n // 6)
    n_test = max(1, n // 6)
    test_pids = list(pids[:n_test])
    val_pids = list(pids[n_test : n_test + n_val])
    train_pids = list(pids[n_test + n_val :])

    config: dict = {
        # ── identity ────────────────────────────────────────────────────────
        "ablation_id": "TEST",
        "model_type": "DeepCNNLSTM",
        "device": "cpu",
        "seed": FIXED_SEED,

        # ── input dims (must match the synthetic fixture exactly) ───────────
        "sequence_length": DEFAULT_T,
        "seq_len": DEFAULT_T,          # TST reads this alias; see LIMITATIONS P4
        "emg_in_ch": DEFAULT_C_EMG,
        "imu_in_ch": DEFAULT_C_IMU,
        "demo_in_dim": DEFAULT_DEMO_DIM,

        # ── modality flags ──────────────────────────────────────────────────
        "multimodal": True,
        "use_imu": True,
        "use_demographics": False,
        "use_film_x_demo": False,

        # ── task setup ──────────────────────────────────────────────────────
        "n_way": 3,
        "k_shot": 1,
        "q_query": 4,
        "pretrain_num_classes": 10,
        "maml_gesture_classes": list(range(10)),
        "target_trial_reps": list(range(1, 11)),

        # ── architecture (tiny; see the GroupNorm note above) ───────────────
        "cnn_base_filters": 8,
        "cnn_layers": 2,
        "cnn_kernel": 5,
        "lstm_hidden": 8,
        "lstm_layers": 3,
        "bidirectional": True,
        "groupnorm_num_groups": 4,
        "use_GlobalAvgPooling": True,
        "use_batch_norm": False,
        "dropout": 0.1,
        "head_type": "mlp",
        "front_end_stride": 0,

        # ── optimisation ────────────────────────────────────────────────────
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "label_smooth": 0.0,
        "gradient_clip_max_norm": 10.0,
        "optimizer": "adam",

        # ── schedule (tiny) ─────────────────────────────────────────────────
        "num_epochs": 1,
        "episodes_per_epoch_train": 4,
        "num_eval_episodes": 2,
        "meta_batchsize": 2,
        "num_workers": 0,
        "use_earlystopping": False,

        # ── MAML++ ──────────────────────────────────────────────────────────
        "meta_learning": True,
        "maml_inner_steps": 2,
        "maml_inner_steps_eval": 2,
        "maml_alpha_init": 1e-2,
        "maml_alpha_init_eval": 1e-2,
        "maml_use_lslr": True,
        "use_lslr_at_eval": False,
        "use_maml_msl": False,
        "maml_msl_num_epochs": 0,
        "maml_opt_order": "first",

        # ── MoE (tiny expert bank) ──────────────────────────────────────────
        "use_MOE": True,
        "MOE_placement": "encoder",
        "num_experts": 3,
        "MOE_top_k": 2,
        "MOE_gate_temperature": 1.0,
        "MOE_aux_coeff": 0.01,
        "MOE_ctx_out_dim": 8,
        "MOE_ctx_hidden_dim": 8,
        "MOE_dropout": 0.0,
        "MOE_expert_expand": 1.0,
        "apply_MOE_aux_loss_inner_outer": "outer",
        "MOE_use_shared_expert": False,
        "MOE_importance_coeff": 0.0,
        "MOE_routing_signal": "context_proj",

        # ── splits (synthetic PIDs) ─────────────────────────────────────────
        "train_PIDs": train_pids,
        "val_PIDs": val_pids,
        "test_PIDs": test_pids,
        "test_procedure": "hpo_test_split",
        "subject_specific_model": False,

        # ── episode sampler behaviour ───────────────────────────────────────
        "use_label_shuf_meta_aug": True,
        "modality_mask": "both",
        "q_query_eval_mode": "all_remaining",
        "strict_n_way": False,
        "augment": False,

        # ── debug switches the pipeline reads unconditionally ───────────────
        "debug_one_episode": False,
        "debug_five_episodes": False,
        "debug_one_user_only": False,
    }

    config.update(overrides)
    return config
