"""
tests/conftest.py
=================
Path and environment setup, plus shared fixtures.

ORDER IS LOAD-BEARING. ``ablation_config.py`` does real work AT IMPORT TIME:

  * reads ``CODE_DIR`` / ``DATA_DIR`` / ``RUN_DIR`` from the environment, with
    ``CODE_DIR`` defaulting to ``"./"``;
  * calls ``RUN_DIR.mkdir(parents=True, exist_ok=True)``;
  * opens ``CODE_DIR/system/fixed_user_splits/hpo_strat_kapanji_split.json``;
  * sets ``torch.backends.cudnn.deterministic = True``.

So importing it from anywhere other than the repo root fails, and importing it
without ``RUN_DIR`` set litters the repo. Everything below the banner runs at
conftest import, i.e. before any test module is imported.

The split JSON is committed, so no synthetic split file is needed -- we point
``CODE_DIR`` at the real repo root and let the real 24/4/4 fold load. Tests then
override ``train_PIDs`` / ``val_PIDs`` / ``test_PIDs`` with synthetic PIDs.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Environment, before any project import
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("CODE_DIR", str(REPO_ROOT))
os.environ.setdefault("DATA_DIR", str(REPO_ROOT / "dataset"))
# Never let RUN_DIR default to "./" -- ablation_config mkdir()s it and the run
# scripts write result JSONs and checkpoints there.
os.environ["RUN_DIR"] = os.environ.get(
    "PGC_TEST_RUN_DIR", tempfile.mkdtemp(prefix="pgc-test-run-")
)

# Keep CPU tests from oversubscribing; also makes timings stable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# The run scripts all do this same sys.path dance (see the header of any file in
# system/NOTS/paper/ablations/test_eval_files/). `system/` is not an installable
# package, so there is no import-time alternative.
for rel in (
    "",
    "system",
    "system/MAML",
    "system/MOE",
    "system/pretraining",
    "system/nonparametric",
    "system/NOTS/paper/ablations",
    "system/NOTS/paper/ablations/test_eval_files",
):
    p = str(REPO_ROOT / rel) if rel else str(REPO_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402  (must follow the sys.path setup)
import torch  # noqa: E402

from synthetic import (  # noqa: E402
    make_synthetic_tensor_dict,
    write_synthetic_pkl,
    pid_names,
    DEFAULT_C_EMG,
    DEFAULT_C_IMU,
    DEFAULT_DEMO_DIM,
    DEFAULT_T,
)


def pytest_configure(config):
    torch.manual_seed(0)
    # Single-threaded: several tests compare tensors bitwise, and intra-op
    # parallelism can reorder float reductions.
    torch.set_num_threads(1)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
N_PIDS = 6
N_CLASSES = 10
N_TRIALS = 10


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_payload() -> dict:
    """
    Full on-disk payload, ``(trials, T, C)`` orientation, session-scoped.

    Session scope is safe ONLY for read-only use. Anything that calls
    ``reorient_tensor_dict`` or could mutate tensors must use
    ``fresh_payload`` instead -- reorientation is in-place and sets a flag on
    the payload, so it would leak across tests.
    """
    return make_synthetic_tensor_dict(
        n_pids=N_PIDS, n_classes=N_CLASSES, n_trials=N_TRIALS, seed=0
    )


@pytest.fixture
def fresh_payload() -> dict:
    """Function-scoped payload for anything that reorients or mutates."""
    return make_synthetic_tensor_dict(
        n_pids=N_PIDS, n_classes=N_CLASSES, n_trials=N_TRIALS, seed=0
    )


@pytest.fixture
def synthetic_pkl(tmp_path) -> str:
    """Path to a pickled synthetic payload, for the path-taking entry points."""
    return write_synthetic_pkl(
        tmp_path, n_pids=N_PIDS, n_classes=N_CLASSES, n_trials=N_TRIALS, seed=0
    )


@pytest.fixture
def synth_pids() -> list[str]:
    return pid_names(N_PIDS)


@pytest.fixture
def reoriented_tensor_dict(fresh_payload, tiny_config) -> dict:
    """``(trials, C, T)`` tensor_dict, i.e. what MetaGestureDataset expects."""
    from MAML.maml_data_pipeline import reorient_tensor_dict

    return reorient_tensor_dict(fresh_payload, tiny_config)


@pytest.fixture
def tiny_config(synth_pids) -> dict:
    """
    Small, fast, CPU-only config.

    Deliberately NOT ``make_base_config()``: that import has side effects and 22
    experts on CPU is slow. ``test_config_defaults.py`` asserts this dict is a
    type-compatible subset of ``make_base_config()`` so the two cannot drift.
    """
    from tiny_config import make_tiny_config

    return make_tiny_config(synth_pids)
