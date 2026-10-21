"""Loads system/fixed_user_splits/kfold_split.py (stdlib-only, not on sys.path) for tests."""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "system" / "fixed_user_splits" / "kfold_split.py"
_spec = importlib.util.spec_from_file_location("kfold_split", _PATH)
ks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ks)
