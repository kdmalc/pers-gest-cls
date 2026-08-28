"""
Regenerate tests/golden_base_config_keys.json.

Deliberately a separate script, never a fixture: the golden baseline is the
mechanical guard on "published numbers reproduce", so updating it must be an
explicit act whose diff a human reads.

    python tests/regen_golden.py
    git diff tests/golden_base_config_keys.json     # READ THIS
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("CODE_DIR", str(REPO))
os.environ.setdefault("DATA_DIR", str(REPO / "dataset"))
os.environ.setdefault("RUN_DIR", "/tmp/pgc-regen-golden")
for rel in ("", "system", "system/MAML", "system/MOE", "system/pretraining",
            "system/NOTS/paper/ablations",
            "system/NOTS/paper/ablations/test_eval_files", "tests"):
    sys.path.insert(0, str(REPO / rel) if rel else str(REPO))

from ablation_config import make_base_config          # noqa: E402
from test_config_defaults import _kind                # noqa: E402

cfg = make_base_config("M0")
types = {k: _kind(v) for k, v in sorted(cfg.items())}
out = REPO / "tests" / "golden_base_config_keys.json"
out.write_text(json.dumps({"n_keys": len(types), "types": types}, indent=1) + "\n")
print(f"wrote {out} ({len(types)} keys)")
print("Now run: git diff tests/golden_base_config_keys.json")
