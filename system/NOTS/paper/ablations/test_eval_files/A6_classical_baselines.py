"""
A6_classical_baselines.py
=========================
Classical / non-parametric baselines (`$B`, kNN, ProtoNet-on-raw, Riemannian, MiniRocket) evaluated
on the SAME cached test episodes as every other method (docs/SPEC_episode_cache.md).

    python A6_classical_baselines.py --method dollar_b --fold-idx 0 [--cells 10x1 3x1 ...]

--cells defaults to the primary cell 10x1; pass `all` for the full 3x3 grid. Deterministic, no
training, CPU only. Output: A6_<method>_*_results.json with the standard results contract
(episode_source hash + per_episode rows), one file per fold.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

CODE_DIR = Path(os.environ["CODE_DIR"]).resolve()
for rel in ("", "system", "system/MAML", "system/MOE", "system/pretraining",
            "system/nonparametric"):
    sys.path.insert(0, str(CODE_DIR / rel) if rel else str(CODE_DIR))

from ablation_config import (  # noqa: E402
    tensor_dict_file,
    _summarise_eval, _print_eval, apply_kfold_fold, add_fold_arg, make_base_config,
    save_results,
)
from MAML.maml_data_pipeline import reorient_tensor_dict  # noqa: E402
from episodes import episode_cache as ec  # noqa: E402
from episodes.episode_numpy import materialize  # noqa: E402
from episodes.episode_torch import outcomes_to_per_episode  # noqa: E402
from nonparametric import classical  # noqa: E402

PRIMARY = (10, 1)


def parse_cells(tokens):
    if tokens == ["all"]:
        return list(ec.CELLS)
    cells = []
    for t in tokens:
        n, k = t.lower().split("x")
        cells.append((int(n), int(k)))
    bad = [c for c in cells if c not in ec.CELLS]
    if bad:
        raise SystemExit(f"cells {bad} not in the cache grid {ec.CELLS}")
    return cells


def main():
    ap = argparse.ArgumentParser(description="A6: classical baselines on cached episodes")
    ap.add_argument("--method", required=True, choices=classical.METHODS)
    ap.add_argument("--cells", nargs="+", default=["10x1"])
    add_fold_arg(ap, required=True)
    args = ap.parse_args()
    cells = parse_cells(args.cells)

    config = apply_kfold_fold(make_base_config(f"A6_{args.method}"), args.fold_idx)
    test_pids = config["test_PIDs"]
    es = ec.load("test", args.fold_idx)
    assert set(test_pids) == set(es.users)

    with open(tensor_dict_file(config), "rb") as f:
        tensor_dict = reorient_tensor_dict(pickle.load(f), config)

    rows_by_cell = {}
    for n, k in cells:
        eps = es.select(n, k)
        correct = []
        for ep in eps:
            mep = materialize(ep, tensor_dict)
            pred = classical.predict(args.method, mep)
            correct.append(int((pred == mep.y_query).sum()))
        rows_by_cell[(n, k)] = outcomes_to_per_episode(eps, correct)
        print(f"[A6/{args.method}] fold {args.fold_idx} cell {n}x{k}: {len(eps)} episodes")

    res = _summarise_eval(es, rows_by_cell)
    _print_eval(res)
    summary = {
        "ablation_id": f"A6_{args.method}", "description": f"Classical baseline: {args.method}",
        "test_procedure": "kfold7", "fold_idxs": [args.fold_idx], "method": args.method,
        "test_results": res,
    }
    save_results(summary, config, tag=f"fold{args.fold_idx:02d}_summary")


if __name__ == "__main__":
    main()
