"""
Aggregator for results JSONs (docs/SPEC_stats.md, docs/SPEC_episode_cache.md).

Collects every block carrying `per_episode` + `episode_source`, then enforces:
  * the episode file hash equals the committed manifest hash,
  * for each (fold, cell), every method evaluated the IDENTICAL set of episode ids
    (methods listed in `subset_ok` may cover a subset of the reference ids),
  * all 28 users are present for every method (all 7 folds),
and builds per-user accuracy tables. Raises on any violation; nothing is skipped.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from episodes import episode_cache as ec
from analysis import stats_utils as su

_ID = re.compile(r"^(?P<split>test|val|dev)-(?P<fold>f\d|x)-(?P<user>P\d+)-n(?P<n>\d+)k(?P<k>\d+)-(?P<i>\d+)$")


def _walk(node, path, stack, out):
    if isinstance(node, dict):
        if "per_episode" in node and "episode_source" in node:
            seed = next((d["seed"] for d in reversed(stack + [node]) if "seed" in d), None)
            out.append((tuple(path), seed, node))
        for key, val in node.items():
            if key in ("per_episode",):
                continue
            _walk(val, path + [key] if isinstance(key, str) else path, stack + [node], out)
    elif isinstance(node, list):
        for val in node:
            _walk(val, path, stack, out)


def collect_blocks(results_path) -> list:
    """-> list of dicts: method, seed, source, rows (per-episode)."""
    obj = json.loads(Path(results_path).read_text())
    if "ablation_id" not in obj:
        raise KeyError(f"{results_path}: no top-level ablation_id")
    if obj["smoke"]:
        raise ValueError(f"{results_path}: SMOKE run results must not be aggregated")
    found = []
    _walk(obj, [], [], found)
    if not found:
        raise ValueError(f"{results_path}: no block with per_episode + episode_source")
    blocks = []
    for path, seed, node in found:
        keys = [p for p in path if p not in ("fold_results", "fold_result", "result")]
        method = "/".join([obj["ablation_id"]] + keys)
        blocks.append({"method": method, "seed": seed, "source": node["episode_source"],
                       "rows": node["per_episode"], "file": str(results_path)})
    return blocks


def parse_id(episode_id: str) -> dict:
    m = _ID.match(episode_id)
    if not m:
        raise ValueError(f"unparseable episode id {episode_id!r}")
    return m.groupdict()


def check_source(block, manifest) -> None:
    name = Path(block["source"]["file"]).name
    if name not in manifest["files"]:
        raise KeyError(f"{block['file']}: episode file {name} not in manifest")
    if block["source"]["sha256"] != manifest["files"][name]:
        raise ValueError(f"{block['file']} ({block['method']}): episode hash "
                         f"{block['source']['sha256'][:12]} != manifest {manifest['files'][name][:12]}")


def aggregate(result_paths, subset_ok=(), manifest=None, expect_folds=range(7)) -> dict:
    """-> {method: {cell: {user: acc}}}  (per-user acc averaged over train seeds)."""
    manifest = ec.load_manifest() if manifest is None else manifest
    blocks = [b for p in result_paths for b in collect_blocks(p)]
    for b in blocks:
        check_source(b, manifest)
        for r in b["rows"]:
            parse_id(r["id"])

    # ids per (method, seed, fold, cell)
    ids = defaultdict(set)
    rows_by = defaultdict(list)
    for b in blocks:
        for r in b["rows"]:
            p = parse_id(r["id"])
            if p["split"] != "test":
                raise ValueError(f"{b['method']}: non-test episode {r['id']} in a results block")
            key = (b["method"], b["seed"], int(p["fold"][1:]), (int(p["n"]), int(p["k"])))
            if r["id"] in ids[key]:
                raise ValueError(f"{b['method']}: duplicate episode id {r['id']}")
            ids[key].add(r["id"])
            rows_by[key].append(r)

    methods = sorted({k[0] for k in ids})
    cells = sorted({k[3] for k in ids})
    # same id set across methods (and across seeds of one method) per (fold, cell)
    for fold in expect_folds:
        for cell in cells:
            ref_key, ref = None, None
            for key, s in sorted(ids.items(), key=lambda kv: str(kv[0])):
                if key[2] != fold or key[3] != cell or key[0] in subset_ok:
                    continue
                if ref is None:
                    ref_key, ref = key, s
                elif s != ref:
                    raise ValueError(
                        f"episode id sets differ at fold {fold} cell {cell}: {key[:2]} vs "
                        f"{ref_key[:2]} (missing {len(ref - s)}, extra {len(s - ref)})")
            for key, s in ids.items():
                if key[2] == fold and key[3] == cell and key[0] in subset_ok and ref is not None:
                    if not s <= ref:
                        raise ValueError(f"subset method {key[0]} has ids outside the reference")

    out = {}
    all_users = {u for f in ec.kfold_split.load_split()["folds"] for u in f["test"]}
    for m in methods:
        out[m] = {}
        for cell in cells:
            seeds = defaultdict(list)
            for key, rows in rows_by.items():
                if key[0] == m and key[3] == cell:
                    seeds[key[1]].extend(rows)
            if not seeds:
                continue
            per_seed = [su.per_user_acc(rows) for rows in seeds.values()]
            users = set().union(*[set(d) for d in per_seed])
            if m not in subset_ok and users != all_users:
                raise ValueError(f"{m} cell {cell}: users missing {sorted(all_users - users)}")
            if any(set(d) != users for d in per_seed):
                raise ValueError(f"{m} cell {cell}: seeds cover different users")
            out[m][cell] = {u: sum(d[u] for d in per_seed) / len(per_seed) for u in sorted(users)}
    return out


def summary_table(agg: dict, cell, n_boot: int = 10000) -> list:
    """One row per method for a cell: mean, 95% user-bootstrap CI, SD over users."""
    import numpy as np
    rows = []
    for m, cells in sorted(agg.items()):
        if cell not in cells:
            continue
        v = np.array(list(cells[cell].values()))
        mean, lo, hi = su.bootstrap_ci(v, n_boot=n_boot)
        rows.append({"method": m, "n_users": len(v), "mean": mean, "ci_lo": lo, "ci_hi": hi,
                     "sd_users": float(v.std(ddof=1))})
    return rows


def compare_to(agg: dict, cell, reference: str, others, n_boot: int = 10000) -> list:
    """Paired comparison of each `others` method vs `reference`, Holm-corrected over the list."""
    res = []
    for m in others:
        users = sorted(set(agg[m][cell]) & set(agg[reference][cell]))
        r = su.paired_compare([agg[m][cell][u] for u in users],
                              [agg[reference][cell][u] for u in users], n_boot=n_boot)
        r["method"], r["vs"] = m, reference
        res.append(r)
    for r, adj in zip(res, su.holm([r["p"] for r in res])):
        r["p_holm"] = adj
    return res
