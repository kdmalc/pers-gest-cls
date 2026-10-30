"""
Generate the 28/4 outer split + 7 inner folds of 20/4/4 (decided 2026-10-21).

  * 4 users are held out as HPO-val users and are NEVER test users. We reuse the
    val users of the original Kapanji-based HPO split (P005, P131, P116, P109) so
    earlier HPO stays interpretable; they cover all three strata.
  * The remaining 28 users are split into 7 disjoint test blocks of 4, stratified
    by group (Non-disabled Kapanji / Disabled Kapanji / Disabled Non-Kapanji) by
    shuffling within stratum (seeded) and dealing round-robin across folds.
  * Fold k: test = block k, val = block (k+1) % 7, train = the other 5 blocks (20).

Stdlib only. Run from anywhere:  python make_28_4_7fold_split.py
Output: kfold28_hpo4_split.json (next to this file).
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "kfold28_hpo4_split.json"
SEED = 0
N_FOLDS = 7

# Source of truth for the Kapanji group labels (copied into the output JSON's "groups").
GROUPS = {
    "Non-disabled Kapanji": ["P011", "P010", "P008", "P006", "P005", "P004"],
    "Disabled Kapanji": ["P111", "P119", "P124", "P110", "P112", "P123", "P132", "P126",
                         "P131", "P104", "P116"],
    "Disabled Non-Kapanji": ["P102", "P114", "P107", "P103", "P125", "P127", "P128", "P118",
                             "P108", "P122", "P106", "P115", "P105", "P109", "P121"],
}
HPO_VAL = ["P005", "P131", "P116", "P109"]


def main():
    groups = GROUPS
    hpo_val = list(HPO_VAL)

    pool = {g: [p for p in pids if p not in hpo_val] for g, pids in groups.items()}
    rng = random.Random(SEED)
    ordered = []
    for g in sorted(pool):  # fixed stratum order
        ps = sorted(pool[g])
        rng.shuffle(ps)
        ordered += ps
    assert len(ordered) == 28, len(ordered)

    blocks = [[] for _ in range(N_FOLDS)]
    for i, p in enumerate(ordered):
        blocks[i % N_FOLDS].append(p)

    folds = []
    for k in range(N_FOLDS):
        test = sorted(blocks[k])
        val = sorted(blocks[(k + 1) % N_FOLDS])
        train = sorted(p for j, b in enumerate(blocks) if j not in (k, (k + 1) % N_FOLDS) for p in b)
        folds.append({"fold": k, "train": train, "val": val, "test": test})

    out = {
        "description": "28/4 outer split; 4 HPO-val users never tested; 7 folds of 20/4/4 over the 28.",
        "created": "2026-10-21",
        "seed": SEED,
        "hpo_val": sorted(hpo_val),
        "groups": groups,
        "folds": folds,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    g_of = {p: g for g, ps in groups.items() for p in ps}
    for f in folds:
        comp = lambda s: {g[:6] + g.split()[-1][:3]: sum(g_of[p] == g for p in f[s]) for g in groups}
        print(f["fold"], "test", f["test"], [g_of[p] for p in f["test"]])


if __name__ == "__main__":
    main()
