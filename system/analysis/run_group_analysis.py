"""
CLI for the per-ability-group MoE gain (docs/specs/SPEC_group_analysis.md).

  python system/analysis/run_group_analysis.py RESULT.json [RESULT.json ...] \
      --cell 10 1 [--treat M0 --ref A4] [--out DIR]

Goes through aggregate.aggregate(), so the episode-hash / id-set / missing-user checks all apply.
Writes group_gain_N{n}K{k}.csv and .md (gains in points) into --out (default: print only).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import aggregate as agg_mod          # noqa: E402
from analysis import group_analysis as ga           # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--cell", nargs=2, type=int, default=[10, 1], metavar=("N", "K"))
    p.add_argument("--treat", default="M0")
    p.add_argument("--ref", default="A4")
    p.add_argument("--out", default=None)
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args(argv)

    cell = tuple(a.cell)
    agg = agg_mod.aggregate(a.results)
    res = ga.group_gain(agg, cell, a.treat, a.ref, n_boot=a.n_boot)
    rows = ga.rows_for_table(res)
    md = ga.to_markdown(rows)
    print(f"{a.treat} - {a.ref}, cell N={cell[0]} K={cell[1]} (points)\n{md}\n")
    for c in ga.all_group_contrasts(res, n_boot=a.n_boot):
        print(f"gain({c['a']}) - gain({c['b']}) = {100*c['diff']:+.2f} "
              f"[{100*c['ci_lo']:+.2f}, {100*c['ci_hi']:+.2f}]")
    r = ga.difficulty_gain_corr(res, agg, cell, a.ref, n_boot=a.n_boot)
    print(f"Spearman(difficulty, gain) = {r['rho']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
    m, lo, hi = ga.fold_cluster_ci(res, n_boot=a.n_boot)
    print(f"overall gain, fold-cluster CI: {100*m:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"group_gain_N{cell[0]}K{cell[1]}"
        (out / f"{stem}.md").write_text(md + "\n")
        with open(out / f"{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
