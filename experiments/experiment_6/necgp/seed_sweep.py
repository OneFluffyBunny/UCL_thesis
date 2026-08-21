"""Run necgp's nested-vs-flat run-to-solution twin (see `stage_run.py`) across
many seeds, to check whether the generation-count advantage seen at seed 0
(RESULTS.md's 2026-08-20 "run-to-solution, NAND-only" entry: nested solved in
32% fewer generations) holds up across seeds or was a one-seed fluke.

For each seed in [0, --seeds), runs the SAME twin `stage_run.run_to_solution`
calls stage_run.py's own `main` runs once (nest_decay as given vs
nest_decay=0.0), same task/driver, no stage-snapshot collection (statistics is
the point here, not imagery). Reports generations-to-solve and wall-clock per
seed, then a paired Wilcoxon signed-rank test on generations-to-solve --
paired because each seed's nested/flat pair shares the same seed (same RNG
stream up to where `compress` starts actually mattering), so this is a
matched-pairs design, not two independent samples.

Run: conda run -n lndp python seed_sweep.py --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import time

import gates as gates_mod
import tasks as tasks_mod
from stage_run import run_to_solution

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="retina_ka2005", choices=list(tasks_mod.TASKS))
    ap.add_argument("--operation", default="xor", choices=list(tasks_mod.OPERATIONS))
    ap.add_argument("--gates", default="nand")
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--popsize", type=int, default=5)
    ap.add_argument("--max-generations", type=int, default=300000)
    ap.add_argument("--mutation-rate", type=float, default=0.03)
    ap.add_argument("--compress-prob", type=float, default=0.1)
    ap.add_argument("--expand-prob", type=float, default=0.2)
    ap.add_argument("--module-point-prob", type=float, default=0.04)
    ap.add_argument("--add-input-prob", type=float, default=0.01)
    ap.add_argument("--remove-input-prob", type=float, default=0.02)
    ap.add_argument("--add-output-prob", type=float, default=0.01)
    ap.add_argument("--remove-output-prob", type=float, default=0.02)
    ap.add_argument("--max-module-size", type=int, default=5)
    ap.add_argument("--nest-decay", type=float, default=0.5)
    ap.add_argument("--fitness", default="raw", choices=["raw", "balanced"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--log-interval", type=int, default=10000)
    ap.add_argument("--out", default="seed_sweep.csv")
    args = ap.parse_args(argv)

    gate_set = gates_mod.build_set(args.gates)
    n_prim = len(gate_set)
    n_in = tasks_mod.n_inputs(args.task)
    in_masks = tasks_mod.input_masks(args.task)
    mask = tasks_mod.full_mask(args.task)
    target = tasks_mod.target_mask(args.task, args.operation)
    n_patterns = tasks_mod.n_patterns(args.task)

    print(f"necgp seed sweep -- task={args.task} operation={args.operation} "
          f"gates={args.gates} nodes={args.nodes} popsize={args.popsize} "
          f"nest_decay={args.nest_decay} seeds=0..{args.seeds - 1} "
          f"max_generations={args.max_generations}\n", flush=True)

    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        nested = run_to_solution(seed, args.nest_decay, args, gate_set, n_prim,
                                 n_in, in_masks, mask, target, n_patterns,
                                 collect_stages=False)
        flat = run_to_solution(seed, 0.0, args, gate_set, n_prim, n_in,
                               in_masks, mask, target, n_patterns,
                               collect_stages=False)
        row = dict(seed=seed,
                   nested_gen=nested["solved_gen"], nested_time=round(nested["elapsed"], 1),
                   flat_gen=flat["solved_gen"], flat_time=round(flat["elapsed"], 1))
        rows.append(row)
        diff = (row["nested_gen"] - row["flat_gen"]) \
            if row["nested_gen"] is not None and row["flat_gen"] is not None else "NA"
        print(f"seed {seed}: nested={row['nested_gen']} ({row['nested_time']}s)  "
              f"flat={row['flat_gen']} ({row['flat_time']}s)  diff={diff}  "
              f"[{time.time() - t0:.0f}s this pair]", flush=True)

    out_path = pathlib.Path(__file__).parent / args.out
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    solved_pairs = [(r["nested_gen"], r["flat_gen"]) for r in rows
                    if r["nested_gen"] is not None and r["flat_gen"] is not None]
    n_gens = [p[0] for p in solved_pairs]
    f_gens = [p[1] for p in solved_pairs]
    diffs = [n - f for n, f in solved_pairs]

    print("\n--- summary across seeds ---")
    print(f"solved pairs: {len(solved_pairs)}/{len(rows)}")
    if n_gens:
        print(f"nested gens: median {statistics.median(n_gens):.0f}  "
              f"mean {statistics.mean(n_gens):.1f}  "
              f"stdev {statistics.stdev(n_gens):.1f}" if len(n_gens) > 1 else
              f"nested gens: {n_gens[0]}")
        print(f"flat   gens: median {statistics.median(f_gens):.0f}  "
              f"mean {statistics.mean(f_gens):.1f}  "
              f"stdev {statistics.stdev(f_gens):.1f}" if len(f_gens) > 1 else
              f"flat gens: {f_gens[0]}")
        wins = sum(1 for d in diffs if d < 0)
        ties = sum(1 for d in diffs if d == 0)
        print(f"nested faster (fewer gens) in {wins}/{len(diffs)} seeds "
              f"({ties} ties, {len(diffs) - wins - ties} flat-faster)")
        print(f"mean paired diff (nested - flat): {statistics.mean(diffs):+.1f} generations")

    if scipy_stats is not None and len(diffs) >= 2 and any(d != 0 for d in diffs):
        stat, p = scipy_stats.wilcoxon(n_gens, f_gens)
        print(f"\nWilcoxon signed-rank (paired, nested vs flat gens-to-solve): "
              f"statistic={stat:.3f}  p={p:.4f}  "
              f"({'significant at alpha=0.05' if p < 0.05 else 'NOT significant at alpha=0.05'})")
    else:
        print("\nscipy unavailable, too few pairs, or no variation; skipping significance test")

    print(f"\nwrote per-seed data to {out_path}")


if __name__ == "__main__":
    main()
