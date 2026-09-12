"""Regenerate every table and figure for one study root, in one command.

    python analysis/run_all.py --root experiment_1/runs/fgmvg
    python analysis/run_all.py --root experiment_2/runs/fgmvg --quick

Produces, next to the runs:

    metrics_per_seed.csv          one row per run, all four metrics
    metrics_summary.json          the 4-group table, as markdown + numbers
    progress_fg_vs_mvg.png        whole-run trajectories, median + seed range
    switch_window_<c>_seed<N>.png accuracy + modularity across goal switches
    brains_grid_purity.png        every run's goal-matched champion, drawn
    brains_grid_community.png     the same, coloured by Newman community

`--quick` drops the null-model count and the sampling density, which turns a
~20 minute pass into ~2. Use it while iterating; use the default for anything
that goes in the thesis.

Nothing here is copied into `latex_figures/` -- that needs a human to look at the
picture first (see the figure-approval convention).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(script, *args):
    cmd = [PY, os.path.join(HERE, script), *map(str, args)]
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    t = time.time()
    rc = subprocess.call(cmd)
    print(f"  -> rc={rc} in {time.time()-t:.0f}s", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", default="0,1,2",
                   help="seeds to draw a switch-window figure for")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    n_rand = 50 if args.quick else 200
    n_points = 25 if args.quick else 60
    width = 200 if args.quick else 400

    failures = []
    if run("score_table.py", "--root", root, "--n-rand", n_rand):
        failures.append("score_table")
    if run("fig_progress.py", "--root", root, "--n-points", n_points):
        failures.append("fig_progress")

    for constraint in ("budget", "nobudget"):
        for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
            if run("fig_switch_window.py", "--root", root, "--seed", seed,
                   "--constraint", constraint, "--width", width):
                failures.append(f"switch_window/{constraint}/seed{seed}")

    for mode in ("purity", "community"):
        if run("fig_brains.py", "--root", root, "--grid", "--color-by", mode):
            failures.append(f"brains_grid/{mode}")

    print("\n" + "=" * 60)
    if failures:
        # Not fatal: a missing arm or a seed with no archive should not throw
        # away the outputs that did work.
        print("FAILED (or had nothing to do): " + ", ".join(failures))
    print(f"outputs in {root}")
    for f in sorted(os.listdir(root)):
        if f.endswith((".png", ".csv", ".json")):
            print(f"  {f}")


if __name__ == "__main__":
    main()
