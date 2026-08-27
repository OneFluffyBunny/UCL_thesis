"""Driver for the entrenchment sub-study -- see ENTRENCHMENT.md.

⚠️ AI-AUTHORED. Hypothesis, arms and predictions are Claude Code's, not a human's.

12 cells (6 stage-1 lengths x 2 gate sets) x 200 seeds, each a separate `train.py`
invocation so any cell can be reproduced by copying one line out of the log.

    .venv-pypy/Scripts/python run_entrenchment.py
    .venv-pypy/Scripts/python run_entrenchment.py --dry-run
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT_ROOT = "runs/_ent"

# --- frozen by ENTRENCHMENT.md before any confirmatory run
#
# G1 is pinned to the measured stage-1 solve distribution (pilot: median ~2 300-2 700,
# p90 ~9 000-11 500), not to round numbers: the whole design turns on whether damage
# appears when stage 1 is SOLVED or keeps accruing with stage-1 TIME, so the ladder has
# to straddle the solve distribution and then run far past it.
G1_LEVELS = (0, 1_000, 3_000, 10_000, 20_000, 100_000)
GATESETS = {"cgp4": "and,nand,or,nor", "nand": "nand"}

# Stage-2 budget, IDENTICAL for every arm. Total generations = G1 + STAGE2_BUDGET, so a
# long-G1 arm is not quietly given less stage-2 search than a short one -- that would
# manufacture the slowdown being measured, as censoring.
STAGE2_BUDGET = 400_000
SEED0, N_SEEDS = 2000, 200        # disjoint from phase 3 (0-49) and 3.5 (1000-1199)
NODES = 100
WORKERS = 8


def cells():
    for gname, gates in GATESETS.items():
        for g1 in G1_LEVELS:
            yield f"g1-{g1}-{gname}", g1, gname, gates


def command(tag, g1, gates):
    cmd = [sys.executable, str(HERE / "train.py"),
           "--task", "pair_partial",
           "--gates", gates,
           "--generations", str(g1 + STAGE2_BUDGET),
           "--post-solve-gens", "0",      # outcome is time-to-solve, not final state
           # stage1_active / stage1_solved_gen are not in the checkpoint format, so
           # these runs are deliberately non-resumable rather than silently losing them
           "--checkpoint-interval", "0",
           "--seed", str(SEED0),
           "--n-seeds", str(N_SEEDS),
           "--nodes", str(NODES),
           "--log-interval", "100000",
           "--workers", str(WORKERS),
           "--no-viz", "--no-resume",
           "--out-dir", OUT_ROOT,
           # the run name carries task/nodes/generations but NOT the gate set, and here
           # generations is a function of G1, so two cells could otherwise collide in
           # one directory. The tag is what keeps them apart.
           "--tag", tag]
    if g1:
        cmd += ["--stage1-gens", str(g1)]
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default="")
    args = p.parse_args(argv)

    todo = [c for c in cells() if args.only in c[0]]
    print(f"{len(todo)} cells x {N_SEEDS} seeds (seeds {SEED0}-{SEED0 + N_SEEDS - 1})")
    print(f"interpreter: {sys.executable}\n")
    t0 = time.time()
    for i, (tag, g1, gname, gates) in enumerate(todo, 1):
        cmd = command(tag, g1, gates)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        print(f"[{i}/{len(todo)}] {tag}", flush=True)
        t = time.time()
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}) -- stopping", flush=True)
            return r.returncode
        print(f"  done in {time.time() - t:.0f}s", flush=True)
    if not args.dry_run:
        print(f"\nall cells done in {time.time() - t0:.0f}s -> {OUT_ROOT}")
        print("now: conda run -n lndp python analyse_entrenchment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
