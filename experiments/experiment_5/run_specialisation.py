"""Driver for the specialisation sub-study -- see SPECIALISATION.md.

⚠️ AI-AUTHORED. This whole sub-study (hypothesis, arms, metric, predictions) was
designed by Claude Code, not by a human. `SPECIALISATION.md` carries the full caveat.

Runs the 2 x 3 x 4 design as 24 independent `train.py` invocations and leaves the
aggregation to `analyse_specialisation.py`. Shelling out rather than importing
`run_seed` is deliberate: every cell then goes through the *same* entry point as any
hand-run command in `RESULTS.md`, so a cell can be reproduced by copying one line out
of the log, and a bug in this file cannot change the search.

    .venv-pypy/Scripts/python run_specialisation.py            # all 24 cells
    .venv-pypy/Scripts/python run_specialisation.py --dry-run  # print the commands
    .venv-pypy/Scripts/python run_specialisation.py --only cgp4     # one encoding

PyPy, because the task is 8 inputs and that is the side of the crossover where PyPy
wins ~6x (README, "Which interpreter"). The analysis script needs numpy/scipy and so
runs under CPython instead.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT_ROOT = "runs/_spec"

# --- frozen by the phase-2 pilot, 2026-08-21. Do not tune these against results.
#
#   STAGE1_GENS  stage 1 (the LEFT object alone) solved in 20/20 pilot seeds with a
#                median of 3 250 generations and a maximum of 17 250, so 20 000 gives
#                every seed a solved stage 1 with headroom without being mostly drift.
#   GENERATIONS  the cold arm solved `pair_partial` in 20/20 seeds, median ~47 000,
#                max ~86 000; NAND-only was 11/12 at 300 000. 500 000 pushes the
#                censoring rate toward zero, which matters because the primary
#                analysis conditions on solving.
#   POST_SOLVE   SPEC moves from ~0.65 to ~0.62 in the first ~20 000 generations after
#                a solution and is then flat for at least 100 000 more (pilot P3), so
#                measuring 20 000 generations past the solution clears the transient
#                and equalises post-solution drift across arms.
#   N_SEEDS      50, raised from the 30 written in SPECIALISATION.md before any
#                confirmatory run, purely because the pilot showed compute is not the
#                binding constraint. Fixed now; no seeds will be added later.
STAGE1_GENS = 20_000
GENERATIONS = 500_000
POST_SOLVE = 20_000
N_SEEDS = 50
NODES = 100
LOG_INTERVAL = 50_000
WORKERS = 8

SCHEDULES = {"staged": STAGE1_GENS, "cold": 0}
OVERLAPS = {"full": "pair_full", "partial": "pair_partial", "zero": "pair_zero"}
# name -> extra flags. `cgp4` is the preregistered baseline; the other three are
# exploratory (SPECIALISATION.md section 5).
ENCODINGS = {
    "cgp4":     [],
    "ecgp4":    ["--ecgp"],
    "cgpnand":  ["--gates", "nand"],
    "ecgpnand": ["--ecgp", "--gates", "nand"],
}


def cells():
    for sched, g1 in SCHEDULES.items():
        for ov, task in OVERLAPS.items():
            for enc, flags in ENCODINGS.items():
                yield f"{sched}-{ov}-{enc}", sched, ov, enc, task, g1, flags


def command(tag, task, g1, flags):
    cmd = [sys.executable, str(HERE / "train.py"),
           "--task", task,
           "--generations", str(GENERATIONS),
           "--post-solve-gens", str(POST_SOLVE),
           "--n-seeds", str(N_SEEDS),
           "--nodes", str(NODES),
           "--log-interval", str(LOG_INTERVAL),
           "--workers", str(WORKERS),
           "--no-viz",
           "--out-dir", OUT_ROOT,
           # The tag is the ONLY thing separating two cells' run directories: the
           # generated run name carries the task, node count and generation count but
           # NOT the gate set or the stage schedule, so without a distinct tag the
           # cgp4 and cgpnand cells would share a directory and --resume would happily
           # report one cell's finished seeds as the other's. This exact collision
           # already produced a nonsense benchmark once (RESULTS.md, 2026-08-19).
           "--tag", tag]
    if g1:
        cmd += ["--stage1-gens", str(g1)]
    return cmd + list(flags)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    p.add_argument("--only", default="", help="substring filter on the cell name")
    args = p.parse_args(argv)

    todo = [c for c in cells() if args.only in c[0]]
    print(f"{len(todo)} cell(s) x {N_SEEDS} seeds x up to {GENERATIONS:,} generations")
    print(f"interpreter: {sys.executable}")
    print()
    t0 = time.time()
    for i, (tag, sched, ov, enc, task, g1, flags) in enumerate(todo, 1):
        cmd = command(tag, task, g1, flags)
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
        print("now: conda run -n lndp python analyse_specialisation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
