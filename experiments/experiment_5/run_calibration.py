"""Metric-validation run for SPEC -- see CALIBRATION.md.

⚠️ AI-AUTHORED. Not a hypothesis test: this asks whether the SPEC statistic defined in
`cgp.specialisation` measures what it is supposed to, by running it on tasks whose
ground-truth decomposition is already known.

    .venv-pypy/Scripts/python run_calibration.py          # (dispatches per task)
    conda run -n lndp python run_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import tasks

HERE = pathlib.Path(__file__).resolve().parent
PYPY = HERE / ".venv-pypy" / "Scripts" / "python.exe"
OUT_ROOT = "runs/_cal"
SEED0 = 5000
GATES = "and,nand,or,nor"
NODES = 100

# task -> (generations, n_seeds). The three `pair_*` tasks share a shape (8 in, 2 out)
# and differ ONLY in how much the two outputs must share, which is what makes them the
# calibration's ground truth. add4/mult4/retina_x2 probe whether that survives at more
# outputs and more inputs.
LADDER = {
    "pair_full":    (300_000, 100),   # O1 = O2  -> outputs MUST share everything
    "pair_partial": (300_000, 100),   # O1 = L, O2 = L XOR R -> partial overlap
    "pair_zero":    (300_000, 100),   # O1 = L, O2 = R -> outputs need share NOTHING
    "add4":         (500_000,  50),   # 5 outputs chained by a carry
    "mult4":        (500_000,  50),   # 8 outputs, no clean decomposition (control)
    "retina_x2":    (300_000,  30),   # 16 in, 2 independent retinas
}

# PyPy wins below the measured ~14-15 input crossover and loses above it (CLAUDE.md).
CROSSOVER = 15


def interpreter(task: str) -> str:
    return str(PYPY) if tasks.n_inputs(task) < CROSSOVER else sys.executable


def command(task: str) -> list[str]:
    gens, n_seeds = LADDER[task]
    return [interpreter(task), str(HERE / "train.py"),
            "--task", task, "--gates", GATES,
            "--generations", str(gens),
            "--post-solve-gens", "0",
            "--checkpoint-interval", "0",
            "--seed", str(SEED0), "--n-seeds", str(n_seeds),
            "--nodes", str(NODES), "--log-interval", "1000000",
            "--workers", "8", "--no-viz", "--no-resume",
            "--out-dir", OUT_ROOT, "--tag", f"cal-{task}"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default="")
    args = p.parse_args(argv)

    todo = [t for t in LADDER if args.only in t]
    print(f"{len(todo)} tasks, seeds from {SEED0}\n")
    t0 = time.time()
    for i, task in enumerate(todo, 1):
        cmd = command(task)
        if args.dry_run:
            print(" ".join(cmd)); continue
        gens, n_seeds = LADDER[task]
        print(f"[{i}/{len(todo)}] {task}  ({tasks.n_inputs(task)} in, "
              f"{tasks.n_outputs(task)} out, {n_seeds} seeds x {gens:,} gens, "
              f"{'pypy' if 'venv-pypy' in cmd[0] else 'cpython'})", flush=True)
        t = time.time()
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode})", flush=True)
            return r.returncode
        print(f"  done in {time.time() - t:.0f}s", flush=True)
    if not args.dry_run:
        print(f"\nall done in {time.time() - t0:.0f}s")
        print("now: conda run -n lndp python analyse_calibration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
