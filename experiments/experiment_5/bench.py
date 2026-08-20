"""How much faster is experiment 5, and how much of that is PyPy?

    conda run -n lndp python bench.py              # the standard sweep
    conda run -n lndp python bench.py --quick      # a third of the generations
    conda run -n lndp python bench.py --csv out.csv
    conda run -n lndp python bench.py --crossover  # where does PyPy stop winning?

Runs the same (1+4) search under three implementations and prints a markdown table of
ms/generation plus the speedups:

    exp4 / CPython   the frozen baseline -- what experiment 4 costs today
    exp5 / CPython   the fork, same interpreter: isolates the code changes
    exp5 / PyPy      the fork under the JIT: the reason this branch exists

Reading the three columns in that order separates "the code got faster" from "the
interpreter got faster", which matters because they are not the same lever and only
one of them is portable to a machine without PyPy.

Every row is a real search, not a microbenchmark: mutate four offspring, evaluate
them, select, repeat. Experiment 4's `RESULTS.md` records an earlier estimate that was
fitted to a microbenchmark and came out 4x wrong, so this drives the loop itself.

!! The canary compares exp5 CPython against exp5 PyPy, NOT exp4 against exp5. exp4
and exp5 legitimately reach different circuits from the same seed -- that is the
documented draw-order change, and `test_equivalence.py` is where it is proved to be
the *only* difference. What must never differ is one implementation under two
interpreters, so that is what is asserted here: if PyPy ever scored differently from
CPython the timings below would be timing two different searches.
"""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
EXP4 = HERE.parent / "experiment_4"
PYPY = HERE / ".venv-pypy" / ("Scripts/python.exe" if sys.platform == "win32"
                              else "bin/python")

# (mode, task, nodes, generations). Generation counts are set so every row takes a
# few seconds: the small ones need many generations for PyPy's JIT to be warm and for
# the timer to have something to resolve, the 16-input ones are inherently slow
# because a wire there is a 64 kbit integer.
SWEEP = [
    ("cgp",  "retina_ka2005", 100, 40000),
    ("cgp",  "retina_ka2005", 400, 20000),
    ("cgp",  "retina_ka2005", 800, 15000),
    ("ecgp", "retina_ka2005", 100, 10000),
    ("ecgp", "retina_ka2005", 400,  6000),
    ("cgp",  "mult3",         200, 10000),
    ("cgp",  "add4",          200, 10000),
    ("cgp",  "retina_x2",     200,  1500),
]

# Tasks experiment 4 cannot run at all: its `tasks.py` has a fixed four-name registry
# and its fitness reads one program output. Those rows measure exp5 CPython vs PyPy
# only, and say so rather than printing a fake baseline.
EXP4_TASKS = {"retina_ka2005", "left", "and2", "copy"}

# `--crossover`: `parityN` is the only family whose input count moves ONE AT A TIME
# while the output count stays at 1, so it isolates the single variable that decides
# whether PyPy helps -- the width of a truth-table integer, `2**n_in` bits. Generation
# counts fall as the width rises because each generation costs ~2x more per input.
CROSSOVER = [(8, 20000), (10, 12000), (12, 8000), (13, 5000), (14, 4000),
             (15, 2500), (16, 1500), (17, 900), (18, 500)]


def _run(python: str, exp_dir: pathlib.Path, mode: str, task: str, nodes: int,
         gens: int, reps: int) -> dict | None:
    cmd = [python, str(HERE / "bench_driver.py"), str(exp_dir), mode, task,
           str(nodes), str(gens), str(reps)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    if r.returncode != 0:
        print(f"    ! failed ({exp_dir.name} / {pathlib.Path(python).name}): "
              f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}")
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def crossover(reps: int) -> int:
    """Sweep truth-table width and print where CPython overtakes PyPy.

    The headline table shows PyPy winning 3-6x on 8-input tasks and LOSING on a
    16-input one. That is not a contradiction, it is a crossover: PyPy's speed comes
    from compiling away interpreter overhead, and CPython's big integers are already
    hand-written C, so as soon as one bitwise op costs more than the interpreting of
    it, the advantage is gone and PyPy's slower `rbigint` shows through. Since "big
    brains" means exactly "wider truth tables", knowing where the line sits decides
    which interpreter a given run should use -- so it is measured, not guessed.
    """
    if not PYPY.exists():
        print(f"  --crossover needs the PyPy venv ({PYPY} missing)")
        return 1
    print(f"bench.py --crossover | CPython {sys.version.split()[0]} vs PyPy "
          f"| CGP parityN, n=200, best of {reps}\n")
    rows = []
    for n_in, gens in CROSSOVER:
        task = f"parity{n_in}"
        print(f"  {task} ({1 << n_in} patterns, {(1 << n_in) // 8} B per wire, "
              f"{gens} gens) ...", flush=True)
        b = _run(sys.executable, HERE, "cgp", task, 200, gens, reps)
        c = _run(str(PYPY), HERE, "cgp", task, 200, gens, reps)
        if b and c:
            rows.append((n_in, (1 << n_in) // 8, b["ms_per_gen"], c["ms_per_gen"]))
    print("\n| inputs | patterns | bytes/wire | CPython ms/gen | PyPy ms/gen | PyPy speedup |")
    print("|---|---|---|---|---|---|")
    for n_in, nbytes, cp, pp in rows:
        print(f"| {n_in} | {1 << n_in} | {nbytes} | {cp:.3f} | {pp:.3f} | "
              f"{'**' if cp / pp < 1 else ''}{cp / pp:.2f}x"
              f"{'**' if cp / pp < 1 else ''} |")
    winning = [n for n, _, cp, pp in rows if cp / pp > 1.0]
    losing = [n for n, _, cp, pp in rows if cp / pp <= 1.0]
    if winning and losing:
        print(f"\nCrossover between {max(winning)} and {min(losing)} inputs: PyPy wins "
              f"at or below {max(winning)}, CPython wins at or above {min(losing)}.")
    elif winning:
        print(f"\nPyPy wins across the whole swept range (up to {max(winning)} inputs).")
    else:
        print("\nCPython wins across the whole swept range.")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    if "--crossover" in argv:
        return crossover(1 if quick else 3)
    reps = 1 if quick else 3
    csv_path = None
    if "--csv" in argv:
        csv_path = pathlib.Path(argv[argv.index("--csv") + 1])

    have_pypy = PYPY.exists()
    have_exp4 = EXP4.exists()
    print(f"bench.py | CPython {sys.version.split()[0]}"
          f" | PyPy {'yes' if have_pypy else 'MISSING -- run setup_pypy.py'}"
          f" | experiment_4 baseline {'yes' if have_exp4 else 'MISSING'}")
    print(f"  {len(SWEEP)} configurations, best of {reps} timed repeat(s) after one "
          f"untimed warm-up pass\n")

    rows = []
    t0 = time.time()
    for mode, task, nodes, gens in SWEEP:
        if quick:
            gens = max(200, gens // 3)
        label = f"{mode.upper()} {task} n={nodes}"
        print(f"  {label} ({gens} gens) ...", flush=True)
        a = (_run(sys.executable, EXP4, mode, task, nodes, gens, reps)
             if have_exp4 and task in EXP4_TASKS else None)
        b = _run(sys.executable, HERE, mode, task, nodes, gens, reps)
        c = _run(str(PYPY), HERE, mode, task, nodes, gens, reps) if have_pypy else None
        if b is None:
            continue
        if c is not None and c["hits"] != b["hits"]:
            print(f"    !! PyPy scored {c['hits']} where CPython scored {b['hits']} on "
                  f"the SAME code and seed -- the timings below are not comparable. "
                  f"Run test_equivalence.py.")
        rows.append(dict(
            mode=mode, task=task, nodes=nodes, gens=gens,
            n_in=b["n_in"], n_out=b["n_out"], hits=b["hits"],
            exp4_cpython=a["ms_per_gen"] if a else None,
            exp5_cpython=b["ms_per_gen"],
            exp5_pypy=c["ms_per_gen"] if c else None))

    print(f"\n  (measured in {time.time() - t0:.0f}s)\n")
    _table(rows)
    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  -> {csv_path}")
    return 0


def _fmt(x) -> str:
    return "--" if x is None else f"{x:.3f}"


def _speedup(base, new) -> str:
    return "--" if not base or not new else f"{base / new:.1f}x"


def _table(rows) -> None:
    head = ("| config | in/out | exp4 CPython | exp5 CPython | exp5 PyPy | "
            "code | PyPy vs exp5 | **total** |")
    print(head)
    print("|" + "|".join(["---"] * 8) + "|")
    for r in rows:
        cfg = f"{r['mode'].upper()} {r['task']} n={r['nodes']}"
        print(f"| {cfg} | {r['n_in']}/{r['n_out']} | {_fmt(r['exp4_cpython'])} | "
              f"{_fmt(r['exp5_cpython'])} | {_fmt(r['exp5_pypy'])} | "
              f"{_speedup(r['exp4_cpython'], r['exp5_cpython'])} | "
              f"{_speedup(r['exp5_cpython'], r['exp5_pypy'])} | "
              f"**{_speedup(r['exp4_cpython'], r['exp5_pypy'])}** |")
    print("\nms per generation (one parent, four offspring: mutate, evaluate, select). "
          "Lower is better.")
    got = [r for r in rows if r["exp5_pypy"] and r["exp5_cpython"]]
    if got:
        sp = sorted(r["exp5_cpython"] / r["exp5_pypy"] for r in got)
        print(f"PyPy speedup over the same code on CPython: "
              f"{sp[0]:.1f}x .. {sp[-1]:.1f}x (median {sp[len(sp) // 2]:.1f}x).")


if __name__ == "__main__":
    raise SystemExit(main())
