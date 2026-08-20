"""Does experiment 5 still run experiment 4's algorithm, and does PyPy agree?

Run:  conda run -n lndp python test_equivalence.py

Experiment 5 forked experiment 4's search so it could grow many outputs and run under
PyPy. A fork is only worth anything if the thing that was forked is still in there, so
this file answers two questions with evidence rather than with assertion:

  1. IS IT THE SAME SEARCH?  Experiment 5 changed exactly one thing about the
     algorithm: `_draw_slots` returns its picks in draw order instead of as a `set`,
     which was necessary because CPython and PyPy iterate a set of ints differently.
     Two tests pin that down. `test_same_slots_drawn` shows the two samplers consume
     the same random numbers and select the same slots from any RNG state -- so the
     *distribution* over mutations is untouched. `test_full_run_matches_exp4` then
     applies experiment 5's ordering to experiment 4's code and runs both searches
     end to end: the generation-by-generation fitness traces and the final genotypes
     hash equal, which leaves no room for a second difference to hide in.

  2. DOES PyPy GIVE THE SAME ANSWERS?  `test_pypy_matches_cpython` runs experiment 5
     under both interpreters and compares the same digests. This is the pay-off of
     the draw-order change: not "PyPy is close", but the identical run.

The PyPy test skips (loudly) when the interpreter is missing, so the file is still
useful on a machine that only has CPython.
"""

from __future__ import annotations

import json
import pathlib
import random
import subprocess
import sys

import cgp

HERE = pathlib.Path(__file__).resolve().parent
EXP4 = HERE.parent / "experiment_4"
PYPY = HERE / ".venv-pypy" / ("Scripts/python.exe" if sys.platform == "win32"
                              else "bin/python")


def _exp4_draw_slots(rnd, total, k):
    """`experiment_4/cgp.py::_draw_slots`, verbatim -- the set-returning original.

    Copied rather than imported: importing experiment 4's `cgp` into this process
    would collide with experiment 5's in `sys.modules`. Kept in sync by
    `test_full_run_matches_exp4`, which drives the real file in a subprocess.
    """
    if k >= total:
        return list(range(total))
    if k * 2 > total:
        return rnd.sample(range(total), k)
    rand = rnd.random
    picked = set()
    while len(picked) < k:
        picked.add(int(rand() * total))
    return picked


def test_same_slots_drawn() -> None:
    """Same RNG state in -> same slots out, and the same RNG state left behind.

    The three things that would change the experiment are all checked: which slots
    get mutated, how many random numbers the sampler burns (anything else drawing
    afterwards would shift), and the resulting state of the generator.
    """
    rnd_a, rnd_b = random.Random(11), random.Random(11)
    checked = 0
    for total, k in ((301, 9), (1601, 48), (2401, 72), (37, 18), (10, 10)):
        for _ in range(200):
            old = _exp4_draw_slots(rnd_a, total, k)
            new = cgp._draw_slots(rnd_b, total, k)
            assert set(old) == set(new), (total, k)
            assert len(new) == len(set(new)) == min(k, total)
            assert rnd_a.getstate() == rnd_b.getstate(), (total, k, "RNG diverged")
            checked += 1
    assert isinstance(cgp._draw_slots(random.Random(1), 100, 5), list)
    print(f"  ok  same slots, same RNG state, {checked} draws "
          f"(experiment 4 returned a set; experiment 5 returns them in draw order)")


def test_same_slots_drawn_biased() -> None:
    """The wiring-weighted sampler too -- it is a separate code path."""
    def exp4_biased(rnd, total, n_func, k, w):
        rand = rnd.random
        picked = set()
        while len(picked) < k:
            s = int(rand() * total)
            if s >= n_func and rand() >= w:
                continue
            picked.add(s)
        return picked

    rnd_a, rnd_b = random.Random(5), random.Random(5)
    for w in (0.1, 0.5, 0.9):
        for _ in range(200):
            old = exp4_biased(rnd_a, 301, 100, 9, w)
            new = cgp._draw_slots_biased(rnd_b, 301, 100, 9, w)
            assert set(old) == set(new), w
            assert rnd_a.getstate() == rnd_b.getstate(), w
    print("  ok  same slots for the wiring-weighted sampler too (w = 0.1 / 0.5 / 0.9)")


def _drive(python: str, exp_dir: pathlib.Path, mode: str, nodes: int, gens: int,
           seed: int, draw_order: bool) -> dict:
    cmd = [python, str(HERE / "equivalence_driver.py"), str(exp_dir), mode,
           str(nodes), str(gens), str(seed)]
    if draw_order:
        cmd.append("--draw-order")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    if r.returncode != 0:
        raise AssertionError(f"driver failed ({exp_dir.name}, {mode}):\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_full_run_matches_exp4() -> None:
    """End to end: with the ordering equalised, the two implementations are one run.

    Experiment 4 gets experiment 5's draw order patched in; nothing else about it is
    touched. If any other change had crept into the fork -- a different mutation
    range, a different tie-break, a different gate, a different fitness -- the traces
    would part company at the generation it first mattered, and the trace hash covers
    every generation, not just the last.
    """
    if not EXP4.exists():
        print(f"  SKIP test_full_run_matches_exp4 ({EXP4} not found)")
        return
    for mode, nodes, gens in (("cgp", 100, 400), ("cgp", 400, 200),
                              ("ecgp", 100, 300)):
        for seed in (0, 3):
            a = _drive(sys.executable, EXP4, mode, nodes, gens, seed, draw_order=True)
            b = _drive(sys.executable, HERE, mode, nodes, gens, seed, draw_order=False)
            assert not a["multi_output_api"] and b["multi_output_api"], "wrong fork"
            for key in ("hits", "score", "trace_sha", "genome_sha", "rng_sha"):
                assert a[key] == b[key], (mode, nodes, seed, key, a[key], b[key])
            print(f"  ok  {mode.upper():4s} n={nodes:3d} seed={seed} gens={gens}: "
                  f"identical to experiment 4 (hits {a['hits']}, "
                  f"trace {a['trace_sha'][:12]})")


def test_exp4_and_exp5_differ_without_the_patch() -> None:
    """The patch is doing real work -- i.e. the test above is not vacuous.

    If experiment 4 runs unpatched and still matched, the draw-order change would be
    a no-op and something about this comparison would be wrong. A control that can
    never fail proves nothing, so this asserts the difference exists.
    """
    if not EXP4.exists():
        print(f"  SKIP test_exp4_and_exp5_differ_without_the_patch ({EXP4} not found)")
        return
    a = _drive(sys.executable, EXP4, "cgp", 100, 400, 0, draw_order=False)
    b = _drive(sys.executable, HERE, "cgp", 100, 400, 0, draw_order=False)
    assert a["trace_sha"] != b["trace_sha"], (
        "experiment 4 unpatched matched experiment 5 -- then the draw-order patch is "
        "a no-op and test_full_run_matches_exp4 is not testing what it claims")
    print(f"  ok  unpatched, the two DO differ (exp4 {a['hits']} hits vs exp5 "
          f"{b['hits']}) -- so the patched match above is meaningful")


def test_pypy_matches_cpython() -> None:
    """The same experiment-5 seed gives the identical run under PyPy and CPython."""
    if not PYPY.exists():
        print(f"  SKIP test_pypy_matches_cpython (no PyPy venv at {PYPY})")
        return
    for mode, nodes, gens in (("cgp", 100, 800), ("cgp", 800, 200), ("ecgp", 100, 500)):
        for seed in (0, 7):
            a = _drive(sys.executable, HERE, mode, nodes, gens, seed, draw_order=False)
            b = _drive(str(PYPY), HERE, mode, nodes, gens, seed, draw_order=False)
            for key in ("hits", "score", "trace_sha", "genome_sha", "rng_sha"):
                assert a[key] == b[key], (mode, nodes, seed, key, a[key], b[key])
            print(f"  ok  {mode.upper():4s} n={nodes:3d} seed={seed} gens={gens}: "
                  f"PyPy == CPython (hits {a['hits']}, trace {a['trace_sha'][:12]})")


def main() -> int:
    print("test_equivalence.py")
    test_same_slots_drawn()
    test_same_slots_drawn_biased()
    test_full_run_matches_exp4()
    test_exp4_and_exp5_differ_without_the_patch()
    test_pypy_matches_cpython()
    print("all equivalence tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
