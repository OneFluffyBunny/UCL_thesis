"""Performance and parallelism tests for experiment 4.

`test_cgp.py` checks that the search is CORRECT. This file checks that it is fast
and that the machine is actually being used, which are separate failure modes: a
silently serial run or a 3x regression in the hot loop both leave every assertion
in `test_cgp.py` passing.

Run:  conda run -n lndp python test_perf.py
      conda run -n lndp python test_perf.py --quick     (skips the timed runs)

Three questions, one test each:

  1. THROUGHPUT   -- is a generation still cheap? A ceiling on ms/gen catches a
                     regression in the hot loop (`cgp.evaluate` / `cgp.mutate`).
  2. UTILISATION  -- when we ask for N workers, do we actually get ~N cores of
                     work done, or are we paying for processes that idle?
  3. ADAPTIVITY   -- does `plan_workers` respond to the job it is given (seed
                     count, core count, run length) rather than returning a
                     constant that happens to suit one configuration?

Plus a correctness guard that belongs with the parallel work rather than in
`test_cgp.py`: running the same seeds through the pool must produce exactly the
results the serial path produces.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time

import train
from config import RunConfig, parse

RUNS = pathlib.Path("runs/_perftest")

# Ceilings, not targets, and deliberately loose.
#
# Repeated timing of the SAME deterministic work spreads ~1.4x run to run on a
# hybrid CPU (a Core Ultra 7 155H mixes fast P-cores with much slower E-cores and
# Windows moves the process between them), so a tight ceiling would fail on noise.
# These sit between what this implementation achieves (0.18-0.62 ms/gen best-of
# over 100-800 nodes) and what the numpy-genotype version it replaced achieved
# (0.55 / 1.57 / 3.23) -- loose enough never to fire on scheduling, tight enough
# that reverting the representation trips all three.
MS_PER_GEN_CEILING = {100: 0.45, 400: 1.20, 800: 2.20}
TIMING_REPEATS = 3          # report the best: min removes scheduling noise, and
                            # no amount of luck makes slow code look fast


def _cfg(**over) -> RunConfig:
    args = ["--task", "retina_ka2005", "--operation", "and", "--no-viz",
            "--no-resume", "--checkpoint-interval", "0", "--out-dir", str(RUNS)]
    for k, v in over.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return parse(args)


def _time_seed(nodes: int, gens: int) -> float:
    """ms per generation for one seed, logging and drawing off."""
    shutil.rmtree(RUNS, ignore_errors=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(nodes=nodes, n_seeds=1, seed=0, generations=gens,
               log_interval=gens * 2)
    c = train._task_context(cfg)
    t = time.time()
    train.run_seed(cfg, 0, c["gate_set"], c["in_masks"], c["mask"], c["n_in"],
                   c["n_patterns"], c["targets"], c["split"], RUNS, "perf")
    return (time.time() - t) / gens * 1e3


def test_throughput(gens: int = 2000) -> None:
    """A generation must stay under its ceiling at every genotype size."""
    print("  nodes |  best ms/gen | ceiling | all repeats")
    for nodes, ceiling in MS_PER_GEN_CEILING.items():
        runs = [_time_seed(nodes, gens) for _ in range(TIMING_REPEATS)]
        ms = min(runs)
        flag = "OK" if ms <= ceiling else "SLOW"
        print(f"  {nodes:5d} | {ms:12.3f} | {ceiling:7.2f} | "
              f"{', '.join(f'{r:.3f}' for r in runs)}  {flag}")
        assert ms <= ceiling, (
            f"{nodes} nodes: {ms:.3f} ms/gen exceeds the {ceiling} ms ceiling -- "
            f"the hot loop has regressed (see cgp.evaluate / cgp.mutate)")
    print("ok  throughput within ceiling at every genotype size")


def test_cpu_utilisation(nodes: int = 400, gens: int = 6000) -> None:
    """Workers must be busy, not idling.

    Efficiency = (core-seconds of useful work) / (wall seconds x workers). Each
    seed is single-threaded and CPU-bound, so a seed's wall time IS its core time
    and summing them gives the numerator honestly.

    Two seeds per worker so the measurement is not dominated by process startup
    (Windows spawns a fresh interpreter per worker, ~0.5-1 s each). Anything below
    the threshold means we are paying for cores we are not using.
    """
    workers = min(6, train._physical_cores())
    n_seeds = workers * 2
    shutil.rmtree(RUNS, ignore_errors=True)
    cfg = _cfg(nodes=nodes, n_seeds=n_seeds, seed=0, generations=gens,
               log_interval=gens * 2, workers=workers)

    t0 = time.time()
    rc = train.main(["--task", "retina_ka2005", "--operation", "and", "--no-viz",
                     "--no-resume", "--checkpoint-interval", "0",
                     "--out-dir", str(RUNS), "--nodes", str(nodes),
                     "--n-seeds", str(n_seeds), "--seed", "0",
                     "--generations", str(gens), "--log-interval", str(gens * 2),
                     "--workers", str(workers), "--tag", "util"])
    wall = time.time() - t0
    assert rc == 0

    import json
    d = next(RUNS.glob("*_util"))
    secs = [json.loads(p.read_text())["seconds"]
            for p in d.glob("*_result.json")]
    busy = sum(secs)
    eff = busy / (wall * workers)
    serial_equiv = busy
    print(f"  {n_seeds} seeds on {workers} workers: {busy:.1f} core-s of work in "
          f"{wall:.1f} s wall")
    print(f"  parallel efficiency {100 * eff:.0f}%  |  "
          f"speedup {serial_equiv / wall:.1f}x over serial")
    assert eff >= 0.70, (
        f"parallel efficiency {100 * eff:.0f}% -- workers are idle. Either the pool "
        f"is not being used or per-seed work is too small to amortise spawn cost.")
    print("ok  workers are saturated (>=70% efficiency)")


def test_parallel_matches_serial(nodes: int = 100, gens: int = 2500,
                                 n_seeds: int = 4) -> None:
    """The pool must not change results. Same seeds, same answers.

    Guards the real risk in the multiprocessing change: a worker rebuilding its own
    task context (`_task_context`) could silently differ from the parent's, and
    every seed would still 'work' while measuring something else.
    """
    import json
    got = {}
    for workers, tag in ((1, "ser"), (n_seeds, "par")):
        shutil.rmtree(RUNS / f"x{tag}", ignore_errors=True)
        rc = train.main(["--task", "retina_ka2005", "--operation", "and", "--no-viz",
                         "--no-resume", "--checkpoint-interval", "0",
                         "--out-dir", str(RUNS / f"x{tag}"), "--nodes", str(nodes),
                         "--n-seeds", str(n_seeds), "--seed", "0",
                         "--generations", str(gens), "--log-interval", str(gens * 2),
                         "--workers", str(workers), "--tag", tag])
        assert rc == 0
        d = next((RUNS / f"x{tag}").glob(f"*_{tag}"))
        got[tag] = [json.loads(p.read_text())
                    for p in sorted(d.glob("*_result.json"))]

    for a, b in zip(got["ser"], got["par"]):
        for key in ("seed", "best_hits", "final_hits", "solved_gen", "gens_run",
                    "evals", "active_nodes", "left", "right", "mixed"):
            assert a[key] == b[key], (
                f"seed {a['seed']}: {key} differs serial={a[key]} parallel={b[key]}")
    print(f"ok  {n_seeds} seeds give byte-identical results serial vs pooled")


def test_worker_planning_is_adaptive() -> None:
    """`plan_workers` must respond to the job, not return a constant.

    Table-driven over (seeds, generations, nodes, --workers, cores) so the policy
    is pinned rather than described. `cores` is passed explicitly so the test gives
    the same answer on any machine.
    """
    def plan(seeds, gens, nodes, workers, cores):
        return train.plan_workers(
            _cfg(n_seeds=seeds, generations=gens, nodes=nodes, workers=workers),
            cores=cores)

    cases = [
        # (seeds, gens, nodes, --workers, cores) -> expected, why
        ((50, 40000, 400, 0, 16), 16, "big job, plenty of seeds -> fill the cores"),
        ((50, 40000, 400, 0, 8),  8,  "same job, smaller box -> fewer workers"),
        ((4, 40000, 400, 0, 16),  4,  "only 4 seeds -> never more workers than seeds"),
        ((1, 999999, 800, 0, 16), 1,  "one seed is unparallelisable, however long"),
        ((12, 100, 100, 0, 16),   1,  "tiny job -> serial, spawning would cost more"),
        ((50, 40000, 400, 4, 16), 4,  "explicit --workers 4 is obeyed"),
        ((3, 40000, 400, 8, 16),  3,  "explicit --workers capped by seed count"),
        ((50, 40000, 400, 1, 16), 1,  "--workers 1 forces the serial path"),
    ]
    for (seeds, gens, nodes, w, cores), want, why in cases:
        got = plan(seeds, gens, nodes, w, cores)
        assert got == want, (f"plan_workers(seeds={seeds}, gens={gens}, "
                             f"nodes={nodes}, workers={w}, cores={cores}) "
                             f"= {got}, expected {want} -- {why}")
        print(f"  {got:2d} workers | {why}")

    # The policy must genuinely vary with each input, not coincidentally match.
    assert plan(50, 40000, 400, 0, 4) < plan(50, 40000, 400, 0, 16), "ignores cores"
    assert plan(2, 40000, 400, 0, 16) < plan(20, 40000, 400, 0, 16), "ignores seeds"
    assert plan(20, 50, 100, 0, 16) < plan(20, 40000, 800, 0, 16), "ignores job size"
    print("ok  worker planning adapts to seeds, cores and job size")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    try:
        print("\n[3] worker planning")
        test_worker_planning_is_adaptive()
        if quick:
            print("\n--quick: skipping timed runs")
        else:
            print("\n[1] throughput")
            test_throughput()
            print("\n[correctness] parallel == serial")
            test_parallel_matches_serial()
            print("\n[2] cpu utilisation")
            test_cpu_utilisation()
        print("\nall performance tests passed")
    finally:
        shutil.rmtree(RUNS, ignore_errors=True)
