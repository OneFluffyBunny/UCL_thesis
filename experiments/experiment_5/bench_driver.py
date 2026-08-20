"""Time one (1+4) search configuration. Prints one JSON line. Used by `bench.py`.

    python bench_driver.py <exp_dir> <cgp|ecgp> <task> <nodes> <gens> <reps>

Separate process per measurement for the same reason `equivalence_driver.py` is:
`experiment_4` and `experiment_5` both own modules called `cgp`, so only one of them
can be the `cgp` in any given interpreter.

WARM-UP IS NOT OPTIONAL HERE. PyPy's JIT needs to see a loop run a few thousand times
before it compiles it, so a short timing run measures the interpreter warming up
rather than the code, and reports PyPy as *slower* than CPython. Every measurement
below therefore runs the full workload once untimed and only then starts the clock.

BEST-OF, NOT MEAN. Repeated timings of identical deterministic work spread ~1.4x on a
hybrid CPU (P-cores and E-cores, with Windows moving the process between them). The
minimum is the estimate with the scheduling noise removed -- and no amount of luck
makes slow code look fast, so the minimum cannot flatter a regression.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
import time


def build(exp_dir: str, mode: str, task: str, nodes: int):
    """A closure that runs `gens` generations of the search, returning parent hits."""
    sys.path.insert(0, str(pathlib.Path(exp_dir).resolve()))
    import cgp
    import ecgp
    import gates as gates_mod
    import tasks as tasks_mod

    multi = hasattr(cgp, "max_hits")            # experiment 5's many-output API
    gate_set = gates_mod.build_set(gates_mod.DEFAULT_GATES)
    n_in = tasks_mod.n_inputs(task)
    n_out = tasks_mod.n_outputs(task) if multi else 1
    in_masks = tasks_mod.input_masks(task)
    mask = tasks_mod.full_mask(task)
    target = (tasks_mod.target_masks(task, "and") if multi
              else tasks_mod.target_mask(task, "and"))
    arity = gates_mod.max_arity(gate_set)
    n_funcs = len(gate_set)

    def run(gens: int, seed: int) -> int:
        rnd = random.Random(seed)
        if mode == "ecgp":
            params = ecgp.Params()
            new = lambda: ecgp.random_individual(rnd, nodes, n_in, n_out, n_funcs)
            kid = lambda p: ecgp.mutate(p, rnd, n_in, n_funcs, params)
            sc = lambda g: ecgp.fitness(g, gate_set, in_masks, target, mask, n_in)
        else:
            n_mut = cgp.n_mutations(nodes, arity, n_out, 0.03)
            new = lambda: cgp.random_genotype(rnd, nodes, n_in, n_out, n_funcs, arity)
            kid = lambda p: cgp.mutate(p, rnd, n_mut, n_in, n_funcs, 1.0)
            sc = lambda g: cgp.fitness(g, gate_set, in_masks, target, mask, n_in)

        pop = [new() for _ in range(5)]
        scored = [sc(g) for g in pop]
        i = max(range(5), key=lambda k: scored[k][0])
        parent, (p_score, p_hits) = pop[i], scored[i]
        for _ in range(gens):
            kids = [kid(parent) for _ in range(4)]
            o = [sc(g) for g in kids]
            s = [x for x, _ in o]
            top = max(s)
            if top > p_score:
                i = s.index(top)
                parent, (p_score, p_hits) = kids[i], o[i]
            elif top == p_score:
                i = rnd.choice([k for k, v in enumerate(s) if v == top])
                parent, (p_score, p_hits) = kids[i], o[i]
                if mode == "ecgp":
                    ecgp.prune_modules(parent)
        return p_hits

    return run, n_in, n_out, multi


def main(argv) -> int:
    exp_dir, mode, task, nodes, gens, reps = (argv[0], argv[1], argv[2], int(argv[3]),
                                              int(argv[4]), int(argv[5]))
    run, n_in, n_out, multi = build(exp_dir, mode, task, nodes)

    run(gens, 0)                                       # warm-up (JIT + caches)
    best, hits = None, None
    for r in range(reps):
        t = time.perf_counter()
        hits = run(gens, 0)
        el = time.perf_counter() - t
        best = el if best is None else min(best, el)

    print(json.dumps(dict(
        impl=pathlib.Path(exp_dir).name, python=("pypy" if hasattr(sys, "pypy_version_info")
                                                 else "cpython"),
        version=sys.version.split()[0], mode=mode, task=task, nodes=nodes,
        n_in=n_in, n_out=n_out, gens=gens, reps=reps,
        secs=round(best, 4), ms_per_gen=round(best / gens * 1e3, 5), hits=int(hits))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
