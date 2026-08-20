"""Run one fixed (1+4) search inside a chosen experiment's code and print a digest.

Used by `test_equivalence.py`. It exists as a separate process because
`experiment_4` and `experiment_5` both contain modules called `cgp`, `ecgp`, `gates`
and `tasks`: importing both into one interpreter would have them fight over
`sys.modules` and silently mix halves of the two implementations, which is exactly
the failure this whole comparison is supposed to detect.

    python equivalence_driver.py <exp_dir> <cgp|ecgp> <nodes> <gens> <seed> [--draw-order]

`--draw-order` monkey-patches the target experiment's slot sampler to return its
picks in DRAW order. Experiment 5's already does; applying it to experiment 4 removes
the one documented difference between them, so a digest match then proves that
*nothing else* changed. Without the flag, experiment 4 runs exactly as it always has.

The digest is a rolling hash of `(generation, parent hits)` for every generation plus
a hash of the final genotype, so a divergence anywhere in the run shows up, not just
one at the end.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import sys


def _draw_slots_ordered(rnd, total, k):
    """Experiment 5's sampler: identical draws, returned in draw order.

    Deliberately re-implemented here rather than imported, so the patch applied to
    experiment 4 cannot accidentally pull in any other experiment-5 behaviour.
    """
    if k >= total:
        return list(range(total))
    if k * 2 > total:
        return rnd.sample(range(total), k)
    rand = rnd.random
    picked, order = set(), []
    while len(picked) < k:
        s = int(rand() * total)
        if s not in picked:
            picked.add(s)
            order.append(s)
    return order


def _draw_slots_biased_ordered(rnd, total, n_func, k, w):
    rand = rnd.random
    picked, order = set(), []
    while len(picked) < k:
        s = int(rand() * total)
        if s >= n_func and rand() >= w:
            continue
        if s not in picked:
            picked.add(s)
            order.append(s)
    return order


def main(argv) -> int:
    exp_dir, mode, nodes, gens, seed = (argv[0], argv[1], int(argv[2]),
                                        int(argv[3]), int(argv[4]))
    patch = "--draw-order" in argv[5:]

    sys.path.insert(0, str(pathlib.Path(exp_dir).resolve()))
    import cgp
    import ecgp
    import gates as gates_mod
    import tasks as tasks_mod

    if patch:
        cgp._draw_slots = _draw_slots_ordered
        cgp._draw_slots_biased = _draw_slots_biased_ordered

    # Experiment 5 scores a TUPLE of target masks (one per program output);
    # experiment 4 scores a single mask. One output either way here, so the only
    # difference is the container -- detected, not assumed.
    multi = hasattr(cgp, "max_hits")
    task = "retina_ka2005"
    gate_set = gates_mod.build_set(gates_mod.DEFAULT_GATES)
    n_in = tasks_mod.n_inputs(task)
    in_masks = tasks_mod.input_masks(task)
    mask = tasks_mod.full_mask(task)
    target = (tasks_mod.target_masks(task, "and") if multi
              else tasks_mod.target_mask(task, "and"))
    arity = gates_mod.max_arity(gate_set)
    n_funcs = len(gate_set)

    rnd = random.Random(seed)
    if mode == "ecgp":
        params = ecgp.Params()
        def new():
            return ecgp.random_individual(rnd, nodes, n_in, 1, n_funcs)
        def offspring(par):
            return ecgp.mutate(par, rnd, n_in, n_funcs, params)
        def score(g):
            return ecgp.fitness(g, gate_set, in_masks, target, mask, n_in, "raw")
        def flat(g):
            return ecgp.flatten(g, n_in)
    else:
        n_mut = cgp.n_mutations(nodes, arity, 1, 0.03)
        def new():
            return cgp.random_genotype(rnd, nodes, n_in, 1, n_funcs, arity)
        def offspring(par):
            return cgp.mutate(par, rnd, n_mut, n_in, n_funcs, 1.0)
        def score(g):
            return cgp.fitness(g, gate_set, in_masks, target, mask, n_in, "raw")
        def flat(g):
            return g

    # Step 1 of the ES: a random population of 5, keep the fittest.
    pop = [new() for _ in range(5)]
    scored = [score(g) for g in pop]
    i = max(range(len(scored)), key=lambda k: scored[k][0])
    parent, (p_score, p_hits) = pop[i], scored[i]

    h = hashlib.sha256()
    for gen in range(gens):
        kids = [offspring(parent) for _ in range(4)]
        o = [score(g) for g in kids]
        s = [x for x, _ in o]
        top = max(s)
        if top > p_score:                                   # rule 4a
            i = s.index(top)
            parent, (p_score, p_hits) = kids[i], o[i]
        elif top == p_score:                                # rule 4b, neutral drift
            i = rnd.choice([k for k, v in enumerate(s) if v == top])
            parent, (p_score, p_hits) = kids[i], o[i]
            if mode == "ecgp":
                ecgp.prune_modules(parent)
        elif mode == "ecgp":
            pass                                            # rule 4c: parent stays
        h.update(f"{gen}:{p_hits}\n".encode())

    g = flat(parent)
    genome = json.dumps([g.func, g.ntype, g.conn, g.cout, g.ogene, g.ocout, g.arity])
    print(json.dumps(dict(
        impl=pathlib.Path(exp_dir).name, mode=mode, multi_output_api=multi,
        patched=patch, hits=int(p_hits), score=float(p_score),
        trace_sha=h.hexdigest()[:32],
        genome_sha=hashlib.sha256(genome.encode()).hexdigest()[:32],
        rng_sha=hashlib.sha256(repr(rnd.getstate()).encode()).hexdigest()[:32],
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
