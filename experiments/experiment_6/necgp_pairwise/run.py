"""EXPERIMENTAL. Run necgp_pairwise to a solution on the KA retina task and report
generation count plus a module-composition report (depth histogram, and a fake-
module check that should come back 0/0 by construction -- see ecgp.py's module
docstring). Adapted from ../necgp/stage_run.py, trimmed for this variant's smaller
Params (no module_point/add_input/remove_input/add_output/remove_output) and for
a single-arm run (this variant's `compress` always potentially nests, gated by
`nest_decay`, same as necgp/ -- no separate nested-vs-flat twin run here; that
comparison already exists for necgp/, this script is about the fake-module
question, not speed).

See ../README.md and ../RESULTS.md for status; do not cite this as `necgp/`.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter

import ecgp
import gates as gates_mod
import tasks as tasks_mod


def run_to_solution(seed: int, args, gate_set, n_prim, n_in, in_masks, mask,
                    target, n_patterns):
    rnd = random.Random(seed)
    p = ecgp.Params(compress=args.compress_prob, expand=args.expand_prob,
                    max_module_size=args.max_module_size,
                    mutation_rate=args.mutation_rate, nest_decay=args.nest_decay)

    pop = [ecgp.random_individual(rnd, args.nodes, n_in, 1, n_prim)
          for _ in range(args.popsize)]
    scored = [ecgp.fitness(g, gate_set, in_masks, target, mask, n_in, args.fitness)
             for g in pop]
    i = max(range(len(scored)), key=lambda k: scored[k][0])
    parent, (p_score, p_hits) = pop[i], scored[i]

    t0 = time.time()
    gen = 0
    solved_gen = None

    for gen in range(1, args.max_generations + 1):
        best_child, best_score, best_hits = None, -1.0, -1
        for _ in range(args.popsize - 1):
            child = ecgp.mutate(parent, rnd, n_in, n_prim, p)
            ecgp.prune_modules(child)
            score, hits = ecgp.fitness(child, gate_set, in_masks, target, mask,
                                       n_in, args.fitness)
            if score > best_score:
                best_child, best_score, best_hits = child, score, hits
        if best_score >= p_score:
            parent, p_score, p_hits = best_child, best_score, best_hits

        if gen % args.log_interval == 0:
            depth_hist: dict[int, int] = {}
            for m in parent.modules.values():
                depth_hist[m.depth] = depth_hist.get(m.depth, 0) + 1
            print(f"  [seed={seed}] gen {gen:>7}  score {p_score:.4f}  "
                  f"hits {p_hits}/{n_patterns}  modules {len(parent.modules)}  "
                  f"depth_hist {dict(sorted(depth_hist.items()))}", flush=True)

        if p_hits == n_patterns:
            solved_gen = gen
            break

    elapsed = time.time() - t0
    return dict(parent=parent, p_hits=p_hits, solved_gen=solved_gen, gen=gen,
               elapsed=elapsed)


def _module_active_body(mod: ecgp.Module) -> set[int]:
    active: set[int] = set()
    stack = [lbl - mod.n_in for lbl in mod.out if lbl >= mod.n_in]
    while stack:
        b = stack.pop()
        if b in active:
            continue
        active.add(b)
        for lbl in mod.conn[b]:
            if lbl >= mod.n_in:
                stack.append(lbl - mod.n_in)
    return active


def transitive_module_counts(parent: ecgp.Individual, n_in: int) -> Counter:
    """Same walk as necgp/scratch_decompose_final.py: how many times each module
    id is actually called, top level OR nested inside another active module."""
    counts: Counter = Counter()

    def walk(func, ntype, active_idx, modules) -> None:
        for b in active_idx:
            if ntype[b] == 0:
                continue
            mid = func[b]
            counts[mid] += 1
            mod = modules[mid]
            walk(mod.func, mod.ntype, _module_active_body(mod), modules)

    walk(parent.func, parent.ntype, ecgp.active_nodes(parent, n_in), parent.modules)
    return counts


def report(label: str, res: dict, n_in: int, n_prim: int, n_patterns: int) -> None:
    parent = res["parent"]
    status = f"SOLVED at gen {res['solved_gen']}" if res["solved_gen"] is not None \
        else "NOT SOLVED within budget"
    print(f"\n=== {label}: {status}  ({res['p_hits']}/{n_patterns} hits, "
          f"{res['elapsed']:.1f}s, {res['gen'] / max(res['elapsed'], 1e-9):.0f} gen/s) ===")

    ecgp.validate(parent, n_in, n_prim)
    print(f"  validate() passed -- including the 'no fake module survived' assertion")

    counts = transitive_module_counts(parent, n_in)
    fake = [m for m in counts if not ecgp.module_has_interaction(parent.modules[m], parent.modules)]
    depth_hist: dict[int, int] = {}
    for m in parent.modules.values():
        depth_hist[m.depth] = depth_hist.get(m.depth, 0) + 1
    print(f"  modules alive: {len(parent.modules)}  depth_hist {dict(sorted(depth_hist.items()))}")
    print(f"  module types reachable in active circuit: {len(counts)}  "
          f"({len(fake)} fake -- should be 0)")
    total_calls = sum(counts.values())
    fake_calls = sum(counts[m] for m in fake)
    print(f"  module calls in active circuit: {total_calls}  "
          f"({fake_calls} fake -- should be 0)")
    for m in sorted(counts, key=lambda m: -counts[m]):
        mod = parent.modules[m]
        prims = ecgp.module_active_primitive_count(mod, parent.modules)
        print(f"    {ecgp.module_name(m, n_prim):>6s}  x{counts[m]:<3d}  "
              f"depth={mod.depth}  flattened_primitives={prims}")


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
    ap.add_argument("--max-module-size", type=int, default=5)
    ap.add_argument("--nest-decay", type=float, default=0.5)
    ap.add_argument("--fitness", default="raw", choices=["raw", "balanced"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=10000)
    args = ap.parse_args(argv)

    gate_set = gates_mod.build_set(args.gates)
    n_prim = len(gate_set)
    n_in = tasks_mod.n_inputs(args.task)
    in_masks = tasks_mod.input_masks(args.task)
    mask = tasks_mod.full_mask(args.task)
    target = tasks_mod.target_mask(args.task, args.operation)
    n_patterns = tasks_mod.n_patterns(args.task)

    print(f"necgp_pairwise run [EXPERIMENTAL] -- task={args.task} operation={args.operation} "
          f"gates={args.gates} nodes={args.nodes} popsize={args.popsize} "
          f"seed={args.seed} max_module_size={args.max_module_size} "
          f"nest_decay={args.nest_decay} max_generations={args.max_generations}\n",
          flush=True)

    res = run_to_solution(args.seed, args, gate_set, n_prim, n_in, in_masks, mask,
                          target, n_patterns)
    report(f"seed {args.seed}", res, n_in, n_prim, n_patterns)


if __name__ == "__main__":
    main()
