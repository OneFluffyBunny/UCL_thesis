"""Smoke-test driver for nested ECGP -- experiment 6's own extension of ecgp.py.

A (1+lambda) ES, same shape as experiment_4/train.py's core loop (its `run_seed`):
one parent, `--popsize - 1` offspring per generation, the best offspring replaces
the parent when it is at least as good (neutral drift allowed on purpose -- see
train.py's own comment on this, ECGP's modules only take hold if fitness-neutral
structural change can be kept). Stripped of train.py's checkpointing / CSV /
multiprocessing -- this is for looking at what nesting does, not a logged run.
See ../RESULTS.md for what this smoke run found.
"""

from __future__ import annotations

import argparse
import random
import time

import cgp
import ecgp
import gates as gates_mod
import tasks as tasks_mod


def run(args: argparse.Namespace) -> None:
    gate_set = gates_mod.build_set(args.gates)
    n_prim = len(gate_set)
    n_in = tasks_mod.n_inputs(args.task)
    in_masks = tasks_mod.input_masks(args.task)
    mask = tasks_mod.full_mask(args.task)
    target = tasks_mod.target_mask(args.task, args.operation)
    n_patterns = tasks_mod.n_patterns(args.task)

    rnd = random.Random(args.seed)
    p = ecgp.Params(compress=args.compress_prob, expand=args.expand_prob,
                    module_point=args.module_point_prob,
                    add_input=args.add_input_prob, remove_input=args.remove_input_prob,
                    add_output=args.add_output_prob, remove_output=args.remove_output_prob,
                    max_module_size=args.max_module_size,
                    mutation_rate=args.mutation_rate, nest_decay=args.nest_decay)

    print(f"necgp smoke run -- task={args.task} operation={args.operation} "
          f"gates={args.gates} nodes={args.nodes} popsize={args.popsize} "
          f"generations={args.generations} nest_decay={args.nest_decay} seed={args.seed}",
          flush=True)

    pop = [ecgp.random_individual(rnd, args.nodes, n_in, 1, n_prim)
          for _ in range(args.popsize)]
    scored = [ecgp.fitness(g, gate_set, in_masks, target, mask, n_in, args.fitness)
             for g in pop]
    i = max(range(len(scored)), key=lambda k: scored[k][0])
    parent, (p_score, p_hits) = pop[i], scored[i]

    max_depth_ever = 1
    t0 = time.time()

    for gen in range(1, args.generations + 1):
        best_child, best_score, best_hits = None, -1.0, -1
        for _ in range(args.popsize - 1):
            child = ecgp.mutate(parent, rnd, n_in, n_prim, p)
            ecgp.prune_modules(child)
            for m in child.modules.values():
                if m.depth > max_depth_ever:
                    max_depth_ever = m.depth
            score, hits = ecgp.fitness(child, gate_set, in_masks, target, mask,
                                       n_in, args.fitness)
            if score > best_score:
                best_child, best_score, best_hits = child, score, hits
        if best_score >= p_score:
            parent, p_score, p_hits = best_child, best_score, best_hits

        if gen % args.log_interval == 0 or gen == args.generations:
            depth_hist: dict[int, int] = {}
            for m in parent.modules.values():
                depth_hist[m.depth] = depth_hist.get(m.depth, 0) + 1
            print(f"gen {gen:>6}  score {p_score:.4f}  hits {p_hits}/{n_patterns}  "
                  f"modules {len(parent.modules)}  depth_hist {dict(sorted(depth_hist.items()))}  "
                  f"max_depth_ever {max_depth_ever}", flush=True)

    elapsed = time.time() - t0
    ecgp.validate(parent, n_in, n_prim, p.max_module_size)
    print(f"\ndone in {elapsed:.1f}s ({args.generations / max(elapsed, 1e-9):.0f} gen/s) "
          f"-- final score {p_score:.4f}, hits {p_hits}/{n_patterns}, "
          f"{len(parent.modules)} modules alive, deepest module ever seen: depth {max_depth_ever}")
    for mid, mod in sorted(parent.modules.items()):
        nested = [b for b in range(mod.n_nodes) if mod.ntype[b] != 0]
        print(f"  {ecgp.module_name(mid, n_prim)}: n_in={mod.n_in} n_out={mod.n_out} "
              f"n_nodes={mod.n_nodes} depth={mod.depth} "
              f"nested_children={[ecgp.module_name(mod.func[b], n_prim) for b in nested]}")

    flat = ecgp.flatten(parent, n_in)
    direct = ecgp.evaluate(parent, gate_set, in_masks, mask, n_in)
    via_flat = cgp.evaluate(flat, gate_set, in_masks, mask, n_in)
    assert direct == via_flat, "evaluate() disagrees with flatten()+cgp.evaluate() -- BUG"
    print("cross-check OK: evaluate() == cgp.evaluate(flatten(...))")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="retina_ka2005", choices=list(tasks_mod.TASKS))
    ap.add_argument("--operation", default="xor", choices=list(tasks_mod.OPERATIONS))
    ap.add_argument("--gates", default="nand")
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--popsize", type=int, default=5)
    ap.add_argument("--generations", type=int, default=20000)
    ap.add_argument("--mutation-rate", type=float, default=0.03)
    ap.add_argument("--compress-prob", type=float, default=0.1)
    ap.add_argument("--expand-prob", type=float, default=0.2)
    ap.add_argument("--module-point-prob", type=float, default=0.04)
    ap.add_argument("--add-input-prob", type=float, default=0.01)
    ap.add_argument("--remove-input-prob", type=float, default=0.02)
    ap.add_argument("--add-output-prob", type=float, default=0.01)
    ap.add_argument("--remove-output-prob", type=float, default=0.02)
    ap.add_argument("--max-module-size", type=int, default=5)
    ap.add_argument("--nest-decay", type=float, default=0.5)
    ap.add_argument("--fitness", default="raw", choices=["raw", "balanced"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=1000)
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
