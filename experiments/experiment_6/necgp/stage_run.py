"""Run necgp to a solution, NAND-only, on the KA retina task, and draw its
evolutionary history as a 2x3 stage sheet (`visualize.stage_grid`), module boxes
tagged with their nesting factor (`|1` primitive, `|2` a module built only from
primitives, `|3` a module that nests a `|2` module, ...).

Runs TWO twin (1+popsize-1) ES searches on the same seed, same task, same
mutation operators -- `--nest-decay` given (default 0.5, nesting on) and
`--nest-decay 0.0` (nesting structurally impossible, see `ecgp.py`'s
`test_decay_zero_blocks_all_nesting`) -- back to back, so the only thing that
differs between the two runs is whether `compress` may fold an existing module
into a new one. That isolates what nesting itself buys or costs in
generations/wall-clock to reach 256/256, inside the SAME driver -- not a
comparison against experiment_4's numbers, which used a different codebase.

Stops on first solve (`hits == n_patterns`), not at a fixed generation count,
so "generations to solve" is a real number to compare, not an artifact of a
budget. Safety cap `--max-generations` in case a run never solves.

See ../RESULTS.md for what this run found.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import time

import ecgp
import gates as gates_mod
import tasks as tasks_mod
import visualize as viz


def _even_picks(seq: list, k: int) -> list:
    """`k` roughly evenly spaced items of `seq`, always including the first and last."""
    if len(seq) <= k:
        return list(seq)
    idx = sorted({round(i * (len(seq) - 1) / (k - 1)) for i in range(k)})
    return [seq[i] for i in idx]


def run_to_solution(seed: int, nest_decay: float, args, gate_set, n_prim, n_in,
                    in_masks, mask, target, n_patterns, collect_stages: bool):
    rnd = random.Random(seed)
    p = ecgp.Params(compress=args.compress_prob, expand=args.expand_prob,
                    module_point=args.module_point_prob,
                    add_input=args.add_input_prob, remove_input=args.remove_input_prob,
                    add_output=args.add_output_prob, remove_output=args.remove_output_prob,
                    max_module_size=args.max_module_size,
                    mutation_rate=args.mutation_rate, nest_decay=nest_decay)

    pop = [ecgp.random_individual(rnd, args.nodes, n_in, 1, n_prim)
          for _ in range(args.popsize)]
    scored = [ecgp.fitness(g, gate_set, in_masks, target, mask, n_in, args.fitness)
             for g in pop]
    i = max(range(len(scored)), key=lambda k: scored[k][0])
    parent, (p_score, p_hits) = pop[i], scored[i]

    # Snapshots for the stage grid, decimated the same way experiment_4/train.py
    # does it: keep <=18, halving resolution (doubling the stride) whenever the
    # list overflows, so a long run still ends with an evenly spread history.
    stages: list[tuple] = []
    stride = [1]
    n_seen = [0]

    def snapshot(gen: int) -> None:
        if not collect_stages:
            return
        n_seen[0] += 1
        if (n_seen[0] - 1) % stride[0] == 0:
            stages.append((gen, p_hits, parent.copy()))
            if len(stages) > 18:
                del stages[1::2]
                stride[0] *= 2

    snapshot(0)
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
            snapshot(gen)
            depth_hist: dict[int, int] = {}
            for m in parent.modules.values():
                depth_hist[m.depth] = depth_hist.get(m.depth, 0) + 1
            print(f"  [nest_decay={nest_decay}] gen {gen:>7}  score {p_score:.4f}  "
                  f"hits {p_hits}/{n_patterns}  modules {len(parent.modules)}  "
                  f"depth_hist {dict(sorted(depth_hist.items()))}", flush=True)

        if p_hits == n_patterns:
            solved_gen = gen
            break

    elapsed = time.time() - t0
    if not stages or stages[-1][0] != gen:
        snapshot(gen)

    return dict(parent=parent, p_hits=p_hits, solved_gen=solved_gen, gen=gen,
               elapsed=elapsed, stages=stages)


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
    ap.add_argument("--module-point-prob", type=float, default=0.04)
    ap.add_argument("--add-input-prob", type=float, default=0.01)
    ap.add_argument("--remove-input-prob", type=float, default=0.02)
    ap.add_argument("--add-output-prob", type=float, default=0.01)
    ap.add_argument("--remove-output-prob", type=float, default=0.02)
    ap.add_argument("--max-module-size", type=int, default=5)
    ap.add_argument("--nest-decay", type=float, default=0.5)
    ap.add_argument("--fitness", default="raw", choices=["raw", "balanced"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=2000)
    args = ap.parse_args(argv)

    gate_set = gates_mod.build_set(args.gates)
    n_prim = len(gate_set)
    n_in = tasks_mod.n_inputs(args.task)
    in_masks = tasks_mod.input_masks(args.task)
    mask = tasks_mod.full_mask(args.task)
    target = tasks_mod.target_mask(args.task, args.operation)
    n_patterns = tasks_mod.n_patterns(args.task)

    print(f"necgp stage run -- task={args.task} operation={args.operation} "
          f"gates={args.gates} nodes={args.nodes} popsize={args.popsize} "
          f"seed={args.seed} max_generations={args.max_generations}\n", flush=True)

    print(f"=== run A: nest_decay={args.nest_decay} (nesting ON) ===", flush=True)
    res_nested = run_to_solution(args.seed, args.nest_decay, args, gate_set, n_prim,
                                 n_in, in_masks, mask, target, n_patterns,
                                 collect_stages=True)

    print(f"\n=== run B: nest_decay=0.0 (nesting OFF, everything else identical) ===",
          flush=True)
    res_flat = run_to_solution(args.seed, 0.0, args, gate_set, n_prim, n_in,
                               in_masks, mask, target, n_patterns,
                               collect_stages=False)

    def _report(label: str, res: dict) -> None:
        status = f"SOLVED at gen {res['solved_gen']}" if res["solved_gen"] is not None \
            else f"NOT SOLVED within {args.max_generations} generations"
        print(f"  {label}: {status}  ({res['p_hits']}/{n_patterns} hits, "
              f"{res['elapsed']:.1f}s, {res['gen'] / max(res['elapsed'], 1e-9):.0f} gen/s)")

    print("\n--- comparison ---")
    _report("nested  (nest_decay=%.2f)" % args.nest_decay, res_nested)
    _report("flat    (nest_decay=0.0)  ", res_flat)
    if res_nested["solved_gen"] is not None and res_flat["solved_gen"] is not None:
        dg = res_nested["solved_gen"] - res_flat["solved_gen"]
        dt = res_nested["elapsed"] - res_flat["elapsed"]
        print(f"  nested vs flat: {dg:+d} generations, {dt:+.1f}s "
              f"({'nested slower' if dg > 0 else 'nested faster' if dg < 0 else 'tied'})")

    ecgp.validate(res_nested["parent"], n_in, n_prim, args.max_module_size)
    nested_ids = {mid for mid, m in res_nested["parent"].modules.items() if m.depth > 1}
    print(f"\nnested run final population: {len(res_nested['parent'].modules)} modules alive, "
          f"{len(nested_ids)} of depth > 1 (real nesting): {sorted(nested_ids)}")

    panels_data = _even_picks(res_nested["stages"], viz.STAGE_ROWS * viz.STAGE_COLS)
    panels = []
    for gen_, hits_, snap in panels_data:
        hidden = viz.n_hidden_nodes(snap, n_in)
        panels.append((snap, None, viz.stage_title(gen_, hits_, n_patterns, hidden), None))

    sheet_title = (f"necgp (nested ECGP)  |  FG {args.operation}  |  "
                   f"1 starting gate (NAND)  |  {args.nodes} genes  |  "
                   f"nest_decay={args.nest_decay}  |  seed {args.seed}")
    out_path = pathlib.Path(__file__).parent / f"stages_seed{args.seed}_nand.png"
    viz.stage_grid(panels, gate_set, n_in, out_path, title=sheet_title, n_prim=n_prim)
    print(f"\nstage grid written to {out_path}")


if __name__ == "__main__":
    main()
