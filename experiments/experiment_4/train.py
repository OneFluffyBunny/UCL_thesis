"""CGP (1+4) evolutionary strategy -- experiment 4.

The loop is PAPER_SPEC.md section 3, verbatim:

  1. generate `popsize` random genotypes, select the fittest
  2. mutate the winner into `popsize-1` offspring
  3. new generation = winner + offspring
  4. select the winner:  (a) best offspring if strictly better;
                         (b) else a RANDOM offspring tying the best fitness;
                         (c) else the parent
  5. repeat

Step 4b is the neutral-drift tie-break and is not optional -- it is the mechanism
CGP's efficiency rests on, letting the genotype wander across a fitness plateau
(here, the broad 192/256 plateau of the retina) while staying phenotypically
equal.

Run:  conda run -n lndp python train.py --task retina_ka2005 --operation and
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from dataclasses import asdict

import numpy as np

import cgp
import gates as gates_mod
import tasks as tasks_mod
from config import RunConfig, parse


def _goal_at(cfg: RunConfig, gen: int) -> str:
    """The active operation at generation `gen`."""
    if not cfg.mvg:
        return cfg.operation
    return cfg.mvg_ops[(gen // cfg.switch_interval) % len(cfg.mvg_ops)]


def run_seed(cfg: RunConfig, seed: int, gate_set, in_masks, mask, n_in,
             n_patterns, targets) -> tuple[list[dict], cgp.Genotype, int]:
    """One independent run. Returns (log rows, best genotype, its raw hits)."""
    rng = np.random.default_rng(seed)
    arity = gates_mod.max_arity(gate_set)
    n_funcs = len(gate_set)
    n_mut = cgp.n_mutations(cfg.nodes, arity, 1, cfg.mutation_rate)
    n_off = cfg.popsize - 1

    goal = _goal_at(cfg, 0)
    target = targets[goal]

    # Step 1 -- random population, select the fittest.
    pop = [cgp.random_genotype(rng, cfg.nodes, n_in, 1, n_funcs, arity)
           for _ in range(cfg.popsize)]
    scored = [cgp.fitness(g, gate_set, in_masks, target, mask, n_in, cfg.fitness)
              for g in pop]
    best_i = int(np.argmax([s for s, _ in scored]))
    parent, (p_score, p_hits) = pop[best_i], scored[best_i]

    rows: list[dict] = []
    best_geno, best_hits = parent.copy(), p_hits
    evals = cfg.popsize
    t_interval = time.time()
    solved_gen = -1

    for gen in range(cfg.generations):
        new_goal = _goal_at(cfg, gen)
        if new_goal != goal:
            # The goal moved: the parent's stored score refers to the old target
            # and must be recomputed before any comparison against offspring.
            goal, target = new_goal, targets[new_goal]
            p_score, p_hits = cgp.fitness(parent, gate_set, in_masks, target,
                                          mask, n_in, cfg.fitness)
            evals += 1

        # Steps 2-3 -- mutate the winner into offspring.
        offspring = [cgp.mutate(parent, rng, n_mut, n_in, n_funcs)
                     for _ in range(n_off)]
        o_scored = [cgp.fitness(g, gate_set, in_masks, target, mask, n_in, cfg.fitness)
                    for g in offspring]
        evals += n_off

        # Step 4 -- selection with the neutral tie-break.
        o_scores = [s for s, _ in o_scored]
        top = max(o_scores)
        if top > p_score:                                        # 4a
            i = int(np.argmax(o_scores))
            parent, (p_score, p_hits) = offspring[i], o_scored[i]
        elif top == p_score:                                     # 4b
            ties = [i for i, s in enumerate(o_scores) if s == top]
            i = int(rng.choice(ties))
            parent, (p_score, p_hits) = offspring[i], o_scored[i]
        # else 4c: the parent stays

        if p_hits > best_hits:
            best_geno, best_hits = parent.copy(), p_hits
        if solved_gen < 0 and p_hits == n_patterns:
            solved_gen = gen

        # Logging. Under MVG rows are emitted at the END of each goal epoch, so
        # every row is the same object -- a goal the lineage has had a full epoch
        # to adapt to -- and rows line up with the switches. (Same convention as
        # experiment 1, commit ed70189.) Gen 0 and the last generation always log.
        epoch = max(1, cfg.switch_interval)
        due = ((gen + 1) % epoch == 0) if cfg.mvg else (gen % cfg.log_interval == 0)
        if due or gen == 0 or gen == cfg.generations - 1:
            n_active = len(cgp.active_nodes(parent, n_in, gate_set))
            span = (epoch if cfg.mvg else cfg.log_interval) if gen > 0 else 1
            secs = (time.time() - t_interval) / span
            t_interval = time.time()
            rows.append(dict(seed=seed, gen=gen, goal=goal, score=p_score,
                             hits=p_hits, acc=p_hits / n_patterns,
                             active_nodes=n_active, evals=evals,
                             secs_per_gen=round(secs, 6)))
            print(f"  [seed {seed:3d}] gen {gen:6d} | goal {goal:3s} | "
                  f"hits {p_hits:4d}/{n_patterns} ({p_hits / n_patterns:.4f}) | "
                  f"active {n_active:3d} | {secs * 1e3:.2f} ms/gen", flush=True)

        if cfg.stop_on_solution and p_hits == n_patterns:
            break

    for r in rows:
        r["solved_gen"] = solved_gen
    return rows, best_geno, best_hits


def main(argv=None) -> int:
    cfg = parse(argv)
    gate_set = gates_mod.build_set(cfg.gates)
    n_in = tasks_mod.n_inputs(cfg.task)
    n_patterns = tasks_mod.n_patterns(cfg.task)
    mask = tasks_mod.full_mask(cfg.task)
    in_masks = tasks_mod.input_masks(cfg.task)

    # Precompute every goal's target mask once; MVG then costs a dict lookup.
    ops = cfg.mvg_ops if cfg.mvg else (cfg.operation,)
    targets = {op: tasks_mod.target_mask(cfg.task, op) for op in ops}

    arity = gates_mod.max_arity(gate_set)
    n_mut = cgp.n_mutations(cfg.nodes, arity, 1, cfg.mutation_rate)

    name = (f"cgp_{cfg.task}_{'mvg-' + '-'.join(cfg.mvg_ops) if cfg.mvg else cfg.operation}"
            f"_n{cfg.nodes}_g{cfg.generations}{('_' + cfg.tag) if cfg.tag else ''}")
    out = pathlib.Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"experiment 4 -- CGP | {name}")
    print(f"  task {cfg.task} ({n_in} inputs, {n_patterns} patterns) | "
          f"gates [{cfg.gates}] arity {arity} | fitness {cfg.fitness}")
    print(f"  {cfg.nodes} nodes, {cgp.n_gene_slots(cfg.nodes, arity, 1)} gene slots, "
          f"{n_mut} mutated/application | (1+{cfg.popsize - 1}) ES")
    print(f"  {cfg.n_seeds} seeds x {cfg.generations} generations -> {out}", flush=True)

    all_rows: list[dict] = []
    summary: list[dict] = []
    t0 = time.time()
    for k in range(cfg.n_seeds):
        seed = cfg.seed + k
        rows, best, best_hits = run_seed(cfg, seed, gate_set, in_masks, mask,
                                         n_in, n_patterns, targets)
        all_rows.extend(rows)
        solved = rows[-1]["solved_gen"] if rows else -1
        summary.append(dict(seed=seed, best_hits=best_hits,
                            best_acc=best_hits / n_patterns, solved_gen=solved,
                            evals=rows[-1]["evals"] if rows else 0))
        if cfg.save_best:
            np.savez(out / f"best_seed{seed}.npz", **{k_: v for k_, v in
                                                      asdict(best).items()})

    if all_rows:
        with (out / "log.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    accs = [s["best_acc"] for s in summary]
    n_solved = sum(1 for s in summary if s["solved_gen"] >= 0)
    print(f"\ndone in {time.time() - t0:.1f}s | best acc {np.mean(accs):.4f} "
          f"+/- {np.std(accs):.4f} (max {max(accs):.4f}) | "
          f"solved {n_solved}/{cfg.n_seeds} | -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
