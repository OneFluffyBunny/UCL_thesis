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
(here, the broad 192/256 plateau of the retina) while staying phenotypically equal.

OUTPUT LAYOUT follows kashtan_alon/train.py so the two are navigable the same way:

    runs/<name>/<run>_seed<k>_log.csv       one row per log point
    runs/<name>/<run>_seed<k>_ckpt.pkl      full state; deleted on completion
    runs/<name>/<run>_seed<k>_result.json   written when the seed finishes
    runs/<name>/<run>_seed<k>_best.npz      best genotype (--save-best)
    runs/<name>/frames/seed<k>_gen<g>.png   circuit diagram per log row

Resume is on by default: a finished seed is skipped, an interrupted one restarts
from its checkpoint with the RNG state restored, so a resumed run is identical to
an uninterrupted one.

NOT logged: density. A CGP genotype has a fixed edge count (`n_nodes * arity`) so
density is a constant, and it was a flawed readout in experiment 1 anyway. The
structural columns here are the input-cone decomposition instead -- see
`cgp.Phenotype`.

Run:  conda run -n lndp python train.py --task retina_ka2005 --operation and
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import pickle
import sys
import time
from dataclasses import asdict

import numpy as np

import cgp
import gates as gates_mod
import tasks as tasks_mod
import visualize as viz_mod
from config import RunConfig, parse

LOG_FIELDS = ["seed", "gen", "goal", "score", "hits", "acc", "active_nodes",
              "left", "right", "mixed", "const", "depth", "evals", "secs_per_gen"]


def _goal_at(cfg: RunConfig, gen: int) -> str:
    """The active operation at generation `gen`."""
    if not cfg.mvg:
        return cfg.operation
    return cfg.mvg_ops[(gen // cfg.switch_interval) % len(cfg.mvg_ops)]


def save_checkpoint(path: pathlib.Path, gen: int, rng, parent, p_score, p_hits,
                    best, best_hits, solved_gen, evals) -> None:
    """Atomically pickle full search state so a run resumes exactly where it stopped."""
    obj = {"gen": gen, "rng_state": rng.bit_generator.state,
           "parent": parent, "p_score": p_score, "p_hits": p_hits,
           "best": best, "best_hits": best_hits,
           "solved_gen": solved_gen, "evals": evals}
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)   # atomic: a crash mid-write cannot corrupt the checkpoint


def run_seed(cfg: RunConfig, seed: int, gate_set, in_masks, mask, n_in,
             n_patterns, targets, split, out: pathlib.Path, run: str):
    """One independent run. Returns its result dict, or None if already finished."""
    csv_path = out / f"{run}_seed{seed}_log.csv"
    ckpt_path = out / f"{run}_seed{seed}_ckpt.pkl"
    result_path = out / f"{run}_seed{seed}_result.json"
    frames = out / "frames"

    if cfg.resume and result_path.exists():
        print(f"[seed {seed}] already finished -> skipping "
              f"(--no-resume to redo)", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    rng = np.random.default_rng(seed)
    arity = gates_mod.max_arity(gate_set)
    n_funcs = len(gate_set)
    n_mut = cgp.n_mutations(cfg.nodes, arity, 1, cfg.mutation_rate)
    n_off = cfg.popsize - 1

    goal = _goal_at(cfg, 0)
    target = targets[goal]

    if cfg.resume and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            c = pickle.load(f)
        rng.bit_generator.state = c["rng_state"]
        start_gen = c["gen"]
        parent, p_score, p_hits = c["parent"], c["p_score"], c["p_hits"]
        best_geno, best_hits = c["best"], c["best_hits"]
        solved_gen, evals = c["solved_gen"], c["evals"]
        csv_f = csv_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_f, fieldnames=LOG_FIELDS)
        print(f"[seed {seed}] RESUME from gen {start_gen}/{cfg.generations} "
              f"(hits {p_hits}/{n_patterns})", flush=True)
    else:
        # Step 1 -- random population, select the fittest.
        pop = [cgp.random_genotype(rng, cfg.nodes, n_in, 1, n_funcs, arity)
               for _ in range(cfg.popsize)]
        scored = [cgp.fitness(g, gate_set, in_masks, target, mask, n_in, cfg.fitness)
                  for g in pop]
        i = int(np.argmax([s for s, _ in scored]))
        parent, (p_score, p_hits) = pop[i], scored[i]
        best_geno, best_hits = parent.copy(), p_hits
        start_gen, solved_gen, evals = 0, -1, cfg.popsize
        csv_f = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_f, fieldnames=LOG_FIELDS)
        writer.writeheader()

    draw_frames = cfg.viz and seed < cfg.viz_seeds
    t_interval = time.time()
    t_seed = time.time()

    def log_row(gen: int) -> None:
        nonlocal t_interval
        pheno = cgp.phenotype(parent, n_in, gate_set, split)
        c = pheno.counts()
        span = ((max(1, cfg.switch_interval) if cfg.mvg else cfg.log_interval)
                if gen > start_gen else 1)
        secs = (time.time() - t_interval) / span
        t_interval = time.time()
        writer.writerow(dict(
            seed=seed, gen=gen, goal=goal, score=round(float(p_score), 6),
            hits=p_hits, acc=round(p_hits / n_patterns, 6),
            active_nodes=pheno.n_active, left=c["left"], right=c["right"],
            mixed=c["mixed"], const=c["const"],
            depth=max(pheno.depth.values(), default=0),
            evals=evals, secs_per_gen=round(secs, 6)))
        csv_f.flush()
        print(f"  [seed {seed:3d}] gen {gen:7d} | goal {goal:3s} | "
              f"hits {p_hits:4d}/{n_patterns} ({p_hits / n_patterns:.4f}) | "
              f"active {pheno.n_active:3d} (L{c['left']} R{c['right']} "
              f"M{c['mixed']}) | {secs * 1e3:.2f} ms/gen", flush=True)
        if draw_frames:
            frames.mkdir(parents=True, exist_ok=True)
            viz_mod.draw(parent, pheno, gate_set, n_in,
                         frames / f"seed{seed}_gen{gen:07d}.png",
                         title=viz_mod.frame_title(gen, goal, p_hits, n_patterns,
                                                   pheno, seed), split=split)

    gen = start_gen
    for gen in range(start_gen, cfg.generations):
        new_goal = _goal_at(cfg, gen)
        if new_goal != goal:
            # The goal moved: the parent's stored score refers to the old target and
            # must be recomputed before any comparison against offspring.
            goal, target = new_goal, targets[new_goal]
            p_score, p_hits = cgp.fitness(parent, gate_set, in_masks, target,
                                          mask, n_in, cfg.fitness)
            evals += 1

        if gen == start_gen or (gen % cfg.log_interval == 0 and not cfg.mvg):
            log_row(gen)                        # baseline row before this generation

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

        # Under MVG rows land at the END of each goal epoch, so every row is the same
        # object -- a goal the lineage has had a full epoch to adapt to -- and rows
        # line up with the switches (same convention as experiment 1, ed70189).
        if cfg.mvg and (gen + 1) % max(1, cfg.switch_interval) == 0:
            log_row(gen)

        if cfg.checkpoint_interval and (gen + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(ckpt_path, gen + 1, rng, parent, p_score, p_hits,
                            best_geno, best_hits, solved_gen, evals)

        if cfg.stop_on_solution and p_hits == n_patterns:
            break

    log_row(gen)                                # endpoint row
    csv_f.close()

    # The final circuit of EVERY seed is drawn, even when per-log frames are capped.
    pheno = cgp.phenotype(parent, n_in, gate_set, split)
    if cfg.viz:
        frames.mkdir(parents=True, exist_ok=True)
        viz_mod.draw(parent, pheno, gate_set, n_in,
                     frames / f"seed{seed}_final.png",
                     title=viz_mod.frame_title(gen, goal, p_hits, n_patterns,
                                               pheno, seed), split=split)
    if cfg.save_best:
        np.savez(out / f"{run}_seed{seed}_best.npz", **asdict(best_geno))

    c = pheno.counts()
    result = dict(seed=seed, best_hits=int(best_hits),
                  best_acc=best_hits / n_patterns, final_hits=int(p_hits),
                  solved_gen=int(solved_gen), gens_run=int(gen + 1),
                  evals=int(evals), active_nodes=pheno.n_active,
                  left=c["left"], right=c["right"], mixed=c["mixed"],
                  seconds=round(time.time() - t_seed, 2))
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink()          # finished -> the checkpoint is no longer needed
    return result


def main(argv=None) -> int:
    cfg = parse(argv)
    gate_set = gates_mod.build_set(cfg.gates)
    n_in = tasks_mod.n_inputs(cfg.task)
    n_patterns = tasks_mod.n_patterns(cfg.task)
    mask = tasks_mod.full_mask(cfg.task)
    in_masks = tasks_mod.input_masks(cfg.task)
    # inputs 0..3 are the left retina block, 4..7 the right one; used only to colour
    # and classify nodes, never by the search itself
    split = n_in // 2 if tasks_mod.uses_operation(cfg.task) else None

    ops = cfg.mvg_ops if cfg.mvg else (cfg.operation,)
    targets = {op: tasks_mod.target_mask(cfg.task, op) for op in ops}

    arity = gates_mod.max_arity(gate_set)
    n_mut = cgp.n_mutations(cfg.nodes, arity, 1, cfg.mutation_rate)

    mode = f"mvg-{'-'.join(cfg.mvg_ops)}" if cfg.mvg else f"fg-{cfg.operation}"
    run = f"cgp_{cfg.task}_{mode}_n{cfg.nodes}_m{cfg.mutation_rate}"
    name = f"{run}_g{cfg.generations}" + (f"_{cfg.tag}" if cfg.tag else "")
    out = pathlib.Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"experiment 4 -- CGP | {name}")
    print(f"  task {cfg.task} ({n_in} inputs, {n_patterns} patterns) | "
          f"gates [{cfg.gates}] arity {arity} | fitness {cfg.fitness}")
    print(f"  {cfg.nodes} nodes, {cgp.n_gene_slots(cfg.nodes, arity, 1)} gene slots, "
          f"{n_mut} mutated/application | (1+{cfg.popsize - 1}) ES")
    print(f"  {cfg.n_seeds} seeds x {cfg.generations} generations | "
          f"ckpt every {cfg.checkpoint_interval or '-'} | resume {cfg.resume} | "
          f"viz {cfg.viz} (first {cfg.viz_seeds} seed(s))")
    print(f"  -> {out}", flush=True)

    summary = []
    t0 = time.time()
    for k in range(cfg.n_seeds):
        summary.append(run_seed(cfg, cfg.seed + k, gate_set, in_masks, mask, n_in,
                                n_patterns, targets, split, out, run))

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    accs = [s["best_acc"] for s in summary]
    solved = [s for s in summary if s["solved_gen"] >= 0]
    line = (f"\ndone in {time.time() - t0:.1f}s | best acc {np.mean(accs):.4f} "
            f"+/- {np.std(accs):.4f} (max {max(accs):.4f}) | "
            f"solved {len(solved)}/{cfg.n_seeds}")
    if solved:
        g = [s["solved_gen"] for s in solved]
        line += f" | median gens-to-solve {int(np.median(g))}"
    print(line + f" | -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
