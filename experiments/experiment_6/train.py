"""SMCGP (1+4) evolutionary strategy on the even-parity curriculum -- experiment 6.

The loop is PAPER_SPEC.md section IV, verbatim except where marked:

  1. bootstrap `cfg.bootstrap` random genotypes, score all, keep the fittest as parent
  2. mutate the parent into `popsize - 1` offspring
  3. score all offspring; promote the best if it beats the parent; on a TIE, promote
     a random tied offspring [our choice -- neutral drift, the same tie-break
     experiment_4/5's CGP loop uses; the paper's own (1+4) ES does not state one]
  4. repeat until the whole curriculum (2..max_inputs) is solved by one genotype, or
     the evaluation budget runs out

FITNESS (PAPER_SPEC.md section VI-A): for each test-case size n_in = 2..max_inputs,
in order, develop the genotype for (n_in - 2) iterations, evaluate the resulting
phenotype exhaustively, and add its hit count to the running score. The FIRST
test case a genotype fails ends its scoring for that call -- larger sizes are never
attempted, which is both the paper's growth-forcing curriculum and the reason
scoring stays cheap even near `--max-inputs 20`.

OUTPUT LAYOUT mirrors experiment_4/5:
    runs/<name>/<run>_seed<k>_log.csv
    runs/<name>/<run>_seed<k>_ckpt.pkl      (deleted on completion)
    runs/<name>/<run>_seed<k>_result.json

Run:  conda run -n lndp python train.py --max-inputs 6 --max-evals 20000
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import pickle
import random
import sys
import time
from dataclasses import asdict

import gates as gates_mod
import smcgp
import tasks as tasks_mod
from config import RunConfig, parse

LOG_FIELDS = ["seed", "gen", "score", "hits", "patterns", "solved_upto",
             "genotype_nodes", "phenotype_nodes", "evals", "secs_per_gen"]


def _task_cache(max_inputs: int) -> dict[int, tuple[list[int], int, int]]:
    """(in_masks, mask, target) per test-case size, built once -- cheap (<= 20
    entries) and shared across every individual scored in the run."""
    return {n: (tasks_mod.input_masks(n), tasks_mod.full_mask(n),
               tasks_mod.target_parity(n))
           for n in range(2, max_inputs + 1)}


def fitness(genotype: smcgp.Genotype, ftable, n_funcs: int, cfg: RunConfig,
           cache: dict, rnd: random.Random) -> tuple[float, int, int, int, int]:
    """(score, total_hits, total_patterns, solved_upto, last_phenotype_nodes).

    `solved_upto` is the largest n_in fully solved (1 if none, i.e. even the
    2-input case failed). `last_phenotype_nodes` is the developed phenotype size
    at the last test case attempted -- the structural readout for "is it growing".
    """
    total_hits = total_patterns = 0
    solved_upto = 1
    phen_nodes = genotype.n_nodes
    for n_in in range(2, cfg.max_inputs + 1):
        in_masks, mask, target = cache[n_in]
        phen = smcgp.develop(genotype, ftable, 1, n_in - 2, cfg.todo_cap,
                             n_funcs, cfg.addr_max, cfg.param_range, rnd)
        phen_nodes = len(phen)
        out = smcgp.evaluate(phen, ftable, 1, in_masks, mask)
        n_patterns = 1 << n_in
        if out is None:
            break
        h = smcgp.hits(out[0], target, mask)
        total_hits += h
        total_patterns += n_patterns
        if h == n_patterns:
            solved_upto = n_in
        else:
            break
    return float(total_hits), total_hits, total_patterns, solved_upto, phen_nodes


def save_checkpoint(path: pathlib.Path, gen: int, rnd, parent, p_score, p_stats,
                    best, best_stats, solved_gen, evals) -> None:
    obj = dict(gen=gen, rng_state=rnd.getstate(), parent=parent, p_score=p_score,
              p_stats=p_stats, best=best, best_stats=best_stats,
              solved_gen=solved_gen, evals=evals)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def run_seed(cfg: RunConfig, seed: int, ftable, n_funcs: int, cache: dict,
            out: pathlib.Path, run: str) -> dict:
    csv_path = out / f"{run}_seed{seed}_log.csv"
    ckpt_path = out / f"{run}_seed{seed}_ckpt.pkl"
    result_path = out / f"{run}_seed{seed}_result.json"

    if cfg.resume and result_path.exists():
        print(f"[seed {seed}] already finished -> skipping (--no-resume to redo)",
             flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    rnd = random.Random(seed)

    if cfg.resume and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            c = pickle.load(f)
        rnd.setstate(c["rng_state"])
        start_gen = c["gen"]
        parent, p_score, p_stats = c["parent"], c["p_score"], c["p_stats"]
        best_geno, best_stats = c["best"], c["best_stats"]
        solved_gen, evals = c["solved_gen"], c["evals"]
        csv_f = csv_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_f, fieldnames=LOG_FIELDS)
        print(f"[seed {seed}] RESUME from gen {start_gen} "
             f"(solved up to {p_stats[2]}-input)", flush=True)
    else:
        pop = [smcgp.random_genotype(rnd, cfg.nodes, n_funcs, cfg.addr_max,
                                     cfg.param_range) for _ in range(cfg.bootstrap)]
        scored = [fitness(g, ftable, n_funcs, cfg, cache, rnd) for g in pop]
        i = max(range(len(scored)), key=lambda k: scored[k][0])
        parent = pop[i]
        p_score, p_stats = scored[i][0], scored[i][1:]
        best_geno, best_stats = parent.copy(), p_stats
        start_gen, solved_gen, evals = 0, -1, cfg.bootstrap
        csv_f = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_f, fieldnames=LOG_FIELDS)
        writer.writeheader()

    def log_row(gen: int, secs: float) -> None:
        hits, patterns, solved_upto, phen_nodes = p_stats
        writer.writerow(dict(
            seed=seed, gen=gen, score=round(p_score, 6), hits=hits,
            patterns=patterns, solved_upto=solved_upto,
            genotype_nodes=parent.n_nodes, phenotype_nodes=phen_nodes,
            evals=evals, secs_per_gen=round(secs, 6)))
        csv_f.flush()
        print(f"  [seed {seed:3d}] gen {gen:6d} | solved up to {solved_upto:2d}-input "
             f"| hits {hits}/{patterns} | phenotype {phen_nodes} nodes "
             f"(genotype {parent.n_nodes}) | evals {evals} | {secs * 1e3:.2f} ms/gen",
             flush=True)

    t_interval = time.time()
    t_seed = time.time()
    gen = start_gen
    while evals < cfg.max_evals:
        if gen == start_gen or gen % cfg.log_interval == 0:
            log_row(gen, (time.time() - t_interval) / max(1, cfg.log_interval))
            t_interval = time.time()

        kids = [smcgp.mutate(parent, rnd, n_funcs, cfg.addr_max, cfg.param_range,
                             cfg.mutation_rate, cfg.param_randomize_prob, cfg.sigma)
               for _ in range(cfg.popsize - 1)]
        o_scored = [fitness(k, ftable, n_funcs, cfg, cache, rnd) for k in kids]
        evals += cfg.popsize - 1

        o_scores = [s[0] for s in o_scored]
        top = max(o_scores)
        if top > p_score:
            i = o_scores.index(top)
            parent, p_score, p_stats = kids[i], o_scored[i][0], o_scored[i][1:]
        elif top == p_score:
            ties = [i for i, s in enumerate(o_scores) if s == top]
            i = rnd.choice(ties)
            parent, p_score, p_stats = kids[i], o_scored[i][0], o_scored[i][1:]

        if p_stats[2] > best_stats[2] or (p_stats[2] == best_stats[2]
                                          and p_stats[0] > best_stats[0]):
            best_geno, best_stats = parent.copy(), p_stats
        if solved_gen < 0 and p_stats[2] == cfg.max_inputs:
            solved_gen = gen

        gen += 1
        if cfg.checkpoint_interval and gen % cfg.checkpoint_interval == 0:
            save_checkpoint(ckpt_path, gen, rnd, parent, p_score, p_stats,
                            best_geno, best_stats, solved_gen, evals)
        if solved_gen >= 0:
            break

    log_row(gen, (time.time() - t_interval) / max(1, gen % cfg.log_interval or cfg.log_interval))
    csv_f.close()

    result = dict(seed=seed, best_solved_upto=int(best_stats[2]),
                 best_hits=int(best_stats[0]), best_patterns=int(best_stats[1]),
                 best_phenotype_nodes=int(best_stats[3]),
                 solved_gen=int(solved_gen), gens_run=int(gen), evals=int(evals),
                 genotype_nodes=int(parent.n_nodes),
                 seconds=round(time.time() - t_seed, 2))
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink()
    return result


def main(argv=None) -> int:
    cfg = parse(argv)
    gate_set = gates_mod.build_set(cfg.gates)
    ftable = smcgp.build_function_table(gate_set)
    n_funcs = len(ftable)
    cache = _task_cache(cfg.max_inputs)

    run = f"smcgp_parity_n{cfg.nodes}_m{cfg.mutation_rate}_maxin{cfg.max_inputs}"
    name = run + (f"_{cfg.tag}" if cfg.tag else "")
    out = pathlib.Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"experiment 6 -- SMCGP | {name}")
    print(f"  curriculum 2..{cfg.max_inputs}-input parity"
         + (f"  [paper default is {20}]" if cfg.max_inputs < 20 else ""))
    print(f"  {cfg.nodes} genotype nodes x 7 genes | function table: "
         f"1 INP + {len(gate_set)} gates ({cfg.gates}) + {len(smcgp.SM_OPS)} SM ops "
         f"= {n_funcs} functions")
    print(f"  todo-cap {cfg.todo_cap} | mutation-rate {cfg.mutation_rate} | "
         f"param-randomize {cfg.param_randomize_prob} | sigma {cfg.sigma}")
    print(f"  bootstrap {cfg.bootstrap} | (1+{cfg.popsize - 1}) ES | "
         f"max-evals {cfg.max_evals}"
         + (f"  [paper default is {10_000_000}]" if cfg.max_evals < 10_000_000 else ""))
    print(f"  {cfg.n_seeds} seed(s) | ckpt every {cfg.checkpoint_interval or '-'} | "
         f"resume {cfg.resume}")
    print(f"  -> {out}", flush=True)

    t0 = time.time()
    seeds = [cfg.seed + k for k in range(cfg.n_seeds)]
    summary = [run_seed(cfg, s, ftable, n_funcs, cache, out, run) for s in seeds]
    wall = time.time() - t0

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    solved = [s for s in summary if s["solved_gen"] >= 0]
    best_upto = [s["best_solved_upto"] for s in summary]
    print(f"\ndone in {wall:.1f}s | best solved-up-to {max(best_upto)} "
         f"(mean {sum(best_upto) / len(best_upto):.1f}) | "
         f"fully solved the curriculum {len(solved)}/{cfg.n_seeds} | -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
