"""EXPERIMENTAL. Run necgp_pairwise on several seeds in parallel, saving the circuit's
history so the gate mix can be read at any point in evolution.

    # the search, headless -- PyPy is the fast interpreter at 8 inputs (see README)
    ../../experiment_5/.venv-pypy/Scripts/pypy.exe train.py --seeds 0-4 --tag base
    # the pictures afterwards, CPython only (matplotlib)
    conda run -n lndp python render.py runs/base

THE SEARCH IS `run.py`'s, unchanged: same (1+4) loop, same operator order, same RNG
draws, so a seed here walks the generations `run.py` walks (asserted by
`test_train.py`). Everything this file adds is read-only bookkeeping that never
touches the RNG.

OUTPUT, one directory per seed (`<out>/<tag>/seed<k>/`), every file flushed as it is
written so a run in progress can be inspected:

  config.json      every flag, the interpreter, the git commit
  log.csv          one row per snapshot: accuracy, circuit size, module share
  gates.csv        one row per gate type per snapshot -- NAND and every live module:
                   call counts, top-level and flattened shares, truth-table
                   signature, human label (XOR, A|~B, ...), depth, birth generation
  snapshots.jsonl  the parent genotype at every snapshot, for drawing any stage later
  result.json      solved generation, wall clock, final module report

WHEN A SNAPSHOT IS TAKEN: at generation 0, every `--snapshot-interval` generations,
every time the parent's hits improve (`--no-snapshot-on-improve` turns that off), and
at the end. See `census.py` for what each column means.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pathlib
import platform
import random
import subprocess
import sys
import time

import census as census_mod
import ecgp
import gates as gates_mod
import tasks as tasks_mod

HERE = pathlib.Path(__file__).resolve().parent


def parse_seeds(spec: str) -> list[int]:
    """'0-4' -> [0..4]; '0,3,7' -> [0,3,7]; '2' -> [2]."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # search -- identical names and defaults to run.py
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
    # seeds / parallelism
    ap.add_argument("--seeds", default="0", help="e.g. 0-4 or 0,3,7")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes; 0 = min(#seeds, physical-core guess)")
    # history
    ap.add_argument("--snapshot-interval", type=int, default=1000)
    ap.add_argument("--no-snapshot-on-improve", dest="snapshot_on_improve",
                    action="store_false")
    ap.add_argument("--print-interval", type=int, default=10000)
    ap.add_argument("--out", default=str(HERE / "runs"))
    ap.add_argument("--tag", default="default")
    return ap


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


class _Writer:
    """Append-and-flush CSV writer, so a live run can be read at any time."""

    def __init__(self, path: pathlib.Path, fields: list[str]):
        self.fh = open(path, "w", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=fields)
        self.w.writeheader()

    def rows(self, rows) -> None:
        self.w.writerows(rows)
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def run_seed(seed: int, args: argparse.Namespace, snapshots: bool = True) -> dict:
    gate_set = gates_mod.build_set(args.gates)
    n_prim = len(gate_set)
    n_in = tasks_mod.n_inputs(args.task)
    in_masks = tasks_mod.input_masks(args.task)
    mask = tasks_mod.full_mask(args.task)
    target = tasks_mod.target_mask(args.task, args.operation)
    n_patterns = tasks_mod.n_patterns(args.task)

    # ---- the search setup: run.run_to_solution, verbatim ----
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

    cen = census_mod.Census(gate_set, n_in, n_prim, n_patterns)
    t0 = time.time()
    writers = None
    if snapshots:
        d = pathlib.Path(args.out) / args.tag / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        cfg = dict(vars(args), seed=seed, python=sys.version.split()[0],
                   implementation=platform.python_implementation(),
                   git_commit=_git_commit())
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        writers = (_Writer(d / "log.csv", census_mod.LOG_FIELDS),
                   _Writer(d / "gates.csv", census_mod.CENSUS_FIELDS),
                   open(d / "snapshots.jsonl", "w"))

    def snap(gen: int, reason: str) -> None:
        if writers is None:
            return
        log, rows = cen.take(parent, gen, p_hits, reason, time.time() - t0)
        writers[0].rows([log])
        writers[1].rows(rows)
        writers[2].write(json.dumps(dict(gen=gen, hits=p_hits, reason=reason,
                                         ind=census_mod.to_json(parent))) + "\n")
        writers[2].flush()

    snap(0, "init")
    gen = 0
    solved_gen = None
    n_snaps, last_snap = 1, 0

    # ---- the search loop: run.run_to_solution, verbatim, plus bookkeeping ----
    for gen in range(1, args.max_generations + 1):
        best_child, best_score, best_hits = None, -1.0, -1
        for _ in range(args.popsize - 1):
            child = ecgp.mutate(parent, rnd, n_in, n_prim, p)
            ecgp.prune_modules(child)
            score, hits = ecgp.fitness(child, gate_set, in_masks, target, mask,
                                       n_in, args.fitness)
            if score > best_score:
                best_child, best_score, best_hits = child, score, hits
        improved = False
        if best_score >= p_score:
            improved = best_hits > p_hits
            parent, p_score, p_hits = best_child, best_score, best_hits
            cen.note_births(parent, gen)

        solved = p_hits == n_patterns
        reasons = []
        if improved and args.snapshot_on_improve:
            reasons.append("improve")
        if gen % args.snapshot_interval == 0:
            reasons.append("interval")
        if solved:
            reasons.append("solved")
        if reasons:
            snap(gen, "+".join(reasons))
            n_snaps += 1
            last_snap = gen
        if gen % args.print_interval == 0 or (improved and args.snapshot_on_improve):
            print(f"  [seed={seed}] gen {gen:>7}  hits {p_hits}/{n_patterns}  "
                  f"modules {len(parent.modules)}", flush=True)

        if solved:
            solved_gen = gen
            break

    if solved_gen is None and last_snap != gen:
        snap(gen, "end")
        n_snaps += 1
    elapsed = time.time() - t0

    ecgp.validate(parent, n_in, n_prim)
    result = dict(seed=seed, solved_gen=solved_gen, gens=gen, hits=p_hits,
                  n_patterns=n_patterns, elapsed_s=round(elapsed, 2),
                  gen_per_s=round(gen / max(elapsed, 1e-9), 1),
                  n_snapshots=n_snaps, n_modules=len(parent.modules),
                  implementation=platform.python_implementation())
    if writers is not None:
        for w in writers[:2]:
            w.close()
        writers[2].close()
        (pathlib.Path(args.out) / args.tag / f"seed{seed}" / "result.json").write_text(
            json.dumps(result, indent=2))
    result["parent"] = parent
    return result


def _worker(job: tuple[int, argparse.Namespace]) -> dict:
    seed, args = job
    res = run_seed(seed, args)
    res.pop("parent")
    return res


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seeds = parse_seeds(args.seeds)
    workers = args.workers or min(len(seeds), max(1, (os.cpu_count() or 2) // 2))
    impl = platform.python_implementation()
    print(f"necgp_pairwise train [EXPERIMENTAL] -- {impl} {sys.version.split()[0]} | "
          f"task={args.task}/{args.operation} gates={args.gates} nodes={args.nodes} "
          f"nest_decay={args.nest_decay} ms={args.max_module_size} | seeds {seeds} on "
          f"{workers} worker(s) -> {pathlib.Path(args.out) / args.tag}", flush=True)
    if impl != "PyPy" and tasks_mod.n_inputs(args.task) < 14:
        print("  note: PyPy is faster below ~14 inputs -- see README 'Speed'", flush=True)

    t0 = time.time()
    jobs = [(s, args) for s in seeds]
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = list(pool.imap_unordered(_worker, jobs))
    results.sort(key=lambda r: r["seed"])

    print(f"\n=== {len(results)} seed(s) in {time.time() - t0:.1f}s wall ===")
    for r in results:
        status = f"solved at gen {r['solved_gen']}" if r["solved_gen"] else \
            f"NOT solved ({r['hits']}/{r['n_patterns']})"
        print(f"  seed {r['seed']}: {status}  |  {r['elapsed_s']}s, {r['gen_per_s']} gen/s, "
              f"{r['n_snapshots']} snapshots, {r['n_modules']} modules")


if __name__ == "__main__":
    main()
