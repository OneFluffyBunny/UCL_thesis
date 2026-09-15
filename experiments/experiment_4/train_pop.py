"""CGP circuits searched by Kashtan-Alon's population GA instead of the (1+4) ES.

    python train_pop.py --mvg --switch-interval 2000 --generations 40000 --n-seeds 5

WHY THIS EXISTS. Under the (1+4) ES, modularly varying goals neither build nor
remove modularity at matched accuracy (RESULTS.md, section 2).
KA's mechanism is selection AMONG VARIANTS: in a population, lineages that happen to
be modular recover faster after a switch and take over. A single lineage cannot
express that. This script changes ONLY the search loop; genotype, gate set, task,
fitness and the mutation operator are experiment 4's own (`cgp.py`). `train.py` --
the frozen (1+4) search -- is untouched.

THE GA, mirrored from `kashtan_alon/ga.py::reproduce` (our KA reproduction):
  - population S, the top L by score carried over unchanged (elite strategy)
  - the other S-L are offspring of two elite parents drawn uniformly
  - crossover with probability pc, else the child clones parent A
  - mutation with probability pm per child
  - ranking is a stable sort on score with LATER index first among ties, exactly
    as `np.argsort(fit, kind="stable")[::-1]` does there; offspring are appended
    after the elites, so a child tied with an elite outranks it (neutral drift)

TRANSLATIONS TO CGP -- our choices, not KA's:
  - CROSSOVER is per NODE: each node takes its function gene and input genes from
    A or B (probability 1/2), the analogue of KA's per-destination-neuron choice
    of an entire incoming weight row. The output gene comes from A or B likewise.
    Every connection stays valid, since node j may reference any label < n_in + j
    in both parents.
  - MUTATION is experiment 4's own point mutation (`cgp.mutate`, 3% of gene slots),
    applied with probability pm, instead of KA's single edge edit -- so the
    variation operator per mutation event matches the (1+4) runs being compared.

Outputs match `train.py` (config.json, log.csv, recovery.csv, archive.csv,
result.json), so `analysis/fgmvg_common.Arm` reads a run directory as-is.
`pop_mean_hits` in the archive is the mean over all S individuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import pathlib
import pickle
import random
import time
from types import SimpleNamespace

import cgp
import gates as gates_mod
from train import ARCHIVE_FIELDS, RECOVERY_FIELDS, _goal_at, _task_context

LOG_FIELDS = ["seed", "gen", "goal", "hits", "acc", "pop_mean_hits", "active_nodes",
              "n_elite_at_best", "evals", "secs_per_gen"]


def crossover(a: cgp.Genotype, b: cgp.Genotype, rnd: random.Random) -> cgp.Genotype:
    """Per-node uniform crossover (see module docstring)."""
    ar = a.arity
    func, conn = a.func[:], a.conn[:]
    for j in range(len(func)):
        if rnd.random() < 0.5:
            func[j] = b.func[j]
            conn[j * ar:(j + 1) * ar] = b.conn[j * ar:(j + 1) * ar]
    ogene = [(x if rnd.random() < 0.5 else y) for x, y in zip(a.ogene, b.ogene)]
    return cgp.Genotype(func, [0] * len(func), conn, [0] * len(conn),
                        ogene, [0] * len(ogene), ar)


def rank(scores: list[float]) -> list[int]:
    """Indices best first; ties -> later index first (= KA's reversed stable argsort)."""
    return sorted(range(len(scores)), key=lambda i: (scores[i], i), reverse=True)


def run_seed(cfg, seed: int, out: pathlib.Path, run: str) -> dict:
    ctx = _task_context(cfg)
    gate_set, in_masks, mask, n_in = ctx["gate_set"], ctx["in_masks"], ctx["mask"], ctx["n_in"]
    n_patterns, targets = ctx["n_patterns"], ctx["targets"]
    arity = gates_mod.max_arity(gate_set)
    n_funcs = len(gate_set)
    n_mut = cgp.n_mutations(cfg.nodes, arity, 1, cfg.mutation_rate)
    S, L = cfg.pop, cfg.n_elite

    p = {k: out / f"{run}_seed{seed}_{k}" for k in
         ("log.csv", "recovery.csv", "archive.csv", "result.json", "ckpt.pkl")}
    if p["result.json"].exists():
        print(f"[seed {seed}] already finished -> skipping", flush=True)
        return json.loads(p["result.json"].read_text(encoding="utf-8"))

    def score(g, tgt):
        return cgp.fitness(g, gate_set, in_masks, tgt, mask, n_in, cfg.fitness)

    if p["ckpt.pkl"].exists():
        with open(p["ckpt.pkl"], "rb") as f:
            c = pickle.load(f)
        rnd = random.Random()
        rnd.setstate(c["rng"])
        start, pop, sc, goal = c["gen"], c["pop"], c["sc"], c["goal"]
        recoveries, rec_open, best_hits, solved_gen, evals = (
            c["recoveries"], c["rec_open"], c["best_hits"], c["solved_gen"], c["evals"])
        for k in ("log.csv", "archive.csv"):          # drop rows past the checkpoint
            with p[k].open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            with p[k].open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                                   (LOG_FIELDS if k == "log.csv" else ARCHIVE_FIELDS))
                w.writeheader()
                w.writerows(r for r in rows if int(r["gen"]) < start)
        print(f"[seed {seed}] RESUME from gen {start}", flush=True)
    else:
        rnd = random.Random(seed)
        start, goal = 0, _goal_at(cfg, 0)
        pop = [cgp.random_genotype(rnd, cfg.nodes, n_in, 1, n_funcs, arity)
               for _ in range(S)]
        sc = [score(g, targets[goal]) for g in pop]
        recoveries, rec_open, best_hits, solved_gen, evals = [], None, 0, -1, S
        for k in ("log.csv", "archive.csv"):
            with p[k].open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS if k == "log.csv"
                               else ARCHIVE_FIELDS).writeheader()

    log_f = p["log.csv"].open("a", newline="", encoding="utf-8")
    log_w = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
    arch_f = p["archive.csv"].open("a", newline="", encoding="utf-8")
    arch_w = csv.DictWriter(arch_f, fieldnames=ARCHIVE_FIELDS)
    t0 = t_int = time.time()
    last_log = start

    for gen in range(start, cfg.generations):
        new_goal = _goal_at(cfg, gen)
        if new_goal != goal:
            order = rank([s for s, _ in sc])
            hits_before = sc[order[0]][1]
            goal = new_goal
            sc = [score(g, targets[goal]) for g in pop]      # everyone, on the new goal
            evals += S
            h_after = max(h for _, h in sc)
            if rec_open is not None:
                rec_open.update(gens_to_recover=gen - rec_open["gen"], censored=1)
                recoveries.append(rec_open)
            rec_open = dict(seed=seed, epoch=len(recoveries), gen=gen, goal=goal,
                            hits_before=hits_before, hits_after=h_after,
                            drop=hits_before - h_after, gens_to_recover=-1, censored=0)
            if h_after >= hits_before:
                rec_open["gens_to_recover"] = 0
                recoveries.append(rec_open)
                rec_open = None

        # --- reproduce (kashtan_alon/ga.py::reproduce) -------------------------
        order = rank([s for s, _ in sc])
        elite = order[:L]
        tgt = targets[goal]
        kids, k_sc = [], []
        for _ in range(S - L):
            ia, ib = elite[int(rnd.random() * L)], elite[int(rnd.random() * L)]
            a = pop[ia]
            child = crossover(a, pop[ib], rnd) if rnd.random() < cfg.pc else a
            if rnd.random() < cfg.pm:
                child = cgp.mutate(child, rnd, n_mut, n_in, n_funcs)
            if child is a:                  # an unmutated clone: its score is known
                kids.append(a.copy())
                k_sc.append(sc[ia])
            else:
                kids.append(child)
                k_sc.append(score(child, tgt))
                evals += 1
        pop = [pop[i] for i in elite] + kids
        sc = [sc[i] for i in elite] + k_sc

        order = rank([s for s, _ in sc])
        champ, c_hits = pop[order[0]], sc[order[0]][1]
        pop_mean = sum(h for _, h in sc) / S
        best_hits = max(best_hits, c_hits)
        if solved_gen < 0 and c_hits == n_patterns:
            solved_gen = gen
        if rec_open is not None and c_hits >= rec_open["hits_before"]:
            rec_open["gens_to_recover"] = gen + 1 - rec_open["gen"]
            recoveries.append(rec_open)
            rec_open = None

        if (gen + 1) % cfg.archive_interval == 0:
            arch_w.writerow(dict(gen=gen, goal=goal, hits=c_hits,
                                 pop_mean_hits=round(pop_mean, 4),
                                 func=" ".join(map(str, champ.func)),
                                 conn=" ".join(map(str, champ.conn)),
                                 ogene=" ".join(map(str, champ.ogene))))
        if (gen + 1) % cfg.log_interval == 0:
            ph = cgp.phenotype(champ, n_in, gate_set, n_in // 2)
            secs = (time.time() - t_int) / (gen + 1 - last_log)
            t_int, last_log = time.time(), gen + 1
            n_top = sum(1 for s, _ in sc if s == sc[order[0]][0])
            log_w.writerow(dict(seed=seed, gen=gen, goal=goal, hits=c_hits,
                                acc=round(c_hits / n_patterns, 6),
                                pop_mean_hits=round(pop_mean, 4),
                                active_nodes=ph.n_active, n_elite_at_best=n_top,
                                evals=evals, secs_per_gen=round(secs, 5)))
            log_f.flush()
            arch_f.flush()
            print(f"  [seed {seed}] gen {gen + 1:6d} | {goal:3s} | best {c_hits}/"
                  f"{n_patterns} | mean {pop_mean / n_patterns:.3f} | "
                  f"{ph.n_active} gates | {n_top} at best | {secs * 1e3:.1f} ms/gen",
                  flush=True)
        if cfg.checkpoint_interval and (gen + 1) % cfg.checkpoint_interval == 0:
            log_f.flush()
            arch_f.flush()
            tmp = p["ckpt.pkl"].with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(dict(gen=gen + 1, rng=rnd.getstate(), pop=pop, sc=sc,
                                 goal=goal, recoveries=recoveries, rec_open=rec_open,
                                 best_hits=best_hits, solved_gen=solved_gen,
                                 evals=evals), f)
            tmp.replace(p["ckpt.pkl"])

    log_f.close()
    arch_f.close()
    if rec_open is not None:
        rec_open.update(gens_to_recover=cfg.generations - rec_open["gen"], censored=1)
        recoveries.append(rec_open)
    with p["recovery.csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RECOVERY_FIELDS)
        w.writeheader()
        w.writerows(recoveries)
    order = rank([s for s, _ in sc])
    res = dict(seed=seed, generations=cfg.generations, final_goal=goal,
               final_hits=sc[order[0]][1], best_hits=best_hits, solved_gen=solved_gen,
               evals=evals, seconds=round(time.time() - t0, 1))
    p["result.json"].write_text(json.dumps(res, indent=2), encoding="utf-8")
    p["ckpt.pkl"].unlink(missing_ok=True)
    print(f"[seed {seed}] done: {res}", flush=True)
    return res


def _seed_task(payload):
    cfg, seed, out, run = payload
    return run_seed(cfg, seed, pathlib.Path(out), run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="retina_ka2005")
    ap.add_argument("--operation", default="and")
    ap.add_argument("--gates", default="and,nand,or,nor")
    ap.add_argument("--nodes", type=int, default=50)
    ap.add_argument("--mutation-rate", type=float, default=0.03)
    ap.add_argument("--fitness", default="raw", choices=("raw", "balanced"))
    ap.add_argument("--mvg", action="store_true")
    ap.add_argument("--mvg-ops", default="and,or")
    ap.add_argument("--switch-interval", type=int, default=20)
    ap.add_argument("--pop", type=int, default=600)
    ap.add_argument("--n-elite", type=int, default=150)
    ap.add_argument("--pc", type=float, default=0.5)
    ap.add_argument("--pm", type=float, default=0.5)
    ap.add_argument("--generations", type=int, default=25000)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--archive-interval", type=int, default=1)
    ap.add_argument("--checkpoint-interval", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("runs/pop"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if a.n_elite > a.pop:
        ap.error("--n-elite must be <= --pop")

    cfg = SimpleNamespace(**{k: v for k, v in vars(a).items() if k != "out_dir"})
    cfg.mvg_ops = cfg.mvg_ops.split(",")
    cfg.out_dir = str(a.out_dir)
    cfg.search = "ka_ga"
    arm = f"mvg-{'-'.join(cfg.mvg_ops)}" if a.mvg else f"fg-{a.operation}"
    run = (f"cgppop_{a.task}_{arm}_n{a.nodes}_S{a.pop}L{a.n_elite}_g{a.generations}"
           + (f"_{a.tag}" if a.tag else ""))
    out = a.out_dir / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(cfg), indent=2), encoding="utf-8")
    seeds = list(range(a.seed, a.seed + a.n_seeds))
    payloads = [(cfg, s, str(out), run) for s in seeds]
    print(f"{run}: seeds {seeds}, {min(a.workers, len(seeds))} workers -> {out}", flush=True)
    if a.workers > 1 and len(seeds) > 1:
        with mp.get_context("spawn").Pool(min(a.workers, len(seeds))) as pool:
            results = pool.map(_seed_task, payloads)
    else:
        results = [_seed_task(pl) for pl in payloads]
    for r in results:
        print(f"seed {r['seed']}: final {r['final_goal']} {r['final_hits']} | best "
              f"{r['best_hits']} | solved_gen {r['solved_gen']} | {r['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
