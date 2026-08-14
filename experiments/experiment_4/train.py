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
import multiprocessing as mp
import os
import pathlib
import pickle
import random
import sys
import time
from dataclasses import asdict

import numpy as np

import cgp
import ecgp
import gates as gates_mod
import tasks as tasks_mod
import visualize as viz_mod
from config import RunConfig, parse

# `n_modules` / `module_nodes` are 0 for every CGP run and only move under --ecgp;
# they are in the shared schema so one analysis script reads both arms.
LOG_FIELDS = ["seed", "gen", "goal", "score", "hits", "acc", "active_nodes",
              "left", "right", "mixed", "const", "depth", "n_modules",
              "module_nodes", "evals", "secs_per_gen"]

# One row per goal epoch under --mvg: how far the lineage fell when the goal moved
# and how long it took to climb back. Written as its own file rather than as columns
# on the main log because it is EVENT-driven -- a row exists per switch, not per
# --log-interval -- so recovery times are exact and do not depend on sampling.
RECOVERY_FIELDS = ["seed", "epoch", "gen", "goal", "hits_before", "hits_after",
                   "drop", "gens_to_recover", "censored"]

# What the circuit is BUILT FROM, one row per distinct gate per log point: the
# primitives by name (AND, OR, ...) and, under ECGP, each module by its `M<k>` name.
# Long format rather than columns because the set of names is open -- modules are
# created and destroyed during the run, and a name is never reused (see
# `ecgp.module_name`), so a wide table would need a column per module ever created.
# `nodes`/`ins`/`outs` travel with the row because `M7` means nothing on its own.
GATE_FIELDS = ["seed", "gen", "goal", "gate", "count", "copies", "kind",
               "nodes", "ins", "outs", "label", "sig", "expr"]

# Below this estimated total runtime, spawning processes costs more than it saves.
# Windows has no fork: every worker starts a fresh interpreter and re-imports
# matplotlib, which measures ~0.5-1 s each.
_SPAWN_BUDGET_SECS = 8.0


def _physical_cores() -> int:
    """Physical cores, not logical.

    SMT siblings are excluded on purpose: this loop is integer- and branch-bound
    with a working set of a few kilobytes, so there are no memory stalls for a
    second thread to fill and the sibling mostly adds contention. Measured on a
    16-physical/22-logical machine, oversubscribing past the physical count did not
    improve throughput.
    """
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    n = os.cpu_count() or 1
    return max(1, n // 2) if n > 4 else n      # assume SMT when psutil is absent


# Measured cost of one full generation of `run_seed` -- mutate + evaluate 4
# offspring + selection -- on the retina with AND/NAND/OR/NOR, logging off. These
# are wall-clock numbers from the real loop, NOT an isolated microbenchmark: an
# earlier model fitted to a microbenchmark underestimated by 4x and wrongly sent a
# 25 s job down the serial path.
#
# Repeated measurement varies by up to ~2x with machine load and clock boost, and
# these anchors are the SLOW end of that band on purpose: over-estimating the job
# only risks spawning workers for a job that would have been fine serially, while
# under-estimating strands a long job on one core. Re-measure if the hot loop
# changes.
_GEN_COST_MS = ((100, 0.239), (400, 0.893), (800, 1.071))


def estimated_secs_per_gen(nodes: int) -> float:
    """Cost of one generation, interpolated over `_GEN_COST_MS`.

    Used only to decide whether a job is too small to be worth spawning processes
    for -- never for reporting. Piecewise-linear between the measured anchors, and
    linear extrapolation from the nearest pair outside them; the growth is
    sublinear in `nodes` (active-node count grows far slower than genotype length),
    so a single slope would misestimate at both ends.
    """
    pts = _GEN_COST_MS
    if nodes <= pts[0][0]:
        lo, hi = pts[0], pts[1]
    elif nodes >= pts[-1][0]:
        lo, hi = pts[-2], pts[-1]
    else:
        lo, hi = next((pts[i], pts[i + 1]) for i in range(len(pts) - 1)
                      if pts[i][0] <= nodes <= pts[i + 1][0])
    frac = (nodes - lo[0]) / (hi[0] - lo[0])
    return max(1e-6, 1e-3 * (lo[1] + frac * (hi[1] - lo[1])))


def plan_workers(cfg: RunConfig, cores: int | None = None) -> int:
    """How many seeds to run concurrently. Adaptive, not a fixed number.

    Seeds are the only parallel axis available. Generations are serial by
    construction (generation t+1's parent is generation t's outcome) and a single
    offspring is ~0.1 ms of work -- far below process-dispatch cost -- so neither
    can be parallelised usefully. See RESULTS.md for the measurements.

    Policy, in order:
      1. An explicit `--workers N` is obeyed, but still capped at `n_seeds`: a
         worker with no seed to run is pure overhead.
      2. Auto caps at `n_seeds` and at the machine's PHYSICAL core count.
      3. Auto falls back to serial (1) when the whole job is estimated to take less
         than `_SPAWN_BUDGET_SECS`, because interpreter startup would dominate.

    This adapts to the actual run: 1 long seed stays serial, 50 short seeds at 100
    nodes go parallel, 4 seeds on a 16-core box use 4 workers rather than 16.
    """
    if cores is None:
        cores = _physical_cores()
    if cfg.workers >= 1:
        return max(1, min(cfg.workers, cfg.n_seeds))
    if cfg.n_seeds <= 1:
        return 1
    est = cfg.n_seeds * cfg.generations * estimated_secs_per_gen(cfg.nodes)
    if est < _SPAWN_BUDGET_SECS:
        return 1
    return max(1, min(cores, cfg.n_seeds))


def _goal_at(cfg: RunConfig, gen: int) -> str:
    """The active operation at generation `gen`."""
    if not cfg.mvg:
        return cfg.operation
    return cfg.mvg_ops[(gen // cfg.switch_interval) % len(cfg.mvg_ops)]


def _even_picks(seq: list, k: int) -> list:
    """`k` roughly evenly spaced items of `seq`, always including the first and last."""
    if len(seq) <= k:
        return list(seq)
    idx = sorted({round(i * (len(seq) - 1) / (k - 1)) for i in range(k)})
    return [seq[i] for i in idx]


def _module_caption(rows: list[dict]) -> str:
    """The module half of a gate histogram, for a diagram caption ("" under CGP).

    Worth carrying into the picture because the picture cannot show it: diagrams are
    drawn on the flattened circuit, where the modules have already been inlined.
    """
    mods = [f"{r['gate']}x{r['count']}" for r in rows if r["kind"] == "module"]
    return ("modules " + " ".join(mods)) if mods else ""


def _spearman(ys: list[float]) -> float:
    """Rank correlation of `ys` against its own index, i.e. "is this trending?".

    Spearman rather than Pearson because recovery times are heavily right-skewed and
    a single slow epoch would dominate a least-squares slope. Negative = recovery is
    getting FASTER as epochs go by, which is the MVG prediction. Ties get average
    ranks. Returns 0.0 when there is nothing to correlate.
    """
    n = len(ys)
    if n < 3:
        return 0.0
    order = sorted(range(n), key=lambda i: ys[i])
    rank = [0.0] * n
    i = 0
    while i < n:                                    # average ranks within a tie group
        j = i
        while j + 1 < n and ys[order[j + 1]] == ys[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    mx = (n - 1) / 2.0                              # mean of the index 0..n-1
    my = sum(rank) / n
    num = sum((x - mx) * (r - my) for x, r in enumerate(rank))
    den = (sum((x - mx) ** 2 for x in range(n)) * sum((r - my) ** 2 for r in rank)) ** 0.5
    return round(num / den, 4) if den else 0.0


def save_checkpoint(path: pathlib.Path, gen: int, rnd, parent, p_score, p_hits,
                    best, best_hits, solved_gen, evals, goal,
                    recoveries=None, rec_open=None) -> None:
    """Atomically pickle full search state so a run resumes exactly where it stopped.

    `goal` is part of that state and cannot be recomputed from `gen`: p_score/p_hits
    are scores ON that goal, and the checkpoint is written at the END of generation
    `gen - 1`, so deriving the goal from `gen` would be off by one epoch at exactly
    the boundaries where it matters. Restoring the wrong goal silently diverges the
    search -- it makes the loop skip the switch re-evaluation and compare offspring
    against a stale parent score.
    """
    obj = {"gen": gen, "rng_state": rnd.getstate(),
           "parent": parent, "p_score": p_score, "p_hits": p_hits,
           "best": best, "best_hits": best_hits, "goal": goal,
           "solved_gen": solved_gen, "evals": evals,
           "recoveries": recoveries or [], "rec_open": rec_open}
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)   # atomic: a crash mid-write cannot corrupt the checkpoint


def run_seed(cfg: RunConfig, seed: int, gate_set, in_masks, mask, n_in,
             n_patterns, targets, split, out: pathlib.Path, run: str):
    """One independent run. Returns its result dict, or None if already finished."""
    csv_path = out / f"{run}_seed{seed}_log.csv"
    gate_path = out / f"{run}_seed{seed}_gates.csv"
    ckpt_path = out / f"{run}_seed{seed}_ckpt.pkl"
    result_path = out / f"{run}_seed{seed}_result.json"
    frames = out / "frames"

    if cfg.resume and result_path.exists():
        print(f"[seed {seed}] already finished -> skipping "
              f"(--no-resume to redo)", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    rnd = random.Random(seed)
    arity = gates_mod.max_arity(gate_set)
    n_funcs = len(gate_set)
    n_mut = cgp.n_mutations(cfg.nodes, arity, 1, cfg.mutation_rate)
    n_off = cfg.popsize - 1

    # --- representation backend. The (1+4) loop, the MVG switch handling, the
    # recovery instrumentation and the checkpointing below are IDENTICAL for CGP and
    # ECGP; only these four operations differ. Keeping the difference to four
    # closures is deliberate -- it is what makes the two arms comparable, since any
    # divergence in the search loop would be a confound rather than a treatment.
    #
    #   new()      a fresh random individual
    #   offspring  one mutated child of the parent
    #   score      (selection score, hits) against a given target
    #   as_cgp     the individual as a plain CGP genotype, for the structural
    #              readout and the diagrams: identity for CGP, `flatten` for ECGP,
    #              so both arms are measured by exactly the same `cgp.phenotype`.
    if cfg.ecgp:
        eparams = ecgp.Params(
            compress=cfg.compress_prob, expand=cfg.expand_prob,
            module_point=cfg.module_point_prob, add_input=cfg.add_input_prob,
            remove_input=cfg.remove_input_prob, add_output=cfg.add_output_prob,
            remove_output=cfg.remove_output_prob,
            max_module_size=cfg.max_module_size, mutation_rate=cfg.mutation_rate)
        def new():
            return ecgp.random_individual(rnd, cfg.nodes, n_in, 1, n_funcs)
        def offspring(par):
            return ecgp.mutate(par, rnd, n_in, n_funcs, eparams)
        def score(g, tgt):
            return ecgp.fitness(g, gate_set, in_masks, tgt, mask, n_in, cfg.fitness)
        def as_cgp(g):
            return ecgp.flatten(g, n_in)
        def with_origin(g):
            return ecgp.flatten_with_origin(g, n_in, n_funcs)
    else:
        def new():
            return cgp.random_genotype(rnd, cfg.nodes, n_in, 1, n_funcs, arity)
        def offspring(par):
            return cgp.mutate(par, rnd, n_mut, n_in, n_funcs, cfg.wiring_weight)
        def score(g, tgt):
            return cgp.fitness(g, gate_set, in_masks, tgt, mask, n_in, cfg.fitness)
        def as_cgp(g):
            return g
        def with_origin(g):
            return g, None

    goal = _goal_at(cfg, 0)
    target = targets[goal]

    if cfg.resume and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            c = pickle.load(f)
        rnd.setstate(c["rng_state"])
        start_gen = c["gen"]
        parent, p_score, p_hits = c["parent"], c["p_score"], c["p_hits"]
        best_geno, best_hits = c["best"], c["best_hits"]
        solved_gen, evals = c["solved_gen"], c["evals"]
        recoveries, rec_open = c.get("recoveries", []), c.get("rec_open")
        goal = c["goal"]                    # the goal p_score/p_hits were scored on
        target = targets[goal]
        csv_f = csv_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_f, fieldnames=LOG_FIELDS)
        gate_f = gate_path.open("a", newline="", encoding="utf-8")
        gate_w = csv.DictWriter(gate_f, fieldnames=GATE_FIELDS)
        print(f"[seed {seed}] RESUME from gen {start_gen}/{cfg.generations} "
              f"(hits {p_hits}/{n_patterns})", flush=True)
    else:
        # Step 1 -- random population, select the fittest.
        pop = [new() for _ in range(cfg.popsize)]
        scored = [score(g, target) for g in pop]
        i = max(range(len(scored)), key=lambda k: scored[k][0])
        parent, (p_score, p_hits) = pop[i], scored[i]
        best_geno, best_hits = parent.copy(), p_hits
        start_gen, solved_gen, evals = 0, -1, cfg.popsize
        recoveries, rec_open = [], None
        csv_f = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        gate_f = gate_path.open("w", newline="", encoding="utf-8")
        gate_w = csv.DictWriter(gate_f, fieldnames=GATE_FIELDS)
        gate_w.writeheader()

    draw_frames = cfg.viz and seed < cfg.viz_seeds
    t_interval = time.time()
    t_seed = time.time()

    # Snapshots for the stage grid: the run's own history, one seed followed through
    # generational time. Kept as cheap genotype copies and rendered ONCE at the end --
    # drawing a PNG at every log row instead is what made a 300k-generation run take
    # minutes of matplotlib. The list is decimated as it grows, so a run of any length
    # ends with ~9-18 evenly spaced snapshots for ~9 renders.
    collect = cfg.grid and cfg.viz and seed == cfg.grid_seed
    stages: list[tuple] = []
    stride = [1]
    n_seen = [0]

    def gate_rows(view, pheno) -> list[dict]:
        """What the circuit is built from: one row per distinct gate, per log point.

        `count` is instances in the ACTIVE circuit -- "used" in the literal sense.
        `copies` is instances anywhere in the genotype, active or not; the gap between
        them is the inactive reservoir CGP drifts through, and a module can sit at
        `count 0, copies 1` for a long time before selection finds a use for it.

        Under ECGP this is counted in the individual's OWN node space, not the
        flattened one: flattening inlines a module into ordinary gates, dissolving
        exactly the identity being counted. Every module in the list gets a row even at
        `count 0` -- the question "what was M73 doing?" has to be answerable for a
        module that exists, not only for one currently wired in.

        `label` / `sig` / `expr` come from `ecgp.module_info`: what the thing computes,
        a canonical key for it, and its body. Primitives are described by the same
        columns (via `ecgp.primitive_module`) so a module can be compared against a
        plain gate directly -- a module whose `sig` equals AND's is a re-encapsulated
        AND, which is worth knowing.
        """
        counts: dict[str, list[int]] = {}                 # name -> [active, copies]
        shape: dict[str, tuple] = {}

        def describe(j: int):
            """Node `j` as (display name, kind, the Module describing what it computes)."""
            if cfg.ecgp:
                f = parent.func[j]
                if parent.ntype[j]:
                    return ecgp.module_name(f, n_funcs), "module", parent.modules[f]
            else:
                f = view.func[j]
            return gate_set[f].name.upper(), "prim", ecgp.primitive_module(f)

        def note(name: str, kind: str, mod) -> list[int]:
            if name not in shape:
                label, sig, expr = ecgp.module_info(mod, gate_set)
                shape[name] = (kind, mod.n_nodes, mod.n_in, mod.n_out, label, sig, expr)
            return counts.setdefault(name, [0, 0])

        if cfg.ecgp:
            # every module in the list gets a row, even one no node currently uses
            for mid, m in parent.modules.items():
                note(ecgp.module_name(mid, n_funcs), "module", m)
            genotype = range(parent.n_nodes)
            active = ecgp.active_nodes(parent, n_in)
        else:
            genotype = range(len(view.func))
            active = pheno.active

        for j in genotype:
            note(*describe(j))[1] += 1
        for j in active:
            note(*describe(j))[0] += 1

        # primitives first (alphabetical), then modules in creation order, so a row
        # block reads the same way at every log point
        order = sorted(counts, key=lambda s: (shape[s][0] == "module",
                                              int(s[1:]) if shape[s][0] == "module" else s))
        return [dict(gate=s, count=counts[s][0], copies=counts[s][1],
                     kind=shape[s][0], nodes=shape[s][1], ins=shape[s][2],
                     outs=shape[s][3], label=shape[s][4], sig=shape[s][5],
                     expr=shape[s][6]) for s in order]

    def log_row(gen: int) -> None:
        nonlocal t_interval
        view = as_cgp(parent)               # the effective circuit, modules inlined
        pheno = cgp.phenotype(view, n_in, gate_set, split)
        c = pheno.counts()
        n_mods = len(parent.modules) if cfg.ecgp else 0
        mod_nodes = sum(1 for t in parent.ntype if t) if cfg.ecgp else 0
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
            n_modules=n_mods, module_nodes=mod_nodes,
            evals=evals, secs_per_gen=round(secs, 6)))
        csv_f.flush()
        rows = gate_rows(view, pheno)
        gate_w.writerows(dict(seed=seed, gen=gen, goal=goal, **r) for r in rows)
        gate_f.flush()
        # `M9x2=XOR` reads better than `M9x2`: the whole question about a module is
        # what it computes, and the label is already sitting in the row
        used = " ".join(f"{r['gate']}x{r['count']}"
                        + (f"={r['label']}" if r["label"] and r["kind"] == "module" else "")
                        for r in rows if r["count"])
        print(f"  [seed {seed:3d}] gen {gen:7d} | goal {goal:3s} | "
              f"hits {p_hits:4d}/{n_patterns} ({p_hits / n_patterns:.4f}) | "
              f"active {pheno.n_active:3d} (L{c['left']} R{c['right']} "
              f"M{c['mixed']})"
              + (f" | mods {n_mods} in {mod_nodes} nodes" if cfg.ecgp else "")
              + f" | {used}"
              + f" | {secs * 1e3:.2f} ms/gen", flush=True)
        if collect:
            n_seen[0] += 1
            if (n_seen[0] - 1) % stride[0] == 0:
                stages.append((gen, goal, p_hits, parent.copy()))
                if len(stages) > 18:
                    del stages[1::2]                # halve the resolution, keep the span
                    stride[0] *= 2
        if draw_frames:
            frames.mkdir(parents=True, exist_ok=True)
            _, org = with_origin(parent)        # so per-log frames name modules too
            ttl = viz_mod.frame_title(gen, goal, p_hits, n_patterns, pheno, seed,
                                      alg="ECGP" if cfg.ecgp else "CGP",
                                      mods=_module_caption(rows))
            path = frames / f"seed{seed}_gen{gen:07d}.png"
            if cfg.ecgp:
                viz_mod.draw_modular(parent, n_in, gate_set, n_funcs, path,
                                     title=ttl, split=split)
            else:
                viz_mod.draw(view, pheno, gate_set, n_in, path, title=ttl, split=split,
                             origin=org, colour_cones=cfg.colour_cones)

    gen = start_gen
    for gen in range(start_gen, cfg.generations):
        new_goal = _goal_at(cfg, gen)
        if new_goal != goal:
            # The goal moved: the parent's stored score refers to the old target and
            # must be recomputed before any comparison against offspring.
            hits_before = p_hits            # its level on the goal it just mastered
            goal, target = new_goal, targets[new_goal]
            p_score, p_hits = score(parent, target)
            evals += 1

            # An epoch that ran out before recovering is RIGHT-CENSORED, not dropped
            # and not recorded as "recovered at E". Either shortcut would bias the
            # trend toward exactly the speed-up we are testing for: early epochs fail
            # to recover most often, so discarding them makes the early mean look
            # small, and capping them at E understates how much slower they were.
            if rec_open is not None:
                rec_open["gens_to_recover"] = gen - rec_open["gen"]
                rec_open["censored"] = 1
                recoveries.append(rec_open)
            rec_open = dict(seed=seed, epoch=len(recoveries), gen=gen, goal=goal,
                            hits_before=int(hits_before), hits_after=int(p_hits),
                            drop=int(hits_before - p_hits),
                            gens_to_recover=-1, censored=0)
            if p_hits >= hits_before:
                # No drop: this genotype already satisfied the new goal at least as
                # well. Real data, not a degenerate case -- it is what a lineage
                # carrying a reusable module is expected to look like.
                rec_open["gens_to_recover"] = 0
                recoveries.append(rec_open)
                rec_open = None

        if gen == start_gen or (gen % cfg.log_interval == 0 and not cfg.mvg):
            log_row(gen)                        # baseline row before this generation

        # Steps 2-3 -- mutate the winner into offspring.
        kids = [offspring(parent) for _ in range(n_off)]
        o_scored = [score(g, target) for g in kids]
        evals += n_off

        # Step 4 -- selection with the neutral tie-break.
        o_scores = [s for s, _ in o_scored]
        top = max(o_scores)
        promoted = False
        if top > p_score:                                        # 4a
            i = o_scores.index(top)
            parent, (p_score, p_hits) = kids[i], o_scored[i]
            promoted = True
        elif top == p_score:                                     # 4b
            ties = [i for i, s in enumerate(o_scores) if s == top]
            i = rnd.choice(ties)
            parent, (p_score, p_hits) = kids[i], o_scored[i]
            promoted = True
        # else 4c: the parent stays

        # PAPER_SPEC section 5: on promotion the module list is pruned to exactly the
        # modules present in the new winner. This is the paper's second-level
        # "copy-or-die" and its stated reason the list stays bounded -- not an
        # optimisation. Every module found only in a losing offspring dies here.
        if promoted and cfg.ecgp:
            ecgp.prune_modules(parent)

        if p_hits > best_hits:
            best_geno, best_hits = parent.copy(), p_hits
        if solved_gen < 0 and p_hits == n_patterns:
            solved_gen = gen

        # Recovered = back to the level held on the goal it was just moved off. The
        # comparison is against `hits_before` rather than a fixed threshold so it
        # measures re-adaptation, not raw difficulty of the new goal.
        if rec_open is not None and p_hits >= rec_open["hits_before"]:
            rec_open["gens_to_recover"] = gen + 1 - rec_open["gen"]
            recoveries.append(rec_open)
            rec_open = None

        # Under MVG rows land at the END of each goal epoch, so every row is the same
        # object -- a goal the lineage has had a full epoch to adapt to -- and rows
        # line up with the switches (same convention as experiment 1, ed70189).
        if cfg.mvg and (gen + 1) % max(1, cfg.switch_interval) == 0:
            log_row(gen)

        if cfg.checkpoint_interval and (gen + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(ckpt_path, gen + 1, rnd, parent, p_score, p_hits,
                            best_geno, best_hits, solved_gen, evals, goal,
                            recoveries, rec_open)

        if cfg.stop_on_solution and p_hits == n_patterns:
            break

    log_row(gen)                                # endpoint row
    csv_f.close()
    gate_f.close()

    # The budget ended mid-epoch: censored, same as a switch arriving too soon.
    if rec_open is not None:
        rec_open["gens_to_recover"] = gen + 1 - rec_open["gen"]
        rec_open["censored"] = 1
        recoveries.append(rec_open)
    if recoveries:
        with (out / f"{run}_seed{seed}_recovery.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RECOVERY_FIELDS)
            w.writeheader()
            w.writerows(recoveries)

    # The final circuit of EVERY seed is drawn, even when per-log frames are capped.
    view = as_cgp(parent)
    pheno = cgp.phenotype(view, n_in, gate_set, split)
    final_gates = gate_rows(view, pheno)
    if cfg.viz:
        frames.mkdir(parents=True, exist_ok=True)
        _, org = with_origin(parent)
        ttl = viz_mod.frame_title(gen, goal, p_hits, n_patterns, pheno, seed,
                                  alg="ECGP" if cfg.ecgp else "CGP",
                                  mods=_module_caption(final_gates))
        if cfg.ecgp:
            viz_mod.draw_modular(parent, n_in, gate_set, n_funcs,
                                 frames / f"seed{seed}_final.png",
                                 title=ttl, split=split)
            # ...and, for the final circuit only, the FLATTENED twin. It is the graph
            # that actually gets evaluated, so it is worth keeping; but it is a
            # secondary view, because inlining is exactly what destroys the module
            # boundaries the run is being read for.
            viz_mod.draw(view, pheno, gate_set, n_in,
                         frames / f"seed{seed}_final_flat.png",
                         title=ttl + "  |  flattened (modules inlined)", split=split,
                         origin=org, colour_cones=cfg.colour_cones)
        else:
            viz_mod.draw(view, pheno, gate_set, n_in,
                         frames / f"seed{seed}_final.png", title=ttl, split=split,
                         origin=org, colour_cones=cfg.colour_cones)
    # The stage grid: six points in ONE seed's history, so the picture reads as
    # evolutionary time rather than as a comparison between independent runs. Six
    # rather than nine because each panel has to stay legible at printed size, laid
    # out 2x3 so the sheet is wider than tall like the circuits inside it.
    if collect:
        if not stages or stages[-1][0] != gen:
            stages.append((gen, goal, p_hits, parent.copy()))
        n_panels = viz_mod.STAGE_ROWS * viz_mod.STAGE_COLS
        panels = []
        for g_, goal_, hits_, snap in _even_picks(stages, n_panels):
            v, org = with_origin(snap)
            ph = cgp.phenotype(v, n_in, gate_set, split)
            # ECGP panels carry the INDIVIDUAL, so the sheet is drawn unflattened like
            # every other ECGP frame; CGP panels carry the genotype, which is the same
            # object either way.
            panels.append((snap if cfg.ecgp else v, ph,
                           viz_mod.stage_title(g_, hits_, n_patterns), org))
        viz_mod.stage_grid(
            panels, gate_set, n_in, out / f"seed{seed}_stages.png",
            title="ECGP" if cfg.ecgp else "CGP",
            n_prim=n_funcs if cfg.ecgp else None,
            split=split, colour_cones=cfg.colour_cones)

    if cfg.save_best:
        # Always the FLATTENED circuit: an npz holds arrays, not a module dict, and
        # the inlined graph is what any downstream consumer (--init-from, analysis,
        # a diagram) actually wants. The module structure lives in the log columns.
        np.savez(out / f"{run}_seed{seed}_best.npz", **asdict(as_cgp(best_geno)))

    c = pheno.counts()
    result = dict(seed=seed, best_hits=int(best_hits),
                  best_acc=best_hits / n_patterns, final_hits=int(p_hits),
                  solved_gen=int(solved_gen), gens_run=int(gen + 1),
                  evals=int(evals), active_nodes=pheno.n_active,
                  left=c["left"], right=c["right"], mixed=c["mixed"],
                  # a string, not a dict: this row also goes into summary.csv, and the
                  # per-log breakdown already lives in the seed's gates.csv
                  gates=" ".join(f"{r['gate']}x{r['count']}"
                                 for r in final_gates if r["count"]),
                  seconds=round(time.time() - t_seed, 2))
    if cfg.ecgp:
        # Genotype length is not constant under ECGP -- compress shrinks it and expand
        # grows it -- so it is reported rather than assumed to be `--nodes`.
        result.update(n_modules=len(parent.modules),
                      module_nodes=sum(1 for t in parent.ntype if t),
                      genotype_nodes=parent.n_nodes,
                      module_sizes=sorted(m.n_nodes for m in parent.modules.values()))
    if recoveries:
        # Uncensored epochs only: a censored row is a lower bound on its recovery
        # time, so mixing it into a median or a correlation understates both.
        clean = [r for r in recoveries if not r["censored"]]
        times = [r["gens_to_recover"] for r in clean]
        half = len(times) // 2
        result.update(
            n_epochs=len(recoveries), n_censored=len(recoveries) - len(clean),
            mean_drop=round(sum(r["drop"] for r in recoveries) / len(recoveries), 2),
            recovery_median=(sorted(times)[len(times) // 2] if times else None),
            recovery_first_half=(round(sum(times[:half]) / half, 2) if half else None),
            recovery_second_half=(round(sum(times[half:]) / (len(times) - half), 2)
                                  if half else None),
            recovery_rho=_spearman([float(t) for t in times]))
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink()          # finished -> the checkpoint is no longer needed
    return result


def _task_context(cfg: RunConfig):
    """Everything `run_seed` needs that is derived from `cfg`.

    Rebuilt inside each worker rather than sent to it: the gate set holds lambdas,
    which pickle cannot serialise. Only `cfg` (plain fields) crosses the process
    boundary, so this is also the cheapest thing to send.
    """
    gate_set = gates_mod.build_set(cfg.gates)
    n_in = tasks_mod.n_inputs(cfg.task)
    ops = cfg.mvg_ops if cfg.mvg else (cfg.operation,)
    return dict(
        gate_set=gate_set, n_in=n_in,
        n_patterns=tasks_mod.n_patterns(cfg.task),
        mask=tasks_mod.full_mask(cfg.task),
        in_masks=tasks_mod.input_masks(cfg.task),
        targets={op: tasks_mod.target_mask(cfg.task, op) for op in ops},
        # inputs 0..3 are the left retina block, 4..7 the right one; used only to
        # colour and classify nodes, never by the search itself
        split=n_in // 2 if tasks_mod.uses_operation(cfg.task) else None,
    )


def _seed_task(payload):
    """Pool entry point. Must be module-level and take picklable arguments only."""
    cfg, seed, out_str, run = payload
    ctx = _task_context(cfg)
    return run_seed(cfg, seed, ctx["gate_set"], ctx["in_masks"], ctx["mask"],
                    ctx["n_in"], ctx["n_patterns"], ctx["targets"], ctx["split"],
                    pathlib.Path(out_str), run)


def main(argv=None) -> int:
    cfg = parse(argv)
    ctx = _task_context(cfg)
    gate_set, n_in = ctx["gate_set"], ctx["n_in"]
    n_patterns, mask, in_masks = ctx["n_patterns"], ctx["mask"], ctx["in_masks"]
    targets, split = ctx["targets"], ctx["split"]

    arity = gates_mod.max_arity(gate_set)
    n_mut = cgp.n_mutations(cfg.nodes, arity, 1, cfg.mutation_rate)

    mode = f"mvg-{'-'.join(cfg.mvg_ops)}" if cfg.mvg else f"fg-{cfg.operation}"
    alg = "ecgp" if cfg.ecgp else "cgp"
    run = f"{alg}_{cfg.task}_{mode}_n{cfg.nodes}_m{cfg.mutation_rate}"
    if cfg.wiring_weight != 1.0:        # keep paper-default runs named as before
        run += f"_w{cfg.wiring_weight}"
    name = f"{run}_g{cfg.generations}" + (f"_{cfg.tag}" if cfg.tag else "")
    out = pathlib.Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"experiment 4 -- {alg.upper()} | {name}")
    print(f"  task {cfg.task} ({n_in} inputs, {n_patterns} patterns) | "
          f"gates [{cfg.gates}] arity {arity} | fitness {cfg.fitness}")
    # share of mutations landing on a function gene: weight 1 each against
    # `wiring_weight` for each of the n*arity input genes and the output gene
    p_func = cfg.nodes / (cfg.nodes + cfg.wiring_weight * (cfg.nodes * arity + 1))
    print(f"  {cfg.nodes} nodes, {cgp.n_gene_slots(cfg.nodes, arity, 1)} gene slots, "
          f"{n_mut} mutated/application | (1+{cfg.popsize - 1}) ES")
    print(f"  wiring-weight {cfg.wiring_weight} -> {100 * p_func:.0f}% of mutations "
          f"hit a function gene, {100 * (1 - p_func):.0f}% rewire "
          f"({n_mut * (1 - p_func):.2f} rewires/offspring)")
    if cfg.ecgp:
        print(f"  ECGP: compress {cfg.compress_prob} / expand {cfg.expand_prob} | "
              f"module point {cfg.module_point_prob} | "
              f"+in {cfg.add_input_prob} -in {cfg.remove_input_prob} "
              f"+out {cfg.add_output_prob} -out {cfg.remove_output_prob} | "
              f"ms {cfg.max_module_size} | module list starts EMPTY")
    workers = plan_workers(cfg)
    print(f"  {cfg.n_seeds} seeds x {cfg.generations} generations | "
          f"ckpt every {cfg.checkpoint_interval or '-'} | resume {cfg.resume} | "
          f"viz {cfg.viz} (first {cfg.viz_seeds} seed(s))")
    print(f"  {workers} worker process(es) over seeds "
          f"({_physical_cores()} physical cores; generations are serial by "
          f"construction)")
    print(f"  -> {out}", flush=True)

    t0 = time.time()
    seeds = [cfg.seed + k for k in range(cfg.n_seeds)]
    if workers == 1:
        summary = [run_seed(cfg, s, gate_set, in_masks, mask, n_in,
                            n_patterns, targets, split, out, run) for s in seeds]
    else:
        payloads = [(cfg, s, str(out), run) for s in seeds]
        # imap preserves input order, so `summary` is seed-ordered either way.
        with mp.Pool(processes=workers) as pool:
            summary = list(pool.imap(_seed_task, payloads))
    wall = time.time() - t0

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    accs = [s["best_acc"] for s in summary]
    solved = [s for s in summary if s["solved_gen"] >= 0]
    line = (f"\ndone in {wall:.1f}s | best acc {np.mean(accs):.4f} "
            f"+/- {np.std(accs):.4f} (max {max(accs):.4f}) | "
            f"solved {len(solved)}/{cfg.n_seeds}")
    if solved:
        g = [s["solved_gen"] for s in solved]
        line += f" | median gens-to-solve {int(np.median(g))}"
    print(line + f" | -> {out}")

    # The 3x3 stage grid, written by `run_seed` for --grid-seed. Opened here rather
    # than there so a parallel run does not have every worker racing to open a window.
    gp = out / f"seed{cfg.grid_seed}_stages.png"
    if cfg.grid and cfg.viz and gp.exists():
        print(f"  stages -> {gp}")
        try:
            os.startfile(str(gp))       # Windows only; a remote/headless run just skips
        except Exception as e:
            print(f"  (not opened automatically: {e})")

    # Parallel efficiency, reported so an under-utilised run is visible rather than
    # assumed. Each seed is single-threaded and CPU-bound, so its wall time is
    # essentially its CPU time; summing them gives the total core-seconds of useful
    # work done inside `wall` seconds across `workers` cores.
    busy = sum(s["seconds"] for s in summary)
    if wall > 0 and workers > 0:
        util = busy / (wall * workers)
        print(f"  cpu: {busy:.1f} core-s of work in {wall:.1f}s wall x {workers} "
              f"workers = {100 * util:.0f}% utilisation "
              f"(speedup {busy / wall:.1f}x over serial)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
