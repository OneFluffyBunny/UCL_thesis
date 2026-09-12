"""CMA-ES training for experiment 2 (direct encoding).

Mirrors experiment_1/train.py — same CMA-ES loop, same accuracy metric, same
optional margin surrogate, same fixed-goal / modularly-varying-goal (--mvg)
switch, same flags (via config.py), the same per-run output directory and the
same champion archive — so results are directly comparable. The ONLY difference
is the model: here the genome is the raw weight vector (model.DirectGenome), not
a compressed DNA grown through a connection rule.

This produces the baseline the genomic bottleneck (exp 1) is measured against:
the best a fixed-topology network can do when every edge is a free parameter,
with no encoding-induced regularity.

Output layout (changed 2026-09-12 to match experiment 1 exactly — one directory
per seed, was flat files in --out-dir):

    <out-dir>/<task>_<arm>[_b<S>s<tau>]_seed<N>/
        log.csv          per-log-interval trace, flushed as it goes
        champions.npz    per-generation champion DNA + accuracy on EVERY goal
        config.json      everything needed to regrow these brains
        result.json      the numbers, with "complete": true as a resume marker
        {best,final,matched,centroid}_dna.eqx / _brain.png

One layout for both encodings means one analysis pipeline for both.

Run `python train.py --help` for all flags.
"""

from __future__ import annotations

import sys
# line-buffer stdout/stderr so logs flush immediately under redirection/background
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import csv
import dataclasses
import json
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import evosax as ex
import numpy as np

import tasks
from model import DirectGenome
from config import parse_args
from visualize import brain_stats, visualize_brain


def _encode(X, encoding):
    return X * 2.0 - 1.0 if encoding == "bipolar" else X


# hinge cap for the margin surrogate (see experiment_1/train.py for the rationale)
_MARGIN_CAP = 0.5


def _balanced_mean(vals, yf):
    """Mean of per-class means of `vals` (balanced; chance-corrected)."""
    pos = jnp.sum(vals * yf) / (jnp.sum(yf) + 1e-8)
    neg = jnp.sum(vals * (1.0 - yf)) / (jnp.sum(1.0 - yf) + 1e-8)
    return 0.5 * (pos + neg)


def _make_eval(static, reshaper, brain_cfg, X_enc, balanced, fitness):
    """Jitted, vmapped eval: flat weights + targets -> (selection_fitness, accuracy).

    Identical to experiment_1's eval except the genome is a DirectGenome.
    `accuracy` is the raw 0/1 (balanced) metric we log / early-stop / report;
    `selection_fitness` is what CMA-ES maximises (== accuracy for fitness=accuracy,
    or a smooth hinged signed-margin surrogate for fitness=margin).
    """
    def eval_genome(flat_params, y):
        genome = eqx.combine(reshaper.reshape_single(flat_params), static)
        yf = y.astype(jnp.float32)
        out = jax.vmap(lambda o: genome.forward(o, brain_cfg))(X_enc)[:, 0]   # raw tanh
        correct = ((out > 0).astype(jnp.int32) == y).astype(jnp.float32)
        acc = _balanced_mean(correct, yf) if balanced else jnp.mean(correct)
        if fitness == "margin":
            s = 2.0 * yf - 1.0
            m = jnp.minimum(out * s, _MARGIN_CAP)
            score = (m + 1.0) / (1.0 + _MARGIN_CAP)
            sel = _balanced_mean(score, yf) if balanced else jnp.mean(score)
        else:
            sel = acc
        return sel, acc
    return jax.jit(jax.vmap(eval_genome, in_axes=(0, None)))


def _git_commit() -> str:
    """Short HEAD hash, so a result can be traced back to the code that made it."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_name_for(brain_cfg, run_cfg, seed) -> str:
    """Directory name for one seed of one arm — same scheme as experiment 1.

    Everything that changes the phenotype has to appear here or two arms write
    the same files: the ARM (an FG and an MVG run of the same task would collide)
    and the synaptic BUDGET (a constrained run would silently clobber an
    unconstrained one, and the only record of the difference is inside the config
    sidecar it just overwrote).
    """
    if tasks.uses_operation(run_cfg.task):
        arm = "mvg-" + "-".join(run_cfg.mvg_ops) if run_cfg.mvg else f"fg-{run_cfg.operation}"
        name = f"{run_cfg.task}_{arm}"
    else:
        name = run_cfg.task
    if brain_cfg.synaptic_budget > 0:
        name += f"_b{brain_cfg.synaptic_budget:g}"
        if brain_cfg.shrink > 0:
            name += f"s{brain_cfg.shrink:g}"
    return f"{name}_seed{seed}"


@dataclasses.dataclass
class SeedResult:
    """Everything one seed produced (mirrors experiment_1.train.SeedResult)."""
    seed: int
    run_name: str
    run_dir: str
    best: float             # best accuracy at ANY generation
    best_genome: object
    final: float            # best accuracy in the LAST generation run
    final_genome: object
    centroid_acc: float     # accuracy of the CMA-ES distribution mean
    centroid_genome: object
    gens_run: int
    wall_s: float
    matched: float = float("nan")   # accuracy of the goal-matched champion
    matched_genome: object = None
    matched_op: str = ""
    matched_gen: int = -1
    acc_by_op: dict = dataclasses.field(default_factory=dict)
    archive_path: str = ""
    archive_n: int = 0


def train_seed(brain_cfg, run_cfg, seed) -> SeedResult:
    """Run one seed; return a SeedResult.

    Four champions, same rationale (and the same goal-matching fix) as
    experiment 1:

    * ``best_*``     -- highest accuracy at ANY generation. Fine under a fixed
      goal; under ``--mvg`` the peak belongs to whichever AND/OR phase happened
      to suit it, so it is not comparable across arms.
    * ``final_*``    -- best member of the LAST generation.
    * ``matched_*``  -- THE ONE TO REPORT UNDER ``--mvg``. The last champion
      selected under the REFERENCE goal (``--operation``), i.e. the goal the
      fixed-goal arm also ran. The switch schedule is deterministic, so a run
      with an even number of epochs always ends mid-OR and ``final_*`` would be
      an OR-selected network reported as an AND result — the bug ``kashtan_alon/``
      hit and fixed. Under a fixed goal this is identical to ``final_*``.
    * ``centroid_*`` -- the CMA-ES distribution MEAN: where the search settled,
      as opposed to the single lucky draw the champions are.

    Every saved champion is additionally scored against EVERY goal in play and
    returned as ``acc_by_op``. With ``--archive-interval > 0`` the champion of
    every Nth generation is kept and written to ``<run_dir>/champions.npz``.
    """
    key = jr.PRNGKey(seed)

    X = tasks.all_binary_inputs(brain_cfg.n_in)
    X_enc = _encode(X, run_cfg.input_encoding)

    key, init_key = jr.split(key)
    template = DirectGenome.init(init_key, brain_cfg)
    params, static = eqx.partition(template, eqx.is_inexact_array)
    reshaper = ex.ParameterReshaper(params)
    batched_eval = _make_eval(static, reshaper, brain_cfg, X_enc, run_cfg.balanced,
                              run_cfg.fitness)

    strategy = ex.Strategies[run_cfg.strategy](
        popsize=run_cfg.popsize, num_dims=reshaper.total_params, sigma_init=run_cfg.sigma_init,
    )
    es_params = strategy.default_params
    key, es_key = jr.split(key)
    state = strategy.initialize(es_key, es_params)
    shaper = ex.FitnessShaper(maximize=True)

    def goal_op(gen):
        if not run_cfg.mvg:
            return run_cfg.operation
        ops = run_cfg.mvg_ops
        return ops[(gen // run_cfg.switch_interval) % len(ops)]

    # Every goal this run could be scored against; the reference goal
    # (--operation) is always included, because it is the only goal an FG arm and
    # an MVG arm share.
    if tasks.uses_operation(run_cfg.task):
        ops_in_play = tuple(dict.fromkeys(
            (run_cfg.mvg_ops if run_cfg.mvg else ()) + (run_cfg.operation,)))
    else:
        ops_in_play = (run_cfg.operation,)
    y_by_op = {op: tasks.targets(run_cfg.task, op, X) for op in ops_in_play}

    def acc_on(flat, op):
        return float(batched_eval(jnp.asarray(flat)[None, :], y_by_op[op])[1][0])

    run_name = run_name_for(brain_cfg, run_cfg, seed)
    run_dir = os.path.join(run_cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[seed {seed}] task={run_cfg.task} op0={goal_op(0)} mvg={run_cfg.mvg} "
          f"balanced={run_cfg.balanced} fitness={run_cfg.fitness} n_in={brain_cfg.n_in} "
          f"n_hidden={brain_cfg.n_hidden} edges={template.n_edges} dims={reshaper.total_params} "
          f"pop={run_cfg.popsize} budget={brain_cfg.synaptic_budget} shrink={brain_cfg.shrink}")
    print(f"[seed {seed}] -> {run_dir}")

    csv_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["gen", "op", "best_acc", "mean_acc", "sel_best",
                     "edges", "max_edges", "density", "sigma", "secs_per_gen"])

    best_flat, best = None, -1.0        # best-EVER (spans goal switches under --mvg)
    final_flat, final = None, -1.0      # best of the last generation actually run
    matched_flat, matched, matched_gen = None, -1.0, -1
    arc_flat, arc_gen, arc_op, arc_acc = [], [], [], {op: [] for op in ops_in_play}
    cur_op, y = None, None
    gens_run = 0
    t_start = time.time()
    interval_start = t_start
    for gen in range(run_cfg.generations):
        gens_run = gen + 1
        op = goal_op(gen)
        if op != cur_op:
            cur_op, y = op, y_by_op[op]

        key, ask_key = jr.split(key)
        x, state = strategy.ask(ask_key, state, es_params)
        sel, acc = batched_eval(x, y)                     # CMA-ES selects on `sel`...
        state = strategy.tell(x, shaper.apply(x, sel), state, es_params)

        # ...but we track/save/report the best *accuracy* member (the real metric).
        best_idx = int(acc.argmax())
        gen_best = float(acc[best_idx])
        gen_best_flat = x[best_idx]
        if gen_best > best:
            best, best_flat = gen_best, jnp.asarray(gen_best_flat)
        final, final_flat = gen_best, jnp.asarray(gen_best_flat)
        if cur_op == run_cfg.operation:
            matched, matched_flat, matched_gen = gen_best, jnp.asarray(gen_best_flat), gen

        if run_cfg.archive_interval > 0 and gen % run_cfg.archive_interval == 0:
            arc_flat.append(np.asarray(gen_best_flat, dtype=np.float32))
            arc_gen.append(gen)
            arc_op.append(cur_op)
            for o in ops_in_play:
                arc_acc[o].append(gen_best if o == cur_op else acc_on(gen_best_flat, o))

        # Under --mvg, log at the END of each goal epoch instead of on
        # --log-interval, so every row is a goal the population has had the full
        # interval to adapt to and rows line up with the switches (matches
        # experiment 1). --log-interval still applies whenever the goal is fixed.
        epoch_len = max(1, run_cfg.switch_interval)
        due = ((gen + 1) % epoch_len == 0 if run_cfg.mvg
               else gen % run_cfg.log_interval == 0)
        if due or gen == 0 or gen == run_cfg.generations - 1:
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            st = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)
            sigma = float(getattr(state, "sigma", float("nan")))
            sigma_str = "" if sigma != sigma else f" | sigma: {sigma:.4f}"
            gens_in = (epoch_len if run_cfg.mvg else run_cfg.log_interval) if gen > 0 else 1
            secs_per_gen = (time.time() - interval_start) / gens_in
            interval_start = time.time()
            mean_acc = float(acc.mean())
            sel_best = float(sel.max())
            op_str = f" | op: {cur_op}" if tasks.uses_operation(run_cfg.task) else ""
            sel_str = f" | Sel: {sel_best:.3f}" if run_cfg.fitness != "accuracy" else ""
            print(f"  Gen {gen:5d} | Best: {gen_best:.3f}{sel_str} | Pop mean: {mean_acc:.3f}"
                  f" | Edges: {st['n_edges']}/{st['max_edges']} | Density: {st['density']:.1f}%"
                  f"{sigma_str}{op_str} | {secs_per_gen:.2f}s/gen")
            writer.writerow([gen, cur_op, f"{gen_best:.4f}", f"{mean_acc:.4f}",
                             f"{sel_best:.4f}", st["n_edges"], st["max_edges"],
                             f"{st['density']:.2f}", f"{sigma:.6f}", f"{secs_per_gen:.3f}"])
            csv_f.flush()   # intermediary results readable mid-run

        # live visualisation of the current best brain during training
        if (run_cfg.viz_interval > 0 and gen > 0 and gen % run_cfg.viz_interval == 0):
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            vp = os.path.join(run_dir, f"gen{gen}.png")
            visualize_brain(genome, brain_cfg, run_cfg.prune_threshold, vp,
                            title=f"{run_cfg.task} seed{seed} gen{gen} - acc {gen_best:.3f}",
                            open_after=run_cfg.open_image)

        # Early stop on target (only meaningful with a fixed goal, and OFF under
        # --mvg). Pass --no-early-stop for a fixed goal too in any FG-vs-MVG
        # comparison: otherwise the FG arm exits the moment it solves the task
        # while the MVG arm runs the full budget -- unequal generations AND
        # unequal post-solution drift, the very difference being measured.
        if run_cfg.early_stop and (not run_cfg.mvg) and best >= run_cfg.target:
            print(f"  early stop: best {best:.3f} >= target {run_cfg.target:.3f} at gen {gen}")
            break

    csv_f.close()

    archive_path, archive_n = "", 0
    if arc_flat:
        archive_path = os.path.join(run_dir, "champions.npz")
        payload = {"flat": np.stack(arc_flat),
                   "gen": np.asarray(arc_gen, dtype=np.int32),
                   "op": np.asarray(arc_op),
                   "ops_in_play": np.asarray(ops_in_play),
                   "reference_op": np.asarray(run_cfg.operation)}
        for o in ops_in_play:
            payload[f"acc_{o}"] = np.asarray(arc_acc[o], dtype=np.float32)
        tmp = archive_path + ".tmp.npz"
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, archive_path)       # atomic: a crash cannot corrupt it
        archive_n = len(arc_flat)
        print(f"[seed {seed}] archived {archive_n} champions -> {archive_path}")

    best_genome = eqx.combine(reshaper.reshape_single(best_flat), static)
    final_genome = eqx.combine(reshaper.reshape_single(final_flat), static)
    if matched_flat is None:            # cannot happen under a fixed goal
        matched_flat, matched, matched_gen = final_flat, final, gens_run - 1
    matched_genome = eqx.combine(reshaper.reshape_single(matched_flat), static)

    mean_flat = getattr(state, "mean", None)
    if mean_flat is None:
        centroid_genome, centroid_acc = None, float("nan")
    else:
        mean_flat = jnp.asarray(mean_flat)
        centroid_genome = eqx.combine(reshaper.reshape_single(mean_flat), static)
        centroid_acc = float(batched_eval(mean_flat[None, :], y)[1][0])

    acc_by_op = {"best": {o: acc_on(best_flat, o) for o in ops_in_play},
                 "final": {o: acc_on(final_flat, o) for o in ops_in_play},
                 "matched": {o: acc_on(matched_flat, o) for o in ops_in_play}}
    if mean_flat is not None:
        acc_by_op["centroid"] = {o: acc_on(mean_flat, o) for o in ops_in_play}

    return SeedResult(seed=seed, run_name=run_name, run_dir=run_dir,
                      best=best, best_genome=best_genome,
                      final=final, final_genome=final_genome,
                      centroid_acc=centroid_acc, centroid_genome=centroid_genome,
                      gens_run=gens_run, wall_s=time.time() - t_start,
                      matched=matched, matched_genome=matched_genome,
                      matched_op=run_cfg.operation, matched_gen=matched_gen,
                      acc_by_op=acc_by_op,
                      archive_path=archive_path, archive_n=archive_n)


def main():
    brain_cfg, run_cfg, _ = parse_args()
    need = tasks.min_inputs(run_cfg.task)
    assert brain_cfg.n_in >= need, f"task {run_cfg.task!r} needs n_in >= {need}"
    if run_cfg.mvg and not tasks.uses_operation(run_cfg.task):
        raise SystemExit(f"--mvg is meaningless for task {run_cfg.task!r}: its target does not "
                         f"depend on --operation, so the goal would never actually change. "
                         f"Use one of {sorted(t for t in tasks.TASKS if tasks.uses_operation(t))}.")
    os.makedirs(run_cfg.out_dir, exist_ok=True)

    # Which champion is THE result depends on the arm (see train_seed):
    # fixed goal -> "best"; --mvg -> "matched", NOT "final".
    headline = "matched" if run_cfg.mvg else "best"

    commit = _git_commit()
    results = []
    for i in range(run_cfg.n_seeds):
        seed = run_cfg.seed + i

        if run_cfg.resume:
            done = os.path.join(run_cfg.out_dir, run_name_for(brain_cfg, run_cfg, seed),
                                "result.json")
            if os.path.exists(done):
                try:
                    with open(done) as fh:
                        prev = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    prev = {}
                if prev.get("complete"):
                    hl = prev.get("stats", {}).get(prev.get("headline", ""), {})
                    print(f"[seed {seed}] already complete "
                          f"({prev.get('headline')} {hl.get('accuracy', float('nan')):.3f}) -> skip")
                    continue

        res = train_seed(brain_cfg, run_cfg, seed)

        with open(os.path.join(res.run_dir, "config.json"), "w") as fh:
            json.dump({"brain": {f.name: getattr(brain_cfg, f.name)
                                 for f in dataclasses.fields(brain_cfg)
                                 if f.name != "activation"},
                       "activation": getattr(brain_cfg.activation, "__name__", "custom"),
                       "seed": seed,
                       "git_commit": commit,
                       "run": {f.name: getattr(run_cfg, f.name)
                               for f in dataclasses.fields(run_cfg)}},
                      fh, indent=2, default=str)

        pngs, stats = {}, {}
        tagged = [("best", res.best, res.best_genome), ("final", res.final, res.final_genome),
                  ("matched", res.matched, res.matched_genome)]
        if res.centroid_genome is not None:
            tagged.append(("centroid", res.centroid_acc, res.centroid_genome))
        for tag, acc, genome in tagged:
            dna_path = os.path.join(res.run_dir, f"{tag}_dna.eqx")
            eqx.tree_serialise_leaves(dna_path, genome)
            st = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)
            stats[tag] = {k: st[k] for k in
                          ("n_edges", "max_edges", "density", "n_exc", "n_inh")}
            stats[tag]["accuracy"] = acc
            stats[tag]["acc_by_op"] = res.acc_by_op.get(tag, {})
            print(f"[seed {seed}] {tag:8s} accuracy {acc:.3f} | edges {st['n_edges']}/{st['max_edges']}"
                  f" | density {st['density']:.1f}% | exc(+) {st['n_exc']} inh(-) {st['n_inh']}")

            pngs[tag] = os.path.join(res.run_dir, f"{tag}_brain.png")
            visualize_brain(genome, brain_cfg, run_cfg.prune_threshold, pngs[tag],
                            title=f"{tag.capitalize()} DNA - {res.run_name} - accuracy {acc:.3f}",
                            open_after=(run_cfg.open_image and run_cfg.n_seeds == 1
                                        and tag == headline))

        with open(os.path.join(res.run_dir, "result.json"), "w") as fh:
            json.dump({"run_name": res.run_name, "seed": seed, "git_commit": commit,
                       "headline": headline, "gens_run": res.gens_run,
                       "wall_s": round(res.wall_s, 1), "stats": stats,
                       "reference_op": run_cfg.operation,
                       "matched_gen": res.matched_gen,
                       "acc_by_op": res.acc_by_op,
                       "archive": {"path": os.path.basename(res.archive_path),
                                   "n": res.archive_n} if res.archive_path else None,
                       "complete": True},
                      fh, indent=2, default=str)
        print(f"[seed {seed}] artifacts -> {res.run_dir}")
        results.append((res, pngs[headline]))

    if results and run_cfg.n_seeds > 1:
        print("\n=== summary ===")
        for res, _ in results:
            print(f"  seed {res.seed}: best {res.best:.3f} | final {res.final:.3f} "
                  f"| matched({res.matched_op}) {res.matched:.3f} "
                  f"| centroid {res.centroid_acc:.3f}")
        key = (lambda r: getattr(r[0], headline))
        top, best_png = max(results, key=key)
        print(f"best seed by {headline}: {top.seed} (best {top.best:.3f} | final {top.final:.3f}"
              f" | matched {top.matched:.3f})")
        print(f"mean {headline} {np.mean([getattr(r[0], headline) for r in results]):.3f}")
        if run_cfg.open_image:
            from visualize import _auto_open
            _auto_open(best_png)


if __name__ == "__main__":
    main()
