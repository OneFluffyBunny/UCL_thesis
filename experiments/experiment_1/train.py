"""CMA-ES training loop for experiment 1.

Evolves the DNA (Genome) so that the static brain it grows solves a boolean task.
Fitness = (balanced) fraction of correct outputs over the full input enumeration
-> deterministic, noise-free evaluation (no winner's-curse possible).

Modularly-varying goal (--mvg): the target operation switches every
--switch-interval generations, alternating AND <-> OR (the two goals share the
left/right module structure -- the Kashtan-Alon setup).

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

import tasks
from model import Genome
from config import parse_args
from visualize import brain_stats, visualize_brain


def _encode(X, encoding):
    return X * 2.0 - 1.0 if encoding == "bipolar" else X


# hinge cap for the margin surrogate: once the raw output is this far onto the
# correct side of 0, a case stops paying, so all selection pressure flows to the
# still-wrong cases (why hinge losses exist). Output is tanh-bounded to [-1, 1].
_MARGIN_CAP = 0.5


def _balanced_mean(vals, yf):
    """Mean of per-class means of `vals` (balanced; chance-corrected)."""
    pos = jnp.sum(vals * yf) / (jnp.sum(yf) + 1e-8)
    neg = jnp.sum(vals * (1.0 - yf)) / (jnp.sum(1.0 - yf) + 1e-8)
    return 0.5 * (pos + neg)


def _make_eval(genome_template_static, reshaper, brain_cfg, X_enc, balanced, fitness):
    """Build a jitted, vmapped eval: flat DNA + targets -> (selection_fitness, accuracy).

    `accuracy` is always the raw 0/1 (balanced) metric -- what we log, early-stop
    and report. `selection_fitness` is what CMA-ES maximises: identical to accuracy
    when fitness='accuracy', or a smooth hinged signed-margin surrogate on the raw
    tanh output when fitness='margin' (the decision is still sign(output)).
    """
    static = genome_template_static

    def eval_genome(flat_params, y):
        genome = eqx.combine(reshaper.reshape_single(flat_params), static)
        yf = y.astype(jnp.float32)

        out = jax.vmap(lambda o: genome.forward(o, brain_cfg))(X_enc)[:, 0]  # raw tanh
        correct = ((out > 0).astype(jnp.int32) == y).astype(jnp.float32)
        acc = _balanced_mean(correct, yf) if balanced else jnp.mean(correct)

        if fitness == "margin":
            s = 2.0 * yf - 1.0                              # target sign in {-1,+1}
            m = jnp.minimum(out * s, _MARGIN_CAP)          # hinge on signed margin
            score = (m + 1.0) / (1.0 + _MARGIN_CAP)        # rescale to [0, 1]
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
    """Directory//artifact name for one seed of one arm.

    Everything that changes the phenotype has to appear here, or two runs write
    the same files: the ARM (an FG and an MVG run of the same task would collide)
    and the synaptic GATE (a t=0.2 run would silently clobber a t=0 one, and the
    only record of the difference is inside the config sidecar it just overwrote).
    """
    if tasks.uses_operation(run_cfg.task):
        arm = "mvg-" + "-".join(run_cfg.mvg_ops) if run_cfg.mvg else f"fg-{run_cfg.operation}"
        name = f"{run_cfg.task}_{arm}"
    else:
        name = run_cfg.task
    if brain_cfg.w_threshold > 0:
        name += f"_w{brain_cfg.w_threshold:g}"
    return f"{name}_seed{seed}"


@dataclasses.dataclass
class SeedResult:
    """Everything one seed produced. Returned instead of a tuple because the
    caller needs the artifacts, the metrics AND the run metadata."""
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


def train_seed(brain_cfg, run_cfg, seed) -> SeedResult:
    """Run one seed; return a SeedResult.

    TWO champions are returned on purpose, and which one you should measure
    depends on the arm:

    * ``best_*``  -- highest accuracy seen at ANY generation. Fine under a fixed
      goal. Under ``--mvg`` it is the peak across a CHANGING target, i.e.
      whichever member happened to top out during whichever AND/OR phase, so it
      is not comparable to a fixed-goal champion.
    * ``final_*`` -- best member of the LAST generation. This is the one to use
      for any FG-vs-MVG structural/modularity comparison: it is the endpoint of
      the same number of generations of the same pressure in both arms.
    * ``centroid_*`` -- the CMA-ES distribution MEAN, not a sampled member. The
      champions above are single draws and can be lucky; the mean is where the
      search actually settled, so it is the more stable thing to measure
      structure on when the two disagree.

    Also writes ``<run_dir>/log.csv`` as it goes (flushed every log interval, so
    it is readable mid-run) -- the FG-vs-MVG claim is about trajectories, and a
    trajectory that only exists in terminal scrollback cannot be plotted.
    """
    key = jr.PRNGKey(seed)

    X = tasks.all_binary_inputs(brain_cfg.n_in)
    X_enc = _encode(X, run_cfg.input_encoding)

    key, init_key = jr.split(key)
    template = Genome.init(init_key, brain_cfg)
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

    # One directory per run, so artifacts of different arms/gates can never mix.
    run_name = run_name_for(brain_cfg, run_cfg, seed)
    run_dir = os.path.join(run_cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[seed {seed}] task={run_cfg.task} op0={goal_op(0)} mvg={run_cfg.mvg} "
          f"balanced={run_cfg.balanced} fitness={run_cfg.fitness} n_in={brain_cfg.n_in} "
          f"n_hidden={brain_cfg.n_hidden} K={brain_cfg.n_types} dims={reshaper.total_params} "
          f"pop={run_cfg.popsize} w_threshold={brain_cfg.w_threshold}")
    print(f"[seed {seed}] -> {run_dir}")

    csv_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["gen", "op", "best_acc", "mean_acc", "sel_best",
                     "edges", "max_edges", "density", "sigma", "secs_per_gen"])

    best_flat, best = None, -1.0        # best-EVER (spans goal switches under --mvg)
    final_flat, final = None, -1.0      # best of the last generation actually run
    cur_op, y = None, None
    gens_run = 0
    t_start = time.time()
    interval_start = t_start
    for gen in range(run_cfg.generations):
        gens_run = gen + 1
        op = goal_op(gen)
        if op != cur_op:
            cur_op, y = op, tasks.targets(run_cfg.task, op, X)

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
        # ...and always keep the CURRENT generation's champion, so whatever
        # generation the loop exits on we still have the endpoint genome.
        final, final_flat = gen_best, jnp.asarray(gen_best_flat)

        # gen 0 is always logged (0 % anything == 0), so every run has a baseline
        # row before selection has done anything.
        if gen % run_cfg.log_interval == 0 or gen == run_cfg.generations - 1:
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            st = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)
            density, n_edges = st["density"], st["n_edges"]
            sigma = float(getattr(state, "sigma", float("nan")))
            sigma_str = "" if sigma != sigma else f" | σ: {sigma:.4f}"
            gens_in = run_cfg.log_interval if gen > 0 else 1
            secs_per_gen = (time.time() - interval_start) / gens_in
            interval_start = time.time()
            mean_acc = float(acc.mean())
            sel_best = float(sel.max())
            op_str = f" | op: {cur_op}" if tasks.uses_operation(run_cfg.task) else ""
            # show the selection surrogate too when it isn't just accuracy
            sel_str = f" | Sel: {sel_best:.3f}" if run_cfg.fitness != "accuracy" else ""
            print(f"  Gen {gen:4d} | Best: {gen_best:.3f}{sel_str} | Pop mean: {mean_acc:.3f}"
                  f" | Edges: {n_edges}/{st['max_edges']} | Density: {density:.1f}%"
                  f"{sigma_str}{op_str} | {secs_per_gen:.2f}s/gen")
            writer.writerow([gen, cur_op, f"{gen_best:.4f}", f"{mean_acc:.4f}",
                             f"{sel_best:.4f}", n_edges, st["max_edges"],
                             f"{density:.2f}", f"{sigma:.6f}", f"{secs_per_gen:.3f}"])
            csv_f.flush()   # intermediary results readable mid-run

        # live visualisation of the current best brain during training
        if (run_cfg.viz_interval > 0 and gen > 0 and gen % run_cfg.viz_interval == 0):
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            vp = os.path.join(run_dir, f"gen{gen}.png")
            visualize_brain(genome, brain_cfg, run_cfg.prune_threshold, vp,
                            title=f"{run_cfg.task} seed{seed} gen{gen} - acc {gen_best:.3f}",
                            open_after=run_cfg.open_image)

        # Early stop on target (only meaningful with a fixed goal, and OFF under
        # --mvg). Pass --no-early-stop to disable it for a fixed goal too: it
        # otherwise makes an FG arm exit the moment it solves the task while the
        # MVG arm always runs the full budget -- unequal generations AND unequal
        # opportunity for post-solution drift, which is exactly the structural
        # difference an FG-vs-MVG modularity comparison is trying to measure.
        if run_cfg.early_stop and (not run_cfg.mvg) and best >= run_cfg.target:
            print(f"  early stop: best {best:.3f} >= target {run_cfg.target:.3f} at gen {gen}")
            break

    csv_f.close()

    best_genome = eqx.combine(reshaper.reshape_single(best_flat), static)
    final_genome = eqx.combine(reshaper.reshape_single(final_flat), static)

    # CMA-ES distribution mean: where the search settled, as opposed to the
    # single lucky draw the champions are. Scored on the goal that was active at
    # the end (the only one it can be compared against under --mvg).
    mean_flat = getattr(state, "mean", None)
    if mean_flat is None:
        centroid_genome, centroid_acc = None, float("nan")
    else:
        mean_flat = jnp.asarray(mean_flat)
        centroid_genome = eqx.combine(reshaper.reshape_single(mean_flat), static)
        centroid_acc = float(batched_eval(mean_flat[None, :], y)[1][0])

    return SeedResult(seed=seed, run_name=run_name, run_dir=run_dir,
                      best=best, best_genome=best_genome,
                      final=final, final_genome=final_genome,
                      centroid_acc=centroid_acc, centroid_genome=centroid_genome,
                      gens_run=gens_run, wall_s=time.time() - t_start)


def main():
    brain_cfg, run_cfg, _ = parse_args()
    need = tasks.min_inputs(run_cfg.task)
    assert brain_cfg.n_in >= need, f"task {run_cfg.task!r} needs n_in >= {need}"
    if run_cfg.mvg and not tasks.uses_operation(run_cfg.task):
        raise SystemExit(f"--mvg is meaningless for task {run_cfg.task!r}: its target does not "
                         f"depend on --operation, so the goal would never actually change. "
                         f"Use one of {sorted(t for t in tasks.TASKS if tasks.uses_operation(t))}.")
    os.makedirs(run_cfg.out_dir, exist_ok=True)

    # Under --mvg the best-EVER champion peaked on whichever goal happened to be
    # active at the time, so the final-generation champion is the comparable one.
    headline = "final" if run_cfg.mvg else "best"

    commit = _git_commit()
    results = []
    for i in range(run_cfg.n_seeds):
        seed = run_cfg.seed + i
        res = train_seed(brain_cfg, run_cfg, seed)

        # Sidecar config. The gate (--w-threshold) is part of the PHENOTYPE but
        # not of the genome, so a .eqx reloaded without it grows a different
        # brain than the one that was evaluated. Record what it takes to regrow
        # this brain exactly.
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
        tagged = [("best", res.best, res.best_genome), ("final", res.final, res.final_genome)]
        if res.centroid_genome is not None:
            tagged.append(("centroid", res.centroid_acc, res.centroid_genome))
        for tag, acc, genome in tagged:
            dna_path = os.path.join(res.run_dir, f"{tag}_dna.eqx")
            eqx.tree_serialise_leaves(dna_path, genome)
            st = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)
            stats[tag] = {k: st[k] for k in
                          ("n_edges", "max_edges", "density", "n_exc", "n_inh", "type_counts")}
            stats[tag]["accuracy"] = acc
            print(f"[seed {seed}] {tag:8s} accuracy {acc:.3f} | edges {st['n_edges']}/{st['max_edges']}"
                  f" | density {st['density']:.1f}% | exc(+) {st['n_exc']} inh(-) {st['n_inh']}"
                  f" | hidden type counts {st['type_counts']}")

            pngs[tag] = os.path.join(res.run_dir, f"{tag}_brain.png")
            # auto-open only the headline arm, and only for a single seed
            visualize_brain(genome, brain_cfg, run_cfg.prune_threshold, pngs[tag],
                            title=f"{tag.capitalize()} DNA - {res.run_name} - accuracy {acc:.3f}",
                            open_after=(run_cfg.open_image and run_cfg.n_seeds == 1
                                        and tag == headline))

        # Results on disk, not just in scrollback. `headline` is recorded because
        # which champion is the comparable one depends on the arm.
        with open(os.path.join(res.run_dir, "result.json"), "w") as fh:
            json.dump({"run_name": res.run_name, "seed": seed, "git_commit": commit,
                       "headline": headline, "gens_run": res.gens_run,
                       "wall_s": round(res.wall_s, 1), "stats": stats},
                      fh, indent=2, default=str)
        print(f"[seed {seed}] artifacts -> {res.run_dir}")
        results.append((res, pngs[headline]))

    if run_cfg.n_seeds > 1:
        print("\n=== summary ===")
        for res, _ in results:
            print(f"  seed {res.seed}: best {res.best:.3f} | final {res.final:.3f} "
                  f"| centroid {res.centroid_acc:.3f}")
        # rank on the headline metric for this arm (see `headline` above)
        key = (lambda r: r[0].final) if headline == "final" else (lambda r: r[0].best)
        top, best_png = max(results, key=key)
        print(f"best seed by {headline}: {top.seed} (best {top.best:.3f} | final {top.final:.3f})")
        if run_cfg.open_image:
            from visualize import _auto_open
            _auto_open(best_png)


if __name__ == "__main__":
    main()
