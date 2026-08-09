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

import dataclasses
import json
import os
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


def train_seed(brain_cfg, run_cfg, seed):
    """Run one seed; return (best_acc, best_genome, final_acc, final_genome, run_name).

    TWO champions are returned on purpose, and which one you should measure
    depends on the arm:

    * ``best_*``  -- highest accuracy seen at ANY generation. Fine under a fixed
      goal. Under ``--mvg`` it is the peak across a CHANGING target, i.e.
      whichever member happened to top out during whichever AND/OR phase, so it
      is not comparable to a fixed-goal champion.
    * ``final_*`` -- best member of the LAST generation. This is the one to use
      for any FG-vs-MVG structural/modularity comparison: it is the endpoint of
      the same number of generations of the same pressure in both arms.
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

    # Stamp the ARM into the run name. Without this an FG and an MVG run of the
    # same task write the same DNA/PNG filenames and silently clobber each other.
    if tasks.uses_operation(run_cfg.task):
        arm = "mvg-" + "-".join(run_cfg.mvg_ops) if run_cfg.mvg else f"fg-{run_cfg.operation}"
        run_name = f"{run_cfg.task}_{arm}_seed{seed}"
    else:
        run_name = f"{run_cfg.task}_seed{seed}"

    print(f"[seed {seed}] task={run_cfg.task} op0={goal_op(0)} mvg={run_cfg.mvg} "
          f"balanced={run_cfg.balanced} fitness={run_cfg.fitness} n_in={brain_cfg.n_in} "
          f"n_hidden={brain_cfg.n_hidden} K={brain_cfg.n_types} dims={reshaper.total_params} "
          f"pop={run_cfg.popsize} w_threshold={brain_cfg.w_threshold}")

    best_flat, best = None, -1.0        # best-EVER (spans goal switches under --mvg)
    final_flat, final = None, -1.0      # best of the last generation actually run
    cur_op, y = None, None
    interval_start = time.time()
    for gen in range(run_cfg.generations):
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

        if gen % run_cfg.log_interval == 0 or gen == run_cfg.generations - 1:
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            density = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)["density"]
            try:
                sigma_str = f" | σ: {float(state.sigma):.4f}"
            except AttributeError:
                sigma_str = ""
            gens_in = run_cfg.log_interval if gen > 0 else 1
            secs_per_gen = (time.time() - interval_start) / gens_in
            interval_start = time.time()
            op_str = f" | op: {cur_op}" if tasks.uses_operation(run_cfg.task) else ""
            # show the selection surrogate too when it isn't just accuracy
            sel_str = f" | Sel: {float(sel.max()):.3f}" if run_cfg.fitness != "accuracy" else ""
            print(f"  Gen {gen:4d} | Best: {gen_best:.3f}{sel_str} | Pop mean: {float(acc.mean()):.3f}"
                  f" | Density: {density:.1f}%{sigma_str}{op_str} | {secs_per_gen:.2f}s/gen")

        # live visualisation of the current best brain during training
        if (run_cfg.viz_interval > 0 and gen > 0 and gen % run_cfg.viz_interval == 0):
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            vp = os.path.join(run_cfg.out_dir, f"{run_name}_gen{gen}.png")
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

    best_genome = eqx.combine(reshaper.reshape_single(best_flat), static)
    final_genome = eqx.combine(reshaper.reshape_single(final_flat), static)
    return best, best_genome, final, final_genome, run_name


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

    results = []
    for i in range(run_cfg.n_seeds):
        seed = run_cfg.seed + i
        best, best_genome, final, final_genome, run_name = train_seed(brain_cfg, run_cfg, seed)

        # Sidecar config. The gate (--w-threshold) is part of the PHENOTYPE but
        # not of the genome, so a .eqx reloaded without it grows a different
        # brain than the one that was evaluated. Record what it takes to regrow
        # this brain exactly.
        cfg_path = os.path.join(run_cfg.out_dir, f"{run_name}_config.json")
        with open(cfg_path, "w") as fh:
            json.dump({"brain": {f.name: getattr(brain_cfg, f.name)
                                 for f in dataclasses.fields(brain_cfg)
                                 if f.name != "activation"},
                       "activation": getattr(brain_cfg.activation, "__name__", "custom"),
                       "seed": seed,
                       "run": {f.name: getattr(run_cfg, f.name)
                               for f in dataclasses.fields(run_cfg)}},
                      fh, indent=2, default=str)

        pngs = {}
        for tag, acc, genome in (("best", best, best_genome), ("final", final, final_genome)):
            dna_path = os.path.join(run_cfg.out_dir, f"{run_name}_{tag}_dna.eqx")
            eqx.tree_serialise_leaves(dna_path, genome)
            st = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)
            print(f"[seed {seed}] {tag:5s} accuracy {acc:.3f} | edges {st['n_edges']}/{st['max_edges']}"
                  f" | density {st['density']:.1f}% | exc(+) {st['n_exc']} inh(-) {st['n_inh']}"
                  f" | hidden type counts {st['type_counts']}")
            print(f"[seed {seed}] {tag:5s} DNA saved -> {dna_path}")

            pngs[tag] = os.path.join(run_cfg.out_dir, f"{run_name}_{tag}_brain.png")
            # auto-open only the headline arm, and only for a single seed
            visualize_brain(genome, brain_cfg, run_cfg.prune_threshold, pngs[tag],
                            title=f"{tag.capitalize()} DNA - {run_name} - accuracy {acc:.3f}",
                            open_after=(run_cfg.open_image and run_cfg.n_seeds == 1
                                        and tag == headline))
        results.append((seed, best, final, pngs[headline]))

    if run_cfg.n_seeds > 1:
        print("\n=== summary ===")
        for seed, best, final, _ in results:
            print(f"  seed {seed}: best {best:.3f} | final {final:.3f}")
        # rank on the headline metric for this arm (see `headline` above)
        key = (lambda r: r[2]) if headline == "final" else (lambda r: r[1])
        best_seed, b, f, best_png = max(results, key=key)
        print(f"best seed by {headline}: {best_seed} (best {b:.3f} | final {f:.3f})")
        if run_cfg.open_image:
            from visualize import _auto_open
            _auto_open(best_png)


if __name__ == "__main__":
    main()
