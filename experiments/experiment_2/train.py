"""CMA-ES training for experiment 2 (direct encoding).

Mirrors experiment_1/train.py — same CMA-ES loop, same balanced-accuracy metric,
same optional margin surrogate, same fixed-goal / modularly-varying-goal (--mvg)
switch, same flags (via config.py) and the same brain-image / DNA outputs — so
results are directly comparable. The ONLY difference is the model: here the
genome is the raw weight vector (model.DirectGenome), not a compressed DNA grown
through a connection rule.

This produces the baseline the genomic bottleneck (exp 1) is measured against:
the best a fixed-topology network can do when every edge is a free parameter,
with no encoding-induced regularity.

Run `python train.py --help` for all flags.
"""

from __future__ import annotations

import sys
# line-buffer stdout/stderr so logs flush immediately under redirection/background
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
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


def train_seed(brain_cfg, run_cfg, seed):
    """Run one seed; return (best_accuracy, best_genome, run_name)."""
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

    run_name = f"{run_cfg.task}_seed{seed}"
    print(f"[seed {seed}] task={run_cfg.task} op0={goal_op(0)} mvg={run_cfg.mvg} "
          f"balanced={run_cfg.balanced} fitness={run_cfg.fitness} n_in={brain_cfg.n_in} "
          f"n_hidden={brain_cfg.n_hidden} edges={template.n_edges} dims={reshaper.total_params} "
          f"pop={run_cfg.popsize}")

    best_flat, best = None, -1.0
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
            op_str = f" | op: {cur_op}" if run_cfg.task == "retina" else ""
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

        # early stop on target (only meaningful with a fixed goal)
        if (not run_cfg.mvg) and best >= run_cfg.target:
            print(f"  early stop: best {best:.3f} >= target {run_cfg.target:.3f} at gen {gen}")
            break

    best_genome = eqx.combine(reshaper.reshape_single(best_flat), static)
    return best, best_genome, run_name


def main():
    brain_cfg, run_cfg, _ = parse_args()
    need = tasks.min_inputs(run_cfg.task)
    assert brain_cfg.n_in >= need, f"task {run_cfg.task!r} needs n_in >= {need}"
    os.makedirs(run_cfg.out_dir, exist_ok=True)

    results = []
    for i in range(run_cfg.n_seeds):
        seed = run_cfg.seed + i
        best, best_genome, run_name = train_seed(brain_cfg, run_cfg, seed)

        dna_path = os.path.join(run_cfg.out_dir, f"{run_name}_best_dna.eqx")
        eqx.tree_serialise_leaves(dna_path, best_genome)
        st = brain_stats(best_genome, brain_cfg, run_cfg.prune_threshold)
        print(f"[seed {seed}] best accuracy {best:.3f} | edges {st['n_edges']}/{st['max_edges']}"
              f" | density {st['density']:.1f}% | exc(+) {st['n_exc']} inh(-) {st['n_inh']}")
        print(f"[seed {seed}] best DNA saved -> {dna_path}")

        png_path = os.path.join(run_cfg.out_dir, f"{run_name}_best_brain.png")
        # auto-open only for a single seed (avoid spamming windows on multi-seed runs)
        visualize_brain(best_genome, brain_cfg, run_cfg.prune_threshold, png_path,
                        title=f"Best DNA - {run_cfg.task} seed{seed} - accuracy {best:.3f}",
                        open_after=run_cfg.open_image and run_cfg.n_seeds == 1)
        results.append((seed, best, png_path))

    if run_cfg.n_seeds > 1:
        print("\n=== summary ===")
        for seed, best, _ in results:
            print(f"  seed {seed}: best accuracy {best:.3f}")
        best_seed, best_acc, best_png = max(results, key=lambda r: r[1])
        print(f"best seed: {best_seed} ({best_acc:.3f})  |  mean {np.mean([r[1] for r in results]):.3f}")
        if run_cfg.open_image:
            from visualize import _auto_open
            _auto_open(best_png)


if __name__ == "__main__":
    main()
