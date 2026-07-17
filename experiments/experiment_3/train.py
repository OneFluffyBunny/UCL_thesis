"""Gradient-descent training for experiment 3 (direct encoding).

The OPTIMISER-axis counterpart to experiment 2: same direct-encoding network
(model.DirectGenome), same tasks, same reported metric (balanced accuracy) --
the ONLY difference is that the weights are trained by backprop + Optax instead
of CMA-ES. Where CMA-ES samples a population and moves its distribution toward
the fitter members, gradient descent computes d(loss)/d(weights) on the *whole*
enumerated dataset and steps straight downhill.

Differentiability. The forward pass (scan of tanh(a @ w + b)) is smooth in the
weights, so the network itself backprops fine. Accuracy is NOT differentiable
(the sign threshold has zero gradient), so it is never the loss -- it stays the
reported metric. The loss is one of two differentiable objectives (--loss):
  * margin : the negative of the exact hinged signed-margin surrogate that
             experiment 2's CMA-ES *maximises*  ->  fair optimiser-only comparison.
  * bce    : a balanced logistic loss on p=(tanh_out+1)/2  ->  the gradient-oracle
             "best a GD method can do here" upper bound.

One step here = one full-batch gradient update (1 forward+backward over all
2**n_in inputs). NOTE this is not the same unit as a CMA-ES generation, which
costs `popsize` forward evals -- the fair cross-experiment x-axis is function
evaluations, not step/generation count.

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
import optax
import numpy as np

import tasks
from model import DirectGenome
from config import parse_args
from visualize import brain_stats, visualize_brain


def _encode(X, encoding):
    return X * 2.0 - 1.0 if encoding == "bipolar" else X


# hinge cap for the margin surrogate (identical to experiment_2/train.py)
_MARGIN_CAP = 0.5


def _balanced_mean(vals, yf):
    """Mean of per-class means of `vals` (balanced; chance-corrected)."""
    pos = jnp.sum(vals * yf) / (jnp.sum(yf) + 1e-8)
    neg = jnp.sum(vals * (1.0 - yf)) / (jnp.sum(1.0 - yf) + 1e-8)
    return 0.5 * (pos + neg)


def _accuracy(out, y, balanced):
    """Reported (non-differentiable) balanced 0/1 accuracy on the tanh output."""
    yf = y.astype(jnp.float32)
    correct = ((out > 0).astype(jnp.int32) == y).astype(jnp.float32)
    return _balanced_mean(correct, yf) if balanced else jnp.mean(correct)


def _loss(out, y, balanced, loss_kind):
    """Differentiable training loss (to MINIMISE) on the raw tanh output `out`."""
    yf = y.astype(jnp.float32)
    if loss_kind == "bce":
        # map tanh output in (-1, 1) to a probability, then balanced cross-entropy
        p = jnp.clip((out + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
        ce = -(yf * jnp.log(p) + (1.0 - yf) * jnp.log1p(-p))
        return _balanced_mean(ce, yf) if balanced else jnp.mean(ce)
    # margin (default): descend the negative of exp 2's *maximised* surrogate
    s = 2.0 * yf - 1.0                               # target sign in {-1, +1}
    m = jnp.minimum(out * s, _MARGIN_CAP)            # hinge on signed margin
    score = (m + 1.0) / (1.0 + _MARGIN_CAP)          # rescale to [0, 1]
    surr = _balanced_mean(score, yf) if balanced else jnp.mean(score)
    return -surr


def _make_optimizer(run_cfg):
    base = optax.adam(run_cfg.lr) if run_cfg.optimizer == "adam" else optax.sgd(run_cfg.lr)
    if run_cfg.grad_clip > 0:
        return optax.chain(optax.clip_by_global_norm(run_cfg.grad_clip), base)
    return base


def train_seed(brain_cfg, run_cfg, seed):
    """Run one seed of gradient descent; return (best_accuracy, best_genome, run_name)."""
    key = jr.PRNGKey(seed)

    X = tasks.all_binary_inputs(brain_cfg.n_in)
    X_enc = _encode(X, run_cfg.input_encoding)

    key, init_key = jr.split(key)
    genome = DirectGenome.init(init_key, brain_cfg)
    # split into trainable (weights, bias -- the inexact arrays) and static
    # (edge_rows / edge_cols -- integer scatter indices); gradients flow to
    # `params` only, which is exactly the direct genome.
    params, static = eqx.partition(genome, eqx.is_inexact_array)

    optim = _make_optimizer(run_cfg)
    opt_state = optim.init(params)

    def loss_only(params, y):
        g = eqx.combine(params, static)
        out = jax.vmap(lambda o: g.forward(o, brain_cfg))(X_enc)[:, 0]   # raw tanh
        loss = _loss(out, y, run_cfg.balanced, run_cfg.loss)
        acc = _accuracy(out, y, run_cfg.balanced)                       # aux (not differentiated)
        return loss, acc

    grad_fn = eqx.filter_value_and_grad(loss_only, has_aux=True)

    @eqx.filter_jit
    def step(params, opt_state, y):
        (loss, acc), grads = grad_fn(params, y)        # loss/acc at the CURRENT params
        updates, opt_state = optim.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss, acc

    def goal_op(step_i):
        if not run_cfg.mvg:
            return run_cfg.operation
        ops = run_cfg.mvg_ops
        return ops[(step_i // run_cfg.switch_interval) % len(ops)]

    run_name = f"{run_cfg.task}_seed{seed}"
    print(f"[seed {seed}] task={run_cfg.task} op0={goal_op(0)} mvg={run_cfg.mvg} "
          f"balanced={run_cfg.balanced} loss={run_cfg.loss} opt={run_cfg.optimizer} "
          f"lr={run_cfg.lr} n_in={brain_cfg.n_in} n_hidden={brain_cfg.n_hidden} "
          f"edges={genome.n_edges} steps={run_cfg.steps}")

    best_params, best = None, -1.0
    cur_op, y = None, None
    interval_start = time.time()
    for s in range(run_cfg.steps):
        op = goal_op(s)
        if op != cur_op:
            cur_op, y = op, tasks.targets(run_cfg.task, op, X)

        prev_params = params                            # params that `acc` is measured at
        params, opt_state, loss, acc = step(params, opt_state, y)

        acc_f = float(acc)
        if acc_f > best:
            best, best_params = acc_f, prev_params      # keep the params that achieved it

        if s % run_cfg.log_interval == 0 or s == run_cfg.steps - 1:
            density = brain_stats(eqx.combine(prev_params, static),
                                  brain_cfg, run_cfg.prune_threshold)["density"]
            gens_in = run_cfg.log_interval if s > 0 else 1
            secs_per_step = (time.time() - interval_start) / gens_in
            interval_start = time.time()
            op_str = f" | op: {cur_op}" if run_cfg.task == "retina" else ""
            print(f"  Step {s:4d} | Loss: {float(loss):.4f} | Acc: {acc_f:.3f} | Best: {best:.3f}"
                  f" | Density: {density:.1f}%{op_str} | {secs_per_step:.3f}s/step")

        # live visualisation of the current best brain during training
        if (run_cfg.viz_interval > 0 and s > 0 and s % run_cfg.viz_interval == 0):
            vp = os.path.join(run_cfg.out_dir, f"{run_name}_step{s}.png")
            visualize_brain(eqx.combine(prev_params, static), brain_cfg,
                            run_cfg.prune_threshold, vp,
                            title=f"{run_cfg.task} seed{seed} step{s} - acc {acc_f:.3f}",
                            open_after=run_cfg.open_image)

        # early stop on target (only meaningful with a fixed goal)
        if (not run_cfg.mvg) and best >= run_cfg.target:
            print(f"  early stop: best {best:.3f} >= target {run_cfg.target:.3f} at step {s}")
            break

    best_genome = eqx.combine(best_params, static)
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
                        title=f"Best (GD) - {run_cfg.task} seed{seed} - accuracy {best:.3f}",
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
