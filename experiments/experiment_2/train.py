"""CMA-ES training for experiment 2 (direct encoding).

Mirrors experiment_1/train.py — same CMA-ES loop, same balanced-accuracy metric,
same optional margin surrogate, same fixed-goal / modularly-varying-goal (--mvg)
switch — so results are directly comparable. The ONLY difference is the model:
here the genome is the raw weight vector (model.DirectGenome), not a compressed
DNA grown through a connection rule.

Run `python train.py --help` for all flags.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import evosax as ex
import numpy as np

import tasks
from model import BrainConfig, DirectGenome


_ACTIVATIONS = {"tanh": jnp.tanh, "relu": jax.nn.relu,
                "sigmoid": jax.nn.sigmoid, "linear": lambda x: x}
_MARGIN_CAP = 0.5


def _encode(X, encoding):
    return X * 2.0 - 1.0 if encoding == "bipolar" else X


def _balanced_mean(vals, yf):
    pos = jnp.sum(vals * yf) / (jnp.sum(yf) + 1e-8)
    neg = jnp.sum(vals * (1.0 - yf)) / (jnp.sum(1.0 - yf) + 1e-8)
    return 0.5 * (pos + neg)


def _make_eval(static, reshaper, cfg, X_enc, balanced, fitness):
    """Jitted, vmapped eval: flat weights + targets -> (selection_fitness, accuracy)."""
    def eval_genome(flat_params, y):
        genome = eqx.combine(reshaper.reshape_single(flat_params), static)
        yf = y.astype(jnp.float32)
        out = jax.vmap(lambda o: genome.forward(o, cfg))(X_enc)[:, 0]   # raw tanh
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


def brain_stats(genome, cfg, threshold):
    """Edge count / density / sign of the grown weight matrix (analysis only)."""
    w = np.asarray(genome.build_weights(cfg))
    present = np.abs(w) > threshold
    n_edges = int(present.sum())
    max_edges = (cfg.n_in * cfg.n_hidden + cfg.n_hidden * (cfg.n_hidden - 1)
                 + cfg.n_hidden * cfg.n_out)
    signed = w[present]
    return dict(n_edges=n_edges, max_edges=max_edges,
                density=100.0 * n_edges / max_edges if max_edges else 0.0,
                n_exc=int((signed > 0).sum()), n_inh=int((signed < 0).sum()))


def train_seed(cfg, args, seed):
    key = jr.PRNGKey(seed)
    X = tasks.all_binary_inputs(cfg.n_in)
    X_enc = _encode(X, args.input_encoding)

    key, init_key = jr.split(key)
    template = DirectGenome.init(init_key, cfg)
    params, static = eqx.partition(template, eqx.is_inexact_array)
    reshaper = ex.ParameterReshaper(params)
    batched_eval = _make_eval(static, reshaper, cfg, X_enc, not args.no_balanced, args.fitness)

    strategy = ex.Strategies[args.strategy](
        popsize=args.popsize, num_dims=reshaper.total_params, sigma_init=args.sigma_init)
    es_params = strategy.default_params
    key, es_key = jr.split(key)
    state = strategy.initialize(es_key, es_params)
    shaper = ex.FitnessShaper(maximize=True)

    def goal_op(gen):
        if not args.mvg:
            return args.operation
        return ["and", "or"][(gen // args.switch_interval) % 2]

    print(f"[seed {seed}] task={args.task} op0={goal_op(0)} mvg={args.mvg} "
          f"fitness={args.fitness} n_in={cfg.n_in} n_hidden={cfg.n_hidden} "
          f"edges={template.n_edges} dims={reshaper.total_params} pop={args.popsize}")

    best_flat, best = None, -1.0
    cur_op, y = None, None
    t0 = time.time()
    for gen in range(args.generations):
        op = goal_op(gen)
        if op != cur_op:
            cur_op, y = op, tasks.targets(args.task, op, X)

        key, ask_key = jr.split(key)
        x, state = strategy.ask(ask_key, state, es_params)
        sel, acc = batched_eval(x, y)
        state = strategy.tell(x, shaper.apply(x, sel), state, es_params)

        best_idx = int(acc.argmax())
        gen_best = float(acc[best_idx])
        if gen_best > best:
            best, best_flat = gen_best, jnp.asarray(x[best_idx])

        if gen % args.log_interval == 0 or gen == args.generations - 1:
            genome = eqx.combine(reshaper.reshape_single(x[best_idx]), static)
            st = brain_stats(genome, cfg, args.prune_threshold)
            try:
                sig = f" | σ: {float(state.sigma):.3f}"
            except AttributeError:
                sig = ""
            sel_str = f" | Sel: {float(sel.max()):.3f}" if args.fitness != "accuracy" else ""
            op_str = f" | op: {cur_op}" if args.task == "retina" else ""
            sps = (time.time() - t0) / (gen + 1)
            print(f"  Gen {gen:4d} | Best: {gen_best:.3f}{sel_str} | Pop mean: {float(acc.mean()):.3f}"
                  f" | Density: {st['density']:.1f}%{sig}{op_str} | {sps:.2f}s/gen")

        if (not args.mvg) and best >= args.target:
            print(f"  early stop: best {best:.3f} >= target {args.target:.3f} at gen {gen}")
            break

    best_genome = eqx.combine(reshaper.reshape_single(best_flat), static)
    return best, best_genome


def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment 2: evolve a DIRECTLY-encoded network (K-A-style control).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # architecture (no K / type_dim / g — direct encoding has no encoding knobs)
    p.add_argument("--n-in", type=int, default=8)
    p.add_argument("--n-hidden", type=int, default=10)
    p.add_argument("--n-out", type=int, default=1)
    p.add_argument("--rnn-iters", type=int, default=8)
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--activation", choices=sorted(_ACTIVATIONS), default="tanh")
    # search
    p.add_argument("--strategy", default="CMA_ES")
    p.add_argument("--fitness", choices=["accuracy", "margin"], default="accuracy")
    p.add_argument("--popsize", type=int, default=64)
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--sigma-init", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--target", type=float, default=1.0)
    # task
    p.add_argument("--task", default="retina", choices=["copy", "and2", "left", "retina"])
    p.add_argument("--operation", choices=["and", "or", "xor"], default="xor")
    p.add_argument("--input-encoding", choices=["bipolar", "binary"], default="bipolar")
    p.add_argument("--mvg", action="store_true", help="modularly-varying goal: switch AND<->OR periodically")
    p.add_argument("--switch-interval", type=int, default=20)
    # analysis
    p.add_argument("--prune-threshold", type=float, default=0.05)
    p.add_argument("--no-balanced", action="store_true", help="use raw accuracy instead of balanced (chance=50%%)")
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--out-dir", default="./runs")
    return p


def main():
    args = build_parser().parse_args()
    cfg = BrainConfig(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out,
                      rnn_iters=args.rnn_iters, use_bias=not args.no_bias,
                      activation=_ACTIVATIONS[args.activation])
    assert cfg.n_in >= tasks.min_inputs(args.task)
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for i in range(args.n_seeds):
        seed = args.seed + i
        best, best_genome = train_seed(cfg, args, seed)
        st = brain_stats(best_genome, cfg, args.prune_threshold)
        print(f"[seed {seed}] best accuracy {best:.3f} | edges {st['n_edges']}/{st['max_edges']}"
              f" | density {st['density']:.1f}% | exc(+) {st['n_exc']} inh(-) {st['n_inh']}")
        dna_path = os.path.join(args.out_dir, f"{args.task}_seed{seed}_best_dna.eqx")
        eqx.tree_serialise_leaves(dna_path, best_genome)
        print(f"[seed {seed}] best DNA saved -> {dna_path}")
        results.append((seed, best))

    if args.n_seeds > 1:
        print("\n=== summary ===")
        for seed, best in results:
            print(f"  seed {seed}: best accuracy {best:.3f}")
        bs, ba = max(results, key=lambda r: r[1])
        print(f"best seed: {bs} ({ba:.3f})  |  mean {np.mean([r[1] for r in results]):.3f}")


if __name__ == "__main__":
    main()
