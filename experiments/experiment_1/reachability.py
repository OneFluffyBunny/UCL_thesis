"""Reachability / basin-of-attraction probe for experiment 1.

`oracle.py` showed a perfect genome for `left` EXISTS (representability is fine).
This asks the next question: how LARGE is the region of DNA-space around that
optimum from which ordinary CMA-ES can still climb back to it?

Procedure (no "reverse search" -- we run the *normal* uphill optimiser, only the
starting point changes):

  1. Build the oracle DNA (score 1.000) and flatten it to a parameter vector.
  2. For a sweep of perturbation radii r:
       - draw `n_starts` random starts = oracle + r * N(0, I)   (r = per-coord std,
         directly comparable to CMA-ES sigma_init and the training mutation scale);
       - run CMA-ES with its mean seeded at each start;
       - record the best balanced accuracy it reaches.
  3. Report, per radius, the fraction of starts that RECOVER (>= threshold) and
     the mean/min best fitness. A `rand` baseline (CMA-ES from fresh random
     genomes) anchors the wide monotone basin (~0.929).

The radius where the recovery fraction crosses ~0.5 is the approximate basin
radius of the optimum. Small basin -> the optimum is a needle -> the training
failure is reachability, measured.

Run:  python reachability.py            # left task, g_width=16
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import evosax as ex
import numpy as np

import tasks
from model import Genome
from oracle import left_config, build_oracle_left, fit_g_ols


# ---------------------------------------------------------------------------
# Build the oracle DNA (reuse oracle.py's hand-wiring + OLS fit of g)
# ---------------------------------------------------------------------------

def build_oracle_dna(cfg):
    base = Genome.init(jr.PRNGKey(0), cfg)
    w_u_target, type_bias = build_oracle_left(cfg)
    g_fit, resid = fit_g_ols(base, cfg, w_u_target)
    oracle_dna = eqx.tree_at(lambda m: (m.g, m.type_bias), base, (g_fit, type_bias))
    return oracle_dna, resid


# ---------------------------------------------------------------------------
# Fitness (balanced accuracy on `left`, full enumeration) over a flat batch
# ---------------------------------------------------------------------------

def make_eval(cfg, static, unravel, X_enc, y):
    def eval_one(flat):
        genome = eqx.combine(unravel(flat), static)
        preds = jax.vmap(lambda o: genome.predict(o, cfg))(X_enc)[:, 0]
        correct = (preds == y).astype(jnp.float32)
        yf = y.astype(jnp.float32)
        pos = jnp.sum(correct * yf) / (jnp.sum(yf) + 1e-8)
        neg = jnp.sum(correct * (1.0 - yf)) / (jnp.sum(1.0 - yf) + 1e-8)
        return 0.5 * (pos + neg)
    return jax.jit(jax.vmap(eval_one))


# ---------------------------------------------------------------------------
# One CMA-ES run from a given starting mean -> best fitness reached
# ---------------------------------------------------------------------------

def run_cma_from(start_flat, eval_pop, key, gens, popsize, sigma_init, stop=0.9999):
    strat = ex.Strategies["CMA_ES"](popsize=popsize, num_dims=int(start_flat.shape[0]),
                                    sigma_init=sigma_init)
    esp = strat.default_params
    key, ik = jr.split(key)
    state = strat.initialize(ik, esp)
    state = state.replace(mean=start_flat)        # seed the search center at the start
    shaper = ex.FitnessShaper(maximize=True)

    best = -1.0
    for _ in range(gens):
        key, ak = jr.split(key)
        x, state = strat.ask(ak, state, esp)
        fit = eval_pop(x)
        state = strat.tell(x, shaper.apply(x, fit), state, esp)
        best = max(best, float(fit.max()))
        if best >= stop:
            break
    return best


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def probe(cfg, radii, n_starts, gens, popsize, sigma_init, seed, recover_thr,
          n_rand):
    oracle_dna, resid = build_oracle_dna(cfg)
    params0, static = eqx.partition(oracle_dna, eqx.is_inexact_array)
    flat0, unravel = jax.flatten_util.ravel_pytree(params0)

    X = tasks.all_binary_inputs(cfg.n_in)
    X_enc = X * 2.0 - 1.0
    y = tasks.targets("left", "xor", X)
    eval_pop = make_eval(cfg, static, unravel, X_enc, y)

    # sanity: the unperturbed oracle really scores 1.0 through this eval path
    oracle_acc = float(eval_pop(flat0[None, :])[0])
    print(f"g_width = {cfg.g_width} | dims = {flat0.shape[0]} | OLS resid = {resid:.4f}")
    print(f"oracle DNA balanced accuracy (sanity) = {oracle_acc:.3f}\n")

    key = jr.PRNGKey(seed)
    rows = []
    for r in radii:
        bests = []
        for _ in range(n_starts):
            key, pk, rk = jr.split(key, 3)
            start = flat0 if r == 0.0 else flat0 + r * jr.normal(pk, flat0.shape)
            bests.append(run_cma_from(start, eval_pop, rk, gens, popsize, sigma_init))
        bests = np.array(bests)
        frac = float((bests >= recover_thr).mean())
        rows.append((f"{r:.2f}", frac, bests.mean(), bests.min(), bests.max()))
        print(f"  r={r:5.2f} | recover {frac*100:5.1f}% | "
              f"best mean {bests.mean():.3f} min {bests.min():.3f} max {bests.max():.3f}")

    # random baseline: CMA-ES from fresh random genomes (the real training start)
    if n_rand > 0:
        bests = []
        for i in range(n_rand):
            key, gk, rk = jr.split(key, 3)
            rand_genome = Genome.init(gk, cfg)
            rp, _ = eqx.partition(rand_genome, eqx.is_inexact_array)
            rflat, _ = jax.flatten_util.ravel_pytree(rp)
            bests.append(run_cma_from(rflat, eval_pop, rk, gens, popsize, sigma_init))
        bests = np.array(bests)
        frac = float((bests >= recover_thr).mean())
        rows.append(("rand", frac, bests.mean(), bests.min(), bests.max()))
        print(f"  r= rand | recover {frac*100:5.1f}% | "
              f"best mean {bests.mean():.3f} min {bests.min():.3f} max {bests.max():.3f}")

    return rows, oracle_acc


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(rows, save_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    swept = [row for row in rows if row[0] != "rand"]
    rs = [float(r[0]) for r in swept]
    frac = [r[1] for r in swept]
    mean_best = [r[2] for r in swept]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rs, frac, "o-", color="#2ECC71", label="recovery fraction")
    ax1.set_xlabel("perturbation radius r (per-coordinate std)")
    ax1.set_ylabel("fraction recovered to optimum", color="#2ECC71")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor="#2ECC71")

    ax2 = ax1.twinx()
    ax2.plot(rs, mean_best, "s--", color="#4A90D9", label="mean best fitness")
    ax2.set_ylabel("mean best balanced accuracy", color="#4A90D9")
    ax2.tick_params(axis="y", labelcolor="#4A90D9")

    rand = [row for row in rows if row[0] == "rand"]
    if rand:
        ax2.axhline(rand[0][2], color="#E74C3C", ls=":", lw=1.5,
                    label=f"random-start baseline ({rand[0][2]:.3f})")
        ax2.legend(loc="lower left", fontsize=8)

    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nplot saved -> {save_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="reachability / basin probe for `left`")
    p.add_argument("--g-width", type=int, default=16)
    p.add_argument("--radii", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    p.add_argument("--n-starts", type=int, default=8, help="random starts per radius")
    p.add_argument("--gens", type=int, default=120, help="CMA-ES generations per start")
    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--sigma-init", type=float, default=0.1, help="CMA-ES local search scale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recover-thr", type=float, default=0.999)
    p.add_argument("--n-rand", type=int, default=8, help="random-start baseline runs (0=skip)")
    p.add_argument("--out", default="./runs/reachability_left.png")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    cfg = left_config(g_width=args.g_width)
    rows, oracle_acc = probe(cfg, args.radii, args.n_starts, args.gens, args.popsize,
                             args.sigma_init, args.seed, args.recover_thr, args.n_rand)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot(rows, args.out,
         f"left basin probe (g_width={cfg.g_width}, {args.n_starts} starts/radius, "
         f"{args.gens} gens)")
    if not args.no_open:
        from visualize import _auto_open
        _auto_open(args.out)


if __name__ == "__main__":
    main()
