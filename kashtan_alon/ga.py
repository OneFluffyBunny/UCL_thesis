"""Mutation-based genetic algorithm (the Kashtan-Alon search).

Discrete, asexual (mutation-only) reproduction with the paper's operators:
  * 20% of networks add one connection;
  * 20% of networks remove one connection;
  * each present connection: prob 2/n of a weight +/-1 step (n = #connections);
  * each node: prob 1/24 of a bias +/-1 step.
Weights are kept in {-2,-1,1,2} (a +/-1 step that would hit 0 flips sign instead;
steps past +/-2 clamp), biases in {-2..2}. Selection is single-objective
tournament + elitism (performance only -- no connection-cost term; that is
Clune's modularity driver, not Kashtan-Alon's MVG).
"""

from __future__ import annotations

import numpy as np

from model import WEIGHT_VALUES, WEIGHT_MAG_MAX, BIAS_LO, BIAS_HI


def _mutate_weights(rng, weights, pop):
    n_conn = np.zeros(pop)
    for W in weights:
        n_conn += np.count_nonzero(W, axis=(1, 2))
    n_conn = np.maximum(n_conn, 1.0)
    for W in weights:
        present = W != 0
        prob = (2.0 / n_conn)[:, None, None]
        do = present & (rng.random(W.shape) < prob)
        steps = rng.choice(np.array([-1, 1], np.int8), size=W.shape)
        newvals = (W + do * steps).astype(np.int16)
        zero_created = do & (newvals == 0)
        newvals = np.where(zero_created, -W.astype(np.int16), newvals)   # 1<->-1 instead of 0
        newvals = np.clip(newvals, -WEIGHT_MAG_MAX, WEIGHT_MAG_MAX).astype(np.int8)
        W[do] = newvals[do]


def _mutate_biases(rng, biases, bias_prob):
    for b in biases:
        do = rng.random(b.shape) < bias_prob
        steps = rng.choice(np.array([-1, 1], np.int8), size=b.shape)
        b[:] = np.clip(b + do * steps, BIAS_LO, BIAS_HI).astype(np.int8)


def _pick_slot(rng, per_block_counts):
    """Pick a random flat slot across blocks; return (block, rank-within-block)."""
    total = int(sum(per_block_counts))
    if total == 0:
        return None
    r = int(rng.integers(0, total))
    for l, c in enumerate(per_block_counts):
        if r < c:
            return l, r
        r -= c
    return None  # unreachable


def _add_remove(rng, weights, pop, p_add, p_remove, cfg):
    add = np.nonzero(rng.random(pop) < p_add)[0]
    for p in add.tolist():
        counts = [int(np.count_nonzero(weights[l][p] == 0)) for l in range(cfg.n_blocks)]
        pick = _pick_slot(rng, counts)
        if pick is None:
            continue
        l, r = pick
        i, j = np.argwhere(weights[l][p] == 0)[r]
        weights[l][p, i, j] = WEIGHT_VALUES[int(rng.integers(0, len(WEIGHT_VALUES)))]
    rem = np.nonzero(rng.random(pop) < p_remove)[0]
    for p in rem.tolist():
        counts = [int(np.count_nonzero(weights[l][p])) for l in range(cfg.n_blocks)]
        pick = _pick_slot(rng, counts)
        if pick is None:
            continue
        l, r = pick
        i, j = np.argwhere(weights[l][p] != 0)[r]
        weights[l][p, i, j] = 0


def mutate(rng, weights, biases, cfg, p_add=0.2, p_remove=0.2, bias_prob=1.0 / 24.0):
    """Mutate the whole population in place."""
    pop = weights[0].shape[0]
    _mutate_weights(rng, weights, pop)
    _mutate_biases(rng, biases, bias_prob)
    _add_remove(rng, weights, pop, p_add, p_remove, cfg)
    return weights, biases


def tournament_select(rng, fit, n_out, k=3):
    """Return `n_out` parent indices by k-way tournament (higher fitness wins)."""
    pop = fit.shape[0]
    contenders = rng.integers(0, pop, size=(n_out, k))
    return contenders[np.arange(n_out), np.argmax(fit[contenders], axis=1)]
