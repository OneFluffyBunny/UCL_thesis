"""The Kashtan-Alon 2005 retina genetic algorithm (verified against PMC1236541).

Faithful to the paper's NEURAL-NETWORK experiment (NOT Clune 2013, NOT the paper's
separate NAND-circuit experiment):

  * **Elite strategy** -- the best L genomes replicate UNCHANGED each generation;
    the remaining S-L are offspring of the elite.  (Paper: S=600, L=150.)
  * **Crossover, Pc = 0.5** -- *sexual* reproduction. An offspring recombines two
    elite parents; for each destination neuron it inherits that neuron's entire
    incoming-connection column (weights + threshold) from one parent or the other.
    Recombining whole neuron-columns is what lets modular building blocks
    (left-/right-object detectors) mix -- the mechanism a mutation-only GA lacks,
    and the leading reason our earlier mutation-only port produced a null.
  * **Mutation, Pm = 0.5 per genome** -- a mutated genome gets ONE random edit:
    add an edge (respecting the fan-in cap), remove an edge, flip a weight's sign
    (weights are +-1, so magnitude never changes), or nudge a threshold by +-1.

VERIFIED vs RECONSTRUCTED: S, L, Pc, Pm, the elite strategy, and +-1 weights are
from KA's methods text. The exact *set* of mutation operators and the threshold
range live in the paper's SI (not the main text); the operator mix above is a
faithful reconstruction -- flagged so it can be tightened against the SI later.
"""

from __future__ import annotations

import numpy as np

from model import WEIGHT_VALUES, BIAS_LO, BIAS_HI, _fan_in


# --------------------------------------------------------------------------- #
# Reproduction: elitism + crossover + mutation                                 #
# --------------------------------------------------------------------------- #
def reproduce(rng, weights, biases, fit, cfg, n_elite=150, pc=0.5, pm=0.5):
    """Build the next generation. Returns fresh (weights, biases) lists.

    Top `n_elite` genomes are carried over unchanged (elite strategy). The other
    pop-n_elite slots are offspring: two elite parents -> crossover (prob pc) or
    clone -> mutation (prob pm)."""
    pop = fit.shape[0]
    n_elite = min(n_elite, pop)
    order = np.argsort(fit, kind="stable")[::-1]
    elite = order[:n_elite]
    n_off = pop - n_elite

    # Parents are drawn from the elite pool (offspring of the elite).
    pa = elite[rng.integers(0, n_elite, size=n_off)]
    pb = elite[rng.integers(0, n_elite, size=n_off)]
    do_cross = rng.random(n_off) < pc

    new_w, new_b = [], []
    for l in range(cfg.n_blocks):
        no = cfg.layers[l + 1]
        wA, wB = weights[l][pa], weights[l][pb]          # (n_off, ni, no)
        bA, bB = biases[l][pa], biases[l][pb]            # (n_off, no)
        # per-destination-neuron parent choice; non-crossover offspring clone A.
        selA = rng.random((n_off, no)) < 0.5
        selA |= ~do_cross[:, None]
        new_w.append(np.where(selA[:, None, :], wA, wB).astype(np.int8))
        new_b.append(np.where(selA, bA, bB).astype(np.int8))

    mutate(rng, new_w, new_b, cfg, pm)                   # Pm per genome, in place

    # Prepend the untouched elites -> full population of size `pop`.
    for l in range(cfg.n_blocks):
        new_w[l] = np.concatenate([weights[l][elite].copy(), new_w[l]], axis=0)
        new_b[l] = np.concatenate([biases[l][elite].copy(), new_b[l]], axis=0)
    return new_w, new_b


# --------------------------------------------------------------------------- #
# Mutation: one random edit per selected genome                                #
# --------------------------------------------------------------------------- #
def mutate(rng, weights, biases, cfg, pm=0.5):
    """With probability `pm`, apply ONE random mutation to each genome, in place."""
    pop = weights[0].shape[0]
    for p in np.nonzero(rng.random(pop) < pm)[0].tolist():
        t = int(rng.integers(0, 4))
        if t == 0:
            _add_edge(rng, weights, cfg, p)
        elif t == 1:
            _remove_edge(rng, weights, cfg, p)
        elif t == 2:
            _flip_weight(rng, weights, cfg, p)
        else:
            _change_threshold(rng, biases, cfg, p)
    return weights, biases


def _add_edge(rng, weights, cfg, p):
    """Add one edge into a destination neuron that is below its fan-in cap."""
    elig = []                                   # (block, dst) with room + a free source
    for l in range(cfg.n_blocks):
        W = weights[l][p]
        cap = _fan_in(cfg, l)
        fanin = np.count_nonzero(W, axis=0)     # per destination neuron
        for j in range(W.shape[1]):
            if fanin[j] < cap and np.any(W[:, j] == 0):
                elig.append((l, j))
    if not elig:
        return
    l, j = elig[int(rng.integers(0, len(elig)))]
    zero_src = np.nonzero(weights[l][p][:, j] == 0)[0]
    i = int(zero_src[int(rng.integers(0, len(zero_src)))])
    weights[l][p, i, j] = WEIGHT_VALUES[int(rng.integers(0, len(WEIGHT_VALUES)))]


def _present_edges(weights, cfg, p):
    edges = []
    for l in range(cfg.n_blocks):
        rows, cols = np.nonzero(weights[l][p])
        edges.extend((l, int(i), int(j)) for i, j in zip(rows, cols))
    return edges


def _remove_edge(rng, weights, cfg, p):
    edges = _present_edges(weights, cfg, p)
    if not edges:
        return
    l, i, j = edges[int(rng.integers(0, len(edges)))]
    weights[l][p, i, j] = 0


def _flip_weight(rng, weights, cfg, p):
    edges = _present_edges(weights, cfg, p)
    if not edges:
        return
    l, i, j = edges[int(rng.integers(0, len(edges)))]
    weights[l][p, i, j] = -weights[l][p, i, j]          # +-1 sign flip


def _change_threshold(rng, biases, cfg, p):
    l = int(rng.integers(0, cfg.n_blocks))
    j = int(rng.integers(0, biases[l].shape[1]))
    step = 1 if rng.random() < 0.5 else -1
    biases[l][p, j] = int(np.clip(int(biases[l][p, j]) + step, BIAS_LO, BIAS_HI))
