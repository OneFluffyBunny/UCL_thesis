"""Kashtan-Alon layered feedforward network (population-vectorised, numpy).

The "actual type of network" from the paper: a strictly layered feedforward net
(a node in layer n connects only to layer n-1), steep-tanh units, and DISCRETE
integer weights where an absent connection is simply a zero weight -- so topology
and weight live in one integer matrix per layer and the graph is naturally sparse
(which is what makes Newman's Q meaningful).

Because the search is a discrete mutation GA over small nets, everything is plain
numpy, vectorised across the whole population on the first axis:

  weights[l] : (pop, layers[l],   layers[l+1])  int   values in {-2,-1,0,1,2}, 0 = no edge
  biases[l]  : (pop, layers[l+1])               int   values in {-2,-1,0,1,2}

`l` runs 0..L-2 (one block per adjacent layer pair). Inputs carry no bias.
"""

from __future__ import annotations

import dataclasses

import numpy as np

WEIGHT_VALUES = (-2, -1, 1, 2)          # allowed connection weights (0 == absent, handled separately)
BIAS_VALUES = (-2, -1, 0, 1, 2)         # allowed biases/thresholds
BIAS_LO, BIAS_HI = -2, 2
WEIGHT_MAG_MAX = 2                       # |weight| clamp


@dataclasses.dataclass(frozen=True)
class NetConfig:
    layers: tuple = (8, 8, 4, 2, 1)     # retina architecture: 8 -> 8 -> 4 -> 2 -> 1
    lam: float = 20.0                    # tanh steepness (near-sign decision)

    @property
    def n_blocks(self) -> int:
        return len(self.layers) - 1

    @property
    def offsets(self) -> tuple:
        """Global node-id offset of each layer (for building the whole-net graph)."""
        off, acc = [], 0
        for n in self.layers:
            off.append(acc); acc += n
        return tuple(off)

    @property
    def n_nodes(self) -> int:
        return int(sum(self.layers))

    @property
    def max_edges(self) -> int:
        return int(sum(self.layers[l] * self.layers[l + 1] for l in range(self.n_blocks)))


def init_population(rng: np.random.Generator, cfg: NetConfig, pop: int,
                    init_density: float = 0.5):
    """Random initial population. Each possible connection is present with prob
    `init_density` (weight drawn from {-2,-1,1,2}); biases start at 0."""
    weights, biases = [], []
    for l in range(cfg.n_blocks):
        ni, no = cfg.layers[l], cfg.layers[l + 1]
        present = rng.random((pop, ni, no)) < init_density
        vals = np.array(WEIGHT_VALUES)[rng.integers(0, len(WEIGHT_VALUES), (pop, ni, no))]
        weights.append(np.where(present, vals, 0).astype(np.int8))
        biases.append(np.zeros((pop, no), dtype=np.int8))
    return weights, biases


def forward(weights, biases, X: np.ndarray, cfg: NetConfig) -> np.ndarray:
    """Population forward pass. X: (B, n_in) bipolar. Returns output (pop, B, n_out)."""
    pop = weights[0].shape[0]
    a = np.broadcast_to(X[None, :, :].astype(np.float32), (pop,) + X.shape).copy()
    for l in range(cfg.n_blocks):
        # z[p,b,o] = sum_i a[p,b,i] * W[p,i,o]  (+ bias)
        z = np.einsum("pbi,pio->pbo", a, weights[l].astype(np.float32))
        z += biases[l][:, None, :].astype(np.float32)
        a = np.tanh(cfg.lam * z)
    return a


def decisions(weights, biases, X: np.ndarray, cfg: NetConfig) -> np.ndarray:
    """Predicted class bit (pop, B): output neuron fires (>0) -> 1, else 0."""
    out = forward(weights, biases, X, cfg)[:, :, 0]     # single output neuron
    return (out > 0).astype(np.int32)


def raw_accuracy(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-individual fraction of correct answers (pop,) over ALL patterns.

    This is Kashtan-Alon's own performance measure: "the percentage of correct
    answers" across every input pattern, with no class balancing. It is the
    paper-faithful default. Beware the shortcut trap (CLAUDE.md): a constant
    output can score high on an imbalanced goal like retina/AND."""
    return (pred == y[None, :]).astype(np.float32).mean(axis=1)


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-individual balanced accuracy (pop,). `pred` (pop, B), `y` (B,) in {0,1}.

    Not what Kashtan-Alon used, but the thesis's shortcut-aware convention
    (chance = 0.5 regardless of class imbalance). Available via --fitness balanced."""
    correct = (pred == y[None, :]).astype(np.float32)   # (pop, B)
    pos, neg = (y == 1), (y == 0)
    pos_acc = correct[:, pos].mean(axis=1) if pos.any() else np.zeros(pred.shape[0], np.float32)
    neg_acc = correct[:, neg].mean(axis=1) if neg.any() else np.zeros(pred.shape[0], np.float32)
    return 0.5 * (pos_acc + neg_acc)


_METRICS = {"raw": raw_accuracy, "balanced": balanced_accuracy}


def fitness(weights, biases, X, y, cfg: NetConfig, metric: str = "raw") -> np.ndarray:
    """Performance of every individual against target bits `y`.

    metric="raw" (default) = Kashtan-Alon's fraction-correct; "balanced" = the
    thesis's balanced accuracy (chance 0.5)."""
    return _METRICS[metric](decisions(weights, biases, X, cfg), y)


def gather(weights, biases, idx: np.ndarray):
    """Select individuals by index (idx: (new_pop,)) -> new (weights, biases) lists."""
    return ([w[idx].copy() for w in weights], [b[idx].copy() for b in biases])


def individual(weights, biases, i: int):
    """Extract individual i as (list of int weight matrices, list of int bias vectors)."""
    return ([np.asarray(w[i]) for w in weights], [np.asarray(b[i]) for b in biases])
