"""Kashtan-Alon 2005 retina network (population-vectorised, numpy).

Faithful to Kashtan & Alon, PNAS 2005 (verified against the paper's own methods,
PMC1236541) -- NOT Clune 2013. The network is a strictly layered feedforward net
of **hard-threshold** neurons:

  retina(8 pixels) -> 8 -> 4 -> 2 -> 1        (node graph = 8-8-4-2-1)

  * "Connections from the retina were to the first layer only."  -> layers[0] is
    the 8 retina pixels (input), layers[1] is the first NEURON layer of 8.
  * Each connection weight is **-1 or +1** (0 == absent; topology evolves).
  * Each neuron "activates its outputs if the [weighted] sum exceeds a threshold"
    -> a hard threshold unit with output in {0,1}. We store the threshold as a
    bias (bias = -threshold), so a neuron fires iff  (sum + bias) > 0.
  * **Fan-in caps** (paper): <=3 incoming for neurons in layers 1-3, <=2 for the
    single output neuron in layer 4. Stored per block in cfg.fan_in.

Everything is plain numpy, vectorised across the whole population on axis 0:

  weights[l] : (pop, layers[l], layers[l+1])  int8  in {-1,0,+1}   0 == no edge
  biases[l]  : (pop, layers[l+1])             int8  (= -threshold)

`l` runs 0..n_blocks-1 (one block per adjacent layer pair). The retina carries no
bias. Weight magnitudes are fixed at 1, so a "weight mutation" is a sign flip.
"""

from __future__ import annotations

import dataclasses

import numpy as np

WEIGHT_VALUES = (-1, 1)          # allowed connection weights (0 == absent, handled separately)
WEIGHT_MAG_MAX = 1               # |weight| is always 1 in KA
# Threshold (stored as bias = -threshold). With {0,1} activations, +-1 weights and
# fan-in <=3, the raw weighted sum lies in [-3, 3], so a threshold outside that band
# just pins a neuron on/off. Exact allowed range is in KA's SI (not the main text);
# {-3..3} is a faithful reconstruction covering every non-trivial threshold.
BIAS_LO, BIAS_HI = -3, 3
BIAS_VALUES = tuple(range(BIAS_LO, BIAS_HI + 1))

# Per-block fan-in caps for the 8-8-4-2-1 retina net: destination layers 1,2,3 get
# <=3 incoming, the output neuron (layer 4) gets <=2.  (KA 2005, verified.)
RETINA_FAN_IN = (3, 3, 3, 2)


@dataclasses.dataclass(frozen=True)
class NetConfig:
    layers: tuple = (8, 8, 4, 2, 1)      # retina(8) -> 8 -> 4 -> 2 -> 1  (KA 2005)
    fan_in: tuple = RETINA_FAN_IN        # max incoming edges per destination-layer block

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


def _fan_in(cfg: NetConfig, l: int) -> int:
    """Fan-in cap for block l (edges into layer l+1). Defaults to no cap if unset."""
    if not cfg.fan_in or l >= len(cfg.fan_in):
        return cfg.layers[l]            # effectively unconstrained
    return int(cfg.fan_in[l])


def init_population(rng: np.random.Generator, cfg: NetConfig, pop: int,
                    init_density: float = 0.5):
    """Random initial population that RESPECTS the per-layer fan-in caps.

    For every destination neuron we pick a random number k (1..cap) of incoming
    edges from a random subset of its possible sources; each gets a weight drawn
    from {-1,+1}. Biases (= -threshold) start at 0. `init_density` scales the
    expected fan-in: k ~ round(density * cap), clamped to [1, cap]."""
    weights, biases = [], []
    for l in range(cfg.n_blocks):
        ni, no = cfg.layers[l], cfg.layers[l + 1]
        cap = min(_fan_in(cfg, l), ni)
        W = np.zeros((pop, ni, no), dtype=np.int8)
        # target incoming count per destination neuron
        k = int(round(init_density * cap))
        k = min(max(k, 1), cap)
        vals = np.array(WEIGHT_VALUES, dtype=np.int8)
        for p in range(pop):
            for j in range(no):
                src = rng.choice(ni, size=k, replace=False)
                W[p, src, j] = vals[rng.integers(0, len(vals), size=k)]
        weights.append(W)
        biases.append(np.zeros((pop, no), dtype=np.int8))
    return weights, biases


def forward(weights, biases, X: np.ndarray, cfg: NetConfig) -> np.ndarray:
    """Population forward pass through hard-threshold neurons.

    X: (B, n_in) with pixel values in {0,1}. Returns activations (pop, B, n_out)
    in {0,1}. A neuron fires (1) iff its weighted input sum plus bias exceeds 0
    (bias = -threshold), matching KA's "activates if the sum exceeds a threshold".

    Uses batched matmul (a @ W) -> multithreaded BLAS; this forward pass is ~95%
    of GA runtime (all pop x 256 patterns every generation)."""
    pop = weights[0].shape[0]
    a = np.broadcast_to(X[None, :, :].astype(np.float32), (pop,) + X.shape)
    for l in range(cfg.n_blocks):
        z = np.matmul(a, weights[l].astype(np.float32))          # (pop, B, out)
        z += biases[l][:, None, :].astype(np.float32)
        a = (z > 0.0).astype(np.float32)                         # hard threshold -> {0,1}
    return a


def decisions(weights, biases, X: np.ndarray, cfg: NetConfig) -> np.ndarray:
    """Predicted class bit (pop, B): the single output neuron's {0,1} activation."""
    out = forward(weights, biases, X, cfg)[:, :, 0]     # single output neuron, already {0,1}
    return out.astype(np.int32)


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
