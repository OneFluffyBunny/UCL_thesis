"""Experiment 1 — DNA -> static brain via cell-type identities.

The genome (DNA) does NOT store connection weights. It stores:
  * a small set of evolved cell-type identity vectors,
  * one shared connection rule `g` (a small MLP),
  * per-type "abundance" logits that decide how many hidden neurons are of each
    type (these evolve, so a type can grow, shrink, or go extinct).

The brain is then *grown* deterministically from the genome:
  w_ij = g( feat_i , feat_j )     for every allowed (i -> j) pair,
where `feat` = [evolved type embedding | fixed positional code | role one-hot].
`g` is asymmetric in its two arguments, so the brain is a *directed* graph.

Design decisions (see experiment_1 notes / README):
  * Positional code is given to INPUT/OUTPUT neurons only. Hidden neurons are
    type-only, which keeps the encoding compressed (the bottleneck that should
    bias toward modularity). Consequence: effective hidden capacity ~ n_types.
  * Hidden neurons share K cell types; inputs share 1 type, outputs share 1 type.
  * The brain is STATIC: grown once from the genome, then run. No within-life
    plasticity (NDP-style, not LNDP-style).

Conventions:
  * Neuron ordering is [inputs, hidden, outputs].
  * w[i, j] is the weight of the directed edge i -> j, so the synchronous update
    is  a_j <- sigma( sum_i a_i * w[i, j] + b_j )  i.e.  a <- sigma(a @ w + b).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn


# ---------------------------------------------------------------------------
# Static configuration (NOT part of the evolved genome)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BrainConfig:
    n_in: int                       # number of input neurons
    n_hidden: int                   # number of hidden neurons (fixed total)
    n_out: int                      # number of output neurons
    n_types: int                    # K: number of distinct hidden cell-types
    type_dim: int = 4               # dim of an evolved type identity vector
    pos_dim: int = 4                # dim of the fixed positional code (I/O only)
    g_width: int = 16               # hidden width of the connection rule g
    g_depth: int = 1                # hidden layers in g
    rnn_iters: int = 8              # synchronous recurrent passes at inference
    use_bias: bool = True           # per-type bias on each neuron
    activation: Callable = jnp.tanh  # sigma; default tanh, swap later if needed
    # Synaptic gate: |g(feat_i, feat_j)| < w_threshold -> hard 0, so the brain
    # that is EVALUATED is the sparse one (see Genome.build_weights). 0.0 = off,
    # which is the historical behaviour. Distinct from --prune-threshold, which
    # only decides what gets *counted/drawn* and never touches the forward pass.
    w_threshold: float = 0.0

    @property
    def n_total(self) -> int:
        return self.n_in + self.n_hidden + self.n_out

    @property
    def feat_dim(self) -> int:
        # [type embedding | positional code | role one-hot(3)]
        return self.type_dim + self.pos_dim + 3


# ---------------------------------------------------------------------------
# Fixed (non-evolved) positional code -- HyperNEAT-style "molecular address".
# Identical every lifetime, costs zero evolved parameters. Mirrors LNDP's
# `positional_identity`.
# ---------------------------------------------------------------------------

def _sinusoidal(n: int, dim: int) -> jax.Array:
    """Sinusoidal positional code, shape (n, dim), deterministic in the index."""
    pos = jnp.arange(n)[:, None]
    i = jnp.arange(dim)[None, :]
    angle = pos / jnp.power(10000.0, (2 * (i // 2)) / dim)
    return jnp.where(i % 2 == 0, jnp.sin(angle), jnp.cos(angle))


def _role_mask(cfg: BrainConfig) -> jax.Array:
    """Boolean (N, N) mask of allowed directed edges: IH, HH, HO (no self-loops)."""
    N = cfg.n_total
    idx = jnp.arange(N)
    is_in = idx < cfg.n_in
    is_hid = (idx >= cfg.n_in) & (idx < cfg.n_in + cfg.n_hidden)
    is_out = idx >= cfg.n_in + cfg.n_hidden

    ih = is_in[:, None] & is_hid[None, :]      # input -> hidden
    hh = is_hid[:, None] & is_hid[None, :]      # hidden -> hidden
    ho = is_hid[:, None] & is_out[None, :]      # hidden -> output
    mask = ih | hh | ho
    mask = mask & (~jnp.eye(N, dtype=bool))     # no self-loops
    return mask


# ---------------------------------------------------------------------------
# The genome (DNA) -- this is the *only* thing evolution mutates.
# ---------------------------------------------------------------------------

class Genome(eqx.Module):
    hidden_types: jax.Array   # (K, type_dim)   evolved hidden cell-type identities
    in_type: jax.Array        # (type_dim,)     shared identity for input neurons
    out_type: jax.Array       # (type_dim,)     shared identity for output neurons
    abundance: jax.Array      # (K,)            logits -> per-type hidden counts
    type_bias: jax.Array      # (K+2,)          bias per type [K hidden, in, out]
    g: nn.MLP                 # connection rule: [feat_i, feat_j] -> weight

    @classmethod
    def init(cls, key: jax.Array, cfg: BrainConfig) -> "Genome":
        k1, k2, k3, k5 = jr.split(key, 4)
        return cls(
            hidden_types=0.1 * jr.normal(k1, (cfg.n_types, cfg.type_dim)),
            in_type=0.1 * jr.normal(k2, (cfg.type_dim,)),
            out_type=0.1 * jr.normal(k3, (cfg.type_dim,)),
            abundance=jnp.zeros((cfg.n_types,)),          # start ~uniform split
            type_bias=jnp.zeros((cfg.n_types + 2,)),
            g=nn.MLP(
                in_size=2 * cfg.feat_dim,
                out_size=1,
                width_size=cfg.g_width,
                depth=cfg.g_depth,
                final_activation=jnp.tanh,   # bound edge weights to [-1, 1]
                key=k5,
            ),
        )

    # -- developmental map: genome -> per-neuron structure --------------------

    def hidden_type_ids(self, cfg: BrainConfig) -> jax.Array:
        """Assign each hidden slot to a cell-type from the abundance genes.

        The per-type fraction is softplus(abundance) normalised to sum to 1, and
        cumulative boundaries then deterministically bucket each slot. softplus
        (not softmax) is used on purpose: its response is near-linear, so small
        mutations move counts gradually (+/-1 at a time) and a starved type can
        recover -- no exponential extinction trap. The result is a function of K
        continuous numbers -> stays O(K). abundance = 0 -> equal split.
        """
        w = jax.nn.softplus(self.abundance)               # (K,) >= 0
        p = w / jnp.sum(w)                                 # normalise to a fraction
        boundaries = jnp.cumsum(p) * cfg.n_hidden          # (K,)
        slots = jnp.arange(cfg.n_hidden)[:, None]          # (H, 1)
        ids = jnp.sum(slots >= boundaries[None, :], axis=1)  # 0..K
        return jnp.clip(ids, 0, cfg.n_types - 1)           # (H,)

    def type_counts(self, cfg: BrainConfig) -> jax.Array:
        """How many hidden neurons of each type (for logging/inspection)."""
        ids = self.hidden_type_ids(cfg)
        return jnp.bincount(ids, length=cfg.n_types)

    def unique_structure(self, cfg: BrainConfig):
        """Distinct neuron feature *signatures* plus a map neuron -> signature row.

        Hidden neurons are position-free, so all hidden neurons of a type share a
        single signature. Inputs/outputs carry a positional code, so each is its
        own signature. Hence only U = n_in + K + n_out distinct signatures exist
        (<= N), and that is all `g` ever has to be evaluated on.

        Returns (unique_feats (U, feat_dim), neuron_to_row (N,), hid_ids (H,)).
        """
        hid_ids = self.hidden_type_ids(cfg)
        role_in = jnp.array([1.0, 0.0, 0.0])
        role_hid = jnp.array([0.0, 1.0, 0.0])
        role_out = jnp.array([0.0, 0.0, 1.0])

        # one signature per input neuron (position differs)
        in_feats = jnp.concatenate([
            jnp.broadcast_to(self.in_type, (cfg.n_in, cfg.type_dim)),
            _sinusoidal(cfg.n_in, cfg.pos_dim),
            jnp.broadcast_to(role_in, (cfg.n_in, 3)),
        ], axis=1)
        # one signature per hidden cell-type (no position)
        hid_feats = jnp.concatenate([
            self.hidden_types,                                  # (K, type_dim)
            jnp.zeros((cfg.n_types, cfg.pos_dim)),
            jnp.broadcast_to(role_hid, (cfg.n_types, 3)),
        ], axis=1)
        # one signature per output neuron (position differs)
        out_feats = jnp.concatenate([
            jnp.broadcast_to(self.out_type, (cfg.n_out, cfg.type_dim)),
            _sinusoidal(cfg.n_out, cfg.pos_dim),
            jnp.broadcast_to(role_out, (cfg.n_out, 3)),
        ], axis=1)
        unique_feats = jnp.concatenate([in_feats, hid_feats, out_feats], axis=0)

        # map each of the N neurons to its row in unique_feats
        in_map = jnp.arange(cfg.n_in)
        hid_map = cfg.n_in + hid_ids
        out_map = cfg.n_in + cfg.n_types + jnp.arange(cfg.n_out)
        neuron_to_row = jnp.concatenate([in_map, hid_map, out_map])  # (N,)
        return unique_feats, neuron_to_row, hid_ids

    def node_features(self, cfg: BrainConfig):
        """Per-neuron feature matrix (N, feat_dim) -- for inspection/analysis."""
        unique_feats, neuron_to_row, hid_ids = self.unique_structure(cfg)
        return unique_feats[neuron_to_row], hid_ids

    def node_bias(self, cfg: BrainConfig, hid_ids: jax.Array) -> jax.Array:
        # type_bias rows: 0..K-1 hidden types, K -> input, K+1 -> output
        in_b = jnp.full((cfg.n_in,), self.type_bias[cfg.n_types])
        hid_b = self.type_bias[hid_ids]
        out_b = jnp.full((cfg.n_out,), self.type_bias[cfg.n_types + 1])
        return jnp.concatenate([in_b, hid_b, out_b])

    # -- grow the static brain -----------------------------------------------

    def build_weights(self, cfg: BrainConfig):
        """Grow the directed weight matrix w (N, N) from the genome.

        `g` is evaluated once per distinct signature pair (U x U), then the full
        N x N matrix is produced by gathering -- same result as evaluating g on
        every neuron pair, but only U^2 (<= N^2) calls to g.

        If ``cfg.w_threshold > 0`` a hard synaptic gate is applied to g's output:
        any |w| below it becomes exactly 0, and the network really runs on the
        gated matrix. Applied to the U x U block BEFORE the gather, which is
        both cheaper and the honest description of the mechanism -- the gate
        acts on the RULE's output, so a gated pair silences every clone pair
        that shares that signature at once (~N^2/U^2 synapses per decision).

        Note the gate is a step function, so the fitness landscape is piecewise
        constant in the crossing region -- fine for CMA-ES, but it would need a
        straight-through estimator to survive gradients.
        """
        uf, neuron_to_row, hid_ids = self.unique_structure(cfg)
        U = uf.shape[0]
        fi = jnp.broadcast_to(uf[:, None, :], (U, U, cfg.feat_dim))
        fj = jnp.broadcast_to(uf[None, :, :], (U, U, cfg.feat_dim))
        pair = jnp.concatenate([fi, fj], axis=-1)            # (U, U, 2*feat)
        w_u = jax.vmap(jax.vmap(self.g))(pair)[..., 0]       # (U, U)
        if cfg.w_threshold > 0.0:
            w_u = jnp.where(jnp.abs(w_u) < cfg.w_threshold, 0.0, w_u)
        w = w_u[neuron_to_row][:, neuron_to_row]             # (N, N) gather, no g calls
        w = w * _role_mask(cfg)
        return w, hid_ids

    # -- inference (static brain, synchronous recurrent pass) -----------------

    def forward(self, obs: jax.Array, cfg: BrainConfig) -> jax.Array:
        """Run the brain on one observation, return raw output activations."""
        w, hid_ids = self.build_weights(cfg)
        b = self.node_bias(cfg, hid_ids) if cfg.use_bias else jnp.zeros((cfg.n_total,))

        def step(a, _):
            a = a.at[:cfg.n_in].set(obs)                      # clamp inputs each step
            a = cfg.activation(a @ w + b)
            return a, None

        a0 = jnp.zeros((cfg.n_total,))
        a, _ = jax.lax.scan(step, a0, None, cfg.rnn_iters)
        return a[-cfg.n_out:]                                 # output neurons

    def predict(self, obs: jax.Array, cfg: BrainConfig) -> jax.Array:
        """Binary decision: threshold the output activation at 0."""
        return (self.forward(obs, cfg) > 0).astype(jnp.int32)
