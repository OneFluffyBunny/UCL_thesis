"""Shared DIRECT-encoding model — the SINGLE source of truth for the direct net.

The genome IS the weight vector: every allowed edge is its own free parameter,
with no sharing, no cell-types, no rule. This is deliberately shared, the same
way `shared_tasks.py` is, so that every experiment that uses a direct encoding
trains the *byte-identical* network and they can never silently diverge:

  * experiment 2 — evolves this genome with CMA-ES (the EC control).
  * experiment 3 — optimises this genome with gradient descent (the GD control).

Both differ only in the OPTIMISER; the model, topology and dynamics are this file.
Do NOT copy this class into an experiment — import it through the thin `model.py`
shim in each experiment directory and edit it here.

Contrast with experiment 1 (`experiment_1/model.py`), which grows the brain from
a compressed genome via a shared connection rule  w_ij = g(feat_i, feat_j)
(O(K) params reused across edges — a low-dim manifold of *regular* networks).

Everything else (fixed neuron count, [inputs, hidden, outputs] ordering, allowed
edges IH|HH|HO with no self-loops, synchronous recurrent inference, tanh output)
is kept identical to experiment 1 so the encodings are directly comparable.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


@dataclasses.dataclass(frozen=True)
class BrainConfig:
    """Static (non-evolved) architecture. No K / type_dim / g here — direct
    encoding has no encoding hyper-parameters, just the topology."""
    n_in: int
    n_hidden: int
    n_out: int
    rnn_iters: int = 8
    use_bias: bool = True
    activation: Callable = jnp.tanh

    @property
    def n_total(self) -> int:
        return self.n_in + self.n_hidden + self.n_out


def role_mask(cfg: BrainConfig) -> jax.Array:
    """Boolean (N, N) mask of allowed directed edges: IH, HH, HO (no self-loops).
    Identical rule to experiment_1.model._role_mask."""
    N = cfg.n_total
    idx = jnp.arange(N)
    is_in = idx < cfg.n_in
    is_hid = (idx >= cfg.n_in) & (idx < cfg.n_in + cfg.n_hidden)
    is_out = idx >= cfg.n_in + cfg.n_hidden
    ih = is_in[:, None] & is_hid[None, :]        # input  -> hidden
    hh = is_hid[:, None] & is_hid[None, :]        # hidden -> hidden
    ho = is_hid[:, None] & is_out[None, :]        # hidden -> output
    return (ih | hh | ho) & (~jnp.eye(N, dtype=bool))


class DirectGenome(eqx.Module):
    """The genome is literally the network's free parameters.

    `weights` holds one evolved number per allowed edge; `edge_rows`/`edge_cols`
    are the (static) source/dest indices that scatter those numbers back into the
    full (N, N) matrix. `bias` is one evolved value per non-input neuron.
    """
    weights: jax.Array      # (E,)  evolved weight per allowed edge
    bias: jax.Array         # (n_hidden + n_out,)  evolved bias (inputs are clamped)
    edge_rows: jax.Array    # (E,)  STATIC source index of each allowed edge
    edge_cols: jax.Array    # (E,)  STATIC dest index of each allowed edge

    @classmethod
    def init(cls, key: jax.Array, cfg: BrainConfig) -> "DirectGenome":
        mask = role_mask(cfg)
        rows, cols = jnp.where(mask)                 # concrete at init time
        E = int(rows.shape[0])
        k1, _ = jr.split(key)
        return cls(
            weights=0.1 * jr.normal(k1, (E,)),        # small random init, like exp 1
            bias=jnp.zeros((cfg.n_hidden + cfg.n_out,)),
            edge_rows=rows,
            edge_cols=cols,
        )

    @property
    def n_edges(self) -> int:
        return int(self.weights.shape[0])

    def build_weights(self, cfg: BrainConfig):
        """Scatter the free weights into the (N, N) directed weight matrix."""
        N = cfg.n_total
        w = jnp.zeros((N, N)).at[self.edge_rows, self.edge_cols].set(self.weights)
        return w

    def node_bias(self, cfg: BrainConfig) -> jax.Array:
        # inputs are clamped every step, so they carry no bias
        return jnp.concatenate([jnp.zeros((cfg.n_in,)), self.bias])

    def forward(self, obs: jax.Array, cfg: BrainConfig) -> jax.Array:
        """Synchronous recurrent pass — identical dynamics to experiment 1.

        Fully differentiable in `weights`/`bias` (matmul + activation + scan),
        which is what lets experiment 3 backprop through it."""
        w = self.build_weights(cfg)
        b = self.node_bias(cfg) if cfg.use_bias else jnp.zeros((cfg.n_total,))

        def step(a, _):
            a = a.at[:cfg.n_in].set(obs)              # clamp inputs each step
            a = cfg.activation(a @ w + b)
            return a, None

        a0 = jnp.zeros((cfg.n_total,))
        a, _ = jax.lax.scan(step, a0, None, cfg.rnn_iters)
        return a[-cfg.n_out:]

    def predict(self, obs: jax.Array, cfg: BrainConfig) -> jax.Array:
        return (self.forward(obs, cfg) > 0).astype(jnp.int32)
