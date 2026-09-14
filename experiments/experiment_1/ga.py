"""A Kashtan-Alon-style genetic algorithm for the cell-type genome (2026-09-14).

    python train.py --strategy KA_GA --popsize 600 --ga-elite 150 ...

WHY THIS EXISTS. The FG-vs-MVG study found no MVG modularity effect under CMA-ES
in either encoding. Kashtan & Alon's result was obtained with a GA, and their
mechanism is a claim about how VARIATION is generated: a modular genome can reach
the other goal in few LOCAL mutations, and crossover can recombine intact modules
between parents. CMA-ES has neither -- every sample perturbs every gene, its
"recombination" averages the best samples into one mean, and no individual
survives a generation. This module swaps the optimiser and nothing else, so the
same train.py loop, goal schedule, archive and goal-matched champion apply.

THE OPERATORS, mapped from `kashtan_alon/ga.py` (itself from PMC1236541):

  * ELITE: the best `n_elite` genomes are copied unchanged into the next
    generation; the other `popsize - n_elite` slots are their offspring. Parents
    are drawn uniformly from the elite.                     [KA: 150 of 600]
  * CROSSOVER, probability `pc` per offspring: every GENE BLOCK is inherited
    whole from one parent or the other (50/50); an offspring without crossover
    clones parent A.                                        [KA: pc = 0.5]
    KA's block is a neuron's incoming column (weights + threshold). This genome
    has no per-neuron genes, so the blocks are its functional units -- see
    `gene_blocks`: one per hidden cell type, the input type, the output type,
    one per hidden unit of the rule `g`, and `g`'s output bias.  [our mapping]
  * MUTATION, probability `pm` per offspring: ONE gene, chosen uniformly, gets
    N(0, mut_sigma^2) added.                                [KA: pm = 0.5, one edit]
    KA's weights are +-1, so their edit size has no continuous counterpart;
    `mut_sigma` is our choice.                              [our choice]

API. Mirrors the three evosax calls train.py makes -- `initialize(key, params)`,
`ask(key, state, params)`, `tell(x, fitness, state, params)` -- and, like
evosax after `FitnessShaper(maximize=True)`, `tell` treats LOWER fitness as
better. The initial population is N(0, sigma_init^2) per gene: the same
distribution CMA-ES draws its first generation from (mean 0, sigma_init), so the
two optimisers start from identical conditions.

Elites are re-evaluated every generation. Evaluation is deterministic, so this
only costs time -- and under --mvg it is required: a goal switch changes every
elite's fitness.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


@dataclasses.dataclass
class GAState:
    pop: jax.Array            # (popsize, num_dims) the generation to be evaluated next
    key: jax.Array            # tell() has no key argument in the evosax API


def gene_blocks(params, reshaper, brain_cfg):
    """(num_dims,) int array: the crossover block each flat gene belongs to.

    Built by writing block ids into a pytree of the genome's own shape and
    flattening it with the SAME reshaper train.py uses, so the ids line up with
    the flat vector by construction rather than by assumed leaf order.

    Blocks (K = n_types, W = g_width), for g_depth = 1 only:
      0..K-1    hidden cell type t: hidden_types[t], abundance[t], type_bias[t]
      K         input type:  in_type, type_bias[K]
      K+1       output type: out_type, type_bias[K+1]
      K+2..K+1+W  hidden unit u of g: layer-0 weight row u, layer-0 bias u,
                  layer-1 weight column u (its incoming AND outgoing weights)
      K+2+W     g's output bias
    """
    if brain_cfg.g_depth != 1:
        raise NotImplementedError("gene_blocks maps g_depth = 1 only")
    K, W = brain_cfg.n_types, brain_cfg.g_width
    per_type = jnp.arange(K, dtype=jnp.float32)
    lin0, lin1 = params.g.layers
    ids = eqx.tree_at(
        lambda p: (p.hidden_types, p.abundance, p.type_bias, p.in_type, p.out_type,
                   p.g.layers[0].weight, p.g.layers[0].bias,
                   p.g.layers[1].weight, p.g.layers[1].bias),
        params,
        (jnp.broadcast_to(per_type[:, None], params.hidden_types.shape),
         per_type,
         jnp.concatenate([per_type, jnp.array([K, K + 1], jnp.float32)]),
         jnp.full(params.in_type.shape, K, jnp.float32),
         jnp.full(params.out_type.shape, K + 1, jnp.float32),
         jnp.broadcast_to((K + 2 + jnp.arange(W, dtype=jnp.float32))[:, None],
                          lin0.weight.shape),
         K + 2 + jnp.arange(W, dtype=jnp.float32),
         jnp.broadcast_to((K + 2 + jnp.arange(W, dtype=jnp.float32))[None, :],
                          lin1.weight.shape),
         jnp.full(lin1.bias.shape, K + 2 + W, jnp.float32)))
    flat = reshaper.flatten_single(ids)
    return jnp.asarray(flat, dtype=jnp.int32)


class KAGeneticAlgorithm:
    def __init__(self, popsize, num_dims, sigma_init, block_ids,
                 n_elite=150, pc=0.5, pm=0.5, mut_sigma=0.5):
        if not 0 < n_elite < popsize:
            raise ValueError(f"need 0 < n_elite ({n_elite}) < popsize ({popsize})")
        self.popsize, self.num_dims, self.sigma_init = popsize, num_dims, sigma_init
        self.block_ids = jnp.asarray(block_ids)
        self.n_blocks = int(self.block_ids.max()) + 1
        self.n_elite, self.pc, self.pm, self.mut_sigma = n_elite, pc, pm, mut_sigma
        self.default_params = None
        self._reproduce = jax.jit(self._reproduce_impl)

    def initialize(self, key, params=None):
        k_pop, k_state = jr.split(key)
        pop = self.sigma_init * jr.normal(k_pop, (self.popsize, self.num_dims))
        return GAState(pop=pop, key=k_state)

    def ask(self, key, state, params=None):
        return state.pop, state

    def tell(self, x, fitness, state, params=None):
        key, sub = jr.split(state.key)
        return GAState(pop=self._reproduce(sub, x, fitness), key=key)

    def _reproduce_impl(self, key, x, fitness):
        n_off = self.popsize - self.n_elite
        k_a, k_b, k_blk, k_cx, k_gene, k_mut, k_step = jr.split(key, 7)
        order = jnp.argsort(fitness)                 # LOWER is better (evosax convention)
        elite = x[order[:self.n_elite]]
        pa = elite[jr.randint(k_a, (n_off,), 0, self.n_elite)]
        pb = elite[jr.randint(k_b, (n_off,), 0, self.n_elite)]
        # Whole blocks from parent B, only for offspring that cross over.
        from_b = jr.bernoulli(k_blk, 0.5, (n_off, self.n_blocks))[:, self.block_ids]
        crosses = jr.bernoulli(k_cx, self.pc, (n_off, 1))
        child = jnp.where(from_b & crosses, pb, pa)
        # ONE gene per mutated offspring.
        gene = jr.randint(k_gene, (n_off,), 0, self.num_dims)
        step = self.mut_sigma * jr.normal(k_step, (n_off,)) * jr.bernoulli(k_mut, self.pm, (n_off,))
        child = child.at[jnp.arange(n_off), gene].add(step)
        return jnp.concatenate([elite, child], axis=0)
