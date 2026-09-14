"""Tests for ga.py. Run from experiments/experiment_1: python test_ga.py

The one that matters most is `test_blocks_line_up_with_genome`: crossover is only
Kashtan-Alon-like if a block really is one functional unit of the genome, and
the ids are written through a pytree, so this reshapes them back and checks each
leaf, rather than trusting the leaf order.
"""

from __future__ import annotations

import equinox as eqx
import evosax as ex
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ga import KAGeneticAlgorithm, gene_blocks
from model import BrainConfig, Genome

CFG = BrainConfig(n_in=8, n_hidden=24, n_out=1, n_types=8,
                  synaptic_budget=6.0, shrink=0.9)


def _setup():
    params, _ = eqx.partition(Genome.init(jr.PRNGKey(0), CFG), eqx.is_inexact_array)
    rs = ex.ParameterReshaper(params)
    return params, rs, gene_blocks(params, rs, CFG)


def test_blocks_line_up_with_genome():
    params, rs, ids = _setup()
    K, W = CFG.n_types, CFG.g_width
    assert ids.shape == (rs.total_params,) == (443,)
    back = rs.reshape_single(ids.astype(jnp.float32))
    for t in range(K):
        assert np.all(np.asarray(back.hidden_types[t]) == t)
        assert float(back.abundance[t]) == t and float(back.type_bias[t]) == t
    assert np.all(np.asarray(back.in_type) == K) and float(back.type_bias[K]) == K
    assert np.all(np.asarray(back.out_type) == K + 1) and float(back.type_bias[K + 1]) == K + 1
    l0, l1 = back.g.layers
    for u in range(W):
        assert np.all(np.asarray(l0.weight[u]) == K + 2 + u)
        assert float(l0.bias[u]) == K + 2 + u and float(l1.weight[0, u]) == K + 2 + u
    assert float(l1.bias[0]) == K + 2 + W
    sizes = np.bincount(np.asarray(ids))
    assert sizes.tolist() == [6] * K + [5, 5] + [24] * W + [1]


def _ga(ids, **kw):
    return KAGeneticAlgorithm(popsize=40, num_dims=ids.shape[0], sigma_init=1.0,
                              block_ids=ids, n_elite=10, **kw)


def _next(ga, seed=1):
    st = ga.initialize(jr.PRNGKey(seed))
    x, st = ga.ask(None, st)
    fit = jr.normal(jr.PRNGKey(seed + 1), (ga.popsize,))
    return np.asarray(x), np.asarray(fit), np.asarray(ga.tell(x, fit, st).pop)


def test_elites_are_the_lowest_fitness_and_unchanged():
    _, _, ids = _setup()
    x, fit, new = _next(_ga(ids))
    assert np.array_equal(new[:10], x[np.argsort(fit)[:10]])


def test_no_crossover_no_mutation_clones_an_elite():
    _, _, ids = _setup()
    x, fit, new = _next(_ga(ids, pc=0.0, pm=0.0))
    elite = x[np.argsort(fit)[:10]]
    for child in new[10:]:
        assert any(np.array_equal(child, e) for e in elite)


def test_mutation_touches_exactly_one_gene():
    _, _, ids = _setup()
    x, fit, new = _next(_ga(ids, pc=0.0, pm=1.0))
    elite = x[np.argsort(fit)[:10]]
    for child in new[10:]:
        assert min(int(np.sum(child != e)) for e in elite) == 1


def test_crossover_moves_whole_blocks():
    _, _, ids = _setup()
    ids_np = np.asarray(ids)
    x, fit, new = _next(_ga(ids, pc=1.0, pm=0.0))
    elite = x[np.argsort(fit)[:10]]
    n_mixed = 0
    for child in new[10:]:
        sources = set()
        for b in range(ids_np.max() + 1):
            m = ids_np == b
            hits = [i for i, e in enumerate(elite) if np.array_equal(child[m], e[m])]
            assert hits, f"block {b} is not a whole block of any elite"
            sources.add(hits[0])
        n_mixed += len(sources) > 1
    assert n_mixed > 0, "pc=1 produced no recombined offspring"


if __name__ == "__main__":
    test_blocks_line_up_with_genome()
    test_elites_are_the_lowest_fitness_and_unchanged()
    test_no_crossover_no_mutation_clones_an_elite()
    test_mutation_touches_exactly_one_gene()
    test_crossover_moves_whole_blocks()
    print("all tests passed")
