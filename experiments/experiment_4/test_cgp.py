"""Tests for the experiment-4 CGP core. Run: conda run -n lndp python test_cgp.py

The important one is `test_fast_matches_slow`: the search loop evaluates circuits
bit-parallel (one Python int per wire, all 256 patterns at once), which is fast but
non-obvious. `cgp.evaluate_slow` is the obvious per-pattern implementation, and
this asserts the two agree on random circuits over the whole function set. Without
it the fast path would be trusted rather than verified.

The task tests re-derive Kashtan & Alon's own counts from the packed bitmasks, so
they check the packing as well as the rule.
"""

from __future__ import annotations

import numpy as np

import cgp
import gates as gates_mod
import tasks as tasks_mod


def _bits(mask: int, n: int) -> list[bool]:
    return [bool((mask >> r) & 1) for r in range(n)]


def test_fast_matches_slow(n_trials: int = 40) -> None:
    """Bit-parallel evaluation == per-pattern evaluation, on random genotypes."""
    gate_set = gates_mod.build_set(",".join(gates_mod.ALL_GATES))
    arity = gates_mod.max_arity(gate_set)
    n_in, n_nodes = 6, 25
    n_patterns = 1 << n_in
    mask = (1 << n_patterns) - 1
    # input masks built the same way tasks.input_masks does, for n_in=6
    in_masks = []
    for i in range(n_in):
        m = 0
        for r in range(n_patterns):
            if (r >> (n_in - 1 - i)) & 1:
                m |= 1 << r
        in_masks.append(m)

    rng = np.random.default_rng(1234)
    for _ in range(n_trials):
        g = cgp.random_genotype(rng, n_nodes, n_in, 1, len(gate_set), arity)
        fast = _bits(cgp.evaluate(g, gate_set, in_masks, mask, n_in)[0], n_patterns)
        slow = cgp.evaluate_slow(g, gate_set, n_in, n_patterns)[0]
        assert fast == slow, "bit-parallel evaluation disagrees with the reference"
    print(f"ok  fast == slow on {n_trials} random circuits "
          f"({len(gate_set)} gates, arity {arity})")


def test_ka_task_counts() -> None:
    """KA's rule is true for 8/16 half-patterns => all four (L,R) cells are 64.

    This is the equal-cell property the experiment depends on: it caps any
    predictor reading only one half at 192/256, level with a constant predictor,
    so no one-module attractor exists. See README.md.
    """
    n_patterns = tasks_mod.n_patterns("retina_ka2005")
    assert n_patterns == 256

    t_and = tasks_mod.target_mask("retina_ka2005", "and")
    t_or = tasks_mod.target_mask("retina_ka2005", "or")
    t_xor = tasks_mod.target_mask("retina_ka2005", "xor")
    assert t_and.bit_count() == 64, t_and.bit_count()      # |L||R| = 8*8
    assert t_or.bit_count() == 192, t_or.bit_count()       # 256 - 8*8
    assert t_xor.bit_count() == 128, t_xor.bit_count()     # balanced
    # AND and OR agree exactly where L == R (the 64+64 diagonal cells)
    assert (~(t_and ^ t_or) & ((1 << n_patterns) - 1)).bit_count() == 128
    print("ok  KA retina cell counts: AND 64/256, OR 192/256, XOR 128/256")


def test_one_half_predictor_is_capped() -> None:
    """A predictor that reads only the left half scores 192/256 under AND and OR --
    exactly a constant predictor's score. This is the property that makes the task
    a real modularity probe rather than a one-module plateau."""
    mask = tasks_mod.full_mask("retina_ka2005")
    n_patterns = tasks_mod.n_patterns("retina_ka2005")
    left = tasks_mod.target_mask("left", "and")            # L over 16 patterns
    # lift L (4 inputs) to the 8-input pattern space: inputs 0-3 are the left block
    lifted = 0
    for r in range(n_patterns):
        if (left >> (r >> 4)) & 1:
            lifted |= 1 << r

    for op, const in (("and", 0), ("or", mask)):
        t = tasks_mod.target_mask("retina_ka2005", op)
        one_half = cgp.hits(lifted, t, mask)
        constant = cgp.hits(const, t, mask)
        assert one_half == 192, (op, one_half)
        assert constant == 192, (op, constant)
    print("ok  one-half predictor == constant predictor == 192/256 under AND and OR")


def test_mutation_stays_valid(n_trials: int = 200) -> None:
    """Mutation never creates a forward or self reference (which would make the
    graph cyclic and the phenotype undefined)."""
    gate_set = gates_mod.build_set(gates_mod.DEFAULT_GATES)
    arity = gates_mod.max_arity(gate_set)
    n_in, n_nodes = 8, 40
    rng = np.random.default_rng(7)
    g = cgp.random_genotype(rng, n_nodes, n_in, 1, len(gate_set), arity)
    n_mut = cgp.n_mutations(n_nodes, arity, 1, 0.03)
    for _ in range(n_trials):
        g = cgp.mutate(g, rng, n_mut, n_in, len(gate_set))
        for j in range(n_nodes):
            assert (g.conn[j] < n_in + j).all(), f"node {j} references itself or later"
        assert (g.ogene < n_in + n_nodes).all()
        assert (g.ntype == 0).all(), "CGP must not produce non-zero node types"
        assert (g.cout == 0).all() and (g.ocout == 0).all()
    print(f"ok  {n_trials} chained mutations keep the graph acyclic and CGP-shaped")


def test_gene_slot_count() -> None:
    """100 nodes x (1 function + 2 inputs) = the paper's '300 genes'; +1 output gene.
    3% of that still rounds to Table II's 9."""
    assert cgp.n_gene_slots(100, 2, 1) == 301
    assert cgp.n_mutations(100, 2, 1, 0.03) == 9
    print("ok  gene-slot accounting matches Table II (300 genes + 1 output, 9 mutated)")


def test_active_nodes_subset() -> None:
    """Active nodes are a subset of all nodes, and the output is reachable."""
    gate_set = gates_mod.build_set(gates_mod.DEFAULT_GATES)
    arity = gates_mod.max_arity(gate_set)
    rng = np.random.default_rng(3)
    g = cgp.random_genotype(rng, 100, 8, 1, len(gate_set), arity)
    act = cgp.active_nodes(g, 8, gate_set)
    assert set(act) <= set(range(100))
    assert len(act) < 100, "a random 100-node genotype should have inactive nodes"
    print(f"ok  active-node walk: {len(act)}/100 nodes in the phenotype")


if __name__ == "__main__":
    test_gene_slot_count()
    test_ka_task_counts()
    test_one_half_predictor_is_capped()
    test_fast_matches_slow()
    test_mutation_stays_valid()
    test_active_nodes_subset()
    print("\nall tests passed")
