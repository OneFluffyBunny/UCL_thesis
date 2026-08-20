"""Tests for the experiment-6 SMCGP core. Run: conda run -n lndp python test_smcgp.py

Covers the pieces most likely to be wrong in a from-scratch implementation: relative
addressing resolving out-of-range to a constant, the output-selection fallback
rules (PAPER_SPEC.md III-C), each hand-checkable self-modification operator on a
tiny fixed graph, and an end-to-end smoke run across the develop -> evaluate path
that the search loop actually uses.
"""

from __future__ import annotations

import random

import gates as gates_mod
import smcgp
import tasks as tasks_mod

GATE_SET = gates_mod.build_set(gates_mod.DEFAULT_GATES)
FTABLE = smcgp.build_function_table(GATE_SET)
N_FUNCS = len(FTABLE)
INP = 0
AND = [i for i, f in enumerate(FTABLE) if f.name == "and"][0]
_SM = {f.name: i for i, f in enumerate(FTABLE) if f.kind == "sm"}


def _node(func, c0=1, c1=1, p0=0.0, p1=0.0, p2=0.0, out=0) -> smcgp.Node:
    return smcgp.Node(func, c0, c1, p0, p1, p2, out)


def test_out_of_range_address_is_zero() -> None:
    """A connection reaching before the start of the graph resolves as a
    constant 0, not an error, and does not extend the backward walk."""
    nodes = [_node(INP), _node(AND, c0=5, c1=1, out=1)]   # c0=5 -> position -3
    active, outs, corrupt = smcgp.active_walk(nodes, FTABLE, 1)
    assert not corrupt and outs == [1]
    assert active == [0, 1], "node 0 is still reached via c1=1"
    out = smcgp.evaluate(nodes, FTABLE, 1, tasks_mod.input_masks(1),
                         tasks_mod.full_mask(1))
    x0 = tasks_mod.input_masks(1)[0]
    assert out[0] == (0 & x0), "AND with the out-of-range input acting as constant 0"
    print("ok  out-of-range relative address resolves to constant 0")


def test_output_selection_fallback() -> None:
    """No node flagged -> last n_out nodes. Enough flagged -> leftmost flagged."""
    nodes = [_node(INP), _node(INP), _node(AND, c0=1, c1=2)]
    outs, corrupt = smcgp.select_outputs(nodes, 1)
    assert not corrupt and outs == [2], "none flagged -> falls back to the last node"

    nodes[0].out = 1
    nodes[2].out = 1
    outs, corrupt = smcgp.select_outputs(nodes, 1)
    assert outs == [0], "leftmost flagged node wins when more are flagged than needed"

    nodes = [_node(INP)]
    outs, corrupt = smcgp.select_outputs(nodes, 2)
    assert corrupt, "fewer nodes than required outputs is corrupt"
    print("ok  output-selection fallback rules")


def test_dup_duplicates_a_range() -> None:
    """DUP(P0=-2,P1=2,P2=0) on the node at x=4 copies nodes[P0+x : P0+x+P1] =
    nodes[2:4] and inserts them after position (P0+x+P2)=2 -- i.e. right where
    they came from, so the graph just grows by the copied count."""
    nodes = [_node(INP), _node(INP), _node(AND, c0=1, c1=1),
            _node(AND, c0=1, c1=1), _node(_SM["DUP"], p0=-2, p1=2, p2=0)]
    out = smcgp.apply_todo(nodes, [4], FTABLE, N_FUNCS, 10, 5.0, random.Random(0))
    assert len(out) == 7, "two nodes duplicated -> graph grows by 2"
    assert FTABLE[out[2].func].name == "and" and FTABLE[out[3].func].name == "and"
    assert FTABLE[out[4].func].name == "and", "the copy was inserted right after the original range"
    print("ok  DUP inserts a copy of the addressed range")


def test_del_removes_a_range() -> None:
    nodes = [_node(INP), _node(AND, c0=1, c1=1), _node(AND, c0=1, c1=1),
            _node(_SM["DEL"], p0=-2, p1=2, p2=0)]
    out = smcgp.apply_todo(nodes, [3], FTABLE, N_FUNCS, 10, 5.0, random.Random(0))
    assert len(out) == 2, "the two AND nodes at positions 1,2 were deleted"
    print("ok  DEL removes the addressed range")


def test_chf_targets_p0_directly_not_p0_plus_x() -> None:
    """CHF/CHC/CHP address node P0 directly (PAPER_SPEC.md verbatim) -- unlike
    the range operators, x is NOT added."""
    nodes = [_node(INP), _node(AND, c0=1, c1=1), _node(_SM["CHF"], p0=1, p1=INP, p2=0)]
    out = smcgp.apply_todo(nodes, [2], FTABLE, N_FUNCS, 10, 5.0, random.Random(0))
    assert out[1].func == INP, "node at absolute position P0=1 was retargeted, not node 2-1"
    print("ok  CHF/CHC/CHP address node P0 directly")


def test_add_grows_the_graph() -> None:
    nodes = [_node(INP), _node(_SM["ADD"], p0=-1, p1=3, p2=0)]
    out = smcgp.apply_todo(nodes, [1], FTABLE, N_FUNCS, 10, 5.0, random.Random(0))
    assert len(out) == 5, "3 new random nodes inserted"
    print("ok  ADD inserts new random nodes")


def test_develop_then_evaluate_smoke(n_trials: int = 25) -> None:
    """A random genotype survives develop()+evaluate() across a curriculum of
    sizes without crashing, for many random genotypes and mutation lineages --
    this is what the search loop actually does every generation."""
    rnd = random.Random(1)
    for _ in range(n_trials):
        g = smcgp.random_genotype(rnd, 25, N_FUNCS, 50, 8.0)
        for _ in range(5):
            g = smcgp.mutate(g, rnd, N_FUNCS, 50, 8.0)
            for n_in in range(2, 7):
                phen = smcgp.develop(g, FTABLE, 1, n_in - 2, 2, N_FUNCS, 50, 8.0, rnd)
                out = smcgp.evaluate(phen, FTABLE, 1, tasks_mod.input_masks(n_in),
                                     tasks_mod.full_mask(n_in))
                if out is not None:
                    assert 0 <= out[0] <= tasks_mod.full_mask(n_in)
    print(f"ok  develop+evaluate survives {n_trials} random lineages x 5 mutations "
         f"x 5 curriculum sizes with no crash")


def test_parity_target_is_correct_xor() -> None:
    """Sanity-check the task itself against a hand-computed 2-input truth table."""
    t = tasks_mod.target_parity(2)
    bits = [(t >> r) & 1 for r in range(4)]
    # pattern order: r=0 -> (0,0), r=1 -> (0,1), r=2 -> (1,0), r=3 -> (1,1)
    assert bits == [0, 1, 1, 0], "2-input XOR truth table"
    print("ok  target_parity matches the hand-computed 2-input XOR table")


if __name__ == "__main__":
    test_out_of_range_address_is_zero()
    test_output_selection_fallback()
    test_dup_duplicates_a_range()
    test_del_removes_a_range()
    test_chf_targets_p0_directly_not_p0_plus_x()
    test_add_grows_the_graph()
    test_parity_target_is_correct_xor()
    test_develop_then_evaluate_smoke()
    print("\nall tests passed")
