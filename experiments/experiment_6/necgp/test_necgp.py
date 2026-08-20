"""Tests for necgp's nesting extension. Run: conda run -n lndp python test_necgp.py

Nesting rewired `Module` (flat arity-2 -> variable arity, no cout -> ocout added,
recursive expand/flatten/_run_module) in several places at once (see ecgp.py's
module docstring), so what matters most here is: does the structural invariant
(`validate`) survive many generations of real mutation pressure, does the
bit-parallel evaluator agree with flatten()+cgp.evaluate() once bodies can nest,
does `nest_decay` actually gate depth the way it is supposed to, and does one
`expand` undo exactly one level rather than jumping to primitives.
"""

from __future__ import annotations

import random

import cgp
import ecgp
import gates as gates_mod
import tasks as tasks_mod

GATE_SET = gates_mod.build_set("and,nand,or,nor")
N_PRIM = len(GATE_SET)
N_IN = 8
IN_MASKS = tasks_mod.input_masks("retina_ka2005")
MASK = tasks_mod.full_mask("retina_ka2005")


def _run_generations(rnd: random.Random, n: int, p: ecgp.Params, n_nodes: int = 40):
    """Random start, `n` unconditional mutate() calls, validating throughout."""
    ind = ecgp.random_individual(rnd, n_nodes, N_IN, 1, N_PRIM)
    ecgp.validate(ind, N_IN, N_PRIM, p.max_module_size)
    for _ in range(n):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        ecgp.validate(ind, N_IN, N_PRIM, p.max_module_size)
    return ind


def test_structural_invariants_survive_heavy_mutation() -> None:
    """1000 generations of aggressive compress/expand/module-op pressure, nesting
    turned on hard (nest_decay=1.0 -- every depth accepted unconditionally), never
    produce a structural violation."""
    rnd = random.Random(0)
    p = ecgp.Params(compress=0.5, expand=0.3, module_point=0.2, add_input=0.1,
                    remove_input=0.1, add_output=0.1, remove_output=0.1,
                    max_module_size=5, mutation_rate=0.05, nest_decay=1.0)
    for seed in range(5):
        _run_generations(random.Random(seed), 1000, p)
    print("ok  structural invariants survive 5x1000 generations of heavy nested mutation")


def test_evaluate_matches_flatten_when_nested() -> None:
    """The bit-parallel evaluator and flatten()+cgp.evaluate() must agree exactly,
    including when the individual carries modules of depth > 1."""
    rnd = random.Random(1)
    p = ecgp.Params(compress=0.5, expand=0.1, module_point=0.1, add_input=0.05,
                    remove_input=0.05, add_output=0.05, remove_output=0.05,
                    max_module_size=5, mutation_rate=0.05, nest_decay=1.0)
    ind = ecgp.random_individual(rnd, 60, N_IN, 1, N_PRIM)
    checked_depth2 = False
    for gen in range(400):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        if gen % 10 == 0:
            direct = ecgp.evaluate(ind, GATE_SET, IN_MASKS, MASK, N_IN)
            flat = ecgp.flatten(ind, N_IN)
            via_flat = cgp.evaluate(flat, GATE_SET, IN_MASKS, MASK, N_IN)
            assert direct == via_flat, f"gen {gen}: evaluate() != cgp.evaluate(flatten())"
            if any(m.depth > 1 for m in ind.modules.values()):
                checked_depth2 = True
    assert checked_depth2, "test is only meaningful if depth > 1 actually occurred"
    print("ok  evaluate() == cgp.evaluate(flatten(...)) throughout, including at depth > 1")


def test_decay_zero_blocks_all_nesting() -> None:
    """nest_decay=0.0 must make depth > 1 essentially impossible: any window
    containing a type I/II node fails the `rnd.random() >= 0.0**(depth-1) == 1`
    roll deterministically (0.0**k == 0.0 for k >= 1, and rnd.random() is always
    >= 0.0), so only depth-1 modules can ever form."""
    rnd = random.Random(2)
    p = ecgp.Params(compress=0.6, expand=0.1, module_point=0.1, add_input=0.05,
                    remove_input=0.05, add_output=0.05, remove_output=0.05,
                    max_module_size=5, mutation_rate=0.05, nest_decay=0.0)
    ind = ecgp.random_individual(rnd, 60, N_IN, 1, N_PRIM)
    for _ in range(500):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        assert all(m.depth == 1 for m in ind.modules.values()), \
            "nest_decay=0.0 let a depth>1 module through"
    print("ok  nest_decay=0.0 blocks every attempt to nest beyond depth 1")


def test_decay_one_matches_unconditional_nesting() -> None:
    """nest_decay=1.0 must reach depth > 1 given enough attempts (the geometric
    throttling from compress's own base rate + window placement is still there,
    but nothing ADDITIONAL should suppress it)."""
    rnd = random.Random(3)
    p = ecgp.Params(compress=0.6, expand=0.05, module_point=0.0, add_input=0.0,
                    remove_input=0.0, add_output=0.0, remove_output=0.0,
                    max_module_size=5, mutation_rate=0.03, nest_decay=1.0)
    ind = ecgp.random_individual(rnd, 60, N_IN, 1, N_PRIM)
    max_depth = 1
    for _ in range(3000):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        max_depth = max(max_depth, max((m.depth for m in ind.modules.values()), default=1))
    assert max_depth > 1, "3000 generations at nest_decay=1.0 never reached depth 2"
    print(f"ok  nest_decay=1.0 reaches depth {max_depth} within 3000 generations")


def test_expand_undoes_exactly_one_level() -> None:
    """Force a depth-2 module, expand its owner, and check the result: the inlined
    body must still contain a nested (type I/II) node -- i.e. expand exposed the
    depth-1 module underneath rather than collapsing straight to primitives."""
    rnd = random.Random(4)
    p = ecgp.Params(compress=0.7, expand=0.0, module_point=0.0, add_input=0.0,
                    remove_input=0.0, add_output=0.0, remove_output=0.0,
                    max_module_size=5, mutation_rate=0.01, nest_decay=1.0)
    ind = ecgp.random_individual(rnd, 60, N_IN, 1, N_PRIM)
    for _ in range(2000):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        if any(m.depth > 1 for m in ind.modules.values()):
            break
    depth2 = [mid for mid, m in ind.modules.items() if m.depth > 1]
    assert depth2, "never managed to build a depth>1 module to test expand on"

    # find a TOP-LEVEL owner (type 1) of a depth>1 module
    owners = [j for j, t in enumerate(ind.ntype)
             if t == 1 and ind.modules[ind.func[j]].depth > 1]
    assert owners, "no top-level owner of a depth>1 module -- can't test expand directly"
    p_idx = owners[0]
    mod = ind.modules[ind.func[p_idx]]
    m = mod.n_nodes
    ecgp.validate(ind, N_IN, N_PRIM, p.max_module_size)
    ok = ecgp.expand(ind, rnd, N_IN)
    assert ok
    ecgp.prune_modules(ind)
    ecgp.validate(ind, N_IN, N_PRIM, p.max_module_size)
    inlined_ntypes = ind.ntype[p_idx:p_idx + m]
    assert any(t != 0 for t in inlined_ntypes), \
        "expand on a depth>1 module collapsed straight to primitives -- should stop one level down"
    print("ok  expand on a depth>1 module undoes exactly one level of nesting")


def test_nested_into_module_is_protected_from_interface_ops() -> None:
    """A module referenced from inside another module's body must be refused by
    add_input/remove_input/add_output/remove_output (v1 scope, see
    `ecgp._is_nested_into`'s docstring)."""
    rnd = random.Random(5)
    p = ecgp.Params(compress=0.7, expand=0.0, nest_decay=1.0, max_module_size=5)
    ind = ecgp.random_individual(rnd, 60, N_IN, 1, N_PRIM)
    for _ in range(2000):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        nested_ids = {mod.func[b] for mod in ind.modules.values()
                     for b in range(mod.n_nodes) if mod.ntype[b] != 0}
        if nested_ids:
            break
    assert nested_ids, "never produced a nested-into module to test protection on"
    victim = next(iter(nested_ids))
    assert ecgp._is_nested_into(ind, victim)
    assert ecgp.add_input(ind, victim, rnd, N_IN) is None
    assert ecgp.remove_input(ind, victim, rnd) is False
    assert ecgp.add_output(ind, victim, rnd) is None
    assert ecgp.remove_output(ind, victim, rnd, N_IN) is False
    print("ok  a nested-into module is refused by all four interface operators")


def test_prune_keeps_transitively_nested_modules() -> None:
    """prune_modules must not delete a module that is alive only because another
    module's body calls it (no top-level node references it directly)."""
    rnd = random.Random(6)
    p = ecgp.Params(compress=0.7, expand=0.0, nest_decay=1.0, max_module_size=5)
    ind = ecgp.random_individual(rnd, 60, N_IN, 1, N_PRIM)
    for _ in range(2000):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        nested_ids = {mod.func[b] for mod in ind.modules.values()
                     for b in range(mod.n_nodes) if mod.ntype[b] != 0}
        if nested_ids:
            break
    assert nested_ids, "never produced a nested-into module to test pruning on"
    for mid in nested_ids:
        assert mid in ind.modules, f"module {mid} is nested-into but was pruned"
    print("ok  prune_modules keeps modules that are alive only via nesting")


if __name__ == "__main__":
    test_structural_invariants_survive_heavy_mutation()
    test_evaluate_matches_flatten_when_nested()
    test_decay_zero_blocks_all_nesting()
    test_decay_one_matches_unconditional_nesting()
    test_expand_undoes_exactly_one_level()
    test_nested_into_module_is_protected_from_interface_ops()
    test_prune_keeps_transitively_nested_modules()
    print("\nall tests passed")
