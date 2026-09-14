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


def test_is_trivial_module_hand_built_nested_case() -> None:
    """The nested generalisation of plain ECGP's trivial-module check.

    A is trivial on its own (one active NAND, plus a dead sibling). B does nothing
    but forward to A -- no primitive nodes of its own -- so it must ALSO be
    trivial, however much nesting depth (`Module.depth`) it carries. C nests A too,
    but then combines A's output with a fresh input through a real gate of its
    own: two active primitives once fully flattened, so C is NOT trivial.

    Structural only -- `func` ids are arbitrary primitive indices, since
    `is_trivial_module` counts nodes, it does not evaluate them.
    """
    A = ecgp.Module(mid=100, n_in=2, func=[0, 0], ntype=[0, 0],
                    conn=[[0, 1], [0, 0]], cout=[[0, 0], [0, 0]],
                    out=[2], ocout=[0], depth=1)
    assert ecgp.is_trivial_module(A, {A.mid: A})

    # B's only body node is a nested call to A; B adds no gate of its own.
    B = ecgp.Module(mid=101, n_in=2, func=[A.mid], ntype=[2],
                    conn=[[0, 1]], cout=[[0]], out=[2], ocout=[0], depth=2)
    modules = {A.mid: A, B.mid: B}
    assert ecgp.module_active_primitive_count(B, modules) == 1
    assert ecgp.is_trivial_module(B, modules)

    # C nests A, then feeds A's output and a fresh input into one more real gate.
    C = ecgp.Module(mid=102, n_in=2, func=[A.mid, 0], ntype=[2, 0],
                    conn=[[0, 1], [2, 1]], cout=[[0, 0], [0, 0]],
                    out=[3], ocout=[0], depth=2)
    modules[C.mid] = C
    assert ecgp.module_active_primitive_count(C, modules) == 2
    assert not ecgp.is_trivial_module(C, modules)
    print("ok  is_trivial_module (nested): forward-only wrapper trivial, "
          "wrapper-plus-a-gate not")


def test_is_trivial_module_matches_independent_flatten(n_trials: int = 40) -> None:
    """`module_active_primitive_count` must agree with an independent route: wrap
    the module alone in a synthetic one-node individual and run it through the
    already-tested `flatten` + `cgp.active_nodes`, over many evolved individuals
    (nesting turned on) so both trivial and non-trivial, nested and flat, bodies
    actually get exercised.
    """
    rnd = random.Random(8)
    p = ecgp.Params(compress=0.6, expand=0.1, module_point=0.1, add_input=0.05,
                    remove_input=0.05, add_output=0.05, remove_output=0.05,
                    max_module_size=5, mutation_rate=0.05, nest_decay=0.6)
    ind = ecgp.random_individual(rnd, 50, N_IN, 1, N_PRIM)
    checked = seen_trivial = seen_nontrivial = seen_nested = 0
    for gen in range(800):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        if gen % 4 != 0:
            continue
        for mid, mod in ind.modules.items():
            synth = ecgp.Individual(
                func=[mid], ntype=[1],
                conn=[list(range(mod.n_in))], cout=[[0] * mod.n_in],
                ogene=[mod.n_in] * mod.n_out, ocout=list(range(mod.n_out)),
                modules=dict(ind.modules), next_id=mid + 1)
            flat = ecgp.flatten(synth, mod.n_in)
            want = len(cgp.active_nodes(flat, mod.n_in, GATE_SET))
            got = ecgp.module_active_primitive_count(mod, ind.modules)
            assert got == want, f"module {mid}: {got} != {want} (independent flatten)"
            assert ecgp.is_trivial_module(mod, ind.modules) == (want <= 1)
            checked += 1
            seen_trivial += want <= 1
            seen_nontrivial += want > 1
            seen_nested += mod.depth > 1
    assert checked > 0, "no modules were ever created -- the test proved nothing"
    assert seen_trivial > 0 and seen_nontrivial > 0, \
        (f"need both kinds to exercise the check "
         f"(trivial={seen_trivial}, nontrivial={seen_nontrivial})")
    assert seen_nested > 0, "never checked a nested (depth>1) module -- test is too shallow"
    print(f"ok  is_trivial_module agrees with independent flatten+active_nodes on "
          f"{checked} modules ({seen_trivial} trivial, {seen_nontrivial} nontrivial, "
          f"{seen_nested} of them nested)")


def test_is_fake_module_hand_built_nested_case() -> None:
    """Nested generalisation of `test_is_trivial_module_hand_built_nested_case`,
    plus the case that check alone does not catch: two active primitives that
    both read straight off the module's own inputs -- one a plain gate, one a
    nested call to trivial module A -- never interact once flattened, so the
    whole thing is fake despite `module_active_primitive_count == 2` (not
    trivial).
    """
    A = ecgp.Module(mid=200, n_in=2, func=[0, 0], ntype=[0, 0],
                    conn=[[0, 1], [0, 0]], cout=[[0, 0], [0, 0]],
                    out=[2], ocout=[0], depth=1)

    # C: nests A, then combines A's output with a fresh input -- chained, real.
    C = ecgp.Module(mid=201, n_in=2, func=[A.mid, 0], ntype=[2, 0],
                    conn=[[0, 1], [2, 1]], cout=[[0, 0], [0, 0]],
                    out=[3], ocout=[0], depth=2)
    modules = {A.mid: A, C.mid: C}
    assert ecgp.module_has_interaction(C, modules)
    assert not ecgp.is_fake_module(C, modules)

    # D: a plain gate and a nested call to A, both reading the module's OWN
    # inputs directly, neither depending on the other -- parallel, not real.
    D = ecgp.Module(mid=202, n_in=2, func=[0, A.mid], ntype=[0, 2],
                    conn=[[0, 1], [0, 1]], cout=[[0, 0], [0, 0]],
                    out=[2, 3], ocout=[0, 0], depth=2)
    modules[D.mid] = D
    assert not ecgp.is_trivial_module(D, modules), "fixture must NOT be trivial (2 active)"
    assert not ecgp.module_has_interaction(D, modules)
    assert ecgp.is_fake_module(D, modules), \
        "a plain gate plus an independent nested call must be fake"
    print("ok  is_fake_module (nested): chained real, gate+independent-nested-call fake")


def test_is_fake_module_matches_independent_flatten_edge_check(n_trials: int = 40) -> None:
    """`module_has_interaction`'s edge scan must agree with an independent route:
    wrap the module alone in a synthetic one-node individual, flatten via the
    already-tested `flatten`, find active nodes with `cgp.active_nodes`, then
    check by hand whether any active node's connection targets another active
    node. Run under heavy nested mutation so interacting, parallel, and nested
    bodies all actually get exercised.
    """
    rnd = random.Random(9)
    p = ecgp.Params(compress=0.6, expand=0.1, module_point=0.1, add_input=0.05,
                    remove_input=0.05, add_output=0.05, remove_output=0.05,
                    max_module_size=5, mutation_rate=0.05, nest_decay=0.6)
    ind = ecgp.random_individual(rnd, 50, N_IN, 1, N_PRIM)
    checked = seen_interacting = seen_parallel = seen_nested = 0
    for gen in range(800):
        ind = ecgp.mutate(ind, rnd, N_IN, N_PRIM, p)
        ecgp.prune_modules(ind)
        if gen % 4 != 0:
            continue
        for mid, mod in ind.modules.items():
            synth = ecgp.Individual(
                func=[mid], ntype=[1],
                conn=[list(range(mod.n_in))], cout=[[0] * mod.n_in],
                ogene=[mod.n_in] * mod.n_out, ocout=list(range(mod.n_out)),
                modules=dict(ind.modules), next_id=mid + 1)
            flat = ecgp.flatten(synth, mod.n_in)
            active = set(cgp.active_nodes(flat, mod.n_in, GATE_SET))
            want = any(flat.conn[2 * b + k] - mod.n_in in active
                      for b in active for k in (0, 1)
                      if flat.conn[2 * b + k] >= mod.n_in)
            got = ecgp.module_has_interaction(mod, ind.modules)
            assert got == want, f"module {mid}: {got} != {want} (independent edge check)"
            assert ecgp.is_fake_module(mod, ind.modules) == (not want)
            checked += 1
            seen_interacting += want
            seen_parallel += not want
            seen_nested += mod.depth > 1
    assert checked > 0, "no modules were ever created -- the test proved nothing"
    assert seen_interacting > 0 and seen_parallel > 0, \
        (f"need both kinds to exercise the check "
         f"(interacting={seen_interacting}, parallel/fake={seen_parallel})")
    assert seen_nested > 0, "never checked a nested (depth>1) module -- test is too shallow"
    print(f"ok  module_has_interaction agrees with independent flatten+edge-check on "
          f"{checked} modules ({seen_interacting} interacting, {seen_parallel} not, "
          f"{seen_nested} nested)")


if __name__ == "__main__":
    test_structural_invariants_survive_heavy_mutation()
    test_evaluate_matches_flatten_when_nested()
    test_decay_zero_blocks_all_nesting()
    test_decay_one_matches_unconditional_nesting()
    test_expand_undoes_exactly_one_level()
    test_nested_into_module_is_protected_from_interface_ops()
    test_prune_keeps_transitively_nested_modules()
    test_is_trivial_module_hand_built_nested_case()
    test_is_trivial_module_matches_independent_flatten()
    test_is_fake_module_hand_built_nested_case()
    test_is_fake_module_matches_independent_flatten_edge_check()
    print("\nall tests passed")
