"""Tests for the ECGP core. Run: conda run -n lndp python test_ecgp.py

Three properties carry the weight here, because ECGP's operators rewrite the graph
rather than just perturbing numbers:

1. **`compress` and `expand` are fitness-neutral.** The paper states it ("the
   genotype before and after represents the same directed graph"), so it is a
   property the implementation must have, not a coincidence to hope for. Any error in
   the relabelling arithmetic breaks it immediately.
2. **`evaluate` agrees with `cgp.evaluate(flatten(...))`.** The fast path evaluates
   modules in place; the reference inlines everything and runs the already-verified
   CGP evaluator. Two independent routes to the same truth table.
3. **Every operator leaves a structurally valid individual** -- `ecgp.validate`
   checks the arity, reference and module bounds from PAPER_SPEC sections 1, 4 and 6,
   including "no nesting" and "a module output never connects to a module input".

The operators are exercised at deliberately high probabilities so a few thousand
generations of drift are compressed into a few hundred trials.
"""

from __future__ import annotations

import random

import cgp
import ecgp
import gates as gates_mod
import tasks as tasks_mod

N_IN = 8
TASK = "retina_ka2005"


def _ctx():
    gate_set = gates_mod.build_set(gates_mod.DEFAULT_GATES)
    in_masks = tasks_mod.input_masks(TASK)
    mask = tasks_mod.full_mask(TASK)
    return gate_set, in_masks, mask


def _params(**kw) -> ecgp.Params:
    """Operator-heavy parameters: the paper's rates would need ~10^4 generations to
    build a module list, and these tests need one within a few dozen."""
    p = ecgp.Params(compress=0.9, expand=0.3, module_point=0.5, add_input=0.3,
                    remove_input=0.3, add_output=0.3, remove_output=0.3,
                    max_module_size=5, mutation_rate=0.03)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _evolve(rnd, gate_set, n_gen=60, n_nodes=40, p=None):
    """A drifting lineage that accumulates modules; returns the final individual."""
    p = p or _params()
    ind = ecgp.random_individual(rnd, n_nodes, N_IN, 1, len(gate_set))
    for _ in range(n_gen):
        ind = ecgp.mutate(ind, rnd, N_IN, len(gate_set), p)
    return ind


def test_evaluate_matches_flattened(n_trials: int = 60) -> None:
    """Direct module evaluation == inline-everything-then-use-the-CGP-evaluator."""
    gate_set, in_masks, mask = _ctx()
    rnd = random.Random(0)
    n_mod = 0
    for _ in range(n_trials):
        ind = _evolve(rnd, gate_set)
        n_mod += len(ind.modules)
        fast = ecgp.evaluate(ind, gate_set, in_masks, mask, N_IN)
        flat = cgp.evaluate(ecgp.flatten(ind, N_IN), gate_set, in_masks, mask, N_IN)
        assert fast == flat, "direct evaluation disagrees with the flattened circuit"
    assert n_mod > 0, "no modules were ever created -- the test proved nothing"
    print(f"ok  evaluate == cgp.evaluate(flatten(.)) on {n_trials} evolved "
          f"individuals ({n_mod} modules total)")


def test_compress_expand_are_neutral(n_trials: int = 200) -> None:
    """compress and expand must not change the function the genotype computes."""
    gate_set, in_masks, mask = _ctx()
    rnd = random.Random(1)
    n_c = n_e = 0
    for _ in range(n_trials):
        ind = _evolve(rnd, gate_set, n_gen=25)
        before = ecgp.evaluate(ind, gate_set, in_masks, mask, N_IN)

        after = ind.copy()
        if ecgp.compress(after, rnd, 5, N_IN):
            n_c += 1
            ecgp.validate(after, N_IN, len(gate_set), 5)
            assert ecgp.evaluate(after, gate_set, in_masks, mask, N_IN) == before, \
                "compress changed the circuit's behaviour"

        after2 = after.copy()
        if ecgp.expand(after2, rnd, N_IN):
            n_e += 1
            ecgp.validate(after2, N_IN, len(gate_set), 5)
            assert ecgp.evaluate(after2, gate_set, in_masks, mask, N_IN) == before, \
                "expand changed the circuit's behaviour"
    assert n_c > 20 and n_e > 20, f"too few operator applications ({n_c}, {n_e})"
    print(f"ok  compress ({n_c}) and expand ({n_e}) are fitness-neutral")


def test_expand_inverts_compress(n_trials: int = 100) -> None:
    """compress-then-expand of the SAME node returns the original directed graph.

    Stronger than neutrality: it checks the two operators are actual inverses, so a
    relabelling bug that happened to preserve behaviour would still be caught.
    """
    gate_set, in_masks, mask = _ctx()
    rnd = random.Random(2)
    n_ok = 0
    for _ in range(n_trials):
        ind = ecgp.random_individual(rnd, 30, N_IN, 1, len(gate_set))
        flat_before = ecgp.flatten(ind, N_IN)
        after = ind.copy()
        if not ecgp.compress(after, rnd, 5, N_IN):
            continue
        assert ecgp.expand(after, rnd, N_IN), "the type I node just made did not expand"
        flat_after = ecgp.flatten(after, N_IN)
        assert flat_before.func == flat_after.func, "node functions differ after a round trip"
        assert flat_before.conn == flat_after.conn, "connections differ after a round trip"
        assert flat_before.ogene == flat_after.ogene, "outputs differ after a round trip"
        n_ok += 1
    assert n_ok > 50, f"only {n_ok} round trips ran"
    print(f"ok  expand(compress(g)) == g as a graph, {n_ok} round trips")


def test_operators_keep_the_individual_valid(n_trials: int = 400) -> None:
    """Every operator sequence leaves a structurally legal individual.

    `ecgp.validate` enforces: correct arity per node type, no forward references,
    every output index within its source's output count, module size/input/output
    bounds, primitives-only module bodies (no nesting), and no module output wired
    straight to a module input.
    """
    gate_set, _, _ = _ctx()
    rnd = random.Random(3)
    p = _params()
    ind = ecgp.random_individual(rnd, 60, N_IN, 1, len(gate_set))
    seen_type2 = 0
    max_mods = 0
    for _ in range(n_trials):
        ind = ecgp.mutate(ind, rnd, N_IN, len(gate_set), p)
        ecgp.validate(ind, N_IN, len(gate_set), p.max_module_size)
        seen_type2 += sum(1 for t in ind.ntype if t == 2)
        max_mods = max(max_mods, len(ind.modules))
    assert seen_type2 > 0, "no type II node ever appeared -- module re-use never fired"
    assert max_mods > 1, "the module list never held more than one module"
    print(f"ok  {n_trials} mutation rounds stayed valid "
          f"(max module list {max_mods}, type II nodes seen {seen_type2})")


def test_type_i_function_gene_is_immune() -> None:
    """Point mutation may never rewrite a type I node's function gene (section 6).

    Driven with mutation rate 1.0 so every gene slot is drawn every time; a type I
    function gene that was mutable would flip within a couple of rounds.
    """
    gate_set, _, _ = _ctx()
    rnd = random.Random(4)
    ind = ecgp.random_individual(rnd, 20, N_IN, 1, len(gate_set))
    for _ in range(50):
        ecgp.compress(ind, rnd, 5, N_IN)
    type1 = [j for j, t in enumerate(ind.ntype) if t == 1]
    assert type1, "no type I nodes to test"
    before = {j: ind.func[j] for j in type1}
    for _ in range(30):
        ecgp.point_mutate(ind, rnd, ecgp.n_gene_slots(ind), N_IN, len(gate_set))
        for j, f in before.items():
            assert ind.ntype[j] == 1 and ind.func[j] == f, \
                f"type I node {j} was mutated by the genotype point mutation"
    print(f"ok  type I function genes immune to point mutation ({len(type1)} nodes, "
          f"30 rounds at rate 1.0)")


def test_prune_keeps_only_used_modules() -> None:
    """Promotion prunes the list to exactly the modules present in the individual."""
    gate_set, _, _ = _ctx()
    rnd = random.Random(5)
    ind = _evolve(rnd, gate_set, n_gen=80)
    assert len(ind.modules) > 0, "no modules to prune"
    ecgp.prune_modules(ind)
    used = {ind.func[j] for j in range(ind.n_nodes) if ind.ntype[j] != 0}
    assert set(ind.modules) == used, "pruned list != modules present in the genotype"
    ecgp.validate(ind, N_IN, len(gate_set), 5)
    print(f"ok  prune leaves exactly the {len(used)} module(s) the individual uses")


def test_empty_module_list_matches_cgp() -> None:
    """With no modules, an ECGP individual is a CGP genotype and evaluates identically.

    This is the continuity check between the two arms: `--ecgp` at generation 0 must
    not be a different experiment from the CGP baseline.
    """
    gate_set, in_masks, mask = _ctx()
    for seed in range(20):
        r1, r2 = random.Random(seed), random.Random(seed)
        ind = ecgp.random_individual(r1, 50, N_IN, 1, len(gate_set))
        g = cgp.random_genotype(r2, 50, N_IN, 1, len(gate_set), 2)
        assert ind.func == g.func and ind.ogene == g.ogene
        assert [x for c in ind.conn for x in c] == g.conn, "different random genotypes"
        assert (ecgp.evaluate(ind, gate_set, in_masks, mask, N_IN)
                == cgp.evaluate(g, gate_set, in_masks, mask, N_IN))
    print("ok  module-free ECGP == CGP (same random genotype, same truth table)")


def test_module_bounds_are_respected(n_trials: int = 300) -> None:
    """The section-4 bounds hold under the module operators, at every size.

    Runs `ms` from 2 to 6 so the add/remove operators hit their ceilings and floors
    rather than drifting in the middle of the legal range.
    """
    gate_set, _, _ = _ctx()
    for ms in (2, 3, 5, 6):
        rnd = random.Random(100 + ms)
        p = _params(max_module_size=ms)
        ind = ecgp.random_individual(rnd, 40, N_IN, 1, len(gate_set))
        hit_in = hit_out = 0
        for _ in range(n_trials):
            ind = ecgp.mutate(ind, rnd, N_IN, len(gate_set), p)
            ecgp.validate(ind, N_IN, len(gate_set), ms)
            for m in ind.modules.values():
                hit_in += m.n_in == 2 * m.n_nodes
                hit_out += m.n_out == m.n_nodes
        print(f"ok  ms={ms}: {n_trials} rounds within bounds "
              f"(input ceiling reached {hit_in}x, output ceiling {hit_out}x)")


def test_fitness_matches_cgp_contract() -> None:
    """ecgp.fitness returns the same (score, hits) a flattened CGP genotype would."""
    gate_set, in_masks, mask = _ctx()
    targets = tasks_mod.target_masks(TASK, "and")
    rnd = random.Random(6)
    for _ in range(30):
        ind = _evolve(rnd, gate_set, n_gen=30)
        a = ecgp.fitness(ind, gate_set, in_masks, targets, mask, N_IN)
        b = cgp.fitness(ecgp.flatten(ind, N_IN), gate_set, in_masks, targets, mask, N_IN)
        assert a == b, f"fitness disagrees: {a} vs {b}"
    print("ok  ecgp.fitness == cgp.fitness on the flattened circuit")


def test_active_nodes_are_exactly_the_ones_that_matter() -> None:
    """`active_nodes` names every node the output depends on, and no others.

    The gate histogram is counted over this set, so an error here would silently
    mis-report what the circuit is built from -- the one number the CGP-vs-ECGP
    comparison turns on. Tested behaviourally rather than by re-implementing the walk:
    overwriting an INACTIVE node's function gene must never change the truth table.
    (The converse -- that every node it *does* name is load-bearing -- is not asserted:
    a node can be reachable and still not matter, e.g. an AND of a value with itself,
    so a false positive there is legitimate.)
    """
    gate_set, in_masks, mask = _ctx()
    n_prim = len(gate_set)
    rnd = random.Random(11)
    checked_in = 0
    for _ in range(20):
        ind = _evolve(rnd, gate_set, n_gen=40)
        base = ecgp.evaluate(ind, gate_set, in_masks, mask, N_IN)
        act = set(ecgp.active_nodes(ind, N_IN))
        for j in range(ind.n_nodes):
            if j in act or ind.ntype[j] != 0:
                continue                       # type I/II: the function gene is a module id
            probe = ind.copy()
            probe.func[j] = (probe.func[j] + 1) % n_prim
            assert ecgp.evaluate(probe, gate_set, in_masks, mask, N_IN) == base, \
                f"node {j} was called inactive but changes the output"
            checked_in += 1
    print(f"ok  active-node walk: {checked_in} inactive nodes verified inert "
          f"over 20 individuals")


def test_module_signature_identifies_the_function() -> None:
    """Two modules get the same signature exactly when they compute the same function.

    This is the claim the whole module analysis rests on -- "M12 and M40 are both XOR"
    has to be a fact, not an eyeball judgement -- so it is tested on XOR built three
    incompatible ways: out of NANDs, out of OR/NAND/AND, and again with the inputs
    swapped and a dummy third input bolted on. All three must collapse to one key, and
    must not collide with AND.
    """
    gate_set, _, _ = _ctx()
    gid = {g.name.upper(): i for i, g in enumerate(gate_set)}
    A, NA, O, NO = gid["AND"], gid["NAND"], gid["OR"], gid["NOR"]

    # XOR = NAND(NAND(a, NAND(a,b)), NAND(b, NAND(a,b)))
    xor_nand = ecgp.Module(mid=0, n_in=2, func=[NA, NA, NA, NA],
                           conn=[0, 1, 0, 2, 1, 2, 3, 4], out=[5])
    # XOR = AND(OR(a,b), NAND(a,b))
    xor_mixed = ecgp.Module(mid=1, n_in=2, func=[O, NA, A],
                            conn=[0, 1, 0, 1, 2, 3], out=[4])
    # the same, on inputs (i2, i0), with i1 never read
    xor_permuted = ecgp.Module(mid=2, n_in=3, func=[O, NA, A],
                               conn=[2, 0, 2, 0, 3, 4], out=[5])
    and_mod = ecgp.Module(mid=3, n_in=2, func=[NA, NO], conn=[0, 1, 2, 2], out=[3])

    sig = {}
    for name, m in (("xor_nand", xor_nand), ("xor_mixed", xor_mixed),
                    ("xor_permuted", xor_permuted), ("and", and_mod)):
        label, s, expr = ecgp.module_info(m, gate_set)
        sig[name] = s
        print(f"    {name:13s} sig {s:12s} label {label or '-':5s}  {expr}")

    assert sig["xor_nand"] == sig["xor_mixed"] == sig["xor_permuted"], \
        f"three XORs disagree: {sig}"
    assert sig["and"] != sig["xor_nand"], "AND and XOR share a signature"
    assert ecgp.module_info(xor_nand, gate_set)[0] == "XOR"
    assert ecgp.module_info(and_mod, gate_set)[0] == "AND"
    # a primitive dressed as a module must land on the same key as a module that
    # rediscovers it -- that is how "this module is just an OR" becomes visible
    assert ecgp.module_info(ecgp.primitive_module(A), gate_set)[1] == sig["and"]
    print("ok  signatures identify functions across three encodings of XOR")


if __name__ == "__main__":
    test_empty_module_list_matches_cgp()
    test_evaluate_matches_flattened()
    test_compress_expand_are_neutral()
    test_expand_inverts_compress()
    test_type_i_function_gene_is_immune()
    test_operators_keep_the_individual_valid()
    test_module_bounds_are_respected()
    test_prune_keeps_only_used_modules()
    test_fitness_matches_cgp_contract()
    test_active_nodes_are_exactly_the_ones_that_matter()
    test_module_signature_identifies_the_function()
    print("\nall tests passed")
