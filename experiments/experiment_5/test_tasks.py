"""Tests for experiment 5's task definitions.

Run:  conda run -n lndp python test_tasks.py            (needs numpy: it cross-checks)
      .venv-pypy/Scripts/python test_tasks.py           (skips the numpy cross-check)

`tasks.py` re-expresses the Kashtan-Alon retina in mask algebra instead of loading
`kashtan_alon/tasks.py` through numpy, because it has to import under PyPy. A
re-expression that is only eyeballed is a re-expression that is wrong, so the first
test here rebuilds every shared task through the numpy original and asserts the masks
match bit for bit. That test is the reason the duplication is safe: if the KA rule is
corrected upstream, this fails loudly rather than experiment 5 quietly running a
different task from experiment 4.

The remaining tests cover the definitions numpy has no opinion about -- the adder, the
multiplier, and `cgp.behavioural_deps` -- against independently computed truth.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys

import cgp
import gates as gates_mod
import tasks


def _bit(mask: int, r: int) -> int:
    return (mask >> r) & 1


def _load_ka():
    """`kashtan_alon/tasks.py`, or None when numpy is absent (i.e. under PyPy)."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        return None
    path = pathlib.Path(__file__).resolve().parents[2] / "kashtan_alon" / "tasks.py"
    spec = importlib.util.spec_from_file_location("_ka_tasks_for_exp5", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matches_kashtan_alon() -> None:
    """The mask algebra reproduces the numpy rule exactly, for every shared task.

    This is the load-bearing test of `tasks.py`. `retina_x1` and `retina_ka2005` are
    the same object under two names, so both are checked; `retina_xN` for N>1 has no
    numpy counterpart and is covered by `test_retina_stack` instead.
    """
    ka = _load_ka()
    if ka is None:
        print("  skip test_matches_kashtan_alon (no numpy -- running under PyPy?)")
        return
    import numpy as np

    shared = [("copy", "copy"), ("and2", "and2"), ("left", "left"),
              ("retina_ka2005", "retina"), ("retina_x1", "retina")]
    for ours, theirs in shared:
        n_in = tasks.n_inputs(ours)
        assert n_in == ka.min_inputs(theirs), (ours, n_in)
        X = ka.all_binary_inputs(n_in)

        # input masks: mask bit r must equal X[r, i]
        got = tasks.input_masks(ours)
        for i in range(n_in):
            col = X[:, i].astype(np.int8).tolist()
            assert all(_bit(got[i], r) == col[r] for r in range(len(col))), (ours, i)

        for op in tasks.OPERATIONS:
            y = np.asarray(ka.targets(theirs, op, X), dtype=np.int8).tolist()
            (m,) = tasks.target_masks(ours, op)
            assert all(_bit(m, r) == y[r] for r in range(len(y))), (ours, op)
            if not tasks.uses_operation(ours):
                break                       # one pass is enough; op is ignored
    print("  ok test_matches_kashtan_alon")


def test_retina_stack() -> None:
    """`retina_xN` is N independent copies: output r depends on pixels 8r..8r+7 only.

    Checked behaviourally, not structurally -- `cgp.behavioural_deps` measures which
    inputs actually move the output, which is exactly the claim being made.
    """
    for name, k in (("retina_x1", 1), ("retina_x2", 2)):
        n_in = tasks.n_inputs(name)
        assert n_in == 8 * k and tasks.n_outputs(name) == k
        masks = tasks.input_masks(name)
        full = tasks.full_mask(name)
        outs = tasks.target_masks(name, "and")
        assert len(outs) == k
        for r, m in enumerate(outs):
            deps = cgp.behavioural_deps(m, masks, full, n_in)
            assert deps == frozenset(range(8 * r, 8 * r + 8)), (name, r, sorted(deps))
        # ...and each copy is the SAME function of its own pixels
        if k > 1:
            single = tasks.target_masks("retina_ka2005", "and")[0]
            for r in range(k):
                # restrict pattern space: fix every other retina's pixels to 0 and read
                # off the sub-truth-table, then compare against the 8-input original
                sub = _restrict_to_block(outs[r], masks, n_in, 8 * r)
                assert sub == single, (name, r)
    print("  ok test_retina_stack")


def _restrict_to_block(out_mask: int, in_masks, n_in: int, base: int) -> int:
    """The 8-input truth table of `out_mask` with inputs outside [base, base+8) at 0.

    Reads the surviving patterns out one at a time. Only used by the test, so clarity
    beats the bit trick that would do it in O(1) shifts.
    """
    keep = [base + t for t in range(8)]
    res = 0
    for r8 in range(256):
        r = 0
        for t, i in enumerate(keep):
            if (r8 >> (7 - t)) & 1:
                r |= 1 << (n_in - 1 - i)
        if (out_mask >> r) & 1:
            res |= 1 << r8
    return res


def _int_of(masks, r: int, lo: int, hi: int) -> int:
    """The LSB-first bit field `masks[lo:hi]` read on pattern `r`, as an integer."""
    v = 0
    for k, m in enumerate(masks[lo:hi]):
        v |= ((m >> r) & 1) << k
    return v


def test_adder() -> None:
    """`addN` is a real N-bit adder with carry-in, on every pattern."""
    for k in (1, 2, 3, 4):
        name = f"add{k}"
        n_in, n_out = tasks.n_inputs(name), tasks.n_outputs(name)
        assert (n_in, n_out) == (2 * k + 1, k + 1)
        masks = tasks.input_masks(name)
        outs = tasks.target_masks(name)
        for r in range(1 << n_in):
            a = _int_of(masks, r, 0, k)
            b = _int_of(masks, r, k, 2 * k)
            cin = (masks[2 * k] >> r) & 1
            got = _int_of(outs, r, 0, n_out)
            assert got == a + b + cin, (name, r, a, b, cin, got)
    print("  ok test_adder")


def test_multiplier() -> None:
    """`multN` is a real N x N unsigned multiplier, on every pattern."""
    for k in (1, 2, 3, 4):
        name = f"mult{k}"
        n_in, n_out = tasks.n_inputs(name), tasks.n_outputs(name)
        assert (n_in, n_out) == (2 * k, 2 * k)
        masks = tasks.input_masks(name)
        outs = tasks.target_masks(name)
        for r in range(1 << n_in):
            a = _int_of(masks, r, 0, k)
            b = _int_of(masks, r, k, 2 * k)
            got = _int_of(outs, r, 0, n_out)
            assert got == a * b, (name, r, a, b, got)
    print("  ok test_multiplier")


def test_input_masks_match_pattern_order() -> None:
    """`input_masks` agrees with the ordering `cgp.evaluate_slow` assumes.

    The two are written independently -- one by doubling a square wave, the other by
    shifting the pattern index -- and everything the experiment measures depends on
    them agreeing, so it is asserted rather than assumed.
    """
    for n_in in range(1, 9):
        masks = [tasks._input_mask(i, n_in) for i in range(n_in)]
        for r, combo in enumerate(itertools.product([0, 1], repeat=n_in)):
            for i in range(n_in):
                assert _bit(masks[i], r) == combo[i], (n_in, r, i)
                assert combo[i] == ((r >> (n_in - 1 - i)) & 1)
    print("  ok test_input_masks_match_pattern_order")


def test_groups_partition_the_inputs() -> None:
    """Every task's `input_groups` is a genuine partition -- no gaps, no overlaps."""
    for name in ("retina_ka2005", "retina_x2", "retina_x3", "left", "and2", "copy",
                 "add1", "add4", "mult1", "mult3"):
        n_in = tasks.n_inputs(name)
        groups = tasks.input_groups(name)
        seen: set[int] = set()
        for g in groups:
            assert g, (name, "empty group")
            assert not (g & seen), (name, "overlapping groups")
            seen |= set(g)
        assert seen == set(range(n_in)), (name, sorted(seen))
        assert len(cgp.group_names(groups)) == len(groups)
    print("  ok test_groups_partition_the_inputs")


def test_behavioural_deps_agrees_with_brute_force() -> None:
    """`cgp.behavioural_deps` matches the definition, checked by enumeration.

    The fast version is one shift and three masks per input; the reference walks every
    pattern and flips every bit. They must agree on arbitrary functions, not just on
    task targets, so the masks tested here are random.
    """
    import random
    rnd = random.Random(7)
    for n_in in range(1, 8):
        n_pat = 1 << n_in
        full = (1 << n_pat) - 1
        masks = [tasks._input_mask(i, n_in) for i in range(n_in)]
        for _ in range(20):
            f = rnd.getrandbits(n_pat)
            slow = set()
            for r in range(n_pat):
                for i in range(n_in):
                    flipped = r ^ (1 << (n_in - 1 - i))
                    if ((f >> r) & 1) != ((f >> flipped) & 1):
                        slow.add(i)
            fast = cgp.behavioural_deps(f, masks, full, n_in)
            assert fast == frozenset(slow), (n_in, sorted(slow), sorted(fast))
    print("  ok test_behavioural_deps_agrees_with_brute_force")


def test_behavioural_deps_ignores_dead_wiring() -> None:
    """A gate can be in the cone and carry no influence -- the two readouts differ.

    `x AND (NOT x)` is wired to input 0 and is constant, so the structural cone says
    "depends on input 0" and the behavioural support says "depends on nothing". This
    is the gap `beh_pure` vs `out_pure` exists to expose, so it is pinned down here.
    """
    gate_set = gates_mod.build_set("and,nand,or,nor")
    n_in = 2
    masks = tasks.input_masks("and2")
    full = tasks.full_mask("and2")
    # node0 = NAND(in0, in0) = NOT in0 ; node1 = AND(in0, node0) = 0
    g = cgp.Genotype(func=[1, 0], ntype=[0, 0], conn=[0, 0, 0, 2], cout=[0] * 4,
                     ogene=[3], ocout=[0], arity=2)
    (out,) = cgp.evaluate(g, gate_set, masks, full, n_in)
    assert out == 0
    pheno = cgp.phenotype(g, n_in, gate_set, tasks.input_groups("and2"))
    assert pheno.cone[1] == frozenset({0}), pheno.cone
    assert cgp.behavioural_deps(out, masks, full, n_in) == frozenset()
    print("  ok test_behavioural_deps_ignores_dead_wiring")


def main() -> int:
    print(f"test_tasks.py  ({'PyPy' if hasattr(sys, 'pypy_version_info') else 'CPython'} "
          f"{sys.version.split()[0]})")
    test_matches_kashtan_alon()
    test_retina_stack()
    test_adder()
    test_multiplier()
    test_input_masks_match_pattern_order()
    test_groups_partition_the_inputs()
    test_behavioural_deps_agrees_with_brute_force()
    test_behavioural_deps_ignores_dead_wiring()
    print("all task tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
