"""Task definitions for experiment 5 -- multi-output, pure Python, no numpy.

DIFFERENCE FROM EXPERIMENT 4. `experiment_4/tasks.py` loads `kashtan_alon/tasks.py`
through `importlib` and packs its numpy arrays into bitmasks. That is the right call
there (the rule is inherited, so a correction upstream propagates) but it makes the
module depend on numpy, and experiment 5 has to run under **PyPy**, where numpy is
not available. So the rules are re-expressed here directly in mask algebra.

The re-expression is NOT trusted on inspection: `test_tasks.py` rebuilds every shared
task through `kashtan_alon/tasks.py` under CPython and asserts the masks are equal
bit for bit. That test is the thing that keeps this file honest -- if the KA rule is
ever corrected upstream, the test fails and this file gets updated. Run it under
CPython (it needs numpy); the search itself never does.

MASK ALGEBRA, NOT PER-PATTERN LOOPS. A target is built the same way the evaluator
computes a circuit: as bitwise ops over whole-truth-table ints. `and2`'s target is
literally `in_masks[0] & in_masks[1]`. A ripple-carry adder is the textbook full-adder
recurrence with `&`/`^`/`|` over masks. This costs O(gates) big-int ops instead of
O(2**n_in) Python iterations, which is what makes a 16- or 20-input task buildable at
all: at n_in=20 the loop would be a million iterations per output, the algebra is
about forty shifts.

MANY OUTPUTS. Experiment 4 had one program output and `target_mask` returned one int.
Here a task has `n_out` outputs and `target_masks` returns a tuple of them; fitness
sums hits over all of them (see `cgp.fitness`). This is what the experiment is for:
behavioural modularity needs several jobs to be modular *about*.

THE 2**n_in WALL. Evaluation is exhaustive over every input pattern, so one wire is a
`2**n_in`-bit int: 8 KB at n_in=16, 128 KB at n_in=20, 2 MB at n_in=24. Cost per gate
grows linearly in that, so ~20 inputs is the practical ceiling regardless of how fast
the interpreter is. The interface below is deliberately written in terms of
`n_patterns()` rather than `1 << n_inputs()` everywhere, so moving to a *sampled*
subset of patterns later is a change to this file alone.

INPUT / PATTERN CONVENTION (must match `cgp.evaluate_slow`). Patterns are ordered as
`itertools.product([0, 1], repeat=n_in)`, whose FIRST element varies slowest. So on
pattern index `r`, input `i` has value `(r >> (n_in - 1 - i)) & 1`, and mask bit `r`
of `input_masks()[i]` holds exactly that.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# input masks
# ---------------------------------------------------------------------------


def _input_mask(i: int, n_in: int) -> int:
    """Truth-table mask of program input `i` over all `2**n_in` patterns.

    Input `i` is bit `n_in-1-i` of the pattern index, so its mask is a square wave:
    `stride` zeros, `stride` ones, repeating, with `stride = 2**(n_in-1-i)`. Built by
    doubling (`m |= m << width`) rather than by looping over patterns -- the same
    O(log) trick as repeated squaring, and the reason a 20-input task builds instantly.
    """
    stride = 1 << (n_in - 1 - i)
    m = ((1 << stride) - 1) << stride          # one period: low half 0, high half 1
    width = stride << 1
    total = 1 << n_in
    while width < total:
        m |= m << width
        width <<= 1
    return m


# ---------------------------------------------------------------------------
# the rules, as mask algebra
# ---------------------------------------------------------------------------


def _ka_object(outer_a: int, outer_b: int, inner_a: int, inner_b: int,
               mask: int) -> int:
    """Kashtan & Alon's object rule for one 2x2 retina block, over masks.

    `[verbatim]` (PNAS 2005, Fig. 5a): true iff **three or more** of the four pixels
    are black, **or** one-to-two black pixels confined to the block's OUTER column.
    Arguments are the two outer-column pixels then the two inner-column ones, exactly
    as `kashtan_alon/tasks.py::_object` takes them.

    ">=3 of 4" is written as the four 3-subsets OR-ed together rather than as an
    arithmetic sum, because masks have no arithmetic -- and the four-term form is
    directly checkable against the truth table, which `test_tasks.py` does.
    """
    ge3 = ((outer_a & outer_b & inner_a) | (outer_a & outer_b & inner_b)
           | (outer_a & inner_a & inner_b) | (outer_b & inner_a & inner_b))
    outer_only = ~inner_a & ~inner_b & (outer_a | outer_b) & mask
    return (ge3 | outer_only) & mask


def _retina_halves(x: list[int], base: int, mask: int) -> tuple[int, int]:
    """(left object, right object) for the 8 pixels starting at input `base`.

    Pixel map `[verbatim]` from `kashtan_alon/tasks.py`: the left block's OUTER
    (left) column is pixels 0,1 and its inner column 2,3; the right block's OUTER
    (right) column is 6,7 and its inner column 4,5.
    """
    p = x[base:base + 8]
    return (_ka_object(p[0], p[1], p[2], p[3], mask),
            _ka_object(p[6], p[7], p[4], p[5], mask))


def _combine(op: str, a: int, b: int, mask: int) -> int:
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    raise ValueError(f"unknown operation: {op!r} (known: {OPERATIONS})")


def _full_adder(a: int, b: int, c: int) -> tuple[int, int]:
    """(sum, carry-out) of three mask bits -- the textbook full adder."""
    axb = a ^ b
    return axb ^ c, (a & b) | (c & axb)


def _add_bits(a: list[int], b: list[int], cin: int) -> list[int]:
    """Ripple-carry add of two equal-length LSB-first bit lists. Returns n+1 bits."""
    out, c = [], cin
    for ai, bi in zip(a, b):
        s, c = _full_adder(ai, bi, c)
        out.append(s)
    out.append(c)
    return out


def _multiply_bits(a: list[int], b: list[int]) -> list[int]:
    """Unsigned `len(a)` x `len(b)` product, LSB-first, `len(a)+len(b)` bits.

    Column compression: every partial product `a_i & b_j` lands in column `i+j`, and
    each column is reduced with full/half adders until one bit is left, carries going
    to the next column. The top carry column is asserted empty -- an n x n product
    provably fits in 2n bits, so a non-empty one would mean a bug here, not an
    overflow to truncate.
    """
    n = len(a) + len(b)
    cols: list[list[int]] = [[] for _ in range(n + 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            cols[i + j].append(ai & bj)
    res = []
    for k in range(n):
        col = cols[k]
        while len(col) > 1:
            if len(col) >= 3:
                s, c = _full_adder(col.pop(), col.pop(), col.pop())
            else:
                x, y = col.pop(), col.pop()
                s, c = x ^ y, x & y
            col.append(s)
            cols[k + 1].append(c)
        res.append(col[0] if col else 0)
    assert not cols[n], "multiplier overflowed 2n bits -- impossible, so this is a bug"
    return res


# ---------------------------------------------------------------------------
# the task registry
# ---------------------------------------------------------------------------
#
# A task is named by a string that may carry a size, so the family and the scale live
# in one flag: `retina_x3`, `add6`, `mult4`. `_parse` turns the name into
# (family, size); `_SPECS` says what each family is. Parsing rather than enumerating
# is what makes "big brains" a knob instead of a code edit.

OPERATIONS = ("and", "or", "xor")

_PATTERNS = [
    (re.compile(r"^retina_ka2005$"), "retina", 1),      # experiment 4's name, kept
    (re.compile(r"^retina_x(\d+)$"), "retina", None),
    (re.compile(r"^left$"), "left", 1),
    (re.compile(r"^and2$"), "and2", 1),
    (re.compile(r"^copy$"), "copy", 1),
    (re.compile(r"^add(\d+)$"), "add", None),
    (re.compile(r"^mult(\d+)$"), "mult", None),
    (re.compile(r"^parity(\d+)$"), "parity", None),
]

# family -> (inputs, outputs, does --operation mean anything?) as functions of size
_SHAPE = {
    "retina": (lambda k: 8 * k, lambda k: k, True),
    "left":   (lambda k: 4, lambda k: 1, False),
    "and2":   (lambda k: 2, lambda k: 1, False),
    "copy":   (lambda k: 1, lambda k: 1, False),
    "add":    (lambda k: 2 * k + 1, lambda k: k + 1, False),
    "mult":   (lambda k: 2 * k, lambda k: 2 * k, False),
    "parity": (lambda k: k, lambda k: 1, False),
}

TASK_FAMILIES = tuple(_SHAPE)
EXAMPLE_TASKS = ("retina_ka2005", "retina_x2", "left", "and2", "copy",
                 "add4", "mult3", "parity12")


def _parse(task: str) -> tuple[str, int]:
    for rx, family, fixed in _PATTERNS:
        m = rx.match(task)
        if m:
            k = fixed if fixed is not None else int(m.group(1))
            if k < 1:
                raise ValueError(f"task size must be >= 1: {task!r}")
            return family, k
    raise ValueError(f"unknown task: {task!r} (families: {list(TASK_FAMILIES)}; "
                     f"e.g. {', '.join(EXAMPLE_TASKS)})")


def n_inputs(task: str) -> int:
    family, k = _parse(task)
    return _SHAPE[family][0](k)


def n_outputs(task: str) -> int:
    family, k = _parse(task)
    return _SHAPE[family][1](k)


def uses_operation(task: str) -> bool:
    """True for tasks where `--operation` (and therefore `--mvg`) means anything."""
    family, _ = _parse(task)
    return _SHAPE[family][2]


def n_patterns(task: str) -> int:
    """How many input patterns fitness is measured over.

    Exhaustive today, hence `2**n_in`. Everything downstream reads this rather than
    computing `1 << n_inputs(...)` itself, so a sampled-pattern variant is a change
    here and nowhere else.
    """
    return 1 << n_inputs(task)


def full_mask(task: str) -> int:
    """All-ones word for this task's pattern count (needed by inverting gates)."""
    return (1 << n_patterns(task)) - 1


def input_masks(task: str) -> tuple[int, ...]:
    """One truth-table mask per program input."""
    n_in = n_inputs(task)
    return tuple(_input_mask(i, n_in) for i in range(n_in))


def target_masks(task: str, operation: str = "and") -> tuple[int, ...]:
    """One truth-table mask per program OUTPUT. `operation` only affects retinas."""
    family, k = _parse(task)
    n_in = n_inputs(task)
    mask = full_mask(task)
    x = list(input_masks(task))

    if family == "copy":
        return (x[0],)
    if family == "and2":
        return (x[0] & x[1],)
    if family == "left":
        return (_ka_object(x[0], x[1], x[2], x[3], mask),)
    if family == "retina":
        outs = []
        for r in range(k):
            left, right = _retina_halves(x, 8 * r, mask)
            outs.append(_combine(operation, left, right, mask) & mask)
        return tuple(outs)
    if family == "add":
        a, b, cin = x[:k], x[k:2 * k], x[2 * k]
        return tuple(_add_bits(a, b, cin))
    if family == "mult":
        return tuple(_multiply_bits(x[:k], x[k:2 * k]))
    if family == "parity":
        # Even parity over all k inputs. One output, exactly `k` inputs, so it is the
        # one family that lets the pattern-count axis be swept ONE INPUT AT A TIME --
        # which is what `bench.py --crossover` needs to find where PyPy's big
        # integers stop paying. It is also a genuine CGP benchmark in its own right,
        # and a deliberately NON-modular one: every input matters to the single
        # output, so it is the control against which a decomposable task is read.
        acc = 0
        for m in x:
            acc ^= m
        return (acc & mask,)
    raise AssertionError(f"unhandled family {family!r}")


def input_groups(task: str) -> tuple[frozenset[int], ...]:
    """The task's ground-truth partition of the program inputs.

    This is what the structural readout classifies nodes against: a node whose input
    cone lies inside ONE group is "pure" (it is working on a single sub-problem), one
    spanning several is "mixed". It is a property of the TASK, not of any solution --
    it says how the problem decomposes, and the run reports how far the evolved
    circuit's wiring agrees.

    * `retina_ka2005` / `retina_x1` -- the two 2x2 blocks, so the columns match
      experiment 4's `left` / `right` exactly and the two experiments stay comparable.
    * `retina_xN`, N>1 -- one group per retina: the N sub-problems really are
      independent, which is the point of the family.
    * `addN` -- one group per bit position `{a_i, b_i}` (the carry-in joins position
      0), because a ripple-carry adder's modules are per-position full adders.
    * `multN` -- the two operand words. `[caution]` This is NOT a modular
      decomposition (every product bit depends on most of both operands); it is
      reported so the columns exist, and a multiplier is expected to read as almost
      entirely "mixed".
    * `parityN` -- one group, honestly: parity does not decompose at all.
    * `left` / `and2` / `copy` -- one group, so nothing is ever "mixed".
    """
    family, k = _parse(task)
    if family == "retina":
        if k == 1:
            return (frozenset(range(0, 4)), frozenset(range(4, 8)))
        return tuple(frozenset(range(8 * r, 8 * r + 8)) for r in range(k))
    if family == "add":
        g = [{i, k + i} for i in range(k)]
        g[0].add(2 * k)                                   # carry-in joins position 0
        return tuple(frozenset(s) for s in g)
    if family == "mult":
        return (frozenset(range(k)), frozenset(range(k, 2 * k)))
    # `parity`, `left`, `and2`, `copy`: one group. Parity's is a real claim, not a
    # fallback -- the function genuinely does not decompose, so nothing can be "pure"
    # in a way that means anything, and every node lands in the single group.
    return (frozenset(range(n_inputs(task))),)
