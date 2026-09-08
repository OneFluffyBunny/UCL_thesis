# Copied unmodified from experiments/experiment_4/gates.py (2026-08-19) -- see
# ../README.md and cgp.py's header note.
"""Primitive function set for experiment 4 -- pure Python ints, no JAX.

Gates operate on **truth-table masks**: one Python int per wire, where bit `r`
holds that wire's value on input pattern `r` (see `cgp.py` for why). A gate is
therefore a single bitwise op evaluated over *all* patterns at once.

Every gate takes `(args, mask)`. `mask` is the all-ones word for the current
pattern count and is required by every inverting gate: Python ints are
arbitrary-precision two's-complement, so `~x` goes negative and must be re-masked
to stay a well-formed truth table.

Each gate also carries a `slow` implementation over plain bools, used only by the
reference evaluator in `cgp.py` that `test_cgp.py` checks the fast path against.

The default set is the paper's (AND, NAND, OR, NOR) -- PAPER_SPEC.md section 8.
Selecting others via `--gates` is supported but is a deviation: in particular
adding `xor` hands the retina its combiner for free, the same effect the paper
notes about putting half-adders into the multiplier's function set.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence


# Opcodes. `cgp.evaluate` dispatches on these with an inline if/elif chain rather
# than calling `fn`, because a Python call plus building the `args` list costs more
# than the bitwise op itself (measured: gate dispatch was a large slice of a ~2 us
# per-node budget). The opcode keeps that fast path **general** -- it works for any
# `--gates` subset in any order -- where hard-coding function indices would silently
# break the moment someone reordered the flag.
OP_AND, OP_OR, OP_NAND, OP_NOR, OP_XOR, OP_XNOR, OP_NOT, OP_CONST0, OP_CONST1 = range(9)


class Gate(NamedTuple):
    name: str
    arity: int
    fn: Callable[[Sequence[int], int], int]      # over truth-table masks
    slow: Callable[[Sequence[bool]], bool]       # over single bools
    op: int                                      # opcode for cgp.evaluate's fast path


# The paper's set, plus the usual extras. Order here is irrelevant -- a genotype
# stores an index into the *selected* subset, which is built by `build_set`.
_REGISTRY = {
    "and":    Gate("and",    2, lambda a, m: a[0] & a[1],            lambda a: a[0] and a[1],        OP_AND),
    "or":     Gate("or",     2, lambda a, m: a[0] | a[1],            lambda a: a[0] or a[1],         OP_OR),
    "nand":   Gate("nand",   2, lambda a, m: ~(a[0] & a[1]) & m,     lambda a: not (a[0] and a[1]),  OP_NAND),
    "nor":    Gate("nor",    2, lambda a, m: ~(a[0] | a[1]) & m,     lambda a: not (a[0] or a[1]),   OP_NOR),
    "xor":    Gate("xor",    2, lambda a, m: a[0] ^ a[1],            lambda a: a[0] != a[1],         OP_XOR),
    "xnor":   Gate("xnor",   2, lambda a, m: ~(a[0] ^ a[1]) & m,     lambda a: a[0] == a[1],         OP_XNOR),
    "not":    Gate("not",    1, lambda a, m: ~a[0] & m,              lambda a: not a[0],             OP_NOT),
    "const0": Gate("const0", 1, lambda a, m: 0,                      lambda a: False,                OP_CONST0),
    "const1": Gate("const1", 1, lambda a, m: m,                      lambda a: True,                 OP_CONST1),
}

DEFAULT_GATES = "and,nand,or,nor"          # Table II / PAPER_SPEC.md section 8
ALL_GATES = sorted(_REGISTRY)


def build_set(spec: str) -> tuple[Gate, ...]:
    """Parse a comma-separated `--gates` string into an ordered tuple of Gates.

    Duplicates are rejected rather than silently collapsed: a repeated gate would
    change the mutation operator's sampling distribution over functions (it draws
    uniformly over the set), which is a quiet way to run a different experiment
    from the one the flags claim.
    """
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError("--gates is empty")
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"unknown gate(s): {unknown} (known: {ALL_GATES})")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate gate(s) in --gates: {dupes}")
    return tuple(_REGISTRY[n] for n in names)


def max_arity(gates: Sequence[Gate]) -> int:
    """Input genes allocated per node.

    CGP gives every node the same number of input genes -- the max arity over the
    function set -- and a gate simply ignores the surplus. Those surplus genes are
    still mutated and still inherited, which is a (deliberate) source of the
    neutrality CGP depends on.
    """
    return max(g.arity for g in gates)
