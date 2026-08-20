"""Computational (non-self-modifying) primitive functions for experiment 6 -- SMCGP.

Same bit-parallel-truth-table-mask convention as experiment_4/5's `gates.py`: one
Python int per wire, bit `r` holds the wire's value on input pattern `r`. Reused
here verbatim for the "restricted function set" (AND/OR/NAND/NOR -- the paper's
default, Table II of PAPER_SPEC.md, section VII-A) since it is bit-for-bit the same
semantics as experiment_4's default gate set.

Harding/Miller/Banzhaf additionally define a "full function set" BF0..BF15 (Table I
of PAPER_SPEC.md) -- the sixteen possible 2-input boolean functions, indexed by
their truth table read as a 4-bit integer over (a,b) in the order (0,0),(0,1),(1,0),
(1,1). Rather than hand-writing sixteen lambdas, `bf` computes any of them directly
from the four minterm masks -- see its docstring for the derivation. `smcgp.py`
wires whichever set is selected into the unified function table (INP + these +
the self-modifying operators all share one function-gene id space, PAPER_SPEC.md
section III-B).
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence


class Gate(NamedTuple):
    name: str
    fn: Callable[[int, int, int], int]     # (a, b, mask) -> mask; b is ignored by NOT


# Restricted set -- PAPER_SPEC.md section VII-A, and experiment_4's DEFAULT_GATES.
_RESTRICTED = {
    "and":  Gate("and",  lambda a, b, m: a & b),
    "or":   Gate("or",   lambda a, b, m: a | b),
    "nand": Gate("nand", lambda a, b, m: ~(a & b) & m),
    "nor":  Gate("nor",  lambda a, b, m: ~(a | b) & m),
    "not":  Gate("not",  lambda a, b, m: ~a & m),
}

DEFAULT_GATES = "and,nand,or,nor"


def bf(k: int, a: int, b: int, mask: int) -> int:
    """The 2-input boolean function indexed `k` in 0..15 (PAPER_SPEC.md Table I).

    Bit `i` of `k` gates whether minterm `i` is included in the result:
    bit 0 -> (a=1,b=1), bit 1 -> (a=1,b=0), bit 2 -> (a=0,b=1), bit 3 -> (a=0,b=0).
    Checked against the paper's own worked reductions (e.g. BF6 = XOR, BF7 = OR,
    BF8 = NOR, BF9 = XNOR, BF14 = NAND, BF15 = TRUE) in PAPER_SPEC.md.
    """
    out = 0
    if k & 1:
        out |= a & b
    if k & 2:
        out |= a & ~b
    if k & 4:
        out |= ~a & b
    if k & 8:
        out |= ~a & ~b
    return out & mask


def build_set(spec: str) -> dict[str, Gate]:
    """Parse a comma-separated `--gates` spec into name -> Gate.

    `spec` may name restricted-set gates (and/or/nand/nor/not) and/or `bf0`..`bf15`
    for the full set. `"full"` is shorthand for all sixteen BF functions.
    """
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError("--gates is empty")
    if names == ["full"]:
        names = [f"bf{k}" for k in range(16)]
    out: dict[str, Gate] = {}
    for n in names:
        if n in _RESTRICTED:
            out[n] = _RESTRICTED[n]
        elif n.startswith("bf") and n[2:].isdigit() and 0 <= int(n[2:]) <= 15:
            k = int(n[2:])
            out[n] = Gate(n, (lambda a, b, m, k=k: bf(k, a, b, m)))
        else:
            raise ValueError(f"unknown gate {n!r} (known: and,or,nand,nor,not,bf0..bf15,full)")
    if len(out) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate gate(s) in --gates: {dupes}")
    return out
