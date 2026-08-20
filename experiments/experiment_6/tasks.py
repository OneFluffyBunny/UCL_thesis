"""The even-parity curriculum -- PAPER_SPEC.md section VI-A, the paper's own task.

Same truth-table-mask packing as experiment_4/5's `tasks.py`: pattern `r`'s bit
for input `i` is bit `(n_in-1-i)` of `r` (input 0 varies slowest). "Parity" here
means n-input XOR; the paper's own worked proof (section VII-C) lands on
`x0 XOR x1 XOR 1` for 2-input, i.e. XOR composed with a fixed inversion -- which
convention is "even" vs "odd" parity is a labelling choice and does not change
task difficulty, so this module just implements XOR.
"""

from __future__ import annotations


def input_masks(n_in: int) -> list[int]:
    n_patterns = 1 << n_in
    return [sum(1 << r for r in range(n_patterns) if (r >> (n_in - 1 - i)) & 1)
           for i in range(n_in)]


def full_mask(n_in: int) -> int:
    return (1 << (1 << n_in)) - 1


def target_parity(n_in: int) -> int:
    """XOR of all `n_in` inputs, as a truth-table mask."""
    m = full_mask(n_in)
    t = 0
    for im in input_masks(n_in):
        t ^= im
    return t & m
