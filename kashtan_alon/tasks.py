"""Retina + boolean tasks for the Kashtan-Alon reproduction -- pure numpy, no JAX.

The **retina task is Kashtan & Alon's actual one** (PNAS 2005, Fig. 5a, verified
against PMC1236541) -- NOT the project's stand-in `(p0&p1)|(p2&p3)` retina used by
experiments 1-3. The 8 pixels are a 4-wide x 2-high grid, split into a left 2x2
block (inputs 0-3) and a right 2x2 block (inputs 4-7):

    input index:   0 2 | 4 6        columns ->  L-outer L-inner | R-inner R-outer
                   1 3 | 5 7        rows: top / bottom
    (x0,x1) = left  block's LEFT (outer) column;  (x2,x3) = its inner column
    (x6,x7) = right block's RIGHT (outer) column; (x4,x5) = its inner column

KA's object rules (verbatim): *"A left object is defined by three or more black
pixels or one or two black pixels in the left column only"*; the right object is
*"defined in a similar way, with one or two black pixels in the right column
only"*.  Each rule is TRUE for exactly 8 of the 16 half-patterns (5 with >=3 black,
3 with 1-2 black confined to the outer column). Goals: G_AND = L AND R,
G_OR = L OR R, shared L/R sub-features -> modularly varying.

Pure numpy so the GA never imports JAX. NOTE: unlike the other tasks here, the
retina is deliberately DIFFERENT from `experiments/shared_tasks.py` -- do not
"re-sync" it; `test_tasks.py` checks the KA definition directly.
"""

from __future__ import annotations

import itertools

import numpy as np


def all_binary_inputs(n_in: int) -> np.ndarray:
    """(2**n_in, n_in) array of every bit combination, values in {0,1} (float32)."""
    grid = list(itertools.product([0, 1], repeat=n_in))
    return np.asarray(grid, dtype=np.float32)


# boolean ops on {0,1}-valued float arrays
def _and(a, b):
    return a * b


def _or(a, b):
    return a + b - a * b


def _xor(a, b):
    return a + b - 2.0 * a * b


OPS = {"and": _and, "or": _or, "xor": _xor}


def _object(outer_a, outer_b, inner_a, inner_b):
    """KA object rule for one 2x2 block. TRUE iff >=3 of the 4 pixels are black, OR
    1-2 black pixels confined to the block's OUTER column (inner column all zero).
    Arguments are the two OUTER-column pixels and the two INNER-column pixels
    (each {0,1}). Returns {0,1} float."""
    nblack = outer_a + outer_b + inner_a + inner_b
    ge3 = nblack >= 3
    outer_only = (inner_a == 0) & (inner_b == 0) & ((outer_a + outer_b) >= 1)
    return (ge3 | outer_only).astype(np.float32)


def _left_feature(x):
    """KA 'left object' over the left 2x2 block. Outer (left) column = x0,x1;
    inner column = x2,x3."""
    return _object(x[..., 0], x[..., 1], x[..., 2], x[..., 3])


def _right_feature(x):
    """KA 'right object' over the right 2x2 block. Outer (right) column = x6,x7;
    inner column = x4,x5."""
    return _object(x[..., 6], x[..., 7], x[..., 4], x[..., 5])


TASKS = ["copy", "and2", "left", "retina"]

_MIN_INPUTS = {"copy": 1, "and2": 2, "left": 4, "retina": 8}


def min_inputs(task: str) -> int:
    return _MIN_INPUTS[task]


def targets(task: str, operation: str, X: np.ndarray) -> np.ndarray:
    """Target bit for every row of X. `operation` only affects retina."""
    if task == "copy":
        y = X[..., 0]
    elif task == "and2":
        y = _and(X[..., 0], X[..., 1])
    elif task == "left":
        y = _left_feature(X)
    elif task == "retina":
        y = OPS[operation](_left_feature(X), _right_feature(X))
    else:
        raise ValueError(f"unknown task: {task!r} (known: {TASKS})")
    return y.astype(np.int32)
