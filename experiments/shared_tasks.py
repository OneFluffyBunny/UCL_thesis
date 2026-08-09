"""Shared task definitions — the SINGLE source of truth for every experiment.

This is the canonical implementation of the boolean tasks. Both experiment 1
(compressed g-encoding) and experiment 2 (direct encoding) train on *exactly*
these tasks, by importing this module through a thin `tasks.py` shim in each
experiment directory. Do NOT copy this logic into an experiment — edit it here
and every experiment picks up the change, so `retina` (etc.) can never diverge.

A difficulty staircase of boolean problems, culminating in a retina-style task
(a modularly-decomposable task: target = OP(left-feature, right-feature)).

  copy          : output = input bit 0          (only 1 input matters -- the LNDP nemesis)
  and2          : output = bit0 AND bit1        (2 inputs matter)
  left          : output = left retina feature  (4 inputs -- one module)
  retina        : output = OP(left, right)      (8 inputs -- STAND-IN object rule)
  retina_ka2005 : output = OP(L, R)             (8 inputs -- the ORIGINAL Kashtan
                                                 & Alon 2005 object rule)

TWO RETINA VARIANTS LIVE HERE -- they are NOT interchangeable. Always record
which one a result used.

`retina` is the **stand-in**: left/right = (p0&p1)|(p2&p3). This repo's own docs
(root CLAUDE.md, kashtan_alon/README.md, RESULTS.md) call it "the Clune (2013)
reimplementation/adaptation", but that attribution is UNVERIFIED: Clune, Mouret
& Lipson 2013 ("The evolutionary origins of modularity", PMC3574393 /
arXiv:1207.2743) also uses an 8-pixel left/right retina derived from KA, but its
exact object rule is given only as a diagram in Figure 2a and is never spelled
out as text/formula anywhere in the paper's body -- so there is nothing to check
this formula against. Treat "Clune-derived" as an inherited, unconfirmed label,
not a fact.

`retina_ka2005` is **Kashtan & Alon's actual rule** from the 2005 PNAS paper
(102(39):13773-13778, doi:10.1073/pnas.0503610102), quoted against the primary
source PMC1236541: *"A left object is defined by three or more black pixels or
one or two black pixels in the left column only"*, the right object *"in a
similar way, with one or two black pixels in the right column only"*. See
kashtan_alon/PAPER_SPEC.md for the full verified spec; this is a JAX port of the
numpy implementation in kashtan_alon/tasks.py, which test_tasks.py checks
against KA's own truth counts.

WHY THE SECOND VARIANT EXISTS (the point, not a detail). KA's rule is TRUE for
exactly 8 of the 16 half-patterns; the stand-in's is TRUE for 7 of 16. That one
pattern decides whether the fitness metric can be gamed:

                          stand-in (7/16)      KA 2005 (8/16)
  (L,R) cell sizes        49 / 63 / 63 / 81    64 / 64 / 64 / 64
  constant-0, raw acc     0.809  <- trap       0.750
  one-side (y=L), raw     0.754                0.750
  constant-0, balanced    0.500                0.500
  one-side, balanced      0.848  <- trap       0.833

With KA's equal cells, raw fraction-correct IS accuracy averaged over the four
(L,R) cells, so EVERY predictor that reads only one half is provably capped at
0.75 -- ties with the constant predictor, and the remaining 0.25 is reachable
only by something that computes both halves. That is the property that makes the
task a real modularity probe, and KA got it for free from their object rule
rather than from a special metric. The stand-in has no such cap: under raw acc
the constant predictor is a 0.809 trap, and under balanced acc the ONE-MODULE
predictor is a 0.848 trap (balancing up-weights the rare positives, and every
positive requires L=1, which is exactly what a one-side brain gets perfectly
right). Prefer `retina_ka2005` for any modularity claim on AND/OR.

Inputs are bit vectors of length n_in (values in {0,1}); the caller may re-encode
to {-1,+1}. Targets are single bits in {0,1}. More tasks may be added later --
register them in TASKS / targets().
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp


def all_binary_inputs(n_in: int) -> jnp.ndarray:
    """(2**n_in, n_in) array of every bit combination, values in {0,1} (float32)."""
    grid = list(itertools.product([0, 1], repeat=n_in))
    return jnp.asarray(grid, dtype=jnp.float32)


# boolean ops on {0,1}-valued float arrays
def _and(a, b):
    return a * b


def _or(a, b):
    return a + b - a * b


def _xor(a, b):
    return a + b - 2.0 * a * b


OPS = {"and": _and, "or": _or, "xor": _xor}


def _left_feature(x):
    """Function of the 4 left pixels: (p0 & p1) | (p2 & p3)."""
    return _or(_and(x[..., 0], x[..., 1]), _and(x[..., 2], x[..., 3]))


def _right_feature(x):
    """Function of the 4 right pixels: (p4 & p5) | (p6 & p7)."""
    return _or(_and(x[..., 4], x[..., 5]), _and(x[..., 6], x[..., 7]))


# ---------------------------------------------------------------------------
# Kashtan & Alon (2005) -- the ORIGINAL object rule from the paper.
# Used by task `retina_ka2005`. Do NOT edit to match the stand-in above.
# ---------------------------------------------------------------------------

def _ka_object(outer_a, outer_b, inner_a, inner_b):
    """KA 2005's object rule for one 2x2 retina block.

    TRUE iff >=3 of the block's 4 pixels are black, OR 1-2 black pixels are
    confined to the block's OUTER column (both inner-column pixels white):

        object = [a+b+c+d >= 3] | [~c & ~d & (a | b)]

    with (a, b) the OUTER column and (c, d) the INNER one. The two clauses are
    disjoint (the outer-only clause caps at 2 black), so the rule is TRUE for
    exactly 5 + 3 = 8 of the 16 patterns -- the equal-halves property the whole
    task depends on (see module docstring).

    Note the rule is MIRROR-symmetric, not translation-symmetric: each block's
    privileged column is the one on the OUTSIDE of the retina.
    """
    n_black = outer_a + outer_b + inner_a + inner_b
    three_or_more = n_black >= 3
    outer_only = (inner_a == 0) & (inner_b == 0) & ((outer_a + outer_b) >= 1)
    return (three_or_more | outer_only).astype(jnp.float32)


def _ka_left_feature(x):
    """KA 'left object'. Left block; OUTER (leftmost) column = x0,x1; inner = x2,x3.

        pixel index:   0 2 | 4 6
                       1 3 | 5 7
    """
    return _ka_object(x[..., 0], x[..., 1], x[..., 2], x[..., 3])


def _ka_right_feature(x):
    """KA 'right object'. Right block; OUTER (rightmost) column = x6,x7; inner = x4,x5."""
    return _ka_object(x[..., 6], x[..., 7], x[..., 4], x[..., 5])


TASKS = ["copy", "and2", "left", "retina", "retina_ka2005"]

_MIN_INPUTS = {"copy": 1, "and2": 2, "left": 4, "retina": 8, "retina_ka2005": 8}

# Tasks of the form OP(L, R) -- the only ones whose target depends on
# --operation, and therefore the only ones for which --mvg means anything.
# Every other task ignores the operation flag entirely, so callers must not log
# it, stamp it into a run name, or let --mvg silently do nothing. Kept here (not
# hardcoded at the call sites) so adding a third retina variant updates every
# experiment at once.
_USES_OPERATION = frozenset({"retina", "retina_ka2005"})


def min_inputs(task: str) -> int:
    return _MIN_INPUTS[task]


def uses_operation(task: str) -> bool:
    """True if `targets(task, op, X)` actually depends on `op`."""
    if task not in _MIN_INPUTS:
        raise ValueError(f"unknown task: {task!r} (known: {TASKS})")
    return task in _USES_OPERATION


def targets(task: str, operation: str, X: jnp.ndarray) -> jnp.ndarray:
    """Compute the target bit for every row of X. `operation` only affects retina*."""
    if task == "copy":
        y = X[..., 0]
    elif task == "and2":
        y = _and(X[..., 0], X[..., 1])
    elif task == "left":
        y = _left_feature(X)
    elif task == "retina":
        y = OPS[operation](_left_feature(X), _right_feature(X))
    elif task == "retina_ka2005":
        y = OPS[operation](_ka_left_feature(X), _ka_right_feature(X))
    else:
        raise ValueError(f"unknown task: {task!r} (known: {TASKS})")
    return y.astype(jnp.int32)
