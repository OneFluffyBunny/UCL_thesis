"""Shared task definitions — the SINGLE source of truth for every experiment.

This is the canonical implementation of the boolean tasks. Both experiment 1
(compressed g-encoding) and experiment 2 (direct encoding) train on *exactly*
these tasks, by importing this module through a thin `tasks.py` shim in each
experiment directory. Do NOT copy this logic into an experiment — edit it here
and every experiment picks up the change, so `retina` (etc.) can never diverge.

A difficulty staircase of boolean problems, culminating in the Kashtan-Alon
retina (a modularly-decomposable task: target = OP(left-feature, right-feature)).

  copy   : output = input bit 0          (only 1 input matters -- the LNDP nemesis)
  and2   : output = bit0 AND bit1        (2 inputs matter)
  left   : output = left retina feature  (4 inputs -- one module)
  retina : output = OP(left, right)      (8 inputs -- full Kashtan-Alon)

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


TASKS = ["copy", "and2", "left", "retina"]

_MIN_INPUTS = {"copy": 1, "and2": 2, "left": 4, "retina": 8}


def min_inputs(task: str) -> int:
    return _MIN_INPUTS[task]


def targets(task: str, operation: str, X: jnp.ndarray) -> jnp.ndarray:
    """Compute the target bit for every row of X. `operation` only affects retina."""
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
    return y.astype(jnp.int32)
