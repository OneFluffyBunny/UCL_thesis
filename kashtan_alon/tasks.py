"""Retina + boolean tasks for the Kashtan-Alon reproduction — pure numpy, no JAX.

Reimplemented in numpy so the GA pulls in ZERO heavy dependencies and never
touches a GPU. The old version re-exported ``experiments/shared_tasks.py``, which
imports ``jax.numpy`` just to build a 256x8 constant grid — and on a CUDA box that
import alone makes JAX preallocate ~75% of VRAM for nothing. The whole compute
path here is numpy on CPU, so importing JAX was pure downside.

These definitions are byte-for-byte the same functions as the project's single
source of truth (``experiments/shared_tasks.py``); ``test_tasks.py`` asserts that
equality across every input pattern and operation. If you change the retina in
shared_tasks.py, mirror it here and re-run ``python test_tasks.py``.
"""

from __future__ import annotations

import itertools

import numpy as np


def all_binary_inputs(n_in: int) -> np.ndarray:
    """(2**n_in, n_in) array of every bit combination, values in {0,1} (float32)."""
    grid = list(itertools.product([0, 1], repeat=n_in))
    return np.asarray(grid, dtype=np.float32)


# boolean ops on {0,1}-valued float arrays (identical algebra to shared_tasks.py)
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
