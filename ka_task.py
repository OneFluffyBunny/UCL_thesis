"""Kashtan-Alon-style retina task (AND of left/right sub-features) for NDP.

This is the project's stand-in retina task, NOT Kashtan & Alon (2005)'s actual
Fig. 5a object rule -- same formula used by experiment_1/experiment_2/experiment_3
in the sibling UCL_thesis/experiments/shared_tasks.py, which is the canonical
source. It is duplicated here (rather than imported cross-repo) on purpose: NDP
is meant to be its own standalone, forkable git repo (see root CLAUDE.md), and a
sys.path import into a sibling folder outside this repo would silently break on
any machine that only clones NDP by itself (e.g. the remote GPU box). If the
formula in shared_tasks.py ever changes, update this file to match by hand.

    left_feature(x)  = (x0 & x1) | (x2 & x3)
    right_feature(x) = (x4 & x5) | (x6 & x7)
    target            = left_feature(x) AND right_feature(x)

256 input patterns (8 bits), single binary target per pattern.
"""

import itertools

import numpy as np


def all_binary_inputs(n_in: int) -> np.ndarray:
    """(2**n_in, n_in) array of every {0,1} bit combination."""
    return np.array(list(itertools.product([0, 1], repeat=n_in)), dtype=np.float64)


def _and(a, b):
    return a * b


def _or(a, b):
    return a + b - a * b


def _left_feature(x):
    return _or(_and(x[..., 0], x[..., 1]), _and(x[..., 2], x[..., 3]))


def _right_feature(x):
    return _or(_and(x[..., 4], x[..., 5]), _and(x[..., 6], x[..., 7]))


def retina_and_dataset():
    """Returns (X, Y): X is (256, 8) in {0,1}, Y is (256,) in {0,1} = left AND right."""
    X = all_binary_inputs(8)
    Y = _and(_left_feature(X), _right_feature(X)).astype(np.int64)
    return X, Y


def to_bipolar(X: np.ndarray) -> np.ndarray:
    """{0,1} -> {-1,+1}."""
    return 2.0 * X - 1.0
