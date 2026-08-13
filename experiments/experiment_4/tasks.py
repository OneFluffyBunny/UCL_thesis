"""Task shim for experiment 4 -- loads the verified KA task, packs it to bitmasks.

The task definition is NOT reimplemented here. It is loaded straight out of
`kashtan_alon/tasks.py`, the pure-numpy implementation that `kashtan_alon/
test_tasks.py` checks against Kashtan & Alon's own truth counts. If that rule is
ever corrected, this experiment inherits the correction.

⚠️ NAMING. The same KA 2005 object rule lives in two places under two names:

    kashtan_alon/tasks.py        task="retina"           (pure numpy)  <- loaded here
    experiments/shared_tasks.py  task="retina_ka2005"    (JAX)

This experiment is pure-Python/numpy by design (a boolean circuit search; JAX buys
nothing and the `kashtan_alon/` convention is no-JAX), so it loads the former but
exposes it under the latter's name, since `retina_ka2005` is the name the rest of
the project uses for this rule. `shared_tasks.py`'s *other* retina -- the
`(p0&p1)|(p2&p3)` stand-in -- is deliberately NOT offered: on that rule a
one-half predictor scores 193/256 under both AND and OR, a goal-invariant
one-module attractor that would fake a null on H1. See README.md.

Loading is by explicit file path via importlib rather than `sys.path` + `import
tasks`, because this module is itself named `tasks` and would otherwise be liable
to import itself.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_KA_TASKS_PY = pathlib.Path(__file__).resolve().parents[2] / "kashtan_alon" / "tasks.py"

_spec = importlib.util.spec_from_file_location("_ka_tasks_for_exp4", _KA_TASKS_PY)
if _spec is None or _spec.loader is None:                      # pragma: no cover
    raise ImportError(f"could not load the KA task module at {_KA_TASKS_PY}")
_ka = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ka)


# Public task names for --task, mapped to the name kashtan_alon/tasks.py uses.
TASKS = {
    "retina_ka2005": "retina",     # KA 2005 Fig. 5a object rule -- the default
    "left":          "left",       # one half-object only (4 inputs) -- a sanity task
    "and2":          "and2",
    "copy":          "copy",
}

# Which tasks actually depend on --operation, i.e. the only ones for which --mvg
# means anything. Mirrors shared_tasks._USES_OPERATION.
USES_OPERATION = frozenset({"retina_ka2005"})

OPERATIONS = ("and", "or", "xor")


def n_inputs(task: str) -> int:
    return int(_ka.min_inputs(TASKS[task]))


def uses_operation(task: str) -> bool:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task!r} (known: {sorted(TASKS)})")
    return task in USES_OPERATION


def _pack(bits: np.ndarray) -> int:
    """Pack a (N,) array of 0/1 into an N-bit int, row r -> bit r."""
    out = 0
    for r, b in enumerate(bits.tolist()):
        if b:
            out |= 1 << r
    return out


def input_masks(task: str) -> tuple[int, ...]:
    """One truth-table mask per program input, over all 2**n_in patterns.

    Pattern order is `itertools.product([0,1], repeat=n_in)` -- the same order the
    KA module uses to build X, so mask bit r and target bit r refer to the same
    pattern by construction.
    """
    n_in = n_inputs(task)
    X = _ka.all_binary_inputs(n_in)
    return tuple(_pack(X[:, i].astype(np.int8)) for i in range(n_in))


def target_mask(task: str, operation: str) -> int:
    """Truth-table mask of the target bit. `operation` is ignored by non-retina tasks."""
    n_in = n_inputs(task)
    X = _ka.all_binary_inputs(n_in)
    y = _ka.targets(TASKS[task], operation, X)
    return _pack(np.asarray(y, dtype=np.int8))


def n_patterns(task: str) -> int:
    return 1 << n_inputs(task)


def full_mask(task: str) -> int:
    """All-ones word for this task's pattern count (needed by inverting gates)."""
    return (1 << n_patterns(task)) - 1
