"""Task shim for necgp_pairwise -- pure Python, so the search runs under PyPy.

CHANGED 2026-09-14. This file used to load `kashtan_alon/tasks.py` (numpy) and pack
its arrays into bitmasks, the same shim `necgp/tasks.py` still uses. numpy does not
exist under PyPy, so it now loads `experiment_5/tasks.py` instead: the same KA 2005
rule re-expressed in mask algebra, no numpy.

The re-expression is not trusted on inspection, twice over:
- `experiment_5/test_tasks.py` asserts it equals `kashtan_alon/tasks.py` bit for bit.
- `test_tasks.py` (next to this file) asserts THIS shim's masks equal the old
  numpy shim's masks on every task and operation offered here, so every seed run
  before the swap still reproduces after it.

Only the single-output tasks are exposed: `ecgp.fitness` scores output 0 only, so a
multi-output task (`retina_x2`, `add4`, ...) would silently be scored on its first
output. Exposing those needs a multi-output fitness first.

Loading is by explicit file path via importlib, because this module is itself named
`tasks` and a `sys.path` import would find itself.
"""

from __future__ import annotations

import importlib.util
import pathlib

_EXP5_TASKS_PY = pathlib.Path(__file__).resolve().parents[2] / "experiment_5" / "tasks.py"

_spec = importlib.util.spec_from_file_location("_exp5_tasks_for_pairwise", _EXP5_TASKS_PY)
if _spec is None or _spec.loader is None:                      # pragma: no cover
    raise ImportError(f"could not load the task module at {_EXP5_TASKS_PY}")
_t5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t5)


# Public task names for --task (unchanged from the numpy shim).
TASKS = ("retina_ka2005", "left", "and2", "copy")

OPERATIONS = ("and", "or", "xor")


def _check(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task!r} (known: {list(TASKS)})")


def n_inputs(task: str) -> int:
    _check(task)
    return _t5.n_inputs(task)


def uses_operation(task: str) -> bool:
    _check(task)
    return _t5.uses_operation(task)


def input_masks(task: str) -> tuple[int, ...]:
    """One truth-table mask per program input, over all 2**n_in patterns."""
    _check(task)
    return tuple(_t5.input_masks(task))


def target_mask(task: str, operation: str) -> int:
    """Truth-table mask of the target bit. `operation` is ignored by non-retina tasks."""
    _check(task)
    (t,) = _t5.target_masks(task, operation)
    return t


def n_patterns(task: str) -> int:
    _check(task)
    return _t5.n_patterns(task)


def full_mask(task: str) -> int:
    """All-ones word for this task's pattern count (needed by inverting gates)."""
    _check(task)
    return _t5.full_mask(task)


def split_index(task: str) -> int | None:
    """Where the left/right input halves meet, for colouring drawings; None if no split."""
    return n_inputs(task) // 2 if task == "retina_ka2005" else None
