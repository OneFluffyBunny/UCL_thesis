"""The pure-Python task shim must give the SAME masks as the numpy shim it replaced.

Run under CPython (needs numpy): conda run -n lndp python test_tasks.py

`necgp/tasks.py` is the old numpy shim, byte-identical to what this directory used
before 2026-09-14 apart from its header comment, so it is the reference. If this
passes, every seed run before the swap reproduces after it.
"""

from __future__ import annotations

import importlib.util
import pathlib

import tasks

_OLD = pathlib.Path(__file__).resolve().parents[1] / "necgp" / "tasks.py"


def _load_old():
    spec = importlib.util.spec_from_file_location("_necgp_numpy_tasks", _OLD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_masks_match_numpy_shim() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("  skip test_masks_match_numpy_shim (no numpy -- running under PyPy?)")
        return
    old = _load_old()
    n = 0
    for task in tasks.TASKS:
        assert tasks.n_inputs(task) == old.n_inputs(task), task
        assert tasks.n_patterns(task) == old.n_patterns(task), task
        assert tasks.full_mask(task) == old.full_mask(task), task
        assert tasks.input_masks(task) == old.input_masks(task), task
        for op in tasks.OPERATIONS:
            assert tasks.target_mask(task, op) == old.target_mask(task, op), (task, op)
            n += 1
    print(f"ok  pure-Python masks == numpy shim on {len(tasks.TASKS)} tasks, {n} targets")


if __name__ == "__main__":
    test_masks_match_numpy_shim()
    print("\nall tests passed")
