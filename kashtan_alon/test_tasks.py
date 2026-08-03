"""Checks for kashtan_alon/tasks.py.

Two things are verified:
  1. `copy` and `and2` still match `experiments/shared_tasks.py` bit-for-bit (they
     are unchanged trivial helpers -- the anti-drift guard for those).
  2. The `retina` and `left` tasks implement KASHTAN & ALON's actual object rules
     (Fig. 5a), which DELIBERATELY differ from shared_tasks.py's stand-in retina.
     These are checked directly, by their exact truth counts.

    conda run -n lndp python test_tasks.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import tasks  # numpy version (this folder)


def check_shared_parity_for_trivial_tasks():
    """copy/and2 are unchanged -> must still equal shared_tasks.py."""
    _EXPERIMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    sys.path.insert(0, _EXPERIMENTS)
    import shared_tasks as S  # jax version  # noqa: E402
    for task in ["copy", "and2"]:
        n = tasks.min_inputs(task)
        Xn = tasks.all_binary_inputs(n)
        Xs = np.asarray(S.all_binary_inputs(n))
        yn = np.asarray(tasks.targets(task, "and", Xn))
        ys = np.asarray(S.targets(task, "and", Xs))
        assert np.array_equal(yn, ys), f"{task} drifted from shared_tasks.py"
        print(f"  ok  parity  {task:6s}  positives={int(yn.sum())}/{len(yn)}")


def check_ka_retina():
    """KA object rules: each side true for exactly 8/16 half-patterns, so over the
    full 256 patterns L and R are each true 128 times; AND=64, OR=192."""
    X = tasks.all_binary_inputs(8)

    # each object over its own 4 pixels, in isolation
    left16 = np.asarray(tasks.targets("left", "and", tasks.all_binary_inputs(4)))
    assert left16.sum() == 8, f"left object should be true for 8/16, got {left16.sum()}"
    print(f"  ok  left object   true for {int(left16.sum())}/16 half-patterns")

    y_and = np.asarray(tasks.targets("retina", "and", X))
    y_or = np.asarray(tasks.targets("retina", "or", X))
    assert y_and.sum() == 64, f"retina AND positives should be 64, got {y_and.sum()}"
    assert y_or.sum() == 192, f"retina OR positives should be 192, got {y_or.sum()}"
    print(f"  ok  retina AND    positives={int(y_and.sum())}/256")
    print(f"  ok  retina OR     positives={int(y_or.sum())}/256")

    # modularity of the TASK: the left object must not depend on right pixels (4-7),
    # and vice-versa -- this is what makes the goals "modularly varying".
    from tasks import _left_feature, _right_feature
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=(500, 8)).astype(np.float32)
    flipR = base.copy(); flipR[:, 4:] = 1 - flipR[:, 4:]      # flip only right pixels
    flipL = base.copy(); flipL[:, :4] = 1 - flipL[:, :4]      # flip only left pixels
    assert np.array_equal(_left_feature(base), _left_feature(flipR)), "left depends on right pixels!"
    assert np.array_equal(_right_feature(base), _right_feature(flipL)), "right depends on left pixels!"
    print("  ok  left/right objects are independent (modular task structure)")


def main():
    print("shared-parity (trivial tasks):")
    check_shared_parity_for_trivial_tasks()
    print("Kashtan-Alon retina rules:")
    check_ka_retina()
    print("ALL TASK CHECKS PASSED")


if __name__ == "__main__":
    main()
