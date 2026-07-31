"""Parity test: kashtan_alon/tasks.py (numpy) == experiments/shared_tasks.py (jax).

The GA uses the numpy `tasks.py` so it never imports JAX. This test is the
anti-drift guard: it imports BOTH and asserts the retina (and every other task /
operation) is bit-for-bit identical across all input patterns. Run it whenever
either file changes:

    conda run -n lndp python test_tasks.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import tasks  # numpy version (this folder)

_EXPERIMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, _EXPERIMENTS)
import shared_tasks as S  # jax version (source of truth)  # noqa: E402


def main():
    assert tasks.TASKS == S.TASKS, f"task lists differ: {tasks.TASKS} vs {S.TASKS}"
    for task in tasks.TASKS:
        n = tasks.min_inputs(task)
        assert n == S.min_inputs(task), f"min_inputs differ for {task}"
        Xn = tasks.all_binary_inputs(n)
        Xs = np.asarray(S.all_binary_inputs(n))
        assert np.array_equal(Xn, Xs), f"inputs differ for {task}"
        ops = ["and", "or", "xor"] if task == "retina" else ["and"]
        for op in ops:
            yn = np.asarray(tasks.targets(task, op, Xn))
            ys = np.asarray(S.targets(task, op, Xs))
            assert np.array_equal(yn, ys), f"targets differ for {task}/{op}"
            print(f"  ok  {task:8s} op={op:3s}  n={n}  positives={int(yn.sum())}/{len(yn)}")
    print("PARITY OK: numpy tasks.py is bit-identical to shared_tasks.py (jax)")


if __name__ == "__main__":
    main()
