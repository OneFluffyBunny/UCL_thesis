"""Ashenhurst-Curtis column multiplicity: how many wires MUST cross a given split.

This measures the TASK, not any circuit -- it is the ceiling that says how modular
the best possible circuit could be, which is the denominator `Q_max`'s hill-climb
was only ever approximating.

Write the truth table as a chart -- rows = assignments to a bound set B, columns =
assignments to the rest. If the chart has mu distinct rows, then F factors as
G(H_1(B), ..., H_k(B), rest) with k = ceil(log2 mu), and no smaller k exists. So mu
is the exact interface width the task imposes on ANY circuit computing it.

Sweeping every 2-way split turns this into a partition FINDER: the split with the
smallest mu is the task's own modular decomposition, found without assuming L/R.

Runs from any working directory, in a few seconds.
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # experiment_4/: tasks

import tasks

TASK = "retina_ka2005"


def multiplicity(t: int, im, n_in: int, n_pat: int, bound: frozenset[int]) -> int:
    free = [i for i in range(n_in) if i not in bound]
    b = sorted(bound)
    rows: dict[int, dict[int, int]] = {}
    for p in range(n_pat):
        kb = sum(((im[i] >> p) & 1) << n for n, i in enumerate(b))
        kf = sum(((im[i] >> p) & 1) << n for n, i in enumerate(free))
        rows.setdefault(kb, {})[kf] = (t >> p) & 1
    seen = {tuple(sorted(r.items())) for r in rows.values()}
    return len(seen)


def main() -> int:
    n_in = tasks.n_inputs(TASK)
    n_pat = tasks.n_patterns(TASK)
    im = tasks.input_masks(TASK)

    for op in ("and", "or", "xor"):
        t = tasks.target_mask(TASK, op)
        lr = multiplicity(t, im, n_in, n_pat, frozenset(range(4)))
        print(f"{op:4s}  L/R split: mu = {lr:3d}  ->  {math.ceil(math.log2(lr))} "
              f"wire(s) must cross")

        best, worst = [], []
        for k in range(1, n_in):
            for b in itertools.combinations(range(n_in), k):
                if 0 not in b:                     # each unordered split once
                    continue
                m = multiplicity(t, im, n_in, n_pat, frozenset(b))
                best.append((m, k, b))
        best.sort()
        print("      narrowest splits: " + ", ".join(
            f"{list(b)}:mu={m}" for m, k, b in best[:4]))
        print(f"      widest mu over all splits: {max(m for m, _, _ in best)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
