"""Re-adaptation speed after a goal switch, compressed (experiment 1) vs direct
(experiment 2) encoding: the numbers in experiment_1/RESULTS.md, section 2.

    python recovery_stats.py
    python recovery_stats.py --arm budget_mvg --windows 100:300,1000:1200

Reads the per-generation replays written by `dense_replay.py`
(`experiment_{1,2}/runs/fgmvg_dense/<arm>/*seed<k>/log.csv`) and uses the
POPULATION MEAN accuracy on the active goal. For every full 20-generation goal
epoch in a window it measures:

  t90      generations to cover 90% of the climb from the trough (the value at the
           switch) to the epoch's own peak (`fig_switch_window.phase_stats`)
  gain3    accuracy gained in the first 3 generations after the switch
  +0.20    generations to climb 0.20 above the trough (censored at 20)
  0.75     generations to reach accuracy 0.75 (censored at 20)
  +0.30    generations to climb 0.30 above the trough (censored at 20)
  trough   accuracy at the switch

A seed's value is its mean over the window's epochs. The last three measures use
no peak, so they cannot be moved by the peak rising between windows.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from fig_switch_window import parse_windows, phase_stats   # noqa: E402

ENCODINGS = {"compressed": "experiment_1", "direct": "experiment_2"}
SWITCH = 20
LOWER_IS_FASTER = {"t90": True, "gain3": False, "+0.20": True, "0.75": True,
                   "+0.30": True, "trough": None}


def _window(run_dir: str, lo: int, hi: int) -> dict:
    with open(os.path.join(run_dir, "log.csv"), newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if lo <= int(r["gen"]) <= hi]
    return {"gen": np.array([int(r["gen"]) for r in rows]),
            "op": np.array([r["op"] for r in rows]),
            "mean_acc": np.array([float(r["mean_acc"]) for r in rows])}


def _epochs(d: dict) -> list[np.ndarray]:
    ops = d["op"]
    starts = [i for i in range(len(ops)) if i == 0 or ops[i] != ops[i - 1]]
    ends = starts[1:] + [len(ops)]
    return [d["mean_acc"][i:i + SWITCH] for i, e in zip(starts, ends)
            if e - i >= SWITCH and len(d["mean_acc"][i:i + SWITCH]) >= SWITCH]


def _first(seg: np.ndarray, level: float) -> int:
    hit = np.nonzero(seg >= level)[0]
    return int(hit[0]) if len(hit) else SWITCH


def seed_stats(run_dir: str, lo: int, hi: int) -> dict[str, float]:
    d = _window(run_dir, lo, hi)
    eps = _epochs(d)
    _, _, t90 = phase_stats(d, None, SWITCH, key="mean_acc")
    return {"t90": float(np.mean(t90)),
            "gain3": float(np.mean([e[3] - e[0] for e in eps])),
            "+0.20": float(np.mean([_first(e, e[0] + 0.2) for e in eps])),
            "0.75": float(np.mean([_first(e, 0.75) for e in eps])),
            "+0.30": float(np.mean([_first(e, e[0] + 0.3) for e in eps])),
            "trough": float(np.mean([e[0] for e in eps]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="budget_mvg")
    ap.add_argument("--windows", default="100:300,1000:1200",
                    help="exactly two windows: early,late")
    args = ap.parse_args()
    (e_lo, e_hi), (l_lo, l_hi) = parse_windows(args.windows)
    base = os.path.dirname(HERE)

    deltas = {}
    for enc, exp in ENCODINGS.items():
        dirs = sorted(glob.glob(os.path.join(base, exp, "runs", "fgmvg_dense", args.arm, "*seed*")))
        if not dirs:
            raise SystemExit(f"no dense replays under {exp}/runs/fgmvg_dense/{args.arm} "
                             f"(run dense_replay.py first)")
        early = [seed_stats(d, e_lo, e_hi) for d in dirs]
        late = [seed_stats(d, l_lo, l_hi) for d in dirs]
        print(f"\n{enc} ({exp}), {len(dirs)} seeds, early [{e_lo},{e_hi}] -> late [{l_lo},{l_hi}]")
        for k, lower in LOWER_IS_FASTER.items():
            a = np.array([s[k] for s in early])
            b = np.array([s[k] for s in late])
            faster = "" if lower is None else \
                f"  {int(np.sum(b < a) if lower else np.sum(b > a))}/{len(a)} seeds faster"
            print(f"  {k:7s} {a.mean():.3f} +- {a.std(ddof=1):.3f} -> "
                  f"{b.mean():.3f} +- {b.std(ddof=1):.3f}{faster}")
        deltas[enc] = np.array([l["t90"] - e["t90"] for e, l in zip(early, late)])
        print("  t90 late - early per seed: " + ", ".join(f"{x:+.1f}" for x in deltas[enc]))

    p = mannwhitneyu(deltas["compressed"], deltas["direct"],
                     alternative="two-sided", method="exact").pvalue
    print(f"\nt90 change, compressed vs direct: exact two-sided Mann-Whitney p = {p:.4f}")


if __name__ == "__main__":
    main()
