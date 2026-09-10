"""Fan-in ablation vs the constrained paper runs, side by side.

The ablation (`run_ablation_no_fanin.py`) changes exactly one thing:
`NetConfig(fan_in=())`, so a neuron may read every node in the layer below
instead of at most 3 (hidden) / 2 (deeper). Everything else -- task, pop 600,
25,000 generations, elite 150, Pc 0.5, Pm 0.5, switch every 20 -- is the paper
preset. It tests whether MVG alone produces KA's modularity, or whether wiring
SCARCITY is doing the work.

Every accuracy here is goal-matched: read from the per-generation CSV's rows
whose `op` is AND, so an MVG arm is never compared on OR against an FG arm on
AND. That needs no saved genomes, so it works for runs made before brains were
archived. Q_m/density come from result.json, i.e. the final-generation champion
-- for MVG that champion is an OR brain, which was measured (RESULTS.md) to move
Q_m by ~0.05 in MVG's disfavour and purity by <0.01.

Usage: conda run -n lndp python kashtan_alon/analysis/ablation_table.py
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

import model as M
import train as T

CONDITIONS = [("capped  (paper)", _HERE.parents[1] / "runs_purity"),
              ("no fan-in cap", _HERE.parents[1] / "runs_no_fanin")]
ARMS = [("mvg", "MVG"), ("fg", "FG ")]
LATE = 2000          # generations at the end of the run averaged over
GOAL = "and"


def read(d, arm, seed):
    p = os.path.join(d, f"retina_{arm}_raw_seed{seed}_log.csv")
    if not os.path.exists(p):
        return None, None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    r = None
    rp = os.path.join(d, f"retina_{arm}_raw_seed{seed}_result.json")
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
    return rows, r


def champion_purity(d, arm, seed):
    """Circuit purity of the saved final-generation champion.

    The ablation runs predate the CSV's `purity` column, so purity is taken from
    the saved genome in BOTH conditions rather than from the log in one and the
    genome in the other -- otherwise the two columns would not be the same
    measurement."""
    p = os.path.join(d, f"retina_{arm}_raw_seed{seed}_best.npz")
    if not os.path.exists(p):
        return np.nan
    npz = np.load(p)
    n = sum(1 for k in npz.files if k.startswith("w"))
    return T.purity_of([npz[f"w{l}"] for l in range(n)], M.NetConfig())


def seed_stats(rows, res, pur):
    g_max = max(int(x["gen"]) for x in rows)
    late = [x for x in rows if int(x["gen"]) >= g_max - LATE]
    on_goal = [x for x in late if x["op"] == GOAL]
    return {
        "acc_and": np.mean([float(x["best_fit"]) for x in on_goal]) if on_goal else np.nan,
        "acc_and_max": np.max([float(x["best_fit"]) for x in on_goal]) if on_goal else np.nan,
        "purity": pur,
        "q_m": res.get("q_m", np.nan) if res else np.nan,
        "edges": res.get("edges", np.nan) if res else np.nan,
        "density": res.get("density", np.nan) if res else np.nan,
    }


def main():
    table = {}
    for cond, d in CONDITIONS:
        for arm, alabel in ARMS:
            per = []
            for s in range(5):
                rows, res = read(str(d), arm, s)
                if rows:
                    per.append(seed_stats(rows, res, champion_purity(str(d), arm, s)))
            if per:
                table[(cond, alabel)] = per

    cols = [("Q_m", "q_m", "{:+.3f}"), ("accuracy on AND", "acc_and", "{:.3f}"),
            ("best on AND", "acc_and_max", "{:.3f}"),
            ("circuit purity", "purity", "{:.3f}"),
            ("edges", "edges", "{:.0f}"), ("density %", "density", "{:.1f}")]

    print(f"Goal-matched over the last {LATE} generations of each run; "
          f"mean +- SD across seeds.\n")
    head = f"{'condition':17s}{'arm':5s}{'n':>3s}  " + "  ".join(
        f"{c[0]:>18s}" for c in cols)
    print(head)
    print("-" * len(head))
    for cond, _ in CONDITIONS:
        for _, alabel in ARMS:
            per = table.get((cond, alabel))
            if not per:
                continue
            cells = []
            for _, key, fmt in cols:
                v = np.array([p[key] for p in per], dtype=float)
                cells.append(f"{fmt.format(np.nanmean(v))}+-{np.nanstd(v, ddof=1):.3f}"
                             if key in ("q_m", "acc_and", "acc_and_max", "purity")
                             else f"{fmt.format(np.nanmean(v))}")
            print(f"{cond:17s}{alabel:5s}{len(per):>3d}  " +
                  "  ".join(f"{c:>18s}" for c in cells))
        print()

    print("MVG - FG gap per condition (the thing the ablation is testing):")
    for cond, _ in CONDITIONS:
        m, f = table.get((cond, "MVG")), table.get((cond, "FG "))
        if not (m and f):
            continue
        for label, key in (("Q_m", "q_m"), ("accuracy on AND", "acc_and"),
                           ("circuit purity", "purity")):
            dm = np.nanmean([p[key] for p in m]) - np.nanmean([p[key] for p in f])
            print(f"  {cond:17s} {label:18s} MVG - FG = {dm:+.3f}")
        print()

    print("Per-seed Q_m:")
    for cond, _ in CONDITIONS:
        for _, alabel in ARMS:
            per = table.get((cond, alabel))
            if per:
                print(f"  {cond:17s}{alabel:5s}"
                      + "  ".join(f"{p['q_m']:+.3f}" for p in per))
    return 0


if __name__ == "__main__":
    sys.exit(main())
