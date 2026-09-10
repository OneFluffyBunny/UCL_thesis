"""Is "MVG reaches higher accuracy than FG" real, or an artefact of the comparison?

The suspicion is specific and reasonable: the MVG curve is scored on whatever goal
is live that generation, so half its points are OR and half are AND, while FG is
always AND. If OR is simply an EASIER goal (raw accuracy is not class-balanced),
MVG would look better while being no better.

Four independent checks, each able to kill the result on its own:

  1. TASK ASYMMETRY -- base rates of AND vs OR over all 256 patterns, and what a
     constant output / a one-side shortcut scores on each. If OR is easier, this
     is where it shows.
  2. THE PLOTTED CURVE, SPLIT BY GOAL -- MVG's own best_fit on AND-phase rows vs
     OR-phase rows, late in the run. If the advantage is goal asymmetry, the two
     must differ by roughly the size of the advantage.
  3. GOAL-MATCHED CHAMPIONS -- every saved final champion scored on BOTH goals.
     MVG-on-AND vs FG-on-AND is the comparison with no asymmetry left in it.
  4. BALANCED ACCURACY -- the same champions under the shortcut-proof metric
     (chance = 0.5), in case raw accuracy is rewarding class priors.

Usage: conda run -n lndp python kashtan_alon/scratch_audit_mvg_accuracy.py
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
import tasks
import train as T
from model import NetConfig

RUNS = {"runs_purity (the figure's data)": _HERE.parents[1] / "runs_purity",
        "runs (the 10 paper runs)": _HERE.parents[1] / "runs"}
OPS = ("and", "or")
N_SEEDS = 5
LATE = 1000        # generations at the end of the run used for the "late" means


def load_champion(d, name):
    npz = np.load(os.path.join(d, f"{name}_best.npz"))
    n = sum(1 for k in npz.files if k.startswith("w"))
    return [npz[f"w{l}"] for l in range(n)], [npz[f"b{l}"] for l in range(n)]


def score(wm, bm, X, y, metric):
    """One individual's accuracy -- pack it as a population of 1."""
    w = [np.asarray(m)[None] for m in wm]
    b = [np.asarray(v)[None] for v in bm]
    return float(M.fitness(w, b, X, y, NetConfig(), metric)[0])


def check_1_task_asymmetry(X, Y):
    print("=" * 78)
    print("1. IS OR AN EASIER GOAL THAN AND?  (all 256 patterns, raw accuracy)")
    print("=" * 78)
    left = np.asarray(tasks.targets("left", "and", X))
    right = np.asarray(tasks._right_feature(X)).astype(int)
    for op in OPS:
        y = Y[op]
        pos = y.mean()
        const = max(pos, 1 - pos)                      # best constant output
        sl = (left == y).mean()                        # "just report the left eye"
        sr = (right == y).mean()
        print(f"  {op.upper():3s}: positives {pos:.3f} of patterns | "
              f"best CONSTANT output {const:.3f} | "
              f"left-eye-only {sl:.3f} | right-eye-only {sr:.3f}")
    print("  -> a goal is 'easier' to the extent these baselines are high.")


def check_2_curve_split(d, label):
    print("=" * 78)
    print(f"2. THE PLOTTED CURVE, SPLIT BY GOAL -- {label}")
    print(f"   mean best_fit over the last {LATE} generations")
    print("=" * 78)
    for arm in ("mvg", "fg"):
        for s in range(N_SEEDS):
            p = os.path.join(d, f"retina_{arm}_raw_seed{s}_log.csv")
            if not os.path.exists(p):
                continue
            with open(p, newline="") as f:
                rows = list(csv.DictReader(f))
            g_max = max(int(r["gen"]) for r in rows)
            late = [r for r in rows if int(r["gen"]) >= g_max - LATE]
            by = {}
            for op in OPS:
                v = [float(r["best_fit"]) for r in late if r["op"] == op]
                if v:
                    by[op] = (np.mean(v), len(v))
            txt = "  ".join(f"on {o.upper()} {m:.4f} (n={n})" for o, (m, n) in by.items())
            gap = (f"   gap {abs(by['and'][0] - by['or'][0]):.4f}"
                   if len(by) == 2 else "")
            print(f"  {arm.upper():3s} seed{s}: {txt}{gap}")
    print("  -> if MVG's edge were goal asymmetry, its AND rows would sit at FG's level.")


def check_3_4_champions(d, label, X, Y):
    print("=" * 78)
    print(f"3+4. SAVED CHAMPIONS, SCORED ON BOTH GOALS -- {label}")
    print("=" * 78)
    print(f"  {'run':22s} {'raw AND':>8s} {'raw OR':>8s} "
          f"{'bal AND':>8s} {'bal OR':>8s}   logged")
    out = {}
    for arm in ("mvg", "fg"):
        rows = []
        for s in range(N_SEEDS):
            name = f"retina_{arm}_raw_seed{s}"
            if not os.path.exists(os.path.join(d, f"{name}_best.npz")):
                continue
            wm, bm = load_champion(d, name)
            raw = {o: score(wm, bm, X, Y[o], "raw") for o in OPS}
            bal = {o: score(wm, bm, X, Y[o], "balanced") for o in OPS}
            with open(os.path.join(d, f"{name}_result.json")) as f:
                r = json.load(f)
            logged = r.get("final_fit", r.get("best_fit"))
            lop = r.get("final_op", r.get("best_op"))
            print(f"  {name:22s} {raw['and']:8.4f} {raw['or']:8.4f} "
                  f"{bal['and']:8.4f} {bal['or']:8.4f}   {logged:.4f} on {lop}")
            rows.append((raw, bal))
        out[arm] = rows
    for metric, idx in (("raw", 0), ("balanced", 1)):
        for op in OPS:
            vals = {a: [r[idx][op] for r in out.get(a, [])] for a in ("mvg", "fg")}
            if not vals["mvg"] or not vals["fg"]:
                continue
            m, f_ = np.mean(vals["mvg"]), np.mean(vals["fg"])
            print(f"  MEAN {metric:8s} on {op.upper():3s}: "
                  f"MVG {m:.4f} {[f'{v:.3f}' for v in vals['mvg']]}   "
                  f"FG {f_:.4f} {[f'{v:.3f}' for v in vals['fg']]}   "
                  f"delta {m - f_:+.4f}")
    return out


def check_5_matched_goal_champions(d, label, X, Y):
    """The strongest test available without a replay.

    train.py archives the champion BRAIN at every logged generation, so for an MVG
    run the champions archived during AND epochs are exactly "the best AND-solver
    the population held at that moment" -- goal-matched to an FG champion by
    construction. Scored under BOTH raw and balanced accuracy, because raw's best
    constant output already scores 0.750 here and could flatter either arm."""
    print("=" * 78)
    print(f"5. GOAL-MATCHED CHAMPIONS FROM THE ARCHIVE -- {label}")
    print(f"   champions archived during AND epochs, last {LATE} generations")
    print("=" * 78)
    agg = {}
    for arm in ("mvg", "fg"):
        raws, bals = [], []
        for s in range(N_SEEDS):
            name = f"retina_{arm}_raw_seed{s}"
            bpath = os.path.join(d, f"{name}_brains.npz")
            lpath = os.path.join(d, f"{name}_log.csv")
            if not (os.path.exists(bpath) and os.path.exists(lpath)):
                continue
            with open(lpath, newline="") as f:
                op_at = {int(r["gen"]): r["op"] for r in csv.DictReader(f)}
            gens, ws, bs = T.BrainArchive.load(bpath)
            g_max = int(gens[-1])
            sel = [i for i, g in enumerate(gens)
                   if op_at.get(int(g)) == "and" and int(g) >= g_max - LATE]
            if not sel:
                continue
            r = [score(ws[i], bs[i], X, Y["and"], "raw") for i in sel]
            b = [score(ws[i], bs[i], X, Y["and"], "balanced") for i in sel]
            print(f"  {arm.upper():3s} seed{s}: on AND raw {np.mean(r):.4f} "
                  f"(max {np.max(r):.4f})  balanced {np.mean(b):.4f} "
                  f"(max {np.max(b):.4f})   n={len(sel)} archived champions")
            raws.append(np.max(r))
            bals.append(np.max(b))
        agg[arm] = (raws, bals)
    if all(agg.get(a, ([], []))[0] for a in ("mvg", "fg")):
        for i, metric in ((0, "raw"), (1, "balanced")):
            m, f_ = agg["mvg"][i], agg["fg"][i]
            print(f"  BEST-ON-AND, {metric:8s}: MVG {np.mean(m):.4f} "
                  f"{[f'{v:.3f}' for v in m]}   FG {np.mean(f_):.4f} "
                  f"{[f'{v:.3f}' for v in f_]}   delta {np.mean(m) - np.mean(f_):+.4f}")
    print("  -> same goal, same metric, same architecture, same GA. Nothing left "
          "for goal asymmetry to explain.")


def main():
    cfg = NetConfig()
    X = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    Y = {o: np.asarray(tasks.targets("retina", o, X)) for o in OPS}

    check_1_task_asymmetry(X, Y)
    for label, d in RUNS.items():
        if not d.exists():
            continue
        print()
        check_2_curve_split(str(d), label)
        print()
        check_3_4_champions(str(d), label, X, Y)
        print()
        check_5_matched_goal_champions(str(d), label, X, Y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
