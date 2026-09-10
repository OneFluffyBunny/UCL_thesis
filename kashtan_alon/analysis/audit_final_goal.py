"""Under MVG the FINAL-generation champion is an OR brain. What does that spoil?

Generation 24,999 lands in an OR epoch for every seed (the schedule is
deterministic), so `<run>_best.npz`, `final_fit`, and the Q/Q_m/purity in
`<run>_result.json` all describe an OR specialist -- while every FG run's
equivalents describe an AND solver. Anything that puts those side by side is
comparing two different tasks.

This measures, per quantity, whether that actually changes the number:

  ACCURACY   the final (OR) champion on OR, vs the best AND champion on AND.
  STRUCTURE  Q, Q_m and circuit purity of the final (OR) champion vs the best
             AND champion -- and, time-matched, the mean of every archived
             champion from AND epochs vs OR epochs over the last LATE
             generations, which removes "the two brains come from different
             points in the run" as an explanation.

Usage: conda run -n lndp python kashtan_alon/scratch_audit_final_goal.py
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
from modularity import normalized_qm

D = str(_HERE.parents[1] / "runs_purity")
OPS = ("and", "or")
N_SEEDS = 5
LATE = 2000
QM_NRAND = 200      # smaller null than the runs' 1000: this compares Q_m to Q_m


def acc(wm, bm, X, y, cfg, metric="raw"):
    w = [np.asarray(m)[None] for m in wm]
    b = [np.asarray(v)[None] for v in bm]
    return float(M.fitness(w, b, X, y, cfg, metric)[0])


def main():
    cfg = NetConfig()
    X = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    Y = {o: np.asarray(tasks.targets("retina", o, X)) for o in OPS}

    print("=" * 92)
    print("A. WHAT THE FINAL-GENERATION CHAMPION ACTUALLY IS")
    print("=" * 92)
    print(f"  {'run':22s} {'last gen':>8s} {'goal':>5s} {'accAND':>7s} {'accOR':>7s}"
          f" {'purity':>7s} {'Q':>6s}   result.json says")
    finals, bests = {}, {}
    for arm in ("mvg", "fg"):
        for s in range(N_SEEDS):
            name = f"retina_{arm}_raw_seed{s}"
            with open(os.path.join(D, f"{name}_log.csv"), newline="") as f:
                op_at = {int(r["gen"]): r["op"] for r in csv.DictReader(f)}
            gens, ws, bs = T.BrainArchive.load(os.path.join(D, f"{name}_brains.npz"))
            i = len(gens) - 1
            g, op = int(gens[i]), op_at[int(gens[i])]
            a = {o: acc(ws[i], bs[i], X, Y[o], cfg) for o in OPS}
            pur = T.purity_of(ws[i], cfg)
            with open(os.path.join(D, f"{name}_result.json")) as f:
                r = json.load(f)
            print(f"  {name:22s} {g:8d} {op:>5s} {a['and']:7.3f} {a['or']:7.3f}"
                  f" {pur:7.3f} {r['q']:6.3f}   fit {r.get('final_fit', r.get('best_fit')):.3f}"
                  f" on {r.get('final_op', r.get('best_op'))}, Q_m {r['q_m']:+.3f}")
            finals[(arm, s)] = (ws[i], bs[i], op, a, pur, r)

            # best champion per goal, from the same archive
            per_goal = {}
            for o in OPS:
                cand = [(acc(ws[j], bs[j], X, Y[o], cfg), j) for j in range(len(gens))
                        if op_at.get(int(gens[j])) == o]
                if cand:
                    per_goal[o] = max(cand)
            bests[(arm, s)] = (per_goal, gens, ws, bs, op_at)

    print()
    print("=" * 92)
    print("B. ACCURACY -- is the reported MVG number inflated by being an OR number?")
    print("=" * 92)
    for arm in ("mvg", "fg"):
        for s in range(N_SEEDS):
            per_goal, gens, ws, bs, _ = bests[(arm, s)]
            _, _, fop, a, _, r = finals[(arm, s)]
            reported = a[fop]
            line = f"  {arm.upper():3s} seed{s}: reported {reported:.4f} (on {fop.upper()})"
            for o in OPS:
                if o in per_goal:
                    v, j = per_goal[o]
                    line += f" | best on {o.upper()} {v:.4f} (gen {int(gens[j])})"
            print(line)

    print()
    print("=" * 92)
    print("C. STRUCTURE -- final (OR) champion vs best AND champion, same run")
    print("=" * 92)
    print(f"  {'run':22s} {'Q fin':>6s} {'Q AND':>6s} | {'Qm fin':>7s} {'Qm AND':>7s}"
          f" | {'pur fin':>7s} {'pur AND':>7s}")
    dq, dp = {"mvg": [], "fg": []}, {"mvg": [], "fg": []}
    for arm in ("mvg", "fg"):
        for s in range(N_SEEDS):
            per_goal, gens, ws, bs, _ = bests[(arm, s)]
            wf, bf, fop, a, purf, r = finals[(arm, s)]
            _, j = per_goal["and"]
            qm_f, pf = normalized_qm(wf, cfg, n_rand=QM_NRAND, seed=s)
            qm_a, pa = normalized_qm(ws[j], cfg, n_rand=QM_NRAND, seed=s)
            pur_a = T.purity_of(ws[j], cfg)
            print(f"  retina_{arm}_raw_seed{s:<7d} {pf['q_real']:6.3f} {pa['q_real']:6.3f} | "
                  f"{qm_f:+7.3f} {qm_a:+7.3f} | {purf:7.3f} {pur_a:7.3f}")
            dq[arm].append((qm_f, qm_a))
            dp[arm].append((purf, pur_a))
    for arm in ("mvg", "fg"):
        f_, a_ = np.array(dq[arm]).mean(axis=0)
        pf_, pa_ = np.array(dp[arm]).mean(axis=0)
        print(f"  MEAN {arm.upper():3s}: Q_m final {f_:+.3f} vs best-AND {a_:+.3f} "
              f"(delta {a_ - f_:+.3f})   purity final {pf_:.3f} vs best-AND {pa_:.3f} "
              f"(delta {pa_ - pf_:+.3f})")

    print()
    print("=" * 92)
    print(f"D. TIME-MATCHED -- every archived champion in the last {LATE} generations,")
    print("   split by the goal live when it was champion (purity and raw Q only)")
    print("=" * 92)
    for s in range(N_SEEDS):
        per_goal, gens, ws, bs, op_at = bests[("mvg", s)]
        g_max = int(gens[-1])
        out = []
        for o in OPS:
            sel = [j for j, g in enumerate(gens)
                   if op_at.get(int(g)) == o and int(g) >= g_max - LATE]
            if not sel:
                continue
            pur = [T.purity_of(ws[j], cfg) for j in sel]
            out.append(f"{o.upper()} purity {np.mean(pur):.4f} (n={len(sel)})")
        print(f"  MVG seed{s}: " + "  |  ".join(out))
    print("  -> if these agree, the goal of the sampled champion does not move purity,")
    print("     and the purity curve is not contaminated by the alternating schedule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
