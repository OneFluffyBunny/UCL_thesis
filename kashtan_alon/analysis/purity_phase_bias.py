"""Is the MVG purity curve an artefact of WHEN inside a goal phase we sample it?

The worry is concrete. Goals switch every `switch_interval` (20) generations and
the CSV logs every `log_interval` (10), so the logged generations land on only two
phase offsets: 0 (the switch generation itself, where the champion is picked under
a goal the population has not adapted to yet) and 10 (exactly mid-phase). Offsets
1-9 and 11-19 are never sampled. If purity varies within a phase, the trajectory is
measuring the sampling grid as much as the evolution.

Two independent checks:

  A. From the archived brains of the finished MVG runs -- purity at offset 0 vs
     offset 10, over every logged generation of every seed. This is free.
  B. A fresh short MVG run logged EVERY generation, giving the full within-phase
     profile at offsets 0..19. This is the check that can actually see a sawtooth.

    conda run -n lndp python kashtan_alon/analysis/purity_phase_bias.py
    ... --fine-gens 6000     # longer fine-grained run
    ... --no-fine            # part A only

Runs from any working directory. Part B costs ~1-2 minutes at the default length.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

import train as T
import model as M
import tasks

SWITCH = 20


def part_a(runs_dir, n_seeds, log_interval):
    cfg = M.NetConfig()
    print(f"A. Archived MVG runs in {os.path.basename(runs_dir)} "
          f"(logged every {log_interval} gens, phase = {SWITCH} gens)")
    by_off = {}
    for s in range(n_seeds):
        path = os.path.join(runs_dir, f"retina_mvg_raw_seed{s}_brains.npz")
        if not os.path.exists(path):
            continue
        gens, ws, _ = T.BrainArchive.load(path)
        for g, wm in zip(gens, ws):
            by_off.setdefault(int(g) % SWITCH, []).append(T.purity_of(wm, cfg))
    if not by_off:
        print("   no archives found -- run fg_mvg_purity.py first\n")
        return
    print(f"   {'offset':>7}{'n':>7}{'mean purity':>14}{'sd':>8}")
    for off in sorted(by_off):
        v = np.array(by_off[off])
        print(f"   {off:>7}{len(v):>7}{v.mean():>14.4f}{v.std(ddof=1):>8.4f}")
    offs = sorted(by_off)
    if len(offs) == 2:
        a, b = (np.array(by_off[o]) for o in offs)
        d = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"   offset {offs[0]} minus offset {offs[1]}: {d:+.4f} "
              f"(SE {se:.4f}, {abs(d) / se:.1f} sigma)")
    print()


def part_b(runs_dir, gens_n, seed):
    """A fresh MVG run logged every generation -> purity by offset 0..19."""
    out = os.path.join(runs_dir, "_phase_probe")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    args = T.build_parser().parse_args([])
    for k, v in dict(task="retina", layers="8,8,4,2,1", input_encoding="binary",
                     pop=600, generations=gens_n, init_density=0.5, n_elite=150,
                     pc=0.5, pm=0.5, fitness="raw", mvg=True, operation="and",
                     switch_interval=SWITCH, mvg_ops="and,or", weighted_q=False,
                     qm_nrand=20, log_interval=1, target=1.0, out_dir=out,
                     viz=False, open_img=False, resume=False,
                     checkpoint_interval=0).items():
        setattr(args, k, v)
    cfg = M.NetConfig()
    X_bits = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    print(f"B. Fresh MVG run, {gens_n} gens, logged EVERY generation (seed {seed})")
    T.train_seed(cfg, X_bits, X_bits, args, seed, open_after=False)

    gens, ws, _ = T.BrainArchive.load(
        os.path.join(out, f"retina_mvg_raw_seed{seed}_brains.npz"))
    pur = np.array([T.purity_of(w, cfg) for w in ws])
    off = np.array([int(g) % SWITCH for g in gens])
    goal_new = (np.array([int(g) // SWITCH for g in gens]) % 2)   # 0 = AND, 1 = OR

    print(f"\n   purity by generations-since-goal-switch (offset), {len(pur)} gens:")
    print(f"   {'offset':>7}{'n':>6}{'mean':>9}{'sd':>8}   "
          f"{'(AND phases)':>14}{'(OR phases)':>13}")
    for o in range(SWITCH):
        m = off == o
        a, b = pur[m & (goal_new == 0)], pur[m & (goal_new == 1)]
        print(f"   {o:>7}{m.sum():>6}{pur[m].mean():>9.4f}{pur[m].std(ddof=1):>8.4f}"
              f"{a.mean():>14.4f}{b.mean():>13.4f}")
    sampled = np.isin(off, [0, 10])
    print(f"\n   sampled offsets (0,10) mean {pur[sampled].mean():.4f}   "
          f"unsampled offsets mean {pur[~sampled].mean():.4f}   "
          f"bias {pur[sampled].mean() - pur[~sampled].mean():+.4f}")
    print(f"   worst-case spread across offsets: "
          f"{max(pur[off == o].mean() for o in range(SWITCH)) - min(pur[off == o].mean() for o in range(SWITCH)):.4f}")
    shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-dir", default=str(_HERE.parents[1] / "runs_purity"))
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--fine-gens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fine", dest="fine", action="store_false", default=True)
    cli = ap.parse_args()
    part_a(cli.runs_dir, cli.n_seeds, cli.log_interval)
    if cli.fine:
        part_b(cli.runs_dir, cli.fine_gens, cli.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
