"""Replay FG seed 0 and MVG seed 0 with PER-GENERATION logging in two windows.

Why a replay and not a re-read: the archived runs_purity/ logs sample every 10
generations while the goal switches every 20, so each phase has exactly 2 points
-- enough to see THAT accuracy drops, not enough to see the shape of the recovery.
The runs are seeded (np.random.default_rng(seed)) and logging draws no randomness,
so replaying seed 0 reproduces the identical evolutionary trajectory; the
verification step below proves it rather than assuming it.

Windows: [1000, 1200] (early) and [10000, 10200] (late), to compare how fast the
population re-converges after a goal switch at two stages of training.

    conda run -n lndp python kashtan_alon/analysis/dense_replay.py --arm fg
    conda run -n lndp python kashtan_alon/analysis/dense_replay.py --arm mvg
    conda run -n lndp python kashtan_alon/analysis/dense_replay.py --verify

Runs from any working directory.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/: train, model, tasks
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

import train as T
import model as M
import tasks

OUT = str(_HERE.parents[1] / "runs_dense")
ARCHIVE = str(_HERE.parents[1] / "runs_purity")
WINDOWS = "100:300,1000:1200,10000:10200"
LAST_GEN = 10200


def preset(mvg):
    """Byte-identical to analysis/fg_mvg_purity.py's paper_preset, plus dense-log.

    Only `generations` differs (we stop just past the second window) and qm_nrand
    (the post-loop Q_m null, which cannot affect the trajectory)."""
    return dict(task="retina", layers="8,8,4,2,1", input_encoding="binary",
                pop=600, generations=LAST_GEN + 1, init_density=0.5,
                n_elite=150, pc=0.5, pm=0.5, fitness="raw",
                switch_interval=20, mvg_ops="and,or",
                weighted_q=False, qm_nrand=100, log_interval=10,
                target=1.0, out_dir=OUT, viz=False, open_img=False,
                resume=False, checkpoint_interval=100000,
                dense_log=WINDOWS, mvg=mvg, operation="and")


def build_args(mvg):
    args = T.build_parser().parse_args([])
    for k, v in preset(mvg).items():
        setattr(args, k, v)
    return args


def run(arm):
    mvg = arm == "mvg"
    args = build_args(mvg)
    cfg = M.NetConfig(layers=tuple(int(x) for x in args.layers.split(",")))
    X_bits = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    os.makedirs(OUT, exist_ok=True)
    T.train_seed(cfg, X_bits, X_bits, args, 0, open_after=False)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def verify():
    """The replay must match the archive on every generation they both logged.

    This is the claim the whole picture rests on: that these dense rows describe
    the SAME run as the published coarse ones."""
    ok = True
    for arm in ("fg", "mvg"):
        name = f"retina_{arm}_raw_seed0_log.csv"
        new = {int(r["gen"]): r for r in rows(os.path.join(OUT, name))}
        old = {int(r["gen"]): r for r in rows(os.path.join(ARCHIVE, name))
               if int(r["gen"]) <= LAST_GEN}
        shared = sorted(set(new) & set(old))
        cols = ["op", "best_fit", "mean_fit", "Q", "purity", "density", "edges"]
        bad = [g for g in shared if any(new[g][c] != old[g][c] for c in cols)]
        extra = len(new) - len(shared)
        print(f"\n{arm.upper()}: {len(shared)} generations logged by both, "
              f"{len(bad)} mismatched, {extra} extra dense rows")
        if bad:
            ok = False
            g = bad[0]
            print(f"  first mismatch at gen {g}:")
            print(f"    archive: {[old[g][c] for c in cols]}")
            print(f"    replay : {[new[g][c] for c in cols]}")
        else:
            print(f"  -> IDENTICAL on all {len(shared)} shared generations "
                  f"(gen {shared[0]}..{shared[-1]})")
    print("\n" + ("VERIFIED: the replay is the same run as the archive."
                  if ok else "FAILED: replay diverged -- do not use this data."))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["fg", "mvg"])
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(verify())
    run(a.arm)
