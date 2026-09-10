"""Big-picture grid of the 10 paper-faithful runs: FG top row, MVG bottom row,
5 seeds each. Subtitle on each panel gives raw Q, Q_m, the
left/right planted-partition score r, and circuit purity. Reuses
highlight_modules.draw_net_purity for the actual network drawing: nodes are
coloured by qmetrics.circuit_purity's own backward-random-walk value (retina
left=0/blue .. right=1/red, every other node the mean of its parents), ordered
within each layer by that same value. Set SHOW_VALUES=True below to print the
numeric value inside every node (off for the figure; on to verify the colouring).

EVERY PANEL IS THE SAME GOAL: the last champion archived during an AND epoch,
generation 24,970. Using `<run>_best.npz` instead would
draw the FINAL-generation champion, and generation 24,999 is mid-OR-epoch for
every MVG seed (the schedule is deterministic), so all five MVG panels would show
an OR specialist -- one that scores 0.50 on AND, below the 0.75 a constant output
gets. Q, Q_m, r and purity are recomputed for the brain actually drawn rather
than read from result.json, which describes the final-generation champion.

Needs `<run>_brains.npz`, so it reads runs_purity/ (runs/ predates archiving).
They are the same 10 runs: identical seeds, identical parameters, identical logs.

Usage: conda run -n lndp python kashtan_alon/analysis/paper_grid.py
Runs from any working directory.
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/: highlight_modules
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

from highlight_modules import draw_net_purity
from modularity import normalized_qm
from model import NetConfig
import model as M
import tasks
import train as T
import qmetrics as qm

RUNS_DIR = str(_HERE.parents[1] / "runs_purity")
OUT = str(_HERE.parents[1] / "runs" / "paper_10runs_grid.png")
GOAL = "and"         # the single goal every panel is scored on
PINNED_LR = {i: i // 4 for i in range(8)}          # 0=left retina, 1=right retina
PINNED_PURITY = {i: (0 if i < 4 else 1) for i in range(8)}
SHOW_VALUES = False  # print each node's flow value inside it; off for the figure
NODE_SCALE = 2.0     # marker area multiplier -- nodes are the thing being read
QM_NRAND = 1000      # null-model size for Q_m, matching the runs' own setting


def last_on_goal(name, X, y):
    """-> (weights, biases, accuracy, gen) for the LAST champion archived while
    GOAL was the live goal.

    Last, not best: a maximum over ~1250 archived champions is a cherry-pick, and
    it lands at a different generation in every run (FG seed2's best AND champion
    is at generation 970). The last one keeps every panel at the end of its run,
    the way a final-generation champion would be, and costs nothing -- every seed's
    late AND champion ties its own best.

    For MVG this is generation 24,970; 24,980-24,999 are one OR epoch."""
    with open(os.path.join(RUNS_DIR, f"{name}_log.csv"), newline="") as f:
        op_at = {int(r["gen"]): r["op"] for r in csv.DictReader(f)}
    gens, ws, bs = T.BrainArchive.load(os.path.join(RUNS_DIR, f"{name}_brains.npz"))
    cfg = NetConfig()
    for i in range(len(gens) - 1, -1, -1):
        if op_at.get(int(gens[i])) != GOAL:
            continue
        w = [np.asarray(m)[None] for m in ws[i]]
        b = [np.asarray(v)[None] for v in bs[i]]
        return ws[i], bs[i], float(M.fitness(w, b, X, y, cfg, "raw")[0]), int(gens[i])
    raise SystemExit(f"{name}: no champion archived during a {GOAL.upper()} epoch")


def extra_metrics(wm, cfg):
    """(r, purity) for one champion, via qmetrics.left_right_q / circuit_purity."""
    blocks = [np.asarray(w, dtype=float) for w in wm]
    out_node = cfg.offsets[-1]

    Gu = qm.from_blocks(blocks, offsets=cfg.offsets, directed=False)
    r = float("nan")
    if Gu.number_of_edges() > 0:
        _, info = qm.left_right_q(Gu, PINNED_LR, exclude=[out_node], n_rand=200, seed=0)
        r = info["r"]

    Gd = qm.from_blocks(blocks, offsets=cfg.offsets, directed=True)
    purity = float("nan")
    if Gd.number_of_edges() > 0:
        purity, _ = qm.circuit_purity(Gd, PINNED_PURITY, exclude=[out_node])
    return r, purity


def main():
    rows = [("fg", "FG"), ("mvg", "MVG")]
    cfg = NetConfig()
    X = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    y = np.asarray(tasks.targets("retina", GOAL, X))

    fig, axes = plt.subplots(2, 5, figsize=(4 * 5, 9))
    for r_idx, (tag, label) in enumerate(rows):
        for s in range(5):
            name = f"retina_{tag}_raw_seed{s}"
            wm, bm, acc, gen = last_on_goal(name, X, y)
            r, purity = extra_metrics(wm, cfg)
            q_m, parts = normalized_qm(wm, cfg, n_rand=QM_NRAND, seed=s)
            # 2 dp everywhere: figure captions are read, not recomputed from
            head = f"{label} seed {s}: accuracy {acc:.2f} ({GOAL.upper()})"
            subtitle = (f"Q={parts['q_real']:.2f}  Q_m={q_m:+.2f}  "
                        f"r={r:+.2f}  purity={purity:.2f}")
            draw_net_purity(axes[r_idx, s], wm, cfg, {}, head,
                             subtitle=subtitle, show_values=SHOW_VALUES,
                             node_scale=NODE_SCALE)
            print(f"{name}: last {GOAL.upper()}-epoch champion, gen {gen}, acc {acc:.4f} "
                  f"| Q={parts['q_real']:.3f} Q_m={q_m:+.3f} "
                  f"r={r:+.3f} purity={purity:.3f}")
    fig.suptitle("Kashtan-Alon paper-faithful repoduction", fontsize=15, y=0.995)
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))

    # Colour is continuous (a node is the mean of its parents), so a gradient bar
    # reads it far more directly than discrete swatches at the two extremes.
    cax = fig.add_axes((0.42, 0.035, 0.16, 0.016))
    cb = matplotlib.colorbar.ColorbarBase(
        cax, cmap=plt.cm.bwr, norm=matplotlib.colors.Normalize(0.0, 1.0),
        orientation="horizontal")
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["LEFT", "RIGHT"])
    cb.ax.tick_params(length=0, labelsize=12)
    cb.outline.set_linewidth(0.6)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
