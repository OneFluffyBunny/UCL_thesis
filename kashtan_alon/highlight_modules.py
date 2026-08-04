"""Make the MODULES in the evolved Kashtan-Alon brains read at a glance.

The per-run PNGs colour nodes by module, but "how modular is it?" is really a
question about EDGES: a modular net has many within-module edges and only a few
between-module edges (the bottleneck where the left-object and right-object
detectors finally meet). This script draws that directly:

  * each Newman-Q module gets a translucent coloured "blob" behind its nodes;
  * within-module edges are drawn in the module colour (solid, faint);
  * BETWEEN-module edges are drawn thick + red -- these are the cross-talk links,
    and a modular network has strikingly few of them.

It renders one comparison sheet, MVG (top row) vs FG (bottom row) across all 5
seeds, so the "MVG looks split, FG looks tangled" contrast is visible in one look.

Run:  conda run -n lndp python kashtan_alon/highlight_modules.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from model import NetConfig
from modularity import newman_q

RUNS = os.path.join(os.path.dirname(__file__), "runs")
MODULE_PALETTE = ["#3498DB", "#E67E22", "#9B59B6", "#1ABC9C", "#F1C40F",
                  "#E91E63", "#8B4513", "#95A5A6"]
CROSS_EDGE = "#E74C3C"       # between-module edge (red = the bottleneck)
INPUT_SPLIT_GAP = 1.2


def load_run(name):
    """-> (weight_mats, cfg, result_dict) for a run stored in runs/."""
    npz = np.load(os.path.join(RUNS, f"{name}_best.npz"))
    n_blocks = sum(1 for k in npz.files if k.startswith("w"))
    weight_mats = [npz[f"w{l}"] for l in range(n_blocks)]
    with open(os.path.join(RUNS, f"{name}_result.json")) as f:
        result = json.load(f)
    return weight_mats, NetConfig(), result


def _positions(cfg):
    pos = {}
    off = cfg.offsets
    for l, n in enumerate(cfg.layers):
        for i in range(n):
            x = i - (n - 1) / 2.0
            if l == 0 and n == 8:
                x += -INPUT_SPLIT_GAP / 2 if i < 4 else INPUT_SPLIT_GAP / 2
            pos[off[l] + i] = (x, float(l))
    return pos


def draw_net(ax, weight_mats, cfg, result, title):
    q, communities = newman_q(weight_mats, cfg)
    node_comm = {n: ci for ci, comm in enumerate(communities) for n in comm}
    off, pos = cfg.offsets, _positions(cfg)

    # --- module blobs: a translucent halo behind every node in a module -------
    for node, (x, y) in pos.items():
        c = MODULE_PALETTE[node_comm.get(node, 0) % len(MODULE_PALETTE)]
        ax.scatter([x], [y], s=1700, marker="o", color=c, alpha=0.16, zorder=0,
                   edgecolors="none")

    # --- edges: within-module faint in module colour, between-module red ------
    n_within = n_between = 0
    for l, W in enumerate(weight_mats):
        W = np.asarray(W)
        for i, j in zip(*np.nonzero(W)):
            s, d = off[l] + i, off[l + 1] + j
            x0, y0 = pos[s]; x1, y1 = pos[d]
            same = node_comm.get(s) == node_comm.get(d)
            if same:
                n_within += 1
                col = MODULE_PALETTE[node_comm.get(s, 0) % len(MODULE_PALETTE)]
                ax.plot([x0, x1], [y0, y1], color=col, lw=1.1, alpha=0.45, zorder=1)
            else:
                n_between += 1
                ax.plot([x0, x1], [y0, y1], color=CROSS_EDGE, lw=2.4, alpha=0.9,
                        zorder=3)

    # --- nodes ----------------------------------------------------------------
    for l, n in enumerate(cfg.layers):
        marker = "s" if l == 0 else "o"
        for i in range(n):
            node = off[l] + i
            x, y = pos[node]
            ax.scatter([x], [y], s=130 if l == 0 else 150, marker=marker, zorder=4,
                       edgecolors="black", linewidths=0.6,
                       color=MODULE_PALETTE[node_comm.get(node, 0) % len(MODULE_PALETTE)])

    q_m = result.get("q_m", float("nan"))
    ax.set_title(f"{title}\nQ_m={q_m:.3f}  |  {len(communities)} modules  |  "
                 f"{n_between} cross / {n_within + n_between} edges", fontsize=9)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-0.7, cfg.n_blocks + 0.4)
    ax.axis("off")


def main():
    seeds = range(5)
    fig, axes = plt.subplots(2, len(list(seeds)), figsize=(4 * 5, 9))
    rows = [("mvg", "MVG"), ("fg", "FG")]
    for r, (tag, label) in enumerate(rows):
        for s in range(5):
            name = f"retina_{tag}_raw_seed{s}"
            wm, cfg, res = load_run(name)
            draw_net(axes[r, s], wm, cfg, res, f"{label} seed{s}")

    fig.suptitle("Evolved retina brains — modules shaded, cross-module edges in RED\n"
                 "(MVG top: fewer red edges, cleaner split  •  FG bottom: more tangled)",
                 fontsize=13, y=0.995)
    handles = [mpatches.Patch(color=CROSS_EDGE, label="between-module edge (bottleneck)"),
               mpatches.Patch(color="#3498DB", alpha=0.4, label="module (shaded blob)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = os.path.join(RUNS, "modules_highlight.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")

    # quick numeric summary of cross-module edge fraction
    print("\nrun            Q_m     cross/total edges")
    for tag, label in rows:
        for s in range(5):
            wm, cfg, res = load_run(f"retina_{tag}_raw_seed{s}")
            q, comms = newman_q(wm, cfg)
            nc = {n: ci for ci, c in enumerate(comms) for n in c}
            off = cfg.offsets
            wi = bt = 0
            for l, W in enumerate(wm):
                for i, j in zip(*np.nonzero(np.asarray(W))):
                    same = nc.get(off[l] + i) == nc.get(off[l + 1] + j)
                    wi += same; bt += (not same)
            print(f"{label} seed{s}   {res['q_m']:+.3f}   {bt}/{wi+bt}"
                  f"  ({100*bt/(wi+bt):.0f}% cross)")


if __name__ == "__main__":
    main()
