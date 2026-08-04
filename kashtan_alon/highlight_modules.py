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


# ---------------------------------------------------------------------------
# Retina-lineage view: KA's actual picture (left detector / right detector /
# integrator), independent of the greedy community algorithm.
# ---------------------------------------------------------------------------
LEFT_COL, RIGHT_COL, MIX_COL, DEAD_COL = "#2E86DE", "#E67E22", "#8E44AD", "#CFCFCF"


def retina_side(weight_mats, cfg, pure=0.8):
    """For every node, the fraction of its input lineage that traces back to the
    LEFT vs RIGHT retina. Retina pixels 0-3 = left, 4-7 = right. A downstream
    neuron inherits the (edge-count-weighted) average side of its live inputs.

    Returns {node: (left_frac, right_frac)} and a classification
    {node: 'L'|'R'|'M'|'-'} (left / right / mixed-integrator / dead)."""
    off = cfg.offsets
    frac = {}
    for i in range(cfg.layers[0]):                 # retina layer
        frac[off[0] + i] = (1.0, 0.0) if i < 4 else (0.0, 1.0)
    for l, W in enumerate(weight_mats):
        W = np.asarray(W)
        for j in range(cfg.layers[l + 1]):
            srcs = np.nonzero(W[:, j])[0]
            if len(srcs) == 0:
                frac[off[l + 1] + j] = (0.0, 0.0)
                continue
            lv = np.mean([frac[off[l] + int(i)][0] for i in srcs])
            rv = np.mean([frac[off[l] + int(i)][1] for i in srcs])
            tot = lv + rv
            frac[off[l + 1] + j] = (lv / tot, rv / tot) if tot > 0 else (0.0, 0.0)
    cls = {}
    for node, (lv, rv) in frac.items():
        if lv + rv == 0:
            cls[node] = "-"
        elif lv >= pure:
            cls[node] = "L"
        elif rv >= pure:
            cls[node] = "R"
        else:
            cls[node] = "M"
    return frac, cls


_SIDE_COL = {"L": LEFT_COL, "R": RIGHT_COL, "M": MIX_COL, "-": DEAD_COL}


def draw_net_lineage(ax, weight_mats, cfg, res, title):
    _, cls = retina_side(weight_mats, cfg)
    off, pos = cfg.offsets, _positions(cfg)
    for l, W in enumerate(weight_mats):
        W = np.asarray(W)
        for i, j in zip(*np.nonzero(W)):
            s, d = off[l] + i, off[l + 1] + j
            x0, y0 = pos[s]; x1, y1 = pos[d]
            ax.plot([x0, x1], [y0, y1], color=_SIDE_COL[cls[s]], lw=1.2,
                    alpha=0.5, zorder=1)
    for l, n in enumerate(cfg.layers):
        marker = "s" if l == 0 else "o"
        for i in range(n):
            node = off[l] + i
            x, y = pos[node]
            ax.scatter([x], [y], s=150 if l == 0 else 170, marker=marker, zorder=3,
                       edgecolors="black", linewidths=0.6, color=_SIDE_COL[cls[node]])
    # purity of the hidden layers (how many non-output neurons stay single-sided)
    hidden = [n for n in range(cfg.layers[0], cfg.n_nodes - 1)]
    pure = sum(cls[n] in ("L", "R") for n in hidden)
    ax.set_title(f"{title}\nQ_m={res['q_m']:+.3f}  |  single-sided neurons "
                 f"{pure}/{len(hidden)}", fontsize=9)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-0.7, cfg.n_blocks + 0.4)
    ax.axis("off")


def _sheet(draw_fn, out_name, suptitle, handles):
    rows = [("mvg", "MVG"), ("fg", "FG")]
    fig, axes = plt.subplots(2, 5, figsize=(4 * 5, 9))
    for r, (tag, label) in enumerate(rows):
        for s in range(5):
            wm, cfg, res = load_run(f"retina_{tag}_raw_seed{s}")
            draw_fn(axes[r, s], wm, cfg, res, f"{label} seed{s}")
    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = os.path.join(RUNS, out_name)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def main():
    rows = [("mvg", "MVG"), ("fg", "FG")]

    # Sheet 1: greedy communities + cross-module edges (the Q_m partition).
    _sheet(draw_net,
           "modules_highlight.png",
           "Evolved retina brains — greedy modules shaded, cross-module edges in RED\n"
           "(this is the partition Q_m scores; MVG top vs FG bottom)",
           [mpatches.Patch(color=CROSS_EDGE, label="between-module edge (bottleneck)"),
            mpatches.Patch(color="#3498DB", alpha=0.4, label="module (shaded blob)")])

    # Sheet 2: KA's actual picture -- left detector / right detector / integrator.
    _sheet(draw_net_lineage,
           "modules_lineage.png",
           "Retina lineage — each neuron coloured by the eye it watches\n"
           "(KA's left-detector / right-detector / integrator; MVG top vs FG bottom)",
           [mpatches.Patch(color=LEFT_COL, label="watches LEFT retina"),
            mpatches.Patch(color=RIGHT_COL, label="watches RIGHT retina"),
            mpatches.Patch(color=MIX_COL, label="integrator (both sides)"),
            mpatches.Patch(color=DEAD_COL, label="no live input")])

    # numeric summary: cross-edge fraction (greedy) + single-sided-neuron count.
    print("\nrun          Q_m     cross/total edges   single-sided hidden neurons")
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
            _, cls = retina_side(wm, cfg)
            hidden = list(range(cfg.layers[0], cfg.n_nodes - 1))
            pure = sum(cls[n] in ("L", "R") for n in hidden)
            print(f"{label} seed{s}  {res['q_m']:+.3f}   {bt}/{wi+bt} "
                  f"({100*bt/(wi+bt):.0f}% cross)      {pure}/{len(hidden)}")


if __name__ == "__main__":
    main()
