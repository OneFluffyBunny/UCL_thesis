"""Draw an evolved Kashtan-Alon network in the style of the paper's figures.

Vertical layout: the 8-pixel retina sits at the BOTTOM (left 4 | right 4, with a
gap marking the split), hidden layers stack upward, the single output is at the
TOP -- the same orientation as the network diagrams in Kashtan & Alon (2005).

Every node is coloured by the module (Newman-Q community) it was assigned to, so a
MODULAR solution shows up immediately: the left retina + its hidden neurons take
one colour, the right retina + its neurons another, meeting only near the output.
Edges are green (excitatory, +) / slate (inhibitory, -), width by |weight|.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from modularity import newman_q

EXC_EDGE = "#2ECC71"   # green (+)
INH_EDGE = "#34495E"   # slate (-)
MODULE_PALETTE = ["#3498DB", "#E67E22", "#9B59B6", "#E74C3C", "#1ABC9C",
                  "#F1C40F", "#E91E63", "#95A5A6", "#8B4513", "#2ECC71"]
INPUT_SPLIT_GAP = 1.2   # extra x-gap between the left-4 and right-4 retina pixels


def _auto_open(path):
    import os, sys, subprocess
    ap = os.path.abspath(path)
    try:
        if sys.platform == "win32":
            os.startfile(ap)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", ap])
        else:
            subprocess.run(["xdg-open", ap])
    except Exception as e:
        print(f"  (could not auto-open image: {e})")


def _positions(cfg):
    """node id -> (x, y). Layer = y row (inputs at y=0 bottom, output on top);
    unit = x, each layer centred. The input retina is split left-4 | right-4."""
    pos = {}
    off = cfg.offsets
    for l, n in enumerate(cfg.layers):
        for i in range(n):
            x = i - (n - 1) / 2.0
            if l == 0 and n == 8:                       # retina: open a left|right gap
                x += -INPUT_SPLIT_GAP / 2 if i < 4 else INPUT_SPLIT_GAP / 2
            pos[off[l] + i] = (x, float(l))
    return pos


def visualize_net(individual, cfg, save_path, title="", weighted_q=False,
                  open_after=False, q_m=None):
    weight_mats, _ = individual
    q, communities = newman_q(weight_mats, cfg, weighted=weighted_q)

    node_comm = {node: ci for ci, comm in enumerate(communities) for node in comm}
    off = cfg.offsets
    pos = _positions(cfg)

    fig, ax = plt.subplots(figsize=(9, 8))

    # edges (directed by construction: layer l -> l+1)
    absmax = max((abs(int(w)) for W in weight_mats for w in np.asarray(W).ravel() if w != 0), default=1)
    for l, W in enumerate(weight_mats):
        W = np.asarray(W)
        for i, j in zip(*np.nonzero(W)):
            w = float(W[i, j])
            x0, y0 = pos[off[l] + i]
            x1, y1 = pos[off[l + 1] + j]
            ax.plot([x0, x1], [y0, y1], color=(EXC_EDGE if w > 0 else INH_EDGE),
                    lw=0.6 + 2.6 * abs(w) / absmax, alpha=0.6, zorder=1)

    # nodes: retina pixels as squares, neurons as circles; fill = module colour
    for l, n in enumerate(cfg.layers):
        marker = "s" if l == 0 else "o"
        size = 300 if l == 0 else 340
        for i in range(n):
            node = off[l] + i
            x, y = pos[node]
            ax.scatter([x], [y], s=size, marker=marker, zorder=2,
                       edgecolors="black", linewidths=0.6,
                       color=MODULE_PALETTE[node_comm.get(node, 0) % len(MODULE_PALETTE)])
            if l == 0:
                ax.text(x, y - 0.42, f"p{i}", ha="center", va="top", fontsize=7, color="#555")

    # left / right retina labels + row labels
    ax.text(-(2 + INPUT_SPLIT_GAP / 2), -0.95, "LEFT retina", ha="center", fontsize=8,
            style="italic", color="#555")
    ax.text((2 + INPUT_SPLIT_GAP / 2), -0.95, "RIGHT retina", ha="center", fontsize=8,
            style="italic", color="#555")
    names = ["retina (in)"] + [f"hidden {l}" for l in range(1, cfg.n_blocks)] + ["output"]
    xr = max(cfg.layers) / 2.0 + INPUT_SPLIT_GAP / 2 + 0.8
    for l, name in enumerate(names):
        ax.text(xr, float(l), name, ha="left", va="center", fontsize=8, color="#777")

    handles = [mpatches.Patch(color=MODULE_PALETTE[c % len(MODULE_PALETTE)],
                              label=f"module {c}  (n={len(communities[c])})")
               for c in range(len(communities))]
    handles += [mpatches.Patch(color=EXC_EDGE, label="excitatory (+)"),
                mpatches.Patch(color=INH_EDGE, label="inhibitory (-)")]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9,
              bbox_to_anchor=(-0.02, 1.0))

    # Headline is KA's normalized Q_m (if provided); the raw Newman Q is the score of
    # the greedy partition the node COLOURS come from (always positive, density-
    # confounded -- not the metric we compare MVG vs FG on).
    head = f"Q_m = {q_m:.3f}   |   " if q_m is not None else ""
    ax.set_title(f"{title}\n{head}raw Newman Q (colour partition) = {q:.3f}   |   "
                 f"modules = {len(communities)}", fontsize=11)
    ax.set_xlim(-(max(cfg.layers) / 2.0 + INPUT_SPLIT_GAP / 2 + 1.5), xr + 2.2)
    ax.set_ylim(-1.4, cfg.n_blocks + 0.4)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved -> {save_path}")
    if open_after:
        _auto_open(save_path)
    return q
