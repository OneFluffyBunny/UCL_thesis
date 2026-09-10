"""Big-picture grid of the 10 paper-faithful runs (`runs/`): FG top row, MVG
bottom row, 5 seeds each. Subtitle on each panel gives raw Q, Q_m, the
left/right planted-partition score r, and circuit purity. Reuses
highlight_modules.draw_net_purity for the actual network drawing: nodes are
coloured by qmetrics.circuit_purity's own backward-random-walk value (retina
left=0/blue .. right=1/red, every other node the mean of its parents), ordered
within each layer by that same value. Set SHOW_VALUES=True below to print the
numeric value inside every node (off for the figure; on to verify the colouring).

Usage: conda run -n lndp python kashtan_alon/analysis/paper_grid.py
Runs from any working directory.
"""
from __future__ import annotations

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

from highlight_modules import load_run, draw_net_purity
import qmetrics as qm

OUT = str(_HERE.parents[1] / "runs" / "paper_10runs_grid.png")
PINNED_LR = {i: i // 4 for i in range(8)}          # 0=left retina, 1=right retina
PINNED_PURITY = {i: (0 if i < 4 else 1) for i in range(8)}
SHOW_VALUES = False  # print each node's flow value inside it; off for the figure
NODE_SCALE = 2.0     # marker area multiplier -- nodes are the thing being read


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
    fig, axes = plt.subplots(2, 5, figsize=(4 * 5, 9))
    for r_idx, (tag, label) in enumerate(rows):
        for s in range(5):
            wm, cfg, res = load_run(f"retina_{tag}_raw_seed{s}")
            r, purity = extra_metrics(wm, cfg)
            # the goal is named because it differs by arm: an MVG run ends on
            # whichever of AND/OR the schedule left live, an FG run always on AND
            head = (f"{label} seed {s}: accuracy {res['final_fit']:.3f} "
                    f"({res['final_op'].upper()})")
            subtitle = (f"Q={res['q_real']:.2f}  Q_m={res['q_m']:+.2f}  "
                        f"r={r:+.2f}  purity={purity:.2f}")
            draw_net_purity(axes[r_idx, s], wm, cfg, res, head,
                             subtitle=subtitle, show_values=SHOW_VALUES,
                             node_scale=NODE_SCALE)
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
