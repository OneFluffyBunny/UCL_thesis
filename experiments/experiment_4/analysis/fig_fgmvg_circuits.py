"""The evolved circuits, all seeds: FG on the top row, MVG on the bottom.

    python fig_fgmvg_circuits.py --root ../runs/study_ga_E2000

Experiment 4's counterpart of `experiments/analysis/fig_brains.py --grid`.

WHICH CIRCUIT. FG: the final champion. MVG: the champion at the end of the LAST AND
epoch (`fgmvg_common.last_and_epoch_row`), so both rows are drawn on the same goal.

COLOUR is circuit purity's own per-gate reading (`circuit_purity`'s `flow`): the
share of a gate's ancestry that traces back to the LEFT retina half (blue, pixels
p0-p3) versus the RIGHT half (red, p4-p7), with white meaning an even mix -- the
`bwr` scale and LEFT/RIGHT colour bar of `kashtan_alon/analysis/paper_grid.py`, so
the two studies read the same. A pure gate is fully blue or fully red. The program
output is coloured the same way but is excluded from the purity number in the
panel title, as in `circuit_purity` itself.

WIRES leave the right side of their source box and enter the left side of the gate
they feed (`visualize._render(side_arrows=True)`).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.colorbar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex

from fgmvg_common import load_study, last_and_epoch_row

import visualize as viz_mod

SIDE_CMAP = plt.cm.bwr          # as in the Kashtan-Alon purity figures
X_GAP = 1.55                    # column spacing: room for side-to-side wires


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: <root>/figures/circuits_fg_vs_mvg.png")
    args = ap.parse_args(argv)

    fg, mvg = load_study(args.root)
    n_cols = max(len(fg.seeds), len(mvg.seeds))
    fig, axes = plt.subplots(2, n_cols, figsize=(5.6 * n_cols, 9.4), squeeze=False)
    in_cols = (to_hex(SIDE_CMAP(0.0)), to_hex(SIDE_CMAP(1.0)))
    fits = []
    for r, arm in enumerate((fg, mvg)):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(arm.seeds):
                ax.axis("off")
                continue
            seed = arm.seeds[c]
            row = last_and_epoch_row(arm, arm.archive(seed))
            g = arm.genotype(row)
            p, d, ph = arm.purity(g)
            colours = {j: to_hex(SIDE_CMAP(d["flow"][arm.n_in + j]))
                       for j in ph.active if arm.n_in + j in d["flow"]}
            title = (f"{arm.name} seed {seed}\n"
                     f"acc {arm.acc(g, 'and'):.3f} | purity {p:.2f} | {ph.n_active} gates")
            labels = viz_mod._render(ax, g, ph, arm.gates, arm.n_in, title=title,
                                     split=arm.split, scale=0.95, node_colours=colours,
                                     input_colours=in_cols, side_arrows=True, x_gap=X_GAP)
            fits.append((ax, labels, viz_mod.BOX_HW))
            print(f"{arm.name} seed {seed}: gen {row['gen']} acc(AND) "
                  f"{arm.acc(g, 'and'):.4f} purity {p:.3f} gates {ph.n_active}")

    fig.suptitle("CGP: FG (top) vs MVG (bottom) final circuits", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965), h_pad=0.4, w_pad=0.2)
    viz_mod._fit_labels(fits)
    # Colour is continuous (a gate is the mean of its parents), as in the KA grid.
    cax = fig.add_axes((0.42, 0.03, 0.16, 0.014))
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=SIDE_CMAP, norm=Normalize(0.0, 1.0),
                                          orientation="horizontal")
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["LEFT", "RIGHT"])
    cb.ax.tick_params(length=0, labelsize=12)
    cb.outline.set_linewidth(0.6)
    out = args.out or args.root / "figures" / "circuits_fg_vs_mvg.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
