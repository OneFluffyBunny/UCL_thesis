"""The evolved circuits, all seeds: FG on the top row, MVG on the bottom.

    python fig_fgmvg_circuits.py --root ../runs/fgmvg50

Experiment 4's counterpart of `experiments/analysis/fig_brains.py --grid`.

WHICH CIRCUIT. FG: the final champion. MVG: the champion at the end of the LAST AND
epoch (`fgmvg_common.last_and_epoch_row`), so both rows are drawn on the same goal.

COLOUR is circuit purity's own per-gate reading (`circuit_purity`'s `flow`): the
share of a gate's ancestry that traces back to the LEFT retina half (blue, the
colour of pixels p0-p3) versus the RIGHT half (orange, p4-p7), with grey meaning an
even mix. A pure gate is fully blue or fully orange. The program output is drawn
with the same colour but is excluded from the purity number in the panel title,
as in `circuit_purity` itself.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex

from fgmvg_common import load_study, last_and_epoch_row

import visualize as viz_mod

SIDE_CMAP = LinearSegmentedColormap.from_list(
    "side", [viz_mod.INPUT_L, "#95A5A6", viz_mod.INPUT_R])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: <root>/figures/circuits_fg_vs_mvg.png")
    args = ap.parse_args(argv)

    fg, mvg = load_study(args.root)
    n_cols = max(len(fg.seeds), len(mvg.seeds))
    fig, axes = plt.subplots(2, n_cols, figsize=(4.4 * n_cols, 9.0), squeeze=False)
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
            title = (f"{arm.name} seed {seed}  (gen {row['gen']:,})\n"
                     f"acc(AND) {arm.acc(g, 'and'):.3f} | purity {p:.2f} | "
                     f"{ph.n_active} gates")
            labels = viz_mod._render(ax, g, ph, arm.gates, arm.n_in, title=title,
                                     split=arm.split, scale=0.95,
                                     node_colours=colours)
            fits.append((ax, labels, viz_mod.BOX_HW))
            print(f"{arm.name} seed {seed}: gen {row['gen']} acc(AND) "
                  f"{arm.acc(g, 'and'):.4f} purity {p:.3f} gates {ph.n_active}")

    fig.suptitle("Experiment 4, CGP (50 nodes, gates AND/NAND/OR/NOR): final circuits. "
                 f"Top: fixed goal (L AND R), {fg.alg}. Bottom: MVG (AND <-> OR every "
                 f"{mvg.E} gens), {mvg.alg}, last AND-epoch champion.\n"
                 "Gate colour = ancestry share from the left retina half (blue) vs "
                 "right half (orange); grey = mixed.", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.2, w_pad=0.2)
    viz_mod._fit_labels(fits)
    out = args.out or args.root / "figures" / "circuits_fg_vs_mvg.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
