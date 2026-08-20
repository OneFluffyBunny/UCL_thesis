"""A 3-gate circuit, drawn with every node's side-mixture x and its purity.

The worked example that validates the SHIPPED `qmetrics.circuit_purity` -- it calls
the real implementation, not a private copy, so the picture is a check on the code
and not merely an illustration of it.

Small enough to check by hand. Two inputs -- i0 pinned LEFT (0.0), i1 pinned
RIGHT (1.0) -- and three gates wired as a chain that re-taps both inputs:

    n2 = f(i0, i1)      n3 = f(n2, i0)      n4 = f(n3, i1)   <- program output

Expected: x = 0.500, 0.250, 0.625 and purity = 0.000, 0.500 over the two counted
gates (n4 is the output and is excluded), so circuit purity = 0.250.

Writes a PNG next to this file (gitignored -- `experiments/**/*.png` is). Runs from
any working directory, instantly.
"""
from __future__ import annotations

import pathlib
import sys

import networkx as nx

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))      # repo root: qmetrics

from qmetrics import circuit_purity, open_file

OUT = _HERE.with_name("purity_demo.png")

NAMES = {0: "i0\nLEFT", 1: "i1\nRIGHT", 2: "n2", 3: "n3", 4: "n4\n(output)"}
POS = {0: (0.0, 1.0), 1: (0.0, -1.0), 2: (1.4, 0.75), 3: (2.8, 0.05), 4: (4.2, -0.7)}
EDGES = [(0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (1, 4)]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    G = nx.DiGraph()
    G.add_edges_from(EDGES)
    pinned = {0: 0, 1: 1}
    purity, d = circuit_purity(G, pinned, exclude=[4])
    x, val = d["flow"], d["values"]

    for v in sorted(G):
        p = 2.0 * abs(x[v] - 0.5)
        tag = ("pinned input" if v in pinned
               else "OUTPUT - excluded from the mean" if v == 4 else "gate, counted")
        print(f"  {v}: x = {x[v]:.4f}   purity = {p:.4f}   [{tag}]")
    print(f"  circuit purity = {purity:.4f} over {d['n_gates']} gates")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    cmap = plt.get_cmap("coolwarm")

    for u, v in EDGES:
        ax.add_patch(FancyArrowPatch(POS[u], POS[v], arrowstyle="-|>",
                                     mutation_scale=16, lw=1.6, color="#6b7280",
                                     shrinkA=26, shrinkB=26, zorder=1))

    for v in sorted(G):
        p = 2.0 * abs(x[v] - 0.5)
        counted = v in val
        ax.scatter(*POS[v], s=2700, c=[cmap(x[v])], zorder=2,
                   edgecolors="black" if counted else "#9ca3af",
                   linewidths=2.4 if counted else 1.4,
                   linestyle="solid" if counted else "dashed")
        ax.text(*POS[v], NAMES[v], ha="center", va="center", zorder=3,
                fontsize=9.5, fontweight="bold",
                color="white" if abs(x[v] - 0.5) > 0.28 else "black")
        ax.annotate(f"x = {x[v]:.3f}\npurity = {p:.3f}",
                    POS[v], (0, -40), textcoords="offset points",
                    ha="center", va="top", fontsize=9.5,
                    color="#111827" if counted else "#9ca3af")

    ax.set_title(f"Circuit purity = {purity:.3f}"
                 f"   =  mean(purity of n2, n3)  =  mean(0.000, 0.500)\n"
                 "solid ring = counted   ·   dashed = excluded (inputs, output)"
                 "   ·   colour = x, blue LEFT → red RIGHT",
                 fontsize=11.5, pad=16)
    ax.set_xlim(-0.9, 5.2)
    ax.set_ylim(-2.1, 1.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
    open_file(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
