"""One capped MVG brain, drawn with the OLD Newman-Q community colouring.

The argument for the planted-bipartition metrics (`left_right_q`'s r, and
circuit purity) in one picture. MVG seed 1's last AND-epoch champion is a
LITERAL two-module network -- every hidden neuron's live ancestry traces to one
retina side only, which is why purity and r are both exactly 1.00 (audited
node-by-node in scratch_purity_audit.py). Greedy Newman Q, handed the same
graph, reports FOUR communities: it cuts the left module in two and makes the
integrator spine a "module" of its own. Its Q_m (+0.26) is also LOWER than seed
0's (+0.46), whose purity is only 0.81 -- the metric ranks the perfectly split
brain below the imperfect one.

So the case for the binary metrics is not "Q is wrong". It is that Q has to
SEARCH for a partition, the search is NP-hard and budget-limited, and nothing
makes the partition it finds the one the task is about. Handing Q the task's own
left/right split removes the search entirely.

Same brain-selection rule as analysis/paper_grid.py: the last champion archived
during an AND epoch, so this is goal-matched to every other figure.

Usage: conda run -n lndp python kashtan_alon/analysis/newman_vs_binary.py
       ... --seed 3          # another MVG seed (0-4)
Runs from any working directory.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

from highlight_modules import (draw_net, purity_and_flow, _positions_by_value,
                               MODULE_PALETTE, CROSS_EDGE)
from modularity import newman_q, normalized_qm, n_edges
from model import NetConfig
import model as M
import tasks
import train as T
import qmetrics as qm

RUNS = str(_HERE.parents[1] / "runs_purity")
OUT_DIR = str(_HERE.parents[1] / "runs_purity")
GOAL = "and"
QM_NRAND = 1000
PINNED_LR = {i: i // 4 for i in range(8)}
PINNED_PURITY = {i: (0 if i < 4 else 1) for i in range(8)}


def last_on_goal(name, X, y, cfg):
    """The last champion archived while GOAL was live -- paper_grid.py's rule."""
    with open(os.path.join(RUNS, f"{name}_log.csv"), newline="") as f:
        op_at = {int(r["gen"]): r["op"] for r in csv.DictReader(f)}
    gens, ws, bs = T.BrainArchive.load(os.path.join(RUNS, f"{name}_brains.npz"))
    for i in range(len(gens) - 1, -1, -1):
        if op_at.get(int(gens[i])) != GOAL:
            continue
        w = [np.asarray(m)[None] for m in ws[i]]
        b = [np.asarray(v)[None] for v in bs[i]]
        return ws[i], float(M.fitness(w, b, X, y, cfg, "raw")[0]), int(gens[i])
    raise SystemExit(f"{name}: no champion archived during a {GOAL.upper()} epoch")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1, help="which MVG seed to draw")
    cli = ap.parse_args()

    cfg = NetConfig()
    X = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    y = np.asarray(tasks.targets("retina", GOAL, X))
    name = f"retina_mvg_raw_seed{cli.seed}"
    wm, acc, gen = last_on_goal(name, X, y, cfg)

    q, comms = newman_q(wm, cfg)
    q_m, parts = normalized_qm(wm, cfg, n_rand=QM_NRAND, seed=cli.seed)
    blocks = [np.asarray(w, dtype=float) for w in wm]
    out_node = cfg.offsets[-1]
    Gd = qm.from_blocks(blocks, offsets=cfg.offsets, directed=True)
    purity, _ = qm.circuit_purity(Gd, PINNED_PURITY, exclude=[out_node])
    Gu = qm.from_blocks(blocks, offsets=cfg.offsets, directed=False)
    _, info = qm.left_right_q(Gu, PINNED_LR, exclude=[out_node], n_rand=200, seed=0)

    # cross-community edge count, the same quantity draw_net draws in red
    node_comm = {n: ci for ci, c in enumerate(comms) for n in c}
    off = cfg.offsets
    cross = sum(1 for l, W in enumerate(wm)
                for i, j in zip(*np.nonzero(np.asarray(W)))
                if node_comm.get(off[l] + i) != node_comm.get(off[l + 1] + j))

    # colour by Newman-Q community, but lay the nodes out by SIDE (left-reading
    # neurons left, right-reading right). With the default index order both the
    # true split and the communities look equally tangled and the figure proves
    # nothing; ordering by side shows a network that IS two modules, coloured by
    # a partition that is not those two modules.
    _, flow = purity_and_flow(wm, cfg)
    off = cfg.offsets

    # Where do the two halves actually meet? The first block holding an edge
    # between a left-reading and a right-reading neuron. Measured, not assumed,
    # so this stays honest on a seed whose split is not perfect.
    def side(n):
        v = flow.get(n)
        return None if v is None or v == 0.5 else int(v > 0.5)
    first_cross = len(wm)
    for l, W in enumerate(wm):
        if any(side(off[l] + i) is not None and side(off[l + 1] + j) is not None
               and side(off[l] + i) != side(off[l + 1] + j)
               for i, j in zip(*np.nonzero(np.asarray(W)))):
            first_cross = l
            break
    meet = ("the two halves meet only at the output neuron"
            if first_cross >= len(wm) - 1
            else f"the two halves first meet in layer {first_cross + 1}")

    fig, ax = plt.subplots(figsize=(8.0, 8.4))
    # Deliberately bare: one title, the legend, and the two retina-half labels.
    # Every number this figure is about (Q, Q_m, purity, r, cross edges, where the
    # halves meet) is printed to stdout below and goes into the caption prose
    # instead -- an earlier version put it all in the title and was unreadable.
    draw_net(ax, wm, cfg, {}, "", subtitle="",
             pos=_positions_by_value(cfg, flow))
    ax.set_title("Newman Q decomposition", fontsize=14)

    # the retina's two halves are already drawn with a gap; name them, because the
    # whole point is whether the communities line up with THIS split
    ax.text(-2.6, -0.55, "LEFT retina", ha="center", fontsize=11, color="#374151")
    ax.text(+2.6, -0.55, "RIGHT retina", ha="center", fontsize=11, color="#374151")
    top = first_cross - 0.2
    lo, hi = ax.get_ylim()
    ax.axvline(0.0, ymin=(-0.7 - lo) / (hi - lo), ymax=(top - lo) / (hi - lo),
               color="#374151", lw=1.0, ls=":")

    handles = [mpatches.Patch(color=MODULE_PALETTE[i], alpha=0.45,
                              label=f"Newman-Q community {i + 1}")
               for i in range(len(comms))]
    handles.append(mpatches.Patch(color=CROSS_EDGE,
                                 label="edge BETWEEN two communities"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"newman_communities_mvg_seed{cli.seed}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"{name}: gen {gen} acc {acc:.4f} Q={q:.3f} Q_m={q_m:+.3f} "
          f"(q_rand={parts['q_rand']:.3f} q_max={parts['q_max']:.3f}) "
          f"purity={purity:.3f} r={info['r']:+.3f} "
          f"{len(comms)} communities, {cross}/{n_edges(wm)} cross edges")
    for ci, c in enumerate(comms):
        print(f"  community {ci + 1}: {sorted(c)}")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
