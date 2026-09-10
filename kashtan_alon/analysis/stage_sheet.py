"""Stages of evolution for ONE Kashtan-Alon run: a 2x3 sheet of the same brain
at six generations, coloured by which eye each neuron watches.

The experiment_4 equivalent (`experiments/experiment_4/visualize.py:stage_grid`)
does this for CGP circuits; this is the layered-network version, and it reads the
per-generation champions that `train.py` archives to `<run>_brains.npz`. Without
that archive there is nothing to draw -- a run predating it has only its final net.

Panels are spaced GEOMETRICALLY by default, not evenly: on this task almost all of
the structural change happens in the first couple of thousand generations, and six
evenly-spaced panels spend five of them showing the same converged brain.

    conda run -n lndp python kashtan_alon/analysis/stage_sheet.py --run retina_mvg_raw_seed0
    ... --purity            # add circuit purity to every panel caption
    ... --style modules     # greedy Newman-Q communities instead of the lineage view
    ... --gens 0,50,200,1000,5000,24990

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
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/: train, model, ...
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import highlight_modules as hm
import train as T
from model import NetConfig
from qmetrics import open_file

ROWS, COLS = 2, 3
N_PATTERNS = 256
# Fractions of the run, geometric-ish: the interesting structure is all up front.
DEFAULT_FRACTIONS = (0.0, 0.002, 0.01, 0.05, 0.25, 1.0)

STYLES = {
    "lineage": (hm.draw_net_lineage,
                [mpatches.Patch(color=hm.LEFT_COL, label="watches LEFT retina"),
                 mpatches.Patch(color=hm.RIGHT_COL, label="watches RIGHT retina"),
                 mpatches.Patch(color=hm.MIX_COL, label="integrator (both sides)"),
                 mpatches.Patch(color=hm.DEAD_COL, label="no live input")]),
    "modules": (hm.draw_net,
                [mpatches.Patch(color=hm.CROSS_EDGE,
                                label="between-module edge (bottleneck)"),
                 mpatches.Patch(color="#3498DB", alpha=0.4,
                                label="module (shaded blob)")]),
}


def read_log(runs_dir, run):
    """-> {gen: row} from the per-generation CSV, for captions."""
    path = os.path.join(runs_dir, f"{run}_log.csv")
    with open(path, newline="") as f:
        return {int(r["gen"]): r for r in csv.DictReader(f)}


def pick_stages(gens, fractions, explicit=None):
    """-> the archive indices to draw, nearest to each requested generation."""
    if explicit:
        want = [int(g) for g in explicit.split(",")]
    else:
        want = [f * gens[-1] for f in fractions]
    idx = sorted({int(np.abs(np.asarray(gens) - w).argmin()) for w in want})
    return idx


def caption(row, gen, with_purity, cfg, wm):
    """`gen | goal | accuracy` and, with --purity, the circuit-purity number."""
    acc = float(row["best_fit"]) if row else float("nan")
    head = f"gen {gen}"
    if row and row.get("op"):
        head += f"  |  goal {row['op'].upper()}"
    head += f"  |  accuracy {100 * acc:.2f}% ({int(round(acc * N_PATTERNS))}/{N_PATTERNS})"
    if not with_purity:
        return head
    pur = float(row["purity"]) if row and "purity" in row else T.purity_of(wm, cfg)
    return f"{head}\ncircuit purity {pur:.3f}"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", default="retina_mvg_raw_seed0",
                    help="run name, e.g. retina_mvg_raw_seed0 / retina_fg_raw_seed2")
    ap.add_argument("--runs-dir", default=str(_HERE.parents[1] / "runs_purity"))
    ap.add_argument("--style", choices=sorted(STYLES), default="lineage")
    ap.add_argument("--purity", action="store_true",
                    help="add circuit purity to every panel caption")
    ap.add_argument("--gens", default="", help="explicit comma-separated generations")
    ap.add_argument("--out", default="", help="output PNG (default: next to the run)")
    ap.add_argument("--no-open", dest="open", action="store_false", default=True)
    return ap


def main():
    cli = build_parser().parse_args()
    run, runs_dir = cli.run, cli.runs_dir
    brains_path = os.path.join(runs_dir, f"{run}_brains.npz")
    if not os.path.exists(brains_path):
        sys.exit(f"no brain archive at {brains_path} -- this run predates archiving, "
                 f"or the name is wrong")

    gens, ws, _ = T.BrainArchive.load(brains_path)
    log = read_log(runs_dir, run)
    cfg = NetConfig()
    draw_fn, handles = STYLES[cli.style]
    idx = pick_stages(gens, DEFAULT_FRACTIONS, cli.gens)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 4.6, ROWS * 4.6))
    for ax, i in zip(axes.flat, list(idx) + [None] * (ROWS * COLS)):
        ax.axis("off")
        if i is None:
            continue
        gen, wm = int(gens[i]), ws[i]
        sub = caption(log.get(gen), gen, cli.purity, cfg, wm)
        draw_fn(ax, wm, cfg, {"q_m": float("nan")}, "", subtitle=sub)

    arm = "MVG (AND <-> OR every 20 gens)" if "_mvg_" in run else "FG (L AND R)"
    style_line = ("each neuron coloured by the eye it watches (KA's left-detector / "
                  "right-detector / integrator)" if cli.style == "lineage" else
                  "greedy Newman-Q modules shaded, cross-module edges in RED")
    fig.suptitle(f"Stages of evolution — {arm} — {run}\n{style_line}", fontsize=13)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out = cli.out or os.path.join(runs_dir, f"{run}_stages_{cli.style}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"stages at generations {[int(gens[i]) for i in idx]}")
    print(f"wrote {out}")
    if cli.open:
        open_file(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
