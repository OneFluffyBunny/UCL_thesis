"""The CGP x ECGP  by  FG x MVG quadrant, across evolution.

The kashtan_alon analogue (`kashtan_alon/analysis/fg_mvg_purity.py`) plots left/right
circuit purity against accuracy for MVG vs FG. This does the same for experiment_4's
boolean circuits, with two extra curves for the encoding, and one substitution:

  ** experiment_4 does not log circuit purity, and archives no per-generation
     genomes, so purity CANNOT be recomputed after the fact. What IS logged every
     interval is the cone classification of every active gate -- left / right /
     mixed -- so this plots SIDEDNESS = (left + right) / (left + right + mixed):
     the fraction of live gates whose input cone touches only one retina. **

That is the hard-threshold ancestor of circuit purity (a gate reading 7 left pixels
and 1 right one counts as fully `mixed` here, but scores ~0.75 purity), so the two
are not interchangeable -- sidedness is the stricter, noisier one. Read this figure
as the cone metric's answer to the same question, not as purity.

    conda run -n lndp python experiments/experiment_4/analysis/fg_mvg_quadrant.py
    ... --nodes 400          # the other node count (MVG only -- FG cell missing)
    ... --no-open

Runs from any working directory. Reads only; runs no evolution.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # experiment_4/
sys.path.insert(0, str(_HERE.parents[3]))      # repo root: qmetrics

from qmetrics import open_file

# (label, run-dir glob, colour, linestyle) -- colour = goal regime, dash = encoding.
CELLS = [
    ("CGP  · FG  (L AND R fixed)", "_purity_fg/cgp_*_n{n}_*", "#2563eb", "-"),
    ("ECGP · FG  (L AND R fixed)", "_purity_fg/ecgp_*_n{n}_*", "#2563eb", "--"),
    ("CGP  · MVG (AND<->OR/2000)", "_purity_mvg/cgp_*_n{n}_*", "#dc2626", "-"),
    ("ECGP · MVG (AND<->OR/2000)", "_purity_mvg/ecgp_*_n{n}_*", "#dc2626", "--"),
]

PANELS = [("sided", "fraction of active gates with a single-retina cone",
           "Structural sidedness  (left+right)/(left+right+mixed)", (0.0, 1.0)),
          ("acc", "accuracy of the champion (fraction of 256 patterns)",
           "Accuracy on the current goal", (0.5, 1.02))]


def read_cell(base, pattern):
    """-> (config, [(gens, {col: values}) per seed]). Empty list if the cell is absent."""
    dirs = [d for d in glob.glob(os.path.join(base, pattern)) if os.path.isdir(d)]
    if not dirs:
        return None, []
    d = sorted(dirs)[0]
    cfg = {}
    if os.path.exists(os.path.join(d, "config.json")):
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
    cfg["_dir"] = os.path.relpath(d, base)
    seeds = []
    for path in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        gens = np.array([int(r["gen"]) for r in rows], dtype=float)
        acc = np.array([float(r["acc"]) for r in rows])
        L = np.array([int(r["left"]) for r in rows], dtype=float)
        R = np.array([int(r["right"]) for r in rows], dtype=float)
        Mx = np.array([int(r["mixed"]) for r in rows], dtype=float)
        seeds.append((gens, {"sided": (L + R) / np.maximum(L + R + Mx, 1.0),
                             "acc": acc}))
    return cfg, seeds


def on_grid(seeds, grid, col):
    """Step-interpolate every seed's column onto a shared generation grid.

    The logs are NOT on a fixed stride (a row is written on an interval AND on every
    improvement), and seeds stop at different generations, so a plain stack is wrong.
    np.interp holds the last logged value, which is exactly right for a champion
    that only changes when it is replaced."""
    return np.array([np.interp(grid, g, v[col]) for g, v in seeds])


def solve_stats(seeds):
    """-> (n_solved, n_seeds, median generation of first 256/256 among solvers)."""
    firsts = []
    for g, v in seeds:
        hit = np.nonzero(v["acc"] >= 0.999)[0]
        if len(hit):
            firsts.append(g[hit[0]])
    return len(firsts), len(seeds), (np.median(firsts) if firsts else float("nan"))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--nodes", type=int, default=50, help="genome size of the cells to plot")
    ap.add_argument("--runs-dir", default=str(_HERE.parents[1] / "runs"))
    ap.add_argument("--smooth", type=int, default=9, help="moving average, log points")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-open", dest="open", action="store_false", default=True)
    return ap


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    pad = np.r_[np.full(w // 2, y[0]), y, np.full(w - w // 2 - 1, y[-1])]
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def main():
    cli = build_parser().parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells, gmax = [], 0
    for label, pat, colour, ls in CELLS:
        cfg, seeds = read_cell(cli.runs_dir, pat.format(n=cli.nodes))
        if not seeds:
            print(f"  [missing cell: {label}]")
            continue
        gmax = max(gmax, max(g[-1] for g, _ in seeds))
        cells.append((label, colour, ls, cfg, seeds))
    if not cells:
        sys.exit(f"no runs at --nodes {cli.nodes} under {cli.runs_dir}")

    grid = np.unique(np.r_[0, np.logspace(2, np.log10(gmax), 400)])

    print(f"\n{'cell':30s}{'gates':18s}{'gens':>9}{'seeds':>6}{'solved':>9}"
          f"{'med gen@solve':>15}{'sided end':>11}{'acc end':>9}")
    for label, _, _, cfg, seeds in cells:
        ns, nt, med = solve_stats(seeds)
        se = np.mean([v["sided"][-1] for _, v in seeds])
        ae = np.mean([v["acc"][-1] for _, v in seeds])
        print(f"{label:30s}{str(cfg.get('gates','?')):18s}"
              f"{cfg.get('generations',''):>9}{nt:>6}{ns:>4}/{nt:<4}"
              f"{med:>15,.0f}{se:>11.3f}{ae:>9.3f}")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(9.0 * len(PANELS), 6.0))
    for ax, (col, ylab, title, ylim) in zip(axes, PANELS):
        for label, colour, ls, cfg, seeds in cells:
            M = on_grid(seeds, grid, col)
            mean, sd = M.mean(axis=0), M.std(axis=0, ddof=1) if len(M) > 1 else None
            if sd is not None:
                ax.fill_between(grid, smooth(mean - sd, cli.smooth),
                                smooth(mean + sd, cli.smooth), color=colour,
                                alpha=0.10, lw=0)
            ax.plot(grid, smooth(mean, cli.smooth), color=colour, ls=ls, lw=2.0,
                    label=f"{label}   ({len(M)} seeds, final {mean[-1]:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("generation (log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        f"experiment_4 retina_ka2005 — CGP vs ECGP  x  FG vs MVG  "
        f"(n={cli.nodes} nodes, pop 5, gates and/nand/or/nor)\n"
        "mean over seeds, shaded ±1 SD  ·  LEFT panel is the CONE metric "
        "(single-retina gates), NOT circuit purity — experiment_4 logs no purity",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = cli.out or os.path.join(cli.runs_dir, f"fg_mvg_quadrant_n{cli.nodes}.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"\nwrote {out}")
    if cli.open:
        open_file(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
