"""Figures for the budget-matched FG vs MVG retina set (1500 generations each),
drawn as experiment 2's figures are: UCL_thesis/experiments/analysis/fig_brains.py
(--grid) and fig_progress.py.

  grid         2x5 grid of final brains, FG top row, MVG bottom row. Retina pinned
               in two eyes on the left, output on the right, hidden neurons placed
               by a seeded spring layout (no spatial claim). Each hidden neuron is
               coloured by the LEFT/RIGHT MODULE it is assigned to in the split
               `lr_r` is measured at, so colours and the caption's number are one
               measurement. Caption: accuracy on AND | density | lr_r.
  progression  three stacked panels -- champion accuracy on AND, champion density,
               champion lr_r -- mean +- 1 SD over the 5 seeds per condition,
               sampled at the end of each AND epoch (every generation qualifies
               for FG), as fig_progress.py does. Needs replay_archive.py's
               per-generation champions.

Modularity metric: `lr_r` only (experiments/shared_brain_metrics.left_right_split),
Newman's discrete assortativity at the planted left/right split, output excluded.
Purity is not used: it is defined on DAGs and NDP brains are undirected and
recurrent. Q_m is not used (reference-only, see that module's REVISED block).

Why this does not import fig_brains / fig_progress: both import runs_io, which
needs jax/equinox/evosax, and the `ndp` env has none of them. The drawing is
copied from fig_brains.draw_brain with three NDP adaptations:
  * NDP stores the output at node 8 and hidden neurons at 9+; W is reordered to
    experiment 2's [inputs | hidden | output] before anything is scored or drawn.
  * NDP brains are undirected, so each edge is drawn once, and brains reach 80
    neurons, so edge alpha and node size scale down with size.
  * An edge is any nonzero weight (NDP has no prune threshold at this stage).
Density here is undirected edges over all pairs except input-input, which NDP
forbids; experiment 2's is over its IH|HH|HO role mask.

Usage: conda run -n ndp python experiments_paper/retina/matched_figures.py grid --out-dir DIR
       conda run -n ndp python experiments_paper/retina/matched_figures.py progression --out-dir DIR
Figures are written to --out-dir (default: current directory) and opened.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

_NDP_ROOT = Path(__file__).resolve().parents[2]
_UCL_ROOT = _NDP_ROOT.parent
sys.path.insert(0, str(_NDP_ROOT))
sys.path.insert(0, str(_UCL_ROOT))
sys.path.insert(0, str(_UCL_ROOT / "experiments"))

from train_backend import grow_network, retina_fitness  # noqa: E402
from qmetrics.plot import open_file  # noqa: E402
from shared_brain_metrics import left_right_split  # noqa: E402

SAVED_MODELS = _NDP_ROOT / "saved_models"
FG_RUNS = ["1789303257", "1789303833", "1789304372", "1789304978", "1789305630"]
MVG_RUNS = ["1786053806", "1788817038", "1788821061", "1788866451", "1788871442"]
N_IN, OUT = 8, 8                      # ka_task.py layout: inputs 0-7 (0-3 left), output 8

# fig_brains.py colours
LEFT_C, RIGHT_C = "#1f5fa9", "#c0392b"
EXC_C, INH_C = "#2e7d32", "#8e24aa"
LR_COLOR = {0: LEFT_C, 1: RIGHT_C}
# fig_progress.py colours / labels
STYLE = {"FG": dict(color="#2563eb", ls="-", label="FG (L AND R)"),
         "MVG": dict(color="#dc2626", ls="-", label="MVG (AND <-> OR, every 20 gens)")}


def load_config(run_id: str) -> dict:
    with open(SAVED_MODELS / run_id / "config.yml") as f:
        return yaml.load(f, Loader=yaml.Loader)


def grow(dna, config) -> np.ndarray:
    W, _ = grow_network(dna, config)
    return np.asarray(W, dtype=float)


def reorder(W: np.ndarray) -> np.ndarray:
    """NDP [inputs | output | hidden] -> experiment 2's [inputs | hidden | output]."""
    n = W.shape[0]
    order = list(range(N_IN)) + list(range(N_IN + 1, n)) + [OUT]
    return W[np.ix_(order, order)]


def density(W: np.ndarray) -> float:
    """% undirected edges over every pair except input-input (forbidden in NDP)."""
    A = np.abs(W) > 0
    np.fill_diagonal(A, False)
    n = W.shape[0]
    edges = int(np.triu(A | A.T, 1).sum())
    return 100.0 * edges / (n * (n - 1) // 2 - N_IN * (N_IN - 1) // 2)


def score_brain(W: np.ndarray, config: dict) -> dict:
    """W in NDP order. acc on AND, density, lr_r (+ its partition, reordered indices)."""
    Wr = reorder(W)
    n_hid = W.shape[0] - N_IN - 1
    # NDP allows direct input<->output edges, which experiment 2's role mask does
    # not; left_right_split drops the output before scoring, so zeroing them here
    # changes nothing but lets the mask check pass.
    Wl = Wr.copy()
    Wl[:N_IN, -1] = Wl[-1, :N_IN] = 0.0
    r, groups = left_right_split(Wl, N_IN, n_hid, 1, threshold=0.0)
    return {"acc": float(retina_fitness(W=W, config=dict(config, current_op="and"))),
            "density": density(W), "lr_r": r, "groups": groups, "nodes": W.shape[0]}


# ---------------------------------------------------------------------------
# grid -- fig_brains.py
# ---------------------------------------------------------------------------

def _layout(present, n_in, n_hidden, n_out, seed=0):
    """fig_brains._layout verbatim: retina pinned at x=0 in two eyes, output at
    x=2, hidden free (seeded spring layout), hidden blob rescaled to the box."""
    half = n_in // 2
    N = n_in + n_hidden + n_out
    pos = {}
    for i in range(n_in):
        eye, k = (0, i) if i < half else (1, i - half)
        pos[i] = np.array([0.0, 1.0 - (k / max(1, half - 1)) * 0.42 - eye * 0.56])
    for k in range(n_out):
        pos[n_in + n_hidden + k] = np.array([2.0, 0.5 + (k - (n_out - 1) / 2) * 0.12])
    rng = np.random.default_rng(seed)
    for n in range(n_in, n_in + n_hidden):
        pos[n] = np.array([0.6 + 0.8 * rng.random(), rng.random()])

    G = nx.Graph()
    G.add_nodes_from(range(N))
    src, dst = np.nonzero(present)
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    fixed = list(range(n_in)) + [n_in + n_hidden + k for k in range(n_out)]
    pos = nx.spring_layout(G, pos=pos, fixed=fixed, seed=seed, iterations=150,
                           k=1.6 / np.sqrt(max(N, 2)))
    hid = list(range(n_in, n_in + n_hidden))
    H = np.array([pos[n] for n in hid], dtype=float)
    box = ((0.45, 1.70), (0.02, 1.00))
    for d, (lo, hi) in enumerate(box):
        span = float(H[:, d].max() - H[:, d].min())
        H[:, d] = lo + (H[:, d] - H[:, d].min()) / span * (hi - lo) if span > 1e-9 else (lo + hi) / 2.0
    out = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}
    out.update({n: (float(H[k, 0]), float(H[k, 1])) for k, n in enumerate(hid)})
    return out


def draw_brain(ax, Wr, groups, subtitle, lw=0.35):
    """fig_brains.draw_brain(color_by='leftright') for an undirected NDP brain, Wr in
    [inputs | hidden | output] order. Drawn light: NDP weights saturate near |w|=1
    (RESULTS.md), so width-by-|w| drew nearly every edge at full width and made
    33%-dense brains look complete. Edges are one thin uniform width, straight,
    at an alpha that falls with edge count; sign is still the colour."""
    N = Wr.shape[0]
    n_hid = N - N_IN - 1
    present = np.triu((np.abs(Wr) > 0) | (np.abs(Wr.T) > 0), 1)
    pos = _layout(present, N_IN, n_hid, 1)

    src, dst = np.nonzero(present)
    alpha = float(np.clip(25.0 / max(len(src), 1), 0.06, 0.35))
    for i, j in zip(src, dst):
        wt = float(Wr[i, j] if Wr[i, j] != 0 else Wr[j, i])
        (x0, y0), (x1, y1) = pos[int(i)], pos[int(j)]
        ax.plot([x0, x1], [y0, y1], lw=lw, color=EXC_C if wt > 0 else INH_C,
                alpha=alpha, solid_capstyle="round", zorder=1)

    s_hid = float(np.clip(2400.0 / max(n_hid, 1), 32, 140))
    for n, (x, y) in pos.items():
        if n < N_IN:
            ax.scatter([x], [y], s=85, c=[LEFT_C if n < N_IN // 2 else RIGHT_C],
                       edgecolors="white", linewidths=0.6, marker="s", zorder=3)
        elif n < N_IN + n_hid:
            ax.scatter([x], [y], s=s_hid, c=[LR_COLOR.get(groups.get(n), "#eeeeee")],
                       edgecolors="white", linewidths=0.6, zorder=3)
        else:
            ax.scatter([x], [y], s=180, c=["#f5f5f5"], edgecolors="0.35", linewidths=0.8, zorder=3)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.06, 1.06)
    ax.axis("off")
    ax.text(0.5, -0.06, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="0.25", linespacing=1.45)


def _legend(fig):
    handles = [
        Line2D([], [], marker="s", ls="", mfc=LEFT_C, mec="white", ms=7, label="left retina input"),
        Line2D([], [], marker="s", ls="", mfc=RIGHT_C, mec="white", ms=7, label="right retina input"),
        Line2D([], [], marker="o", ls="", mfc="#f5f5f5", mec="0.35", ms=9, label="output"),
        Line2D([], [], marker="o", ls="", mfc=LEFT_C, mec="white", ms=7, label="hidden: assigned to the LEFT module"),
        Line2D([], [], marker="o", ls="", mfc=RIGHT_C, mec="white", ms=7, label="hidden: assigned to the RIGHT module"),
        Line2D([], [], color=EXC_C, lw=1.5, label="excitatory (w > 0)"),
        Line2D([], [], color=INH_C, lw=1.5, label="inhibitory (w < 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)


def cmd_grid(out_dir: Path):
    brains = {}
    for rid in FG_RUNS + MVG_RUNS:
        config = load_config(rid)
        W = grow(np.load(SAVED_MODELS / rid / "solution_best.npy"), config)
        brains[rid] = (config, W, score_brain(W, config))
    # identical wiring (same edge set) gets the same letter; printed, not drawn
    wirings = {}
    for rid, (_, W, _) in brains.items():
        wirings.setdefault((W.shape[0], (np.abs(W) > 0).tobytes()), []).append(rid)
    letter = {rid: chr(ord("A") + i) for i, rids in enumerate(wirings.values()) for rid in rids}

    fig, axes = plt.subplots(2, 5, figsize=(3.5 * 5, 3.5 * 2), squeeze=False)
    for row, (cond, runs) in enumerate([("FG", FG_RUNS), ("MVG", MVG_RUNS)]):
        for col, rid in enumerate(runs):
            config, W, s = brains[rid]
            cap = (f"{cond} - seed {col}\n"
                   f"acc {s['acc']:.2f}  |  {s['nodes']} neurons  |  density {s['density']:.0f}%  |  lr_r {s['lr_r']:+.2f}")
            draw_brain(axes[row][col], reorder(W), s["groups"], cap)
            print(f"{cond} {rid}: acc {s['acc']:.4f} nodes {s['nodes']} density {s['density']:.1f}% "
                  f"lr_r {s['lr_r']:+.3f} wiring {letter[rid]}", flush=True)
    fig.suptitle("NDP encoding — FG vs MVG — retina task, 1500 generations", fontsize=13)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.945), h_pad=3.0)
    out = out_dir / "ndp_matched_brains_grid_leftright.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved -> {out}")
    open_file(str(out))


# ---------------------------------------------------------------------------
# progression -- fig_progress.py
# ---------------------------------------------------------------------------

PANELS = [("acc", "champion accuracy (AND)"),
          ("density", "champion density (%)"),
          ("lr_r", "champion L/R modularity ($lr_r$)")]


def reference_epoch_ends(op: np.ndarray, ref: str = "and") -> np.ndarray:
    """fig_progress.reference_epoch_ends: last generation of each AND epoch, or
    every generation for a fixed-goal run."""
    is_ref = op == ref
    if is_ref.all():
        return np.arange(len(op), dtype=int)
    return np.array([i for i in range(len(op))
                     if is_ref[i] and (i + 1 == len(op) or not is_ref[i + 1])], dtype=int)


def sample_run(run_id: str, n_points: int) -> dict:
    arc = np.load(SAVED_MODELS / run_id / "replay_champions.npz")
    config = load_config(run_id)
    cand = reference_epoch_ends(arc["op"])
    idx = cand[np.unique(np.linspace(0, len(cand) - 1, n_points).astype(int))]
    out = {k: [] for k, _ in PANELS}
    out["gen"], fit_gap = [], 0.0
    for i in idx:
        s = score_brain(grow(arc["champions"][i], config), config)
        fit_gap = max(fit_gap, abs(s["acc"] - float(arc["champ_fit"][i])))
        out["gen"].append(int(i))
        for k, _ in PANELS:
            out[k].append(s[k])
    d = {k: np.asarray(v) for k, v in out.items()}
    d["verified"] = bool(arc["verified"])
    print(f"  sampled {run_id} ({len(idx)} points, max |regrown acc - replay fitness| {fit_gap:.2e}, "
          f"verified {d['verified']})", flush=True)
    return d


def cmd_progression(out_dir: Path, n_points: int):
    runs = {"FG": FG_RUNS, "MVG": MVG_RUNS}
    missing = [r for rs in runs.values() for r in rs
               if not (SAVED_MODELS / r / "replay_champions.npz").exists()]
    if missing:
        raise SystemExit(f"no replay archive yet for {missing}; run replay_archive.py first")
    by_cond = {c: [sample_run(r, n_points) for r in rs] for c, rs in runs.items()}
    unverified = [r for c, rs in runs.items() for r, d in zip(rs, by_cond[c]) if not d["verified"]]

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.5, 3.2 * len(PANELS)), sharex=True)
    for ax, (key, ylabel) in zip(axes, PANELS):
        for cond, ds in by_cond.items():
            n = min(len(d["gen"]) for d in ds)
            x = ds[0]["gen"][:n]
            Y = np.vstack([d[key][:n] for d in ds])
            mu, sd = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0, ddof=1)
            st = STYLE[cond]
            ax.plot(x, mu, lw=1.7, **st)
            ax.fill_between(x, mu - sd, mu + sd, color=st["color"], alpha=0.15, lw=0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        if key == "acc":
            ax.axhline(0.8438, color="#9ca3af", lw=1.0, ls=":")   # popcount plateau, RESULTS.md
        if key == "density":
            ax.set_ylim(0, 102)
        if key == "lr_r":
            ax.axhline(0.0, color="#9ca3af", lw=1.0, ls=":")
    axes[-1].set_xlabel("generation")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

    head = "NDP encoding — FG vs MVG — retina task, 1500 generations"
    if unverified:
        head += f"  [UNVERIFIED REPLAY: {', '.join(unverified)}]"
    n = min(len(v) for v in by_cond.values())
    fig.suptitle(f"{head}\n{n} seed mean per arm, shaded ± 1 SD — sampled at reference-goal epoch ends",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "ndp_matched_progress_fg_vs_mvg.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")
    open_file(str(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["grid", "progression"])
    ap.add_argument("--out-dir", type=Path, default=Path.cwd())
    ap.add_argument("--n-points", type=int, default=60,
                    help="progression: generations sampled per run (fig_progress.py default)")
    args = ap.parse_args()
    if args.command == "grid":
        cmd_grid(args.out_dir)
    else:
        cmd_progression(args.out_dir, args.n_points)


if __name__ == "__main__":
    main()
