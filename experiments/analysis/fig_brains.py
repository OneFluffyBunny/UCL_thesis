"""Draw the evolved brains — single portraits and the all-runs grid.

    python fig_brains.py --root ../experiment_1/runs/fgmvg --grid
    python fig_brains.py --root ../experiment_1/runs/fgmvg --arm budget_mvg --seed 1

`kashtan_alon/highlight_modules.py` cannot be reused: every one of its entry
points takes a LIST of per-layer weight matrices plus a cfg with `.layers` and
`.offsets`, because KA's net is strictly feedforward. The brains here are a
single (N, N) recurrent adjacency with a cyclic hidden->hidden block, so they
need their own drawing.

Colouring is by PURITY FLOW rather than by discovered community, and that is the
point of the picture. The flow value is where a neuron's ancestry comes from --
blue = fed by the left half of the retina, red = by the right half, grey =
by both. So the drawing shows the split the TASK is about, not a split some
community-detection algorithm happened to find, and it can be read against
`score_table.py`'s `lr` and `purity` columns directly. `--color-by community`
gives the Newman view for comparison.

Hidden neurons are ordered top-to-bottom by flow, so a modular brain shows two
visually separated bands and a mixed one shows a gradient. That ordering is
cosmetic -- neurons have no position in this model (see CLAUDE.md: no physical
space) -- and it is applied only to make the structure legible.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from runs_io import find_runs                         # noqa: E402
from shared_brain_metrics import recurrent_purity, role_allowed, score_weights  # noqa: E402

LEFT_C, RIGHT_C, MIX_C = "#1f5fa9", "#c0392b", "#b9b9b9"
EXC_C, INH_C = "#2e7d32", "#8e24aa"


def _flow_color(v):
    """0 = purely left-fed, 1 = purely right-fed, 0.5 = balanced."""
    if v is None or not np.isfinite(v):
        return "#eeeeee"
    t = abs(v - 0.5) * 2.0                     # 0 mixed .. 1 pure
    base = np.array(matplotlib.colors.to_rgb(LEFT_C if v < 0.5 else RIGHT_C))
    grey = np.array(matplotlib.colors.to_rgb(MIX_C))
    return tuple(grey + (base - grey) * t)


def draw_brain(ax, w, n_in, n_hidden, n_out, *, rnn_iters=8, threshold=0.05,
               title="", subtitle="", color_by="purity", max_lw=2.2):
    """Draw one (N, N) recurrent brain. Returns the info dict from the colouring."""
    w = np.asarray(w)
    allowed = role_allowed(n_in, n_hidden, n_out)
    present = (np.abs(w) > threshold) & allowed

    _, info = recurrent_purity(w, n_in, n_hidden, rnn_iters, threshold)
    flow = info["flow"]

    if color_by == "community":
        import networkx as nx
        from qmetrics import from_matrix
        G = from_matrix(w, threshold=threshold, directed=False, weighted=True, allowed=allowed)
        try:
            comms = nx.community.greedy_modularity_communities(G, weight="weight")
        except Exception:
            comms = []
        cmap = plt.get_cmap("tab10")
        cid = {n: i for i, c in enumerate(comms) for n in c}
        node_color = lambda n: (cmap(cid[n] % 10) if n in cid else "#eeeeee")   # noqa: E731
    else:
        node_color = lambda n: _flow_color(flow.get(n))                          # noqa: E731

    half = n_in // 2
    pos = {}
    # inputs: two visually separated eyes
    for i in range(n_in):
        eye, k = (0, i) if i < half else (1, i - half)
        pos[i] = (0.0, 1.0 - (k / max(1, half - 1)) * 0.42 - eye * 0.56)
    # hidden: ordered by flow so a split reads as two bands
    order = sorted(range(n_in, n_in + n_hidden),
                   key=lambda n: (flow.get(n, 0.5), n))
    for rank, n in enumerate(order):
        pos[n] = (1.0, 1.0 - rank / max(1, n_hidden - 1))
    for k in range(n_out):
        pos[n_in + n_hidden + k] = (2.0, 0.5 + (k - (n_out - 1) / 2) * 0.12)

    src, dst = np.nonzero(present)
    if len(src):
        mx = float(np.abs(w[present]).max()) or 1.0
        for i, j in zip(src, dst):
            x0, y0 = pos[int(i)]
            x1, y1 = pos[int(j)]
            wt = float(w[i, j])
            rad = 0.28 if abs(x1 - x0) < 1e-9 else 0.0      # curve recurrent edges
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-", lw=0.25 + max_lw * abs(wt) / mx,
                                        color=EXC_C if wt > 0 else INH_C,
                                        alpha=0.42, shrinkA=3.5, shrinkB=3.5,
                                        connectionstyle=f"arc3,rad={rad}"))

    for n, (x, y) in pos.items():
        if n < n_in:
            fc = LEFT_C if n < half else RIGHT_C
            ax.scatter([x], [y], s=58, c=[fc], edgecolors="k", linewidths=0.5, zorder=3)
        elif n < n_in + n_hidden:
            ax.scatter([x], [y], s=96, c=[node_color(n)], edgecolors="k",
                       linewidths=0.5, zorder=3)
        else:
            ax.scatter([x], [y], s=110, c=["#f5f5f5"], edgecolors="k",
                       linewidths=1.0, marker="s", zorder=3)

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.12, 1.12)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9.5, pad=3)
    if subtitle:
        ax.text(0.5, -0.07, subtitle, transform=ax.transAxes, ha="center",
                fontsize=8, color="0.3")
    return info


def _legend(fig, color_by):
    handles = [
        Line2D([], [], marker="o", ls="", mfc=LEFT_C, mec="k", ms=7, label="left retina input"),
        Line2D([], [], marker="o", ls="", mfc=RIGHT_C, mec="k", ms=7, label="right retina input"),
        Line2D([], [], marker="s", ls="", mfc="#f5f5f5", mec="k", ms=7, label="output"),
        Line2D([], [], color=EXC_C, lw=2, label="excitatory (w > 0)"),
        Line2D([], [], color=INH_C, lw=2, label="inhibitory (w < 0)"),
    ]
    if color_by == "purity":
        handles[3:3] = [
            Line2D([], [], marker="o", ls="", mfc=LEFT_C, mec="k", ms=8,
                   label="hidden: left-fed ancestry"),
            Line2D([], [], marker="o", ls="", mfc=MIX_C, mec="k", ms=8,
                   label="hidden: mixed ancestry"),
            Line2D([], [], marker="o", ls="", mfc=RIGHT_C, mec="k", ms=8,
                   label="hidden: right-fed ancestry"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)


def label_for(run, s):
    return (f"acc {s['acc']:.3f} | dens {s['density']:.0f}% | "
            f"purity {s['purity']:.2f} | Q {s.get('q', float('nan')):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--grid", action="store_true", help="all arms x all seeds in one figure")
    p.add_argument("--arm", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tag", default="matched")
    p.add_argument("--color-by", choices=["purity", "community"], default="purity")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"no completed runs under {args.root}")

    if not args.grid:
        sel = [r for r in runs
               if (args.arm is None or r.arm == args.arm)
               and (args.seed is None or r.seed == args.seed)]
        if not sel:
            raise SystemExit("no run matched --arm/--seed")
        r = sel[0]
        n_in, n_hid, n_out = r.shape
        thr = r.run["prune_threshold"]
        w = r.weights(args.tag)
        s = score_weights(w, n_in, n_hid, n_out, rnn_iters=r.cfg.rnn_iters,
                          threshold=thr, qm=False, lr=False)
        s["acc"] = r.accuracy(args.tag)
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        draw_brain(ax, w, n_in, n_hid, n_out, rnn_iters=r.cfg.rnn_iters, threshold=thr,
                   color_by=args.color_by,
                   title=f"{r.encoding} — {r.arm.replace('_', ' ')} — seed {r.seed}",
                   subtitle=label_for(r, s))
        _legend(fig, args.color_by)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        out = args.out or os.path.join(args.root, f"brain_{r.arm}_seed{r.seed}.png")
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
        return

    arms = [a for a in ["budget_fg", "budget_mvg", "nobudget_fg", "nobudget_mvg"]
            if any(r.arm == a for r in runs)]
    seeds = sorted({r.seed for r in runs})
    fig, axes = plt.subplots(len(arms), len(seeds),
                             figsize=(3.5 * len(seeds), 3.5 * len(arms)), squeeze=False)
    for i, arm in enumerate(arms):
        for j, seed in enumerate(seeds):
            ax = axes[i][j]
            sel = [r for r in runs if r.arm == arm and r.seed == seed]
            if not sel:
                ax.axis("off")
                continue
            r = sel[0]
            n_in, n_hid, n_out = r.shape
            thr = r.run["prune_threshold"]
            w = r.weights(args.tag)
            s = score_weights(w, n_in, n_hid, n_out, rnn_iters=r.cfg.rnn_iters,
                              threshold=thr, qm=False, lr=False)
            s["acc"] = r.accuracy(args.tag)
            draw_brain(ax, w, n_in, n_hid, n_out, rnn_iters=r.cfg.rnn_iters,
                       threshold=thr, color_by=args.color_by,
                       title=(f"{arm.replace('_', ' ')} — seed {seed}" if i == 0 or True else ""),
                       subtitle=label_for(r, s))
            print(f"  drew {arm} seed{seed}")

    enc = runs[0].encoding
    fig.suptitle(f"Goal-matched champion of every run — {enc} encoding "
                 f"({os.path.basename(os.path.normpath(args.root))})\n"
                 f"hidden neurons coloured by {args.color_by}, ordered by left/right ancestry",
                 fontsize=12)
    _legend(fig, args.color_by)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = args.out or os.path.join(args.root, f"brains_grid_{args.color_by}.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
