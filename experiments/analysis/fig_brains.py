"""Draw the evolved brains — single portraits and the all-runs grid.

    python fig_brains.py --root ../experiment_1/runs/fgmvg --grid
    python fig_brains.py --root ../experiment_1/runs/fgmvg --arm budget_mvg --seed 1

`kashtan_alon/highlight_modules.py` cannot be reused: every one of its entry
points takes a LIST of per-layer weight matrices plus a cfg with `.layers` and
`.offsets`, because KA's net is strictly feedforward. The brains here are a
single (N, N) recurrent adjacency with a cyclic hidden->hidden block, so they
need their own drawing.

Colouring is by THE LEFT/RIGHT MODULE EACH NEURON WAS ASSIGNED TO, which is the
partition `lr_r` -- the study's primary modularity metric -- is measured at. Blue
= the left module, red = the right. The eight retina pixels are pinned (left 4 |
right 4, that is what the task means by left and right); every hidden neuron is
assigned to whichever side maximises Q, and the output is excluded because a
single readout must connect to both halves by construction. So a panel's colours
and its `lr_r` number are two views of the same measurement, and the drawing can
be read straight against `score_table.py`'s `lr_r` column.

`--color-by purity` restores the old ancestry-flow colouring (descriptive only
since 2026-09-12 -- see `shared_brain_metrics`) and `--color-by community` gives
the Newman discovered-partition view for comparison.

LAYOUT. Only the retina sits in a line, on the left, because those eight nodes
are the ones the task actually groups. Hidden neurons are NOT drawn as a layer:
they have no position in this model, so they are
placed by a spring layout with the inputs and the output pinned, letting
connectivity alone decide where each one lands. The output neuron sits to the
right of them all. The layout is seeded, so the same brain always draws the same
way, but no spatial claim is being made -- read the COLOURS, not the distances.
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
from shared_brain_metrics import (left_right_split, recurrent_purity,   # noqa: E402
                                  role_allowed, score_weights)

LEFT_C, RIGHT_C, MIX_C = "#1f5fa9", "#c0392b", "#b9b9b9"
EXC_C, INH_C = "#2e7d32", "#8e24aa"
LR_COLOR = {0: LEFT_C, 1: RIGHT_C}


def _flow_color(v):
    """0 = purely left-fed, 1 = purely right-fed, 0.5 = balanced."""
    if v is None or not np.isfinite(v):
        return "#eeeeee"
    t = abs(v - 0.5) * 2.0                     # 0 mixed .. 1 pure
    base = np.array(matplotlib.colors.to_rgb(LEFT_C if v < 0.5 else RIGHT_C))
    grey = np.array(matplotlib.colors.to_rgb(MIX_C))
    return tuple(grey + (base - grey) * t)


def _layout(present, n_in, n_hidden, n_out, seed=0):
    """node -> (x, y). Retina pinned in a line at x=0, output pinned at x=2,
    hidden neurons FREE in between.

    The hidden neurons are placed by a spring layout rather than stacked in a
    column because they do not live in a layer -- this model gives neurons no
    position at all, and drawing them as a rank invents a structure the model
    does not have. Pinning only the nodes the task does group (the two retina
    halves, and the readout) leaves connectivity to decide the rest: neurons that
    wire together land together. Seeded, so a given brain always draws the same.
    """
    import networkx as nx

    half = n_in // 2
    N = n_in + n_hidden + n_out
    pos = {}
    for i in range(n_in):                       # two visually separated eyes
        eye, k = (0, i) if i < half else (1, i - half)
        pos[i] = np.array([0.0, 1.0 - (k / max(1, half - 1)) * 0.42 - eye * 0.56])
    for k in range(n_out):
        pos[n_in + n_hidden + k] = np.array([2.0, 0.5 + (k - (n_out - 1) / 2) * 0.12])
    rng = np.random.default_rng(seed)           # hidden: seeded start, then free
    for n in range(n_in, n_in + n_hidden):
        pos[n] = np.array([0.6 + 0.8 * rng.random(), rng.random()])

    G = nx.Graph()
    G.add_nodes_from(range(N))
    src, dst = np.nonzero(present)
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    fixed = list(range(n_in)) + [n_in + n_hidden + k for k in range(n_out)]
    pos = nx.spring_layout(G, pos=pos, fixed=fixed, seed=seed, iterations=150,
                           k=1.6 / np.sqrt(max(N, 2)))
    # Rescale the hidden blob to fill the box between the retina and the output.
      # Spring layout stops wherever the forces balance, which for a dense brain is
      # a small clump in one corner; stretching it changes no relative position
      # (it is affine) and uses the canvas. Degenerate ranges fall back to centred.
    hid = list(range(n_in, n_in + n_hidden))
    H = np.array([pos[n] for n in hid], dtype=float)
    box = ((0.45, 1.70), (0.02, 1.00))          # (x_lo, x_hi), (y_lo, y_hi)
    for d, (lo, hi) in enumerate(box):
        span = float(H[:, d].max() - H[:, d].min())
        if span > 1e-9:
            H[:, d] = lo + (H[:, d] - H[:, d].min()) / span * (hi - lo)
        else:
            H[:, d] = (lo + hi) / 2.0
    out = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}
    out.update({n: (float(H[k, 0]), float(H[k, 1])) for k, n in enumerate(hid)})
    return out


def draw_brain(ax, w, n_in, n_hidden, n_out, *, rnn_iters=8, threshold=0.05,
               title="", subtitle="", color_by="leftright", max_lw=2.2):
    """Draw one (N, N) recurrent brain. Returns the info dict from the colouring."""
    w = np.asarray(w)
    allowed = role_allowed(n_in, n_hidden, n_out)
    present = (np.abs(w) > threshold) & allowed
    info = {}

    if color_by == "leftright":
        # Exactly the partition `lr_r` is scored at, so the colours and the
        # number in the subtitle are two views of one measurement.
        r, groups = left_right_split(w, n_in, n_hidden, n_out, threshold=threshold)
        info = {"lr_r": r, "groups": groups}
        node_color = lambda n: LR_COLOR.get(groups.get(n), "#eeeeee")   # noqa: E731
    elif color_by == "community":
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
        _, info = recurrent_purity(w, n_in, n_hidden, rnn_iters, threshold)
        flow = info["flow"]
        node_color = lambda n: _flow_color(flow.get(n))                          # noqa: E731

    half = n_in // 2
    pos = _layout(present, n_in, n_hidden, n_out)

    src, dst = np.nonzero(present)
    if len(src):
        mx = float(np.abs(w[present]).max()) or 1.0
        for i, j in zip(src, dst):
            x0, y0 = pos[int(i)]
            x1, y1 = pos[int(j)]
            wt = float(w[i, j])
            both_hid = (n_in <= i < n_in + n_hidden) and (n_in <= j < n_in + n_hidden)
            rad = 0.18 if both_hid else 0.0                 # curve recurrent edges
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-", lw=0.25 + max_lw * abs(wt) / mx,
                                        color=EXC_C if wt > 0 else INH_C,
                                        alpha=0.42, shrinkA=3.5, shrinkB=3.5,
                                        connectionstyle=f"arc3,rad={rad}"))

    for n, (x, y) in pos.items():
        # Marker shape follows `kashtan_alon/highlight_modules.py`: the INPUT
        # layer is square ("s") and every other neuron is a circle. The output
        # used to be the square here, so it moves to a circle -- it stays
        # distinguishable by being bigger, white-filled and heavier-edged, which
        # is also how KA's readout reads.
        if n < n_in:
            fc = LEFT_C if n < half else RIGHT_C
            ax.scatter([x], [y], s=70, c=[fc], edgecolors="k", linewidths=0.5,
                       marker="s", zorder=3)
        elif n < n_in + n_hidden:
            ax.scatter([x], [y], s=96, c=[node_color(n)], edgecolors="k",
                       linewidths=0.5, zorder=3)
        else:
            ax.scatter([x], [y], s=150, c=["#f5f5f5"], edgecolors="k",
                       linewidths=1.2, zorder=3)

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.06, 1.06)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9.5, pad=3)
    if subtitle:
        ax.text(0.5, -0.10, subtitle, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="0.25", linespacing=1.45)
    return info


def _legend(fig, color_by, ncol=None):
    handles = [
        Line2D([], [], marker="s", ls="", mfc=LEFT_C, mec="k", ms=7, label="left retina input"),
        Line2D([], [], marker="s", ls="", mfc=RIGHT_C, mec="k", ms=7, label="right retina input"),
        Line2D([], [], marker="o", ls="", mfc="#f5f5f5", mec="k", ms=9, label="output"),
        Line2D([], [], color=EXC_C, lw=2, label="excitatory (w > 0)"),
        Line2D([], [], color=INH_C, lw=2, label="inhibitory (w < 0)"),
    ]
    if color_by == "leftright":
        handles[3:3] = [
            Line2D([], [], marker="o", ls="", mfc=LEFT_C, mec="k", ms=8,
                   label="hidden: assigned to the LEFT module"),
            Line2D([], [], marker="o", ls="", mfc=RIGHT_C, mec="k", ms=8,
                   label="hidden: assigned to the RIGHT module"),
        ]
    elif color_by == "purity":
        handles[3:3] = [
            Line2D([], [], marker="o", ls="", mfc=LEFT_C, mec="k", ms=8,
                   label="hidden: left-fed ancestry"),
            Line2D([], [], marker="o", ls="", mfc=MIX_C, mec="k", ms=8,
                   label="hidden: mixed ancestry"),
            Line2D([], [], marker="o", ls="", mfc=RIGHT_C, mec="k", ms=8,
                   label="hidden: right-fed ancestry"),
        ]
    # Long labels: 4 columns need ~11in of figure width, so the single-brain
    # portrait (7.2in) has to wrap or it is clipped at both edges.
    if ncol is None:
        ncol = 4 if fig.get_figwidth() >= 11.0 else 2
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8,
               frameon=False)


def label_for(run, s, *, with_constraint=True):
    """The panel caption: which run this is, then the three reported numbers.

    Seed and goal regime (FG / MVG) identify the run; the constraint is included
    whenever a figure mixes both sides of the ablation, because `budget_mvg` and
    `nobudget_mvg` are both "MVG" and the panel would otherwise be ambiguous.
    The numbers are accuracy ON THE REFERENCE GOAL (AND) -- for an MVG run that
    is the goal-matched champion, i.e. the network as it stood at the end of an
    AND epoch -- then density, then `lr_r`.
    """
    goal = "MVG" if run.is_mvg else "FG"
    con = ""
    if with_constraint:
        con = " - budget" if run.arm.startswith("budget_") else " - no budget"
    return (f"seed {run.seed} - {goal}{con}\n"
            f"acc {s['acc']:.2f}  |  density {s['density']:.0f}%  |  "
            f"lr_r {s['lr_r']:+.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--grid", action="store_true", help="all arms x all seeds in one figure")
    p.add_argument("--arm", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tag", default="matched")
    p.add_argument("--constraint", choices=["budget", "nobudget"], default=None,
                   help="only one side of the ablation: `budget` = the constrained arms, `nobudget` = the ablation arms. Default: all four arms in one figure.")
    p.add_argument("--color-by", choices=["leftright", "purity", "community"],
                   default="leftright",
                   help="`leftright` (default) colours each hidden neuron by the module "
                        "it was assigned to in the left/right split `lr_r` is scored at.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"no completed runs under {args.root}")
    if args.constraint:
        runs = [r for r in runs if r.arm.startswith(args.constraint + "_")]
        if not runs:
            raise SystemExit(f"no {args.constraint} runs under {args.root}")

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
        s["lr_r"] = left_right_split(w, n_in, n_hid, n_out, threshold=thr)[0]
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        draw_brain(ax, w, n_in, n_hid, n_out, rnn_iters=r.cfg.rnn_iters, threshold=thr,
                   color_by=args.color_by,
                   title=f"{r.encoding} — {r.arm.replace('_', ' ')} — seed {r.seed}",
                   subtitle=label_for(r, s))
        _legend(fig, args.color_by)
        fig.tight_layout(rect=(0, 0.16, 1, 1))
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
            s["lr_r"] = left_right_split(w, n_in, n_hid, n_out, threshold=thr)[0]
            # No per-panel title: the caption already names the run, and a title
            # plus a two-line caption on 20 panels leaves no room for the graph.
            draw_brain(ax, w, n_in, n_hid, n_out, rnn_iters=r.cfg.rnn_iters,
                       threshold=thr, color_by=args.color_by,
                       subtitle=label_for(r, s, with_constraint=args.constraint is None))
            print(f"  drew {arm} seed{seed}  lr_r {s['lr_r']:+.3f}")

    enc = runs[0].encoding
    # The constraint HAS to be in the title. With --constraint set, `label_for`
    # drops it from every panel caption (all 10 panels would repeat one word), so
    # the figure would otherwise carry no way at all to tell a budgeted grid from
    # the ablation -- and that is the difference the densities turn on (~30-48%
    # budgeted vs 83-100% ablated).
    con = {"budget": "synaptic budget",
           "nobudget": "no budget (ablation)"}.get(args.constraint,
                                                   "budget + ablation")
    fig.suptitle(f"{enc} encoding — FG vs MVG — {con}", fontsize=13)
    _legend(fig, args.color_by)
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    suffix = f"_{args.constraint}" if args.constraint else ""
    out = args.out or os.path.join(args.root, f"brains_grid_{args.color_by}{suffix}.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
