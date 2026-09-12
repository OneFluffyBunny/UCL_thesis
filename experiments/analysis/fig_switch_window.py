"""Generation-by-generation accuracy and modularity across goal switches.

    python fig_switch_window.py --root ../experiment_1/runs/fgmvg --seed 0
    python fig_switch_window.py --root ../experiment_1/runs/fgmvg --seed 0 \
        --start 9000 --width 400 --constraint budget

The `kashtan_alon/analysis/switch_window.py` equivalent, and the figure this
study most needs: a goal epoch is only --switch-interval (20) generations long,
so an aggregate curve sampled every 10 or 100 generations cannot show what
happens ACROSS a switch. This reads the per-generation champion archive
(`champions.npz`, written by --archive-interval 1) and scores every single
generation in a window.

Three rows, shared x:

  accuracy    the champion's accuracy on EVERY goal, not just the active one.
              This is the row that shows whether an MVG population HOLDS both
              targets or merely swaps between them -- if the two curves alternate
              in antiphase, the run is trading, not generalising.
  modularity  purity (per-generation, cheap, always defined) and Newman Q.
  density     the confound to watch: a modularity change that is really a
              density change is not a modularity change.

Shading marks the goal epochs; the FG column is the control with no switches.

Q and purity are cheap enough to run every generation. The left/right null model
is NOT (a degree-preserving null per generation would dominate the runtime), so
LR belongs in the endpoint table (`score_table.py`), not here.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from runs_io import find_runs                        # noqa: E402
from shared_brain_metrics import score_weights       # noqa: E402

OP_COLOR = {"and": "#1f77b4", "or": "#d62728", "xor": "#2ca02c"}


def trajectory(run, gens, threshold):
    """Score every generation in `gens`. Cheap metrics only — see the docstring."""
    arc = run.archive()
    if arc is None:
        raise SystemExit(f"{run.dir} has no champions.npz "
                         f"(the run needs --archive-interval > 0)")
    n_in, n_hid, n_out = run.shape
    index = {int(g): i for i, g in enumerate(arc["gen"])}
    rows = []
    for g in gens:
        if g not in index:
            continue
        i = index[g]
        s = score_weights(run.weights_from_flat(arc["flat"][i]), n_in, n_hid, n_out,
                          rnn_iters=run.cfg.rnn_iters, threshold=threshold,
                          n_rand=0, qm=False, lr=False)   # nulls are too slow per-gen
        s["gen"] = g
        s["op"] = arc["op"][i]
        for op in arc["ops_in_play"]:
            s[f"acc_{op}"] = float(arc["acc"][op][i])
        rows.append(s)
    return rows, arc


def shade_epochs(ax, rows, switch_interval):
    """Tint the background by which goal was active."""
    if not rows:
        return
    cur, start = rows[0]["op"], rows[0]["gen"]
    for r in rows[1:] + [None]:
        if r is None or r["op"] != cur:
            end = rows[-1]["gen"] if r is None else r["gen"]
            ax.axvspan(start, end, color=OP_COLOR.get(cur, "#999999"), alpha=0.07, lw=0)
            if r is not None:
                ax.axvline(r["gen"], color="0.55", lw=0.6, ls=":")
                cur, start = r["op"], r["gen"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--constraint", choices=["budget", "nobudget"], default="budget")
    p.add_argument("--start", type=int, default=None,
                   help="first generation of the window (default: near the end of the run)")
    p.add_argument("--width", type=int, default=400, help="generations in the window")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = {r.arm: r for r in find_runs(args.root) if r.seed == args.seed}
    want = [f"{args.constraint}_fg", f"{args.constraint}_mvg"]
    have = [a for a in want if a in runs]
    if not have:
        raise SystemExit(f"no {args.constraint} runs for seed {args.seed} under {args.root}")

    ref = runs[have[0]]
    total = ref.rj.get("gens_run", 0)
    # Default window: the LAST `width` generations, which is where the structure
    # being reported actually comes from.
    start = args.start if args.start is not None else max(0, total - args.width)
    gens = list(range(start, min(start + args.width, total)))
    thr = ref.run["prune_threshold"]
    switch = ref.run["switch_interval"]

    print(f"scoring generations {gens[0]}..{gens[-1]} for {len(have)} arms "
          f"(seed {args.seed}, {ref.encoding} encoding)")
    data = {}
    for arm in have:
        data[arm] = trajectory(runs[arm], gens, thr)[0]
        print(f"  {arm}: {len(data[arm])} generations scored")

    # one Q scale for every column, computed before drawing
    q_all = [r.get("q", np.nan) for rows in data.values() for r in rows]
    q_top = float(np.nanmax(q_all)) * 1.1 if np.isfinite(np.nanmax(q_all)) else 1.0

    fig, axes = plt.subplots(3, len(have), figsize=(7.2 * len(have), 8.4),
                             sharex=True, squeeze=False)
    for col, arm in enumerate(have):
        rows = data[arm]
        if not rows:
            continue
        x = [r["gen"] for r in rows]
        is_mvg = arm.endswith("_mvg")

        # --- row 0: accuracy on every goal --------------------------------
        ax = axes[0][col]
        if is_mvg:
            shade_epochs(ax, rows, switch)
        ops = sorted({k[4:] for k in rows[0] if k.startswith("acc_")})
        for op in ops:
            ax.plot(x, [r[f"acc_{op}"] for r in rows], lw=1.2,
                    color=OP_COLOR.get(op, None), label=f"accuracy on {op.upper()}")
        ax.axhline(0.75, color="0.4", lw=0.8, ls="--")
        ax.text(x[0], 0.757, "0.750 = one-eye cap", fontsize=7.5, color="0.35")
        # FIXED range across columns. Autoscaling made the converged FG panel
        # span 0.76-0.86 and the MVG panel 0.3-0.85, so the same vertical
        # distance meant different things in the two columns and the one-eye
        # cap fell off the FG axis entirely.
        ax.set_ylim(0.25, 1.02)
        ax.set_ylabel("accuracy (raw)")
        ax.set_title(f"{ref.encoding} encoding — {arm.replace('_', ' ')} — seed {args.seed}",
                     fontsize=11)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

        # --- row 1: modularity --------------------------------------------
        ax = axes[1][col]
        if is_mvg:
            shade_epochs(ax, rows, switch)
        ax.plot(x, [r["purity"] for r in rows], lw=1.3, color="#6a3d9a", label="purity (primary)")
        ax.set_ylabel("purity")
        ax.set_ylim(-0.02, 1.02)
        ax2 = ax.twinx()
        ax2.plot(x, [r.get("q", np.nan) for r in rows], lw=1.0, color="#ff7f0e",
                 alpha=0.85, label="Newman Q (secondary)")
        ax2.set_ylabel("Newman Q", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax2.set_ylim(0, q_top)          # shared across columns, same reason
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)

        # --- row 2: density -----------------------------------------------
        ax = axes[2][col]
        if is_mvg:
            shade_epochs(ax, rows, switch)
        ax.plot(x, [r["density"] for r in rows], lw=1.2, color="#2ca02c")
        ax.set_ylabel("density (%)")
        ax.set_xlabel("generation")
        ax.set_ylim(0, 102)

    sub = (f"{os.path.basename(os.path.normpath(args.root))} | "
           f"{args.constraint} | goal switches every {switch} generations | "
           f"edge = |w| > {thr}")
    fig.suptitle("Accuracy and modularity across goal switches, generation by generation\n"
                 + sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = args.out or os.path.join(args.root, f"switch_window_{args.constraint}_seed{args.seed}.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
