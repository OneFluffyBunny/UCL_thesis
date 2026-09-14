"""Aggregate progress: FG vs MVG, constrained vs not, across the whole run.

    python fig_progress.py --root ../experiment_1/runs/fgmvg
    python fig_progress.py --root ../experiment_2/runs/fgmvg --n-points 40

The `kashtan_alon/analysis/fg_mvg_purity.py` equivalent. Where
`fig_switch_window.py` zooms into a few hundred generations, this shows the whole
trajectory with the seeds pooled: median across seeds, with the full seed range
as a band, because 5 seeds is too few for a standard error to mean much.

Accuracy comes from the archive scored ON THE REFERENCE GOAL — NOT from
log.csv's `best_acc`, which under --mvg is the accuracy on whichever goal was
active in that row and therefore alternates between two different tasks. That
distinction is the whole reason `acc_by_op` exists.

Every sample is taken at the END OF A REFERENCE-GOAL EPOCH, not at evenly spaced
generations -- see `reference_epoch_ends`. Even spacing aliases against the
20-generation switch cycle and draws a sawtooth that is an artifact of the
sampling rate.

Modularity is sampled at --n-points points rather than every generation: the
point here is the long-run trend, and the per-generation detail is the other
figure's job.

THREE PANELS, in this order: accuracy, density, `lr_r`. Density sits in the
middle deliberately -- it is the confound, not decoration. The synaptic budget's
first-order effect is to halve density, sparser graphs score higher on any
modularity metric, and within experiment 1's constrained runs `lr_r` correlates
with density at r = -0.471. Any claim read off the bottom panel has to be read
against the middle one. Purity and Newman Q were dropped from this figure on
2026-09-12: purity is no longer offered as evidence, and with two groups of
near-equal degree mass `lr_r` ~ 2Q, so plotting both is plotting one thing twice
(see the REVISED block in `shared_brain_metrics`).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from runs_io import find_runs                         # noqa: E402
from shared_brain_metrics import left_right_split, score_weights   # noqa: E402

# Labels follow `kashtan_alon/analysis/fg_mvg_purity.py`: name the goal regime and
# the switch period, nothing else. The constraint is in the suptitle, so it is
# only repeated here when a figure mixes both sides of the ablation.
STYLE = {
    "budget_fg":    dict(color="#2563eb", ls="-",  label="FG (L AND R)"),
    "budget_mvg":   dict(color="#dc2626", ls="-",  label="MVG (AND <-> OR, every 20 gens)"),
    "nobudget_fg":  dict(color="#2563eb", ls="--", label="FG, no budget"),
    "nobudget_mvg": dict(color="#dc2626", ls="--", label="MVG, no budget"),
}
# Every panel is a property OF THE CHAMPION GENOME of that generation: accuracy
# is the champion scored on the reference goal, and density and lr_r are measured
# on the champion's own weight matrix (see `sample_run`). The curve is then the
# mean of that across seeds -- a mean of per-seed champion values, never a
# population statistic.
PANELS = [("acc", "champion accuracy"),
          ("density", "champion density (%)"),
          ("lr_r", "champion L/R modularity ($lr_r$)")]


def reference_epoch_ends(arc):
    """Indices of the LAST generation of each epoch that ran the reference goal.

    Sampling an MVG run at evenly spaced generations ALIASES: a goal epoch is 20
    generations long, so points 200 apart land at arbitrary phases and the plot
    grows a sawtooth whose period is an artifact of the sampling rate, not of the
    run. Worse, half those points are scored mid-OR and are not comparable to the
    fixed-goal arm at all.

    Sampling the last generation of each AND epoch fixes both: every point is the
    same kind of state (a goal the population has had a full epoch to adapt to),
    and it is the same state `matched` reports at the end, so the trajectory
    actually leads to the number in the table. For a fixed-goal run every
    generation qualifies and this reduces to "all of them".
    """
    op, ref = arc["op"], arc["reference_op"]
    is_ref = op == ref
    if is_ref.all():
        # Fixed goal: the whole run is one contiguous reference block, so asking
        # for "epoch ends" would return exactly one index and the arm would plot
        # as a single invisible point. Every generation is already comparable.
        return np.arange(len(op), dtype=int)
    # last index of each contiguous run of reference-goal generations
    ends = [i for i in range(len(op)) if is_ref[i] and (i + 1 == len(op) or not is_ref[i + 1])]
    return np.asarray(ends, dtype=int)


def sample_run(run, n_points, threshold):
    arc = run.archive()
    if arc is None:
        return None
    gens = arc["gen"]
    cand = reference_epoch_ends(arc)
    if len(cand) == 0:                       # no reference-goal epoch (should not happen)
        cand = np.arange(len(gens))
    idx = cand[np.unique(np.linspace(0, len(cand) - 1, n_points).astype(int))]
    n_in, n_hid, n_out = run.shape
    ref = arc["reference_op"]
    out = {k: [] for k, _ in PANELS}
    out["gen"] = []
    for i in idx:
        w = run.weights_from_flat(arc["flat"][i])
        s = score_weights(w, n_in, n_hid, n_out, rnn_iters=run.cfg.rnn_iters,
                          threshold=threshold, n_rand=0, qm=False, lr=False)
        out["gen"].append(int(gens[i]))
        out["acc"].append(float(arc["acc"][ref][i]))   # always the REFERENCE goal
        out["density"].append(s["density"])
        # `left_right_split`, not score_weights(lr=True): identical `lr_r`, but it
        # skips the q_max rewiring, which is 60-140x the cost and unused here.
        out["lr_r"].append(left_right_split(w, n_in, n_hid, n_out,
                                            threshold=threshold)[0])
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--n-points", type=int, default=60,
                   help="generations sampled per run (evenly spaced)")
    p.add_argument("--constraint", choices=["budget", "nobudget"], default=None,
                   help="only one side of the ablation: `budget` = the constrained arms, `nobudget` = the ablation arms. Default: all four arms in one figure.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"no completed runs under {args.root}")
    if args.constraint:
        runs = [r for r in runs if r.arm.startswith(args.constraint + "_")]
        if not runs:
            raise SystemExit(f"no {args.constraint} runs under {args.root}")
    thr = runs[0].run["prune_threshold"]
    enc = runs[0].encoding

    by_arm = {}
    for r in runs:
        d = sample_run(r, args.n_points, thr)
        if d is None:
            print(f"  {r.arm} seed{r.seed}: no archive, skipped")
            continue
        by_arm.setdefault(r.arm, []).append(d)
        print(f"  sampled {r.arm} seed{r.seed} ({len(d['gen'])} points)")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.5, 3.2 * len(PANELS)), sharex=True)
    for ax, (key, ylabel) in zip(axes, PANELS):
        for arm, ds in by_arm.items():
            n = min(len(d["gen"]) for d in ds)
            x = ds[0]["gen"][:n]
            Y = np.vstack([d[key][:n] for d in ds])
            # Mean and +-1 SD, as in `kashtan_alon/analysis/fg_mvg_purity.py`.
            # The band was previously the full seed range, which a single outlier
            # seed widens to the whole axis and which never narrows as seeds are
            # added; an SD band is comparable between arms and between figures.
            mu = np.nanmean(Y, axis=0)
            st = STYLE.get(arm, dict(label=arm))
            ax.plot(x, mu, lw=1.7, **st)
            if len(Y) > 1:
                sd = np.nanstd(Y, axis=0, ddof=1)
                ax.fill_between(x, mu - sd, mu + sd,
                                color=st.get("color", "0.5"), alpha=0.15, lw=0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        # Bare dotted reference lines, as in KA's figure: 0.750 is the ceiling
        # of every one-eye solution, and 0 is chance for an assortativity at the
        # observed degrees (below it = anti-associated). Both are explained in
        # RESULTS.md; a caption inside each panel is what made this figure wordy.
        if key == "acc":
            ax.axhline(0.75, color="#9ca3af", lw=1.0, ls=":")
        if key == "density":
            ax.set_ylim(0, 102)
        if key == "lr_r":
            ax.axhline(0.0, color="#9ca3af", lw=1.0, ls=":")
    axes[-1].set_xlabel("generation")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

    # KA's `fg_mvg_purity.py` suptitle shape: one headline, then one line of
    # what the curves ARE. Everything else (the sampling rule, why density sits
    # in the middle) is in this module's docstring and in RESULTS.md.
    n = min(len(v) for v in by_arm.values())
    con = {"budget": "synaptic budget",
           "nobudget": "no budget (ablation)"}.get(args.constraint,
                                                   "budget + ablation")
    fig.suptitle(f"{enc} encoding — FG vs MVG — {con}\n"
                 f"{n} seed mean per arm, shaded ± 1 SD — sampled at "
                 f"reference-goal epoch ends",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    suffix = f"_{args.constraint}" if args.constraint else ""
    out = args.out or os.path.join(args.root, f"progress_fg_vs_mvg{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
