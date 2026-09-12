"""Aggregate progress: FG vs MVG, constrained vs not, across the whole run.

    python fig_progress.py --root ../experiment_1/runs/fgmvg
    python fig_progress.py --root ../experiment_2/runs/fgmvg --n-points 40

The `kashtan_alon/analysis/fg_mvg_purity.py` equivalent. Where
`fig_switch_window.py` zooms into a few hundred generations, this shows the whole
trajectory with the seeds pooled: median across seeds, with the full seed range
as a band, because 5 seeds is too few for a standard error to mean much.

Accuracy comes from the archive scored ON THE REFERENCE GOAL at every sampled
generation — NOT from log.csv's `best_acc`, which under --mvg is the accuracy on
whichever goal happened to be active in that row and therefore alternates between
two different tasks. That distinction is the whole reason `acc_by_op` exists.

Modularity is sampled at --n-points generations rather than every one: the point
here is the long-run trend, and the per-generation detail is the other figure's
job.
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
from shared_brain_metrics import score_weights        # noqa: E402

STYLE = {
    "budget_fg":    dict(color="#1f77b4", ls="-",  label="constrained, fixed goal"),
    "budget_mvg":   dict(color="#d62728", ls="-",  label="constrained, MVG"),
    "nobudget_fg":  dict(color="#1f77b4", ls="--", label="ablation (no budget), fixed goal"),
    "nobudget_mvg": dict(color="#d62728", ls="--", label="ablation (no budget), MVG"),
}
PANELS = [("acc", "accuracy on the reference goal (raw)"),
          ("purity", "purity (primary)"),
          ("q", "Newman Q (secondary)"),
          ("density", "density (%)")]


def sample_run(run, n_points, threshold):
    arc = run.archive()
    if arc is None:
        return None
    gens = arc["gen"]
    idx = np.unique(np.linspace(0, len(gens) - 1, n_points).astype(int))
    n_in, n_hid, n_out = run.shape
    ref = arc["reference_op"]
    out = {k: [] for k, _ in PANELS}
    out["gen"] = []
    for i in idx:
        s = score_weights(run.weights_from_flat(arc["flat"][i]), n_in, n_hid, n_out,
                          rnn_iters=run.cfg.rnn_iters, threshold=threshold,
                          n_rand=0, qm=False, lr=False)
        out["gen"].append(int(gens[i]))
        out["acc"].append(float(arc["acc"][ref][i]))   # always the REFERENCE goal
        out["purity"].append(s["purity"])
        out["q"].append(s.get("q", np.nan))
        out["density"].append(s["density"])
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--n-points", type=int, default=60,
                   help="generations sampled per run (evenly spaced)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"no completed runs under {args.root}")
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

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.5, 3.0 * len(PANELS)), sharex=True)
    for ax, (key, ylabel) in zip(axes, PANELS):
        for arm, ds in by_arm.items():
            n = min(len(d["gen"]) for d in ds)
            x = ds[0]["gen"][:n]
            Y = np.vstack([d[key][:n] for d in ds])
            med = np.nanmedian(Y, axis=0)
            lo, hi = np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
            st = STYLE.get(arm, dict(label=arm))
            ax.plot(x, med, lw=1.7, **st)
            ax.fill_between(x, lo, hi, color=st.get("color", "0.5"), alpha=0.12, lw=0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        if key == "acc":
            ax.axhline(0.75, color="0.35", lw=0.9, ls=":")
            ax.text(x[0], 0.755, "0.750 — every one-eye solution caps here",
                    fontsize=7.5, color="0.3")
        if key == "density":
            ax.set_ylim(0, 102)
    axes[-1].set_xlabel("generation")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

    n_seeds = {a: len(v) for a, v in by_arm.items()}
    fig.suptitle(
        f"FG vs MVG x constraint — {enc} encoding — median across seeds, band = full range\n"
        f"seeds per arm: {n_seeds} | edge = |w| > {thr} | accuracy always scored on the "
        f"reference goal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = args.out or os.path.join(args.root, "progress_fg_vs_mvg.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
