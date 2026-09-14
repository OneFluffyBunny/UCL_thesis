"""Aggregate progress over the whole run: FG vs MVG, all seeds pooled.

    python fig_fgmvg_progress.py --root ../runs/fgmvg50

Experiment 4's counterpart of `experiments/analysis/fig_progress.py`. THREE PANELS:

  1  accuracy on AND        -- the goal both arms share, so the curves compare the
                               same task (log.csv's `acc` under MVG alternates goals)
  2  circuit purity         -- `qmetrics.circuit_purity`, output excluded
  3  active gates           -- purity is a mean over gates and falls with circuit size
                               on random circuits, so panel 2 must be read against it

Median across seeds, with the full seed range as a band (5 seeds is too few for a
standard error to mean much).

SAMPLING. Every point is an archived champion at the END OF AN AND EPOCH of the MVG
schedule (generation g with (g+1) = E, 3E, 5E, ...), for BOTH arms: MVG is then
always read after a full epoch of adaptation to AND, and FG is read at exactly the
same generations. Even spacing would alias against the switch cycle. The x axis is
logarithmic because FG solves within the first ~10% of the run.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fgmvg_common import load_study


def series(arm, E: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """gens, {metric: (n_seeds, n_points)} at MVG AND-epoch ends."""
    per_seed = []
    for seed in arm.seeds:
        rows = [r for r in arm.archive(seed) if (r["gen"] + 1) % (2 * E) == E]
        per_seed.append([arm.measure(r) for r in rows])
    n = min(len(s) for s in per_seed)
    gens = np.array([m["gen"] for m in per_seed[0][:n]])
    out = {k: np.array([[m[k] for m in s[:n]] for s in per_seed], dtype=float)
           for k in ("acc_and", "purity", "gates")}
    return gens, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: <root>/figures/progress_fg_vs_mvg.png")
    args = ap.parse_args(argv)

    fg, mvg = load_study(args.root)
    E = mvg.E
    styles = {"FG": dict(color="#2563eb", label=f"FG (L AND R), {fg.alg}"),
              "MVG": dict(color="#dc2626", label=f"MVG (AND <-> OR, every {E} gens), {mvg.alg}")}

    panels = [("acc_and", "accuracy on AND"), ("purity", "circuit purity"),
              ("gates", "active gates")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 10), sharex=True)
    for arm in (fg, mvg):
        gens, data = series(arm, E)
        x = gens + 1
        st = styles[arm.name]
        for ax, (key, _) in zip(axes, panels):
            v = data[key]
            med = np.nanmedian(v, axis=0)
            ax.fill_between(x, np.nanmin(v, axis=0), np.nanmax(v, axis=0),
                            color=st["color"], alpha=0.15, lw=0)
            ax.plot(x, med, color=st["color"], lw=1.6,
                    label=f"{st['label']}, median of {v.shape[0]} seeds")
        last = {k: data[k][:, -1] for k in data}
        print(f"{arm.name}: end (gen {gens[-1]:,}) acc(AND) median "
              f"{np.median(last['acc_and']):.3f} [{last['acc_and'].min():.3f}, "
              f"{last['acc_and'].max():.3f}] | purity median "
              f"{np.nanmedian(last['purity']):.3f} [{np.nanmin(last['purity']):.3f}, "
              f"{np.nanmax(last['purity']):.3f}] | gates median "
              f"{np.median(last['gates']):.0f} [{last['gates'].min():.0f}, "
              f"{last['gates'].max():.0f}]")

    for ax, (key, ylabel) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].axhline(0.75, color="#9ca3af", lw=1.0, ls=":")
    axes[0].text(0.01, 0.752, "0.75 = constant output / best one-sided circuit",
                 transform=axes[0].get_yaxis_transform(), fontsize=8,
                 color="#6b7280", va="bottom")
    axes[0].set_ylim(0.70, 1.01)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("generation (log scale)")
    fig.suptitle("Experiment 4, CGP (50 nodes): FG vs MVG over the whole run\n"
                 "median across seeds, band = seed range; sampled at the end of "
                 "every AND epoch", fontsize=12)
    fig.tight_layout()
    out = args.out or args.root / "figures" / "progress_fg_vs_mvg.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
