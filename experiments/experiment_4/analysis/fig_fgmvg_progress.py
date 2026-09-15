"""Aggregate progress over the whole run: FG vs MVG, all seeds pooled.

    python fig_fgmvg_progress.py --root ../runs/study_ga_E2000 [--every 200]

Experiment 4's counterpart of `experiments/analysis/fig_progress.py`. THREE PANELS,
each the mean across seeds with a +-1 SD band (thesis figure style):

  1  champion accuracy on the goal ACTIVE at that generation -- AND for FG; AND or
     OR for MVG. After a switch the population re-adapts within 1-2 generations, so
     at this sampling the curve is "how well the current task is solved".
  2  champion circuit purity  -- `qmetrics.circuit_purity`, output excluded
  3  champion active gates    -- purity falls slightly with circuit size on random
                                 circuits, so panel 2 is read against it

SAMPLING. Every --every generations (archived champions), on a linear axis. The first
version sampled only at MVG AND-epoch ends (every 2E = 4000 generations), which left
25 points for a 100k-generation run; that was the reason for so few points.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fgmvg_common import load_study


def series(arm, every: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """gens, {metric: (n_seeds, n_points)} every `every` generations."""
    per_seed = []
    for seed in arm.seeds:
        rows = [r for r in arm.archive(seed) if (r["gen"] + 1) % every == 0]
        per_seed.append([arm.measure(r) for r in rows])
    n = min(len(s) for s in per_seed)
    gens = np.array([m["gen"] for m in per_seed[0][:n]])
    out = {k: np.array([[m[k] for m in s[:n]] for s in per_seed], dtype=float)
           for k in ("acc_active", "purity", "gates")}
    return gens, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--every", type=int, default=200,
                    help="sampling interval in generations (a multiple of the archive "
                         "interval)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: <root>/figures/progress_fg_vs_mvg.png")
    args = ap.parse_args(argv)

    fg, mvg = load_study(args.root)
    E = mvg.E
    styles = {"FG": dict(color="#2563eb", label="FG (L AND R)"),
              "MVG": dict(color="#dc2626", label=f"MVG (AND <-> OR, every {E} gens)")}

    panels = [("acc_active", "champion accuracy"), ("purity", "champion circuit purity"),
              ("gates", "champion active gates")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 10), sharex=True)
    for arm in (fg, mvg):
        gens, data = series(arm, args.every)
        x = gens + 1
        st = styles[arm.name]
        for ax, (key, _) in zip(axes, panels):
            v = data[key]
            mean, sd = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
            ax.fill_between(x, mean - sd, mean + sd, color=st["color"], alpha=0.15, lw=0)
            ax.plot(x, mean, color=st["color"], lw=1.3, label=st["label"])
        last = {k: data[k][:, -1] for k in data}
        print(f"{arm.name}: {len(gens)} points; end (gen {gens[-1] + 1:,}) acc mean "
              f"{np.mean(last['acc_active']):.3f} | purity mean "
              f"{np.nanmean(last['purity']):.3f} | gates mean {np.mean(last['gates']):.1f}")

    for ax, (_, ylabel) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].axhline(0.75, color="#9ca3af", lw=1.0, ls=":")
    axes[0].set_ylim(0.70, 1.01)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[-1].set_xlim(0, None)
    axes[-1].set_xlabel("generation")
    fig.suptitle(f"CGP — FG vs MVG\n{len(fg.seeds)} seed mean per arm, shaded ± 1 SD",
                 fontsize=12)
    fig.tight_layout()
    out = args.out or args.root / "figures" / "progress_fg_vs_mvg.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
