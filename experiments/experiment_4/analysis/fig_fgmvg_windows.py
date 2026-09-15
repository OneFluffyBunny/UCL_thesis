"""Per-generation view inside fixed windows: FG seed 0 vs MVG seed 0.

    python fig_fgmvg_windows.py --root ../runs/fgmvg50
    python fig_fgmvg_windows.py --root ../runs/fgmvg50 --windows 0:6000,792000:798000

Experiment 4's counterpart of `experiments/analysis/fig_switch_window.py`. THREE
METRICS (rows) x N WINDOWS (columns), one FG seed and one MVG seed overlaid:

  1  CHAMPION accuracy on the goal ACTIVE that generation   (KA's `best_fit`)
  2  POPULATION MEAN accuracy, same goal: parent + 4 offspring (KA's `mean_fit`).
     A (1+4) population is 5 individuals, so the raw per-generation mean is drawn
     faint with a centred --smooth-generation rolling mean on top; KA's 600-strong
     population needed no smoothing.
  3  CHAMPION circuit purity                                (KA's modularity row)

Shading marks the MVG schedule's OR epochs; FG's goal is AND throughout.

NO REPLAY IS NEEDED, unlike experiments 1/2 and KA: the runs were made with
`--dense-archive` over these windows, so every generation's champion and the
population mean are on disk from the run itself. The CGP search is integer-only and
seeded from `random.Random`, and the archive was verified not to perturb it (seeds
0-3 reproduce the pre-archive logs byte for byte), so these rows ARE the archived
trajectory.

Windows were fixed before the runs were looked at (2026-09-14): early (0-6000), the
FG seed-0 solve (86000-92000; solve at 89,108), and late (792000-798000). Seed 0
likewise. The printed table gives, per MVG switch inside each window, the drop and
the generations to recover (from the run's own recovery.csv).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fgmvg_common import load_study

DEFAULT_WINDOWS = "0:6000,86000:92000,792000:798000"


def window_series(arm, seed: int, lo: int, hi: int) -> dict[str, np.ndarray]:
    rows = [r for r in arm.archive(seed) if lo <= r["gen"] < hi]
    ms = [arm.measure(r) for r in rows]
    have = {m["gen"] for m in ms}
    missing = (hi - lo) - len(have)
    if missing:
        raise SystemExit(f"{arm.name} seed {seed}: {missing} generations of [{lo},{hi}) "
                         f"not in the archive -- was the run made with --dense-archive?")
    return {k: np.array([m[k] for m in ms], dtype=float)
            for k in ("gen", "acc_active", "pop_mean", "purity")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=50,
                    help="rolling-mean width (generations) for the population row")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: <root>/figures/windows_fg_vs_mvg_seed<k>.png")
    args = ap.parse_args(argv)

    fg, mvg = load_study(args.root)
    E = mvg.E
    windows = [tuple(int(v) for v in w.split(":")) for w in args.windows.split(",")]
    rows_spec = [("acc_active", "champion accuracy"),
                 ("pop_mean", "population mean accuracy"),
                 ("purity", "champion circuit purity")]
    styles = {"FG": dict(color="#2563eb", label="FG (L AND R)"),
              "MVG": dict(color="#dc2626", label=f"MVG (AND <-> OR, every {E} gens)")}

    fig, axes = plt.subplots(len(rows_spec), len(windows),
                             figsize=(5.2 * len(windows), 8.5), squeeze=False,
                             sharey="row")
    rec = mvg.recovery(args.seed)
    for c, (lo, hi) in enumerate(windows):
        for arm in (fg, mvg):
            d = window_series(arm, args.seed, lo, hi)
            col = styles[arm.name]["color"]
            for r, (key, _) in enumerate(rows_spec):
                if key == "pop_mean" and args.smooth > 1:
                    axes[r][c].plot(d["gen"], d[key], color=col, lw=0.4, alpha=0.15)
                    sm = np.convolve(d[key], np.ones(args.smooth) / args.smooth,
                                     mode="valid")
                    off = args.smooth // 2
                    axes[r][c].plot(d["gen"][off:off + len(sm)], sm, color=col, lw=1.3)
                else:
                    axes[r][c].plot(d["gen"], d[key], color=col, lw=0.9,
                                    label=styles[arm.name]["label"])
        for r in range(len(rows_spec)):
            ax = axes[r][c]
            start = (lo // E) * E
            for e0 in range(start, hi, E):
                if mvg.goal_at(e0) == "or":
                    ax.axvspan(max(e0, lo), min(e0 + E, hi), color="#f3f4f6", lw=0)
            ax.set_xlim(lo, hi)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(f"generations {lo:,}–{hi:,}")
            if r == len(rows_spec) - 1:
                ax.set_xlabel("generation")
        print(f"window {lo}-{hi}: MVG seed {args.seed} switches")
        for e in rec:
            if lo < e["gen"] < hi:
                rt = (f">={e['gens_to_recover']} (censored)" if e["censored"]
                      else str(e["gens_to_recover"]))
                print(f"  gen {e['gen']:>7} -> {e['goal']:3s} | hits {e['hits_before']} -> "
                      f"{e['hits_after']} (drop {e['drop']}) | recovered in {rt} gens")

    for r, (_, ylabel) in enumerate(rows_spec):
        axes[r][0].set_ylabel(ylabel)
    for r in (0, 1):
        for ax in axes[r]:
            ax.axhline(0.75, color="#9ca3af", lw=0.8, ls=":")
    axes[2][0].set_ylim(0, 1.02)
    axes[0][0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"CGP — FG vs MVG, seed {args.seed}", fontsize=13)
    fig.tight_layout()
    out = args.out or args.root / "figures" / f"windows_fg_vs_mvg_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
