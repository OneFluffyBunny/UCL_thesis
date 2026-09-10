"""Per-generation view of the MVG goal switch: FG seed 0 vs MVG seed 0.

Supersedes the every-10-generations version. With --log-interval 10 against a
20-generation switch, each goal phase had exactly TWO logged points, so the
post-switch recovery could not be resolved at all -- a perfect sawtooth rendered
as jagged noise. dense_replay.py re-runs the same seeds logging EVERY
generation inside two windows; this draws them.

Three metrics (rows) x two windows (columns):
  * champion accuracy    -- max over 600, so late in training the switch barely
                            dents it: the population holds solutions to BOTH goals
  * population mean      -- where the switch actually shows, cratering to ~0.5
  * champion purity      -- left/right circuit purity of the champion

Early window [1000, 1200] vs late window [10000, 10200] answers whether the
population re-converges FASTER after a switch once evolution has had time to
build a modular solution.

Usage: conda run -n lndp python kashtan_alon/analysis/switch_window.py
Runs from any working directory.
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).resolve()
RUNS = str(_HERE.parents[1] / "runs_dense")
OUT = os.path.join(RUNS, "switch_window_seed0.png")
SWITCH = 20
WINDOWS = [(100, 300, "very early"), (1000, 1200, "early"), (10000, 10200, "late")]
ARMS = [("retina_fg_raw_seed0", "FG (L AND R)", "#2563eb"),
        ("retina_mvg_raw_seed0", "MVG (AND <-> OR every 20 gens)", "#dc2626")]
RECOVER = 0.90          # "recovered" = population mean back above this


def load(name, lo, hi):
    path = os.path.join(RUNS, f"{name}_log.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if lo <= int(r["gen"]) <= hi]
    rows.sort(key=lambda r: int(r["gen"]))
    return {"gen": np.array([int(r["gen"]) for r in rows]),
            "best": np.array([float(r["best_fit"]) for r in rows]),
            "mean": np.array([float(r["mean_fit"]) for r in rows]),
            "purity": np.array([float(r["purity"]) for r in rows]),
            "op": [r["op"] for r in rows]}


def phase_stats(d):
    """Per goal-phase: the trough at the switch, the peak reached inside the phase,
    and how many generations it took to get 90% of the way from trough to peak.

    Recovery is measured against the phase's OWN peak rather than a fixed accuracy
    threshold: early in training the population never reaches a fixed 0.9 at all,
    so a fixed threshold censors every early phase at the phase length and hides
    the very comparison these two windows exist to make."""
    troughs, peaks, times = [], [], []
    switches = [g for g in d["gen"] if g % SWITCH == 0 and g > d["gen"][0]]
    idx = {g: i for i, g in enumerate(d["gen"])}
    for s in switches:
        i = idx[s]
        seg = d["mean"][i:i + SWITCH]
        if len(seg) < SWITCH:
            continue
        trough, peak = float(seg[0]), float(seg.max())
        thresh = trough + 0.9 * (peak - trough)
        t = int(np.argmax(seg >= thresh))
        troughs.append(trough)
        peaks.append(peak)
        times.append(t)
    return np.array(troughs), np.array(peaks), np.array(times)


def main():
    if not os.path.exists(os.path.join(RUNS, f"{ARMS[0][0]}_log.csv")):
        sys.exit(f"no dense logs in {RUNS} -- run dense_replay.py first")

    rows_spec = [("best", "champion accuracy\n(best of 600)", (0.40, 1.03)),
                 ("mean", "POPULATION MEAN accuracy\n(600 individuals)", (0.30, 1.03)),
                 ("purity", "champion circuit purity", (0.0, 1.05))]

    fig, axes = plt.subplots(3, len(WINDOWS), figsize=(8.5 * len(WINDOWS), 11),
                             sharey="row")
    data = {(n, lo): load(n, lo, hi) for n, _, _ in ARMS for lo, hi, _ in WINDOWS}

    for ci, (lo, hi, wlabel) in enumerate(WINDOWS):
        mvg = data[(ARMS[1][0], lo)]
        for ri, (key, ylab, ylim) in enumerate(rows_spec):
            ax = axes[ri][ci]
            # shade the OR epochs, straight from the logged goal (not recomputed)
            for g, op in zip(mvg["gen"], mvg["op"]):
                if op == "or":
                    ax.axvspan(g - 0.5, g + 0.5, color="#9ca3af", alpha=0.16, lw=0)
            for g in mvg["gen"]:
                if g % SWITCH == 0:
                    ax.axvline(g, color="#6b7280", lw=0.7, ls=":")
            for name, label, colour in ARMS:
                d = data[(name, lo)]
                ax.plot(d["gen"], d[key], "-", lw=1.5, color=colour, label=label)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
            ax.set_xlim(lo, hi)
            # rows share a scale, but every panel keeps its own tick numbers so a
            # column can be read on its own without tracking back to column 1
            ax.tick_params(labelleft=True)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            if ri == 0:
                ax.set_title(f"{wlabel} — generations [{lo}, {hi}]", fontsize=12)
                ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
            if ri == 2:
                ax.set_xlabel("generation")

    # quantify the thing the two windows exist to compare
    lines = []
    for lo, hi, wlabel in WINDOWS:
        d = data[(ARMS[1][0], lo)]
        trough, peak, t = phase_stats(d)
        lines.append(f"{wlabel} [{lo}-{hi}]: population mean {np.mean(trough):.3f} "
                     f"at the switch -> {np.mean(peak):.3f} within the phase, "
                     f"90% of that recovery in {np.mean(t):.1f} gens "
                     f"(n={len(t)} switches)")
        print("MVG " + lines[-1])
        print(f"     per-switch recovery generations: {t.tolist()}")

    fig.suptitle("Kashtan-Alon retina task seed 0", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
