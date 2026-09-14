"""The three preregistered tests of ENTRENCHMENT.md, and the dose-response curve.

⚠️ AI-AUTHORED sub-study. Read ENTRENCHMENT.md's banner.

    conda run -n lndp python analyse_entrenchment.py

CPython, not PyPy -- numpy/scipy/matplotlib. The search itself ran under PyPy.
Writes `runs/_ent/tidy.csv` and `runs/_ent/entrenchment.png`.
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

OUT = pathlib.Path(__file__).resolve().parent / "runs" / "_ent"
G1_LEVELS = (0, 1_000, 3_000, 10_000, 20_000, 100_000)
GATESETS = ("cgp4", "nand")
ALPHA = 0.05 / 3                     # Bonferroni over the three preregistered tests
NL = chr(10)


def load() -> list[dict]:
    """One row per finished seed, keyed by the run directory's `--tag`."""
    rows = []
    for d in sorted(OUT.glob("*")):
        if not d.is_dir():
            continue
        tag = d.name.rsplit("_", 1)[-1]          # e.g. g1-20000-nand
        parts = tag.split("-")
        if len(parts) != 3 or parts[0] != "g1":
            continue
        g1, gates = int(parts[1]), parts[2]
        for f in sorted(d.glob("*_result.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            solved = r["solved_gen"] >= 0
            rows.append(dict(
                g1=g1, gates=gates, seed=r["seed"], solved=int(solved),
                solved_gen=r["solved_gen"],
                # THE OUTCOME: stage-2 search only. The staged arms are charged for the
                # generations after the second demand arrives, not for acquiring the
                # stage-1 solution -- "did having a detector help or hurt?", not "was
                # staging cheaper overall?" (SPECIALISATION.md already answered that: no).
                stage2_gens=(r["solved_gen"] - g1) if solved else None,
                stage1_active=r.get("stage1_active", -1),
                stage1_solved_gen=r.get("stage1_solved_gen", -1),
                stage1_hits=r.get("stage1_hits", -1),
                active_nodes=r["active_nodes"], seconds=r["seconds"]))
    return rows


def cell(rows, g1, gates) -> list[int]:
    """Stage-2 generations for the solved seeds of one cell."""
    return [r["stage2_gens"] for r in rows
            if r["g1"] == g1 and r["gates"] == gates and r["solved"]]


def med(v):
    return statistics.median(v) if v else float("nan")


def dose_table(rows):
    print(NL + "=" * 78)
    print("DOSE-RESPONSE: stage-2 generations to solve, by stage-1 length")
    print("=" * 78)
    for g in GATESETS:
        print(f"{NL}  gate set: {g}")
        print(f"  {'G1':>8} {'solved':>9} {'median':>10} {'p25':>10} {'p75':>10} "
              f"{'vs cold':>10} {'stage1 active':>14}")
        base = med(cell(rows, 0, g))
        for g1 in G1_LEVELS:
            v = sorted(cell(rows, g1, g))
            n = len([r for r in rows if r["g1"] == g1 and r["gates"] == g])
            if not v:
                continue
            a = [r["stage1_active"] for r in rows
                 if r["g1"] == g1 and r["gates"] == g and r["stage1_active"] >= 0]
            q = lambda p: v[int(p * (len(v) - 1))]
            print(f"  {g1:>8,} {len(v):>4}/{n:<4} {med(v):>10,.0f} {q(.25):>10,.0f} "
                  f"{q(.75):>10,.0f} {med(v) - base:>+10,.0f} "
                  f"{(med(a) if a else float('nan')):>14,.1f}")


def boot_median_diff(a, b, rng, n=10000):
    return np.array([np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
                     for _ in range(n)])


def tests(rows):
    rng = np.random.default_rng(0)
    verdicts = []

    print(NL + "=" * 78)
    print(f"PREREGISTERED TESTS (ENTRENCHMENT.md section 4), alpha = {ALPHA:.4f} each")
    print("=" * 78)

    # --- E-P1: anchor. Third look at a known effect; expected to pass, not new evidence.
    a, b = cell(rows, 20_000, "nand"), cell(rows, 0, "nand")
    u, p = mannwhitneyu(a, b, alternative="greater")
    print(f"{NL}E-P1 (anchor)  NAND, G1=20,000 vs cold: stage-2 generations higher?")
    print(f"  staged n={len(a)} median {med(a):,.0f}   cold n={len(b)} median {med(b):,.0f}"
          f"   diff {med(a) - med(b):+,.0f}")
    print(f"  U={u:,.0f}  one-sided p={p:.5g}  -->  "
          f"{'SUPPORTED' if p < ALPHA else 'NOT SUPPORTED'}")
    verdicts.append(("E-P1", p < ALPHA, f"p={p:.4g}"))

    # --- E-P2: THE discriminating test. Delta_late spans 80,000 generations of extra
    #     stage-1 time, Delta_early spans 3,000. Entrenchment predicts Delta_late ~ 0
    #     because everything worth entrenching is already committed by G1=20,000;
    #     drift-time predicts Delta_late dwarfs Delta_early.
    early_hi, early_lo = cell(rows, 3_000, "nand"), cell(rows, 0, "nand")
    late_hi, late_lo = cell(rows, 100_000, "nand"), cell(rows, 20_000, "nand")
    d_early = med(early_hi) - med(early_lo)
    d_late = med(late_hi) - med(late_lo)
    bd = (boot_median_diff(late_hi, late_lo, np.random.default_rng(1))
          - boot_median_diff(early_hi, early_lo, np.random.default_rng(2)))
    lo, hi = np.percentile(bd, [2.5, 97.5])
    ok = hi < 0
    print(f"{NL}E-P2 (saturation)  is the LATE rise smaller than the EARLY rise?")
    print(f"  Delta_early  G1 0 -> 3,000      = {d_early:+,.0f}   (3,000 gens of dose)")
    print(f"  Delta_late   G1 20,000 -> 100,000 = {d_late:+,.0f}   (80,000 gens of dose)")
    print(f"  Delta_late - Delta_early = {d_late - d_early:+,.0f}  "
          f"95% CI [{lo:+,.0f}, {hi:+,.0f}]")
    # NB: failing to exclude zero is NOT evidence for drift-time -- it is an
    # inconclusive test. The direction of the point estimate is reported separately
    # so the verdict line cannot be read as support for the alternative.
    print(f"  --> {'SUPPORTED (saturates: ENTRENCHMENT)' if ok else 'NOT SUPPORTED (CI includes 0 -- INCONCLUSIVE, not evidence for drift-time)'}")
    print(f"      point estimate direction: "
          f"{'saturating' if d_late < d_early else 'still rising'}")
    verdicts.append(("E-P2", ok, f"CI [{lo:+,.0f}, {hi:+,.0f}]"))

    # --- E-P3: mechanism. Does the AMOUNT of committed structure predict the damage?
    sel = [r for r in rows if r["g1"] == 20_000 and r["gates"] == "nand"
           and r["solved"] and r["stage1_active"] >= 0]
    x = [r["stage1_active"] for r in sel]
    y = [r["stage2_gens"] for r in sel]
    rho, p3 = spearmanr(x, y)
    p3 = p3 / 2 if rho > 0 else 1 - p3 / 2          # one-sided, positive direction
    print(f"{NL}E-P3 (mechanism)  NAND, G1=20,000: does stage-1 circuit SIZE predict the cost?")
    print(f"  n={len(sel)}  median stage1_active {med(x):,.1f}  "
          f"Spearman rho={rho:+.3f}  one-sided p={p3:.5g}")
    print(f"  --> {'SUPPORTED' if p3 < ALPHA and rho > 0 else 'NOT SUPPORTED'}")
    verdicts.append(("E-P3", p3 < ALPHA and rho > 0, f"rho={rho:+.3f} p={p3:.4g}"))

    print(NL + "=" * 78)
    for name, ok, detail in verdicts:
        print(f"  {name}: {'SUPPORTED' if ok else 'NOT SUPPORTED':<15} {detail}")
    print("=" * 78)
    return verdicts


def descriptive(rows):
    """Same three quantities in the other cells. Description, no claims."""
    print(f"{NL}DESCRIPTIVE (no significance claimed) -- rho(stage1_active, stage-2 gens)")
    print(f"  {'gates':<7} {'G1':>9} {'n':>5} {'rho':>8} {'p(2-sided)':>12}")
    for g in GATESETS:
        for g1 in G1_LEVELS:
            if g1 == 0:
                continue
            sel = [r for r in rows if r["g1"] == g1 and r["gates"] == g
                   and r["solved"] and r["stage1_active"] >= 0]
            if len(sel) < 10:
                continue
            rho, p = spearmanr([r["stage1_active"] for r in sel],
                               [r["stage2_gens"] for r in sel])
            print(f"  {g:<7} {g1:>9,} {len(sel):>5} {rho:>+8.3f} {p:>12.4g}")


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    xs = np.arange(len(G1_LEVELS))
    for g, c in (("cgp4", "#2c6fbb"), ("nand", "#c0392b")):
        meds, los, his = [], [], []
        for g1 in G1_LEVELS:
            v = sorted(cell(rows, g1, g))
            meds.append(med(v))
            los.append(v[int(.25 * (len(v) - 1))] if v else np.nan)
            his.append(v[int(.75 * (len(v) - 1))] if v else np.nan)
        meds, los, his = np.array(meds) / 1000, np.array(los) / 1000, np.array(his) / 1000
        ax1.plot(xs, meds, "o-", color=c, label=g, lw=2)
        ax1.fill_between(xs, los, his, color=c, alpha=0.15)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{g:,}" for g in G1_LEVELS], fontsize=8)
    ax1.set_xlabel("stage-1 length G1 (generations); 0 = cold")
    ax1.set_ylabel("stage-2 generations to solve (thousands)")
    ax1.axvspan(1.5, 2.5, color="0.6", alpha=0.18)
    ax1.text(2.0, ax1.get_ylim()[1] * 0.96, "stage 1\ntypically solved",
             ha="center", va="top", fontsize=8, color="0.3")
    ax1.set_title("Dose-response: does the damage plateau once stage 1 is solved?",
                  fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    sel = [r for r in rows if r["g1"] == 20_000 and r["gates"] == "nand"
           and r["solved"] and r["stage1_active"] >= 0]
    if sel:
        ax2.plot([r["stage1_active"] for r in sel],
                 [r["stage2_gens"] / 1000 for r in sel], "o", ms=4, alpha=0.45,
                 color="#c0392b")
        rho, _ = spearmanr([r["stage1_active"] for r in sel],
                           [r["stage2_gens"] for r in sel])
        ax2.set_title(f"E-P3: committed stage-1 size vs the cost\n"
                      f"NAND, G1=20,000, n={len(sel)}, Spearman rho={rho:+.3f}",
                      fontsize=10)
    ax2.set_xlabel("active nodes in the stage-1 circuit")
    ax2.set_ylabel("stage-2 generations to solve (thousands)")
    ax2.grid(alpha=0.3)
    fig.suptitle("Entrenchment sub-study (n=200/cell, seeds 2000-2199)"
                 "  -- AI-designed, see ENTRENCHMENT.md", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "entrenchment.png", dpi=150)
    print(f"{NL}figure -> {OUT / 'entrenchment.png'}")


def main() -> int:
    rows = load()
    if not rows:
        print(f"no results under {OUT} -- run run_entrenchment.py first")
        return 1
    with (OUT / "tidy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} seed-runs -> {OUT / 'tidy.csv'}")
    dose_table(rows)
    tests(rows)
    descriptive(rows)
    plot(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
