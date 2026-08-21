"""Phase 3.5 replication -- EXACTLY the two tests preregistered in SPECIALISATION.md.

⚠️ AI-AUTHORED sub-study. Read SPECIALISATION.md's banner.

    conda run -n lndp python analyse_replication.py

Deliberately a separate file from `analyse_specialisation.py`, and deliberately
hard-coded rather than looped over cells: the value of a replication is that its
analysis was fixed in advance, and a script that can only run two tests cannot quietly
grow a third. Seeds 1000-1199, disjoint from phase 3's 0-49.
"""

from __future__ import annotations

import json
import pathlib
import statistics

import numpy as np
from scipy.stats import mannwhitneyu

OUT = pathlib.Path(__file__).resolve().parent / "runs" / "_specrep"
ALPHA = 0.025            # Bonferroni over the two preregistered contrasts
TESTS = (("R1", "cgp4"), ("R2", "cgpnand"))
NL_ = chr(10)


def arm(tag: str) -> list[float]:
    """SPEC for the solved seeds of one arm, from its run directory's `--tag`."""
    vals = []
    for d in OUT.glob(f"*_{tag}"):
        for f in sorted(d.glob("*_result.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["solved_gen"] >= 0 and r.get("spec") not in ("", None):
                vals.append(float(r["spec"]))
    return vals


def rank_biserial(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    u = (x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()
    return 2 * u / (len(x) * len(y)) - 1


def solve_gens(tag: str, tax: int = 0) -> list[int]:
    out = []
    for d in OUT.glob(f"*_{tag}"):
        for f in sorted(d.glob("*_result.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["solved_gen"] >= 0:
                out.append(r["solved_gen"] - tax)
    return out


def transfer():
    """Generations to solve -- NOT preregistered; an out-of-sample follow-up.

    Phase 3 generated this hypothesis and the statistic was chosen after seeing
    phase-3 data, so it is weaker than the two tests above. What it has is fresh
    seeds: the effect was not discovered in this sample. `staged` is charged only for
    stage 2, by subtracting the stage-1 budget it was given, so the question is "did
    having a solved left-detector help?" rather than "was staging cheaper overall?"
    (it was not).
    """
    print(NL_ + "=" * 74)
    print("FOLLOW-UP (not preregistered): stage-2 generations vs cold. + = staging HURT")
    print("=" * 74)
    for enc in ("cgp4", "cgpnand"):
        a = solve_gens(f"rep-staged-partial-{enc}", 20_000)
        b = solve_gens(f"rep-cold-partial-{enc}")
        if not a or not b:
            continue
        rng = np.random.default_rng(0)
        d = statistics.median(a) - statistics.median(b)
        boot = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
                for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = mannwhitneyu(a, b, alternative="two-sided")[1]
        print(f"  {enc:<9} stage2 {statistics.median(a):>8,.0f}  "
              f"cold {statistics.median(b):>8,.0f}  diff {d:>+8,.0f}  "
              f"95% CI [{lo:+,.0f}, {hi:+,.0f}]  p={p:.4g}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    panels = (
        (ax1, lambda sch, enc: arm(f"rep-{sch}-partial-{enc}"),
         "SPEC  (specialised / influencing nodes)",
         "The preregistered outcome: nothing"),
        (ax2, lambda sch, enc: [g / 1000 for g in solve_gens(
            f"rep-{sch}-partial-{enc}", 20_000 if sch == "staged" else 0)],
         "thousands of generations to solve",
         "The one effect that survived: staging COSTS time"),
    )
    for ax, getter, ylab, title in panels:
        data, labs, cols = [], [], []
        for enc in ("cgp4", "cgpnand"):
            for sch, c in (("staged", "#c0392b"), ("cold", "#2c6fbb")):
                data.append(getter(sch, enc))
                labs.append(enc + NL_ + sch)
                cols.append(c)
        bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=1.6), showfliers=False)
        for box, c in zip(bp["boxes"], cols):
            box.set_facecolor(c)
            box.set_alpha(0.45)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Specialisation sub-study, replication (n=200/arm, seeds 1000-1199)"
                 "  -- AI-designed, see SPECIALISATION.md", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "replication.png", dpi=150)
    print(NL_ + f"figure -> {OUT / 'replication.png'}")


def main() -> int:
    print("PHASE 3.5 REPLICATION -- seeds 1000-1199, alpha = 0.025 (Bonferroni over 2)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    verdicts = []
    for name, enc in TESTS:
        a = arm(f"rep-staged-partial-{enc}")
        b = arm(f"rep-cold-partial-{enc}")
        if not a or not b:
            print(f"{name} ({enc}): MISSING DATA (staged n={len(a)}, cold n={len(b)})")
            continue
        u, p = mannwhitneyu(a, b, alternative="greater")
        r = rank_biserial(a, b)
        boot = [rank_biserial(rng.choice(a, len(a)), rng.choice(b, len(b)))
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ok = p < ALPHA
        verdicts.append((name, enc, ok, p, r))
        print(f"\n{name}  staged > cold, overlap=partial, encoding={enc}")
        print(f"  staged  n={len(a):<4} median SPEC {statistics.median(a):.4f}")
        print(f"  cold    n={len(b):<4} median SPEC {statistics.median(b):.4f}")
        print(f"  median difference {statistics.median(a) - statistics.median(b):+.4f}")
        print(f"  U = {u:.1f}   one-sided p = {p:.5g}   rank-biserial r = {r:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]")
        print(f"  --> {'REJECT null' if ok else 'FAIL TO REJECT null'} at alpha={ALPHA}")

    transfer()
    plot()
    print("\n" + "=" * 74)
    for name, enc, ok, p, r in verdicts:
        print(f"{name} ({enc:<8}): {'SUPPORTED' if ok else 'NOT SUPPORTED':<14} "
              f"p={p:.5g}  r={r:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
