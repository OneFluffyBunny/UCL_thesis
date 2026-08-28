"""Does SPEC measure modular organisation? -- see CALIBRATION.md.

⚠️ AI-AUTHORED sub-study. Read CALIBRATION.md's banner.

    conda run -n lndp python analyse_calibration.py

Two legs, and the second is the one that matters:

  1. RANDOM-GENOTYPE NULL -- SPEC on unevolved circuits. Sets the range the statistic
     can actually move within, per task. Computed here, not run: it needs no search.
  2. GROUND TRUTH -- SPEC on SOLVED circuits for the three `pair_*` tasks, which share
     a shape (8 in, 2 out, same node budget, same null) and differ ONLY in how much
     their two outputs are forced to share. That isolates the metric's response to
     decomposition from every other difference.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics

from scipy.stats import mannwhitneyu

import cgp
import gates as gates_mod
import tasks

OUT = pathlib.Path(__file__).resolve().parent / "runs" / "_cal"
GATES = gates_mod.build_set("and,nand,or,nor")
LADDER = ("pair_full", "pair_partial", "pair_zero", "add4", "mult4", "retina_x2")
N_NULL = 300
NODES = 100
NL = chr(10)

# The three tasks that form the ground-truth contrast, ordered by how much sharing
# between the two outputs the TARGET forces. SPEC should fall as forced sharing rises.
GT = ("pair_zero", "pair_partial", "pair_full")
GT_NOTE = {"pair_zero": "O1=L, O2=R -- outputs need share NOTHING",
           "pair_partial": "O1=L, O2=L XOR R -- partial overlap",
           "pair_full": "O1=O2=L -- outputs MUST share everything"}


def null_spec(task):
    """SPEC on unevolved random genotypes. No search -- the substrate's own baseline."""
    ni, no = tasks.n_inputs(task), tasks.n_outputs(task)
    im, mask = tasks.input_masks(task), tasks.full_mask(task)
    rnd = random.Random(12345)
    vals = []
    for _ in range(N_NULL):
        g = cgp.random_genotype(rnd, NODES, ni, no, len(GATES), 2)
        s, _, _ = cgp.specialisation(g, GATES, im, mask, ni)
        if s == s:                      # drop NaN (nothing influences anything)
            vals.append(s)
    return vals


def results(task):
    for d in OUT.glob("*cal-" + task):
        for f in sorted(d.glob("*_result.json")):
            yield json.loads(f.read_text(encoding="utf-8"))


def evolved(task, solved_only=True):
    """(SPEC values, n_solved, n_total) from the calibration run."""
    vals, n_solved, n_total = [], 0, 0
    for r in results(task):
        n_total += 1
        ok = r["solved_gen"] >= 0
        n_solved += ok
        if (ok or not solved_only) and r.get("spec") not in ("", None):
            vals.append(float(r["spec"]))
    return vals, n_solved, n_total


def med(v):
    return statistics.median(v) if v else float("nan")


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def leg1():
    print(NL + "=" * 88)
    print("LEG 1 -- RANDOM-GENOTYPE NULL: what does an UNEVOLVED circuit score?")
    print("=" * 88)
    print("  {:<14} {:>3} {:>4} {:>9} {:>8} {:>8} {:>17} {:>8}".format(
        "task", "in", "out", "median", "p25", "p75", "evolved(solved)", "shift"))
    out = {}
    for t in LADDER:
        n = null_spec(t)
        ev, ns, nt = evolved(t)
        out[t] = (n, ev)
        shift = (med(ev) - med(n)) if ev else float("nan")
        print("  {:<14} {:>3} {:>4} {:>9.4f} {:>8.3f} {:>8.3f} {:>17.4f} {:>+8.3f}".format(
            t, tasks.n_inputs(t), tasks.n_outputs(t),
            med(n), q(n, .25), q(n, .75), med(ev) if ev else float("nan"), shift))
    print(NL + "  A null near 1.0 means the statistic has almost no room to rise above")
    print("  chance -- 'more specialised than random' would be nearly unmeasurable.")
    return out


def leg2():
    print(NL + "=" * 88)
    print("LEG 2 -- GROUND TRUTH: same shape (8 in, 2 out), different forced sharing")
    print("=" * 88)
    vals = {}
    for t in GT:
        ev, ns, nt = evolved(t)
        vals[t] = ev
        print(NL + "  {:<13} {}".format(t, GT_NOTE[t]))
        print("    solved {}/{}   median SPEC {:>7.4f}   IQR [{:.3f}, {:.3f}]   n={}".format(
            ns, nt, med(ev), q(ev, .25), q(ev, .75), len(ev)))

    print(NL + "  Pairwise (one-sided Mann-Whitney, less-sharing > more-sharing):")
    for a, b in (("pair_zero", "pair_partial"), ("pair_partial", "pair_full"),
                 ("pair_zero", "pair_full")):
        if not vals[a] or not vals[b]:
            continue
        u, p = mannwhitneyu(vals[a], vals[b], alternative="greater")
        print("    {:<13} > {:<13}  diff {:>+7.4f}  U={:>9,.0f}  p={:.4g}  {}".format(
            a, b, med(vals[a]) - med(vals[b]), u, p,
            "ORDERED" if p < 0.05 else "NOT ORDERED"))
    return vals


def unsolved_check():
    """SPEC on circuits that never solved -- is the statistic reading competence?"""
    print(NL + "=" * 88)
    print("CONFOUND CHECK -- SPEC on UNSOLVED circuits (a high value here is a warning)")
    print("=" * 88)
    print("  {:<14} {:>9} {:>10} {:>11} {:>10}".format(
        "task", "solved n", "med SPEC", "unsolved n", "med SPEC"))
    for t in LADDER:
        sol, _, _ = evolved(t, solved_only=True)
        uns = [float(r["spec"]) for r in results(t)
               if r["solved_gen"] < 0 and r.get("spec") not in ("", None)]
        print("  {:<14} {:>9} {:>10.4f} {:>11} {:>10.4f}".format(
            t, len(sol), med(sol), len(uns), med(uns)))


def plot(nulls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    data, labs, cols = [], [], []
    for t in LADDER:
        n, ev = nulls[t]
        data += [n, ev if ev else [float("nan")]]
        labs += [t + NL + "random", t + NL + "evolved"]
        cols += ["#95a5a6", "#2c6fbb"]
    bp = ax1.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.62,
                     medianprops=dict(color="black", lw=1.5), showfliers=False)
    for box, c in zip(bp["boxes"], cols):
        box.set_facecolor(c)
        box.set_alpha(0.5)
    ax1.tick_params(axis="x", labelsize=6.5)
    ax1.set_ylabel("SPEC")
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_title("Leg 1: the random null sits near CEILING for few-output tasks",
                  fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    gt = [nulls[t][1] if nulls[t][1] else [float("nan")] for t in GT]
    bp2 = ax2.boxplot(gt, patch_artist=True, widths=0.55,
                      tick_labels=[t + NL + GT_NOTE[t].split(" -- ")[0] for t in GT],
                      medianprops=dict(color="black", lw=1.6), showfliers=False)
    for box, c in zip(bp2["boxes"], ("#27ae60", "#e67e22", "#c0392b")):
        box.set_facecolor(c)
        box.set_alpha(0.5)
    ax2.axhline(statistics.median(nulls["pair_zero"][0]), ls="--", color="0.4", lw=1.2)
    ax2.text(0.62, statistics.median(nulls["pair_zero"][0]) + 0.02, "random null",
             fontsize=8, color="0.4")
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.set_ylabel("SPEC (solved circuits only)")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_title("Leg 2: does SPEC track ground-truth forced sharing?", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("SPEC metric calibration -- AI-designed, see CALIBRATION.md", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=150)
    print(NL + "figure -> " + str(OUT / "calibration.png"))


def main():
    if not OUT.exists():
        print("no results under " + str(OUT) + " -- run run_calibration.py first")
        return 1
    nulls = leg1()
    leg2()
    unsolved_check()
    plot(nulls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
