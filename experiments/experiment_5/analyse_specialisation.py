"""Aggregate and test the specialisation sub-study -- see SPECIALISATION.md.

⚠️ AI-AUTHORED, like the rest of this sub-study. Read SPECIALISATION.md's banner.

    conda run -n lndp python analyse_specialisation.py

CPython, not PyPy: this needs numpy/scipy/matplotlib, none of which exist in the PyPy
venv. The *search* ran under PyPy; only the statistics run here.

Writes `runs/_spec/tidy.csv` (one row per seed per cell), prints the preregistered
tests, and draws `runs/_spec/specialisation.png`.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent / "runs" / "_spec"
SCHEDULES = ("staged", "cold")
OVERLAPS = ("full", "partial", "zero")
ENCODINGS = ("cgp4", "ecgp4", "cgpnand", "ecgpnand")
BASELINE = "cgp4"                     # the preregistered encoding


def load() -> list[dict]:
    """One row per finished seed. The cell name is the run directory's `--tag`."""
    rows = []
    for d in sorted(OUT.glob("*")):
        if not d.is_dir():
            continue
        cell = d.name.rsplit("_", 1)[-1]
        parts = cell.split("-")
        if len(parts) != 3 or parts[0] not in SCHEDULES:
            continue
        sched, ov, enc = parts
        for f in sorted(d.glob("*_result.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            rows.append(dict(
                cell=cell, schedule=sched, overlap=ov, encoding=enc,
                seed=r["seed"],
                solved=int(r["solved_gen"] >= 0),
                solved_gen=r["solved_gen"], gens_run=r["gens_run"],
                # `spec` is "" when no active node influenced any output; that is a
                # real state (see cgp.specialisation) and must not become 0.0.
                spec=(float(r["spec"]) if r.get("spec") not in ("", None) else None),
                n_spec=r.get("n_spec"), n_pleio=r.get("n_pleio"),
                active_nodes=r["active_nodes"], out_pure=r["out_pure"],
                beh_pure=r["beh_pure"], mixed=r["mixed"], seconds=r["seconds"]))
    return rows


def spec_of(rows, sched, ov, enc) -> list[float]:
    """SPEC for the SOLVED seeds of one cell.

    Conditioning on solving is what makes the comparison fair: every value here comes
    from a circuit at perfect fitness and exactly `--post-solve-gens` generations past
    its solution, so a difference cannot be a fitness difference wearing a disguise.
    It is also a selection step, which is why `solve_table` prints the rate it
    conditioned on -- if two arms of one contrast solve at different rates, the
    conditioning is itself a confound and the contrast must be read with that in mind.
    """
    return [r["spec"] for r in rows
            if r["schedule"] == sched and r["overlap"] == ov
            and r["encoding"] == enc and r["solved"] and r["spec"] is not None]


def mannwhitney(a, b):
    """(U, one-sided p for a > b, rank-biserial effect size). scipy does the test."""
    from scipy.stats import mannwhitneyu
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    u, p = mannwhitneyu(a, b, alternative="greater")
    # rank-biserial r = 2U/(n1*n2) - 1: +1 means every a beats every b, 0 means no
    # separation, -1 the reverse. Reported because a p-value on n=50 says nothing
    # about whether the difference is large enough to care about.
    return u, p, 2 * u / (len(a) * len(b)) - 1


def med(v):
    import statistics
    return statistics.median(v) if v else float("nan")


def solve_table(rows):
    print("\nSOLVE RATE per cell (the conditioning the tests below rely on)")
    print(f"{'cell':<26} {'solved':>8}  {'median gens-to-solve':>21}")
    for enc in ENCODINGS:
        for ov in OVERLAPS:
            for sched in SCHEDULES:
                sel = [r for r in rows if r["schedule"] == sched
                       and r["overlap"] == ov and r["encoding"] == enc]
                if not sel:
                    continue
                s = [r for r in sel if r["solved"]]
                g = med([r["solved_gen"] for r in s])
                print(f"{sched + '-' + ov + '-' + enc:<26} "
                      f"{len(s):>4}/{len(sel):<3} {g:>21,.0f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    rows = load()
    if not rows:
        print(f"no results under {OUT} -- run run_specialisation.py first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tidy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} seed-runs -> {OUT / 'tidy.csv'}")

    solve_table(rows)

    print("\n" + "=" * 72)
    print("PRIMARY (preregistered): staged > cold, overlap=partial, encoding=cgp4")
    print("=" * 72)
    a, b = spec_of(rows, "staged", "partial", BASELINE), spec_of(rows, "cold", "partial", BASELINE)
    u, pv, rb = mannwhitney(a, b)
    print(f"  staged  n={len(a):<3} median SPEC {med(a):.4f}")
    print(f"  cold    n={len(b):<3} median SPEC {med(b):.4f}")
    print(f"  Mann-Whitney U={u:.1f}  one-sided p={pv:.4g}  rank-biserial r={rb:+.3f}")
    print(f"  --> {'REJECT null at 0.05' if pv < 0.05 else 'FAIL TO REJECT null at 0.05'}")

    print("\n" + "=" * 72)
    print("SECONDARY (preregistered): the interaction -- gap present at `partial`,")
    print("absent at `full` and `zero`. Encoding = cgp4.")
    print("=" * 72)
    print(f"{'overlap':<10} {'staged':>8} {'cold':>8} {'gap':>8} {'p(1-sided)':>12} {'r':>8}")
    for ov in OVERLAPS:
        a, b = spec_of(rows, "staged", ov, BASELINE), spec_of(rows, "cold", ov, BASELINE)
        _, pv, rb = mannwhitney(a, b)
        print(f"{ov:<10} {med(a):>8.4f} {med(b):>8.4f} {med(a) - med(b):>+8.4f} "
              f"{pv:>12.4g} {rb:>+8.3f}")

    print("\n" + "=" * 72)
    print("EXPLORATORY -- every encoding. NO significance is claimed for these:")
    print("with 12 contrasts something will look real. Read as description only.")
    print("=" * 72)
    print(f"{'encoding':<10} {'overlap':<9} {'staged':>8} {'cold':>8} {'gap':>8} {'p':>10} {'r':>8}")
    for enc in ENCODINGS:
        for ov in OVERLAPS:
            a, b = spec_of(rows, "staged", ov, enc), spec_of(rows, "cold", ov, enc)
            if not a or not b:
                continue
            _, pv, rb = mannwhitney(a, b)
            print(f"{enc:<10} {ov:<9} {med(a):>8.4f} {med(b):>8.4f} "
                  f"{med(a) - med(b):>+8.4f} {pv:>10.4g} {rb:>+8.3f}")

    robustness(rows)

    if not args.no_plot:
        plot(rows)
    return 0


def robustness(rows):
    """Two ways the primary contrast could be an artefact rather than the mechanism.

    1. SIZE. SPEC is a ratio over influencing nodes, so if one arm simply evolves
       smaller circuits the ratio can move for reasons that have nothing to do with
       specialisation -- a 5-node circuit has fewer ways to be pleiotropic than a
       25-node one.
    2. AGE. The staged arm reaches its solution later in absolute generations (it
       spends the first 20 000 on stage 1). If SPEC drifts with how long a run took,
       the contrast is measuring duration. Pilot P3 said SPEC is flat for 100 000
       generations after a solution, but that was one arm of one cell; this checks it
       against `solved_gen` inside every cell, where a real duration effect would show
       as a consistently signed correlation.
    """
    from scipy.stats import spearmanr
    print(chr(10) + "=" * 72)
    print("ROBUSTNESS -- is the contrast really about specialisation?")
    print("=" * 72)
    print(f"{'cell':<26} {'n':>4} {'med active':>11} {'med infl':>9} "
          f"{'rho(SPEC,solved_gen)':>21} {'p':>9}")
    for enc in ENCODINGS:
        for ov in OVERLAPS:
            for sched in SCHEDULES:
                sel = [r for r in rows if r["schedule"] == sched and r["overlap"] == ov
                       and r["encoding"] == enc and r["solved"] and r["spec"] is not None]
                if len(sel) < 3:
                    continue
                infl = [int(r["n_spec"]) + int(r["n_pleio"]) for r in sel]
                sp = [r["spec"] for r in sel]
                sg = [r["solved_gen"] for r in sel]
                rho, pv = spearmanr(sp, sg)
                print(f"{sched + '-' + ov + '-' + enc:<26} {len(sel):>4} "
                      f"{med([r['active_nodes'] for r in sel]):>11.1f} {med(infl):>9.1f} "
                      f"{rho:>21.3f} {pv:>9.3g}")


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ENCODINGS), figsize=(4.2 * len(ENCODINGS), 4.4),
                             sharey=True)
    colours = {"staged": "#c0392b", "cold": "#2c6fbb"}
    for ax, enc in zip(axes, ENCODINGS):
        data, labels, cols = [], [], []
        for ov in OVERLAPS:
            for sched in SCHEDULES:
                v = spec_of(rows, sched, ov, enc)
                data.append(v if v else [float("nan")])
                labels.append(f"{ov}\n{sched}")
                cols.append(colours[sched])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.62,
                        medianprops=dict(color="black", lw=1.6))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for i, v in enumerate(data, start=1):
            if v and v[0] == v[0]:
                ax.plot([i] * len(v), v, ".", color="0.25", ms=3, alpha=0.5)
        ax.set_title(enc + ("  (preregistered)" if enc == BASELINE else "  (exploratory)"),
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("SPEC  =  specialised / (specialised + pleiotropic)")
    fig.suptitle("Does a staged second demand specialise the circuit?  "
                 "(AI-designed sub-study -- SPECIALISATION.md)", fontsize=11)
    fig.tight_layout()
    path = OUT / "specialisation.png"
    fig.savefig(path, dpi=150)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
