"""Per-generation view inside fixed windows: FG seed 0 vs MVG seed 0.

    python fig_switch_window.py --root ../experiment_1/runs/fgmvg
    python fig_switch_window.py --root ../experiment_2/runs/fgmvg --windows 100:300,1000:1200
    python fig_switch_window.py --root ../experiment_1/runs/fgmvg --constraint nobudget

`kashtan_alon/analysis/switch_window.py`, adapted to this framework. THREE
METRICS (rows) x N WINDOWS (columns), one FG seed and one MVG seed overlaid in
every panel, rows sharing a y scale so one window can be read against another:

  1  CHAMPION accuracy ON THE GOAL ACTIVE THAT GENERATION  (KA's `best_fit`)
  2  POPULATION MEAN accuracy, same goal                   (KA's `mean_fit`)
  3  CHAMPION `lr_r`, left/right assortativity             (KA's `Q`)

`lr_r` replaces KA's circuit purity. Purity is not applicable here: it is only
defined on a DAG (`qmetrics.circuit_purity` raises on a cycle) and every brain in
`experiments/` is recurrent, and purity is no longer offered as evidence of
modularity anyway. `lr_r` is this study's primary modularity metric, and it is
the column KA's `Q` corresponds to.

THE CHAMPION ROWS NEED NO REPLAY, unlike KA. KA's archived logs sampled every 10
generations against a 20-generation switch, giving two points per goal phase, so
every row there had to be replayed. These runs were archived with
`archive_interval: 1`: `champions.npz` holds the champion genome of EVERY
generation, so rows 1 and 3 can be read straight out of any run's archive --
201 points in a 200-generation window.

THE POPULATION MEAN STILL NEEDS A REPLAY, exactly as it did in KA. It cannot be
recovered from an archive of champions, and log.csv only carries it at
`log_interval` -- under MVG that is one row per goal epoch, 10 points in a
200-generation window. So `dense_replay.py` re-runs the arm with
`--dense-log lo:hi,...`, the `--dense-log` flag being itself a port of KA's.

WHICH RUN EACH ROW COMES FROM. If a replay exists under `<root>_dense` this
figure takes ALL THREE ROWS from it; otherwise it falls back to the archived
study and says so, with a coarse row 2. The rows are never mixed across the two,
because a replay is not the archived trajectory: it is bit-identical to another
replay but diverges from the archived run from generation 0, float
reduction-order nondeterminism that CMA-ES amplifies within a few dozen
generations (measured -- see `dense_replay.py`). A replay is therefore a second
sample of the same arm, fine to read a switch off, wrong to graft onto the
archived champion's curve. The subtitle names the source.

`mean_acc` is plotted AS LOGGED, i.e. on whichever goal was active in that row,
exactly as KA plotted `mean_fit`; the shaded epochs say which goal each point
belongs to.

Windows default to [100, 300] and [1000, 1200]: early enough that the population
is still climbing, and one order of magnitude apart, which is the comparison the
figure exists to make -- does the population re-adapt to a returning goal FASTER
once evolution has had time to build structure? The printed per-window recovery
statistic quantifies it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from runs_io import find_runs                        # noqa: E402
from shared_brain_metrics import left_right_split    # noqa: E402

ARM_STYLE = {
    "fg":  dict(color="#2563eb", ls="-", label="FG (L AND R)"),
    "mvg": dict(color="#dc2626", ls="-", label="MVG (AND <-> OR, every 20 gens)"),
}
OFF_GOAL_SHADE = "#9ca3af"      # the epochs running the OTHER goal
DEFAULT_WINDOWS = "100:300,1000:1200"


def parse_windows(spec):
    """"100:300,1000:1200" -> [(100, 300), (1000, 1200)]."""
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition(":")
        out.append((int(lo), int(hi)))
        if out[-1][1] <= out[-1][0]:
            raise SystemExit(f"empty window {part!r}")
    return out


def champion_window(run, lo, hi, threshold):
    """Score the champion at EVERY archived generation in [lo, hi].

    -> dict of arrays: gen, acc (on the ACTIVE goal), lr_r, op.

    ACCURACY IS ON THE GOAL THAT WAS ACTIVE IN THAT GENERATION, which is KA's
    `best_fit` column: `gen_best` is always the champion's fitness on `cur_op`.
    So once the goal flips to OR this is the new champion measured on OR, and the
    curve asks "how good is the population's best network at the job it is
    currently being selected for?". The archive keys accuracy by goal, so the
    active-goal value is a lookup, not a re-evaluation.

    Reading it on the REFERENCE goal instead turns the panel into a square wave
    between 0.81 and 0.31 -- true, but it is a statement about how wrong an
    OR-solver is on AND, not about progress, and it hides the switch cost
    entirely. The end-of-run table is where reference-goal accuracy belongs.
    """
    arc = run.archive()
    if arc is None:
        raise SystemExit(f"{run.dir} has no champions.npz "
                         f"(the run needs --archive-interval > 0)")
    gens = arc["gen"]
    sel = np.where((gens >= lo) & (gens <= hi))[0]
    if len(sel) == 0:
        raise SystemExit(f"{run.dir}: no archived generations in [{lo}, {hi}] "
                         f"(the run reached generation {gens[-1]})")
    n_in, n_hid, n_out = run.shape
    ops = arc["op"][sel]
    acc = np.array([float(arc["acc"][op][i]) for op, i in zip(ops, sel)])
    lr = np.empty(len(sel))
    for k, i in enumerate(sel):
        w = run.weights_from_flat(arc["flat"][i])
        # `left_right_split`, not score_weights(lr=True): identical `lr_r`, but it
        # skips the q_max rewiring, which is 60-140x the cost and unused here.
        lr[k] = left_right_split(w, n_in, n_hid, n_out, threshold=threshold)[0]
    return {"gen": gens[sel], "acc": acc, "lr_r": lr, "op": ops}


def population_window(run, lo, hi):
    """`mean_acc` from log.csv in [lo, hi], as logged (on the active goal)."""
    lg = run.log()
    if not lg or "mean_acc" not in lg:
        return {"gen": np.array([]), "mean_acc": np.array([])}
    keep = (lg["gen"] >= lo) & (lg["gen"] <= hi)
    return {"gen": lg["gen"][keep], "mean_acc": lg["mean_acc"][keep]}


def phase_stats(d, ref_op, switch, key="acc"):
    """Per GOAL EPOCH in the window: the trough at the switch, the peak reached
    inside the epoch, and how many generations it took to cover 90% of that
    climb. -> (troughs, peaks, times).

    EVERY epoch counts, not only the reference-goal ones, because `d[key]` is
    measured on the goal that was active: right after a switch the champion is
    being scored on a job it was not selected for, so the climb is exactly the
    re-adaptation KA's two windows exist to compare. Restricting to AND epochs
    would throw away half the switches for no reason.

    Recovery is measured against the epoch's OWN peak, not a fixed accuracy
    threshold: early in training the population never reaches a fixed 0.9 at all,
    so a fixed threshold censors every early epoch at the epoch length and hides
    the very comparison the two windows exist to make. (KA's reasoning, kept.)

    Returns empty arrays for a fixed-goal run: one goal, no switches, nothing to
    recover from.
    """
    ops = d["op"]
    if len(set(ops.tolist())) < 2:
        return np.array([]), np.array([]), np.array([])
    starts = [i for i in range(len(ops)) if i == 0 or ops[i] != ops[i - 1]]
    bounds = starts[1:] + [len(ops)]
    troughs, peaks, times = [], [], []
    for i, end in zip(starts, bounds):
        seg = np.asarray(d[key][i:i + switch], dtype=float)
        # Skip any epoch the window cut short, at either edge. A window that
        # opens mid-epoch gives a first "trough" that is just where the window
        # starts, and a window that closes mid-epoch gives a peak that is an
        # artifact of where it stops. A full epoch is `switch` generations long.
        if end - i < switch or len(seg) < switch:
            continue
        trough, peak = float(seg[0]), float(seg.max())
        thresh = trough + 0.9 * (peak - trough)
        troughs.append(trough)
        peaks.append(peak)
        times.append(int(np.argmax(seg >= thresh)))
    return np.array(troughs), np.array(peaks), np.array(times)


def shade_goal(ax, d, ref_op, switch, lo, hi):
    """Tint the generations that ran the OTHER goal, straight from the logged op.

    Per generation, not per epoch: the archive records the active goal of every
    generation, so the stripes are the real epoch boundaries rather than a
    reconstruction from `switch_interval`.
    """
    off = d["op"] != ref_op
    for g, is_off in zip(d["gen"], off):
        if is_off:
            ax.axvspan(g - 0.5, g + 0.5, color=OFF_GOAL_SHADE, alpha=0.16, lw=0)
    for k in range(1, len(off)):
        if off[k] != off[k - 1]:
            ax.axvline(d["gen"][k] - 0.5, color="#6b7280", lw=0.7, ls=":")
    return bool(off.any())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--constraint", choices=["budget", "nobudget"], default="budget")
    p.add_argument("--windows", default=DEFAULT_WINDOWS,
                   help=f"comma-separated lo:hi generation windows "
                        f"(default {DEFAULT_WINDOWS})")
    p.add_argument("--dense-root", default=None,
                   help="where dense_replay.py wrote its per-generation runs "
                        "(default: <root>_dense). Used for ALL THREE rows when "
                        "present; --no-dense ignores it.")
    p.add_argument("--no-dense", action="store_true",
                   help="read the archived study instead of the replay, leaving "
                        "the population row at its coarse log_interval cadence")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    windows = parse_windows(args.windows)
    root = os.path.normpath(args.root)
    want = [f"{args.constraint}_fg", f"{args.constraint}_mvg"]

    # Prefer the dense replay, but only if it covers EVERY arm asked for: half a
    # figure from the replay and half from the archive would be two runs drawn as
    # one. See the module docstring.
    dense_root = args.dense_root or (root.rstrip(os.sep) + "_dense")
    dense = ({r.arm: r for r in find_runs(dense_root) if r.seed == args.seed}
             if (os.path.isdir(dense_root) and not args.no_dense) else {})
    if all(a in dense for a in want):
        runs, source = dense, f"dense replay ({os.path.basename(dense_root)})"
    else:
        runs = {r.arm: r for r in find_runs(root) if r.seed == args.seed}
        source = ("archived study — population row is COARSE; run "
                  "dense_replay.py for a per-generation one")
    have = [a for a in want if a in runs]
    if not have:
        raise SystemExit(f"no {args.constraint} runs for seed {args.seed} under {root}")
    print(f"source: {source}")

    ref = runs[have[0]]
    thr = ref.run["prune_threshold"]
    switch = ref.run["switch_interval"]
    ref_op = ref.reference_op

    champ, pop = {}, {}
    for arm in have:
        for lo, hi in windows:
            champ[(arm, lo)] = champion_window(runs[arm], lo, hi, thr)
            pop[(arm, lo)] = population_window(runs[arm], lo, hi)
            print(f"  {arm} [{lo}, {hi}]: "
                  f"{len(champ[(arm, lo)]['gen'])} generations scored, "
                  f"{len(pop[(arm, lo)]['gen'])} population rows")

    # Both accuracy rows are on the ACTIVE goal, so neither may be labelled with
    # the reference goal's name: inside a shaded epoch the number is accuracy on
    # the other task entirely.
    # Short labels, as in `fig_progress.py`. Both accuracy rows are scored on the
    # goal ACTIVE that generation (not the reference goal) -- hence "active goal".
    ROWS = [("acc", "champion accuracy (active goal)"),
            ("mean_acc", "population mean accuracy"),
            ("lr_r", "champion L/R modularity ($lr_r$)")]

    def row_ylim(key, pad=0.05):
        """Limits from the row's OWN data across every window, not a fixed range.

        A fixed (0.48, 1.02) clipped the MVG champion's accuracy on AND, which
        falls well BELOW chance inside an OR epoch -- an OR-solver is actively
        wrong on AND, not merely uninformed -- so the clipped trace read as
        spikes touching the axis instead of as a real drop. The same fixed range
        then left the population row, which lives inside eight accuracy points,
        almost entirely empty. Sharing one limit per ROW is what makes a window
        comparable to a window; sharing it across rows never meant anything.
        """
        src = pop if key == "mean_acc" else champ
        vals = np.concatenate([np.asarray(src[(a, lo)][key], dtype=float)
                               for a in have for lo, _ in windows])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return None
        lo_v, hi_v = float(vals.min()), float(vals.max())
        span = max(hi_v - lo_v, 0.05)
        return (lo_v - pad * span, hi_v + pad * span)

    fig, axes = plt.subplots(3, len(windows),
                             figsize=(7.6 * len(windows), 10.4), sharey="row",
                             squeeze=False)
    mvg_arm = next((a for a in have if a.endswith("_mvg")), None)
    ylims = [row_ylim(k) for k, _ in ROWS]
    # Is the population row actually per-generation? Measured, not assumed from
    # the source name: a replay whose windows were narrower than these would
    # still leave gaps, and the row must not claim a density it does not have.
    dense_pop = all(len(pop[(a, lo)]["gen"]) >= 0.9 * (hi - lo)
                    for a in have for lo, hi in windows)

    for ci, (lo, hi) in enumerate(windows):
        for ri, (key, ylab) in enumerate(ROWS):
            ax = axes[ri][ci]
            ylim = ylims[ri]
            if mvg_arm:
                shade_goal(ax, champ[(mvg_arm, lo)], ref_op, switch, lo, hi)
            for arm in have:
                st = ARM_STYLE["mvg" if arm.endswith("_mvg") else "fg"]
                if key == "mean_acc":
                    d = pop[(arm, lo)]
                    # Markers only when the row is coarse: they say "these are the
                    # only points there are". On a dense replay there are 201 of
                    # them per window and markers would just be a thick line.
                    extra = {} if dense_pop else dict(marker="o", ms=3.4)
                    ax.plot(d["gen"], d["mean_acc"], lw=1.2, **extra, **st)
                else:
                    d = champ[(arm, lo)]
                    ax.plot(d["gen"], d[key], lw=1.5, **st)
            ax.set_xlim(lo, hi)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
            # Rows share a scale, but every panel keeps its own tick numbers so a
            # column can be read on its own without tracking back to column 1.
            ax.tick_params(labelleft=True)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            if ri == 0:
                # 0.750 is where every one-eye solution caps, so it says whether
                # the plateau the champion sits on is a real two-eye solution.
                ax.axhline(0.75, color="0.4", lw=0.8, ls="--")
                ax.set_title(f"generations [{lo}, {hi}]", fontsize=12)
                ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
            if ri == 1 and ci == 0 and not dense_pop:
                ax.text(0.012, 0.03, "markers = the only logged rows there are; "
                        "run dense_replay.py for a per-generation population",
                        transform=ax.transAxes, fontsize=7.5, color="0.3",
                        va="bottom",
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
            if ri == 2:
                # 0 is not a cosmetic gridline: lr_r is an assortativity, so 0 IS
                # chance at the observed degrees and below 0 is anti-associated.
                ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
                ax.set_xlabel("generation")

    # Quantify the thing the two windows exist to compare.
    if mvg_arm:
        print(f"\nMVG re-adaptation after a goal switch, per epoch "
              f"(accuracy on the newly active goal):")
        for lo, hi in windows:
            c = champ[(mvg_arm, lo)]
            series = [("champion  ", c)]
            # KA's headline was the POPULATION mean: the champion is a max over
            # the population, so it barely dents at a switch, while the mean
            # craters. Only a dense replay lines the mean up with `op`.
            pm = pop[(mvg_arm, lo)]
            if dense_pop and len(pm["gen"]) == len(c["gen"]):
                series.append(("population", {"op": c["op"],
                                              "mean_acc": pm["mean_acc"]}))
            for name, d in series:
                key = "mean_acc" if name.strip() == "population" else "acc"
                trough, peak, t = phase_stats(d, ref_op, switch, key=key)
                if len(t) == 0:
                    print(f"  [{lo}-{hi}] {name}: no complete epoch in the window")
                    continue
                print(f"  [{lo}-{hi}] {name}: {np.mean(trough):.3f} at the switch "
                      f"-> {np.mean(peak):.3f} within the epoch, 90% of that "
                      f"recovery in {np.mean(t):.1f} gens (n={len(t)})")
                print(f"                 per-epoch: {t.tolist()}")

    # Same two-line shape as `fig_progress.py` / KA's `fg_mvg_purity.py`: one
    # headline, one line of what the curves ARE. Source, threshold and sampling
    # detail live in the promoted folder's README and in RESULTS.md.
    con = {"budget": "synaptic budget",
           "nobudget": "no budget (ablation)"}[args.constraint]
    what = (f"seed {args.seed} replay, every generation" if dense_pop
            else f"seed {args.seed}, population row at log interval only")
    fig.suptitle(f"{ref.encoding} encoding — FG vs MVG — {con}\n"
                 f"{what} — shaded: MVG epochs on {'OR' if ref_op == 'and' else 'the other goal'}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = args.out or os.path.join(
        args.root, f"switch_window_{args.constraint}_seed{args.seed}.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
