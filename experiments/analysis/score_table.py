"""The 4-group x 4-metric table — the `kashtan_alon/` Run 8 equivalent.

    python score_table.py --root ../experiment_1/runs/fgmvg
    python score_table.py --root ../experiment_2/runs/fgmvg --n-rand 200

Scores the GOAL-MATCHED champion of every seed (`matched`: the last champion
selected under the reference goal), so an FG number and an MVG number are the
same measurement. Writes `metrics_per_seed.csv` and `metrics_summary.json` next
to the runs, and prints a markdown table ready to paste into RESULTS.md.

Read the four metrics in this order, and read the density first:

  density   If an arm converged near 100%, its modularity numbers are not "low",
            they are UNDEFINED -- a complete graph has no communities to find and
            no sparser null to compare against. Say unanswerable, not unmodular.
  lr        PRIMARY. Modularity at the PLANTED left/right split, with a p-value
            from a degree-preserving null. No partition search, so nothing to
            fail; `lr_r` is the left/right correlation KA reports as `r`.
  purity    PRIMARY. Mean over hidden neurons of how one-sided each neuron's
            ancestry is, on the time-unrolled graph. Always defined.
  q, q_m    SECONDARY. A DISCOVERED partition can be real and irrelevant at once
            (experiment_1/RESULTS.md has Q = 0.20 at p = 0.87 on a partition
            unrelated to left/right). Q_m additionally stops discriminating above
            ~50% density, by KA's own data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from runs_io import find_runs                       # noqa: E402
from shared_brain_metrics import score_weights      # noqa: E402

ARM_ORDER = ["budget_fg", "budget_mvg", "nobudget_fg", "nobudget_mvg"]
ARM_LABEL = {"budget_fg": "budget (constrained) | FG",
             "budget_mvg": "budget (constrained) | MVG",
             "nobudget_fg": "no budget (ablation) | FG",
             "nobudget_mvg": "no budget (ablation) | MVG"}


def mean_sd(xs):
    xs = [x for x in xs if x == x]                  # drop NaN
    if not xs:
        return float("nan"), float("nan"), 0
    return (stats.mean(xs), stats.stdev(xs) if len(xs) > 1 else 0.0, len(xs))


def fmt(xs, prec=3):
    m, s, n = mean_sd(xs)
    if n == 0:
        return "n/a"
    return f"{m:.{prec}f}+-{s:.{prec}f}" + ("" if n == len(xs) else f" ({n}/{len(xs)})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="directory holding the arm subdirectories")
    p.add_argument("--tag", default="matched",
                   help="which champion to score (matched = goal-matched, the right one)")
    p.add_argument("--n-rand", type=int, default=200,
                   help="randomisations for Q_m and the left/right p-value")
    p.add_argument("--threshold", type=float, default=None,
                   help="|w| cut defining an edge; default = each run's --prune-threshold")
    args = p.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"no completed runs under {args.root}")
    print(f"scoring {len(runs)} runs from {args.root} (tag={args.tag}, n_rand={args.n_rand})\n")

    rows = []
    for r in runs:
        n_in, n_hid, n_out = r.shape
        thr = args.threshold if args.threshold is not None else r.run["prune_threshold"]
        s = score_weights(r.weights(args.tag), n_in, n_hid, n_out,
                          rnn_iters=r.cfg.rnn_iters, threshold=thr,
                          n_rand=args.n_rand, seed=r.seed)
        s.update(arm=r.arm, seed=r.seed, encoding=r.encoding,
                 acc=r.accuracy(args.tag), reference_op=r.reference_op,
                 gens=r.rj.get("gens_run"), matched_gen=r.rj.get("matched_gen"),
                 run_dir=os.path.relpath(r.dir, args.root))
        # accuracy on the OTHER goals, which is how you see whether an MVG run
        # holds both targets or just swaps between them
        for op, a in (r.rj.get("acc_by_op", {}).get(args.tag) or {}).items():
            s[f"acc_{op}"] = a
        rows.append(s)
        print(f"  {r.arm:14s} seed{r.seed}  acc {s['acc']:.3f}  dens {s['density']:5.1f}%  "
              f"lr {s.get('lr', float('nan')):+.3f}  purity {s['purity']:.3f}  "
              f"q {s.get('q', float('nan')):.3f}  q_m {s.get('q_m', float('nan')):+.3f}")

    # --- per-seed CSV ---
    keys = sorted({k for r in rows for k in r if not isinstance(r[k], (dict, list))})
    lead = ["encoding", "arm", "seed", "acc", "density", "lr", "lr_r", "lr_p",
            "purity", "q", "q_m"]
    keys = lead + [k for k in keys if k not in lead]
    csv_path = os.path.join(args.root, "metrics_per_seed.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- markdown summary ---
    ops = sorted({k[4:] for r in rows for k in r if k.startswith("acc_")})
    head = ["condition", "arm", "n"] + [f"acc ({o.upper()})" for o in ops] + \
           ["density %", "LR (primary)", "purity (primary)", "Q", "Q_m"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    summary = {}
    for arm in ARM_ORDER:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        cond, side = ARM_LABEL[arm].split(" | ")
        cells = [cond, side, str(len(sub))]
        cells += [fmt([r.get(f"acc_{o}", float("nan")) for r in sub]) for o in ops]
        cells += [fmt([r["density"] for r in sub], 1),
                  fmt([r.get("lr", float("nan")) for r in sub]),
                  fmt([r["purity"] for r in sub]),
                  fmt([r.get("q", float("nan")) for r in sub]),
                  fmt([r.get("q_m", float("nan")) for r in sub])]
        lines.append("| " + " | ".join(cells) + " |")
        summary[arm] = {k: mean_sd([r.get(k, float("nan")) for r in sub])[:2]
                        for k in ("acc", "density", "lr", "lr_r", "lr_p", "purity", "q", "q_m")}
        summary[arm]["n"] = len(sub)
        # how many seeds actually beat their own null at the planted split
        ps = [r.get("lr_p") for r in sub if r.get("lr_p") == r.get("lr_p")]
        summary[arm]["n_significant_lr"] = sum(1 for x in ps if x is not None and x < 0.05)
        summary[arm]["n_with_p"] = len(ps)

    print("\n" + "\n".join(lines))

    print("\nSeeds significant at the planted split (p < 0.05, degree-preserving null):")
    for arm in ARM_ORDER:
        if arm in summary:
            print(f"  {arm:14s} {summary[arm]['n_significant_lr']}/{summary[arm]['n_with_p']}")

    with open(os.path.join(args.root, "metrics_summary.json"), "w") as fh:
        json.dump({"tag": args.tag, "n_rand": args.n_rand,
                   "table_markdown": "\n".join(lines), "by_arm": summary}, fh, indent=2)
    print(f"\nwrote {csv_path}")
    print(f"wrote {os.path.join(args.root, 'metrics_summary.json')}")


if __name__ == "__main__":
    main()
