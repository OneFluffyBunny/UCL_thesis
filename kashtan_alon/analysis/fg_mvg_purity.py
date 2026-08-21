"""MVG vs Fixed Goal, across evolution: left/right circuit purity beside accuracy.

Kashtan-Alon's own claim is about Newman Q; this asks the same question with
qmetrics METRIC 4 (circuit purity), which unlike Q knows WHICH side of the retina
each neuron reads from. The accuracy panel sits next to it on purpose -- MVG and FG
do not reach the same fitness, so a purity gap must always be read against the
performance gap that comes with it.

Runs both arms at the paper preset (pop 600, 25k gens, goal switch every 20) into a
SEPARATE out-dir, so the committed `runs/` paper results are never overwritten,
then reads the per-generation CSVs back and plots mean +- 1 SD across seeds.

Finished seeds are skipped (their result.json is the completion marker), so this is
re-runnable and resumable: after a crash it costs only the unfinished seeds, and
once everything is done it just redraws.

    conda run -n lndp python kashtan_alon/analysis/fg_mvg_purity.py --n-seeds 5
    ... --arm mvg          # one arm only; the two share no state, so two processes
    ... --plot-only        # redraw from existing CSVs, no training
    ... --smoke            # pop 60 / 400 gens, checks the pipeline

Runs from any working directory. Training is ~10.5 min per seed per arm.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/: train, model, tasks
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

import train as T
import model as M
import tasks
from qmetrics import open_file

ARMS = {                       # label -> (train.py overrides, colour)
    "MVG (AND <-> OR, every 20 gens)": (dict(mvg=True, operation="and"), "#dc2626"),
    "FG (L AND R, fixed)": (dict(mvg=False, operation="and"), "#2563eb"),
}

PANELS = [("purity", "circuit purity of the champion brain",
           "Left/right circuit purity", (0.0, 1.0)),
          ("best_fit", "accuracy of the champion brain (fraction of 256 patterns)",
           "Accuracy on the current goal", (0.4, 1.02))]


def paper_preset(cli):
    """The run_paper.py parameter block, with the knobs this script exposes."""
    p = dict(task="retina", layers="8,8,4,2,1", input_encoding="binary",
             pop=600, generations=25000, init_density=0.5,
             n_elite=150, pc=0.5, pm=0.5, fitness="raw",
             switch_interval=20, mvg_ops="and,or",
             weighted_q=False, qm_nrand=1000, log_interval=cli.log_interval,
             target=1.0, out_dir=cli.out_dir, viz=False, open_img=False,
             resume=True, checkpoint_interval=1000)
    if cli.smoke:
        p.update(pop=60, generations=400, n_elite=15, qm_nrand=50,
                 log_interval=10, checkpoint_interval=200)
    if cli.generations:
        p["generations"] = cli.generations
    if cli.pop:
        p["pop"] = cli.pop
    return p


def arm_args(preset, overrides):
    args = T.build_parser().parse_args([])
    for k, v in {**preset, **overrides}.items():
        setattr(args, k, v)
    return args


def train_arm(cli, args):
    cfg = M.NetConfig(layers=tuple(int(x) for x in args.layers.split(",")))
    X_bits = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    for i in range(cli.n_seeds):
        seed = cli.seed + i
        done = os.path.join(cli.out_dir, f"{T.run_name_for(args, seed)}_result.json")
        if os.path.exists(done):
            with open(done) as f:
                r = json.load(f)
            print(f"[seed {seed}] already complete (best {r['best_fit']:.3f}) -> skip")
            continue
        T.train_seed(cfg, X_bits, X_bits, args, seed, open_after=False)


def read_columns(out_dir, args, n_seeds, seed0, cols):
    """-> (gens, {col: matrix [seed, gen]}). Seeds truncated to a common grid."""
    per_seed = []
    for i in range(n_seeds):
        path = os.path.join(out_dir, f"{T.run_name_for(args, seed0 + i)}_log.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or any(c not in rows[0] for c in cols):
            print(f"  [{os.path.basename(path)}: missing a column -- rerun it]")
            continue
        per_seed.append(rows)
    if not per_seed:
        return np.array([]), {c: np.zeros((0, 0)) for c in cols}
    n = min(len(s) for s in per_seed)
    gens = np.array([int(r["gen"]) for r in per_seed[0][:n]])
    mats = {c: np.array([[float(r[c]) for r in s[:n]] for s in per_seed])
            for c in cols}
    return gens, mats


def smooth(y, w):
    """Centred moving average; MVG's per-phase sawtooth otherwise buries the trend."""
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    pad = np.r_[np.full(w // 2, y[0]), y, np.full(w - w // 2 - 1, y[-1])]
    return np.convolve(pad, k, mode="valid")


def plot(cli, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(9.0 * len(PANELS), 6.0))
    n_seeds = {len(m["purity"]) for _, _, m in curves.values() if len(m["purity"])}
    for ax, (col, ylab, ptitle, ylim) in zip(axes, PANELS):
        for label, (colour, gens, mats) in curves.items():
            mat = mats[col]
            if not len(mat):
                continue
            mean = mat.mean(axis=0)
            if len(mat) > 1:
                sd = mat.std(axis=0, ddof=1)
                ax.fill_between(gens, smooth(mean - sd, cli.smooth),
                                smooth(mean + sd, cli.smooth), color=colour,
                                alpha=0.15, lw=0)
            ax.plot(gens, mean, color=colour, lw=0.6, alpha=0.25)
            ax.plot(gens, smooth(mean, cli.smooth), color=colour, lw=2.0,
                    label=f"{label}   (final {mean[-1]:.3f})")
        if col == "purity":
            ax.axhline(0.5, color="#9ca3af", lw=1.0, ls=":")
        ax.set_xlabel("generation")
        ax.set_ylabel(ylab)
        ax.set_title(ptitle, fontsize=13)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    seeds = f"{min(n_seeds)} seeds per arm" if n_seeds else "no data"
    fig.suptitle("Kashtan-Alon retina 8-8-4-2-1 (+-1 weights, threshold units): "
                 "MVG vs Fixed Goal across evolution\n"
                 f"mean over {seeds}, shaded +- 1 SD across seeds  ·  "
                 f"bold line smoothed over {cli.smooth} log points "
                 f"(= {cli.smooth * cli.log_interval} generations), faint line raw",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(cli.out_dir, exist_ok=True)
    out = os.path.join(cli.out_dir, "fg_vs_mvg_purity.png")
    fig.savefig(out, dpi=170)
    print(f"\nwrote {out}")
    if cli.open:
        open_file(out)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=0, help="0 = paper's 25000")
    ap.add_argument("--pop", type=int, default=0, help="0 = paper's 600")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=51, help="moving average, log points")
    ap.add_argument("--out-dir", default=str(_HERE.parents[1] / "runs_purity"))
    ap.add_argument("--smoke", action="store_true", help="pop 60 / 400 gens")
    ap.add_argument("--plot-only", action="store_true", help="skip training, redraw")
    ap.add_argument("--arm", choices=["both", "mvg", "fg"], default="both",
                    help="train one arm only -- the arms share no state, so running "
                         "them as two processes halves wall-clock")
    ap.add_argument("--no-open", dest="open", action="store_false", default=True,
                    help="do not launch the finished picture")
    return ap


def main():
    cli = build_parser().parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    preset = paper_preset(cli)
    cols = [c for c, _, _, _ in PANELS]
    curves = {}
    for label, (overrides, colour) in ARMS.items():
        args = arm_args(preset, overrides)
        if cli.arm != "both" and (cli.arm == "mvg") != overrides["mvg"]:
            continue
        print(f"\n========== {label} ==========")
        if not cli.plot_only:
            train_arm(cli, args)
        gens, mats = read_columns(cli.out_dir, args, cli.n_seeds, cli.seed, cols)
        curves[label] = (colour, gens, mats)

    print("\n================ SUMMARY (0/25/50/75/100% of the run) ================")
    for label, (_, _, mats) in curves.items():
        for col in cols:
            mat = mats[col]
            if not len(mat):
                print(f"  {label}: no data")
                break
            q = [mat[:, int(f * (mat.shape[1] - 1))].mean() for f in (0, .25, .5, .75, 1)]
            print(f"  {label:34s} {col:9s} " + "  ".join(f"{v:.3f}" for v in q)
                  + f"   ({len(mat)} seeds)")
    plot(cli, curves)
    return 0


if __name__ == "__main__":
    sys.exit(main())
