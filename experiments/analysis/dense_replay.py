"""Replay one FG seed and one MVG seed with PER-GENERATION logging in windows.

    python dense_replay.py --root ../experiment_1/runs/fgmvg
    python dense_replay.py --root ../experiment_1/runs/fgmvg --windows 100:300,1000:1200
    python dense_replay.py --root ../experiment_1/runs/fgmvg --verify

`kashtan_alon/analysis/dense_replay.py`, ported to experiments 1 and 2.

WHY A REPLAY AND NOT A RE-READ. `champions.npz` is archived every generation, so
the champion's accuracy and its `lr_r` are already per-generation and need no
re-run. A POPULATION statistic is different: the archive stores the CHAMPION, so
`mean_acc` cannot be recovered from it after the fact, and log.csv only has it at
`log_interval` -- which under --mvg is one row per goal epoch, i.e. 10 points in
a 200-generation window. Enough to see THAT the mean moves, not the shape of the
recovery, which is the whole point of looking at a window.

WHAT THE REPLAY IS, AND WHAT IT IS NOT. Measured on 2026-09-12, not assumed:

  * Two replays of the same arm are BIT-IDENTICAL to each other (max |dgenome|
    exactly 0.0), with and without --dense-log. So --dense-log is provably
    neutral, and a replay is regenerable: this figure can always be rebuilt.
  * A replay is NOT bit-identical to the ARCHIVED run of the same config and
    seed. It diverges from generation 0, by a little: sigma 0.097134 vs
    0.097137, population mean 0.5433 vs 0.5308, 576 vs 579 edges. That is float
    reduction-order nondeterminism -- a different XLA backend, device or thread
    count sums the population in a different order -- and CMA-ES amplifies a
    1e-6 relative difference chaotically within a few dozen generations.

So a replay is a SECOND SAMPLE OF THE SAME ARM, not a re-reading of the same
trajectory. The consequence is load-bearing: rows must not be mixed across the
two. `fig_switch_window.py` therefore takes ALL THREE of its rows from the
replay when one exists, so the panel is one coherent run, rather than stitching
a replayed population mean onto the archived champion's curve. The end-of-run
tables and the brains/aggregate figures still come from the archived study; only
the per-generation window figure is a replay, and it says so in its subtitle.

`--verify` measures this rather than asserting it, and reports the divergence.

The replay writes to a SIBLING directory, `<root>_dense`, never inside `<root>`:
`runs_io.find_runs` walks recursively and would otherwise pick these truncated
1200-generation runs up as extra seeds in every table and figure downstream.

Everything else is taken from the original run's own config.json, so the only
deliberate differences are: `generations` (stops just past the last window),
`dense_log`, `out_dir`, `archive_interval` (forced to 1, so --verify can work),
and resume/viz/open being off.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from runs_io import find_runs                        # noqa: E402

DEFAULT_WINDOWS = "100:300,1000:1200"


def parse_windows(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition(":")
        out.append((int(lo), int(hi)))
    return out


def _experiment_dir(run):
    """The experiment package a run belongs to, for sys.path and train.py.

    experiment_2 and experiment_3 share `shared_direct_model.py` but each has its
    own train.py/config.py, and the run's own directory is the only reliable
    witness of which one wrote it -- config.json records the encoding, not the
    package.
    """
    d = run.dir
    while d and os.path.basename(d) not in ("experiment_1", "experiment_2", "experiment_3"):
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit(f"cannot tell which experiment wrote {run.dir}")
        d = parent
    return d


def replay(run, windows, out_root, dry_run=False):
    """Re-run `run` up to just past the last window, logging every generation."""
    exp_dir = _experiment_dir(run)
    # train.py does `import tasks` / `from model import Genome`, so its own
    # directory has to come first on sys.path. Only ONE experiment is loaded per
    # process for exactly the reason runs_io documents: experiments 1 and 2 both
    # have a `model.py`, and importing both by name in one process silently gives
    # you whichever landed first.
    if sys.path[0] != exp_dir:
        sys.path.insert(0, exp_dir)
    import train as T                                 # noqa: E402
    import config as C                                # noqa: E402

    last_gen = max(hi for _, hi in windows)
    rc = dict(run.run)
    rc["mvg_ops"] = tuple(rc["mvg_ops"])
    rc["dense_log"] = ",".join(f"{lo}:{hi}" for lo, hi in windows)
    rc["generations"] = last_gen + 1
    rc["out_dir"] = os.path.join(out_root, os.path.basename(os.path.dirname(run.dir)))
    rc["archive_interval"] = 1       # --verify needs a per-generation archive
    rc["resume"] = False
    rc["viz_interval"] = 0
    rc["open_image"] = False
    rc["n_seeds"] = 1
    run_cfg = C.RunConfig(**rc)

    print(f"\n=== {run.arm} seed {run.seed}: replaying 0..{last_gen} "
          f"(dense in {rc['dense_log']}) -> {rc['out_dir']}")
    if dry_run:
        return None
    os.makedirs(rc["out_dir"], exist_ok=True)
    res = T.train_seed(run.cfg, run_cfg, run.seed)

    # train_seed writes log.csv and champions.npz; config.json and result.json are
    # written by train.py's main(), which we bypassed. Without them `find_runs`
    # skips the directory entirely (it keys on both files plus complete=true), so
    # the figure would silently fall back to the archived study. Write the fields
    # runs_io actually reads -- the brain config to regrow a genome, the run
    # config, the reference goal and gens_run -- and nothing we did not measure:
    # there is no `stats` block here because a replay saves no champion .eqx.
    with open(os.path.join(res.run_dir, "config.json"), "w") as fh:
        json.dump({"brain": {f.name: getattr(run.cfg, f.name)
                             for f in dataclasses.fields(run.cfg)
                             if f.name != "activation"},
                   "activation": getattr(run.cfg.activation, "__name__", "custom"),
                   "seed": run.seed,
                   "run": {f.name: getattr(run_cfg, f.name)
                           for f in dataclasses.fields(run_cfg)}},
                  fh, indent=2, default=str)
    with open(os.path.join(res.run_dir, "result.json"), "w") as fh:
        json.dump({"run_name": res.run_name, "seed": run.seed,
                   "gens_run": res.gens_run,
                   "reference_op": run_cfg.operation,
                   "acc_by_op": res.acc_by_op,
                   "dense_replay_of": run.dir,
                   "dense_log": rc["dense_log"],
                   "complete": True},
                  fh, indent=2, default=str)
    return res.run_dir


def verify(run, out_root, windows):
    """Compare the replay's archive against the original run's, generation by
    generation, over the generations both cover.

    Checks the champion GENOME, not only its accuracy: two different genomes can
    hit the same accuracy, so matching accuracy alone would say nothing about
    whether the search followed the same path.

    A DIVERGENCE HERE IS EXPECTED and is not a bug in the replay -- see the
    module docstring. What this prints is how far apart the two samples are; what
    makes the figure regenerable is replay-vs-replay, which is bit-exact.
    """
    dense_dir = os.path.join(out_root, os.path.basename(os.path.dirname(run.dir)),
                             os.path.basename(run.dir))
    if not os.path.exists(os.path.join(dense_dir, "champions.npz")):
        print(f"  {run.arm} seed {run.seed}: no replay archive at {dense_dir}")
        return False
    a = run.archive()
    b = np.load(os.path.join(dense_dir, "champions.npz"), allow_pickle=False)
    n = min(len(a["gen"]), len(b["gen"]))
    ok = True
    if not np.array_equal(a["gen"][:n], b["gen"][:n]):
        print(f"  {run.arm} seed {run.seed}: GENERATION INDICES DIFFER")
        return False
    if not np.array_equal(a["op"][:n], np.array([str(o) for o in b["op"][:n]])):
        print(f"  {run.arm} seed {run.seed}: ACTIVE GOAL DIFFERS")
        ok = False
    d_flat = float(np.abs(a["flat"][:n] - b["flat"][:n]).max())
    deltas = {f"acc_{o}": float(np.abs(np.asarray(a["acc"][o][:n])
                                       - b[f"acc_{o}"][:n]).max())
              for o in a["ops_in_play"] if f"acc_{o}" in b}
    acc_str = ", ".join(f"{k} {v:.2e}" for k, v in deltas.items())
    print(f"  {run.arm} seed {run.seed}: {n} generations compared | "
          f"max |dgenome| {d_flat:.2e} | {acc_str}")
    if d_flat > 1e-5 or any(v > 1e-5 for v in deltas.values()):
        print("     ^ diverged: a SECOND SAMPLE of this arm, not the archived "
              "trajectory.")
        print("       Expected -- the archived study is not bit-reproducible on "
              "this backend.")
        print("       Never mix replayed and archived rows in one panel; take the "
              "whole panel from one.")
        return "diverged"
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--constraint", choices=["budget", "nobudget"], default="budget")
    p.add_argument("--goal", choices=["fg", "mvg", "both"], default="both",
                   help="which goal regime to replay. `mvg` alone is enough for "
                        "the recovery statistic: a fixed-goal run has ONE goal "
                        "epoch and no switch, so `phase_stats` skips it.")
    p.add_argument("--windows", default=DEFAULT_WINDOWS)
    p.add_argument("--out-root", default=None,
                   help="default: <root>_dense, a SIBLING of the study directory "
                        "so find_runs never sees these truncated runs")
    p.add_argument("--verify", action="store_true",
                   help="only check an existing replay against the original runs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    windows = parse_windows(args.windows)
    root = os.path.normpath(args.root)
    out_root = args.out_root or (root.rstrip(os.sep) + "_dense")

    runs = {r.arm: r for r in find_runs(root) if r.seed == args.seed}
    goals = ["fg", "mvg"] if args.goal == "both" else [args.goal]
    want = [f"{args.constraint}_{g}" for g in goals]
    have = [a for a in want if a in runs]
    if len(have) < len(want):
        print(f"warning: only found {have} (wanted {want})")
    if not have:
        raise SystemExit(f"no {args.constraint} runs for seed {args.seed} under {root}")

    if args.verify:
        print("verifying the replay reproduces the original trajectory:")
        results = {a: verify(runs[a], out_root, windows) for a in have}
        if all(v is True for v in results.values()):
            print("\nevery arm matches the archived run bit-for-bit")
        elif any(v == "diverged" for v in results.values()):
            print("\nat least one arm diverged from the archive (expected). The "
                  "replay is self-consistent and regenerable; use it whole.")
        else:
            print("\nat least one arm could not be compared -- see above")
        return

    for arm in have:
        replay(runs[arm], windows, out_root, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\ndone -> {out_root}")
        print("now run:  python dense_replay.py --root ... --verify")


if __name__ == "__main__":
    main()
