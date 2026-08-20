"""Ablation of run_paper.py: same KA-faithful retina task, MVG vs FG, same GA
hyperparameters as Run 5 in RESULTS.md -- with the fan-in cap REMOVED.

Tests the constraint-necessity hypothesis (add_to_latex.md, "Testing whether a
constraint is necessary for modularity"): if connectivity scarcity, not MVG
alone, is what drives KA's modularity result, then removing the fan-in cap
(the one thing that makes wiring scarce in KA's own model) should collapse
Q_m toward the FG level or below, even under MVG.

Everything is identical to run_paper.py's paper preset (task, pop=600,
generations=25000, n_elite=150, pc=0.5, pm=0.5, switch_interval=20,
qm_nrand=1000, init_density=0.5) except:
  * NetConfig(fan_in=()) -- `_fan_in()` in model.py then falls back to
    "cap = the destination layer's own width", i.e. no constraint at all
    (a neuron may receive from every node in the previous layer).
  * separate --out-dir (default ./runs_no_fanin) so it can't collide with or
    overwrite the constrained run's saved seeds in ./runs.

    conda run -n lndp python run_ablation_no_fanin.py --n-seeds 3
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import train as T
import model as M
import tasks


def build_base_args(overrides):
    args = T.build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def main():
    ap = argparse.ArgumentParser(
        description="KA retina ablation: MVG vs FG with the fan-in cap removed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=3, help="independent runs per condition")
    ap.add_argument("--seed", type=int, default=0, help="first seed (conditions share seeds)")
    ap.add_argument("--out-dir", default="./runs_no_fanin")
    ap.add_argument("--viz", dest="viz", action="store_true", default=True)
    ap.add_argument("--no-viz", dest="viz", action="store_false")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--checkpoint-interval", type=int, default=1000)
    cli = ap.parse_args()

    # Same paper-locked parameters as run_paper.py -- only the fan-in cap differs.
    paper = dict(task="retina", layers="8,8,4,2,1", input_encoding="binary",
                 pop=600, generations=25000, init_density=0.5,
                 n_elite=150, pc=0.5, pm=0.5, fitness="raw",
                 switch_interval=20, mvg_ops="and,or",
                 weighted_q=False, qm_nrand=1000, log_interval=10, target=1.0,
                 out_dir=cli.out_dir, viz=cli.viz,
                 resume=not cli.fresh, checkpoint_interval=cli.checkpoint_interval)

    conditions = {
        "MVG":         dict(mvg=True,  operation="and"),
        "FG(L AND R)": dict(mvg=False, operation="and"),
    }

    os.makedirs(cli.out_dir, exist_ok=True)
    layers = tuple(int(x) for x in paper["layers"].split(","))
    cfg = M.NetConfig(layers=layers, fan_in=())     # <-- the ablation: no fan-in cap
    print(f"*** ABLATION: fan-in cap removed (cfg.fan_in={cfg.fan_in}) -- "
          f"every neuron may receive from every node in the previous layer ***\n")
    X_bits = np.asarray(tasks.all_binary_inputs(layers[0]))
    X = X_bits

    results = {name: [] for name in conditions}
    for name, cond in conditions.items():
        print(f"\n========== condition: {name} (NO fan-in cap) ==========")
        args = build_base_args({**paper, **cond})
        for i in range(cli.n_seeds):
            seed = cli.seed + i
            result_path = os.path.join(cli.out_dir, f"{T.run_name_for(args, seed)}_result.json")
            if args.resume and os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                qm = r.get("q_m", r.get("q"))
                print(f"[seed {seed}] already complete (best {r['best_fit']:.3f} | "
                      f"Q_m {qm:.3f}) -> skip")
                results[name].append((seed, r["best_fit"], qm))
                continue
            open_after = args.viz and (i == cli.n_seeds - 1)
            bf, qm = T.train_seed(cfg, X, X_bits, args, seed, open_after)
            results[name].append((seed, bf, qm))

    print("\n================ ABLATION COMPARISON (no fan-in cap) ================")
    for name in conditions:
        qs = [r[2] for r in results[name]]
        fs = [r[1] for r in results[name]]
        dens = [None for _ in results[name]]
        print(f"  {name:12s} | mean Q_m {np.mean(qs):.3f} | mean best fit {np.mean(fs):.3f} "
              f"| Q_m per seed {[f'{q:.3f}' for q in qs]}")
    print("  Constrained baseline (Run 5, RESULTS.md): MVG Q_m 0.245 +/- 0.049; "
          "FG Q_m 0.025 +/- 0.139.")
    print("  Hypothesis: removing the fan-in cap collapses MVG's Q_m toward/below FG.")


if __name__ == "__main__":
    main()
