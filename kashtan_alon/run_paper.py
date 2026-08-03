"""Run the Kashtan-Alon retina experiment EXACTLY as in the paper.

Faithful to Kashtan & Alon, PNAS 2005 (the NEURAL-NETWORK retina experiment),
verified against the paper's own methods (PMC1236541). Its central comparison --
Modularly-Varying Goals (G_AND = L AND R  <->  G_OR = L OR R, switching every 20
generations) versus a Fixed-Goal control (L AND R held constant) -- with every
parameter locked to the paper:

  * task: KA's real retina objects (4x2 pixels, >=3-black / outer-column-only rule,
    Fig. 5a) -- NOT the project's `(p0&p1)|(p2&p3)` stand-in;
  * network: retina(8) -> 8 -> 4 -> 2 -> 1, HARD-THRESHOLD neurons, weights in
    {-1,+1}, fan-in <=3 in layers 1-3 and <=2 at the output;
  * search: population 600, ELITE strategy (top 150 replicate unchanged),
    CROSSOVER Pc=0.5 + mutation Pm=0.5;
  * fitness: RAW fraction correct over all 256 patterns (KA's own measure);
  * modularity: normalized Q_m = (Q_real-Q_rand)/(Q_max-Q_rand), Q_rand over 1000
    degree-preserving randomizations.

Paper result: Q_m stays HIGH under MVG (0.35 +- 0.02) and LOW under the Fixed Goal
(0.15 +- 0.02). Raw Newman Q is logged per generation only as a cheap live trace;
the headline is the end-of-run Q_m.

    conda run -n lndp python run_paper.py                # full paper run (600 x 25000)
    conda run -n lndp python run_paper.py --n-seeds 5    # multiple independent runs
    conda run -n lndp python run_paper.py --smoke        # tiny, just checks the pipeline

Everything not exposed here is fixed at the paper value on purpose -- this script
is the "as in the paper" preset. For sweeps/variations use train.py directly.
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
    """A fully-defaulted train.py namespace (paper values) with `overrides` applied."""
    args = T.build_parser().parse_args([])   # all argparse defaults = paper values
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def main():
    ap = argparse.ArgumentParser(
        description="Kashtan-Alon retina, exactly as in the paper: MVG vs Fixed-Goal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=1, help="independent runs per condition")
    ap.add_argument("--seed", type=int, default=0, help="first seed (conditions share seeds)")
    ap.add_argument("--out-dir", default="./runs")
    ap.add_argument("--viz", dest="viz", action="store_true", default=True,
                    help="render the best net of each run, K-A-style [default: on]")
    ap.add_argument("--no-viz", dest="viz", action="store_false",
                    help="disable visualisation")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run (pop 60, 300 gens) to verify the pipeline, NOT the paper")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore checkpoints/results and restart every run from scratch")
    ap.add_argument("--checkpoint-interval", type=int, default=1000,
                    help="generations between full-state checkpoints (0 = off); enables resume")
    cli = ap.parse_args()

    # Paper-locked parameters (do not expose -- this is the faithful KA 2005 preset).
    paper = dict(task="retina", layers="8,8,4,2,1", input_encoding="binary",
                 pop=600, generations=25000, init_density=0.5,
                 n_elite=150, pc=0.5, pm=0.5, fitness="raw",
                 switch_interval=20, mvg_ops="and,or",
                 weighted_q=False, qm_nrand=1000, log_interval=10, target=1.0,
                 out_dir=cli.out_dir, viz=cli.viz,
                 resume=not cli.fresh, checkpoint_interval=cli.checkpoint_interval)
    if cli.smoke:
        paper.update(pop=60, generations=300, n_elite=15, qm_nrand=50,
                     log_interval=20, checkpoint_interval=100)
        print("*** SMOKE MODE: pop 60 / 300 gens -- checks the pipeline, NOT a paper result ***\n")

    # Two conditions, identical everything except the goal schedule.
    conditions = {
        "MVG":         dict(mvg=True,  operation="and"),   # G_AND <-> G_OR every 20 gens
        "FG(L AND R)": dict(mvg=False, operation="and"),   # the fixed retina goal (control)
    }

    os.makedirs(cli.out_dir, exist_ok=True)
    layers = tuple(int(x) for x in paper["layers"].split(","))
    cfg = M.NetConfig(layers=layers)
    X_bits = np.asarray(tasks.all_binary_inputs(layers[0]))
    X = X_bits                                              # binary {0,1} pixels (KA)

    results = {name: [] for name in conditions}
    for name, cond in conditions.items():
        print(f"\n========== condition: {name} ==========")
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
            open_after = args.viz and (i == cli.n_seeds - 1)   # open each condition's final brain
            bf, qm = T.train_seed(cfg, X, X_bits, args, seed, open_after)
            results[name].append((seed, bf, qm))

    print("\n================ PAPER COMPARISON ================")
    for name in conditions:
        qs = [r[2] for r in results[name]]
        fs = [r[1] for r in results[name]]
        print(f"  {name:12s} | mean Q_m {np.mean(qs):.3f} | mean best fit {np.mean(fs):.3f} "
              f"| Q_m per seed {[f'{q:.3f}' for q in qs]}")
    print("  Expectation (paper): MVG Q_m ~0.35; Fixed Goal Q_m ~0.15.")


if __name__ == "__main__":
    main()
