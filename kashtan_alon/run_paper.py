"""Run the Kashtan-Alon retina experiment EXACTLY as in the paper.

This is the canonical reproduction. It runs the paper's central comparison --
Modularly-Varying Goals (G_AND = L AND R  <->  G_OR = L OR R, switching every 20
generations) versus a Fixed-Goal control (the retina goal L AND R held constant)
-- with every parameter locked to Kashtan & Alon, PNAS 2005:

  * layered feedforward net 8-8-4-2-1, tanh(lambda*z) with lambda = 20;
  * integer connection weights {-2,-1,1,2} (0 = absent, topology evolves);
    integer biases/thresholds {-2,-1,0,1,2};
  * population 1000, 25000 generations, mutation-only (asexual) GA;
  * mutation: add connection 20% / remove connection 20% per network,
    weight +/-1 at prob 2/n per connection, bias +/-1 at prob 1/24 per node;
  * fitness = RAW fraction of correct answers over all 256 input patterns
    (the paper's own performance measure -- no class balancing).

Newman Q is logged every generation for both conditions; the paper's headline is
that Q rises and stays high under MVG but stays low under the Fixed Goal.

    conda run -n lndp python run_paper.py                # full paper run (slow: 1000 x 25000)
    conda run -n lndp python run_paper.py --n-seeds 5    # paper used multiple independent runs
    conda run -n lndp python run_paper.py --smoke        # tiny, just to check the pipeline runs

Everything not exposed here is fixed at the paper value on purpose -- this script
is the "as in the paper" preset. For sweeps/variations use train.py directly.
"""

from __future__ import annotations

import argparse

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
    cli = ap.parse_args()

    # Paper-locked parameters (do not expose -- this is the faithful preset).
    paper = dict(task="retina", layers="8,8,4,2,1", lam=20.0, input_encoding="bipolar",
                 pop=1000, generations=25000, init_density=0.5,
                 p_add=0.2, p_remove=0.2, bias_prob=1.0 / 24.0,
                 tournament_k=3, n_elite=1, fitness="raw",
                 switch_interval=20, mvg_ops="and,or",
                 weighted_q=False, log_interval=10, target=1.0,
                 out_dir=cli.out_dir, viz=cli.viz)
    if cli.smoke:
        paper.update(pop=60, generations=300, log_interval=20)
        print("*** SMOKE MODE: pop 60 / 300 gens -- checks the pipeline, NOT a paper result ***\n")

    # Two conditions, identical everything except the goal schedule.
    conditions = {
        "MVG":         dict(mvg=True,  operation="and"),   # G_AND <-> G_OR every 20 gens
        "FG(L AND R)": dict(mvg=False, operation="and"),   # the fixed retina goal (control)
    }

    # Shared task tensors (built once).
    import os
    os.makedirs(cli.out_dir, exist_ok=True)
    layers = tuple(int(x) for x in paper["layers"].split(","))
    cfg = M.NetConfig(layers=layers, lam=paper["lam"])
    X_bits = np.asarray(tasks.all_binary_inputs(layers[0]))
    X = X_bits * 2.0 - 1.0   # bipolar {-1,+1}

    results = {name: [] for name in conditions}
    for name, cond in conditions.items():
        print(f"\n========== condition: {name} ==========")
        args = build_base_args({**paper, **cond})
        for i in range(cli.n_seeds):
            seed = cli.seed + i
            open_after = args.viz and (i == cli.n_seeds - 1)   # open each condition's final brain
            bf, q = T.train_seed(cfg, X, X_bits, args, seed, open_after)
            results[name].append((seed, bf, q))

    print("\n================ PAPER COMPARISON ================")
    for name in conditions:
        qs = [r[2] for r in results[name]]
        fs = [r[1] for r in results[name]]
        print(f"  {name:12s} | mean Newman Q {np.mean(qs):.3f} | mean best fit {np.mean(fs):.3f} "
              f"| Q per seed {[f'{q:.3f}' for q in qs]}")
    print("  Expectation (paper): MVG keeps Q high (~0.4+); the Fixed Goal stays low (~0.15-0.2).")


if __name__ == "__main__":
    main()
