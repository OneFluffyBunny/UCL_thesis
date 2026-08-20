"""Score Q_m / left_right_q on saved experiment_1 run(s) -- the constraint x
goal-switching 2x2 (idea #1 from the 2026-08-15 publishability planning
session; see RESULTS.md's synaptic-budget section for the two existing arms
this fills the other half of).

Usage:
    python score_2x2.py runs/budget_fg_b4s0.9 runs/mvg_E20_b4s0.9 \
                        runs/nobudget_fg runs/nobudget_mvg

Reads every seed subdirectory of each given run dir, regrows the headline
brain (config.json + result.json already record which one and with what
architecture), and reports left_right_q (primary, both assign modes) and
normalized_qm (secondary) per the RESULTS.md guidance: "Use left_right_q at
the planted split as the PRIMARY metric for retina claims; report newman_q
[here: normalized_qm] as secondary only."
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import jax.random as jr
import equinox as eqx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import ACTIVATIONS  # noqa: E402
from model import BrainConfig, Genome, _role_mask  # noqa: E402
from qmetrics.graph import from_matrix  # noqa: E402
from qmetrics.metrics import left_right_q, normalized_qm  # noqa: E402


def load_seed(seed_dir: str):
    with open(os.path.join(seed_dir, "config.json")) as fh:
        cfg_json = json.load(fh)
    with open(os.path.join(seed_dir, "result.json")) as fh:
        result = json.load(fh)

    brain = dict(cfg_json["brain"])
    brain["activation"] = ACTIVATIONS[cfg_json["activation"]]
    cfg = BrainConfig(**brain)

    headline = result["headline"]
    template = Genome.init(jr.PRNGKey(0), cfg)
    genome = eqx.tree_deserialise_leaves(
        os.path.join(seed_dir, f"{headline}_dna.eqx"), template)

    return cfg, cfg_json["run"], result, headline, genome


def score_one(cfg: BrainConfig, genome: Genome):
    w = np.asarray(genome.build_weights(cfg)[0])
    allowed = np.asarray(_role_mask(cfg))
    G = from_matrix(w, threshold=0.0, directed=True, weighted=True, allowed=allowed)

    half = cfg.n_in // 2
    pinned = {i: i // half for i in range(cfg.n_in)}
    exclude = list(range(cfg.n_in + cfg.n_hidden, cfg.n_in + cfg.n_hidden + cfg.n_out))

    score_opt, info_opt = left_right_q(G, pinned, exclude=exclude, assign="optimal")
    score_maj, info_maj = left_right_q(G, pinned, exclude=exclude, assign="majority")
    q_m, qm_info = normalized_qm(G)

    n_possible = int(allowed.sum())
    density = G.number_of_edges() / n_possible if n_possible else float("nan")

    return dict(density=density,
                lr_opt=score_opt, lr_opt_r=info_opt["r"], lr_opt_p=info_opt["p"],
                lr_maj=score_maj, lr_maj_r=info_maj["r"], lr_maj_p=info_maj["p"],
                q_m=q_m, q_m_p=qm_info["p"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run output dirs, each holding *_seedN/ subdirs")
    args = ap.parse_args()

    header = (f"{'arm':<22} {'seed':<5} {'headline':<9} {'acc':>6} {'density':>8} "
              f"{'LR_opt':>7} {'r_opt':>7} {'p_opt':>6} {'LR_maj':>7} {'Q_m':>7} {'p_qm':>6}")
    print(header)
    print("-" * len(header))

    for run_dir in args.run_dirs:
        arm = os.path.basename(os.path.normpath(run_dir))
        seed_dirs = sorted(glob.glob(os.path.join(run_dir, "*_seed*")))
        rows = []
        for sd in seed_dirs:
            if not os.path.isfile(os.path.join(sd, "result.json")):
                continue
            cfg, run_cfg, result, headline, genome = load_seed(sd)
            acc = result["stats"][headline]["accuracy"]
            m = score_one(cfg, genome)
            rows.append(m)
            print(f"{arm:<22} {result['seed']:<5} {headline:<9} {acc:>6.3f} {m['density']:>8.1%} "
                  f"{m['lr_opt']:>7.3f} {m['lr_opt_r']:>7.3f} {m['lr_opt_p']:>6.3f} "
                  f"{m['lr_maj']:>7.3f} {m['q_m']:>7.3f} {m['q_m_p']:>6.3f}")
        if rows:
            mean = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
            print(f"{arm:<22} {'mean':<5} {'':<9} {'':>6} {mean['density']:>8.1%} "
                  f"{mean['lr_opt']:>7.3f} {mean['lr_opt_r']:>7.3f} {mean['lr_opt_p']:>6.3f} "
                  f"{mean['lr_maj']:>7.3f} {mean['q_m']:>7.3f} {mean['q_m_p']:>6.3f}")
        print()


if __name__ == "__main__":
    main()
