"""Descriptive stats (nodes, edges, accuracy, ...) for a set of saved retina runs.

Deliberately stops short of modularity metrics (Q, r, z, p via qmetrics) --
that's a separate step. This only reports what's cheaply derivable from
config.yml + solution_best.npy via grow_network(), same as analysis.py.

Usage: python experiments_paper/retina/run_stats.py <run_id> [<run_id> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

_NDP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NDP_ROOT))

from train_backend import grow_network, retina_fitness  # noqa: E402

SAVED_MODELS = _NDP_ROOT / "saved_models"


def run_stats(run_id: str) -> dict:
    conf_path = SAVED_MODELS / run_id / "config.yml"
    dna_path = SAVED_MODELS / run_id / "solution_best.npy"
    with open(conf_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    dna = np.load(dna_path)
    W, _ = grow_network(dna, config)

    nodes = W.shape[0]
    nz = np.abs(W[np.abs(W) > 0])
    edges = len(nz)
    acc = retina_fitness(W=W, config=config)

    obs_dim = config["observation_dim"]
    w_io = W[:obs_dim, obs_dim]
    io_uniform = bool(np.ptp(np.abs(w_io)) < 1e-3) if len(w_io) else False

    condition = "MVG" if config.get("mvg", False) else "FG"

    return {
        "run_id": run_id,
        "seed": config.get("seed"),
        "condition": condition,
        "nodes": nodes,
        "edges": edges,
        "balanced_acc": acc,
        "w_min": float(nz.min()) if edges else float("nan"),
        "w_max": float(nz.max()) if edges else float("nan"),
        "w_std": float(nz.std()) if edges else float("nan"),
        "io_weights_uniform": io_uniform,
        "training_time_s": config.get("training time"),
        "generations": config.get("generations"),
    }


def main():
    run_ids = sys.argv[1:]
    if not run_ids:
        print(__doc__)
        sys.exit(1)

    rows = [run_stats(r) for r in run_ids]

    header = ["run_id", "seed", "cond", "nodes", "edges", "bal_acc",
              "|w| min", "|w| max", "|w| std", "IO-uniform", "train_s"]
    print(" | ".join(header))
    print("-" * 120)
    for r in rows:
        print(" | ".join([
            r["run_id"], str(r["seed"]), r["condition"], str(r["nodes"]), str(r["edges"]),
            f"{r['balanced_acc']:.4f}", f"{r['w_min']:.4f}", f"{r['w_max']:.4f}",
            f"{r['w_std']:.6f}", str(r["io_weights_uniform"]), str(r["training_time_s"]),
        ]))


if __name__ == "__main__":
    main()
