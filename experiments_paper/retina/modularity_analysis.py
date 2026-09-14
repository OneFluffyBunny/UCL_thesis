"""Task 2: modularity metrics (Newman Q, left-right r/z/p) for the 5-seed
FG/MVG retina run set. Read-only over already-completed saved_models/ runs --
no training happens here.

Loads each run exactly like analysis.py's load_run() (config.yml +
solution_best.npy -> grow_network()), builds a graph via qmetrics, and reports
Newman Q (greedy community detection) plus the left-right planted-partition
score (pinned = the 8 inputs split into left/right quadruples, output excluded).

Note on qmetrics' from_matrix(): it does NOT take an `order` kwarg (only
roles_for/role_mask do, and those are for plotting labels, not the metrics
computed here). NDP's "ioh" layout already puts inputs at indices 0-7 and the
output at index 8 (verified against train_backend.py's retina_fitness, which
reads network_state[obs_dim] as the output) -- exactly what left_right_q's
pinned/exclude args below assume -- so from_matrix(W) with no extra args is
correct as-is.

Usage: python experiments_paper/retina/modularity_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

_NDP_ROOT = Path(__file__).resolve().parents[2]
_UCL_ROOT = _NDP_ROOT.parent
sys.path.insert(0, str(_NDP_ROOT))
sys.path.insert(0, str(_UCL_ROOT))

from train_backend import grow_network, retina_fitness  # noqa: E402
from qmetrics.graph import from_matrix  # noqa: E402
from qmetrics.metrics import newman_q, left_right_q, normalized_qm  # noqa: E402

SAVED_MODELS = _NDP_ROOT / "saved_models"

FG_RUNS = ["1786033855", "1786102425", "1788808796", "1788808995", "1788809505"]
MVG_RUNS = ["1786053806", "1788817038", "1788821061", "1788866451", "1788871442"]


def analyze(run_id: str) -> dict:
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

    G = from_matrix(W)
    Q, comms = newman_q(G)
    score, extras = left_right_q(G, {i: i // 4 for i in range(8)}, exclude=[8], assign="optimal")
    Qm, qm_extras = normalized_qm(G)

    condition = "MVG" if config.get("mvg", False) else "FG"

    return {
        "run_id": run_id,
        "seed": config.get("seed"),
        "condition": condition,
        "nodes": nodes,
        "edges": edges,
        "balanced_acc": acc,
        "io_weights_uniform": io_uniform,
        "Q": Q,
        "n_communities": len(comms),
        "Qm": Qm,
        "Qm_z": qm_extras["z"],
        "Qm_p": qm_extras["p"],
        "Qm_saturated": qm_extras["saturated"],
        "lr_score": score,
        "r": extras["r"],
        "z": extras["z"],
        "p": extras["p"],
        "q_lr": extras["q"],
        "crosstalk": extras["crosstalk"],
    }


def main():
    rows = [analyze(r) for r in FG_RUNS + MVG_RUNS]
    header = ["run_id", "seed", "cond", "nodes", "edges", "bal_acc", "Q", "Qm", "Qm_z", "Qm_p", "sat", "r", "z", "p"]
    print(" | ".join(header))
    for r in rows:
        print(" | ".join([
            r["run_id"], str(r["seed"]), r["condition"], str(r["nodes"]), str(r["edges"]),
            f"{r['balanced_acc']:.4f}", f"{r['Q']:.4f}",
            f"{r['Qm']:.4f}" if r["Qm"] == r["Qm"] else "nan",
            f"{r['Qm_z']:.4f}" if r["Qm_z"] == r["Qm_z"] else "nan",
            f"{r['Qm_p']:.4f}", str(r["Qm_saturated"]),
            f"{r['r']:.4f}", f"{r['z']:.4f}" if r["z"] == r["z"] else "nan", f"{r['p']:.4f}",
        ]))

    import json
    out_path = Path(__file__).parent / "modularity_analysis_raw.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    main()
