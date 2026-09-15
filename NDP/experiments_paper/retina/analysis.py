"""Analysis checks backing the mechanistic claims in RESULTS.md.

All of these load a saved run's config.yml + solution_best.npy from
saved_models/<run_id>/ and regrow it via grow_network() -- nothing here
re-trains anything. See RESULTS.md for which command produced which claim.

Run `python experiments_paper/retina/analysis.py --help` for the available checks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

_NDP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NDP_ROOT))

from train_backend import grow_network, retina_fitness  # noqa: E402
from NDP import propagate_features, bfs_diameter  # noqa: E402
from ka_task import retina_and_dataset, to_bipolar  # noqa: E402

SAVED_MODELS = _NDP_ROOT / "saved_models"


def load_run(run_id: str):
    conf_path = SAVED_MODELS / run_id / "config.yml"
    dna_path = SAVED_MODELS / run_id / "solution_best.npy"
    with open(conf_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    dna = np.load(dna_path)
    W, network_state = grow_network(dna, config)
    return W, network_state, config


def balanced_acc(pred: np.ndarray, Y: np.ndarray):
    correct = pred.astype(int) == Y
    tpr = correct[Y == 1].mean()
    tnr = correct[Y == 0].mean()
    return 0.5 * (tpr + tnr), tpr, tnr


def predict_all(W: np.ndarray, config: dict):
    """Run the actual NDP rollout (propagate_features per pattern) and return predictions."""
    X01, Y = retina_and_dataset()
    X = to_bipolar(X01)
    obs_dim = config["observation_dim"]
    diameter = bfs_diameter(W)
    network_thinking_time = diameter + config["network_thinking_time_extra_rollout"]
    preds = np.zeros(256, dtype=int)
    for idx, x in enumerate(X):
        state = np.zeros(W.shape[0])
        state[:obs_dim] = x
        state = propagate_features(
            network_state=state, W=W, network_thinking_time=network_thinking_time,
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"], persistent_observation=x,
            feature_transformation_model=None, use_torch=False,
        )
        preds[idx] = 1 if state[obs_dim] > 0 else 0
    return preds, X01, Y


def cmd_weights(run_id: str):
    """Inspect a solution's grown weight matrix: uniform/saturated? matches retina_fitness()?"""
    W, _, config = load_run(run_id)
    nz = np.abs(W[np.abs(W) > 0])
    print(f"Run {run_id} ({config['environment']}): {W.shape[0]} nodes, {len(nz)} edges")
    print(f"|w| min={nz.min():.4f} max={nz.max():.4f} mean={nz.mean():.4f} std={nz.std():.6f}")

    obs_dim = config.get("observation_dim")
    if obs_dim is not None and W.shape[0] > obs_dim and "retina" in config["environment"]:
        w_io = W[:obs_dim, obs_dim]
        print(f"Input->output weights: {np.round(w_io, 4)}")
        X01, Y = retina_and_dataset()
        X = to_bipolar(X01)
        pred = (X @ w_io > 0).astype(int)
        ba, tpr, tnr = balanced_acc(pred, Y)
        print(f"Hand-computed sign(w.x) balanced accuracy: {ba:.4f} (TPR={tpr:.4f} TNR={tnr:.4f})")
        fit = retina_fitness(W=W, config=config)
        print(f"retina_fitness() balanced accuracy: {fit:.4f}")


def cmd_popcount():
    """Popcount/majority-vote analysis of the retina-AND task itself (no saved run needed)."""
    X01, Y = retina_and_dataset()
    X = to_bipolar(X01)
    Y = Y.astype(int)
    popcount = X01.sum(axis=1)

    majority_pred = (popcount >= 5).astype(int)
    ba, tpr, tnr = balanced_acc(majority_pred, Y)
    print(f"Majority-vote (popcount>=5) balanced_acc = {ba:.10f} (TPR={tpr:.4f} TNR={tnr:.4f})")

    print("\npopcount | n_patterns | n_positive | P(Y=1|popcount)")
    for k in range(9):
        mask = popcount == k
        n = mask.sum()
        npos = Y[mask].sum()
        print(f"   {k}     |    {n:4d}    |    {npos:4d}    |  {npos/n if n else 0:.3f}")

    w0 = np.ones(8)

    def eval_w(w):
        pred = (X @ w > 0).astype(int)
        ba, _, _ = balanced_acc(pred, Y)
        return ba

    base = eval_w(w0)
    print(f"\nUniform weight [1]*8 balanced_acc = {base:.4f}")
    improved = False
    for i in range(8):
        for delta in [-0.5, -0.2, -0.05, 0.05, 0.2, 0.5]:
            w2 = w0.copy()
            w2[i] += delta
            ba2 = eval_w(w2)
            if ba2 > base + 1e-9:
                print(f"  IMPROVEMENT: w[{i}] += {delta} -> {ba2:.4f}")
                improved = True
    if not improved:
        print("  No single-coordinate perturbation improves on uniform weights (local optimum).")


def cmd_compare(run_a: str, run_b: str):
    """Compare which of the 256 patterns two runs' solutions get wrong."""
    Wa, _, ca = load_run(run_a)
    Wb, _, cb = load_run(run_b)
    preds_a, X01, Y = predict_all(Wa, ca)
    preds_b, _, _ = predict_all(Wb, cb)

    acc_a = preds_a == Y
    acc_b = preds_b == Y
    print(f"Run {run_a}: {Wa.shape[0]} nodes, {acc_a.sum()}/256 correct")
    print(f"Run {run_b}: {Wb.shape[0]} nodes, {acc_b.sum()}/256 correct")

    wrong_a = set(np.where(~acc_a)[0])
    wrong_b = set(np.where(~acc_b)[0])
    both = wrong_a & wrong_b
    only_a = wrong_a - wrong_b
    only_b = wrong_b - wrong_a
    print(f"\nWrong: {run_a}={len(wrong_a)}, {run_b}={len(wrong_b)}")
    print(f"Both wrong: {len(both)} | only {run_a}: {len(only_a)} | only {run_b}: {len(only_b)}")
    if wrong_a | wrong_b:
        print(f"Jaccard overlap: {len(both) / len(wrong_a | wrong_b):.3f}")

    popcount = X01.sum(axis=1)
    for name, wrong in [(run_a, wrong_a), (run_b, wrong_b)]:
        print(f"\n{name} errors by popcount:")
        for k in range(9):
            mask = popcount == k
            n_total = mask.sum()
            n_wrong = sum(1 for i in wrong if popcount[i] == k)
            if n_total:
                print(f"  popcount={k}: {n_wrong}/{n_total} wrong")

    print(f"\nOnly-{run_a}-wrong patterns (idx, popcount): {[(i, int(popcount[i])) for i in sorted(only_a)]}")
    print(f"Only-{run_b}-wrong patterns (idx, popcount): {[(i, int(popcount[i])) for i in sorted(only_b)]}")


def cmd_saturation(run_ids: list[str]):
    """Edge-weight magnitude stats across multiple runs (the saturation comparison table)."""
    for run_id in run_ids:
        W, _, config = load_run(run_id)
        nz = np.abs(W[np.abs(W) > 0])
        print(f"{run_id} ({config['environment']}): {len(nz)} edges, "
              f"|w| min={nz.min():.4f} max={nz.max():.4f} mean={nz.mean():.4f} std={nz.std():.6f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_w = sub.add_parser("weights", help="Inspect one run's grown weight matrix")
    p_w.add_argument("run_id")

    sub.add_parser("popcount", help="Popcount/majority-vote analysis of the retina-AND task")

    p_c = sub.add_parser("compare", help="Compare which patterns two runs get wrong")
    p_c.add_argument("run_a")
    p_c.add_argument("run_b")

    p_s = sub.add_parser("saturation", help="Edge-weight magnitude stats across runs")
    p_s.add_argument("run_ids", nargs="+")

    args = p.parse_args()
    if args.cmd == "weights":
        cmd_weights(args.run_id)
    elif args.cmd == "popcount":
        cmd_popcount()
    elif args.cmd == "compare":
        cmd_compare(args.run_a, args.run_b)
    elif args.cmd == "saturation":
        cmd_saturation(args.run_ids)


if __name__ == "__main__":
    main()
