"""Do the budget-matched FG/MVG retina brains compute anything beyond a count of
active inputs? Every number in RESULTS.md's "What the brains compute" entry comes
from this script.

  ceilings    best balanced accuracy, over ALL 256 patterns, of (a) any rule that
              depends only on how many inputs are on, (b) any sign-symmetric
              classifier f(-x) = -f(x), and (c) the count rules NDP can express
              (both at once: output 0 at exactly 4 on) -- for AND and for OR.
  final       the 10 final brains (FG: best-ever genome, MVG: final-generation
              champion): same-count invariance, oddness, 4-on outputs, score
              with 4-on ties set to 0, score after renumbering hidden neurons,
              input-lineage structure, greedy communities, shared wirings.
  or-champs   the best OR-epoch champion of each MVG run (needs
              replay_archive.py): score vs the exact cap, and whether its growth
              was input-symmetric.

The rollout is re-implemented batched for the invariance checks (row form of
NDP.propagate_features: s <- tanh(W.T s), inputs re-clamped, state zeroed per
pattern, no bias); every reported ACCURACY uses the project's own
train_backend.retina_fitness, whose per-pattern arithmetic is what training saw.

Usage: conda run -n ndp python experiments_paper/retina/counting_analysis.py [ceilings|final|or-champs|all]
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)
_NDP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NDP_ROOT))
sys.path.insert(0, str(_NDP_ROOT.parent))

from train_backend import grow_network, retina_fitness  # noqa: E402
from NDP import bfs_diameter  # noqa: E402
from ka_task import retina_dataset, to_bipolar  # noqa: E402
from qmetrics.graph import from_matrix  # noqa: E402
from qmetrics.metrics import newman_q  # noqa: E402

SAVED_MODELS = _NDP_ROOT / "saved_models"
FG_RUNS = ["1789303257", "1789303833", "1789304372", "1789304978", "1789305630"]
MVG_RUNS = ["1786053806", "1788817038", "1788821061", "1788866451", "1788871442"]
N_IN, OUT = 8, 8


def data(op):
    X01, Y = retina_dataset(op=op)
    X = to_bipolar(np.asarray(X01)).astype(float)
    return X, np.asarray(Y).ravel(), (X > 0).sum(1)


def bal(pred, Y):
    return 0.5 * ((pred[Y == 1] == 1).mean() + (pred[Y == 0] == 0).mean())


def load(run_id, dna=None, op="and"):
    with open(SAVED_MODELS / run_id / "config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg["current_op"] = op
    dna = np.load(SAVED_MODELS / run_id / "solution_best.npy") if dna is None else dna
    W, _ = grow_network(dna, cfg)
    return np.asarray(W, dtype=float), cfg


def rollout(W, cfg, X):
    T = bfs_diameter(W) + cfg["network_thinking_time_extra_rollout"]
    s = np.zeros((len(X), W.shape[0]))
    s[:, :N_IN] = X
    for _ in range(T):
        s = np.tanh(s @ W)
        s[:, :N_IN] = X
    return s[:, OUT]


def renumbered_scores(W, cfg, k, rng):
    n = W.shape[0]
    out = []
    for _ in range(k):
        p = np.r_[np.arange(N_IN + 1), N_IN + 1 + rng.permutation(n - N_IN - 1)]
        out.append(round(retina_fitness(W=W[np.ix_(p, p)], config=cfg), 4))
    return sorted(set(out))


def odd_count_rule(s, k):
    """NDP-expressible count rule: s = predictions at counts 5..8, mirrored below 4, 0 at 4."""
    return s[k - 5] if k > 4 else (0 if k == 4 else 1 - s[3 - k])


def cmd_ceilings():
    for op in ["and", "or"]:
        X, Y, pc = data(op)
        count = max(bal(np.array([r[k] for k in pc]), Y) for r in itertools.product([0, 1], repeat=9))
        expr = max((bal(np.array([odd_count_rule(s, k) for k in pc]), Y), s)
                   for s in itertools.product([0, 1], repeat=4))
        w = {1: 0.5 / (Y == 1).sum(), 0: 0.5 / (Y == 0).sum()}
        idx = {tuple(r): i for i, r in enumerate(X.astype(int))}
        seen, odd = set(), 0.0
        for i in range(len(X)):
            j = idx[tuple(-X[i].astype(int))]
            if i in seen:
                continue
            seen |= {i, j}
            odd += max(w[1] * (Y[i] == 1) + w[0] * (Y[j] == 0), w[0] * (Y[i] == 0) + w[1] * (Y[j] == 1),
                       w[0] * (Y[i] == 0) + w[0] * (Y[j] == 0))
        rule = [k for k in range(9) if odd_count_rule(expr[1], k)]
        print(f"{op}: best count-only rule {count:.4f} | best sign-symmetric classifier {odd:.4f} | "
              f"best NDP-expressible count rule {expr[0]:.4f} (predict 1 at counts {rule}) | "
              f"majority vote (>=5 on) {bal((pc >= 5).astype(int), Y):.4f}")
    X01, Y = retina_dataset(op="and")
    for bits in ["11001100", "10101010", "00110011"]:
        i = int(np.where((np.asarray(X01) == np.array([int(b) for b in bits])).all(1))[0][0])
        print(f"  pattern {bits}: {sum(map(int, bits))} on, AND label {np.asarray(Y).ravel()[i]}")


def cmd_final():
    X, Y, pc = data("and")
    idx = {tuple(r): i for i, r in enumerate(X.astype(int))}
    rng = np.random.default_rng(0)
    masks = {}
    for cond, runs in (("FG", FG_RUNS), ("MVG", MVG_RUNS)):
        for rid in runs:
            W, cfg = load(rid)
            n = W.shape[0]
            f = rollout(W, cfg, X)
            spread = max(np.ptp(f[pc == k]) for k in range(9) if k != 4)
            odd = max(abs(f[i] + f[idx[tuple(-X[i].astype(int))]]) for i in range(len(X)))
            tie0 = bal(np.where(pc == 4, 0, f > 0).astype(int), Y)
            A = np.abs(W) > 0
            k_in = A[N_IN + 1:, :N_IN].sum(1)
            _, comms = newman_q(from_matrix(W))
            per = sorted(sum(1 for i in range(N_IN) if i in c) for c in comms)
            masks[rid] = A
            print(f"{cond:3s} {rid} n={n:2d} | acc {retina_fitness(W=W, config=cfg):.4f} | same-count spread "
                  f"(k!=4) {spread:.0e} | max|f(x)+f(-x)| {odd:.0e} | max|out| at 4 on {np.abs(f[pc == 4]).max():.0e} "
                  f"| acc with 4-on ties->0 {tie0:.4f} | acc after 8 hidden renumberings {renumbered_scores(W, cfg, 8, rng)} "
                  f"| I-I edges {int(A[:N_IN, :N_IN].sum())} | hidden touching 1/2-7/8 inputs "
                  f"{int((k_in == 1).sum())}/{int(((k_in > 1) & (k_in < 8)).sum())}/{int((k_in == 8).sum())} "
                  f"| greedy communities {len(comms)}, inputs per community {per}")
    groups = {}
    for rid in FG_RUNS + MVG_RUNS:
        key = next((k for k in groups if masks[k].shape == masks[rid].shape and np.array_equal(masks[k], masks[rid])), rid)
        groups.setdefault(key, []).append(rid)
    print("identical edge sets:", list(groups.values()))


def cmd_or_champs():
    X, Y, pc = data("or")
    rng = np.random.default_rng(1)
    for rid in MVG_RUNS:
        arc_path = SAVED_MODELS / rid / "replay_champions.npz"
        if not arc_path.exists():
            print(f"{rid}: no replay archive")
            continue
        a = np.load(arc_path)
        m = a["op"] == "or"
        g = int(np.where(m)[0][np.argmax(a["champ_fit"][m])])
        W, cfg = load(rid, dna=a["champions"][g], op="or")
        f = rollout(W, cfg, X)
        sizes = []
        for k in range(cfg["number_of_growth_cycles"] + 1):
            c = dict(cfg, number_of_growth_cycles=k)
            sizes.append(np.asarray(grow_network(a["champions"][g], c)[0]).shape[0])
        deg = (np.abs(W[:N_IN]) > 0).sum(1).tolist()
        print(f"{rid}: best OR champion gen {g} | acc {retina_fitness(W=W, config=cfg):.4f} | acc with 4-on ties->0 "
              f"{bal(np.where(pc == 4, 0, f > 0).astype(int), Y):.4f} | after 6 hidden renumberings "
              f"{renumbered_scores(W, cfg, 6, rng)} | neurons per growth cycle {sizes} | input degrees {deg}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("ceilings", cmd_ceilings), ("final", cmd_final), ("or-champs", cmd_or_champs)):
        if what in (name, "all"):
            print(f"== {name}")
            fn()
