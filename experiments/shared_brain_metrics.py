"""Modularity metrics for a RECURRENT (N, N) brain — shared by experiments 1-3.

`kashtan_alon/` reports four numbers per network, and this module produces the
same four for the brains grown in `experiments/`, so the two studies can be read
side by side:

    Q         Newman modularity of a DISCOVERED partition        (secondary)
    Q_m       KA's normalized (Q_real - Q_rand)/(Q_max - Q_rand) (secondary)
    LR / r    modularity at the PLANTED left/right split         (PRIMARY)
    purity    left/right purity of each hidden neuron's ancestry (PRIMARY)

Which are primary is not a style choice. experiment_1/RESULTS.md records a run
where `newman_q` returned Q = 0.20 with p = 0.87 against its own null, having
found a partition that had nothing to do with the retina's left/right structure
— a discovered partition can be real and irrelevant at the same time. The split
this task is *about* is known in advance, so score that one:
`left_right_q` needs no search and so has no optimiser to fail. Q_m is reported
because KA reported it, with the caveat KA's own data shows (and this repo
reproduced): it stops discriminating above ~50% density.

WHY PURITY NEEDED NEW CODE. `qmetrics.circuit_purity` raises
``ValueError("circuit_purity needs an acyclic graph")`` — it calls
`nx.topological_sort`, because it was built for experiment 4's Boolean circuits,
which are DAGs. Every brain in `experiments/` has a recurrent hidden->hidden
block, so it is cyclic by construction and the metric simply cannot run on one.

`recurrent_purity` below is that metric on the TIME-UNROLLED graph, which is
acyclic by construction: node (v, t) has parents (i, t-1) for every synapse
i->v, inputs are re-clamped at every step exactly as in `forward`, and the
network is unrolled for the same `rnn_iters` it is actually evaluated with. Each
node takes the MEAN of its parents, which is circuit_purity's rule verbatim, so
the two agree on any feedforward graph. Read at the final time slice, giving one
value per hidden neuron:

    purity(v) = 2 * |x_v - 0.5|    1 = ancestry all one side, 0 = balanced

Read it knowing what the unrolling does: influence mixes a little more at every
step, so purity decays with `rnn_iters` for any network that is not actually
separated. That is the honest reading — a dense recurrent net at 8 iterations
really does depend on both halves everywhere — but it means purity is comparable
only between networks unrolled the SAME number of steps.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmetrics import from_matrix, left_right_q, newman_q, normalized_qm  # noqa: E402


# ---------------------------------------------------------------------------
# Purity — the recurrence-safe replacement for qmetrics.circuit_purity
# ---------------------------------------------------------------------------

def recurrent_purity(w, n_in: int, n_hidden: int, rnn_iters: int = 8,
                     threshold: float = 0.0, weighted: bool = False):
    """Left/right purity of each hidden neuron, on the time-unrolled graph.

    w          (N, N) weight matrix, w[i, j] = edge i -> j (experiments' convention)
    n_in       input neurons, assumed [left half | right half] in index order
    n_hidden   hidden neurons, indices n_in .. n_in+n_hidden-1
    rnn_iters  unroll depth; MUST match the cfg the network was evaluated with
    threshold  |w| at or below this is not a synapse
    weighted   False: every parent counts equally, exactly as `circuit_purity`
               does (it was built for weightless gates). True: parents count in
               proportion to |w|, which uses information a Boolean circuit does
               not have. Both are returned by `score_weights`; the unweighted one
               is the like-for-like comparison with experiment 4 and KA.

    -> (purity, info). purity is the MEAN over hidden neurons; info carries
       `median` (prefer it when a few dead-but-pure neurons inflate the mean),
       `values` (per hidden neuron, for colouring a drawing), `flow` (the raw
       0..1 side mixture), and `n_undefined` (hidden neurons no input reaches).
    """
    w = np.asarray(w, dtype=np.float64)
    N = w.shape[0]
    active = np.abs(w) > threshold
    P = (np.abs(w) * active) if weighted else active.astype(np.float64)

    half = n_in // 2
    side = (np.arange(n_in) >= half).astype(np.float64)   # left -> 0.0, right -> 1.0

    # t = 0: only the clamped inputs carry a side; everything else is unreached.
    x = np.full(N, np.nan)
    x[:n_in] = side
    for _ in range(int(rnn_iters)):
        known = ~np.isnan(x)
        num = P[known, :].T @ x[known]          # (N,) sum of parent values * weight
        den = P[known, :].sum(axis=0)           # (N,) total parent weight
        with np.errstate(invalid="ignore", divide="ignore"):
            nxt = np.where(den > 0, num / den, np.nan)
        nxt[:n_in] = side                        # inputs are re-clamped every step
        x = nxt

    hid = x[n_in:n_in + n_hidden]
    vals = 2.0 * np.abs(hid - 0.5)
    finite = ~np.isnan(vals)
    purity = float(vals[finite].mean()) if finite.any() else float("nan")
    return purity, {
        "purity": purity,
        "median": float(np.median(vals[finite])) if finite.any() else float("nan"),
        "n_scored": int(finite.sum()),
        "n_undefined": int((~finite).sum()),
        "values": {int(n_in + i): float(v) for i, v in enumerate(vals) if np.isfinite(v)},
        "flow": {int(n_in + i): float(v) for i, v in enumerate(hid) if np.isfinite(v)},
    }


# ---------------------------------------------------------------------------
# All four metrics on one weight matrix
# ---------------------------------------------------------------------------

def role_allowed(n_in: int, n_hidden: int, n_out: int) -> np.ndarray:
    """(N, N) mask of edges the architecture permits: IH | HH | HO, no self-loops.

    Passed to `from_matrix` so a structurally impossible edge is never counted as
    an absent one -- density and every null model are over the ALLOWED edges.
    """
    N = n_in + n_hidden + n_out
    idx = np.arange(N)
    is_in = idx < n_in
    is_hid = (idx >= n_in) & (idx < n_in + n_hidden)
    is_out = idx >= n_in + n_hidden
    m = (np.outer(is_in, is_hid) | np.outer(is_hid, is_hid) | np.outer(is_hid, is_out))
    return m & ~np.eye(N, dtype=bool)


def score_weights(w, n_in: int, n_hidden: int, n_out: int, *,
                  rnn_iters: int = 8, threshold: float = 0.05,
                  n_rand: int = 200, qm: bool = True, lr: bool = True,
                  seed: int = 0, n_jobs: int = 1) -> dict:
    """Score one brain. Returns a flat dict of numbers, ready for a table row.

    threshold  |w| at or below this is not an edge. Default 0.05 is the repo's
               --prune-threshold, i.e. the same cut the logged `density` uses, so
               a metric and its density always describe the same graph. Note
               unconstrained runs converge on a near-complete graph at any
               sensible cut, which does not make them un-modular -- it makes the
               question unanswerable, and Q_m in particular goes undefined.
    n_rand     randomisations behind Q_m and the left/right p-value. 200 for a
               headline table; drop to ~50 when scoring a whole trajectory.
    n_jobs     worker processes for those null models. DEFAULT 1, deliberately:
               qmetrics defaults to -1 (every core), and calling it in a loop
               while training jobs are running oversubscribes the machine so
               badly that scoring makes no progress at all. Raise it only when
               the box is otherwise idle.
    qm / lr    turn off the expensive null models for a cheap density-only pass.
    """
    w = np.asarray(w, dtype=np.float64)
    allowed = role_allowed(n_in, n_hidden, n_out)
    present = (np.abs(w) > threshold) & allowed
    n_edges = int(present.sum())
    max_edges = int(allowed.sum())

    out = {
        "n_edges": n_edges,
        "max_edges": max_edges,
        "density": 100.0 * n_edges / max_edges if max_edges else float("nan"),
        "frac_inh": float((w[present] < 0).mean()) if n_edges else float("nan"),
    }

    # --- METRIC 4: purity (always cheap, always defined) -------------------
    p_u, info_u = recurrent_purity(w, n_in, n_hidden, rnn_iters, threshold, weighted=False)
    p_w, _ = recurrent_purity(w, n_in, n_hidden, rnn_iters, threshold, weighted=True)
    out["purity"] = p_u
    out["purity_median"] = info_u["median"]
    out["purity_weighted"] = p_w
    out["purity_undefined"] = info_u["n_undefined"]

    # TWO graphs, on purpose:
    #  * UNDIRECTED for Q and Q_m. Newman modularity is *defined* on an
    #    undirected graph, KA's Q_m is undirected, and `newman_q`'s greedy
    #    method refuses a DiGraph outright. from_matrix(directed=False) keeps
    #    edge i-j when either direction exists, at the larger |w|.
    #  * DIRECTED for left_right_q, matching experiment_1/score_2x2.py, which is
    #    how the existing 2x2 in add_to_latex.md was scored. Keeping the same
    #    convention means these numbers extend that table instead of being a
    #    second, incomparable set.
    # Building the graphs and running greedy modularity is cheap; only the
    # degree-preserving NULL MODELS behind Q_m and the left/right p-value are
    # expensive, so those are what `qm` / `lr` gate. Q is always computed, which
    # is what lets it be traced every generation in a switch-window figure.
    G_und = from_matrix(w, threshold=threshold, directed=False, weighted=True,
                        allowed=allowed)
    G_dir = from_matrix(w, threshold=threshold, directed=True, weighted=True,
                        allowed=allowed)

    # --- METRICS 1 + 2: discovered partition -------------------------------
    try:
        q, comms = newman_q(G_und)
        out["q"] = float(q)
        out["n_modules"] = len(comms)
    except Exception as e:                     # a complete or empty graph
        out["q"], out["n_modules"], out["q_error"] = float("nan"), 0, repr(e)

    if qm:
        try:
            q_m, parts = normalized_qm(G_und, n_rand=n_rand, seed=seed, n_jobs=n_jobs)
            out["q_m"] = float(q_m)
            out["q_real"], out["q_rand"] = float(parts["q_real"]), float(parts["q_rand"])
            out["q_max"] = float(parts["q_max"])
        except Exception as e:
            out["q_m"] = float("nan")
            out["qm_error"] = repr(e)

    # --- METRIC 3: PRIMARY, the planted left/right split -------------------
    if lr:
        pinned = {i: i // (n_in // 2) for i in range(n_in)}
        # The single output must read both halves by construction, so scoring it
        # charges a fixed penalty that has nothing to do with modularity.
        exclude = list(range(n_in + n_hidden, n_in + n_hidden + n_out))
        try:
            score, info = left_right_q(G_dir, pinned, exclude=exclude, assign="optimal",
                                       n_rand=n_rand, seed=seed, n_jobs=n_jobs)
            out["lr"] = float(score)
            for k in ("q", "r", "crosstalk", "p"):
                if k in info:
                    out[f"lr_{k}"] = float(info[k])
        except Exception as e:
            out["lr"] = float("nan")
            out["lr_error"] = repr(e)

    return out


METRIC_KEYS = ("lr", "purity", "q", "q_m")   # PRIMARY first -- see module docstring
