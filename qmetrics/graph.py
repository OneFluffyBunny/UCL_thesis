"""Adapters: whatever a brain is stored as -> one canonical networkx graph.

Every codebase here stores its brain differently, so every one of them used to
grow its own modularity code. This module is the ONLY place that knows about the
formats; the metrics in `metrics.py` just take a graph.

  NDP/                one square W (n, n), signed float, directed or not
                      (edge iff |W| > thr -- matches train_backend.py:419 pruning)
  experiments/1,2,3   one square w, node order [inputs, hidden, outputs]
                      (edge iff |w| > thr -- matches visualize.py:48 brain_stats)
  kashtan_alon/       per-layer blocks + cfg.offsets, int8 in {-1, 0, +1}
                      (edge iff non-zero -- matches modularity.py:32)

Two conventions the metrics rely on, applied here once:

  * **|w|, not w.** Newman Q is undefined for negative weights, so a weighted
    graph carries abs(weight). Excitatory/inhibitory sign is a separate question
    from "who is wired to whom" -- keep it out of Q.
  * **self-loops dropped** by default. A self-loop sits inside its own community
    by construction and inflates Q for free.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def layered_mask(layers) -> np.ndarray:
    """Legal-edge mask for a layered feedforward net: ADJACENT layers only.
    `layers` is e.g. (8, 8, 4, 2, 1) -> a (23, 23) bool array."""
    layer_of = np.concatenate([[l] * n for l, n in enumerate(layers)])
    return np.abs(layer_of[:, None] - layer_of[None, :]) == 1


def role_mask(n_in: int, n_hidden: int, n_out: int,
              in_to_out: bool = False) -> np.ndarray:
    """Legal-edge mask for [inputs, hidden, outputs] nets: no input-input and no
    output-output edges (and no input-output unless `in_to_out`)."""
    r = np.array([0] * n_in + [1] * n_hidden + [2] * n_out)
    a, b = r[:, None], r[None, :]
    ok = ~(((a == 0) & (b == 0)) | ((a == 2) & (b == 2)))
    if not in_to_out:
        ok &= ~(((a == 0) & (b == 2)) | ((a == 2) & (b == 0)))
    np.fill_diagonal(ok, False)
    return ok


def from_matrix(W, *, threshold: float = 0.0, directed: bool = False,
                weighted: bool = False, drop_self_loops: bool = True,
                roles=None, allowed=None) -> nx.Graph:
    """Square signed adjacency (n, n) -> graph. Edge i->j iff |W[i, j]| > threshold.

    Covers NDP's grown `W` and experiment_1/2/3's `genome.build_weights(cfg)[0]`.

    threshold  -- weak edges below this are treated as absent (0.0 = keep all
                  non-zeros, the "no pruning" default).
    directed   -- keep edge direction (nx.DiGraph). If False, i-j exists when
                  either direction does, and its weight is the larger |w|.
    weighted   -- attach weight=|w|; otherwise the graph is purely topological.
    roles      -- optional per-node labels (see `roles_for`), stored as the
                  "role" node attribute so plots/reports can use them.
    allowed    -- (n, n) bool mask of which edges are STRUCTURALLY POSSIBLE in
                  this model (see `layered_mask` / `role_mask`). Carried on the
                  graph and used by normalized_qm, whose null model must not
                  invent edges the architecture forbids. Raises if W itself
                  violates it -- a wrong mask should fail loudly, not silently
                  produce a wrong Q_m.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {W.shape}")

    A = np.abs(W)
    if drop_self_loops:
        np.fill_diagonal(A, 0.0)
    A = np.where(A > threshold, A, 0.0)
    if not directed:
        A = np.maximum(A, A.T)

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(W.shape[0]))
    rows, cols = np.nonzero(A)
    if weighted:
        G.add_weighted_edges_from(
            (int(i), int(j), float(A[i, j])) for i, j in zip(rows, cols))
    else:
        G.add_edges_from((int(i), int(j)) for i, j in zip(rows, cols))

    if roles is not None:
        nx.set_node_attributes(G, {i: r for i, r in enumerate(roles)}, "role")

    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if not directed:                      # an undirected edge needs both ways legal
            allowed = allowed | allowed.T
        bad = [(u, v) for u, v in G.edges() if not allowed[u, v]]
        if bad:
            raise ValueError(f"{len(bad)} edges violate `allowed`, e.g. {bad[:3]} "
                             "-- the mask does not describe this network")
        G.graph["allowed"] = allowed
    return G


def from_blocks(blocks, offsets=None, constrain: bool = True, **kwargs) -> nx.Graph:
    """Layered feedforward blocks -> graph, by embedding them in one square matrix.

    `blocks[l]` has shape (layers[l], layers[l+1]); node `i` of layer `l` becomes
    global id `offsets[l] + i`. This is the kashtan_alon format -- pass
    `cfg.offsets`, or leave it None to derive offsets from the block shapes.

    `constrain` attaches the adjacent-layers-only mask automatically (a layered
    net cannot have any other edge), unless an explicit `allowed=` is given.
    Other keyword args are forwarded to `from_matrix`.
    """
    blocks = [np.asarray(b, dtype=float) for b in blocks]
    sizes = [b.shape[0] for b in blocks] + [blocks[-1].shape[1]]
    if offsets is None:
        offsets = np.cumsum([0] + sizes[:-1]).tolist()
    n = offsets[-1] + blocks[-1].shape[1]

    W = np.zeros((n, n))
    for l, B in enumerate(blocks):
        r, c = offsets[l], offsets[l + 1]
        W[r:r + B.shape[0], c:c + B.shape[1]] = B

    if constrain and kwargs.get("allowed") is None:
        kwargs["allowed"] = layered_mask(sizes)
    return from_matrix(W, **kwargs)


def roles_for(n_in: int, n_hidden: int, n_out: int) -> list:
    """Node-role labels for the [inputs, hidden, outputs] ordering used by
    experiments/experiment_1-3 (and by NDP's I/O anchor convention)."""
    return ["in"] * n_in + ["hidden"] * n_hidden + ["out"] * n_out


def describe(G: nx.Graph) -> dict:
    """Size/density of a graph, so a Q number is never read without its context.

    Q is strongly confounded by density -- a dense graph scores low no matter how
    it is wired -- so always report these alongside it (or use `normalized_qm`).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    max_edges = n * (n - 1) if G.is_directed() else n * (n - 1) // 2
    return dict(n_nodes=n, n_edges=m, directed=G.is_directed(),
                density=(m / max_edges if max_edges else 0.0),
                n_isolated=int(sum(1 for _, d in G.degree() if d == 0)))
