"""Newman's Q modularity for an evolved Kashtan-Alon network.

This is the metric CLAUDE.md flags as the project's #1 missing tool. It is kept
framework-agnostic (it takes an individual's integer weight matrices, nothing
JAX/GA-specific) so it can later be promoted to a shared module and reused to
score experiment 1-3 networks too.

Q is computed on the UNDIRECTED connection graph over ALL neurons (inputs, hidden,
output): nodes are wired iff the connection weight is non-zero. A greedy
modularity community split is used to partition the graph, then Q = (fraction of
edges inside communities) - (expected fraction if edges were random). Modular
networks score Q ~ 0.4+, non-modular ~ 0.15-0.2 (Kashtan-Alon / Clune).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from model import NetConfig


def to_graph(weight_mats, cfg: NetConfig, weighted: bool = False) -> nx.Graph:
    """Undirected graph over all neurons; an edge exists where |weight| > 0.

    `weight_mats` is one individual's list of int matrices (from model.individual).
    Node ids are global: layer l, unit i -> cfg.offsets[l] + i.
    """
    G = nx.Graph()
    G.add_nodes_from(range(cfg.n_nodes))
    off = cfg.offsets
    for l, W in enumerate(weight_mats):
        src_base, dst_base = off[l], off[l + 1]
        rows, cols = np.nonzero(W)
        for i, j in zip(rows.tolist(), cols.tolist()):
            w = float(W[i, j])
            if weighted:
                G.add_edge(src_base + i, dst_base + j, weight=abs(w))
            else:
                G.add_edge(src_base + i, dst_base + j)
    return G


def newman_q(weight_mats, cfg: NetConfig, weighted: bool = False):
    """Return (Q, communities) for one individual. Isolated nodes are kept; a
    graph with no edges has Q = 0.0 by convention."""
    G = to_graph(weight_mats, cfg, weighted=weighted)
    if G.number_of_edges() == 0:
        return 0.0, [set(G.nodes())]
    weight = "weight" if weighted else None
    communities = list(nx.community.greedy_modularity_communities(G, weight=weight))
    q = nx.community.modularity(G, communities, weight=weight)
    return float(q), communities


def _q_of_graph(G: nx.Graph) -> float:
    """Raw Newman Q of the max-modularity greedy partition (0.0 if no edges)."""
    if G.number_of_edges() == 0:
        return 0.0
    comm = list(nx.community.greedy_modularity_communities(G))
    return float(nx.community.modularity(G, comm))


def _degree_preserving_random(G: nx.Graph, seed: int) -> nx.Graph:
    """A random simple graph with the SAME degree sequence (double-edge-swap)."""
    H = G.copy()
    m = H.number_of_edges()
    if m < 2:
        return H
    try:
        nx.double_edge_swap(H, nswap=10 * m, max_tries=200 * m, seed=seed)
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        pass   # too few swappable edges; H stays as-is (best effort)
    return H


def normalized_qm(weight_mats, cfg: NetConfig, n_rand: int = 100,
                  restarts: int = 6, steps: int = 250, seed: int = 0):
    """Kashtan-Alon's NORMALIZED modularity Q_m (their Eq. 2):

        Q_m = (Q_real - Q_rand) / (Q_max - Q_rand)

    where Q_real is the network's greedy Newman Q, Q_rand the mean greedy Q over
    degree-preserving randomizations, and Q_max the greedy Q of a modularity-
    MAXIMIZING rewiring at the same degree sequence (hill-climb over edge swaps).
    The (subtract Q_rand, divide by Q_max-Q_rand) normalization removes the
    density/size confound that makes raw Q incomparable across sparsities.

    Returns (Q_m, {q_real, q_rand, q_max}). Computed on the undirected, unweighted
    connection graph over all neurons -- same graph as newman_q."""
    G = to_graph(weight_mats, cfg, weighted=False)
    rng = np.random.default_rng(seed)

    q_real = _q_of_graph(G)

    # Q_rand: average over degree-preserving random graphs.
    rands = [_q_of_graph(_degree_preserving_random(G, int(rng.integers(1 << 31))))
             for _ in range(n_rand)]
    q_rand = float(np.mean(rands)) if rands else 0.0

    # Q_max: hill-climb Q over degree-preserving swaps (several random restarts).
    q_max = q_real
    m = G.number_of_edges()
    if m >= 2:
        for _ in range(restarts):
            H = _degree_preserving_random(G, int(rng.integers(1 << 31)))
            cur = _q_of_graph(H)
            for _ in range(steps):
                H2 = H.copy()
                try:
                    nx.double_edge_swap(H2, nswap=1, max_tries=100,
                                        seed=int(rng.integers(1 << 31)))
                except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                    continue
                q2 = _q_of_graph(H2)
                if q2 >= cur:            # accept non-decreasing swaps (allow plateaus)
                    H, cur = H2, q2
            q_max = max(q_max, cur)

    denom = q_max - q_rand
    q_m = (q_real - q_rand) / denom if abs(denom) > 1e-9 else float("nan")
    return float(q_m), {"q_real": q_real, "q_rand": q_rand, "q_max": q_max}


def density(weight_mats, cfg: NetConfig) -> float:
    """Percentage of possible feedforward connections that are present."""
    n_edges = int(sum(int(np.count_nonzero(W)) for W in weight_mats))
    return 100.0 * n_edges / cfg.max_edges if cfg.max_edges else 0.0


def n_edges(weight_mats) -> int:
    return int(sum(int(np.count_nonzero(W)) for W in weight_mats))
