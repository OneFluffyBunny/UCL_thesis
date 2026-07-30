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


def density(weight_mats, cfg: NetConfig) -> float:
    """Percentage of possible feedforward connections that are present."""
    n_edges = int(sum(int(np.count_nonzero(W)) for W in weight_mats))
    return 100.0 * n_edges / cfg.max_edges if cfg.max_edges else 0.0


def n_edges(weight_mats) -> int:
    return int(sum(int(np.count_nonzero(W)) for W in weight_mats))
