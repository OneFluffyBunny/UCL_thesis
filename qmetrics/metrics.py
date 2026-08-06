"""The Q calculators. All of them take a graph from `graph.py` and nothing else.

Two real metrics (newman_q, normalized_qm); threshold_sweep is a sweep over the
first, role_segregation is a task-specific proxy. Note: Newman Q is the objective
function -- Louvain and greedy/CNM are just search algorithms for it, so there is
no separate "Louvain modularity", and every Q here is a lower bound (NP-hard).

Which one to reach for:

  newman_q          the default. "How modular is this graph?" One number.
  normalized_qm     when comparing brains of DIFFERENT density/size. Raw Q is
                    confounded by density, so a dense grown brain scores low even
                    when it IS structured; Q_m divides that confound out.
  threshold_sweep   when the brain is dense but many weights are weak. Answers
                    "is there structure hiding under a tail of near-zero edges,
                    or is it genuinely one blob?" -- which a single Q cannot.
  role_segregation  when you already know what the modules SHOULD be (e.g. the
                    retina's left/right input halves). A known-answer check to
                    validate the generic metrics against.

Reference scale (Kashtan-Alon 2005 / Clune 2013): modular nets score Q ~ 0.4+,
non-modular ~ 0.15-0.2. Those numbers are for sparse nets -- read them next to
`graph.describe`, or use normalized_qm.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .graph import describe, from_matrix


def partition(G: nx.Graph, method: str = "greedy", seed: int = 0,
              restarts: int = 1):
    """Split `G` into communities -- the SEARCH step, not the measurement.

    'greedy'   CNM: repeatedly merge the best pair. DETERMINISTIC, hence the
               default: reproducible, and it lines up with kashtan_alon/RESULTS.md.
               Easily trapped on large graphs. Undirected only.
    'louvain'  move nodes to the best neighbouring community, collapse, repeat.
               Best-of-40 beat greedy on 5/6 KA nets, but it is STOCHASTIC and a
               single run is unreliable on small graphs: seed-to-seed spread hit
               0.10 on the 23-node KA brains, as large as the MVG/FG effect
               itself. Raise `restarts` (Q is a maximisation, so best-of-N is a
               strictly tighter estimate). Required for directed graphs.
    """
    weight = "weight" if nx.get_edge_attributes(G, "weight") else None
    if method == "louvain":
        best = max((nx.community.louvain_communities(G, weight=weight, seed=seed + k)
                    for k in range(max(restarts, 1))),
                   key=lambda c: nx.community.modularity(G, c, weight=weight))
        return [set(c) for c in best]
    if method == "greedy":
        if G.is_directed():
            raise ValueError("'greedy' needs an undirected graph; use 'louvain'")
        return [set(c) for c in nx.community.greedy_modularity_communities(
            G, weight=weight)]
    raise ValueError(f"unknown method {method!r} (want 'louvain' or 'greedy')")


def newman_q(G: nx.Graph, method: str = "greedy", seed: int = 0,
             restarts: int = 1):
    """METRIC 1 -- Newman Q. -> (Q, communities). Edgeless graph = 0.0.

        Q = (1/2m) sum_ij [A_ij - k_i*k_j/(2m)] delta(c_i, c_j)

    = (edges inside communities) - (expected if rewired at random keeping
    degrees). The null term means a hub's many edges aren't evidence by themselves.
    DiGraph -> Leicht-Newman directed form automatically; weights used if present.

    + field standard (comparable to KA/Clune); fast; weighted + directed ok.
    - density confound: dense graphs score ~0 whatever their wiring (use Q_m);
      resolution limit (Fortunato 2007) hides modules under ~sqrt(2m) edges,
      ~9 edges at m=40 -- big for a small brain; partition-dependent lower bound;
      random graphs score nonzero anyway (Guimera 2004), so high Q isn't proof.
    """
    if G.number_of_edges() == 0:
        return 0.0, [set(G.nodes())]
    weight = "weight" if nx.get_edge_attributes(G, "weight") else None
    comms = partition(G, method=method, seed=seed, restarts=restarts)
    return float(nx.community.modularity(G, comms, weight=weight)), comms


def _q(G, method, seed):
    """Raw Q of a graph, ignoring the partition (0.0 if edgeless)."""
    return newman_q(G, method=method, seed=seed)[0]


def _rewire(G: nx.Graph, rng, nswap: int) -> int:
    """In-place degree-preserving rewiring. -> number of swaps actually made.

    Swap (a,b),(c,d) -> (a,d),(c,b): every node keeps its exact degree, and only
    WHERE the edges point changes -- which is the structure Q is meant to detect.

    Honours `G.graph['allowed']` (see graph.py) so the null model can only build
    networks the architecture could actually produce. Without it the null is
    nonsense: on the KA retina, 56% of unconstrained null edges are impossible
    (retina-to-retina, layer-skipping), so Q_rand describes the wrong ensemble.
    Written by hand because nx.double_edge_swap takes no legality predicate.
    """
    allowed = G.graph.get("allowed")
    edges = list(G.edges())
    if len(edges) < 2:
        return 0
    done = tries = 0
    max_tries = 100 * max(nswap, 1)
    while done < nswap and tries < max_tries:
        tries += 1
        i, j = int(rng.integers(len(edges))), int(rng.integers(len(edges)))
        if i == j:
            continue
        (a, b), (c, d) = edges[i], edges[j]
        if rng.random() < 0.5:                       # sample both pairings
            c, d = d, c
        if len({a, b, c, d}) < 4:                    # no self-loops / shared ends
            continue
        if allowed is not None and not (allowed[a, d] and allowed[c, b]):
            continue
        if G.has_edge(a, d) or G.has_edge(c, b):     # keep the graph simple
            continue
        G.remove_edge(a, b), G.remove_edge(c, d)
        G.add_edge(a, d), G.add_edge(c, b)
        edges[i], edges[j] = (a, d), (c, b)
        done += 1
    return done


def _degree_preserving_random(G: nx.Graph, seed: int) -> nx.Graph:
    """A random graph with the SAME degree sequence, and the same legal edges."""
    H = nx.Graph(G)          # swaps need an undirected simple graph; copies .graph
    _rewire(H, np.random.default_rng(seed), 10 * H.number_of_edges())
    return H


def normalized_qm(G: nx.Graph, n_rand: int = 100, restarts: int = 6,
                  steps: int = 250, method: str = "greedy", seed: int = 0):
    """Kashtan-Alon's normalized modularity Q_m (their Eq. 2):

        Q_m = (Q_real - Q_rand) / (Q_max - Q_rand)

    Q_rand is the mean Q over degree-preserving randomizations, Q_max the Q of a
    modularity-MAXIMIZING rewiring at the same degree sequence (hill-climb over
    edge swaps). Subtracting Q_rand and dividing by the achievable range removes
    the density/size confound that makes raw Q incomparable across sparsities --
    which is exactly the problem with dense grown brains.

    -> (Q_m, {q_real, q_rand, q_max}). Always computed on the UNWEIGHTED,
    undirected graph: the null model is a rewiring of the topology.
    Read as: 0 = no better than chance, 1 = as modular as these degrees allow.

    + kills the density confound, so different-sparsity brains compare;
      answers the Guimera critique by construction; reorders what raw Q gets
      wrong (KA fg_seed3: Q=0.40 but Q_m=-0.07, i.e. random).
    - expensive: n_rand + restarts*steps community detections, too slow for a
      fitness function at defaults; q_max is a stuck-prone hill-climb, so it is
      a lower bound and biases Q_m UP; not bounded to [0,1]; nan when
      q_max ~ q_rand (degrees admit no modular arrangement); unweighted only.
    """
    U = nx.Graph(G)
    rng = np.random.default_rng(seed)
    q_real = _q(U, method, seed)

    rands = [_q(_degree_preserving_random(U, int(rng.integers(1 << 31))), method, seed)
             for _ in range(n_rand)]
    q_rand = float(np.mean(rands)) if rands else 0.0

    q_max = q_real
    if U.number_of_edges() >= 2:
        for _ in range(restarts):
            H = _degree_preserving_random(U, int(rng.integers(1 << 31)))
            cur = _q(H, method, seed)
            for _ in range(steps):
                H2 = H.copy()                # .copy() carries the `allowed` mask
                if not _rewire(H2, rng, 1):  # no legal swap available
                    continue
                q2 = _q(H2, method, seed)
                if q2 >= cur:            # accept non-decreasing swaps (allow plateaus)
                    H, cur = H2, q2
            q_max = max(q_max, cur)

    denom = q_max - q_rand
    q_m = (q_real - q_rand) / denom if abs(denom) > 1e-9 else float("nan")
    return float(q_m), {"q_real": q_real, "q_rand": q_rand, "q_max": q_max}


def threshold_sweep(W, thresholds=None, *, quantiles=None, method: str = "greedy",
                    seed: int = 0, weighted: bool = False, normalized: bool = False,
                    **graph_kwargs):
    """Q as a function of how many weak edges you prune. The dense-brain diagnostic.

    A brain whose Q climbs sharply as the weak tail is cut HAS latent structure --
    the modules are there, buried under near-zero edges. A brain whose Q stays
    flat and low is genuinely unstructured. One Q at one arbitrary cutoff cannot
    tell those apart, which is the whole reason this exists.

    thresholds -- absolute |w| cutoffs. Defaults to `quantiles` of the non-zero
                  |w| distribution, which is scale-free and so comparable across
                  runs whose weights live on different scales.
    normalized -- also compute Q_m at each level (slow: it randomizes per level).

    -> list of dicts, one per level: threshold, quantile, Q, n_edges, density...

    + the one question a single Q provably cannot answer; scale-free by default
      so curves overlay across runs; no ground truth or up-front threshold needed.
    - pruning MANUFACTURES modularity (any graph -> Q~1 once fragmented), so a
      rising tail at extreme cutoffs is an artifact: read Q next to n_edges and
      distrust rows below ~2 edges/node. Rows aren't independent (each cutoff
      shifts the degree sequence -> use normalized=True if levels must compare).
      Gives a curve, not a number; where you read it off is a judgement call.
    """
    A = np.abs(np.asarray(W, dtype=float))
    np.fill_diagonal(A, 0.0)
    nz = A[A > 0]

    if thresholds is None:
        quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if quantiles is None else quantiles
        thresholds = [0.0 if q == 0.0 else float(np.quantile(nz, q)) for q in quantiles] \
            if nz.size else [0.0]
    qs = quantiles if quantiles is not None else [None] * len(thresholds)

    rows = []
    for thr, q in zip(thresholds, qs):
        G = from_matrix(W, threshold=thr, weighted=weighted, **graph_kwargs)
        row = dict(threshold=float(thr), quantile=q, **describe(G))
        row["Q"] = newman_q(G, method=method, seed=seed)[0]
        if normalized:
            row["Q_m"] = normalized_qm(G, method=method, seed=seed)[0]
        rows.append(row)
    return rows


def role_segregation(W, n_in: int, n_hidden: int, threshold: float = 0.0):
    """Known-answer check: are hidden neurons driven by ONE half of the inputs?

    Generalises the retina proxy in experiments/experiment_1/curriculum.py:202.
    For each hidden neuron, s = (L - R) / (L + R) over incoming |w| from the left
    vs right input half. s ~ +/-1 means the neuron belongs to one input module;
    s ~ 0 means it mixes both. Node order must be [inputs, hidden, outputs].

    + ground truth: the check that says whether a Q number is believable (Q says
      "modular" but every s ~ 0 -> one of them lies). No partition, so no
      resolution limit, no density confound, no seed. Per-neuron `per_hidden_s`
      shows the distribution: bimodal at +/-1 = modular, piled at 0 = tangled.
    - only where a left/right input split IS the ground truth (not cartpole /
      lunarlander / XOR); only reads input->hidden, ignoring hidden->hidden and
      hidden->out; assumes exactly 2 modules split at n_in//2; not comparable to
      any published Q -- never report it as "modularity" without saying so.
    """
    W = np.abs(np.asarray(W, dtype=float))
    half = n_in // 2
    W_in_hid = W[:n_in, n_in:n_in + n_hidden]
    left, right = W_in_hid[:half].sum(0), W_in_hid[half:].sum(0)
    tot = left + right

    active = tot > threshold
    s = np.where(tot > 1e-9, (left - right) / (tot + 1e-9), 0.0)
    if not active.any():
        return dict(mean_abs_s=0.0, frac_lateralized=0.0, n_active_hidden=0,
                    per_hidden_s=s)
    return dict(mean_abs_s=float(np.abs(s[active]).mean()),
                frac_lateralized=float((np.abs(s[active]) > 0.5).mean()),
                n_active_hidden=int(active.sum()), per_hidden_s=s)
