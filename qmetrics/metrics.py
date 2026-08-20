"""The Q calculators. All of them take a graph from `graph.py` and nothing else.

THREE metrics (newman_q, normalized_qm, left_right_q) and two supporting tools
(threshold_sweep is a sweep over the first; role_segregation is a task-specific
proxy). Note: Newman Q is the objective function -- Louvain and greedy/CNM are
just search algorithms for it, so there is no separate "Louvain modularity", and
every Q here is a lower bound (NP-hard).

The split that matters: the first two DISCOVER a partition and so inherit an
NP-hard search; left_right_q SCORES one you already know and so has no optimiser
in it to fail. Prefer it whenever the question names its own modules.

Which one to reach for:

  newman_q          the default. "How modular is this graph?" One number.
  normalized_qm     when comparing brains of DIFFERENT density/size. Raw Q is
                    confounded by density, so a dense grown brain scores low even
                    when it IS structured; Q_m divides that confound out. Its
                    DIRECTION is reliable; its SCALE moves with the Q_max
                    estimator -- see its docstring.
  left_right_q      when the task names the modules ("are the left 4 and the
                    right 4 retina inputs in separate halves?"). No search, so no
                    budget and no convergence problem; ~10x cheaper than
                    normalized_qm. The published name for what it computes is a
                    PLANTED-bipartition score / discrete assortativity; it is
                    named for the question we ask it, not the other way round.
  threshold_sweep   when the brain is dense but many weights are weak. Answers
                    "is there structure hiding under a tail of near-zero edges,
                    or is it genuinely one blob?" -- which a single Q cannot.
  role_segregation  the older, weaker form of left_right_q's question: reads only
                    input->hidden weights and assumes a 2-way split at n_in//2.
                    Kept as a known-answer cross-check; prefer left_right_q.

Reference scale (Kashtan-Alon 2005 / Clune 2013): modular nets score Q ~ 0.4+,
non-modular ~ 0.15-0.2. Those numbers are for sparse nets -- read them next to
`graph.describe`, or use normalized_qm.
"""

from __future__ import annotations

import ast
import functools
import os
import time

import networkx as nx
import numpy as np

from .graph import _roles_array, describe, from_matrix


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


def _rewire(G: nx.Graph, rng, nswap: int, log: list | None = None,
            block: dict | None = None) -> int:
    """In-place degree-preserving rewiring. -> number of swaps actually made.

    Swap (a,b),(c,d) -> (a,d),(c,b): every node keeps its exact degree, and only
    WHERE the edges point changes -- which is the structure Q is meant to detect.
    Pass `log` to collect the (a,b,c,d) of each swap, so a caller can undo it
    instead of working on a copy.

    Pass `block` (node -> block id) to bias the walk toward that partition:
    swaps that would REDUCE the number of within-block edges are rejected. For a
    FIXED partition, Q = within/m - sum_c (k_c/2m)^2 and the second term is
    invariant under degree-preserving rewiring, so "maximise within-block edges"
    is not a proxy for maximising Q -- it is algebraically the same thing, and it
    costs O(1) per proposal instead of a whole community detection. This is what
    makes `_q_max_planted` ~1000x cheaper than hill-climbing on Q itself.

    Honours `G.graph['allowed']` (see graph.py) so the null model can only build
    networks the architecture could actually produce. Without it the null is
    nonsense: on the KA retina, 56% of unconstrained null edges are impossible
    (retina-to-retina, layer-skipping), so Q_rand describes the wrong ensemble.
    Written by hand because nx.double_edge_swap takes no legality predicate.

    Random draws are BATCHED. Acceptance is intrinsically low on a dense,
    degree-heterogeneous brain (~6% -- picking random *edges* over-samples hubs,
    which are already connected to each other), so the loop runs ~16 attempts per
    swap; drawing each of the 3 numbers per attempt as a separate numpy Generator
    call cost ~5us of Python->C dispatch and was ~90% of this function's runtime.
    """
    allowed = G.graph.get("allowed")
    edges = list(G.edges())
    n_e = len(edges)                                 # constant; hoisted out
    if n_e < 2:
        return 0
    done = tries = 0
    max_tries = 100 * max(nswap, 1)
    bsize = min(8192, max(64, 4 * nswap))            # one C call per `bsize` draws
    buf, k = (), 0
    while done < nswap and tries < max_tries:
        if k >= len(buf):                            # .tolist() -> real Python ints,
            buf = np.column_stack([                  # so indexing costs no boxing
                rng.integers(n_e, size=(bsize, 2)),
                rng.integers(2, size=(bsize, 1)),
            ]).tolist()
            k = 0
        i, j, flip = buf[k]
        k += 1
        tries += 1
        if i == j:
            continue
        (a, b), (c, d) = edges[i], edges[j]
        if flip:                                     # sample both pairings
            c, d = d, c
        if len({a, b, c, d}) < 4:                    # no self-loops / shared ends
            continue
        if allowed is not None and not (allowed[a, d] and allowed[c, b]):
            continue
        if G.has_edge(a, d) or G.has_edge(c, b):     # keep the graph simple
            continue
        if block is not None and (                   # O(1) Q test, see docstring
                (block[a] == block[d]) + (block[c] == block[b]) <
                (block[a] == block[b]) + (block[c] == block[d])):
            continue                                 # would move edges OUT of blocks
        G.remove_edge(a, b), G.remove_edge(c, d)
        G.add_edge(a, d), G.add_edge(c, b)
        edges[i], edges[j] = (a, d), (c, b)
        if log is not None:
            log.append((a, b, c, d))
        done += 1
    return done


def _unswap(G: nx.Graph, a, b, c, d) -> None:
    """Undo one `_rewire` swap, so a hill-climb needn't copy the graph to trial it."""
    G.remove_edge(a, d), G.remove_edge(c, b)
    G.add_edge(a, b), G.add_edge(c, d)


def _degree_preserving_random(G: nx.Graph, seed: int, swaps: int = 2) -> nx.Graph:
    """A random graph with the SAME degree sequence, and the same legal edges.

    `swaps` = attempted swaps per edge. Was 10; measured flat at 1/2/5/10 on both
    the 34-edge KA retina and a 251-edge NDP brain (q_rand moved <0.002, vs a
    0.007 spread between samples), so 2 is 5x cheaper for nothing.
    """
    H = nx.Graph(G)          # swaps need an undirected simple graph; copies .graph
    _rewire(H, np.random.default_rng(seed), swaps * H.number_of_edges())
    return H


def _unweighted(G: nx.Graph) -> nx.Graph:
    """Topology only: same nodes/edges, no weight attributes, `allowed` carried over.

    The null model rewires topology and `_rewire`'s new edges carry no weight, so
    a weighted input would give a weighted q_real against half-weighted nulls --
    incomparable. Stripping once here also spares every inner newman_q the
    weight lookup.
    """
    U = nx.Graph()
    U.add_nodes_from(G)
    U.add_edges_from(G.edges())
    U.graph.update(G.graph)
    return U


def _rand_q(args):
    """One null sample. Module-level and self-contained so it can be pickled."""
    G, seed, swaps, method, qseed = args
    return _q(_degree_preserving_random(G, seed, swaps), method, qseed)


def _climb_q(args):
    """One hill-climb restart -> best Q reached. Own RNG, so restarts are independent.

    `from_g` starts the climb at G itself instead of a random rewiring. Since the
    climb never accepts a decrease, that restart returns >= q_real BY CONSTRUCTION,
    which is what stops q_max collapsing onto q_real and making Q_m exactly 1.0
    (see normalized_qm). The other restarts still start random, so q_max is not
    confined to the basin G happens to sit in.
    """
    G, seed, steps, swaps, method, qseed, from_g = args
    rng = np.random.default_rng(seed)
    H = nx.Graph(G) if from_g else _degree_preserving_random(
        G, int(rng.integers(1 << 31)), swaps)
    cur = _q(H, method, qseed)
    for _ in range(steps):
        log = []
        if not _rewire(H, rng, 1, log):   # no legal swap available
            continue
        q2 = _q(H, method, qseed)
        if q2 >= cur:              # accept non-decreasing swaps (allow plateaus)
            cur = q2
        else:
            _unswap(H, *log[0])    # revert in place; cheaper than trialling a copy
    return cur


def _plant_blocks(G: nx.Graph, k: int, rng=None) -> dict:
    """Split the nodes into k blocks of equal DEGREE SUM (not equal size).

    Q for a fixed partition pays a penalty sum_c (k_c/2m)^2, which for a fixed
    total is minimised when the k_c are equal -- so balancing degree, not node
    count, is what makes the target partition the best one of its size. Greedy
    largest-first into the least-loaded block (the standard LPT heuristic).

    Blocks must also be FILLABLE: a block is only useful if its members are
    allowed to connect to each other. Degree balance alone is not enough on a
    constrained architecture -- on the layered KA retina it plants blocks whose
    internal edges are all illegal, so nothing can be rewired into them and q_max
    comes back <= q_real (measured: 2 of 10 champions returned nan). So prefer,
    among the under-loaded blocks, the one where v has the highest FRACTION of
    legal partners. On an unconstrained graph that fraction is 1.0 everywhere and
    this degrades exactly to plain LPT.

    `rng` randomises the tie-breaks, so the construction can be restarted. One
    deterministic pass is fragile on a small tightly-masked graph (23 nodes /
    32 edges with fan-in caps): it is a greedy heuristic, and a single bad early
    assignment is unrecoverable. Restarts are cheap -- the whole construction is
    O(nk) plus a rewire, not a community detection.
    """
    allowed = G.graph.get("allowed")
    load, members, blocks = [0] * k, [[] for _ in range(k)], {}
    target = 2 * G.number_of_edges() / k          # degree-sum each block should get
    jitter = (lambda: 0.0) if rng is None else (lambda: rng.random() * 1e-6)
    order = sorted(G.nodes(), key=lambda v: (-G.degree(v), jitter()))
    for v in order:
        open_ = [i for i in range(k) if load[i] < target] or list(range(k))
        def score(i):
            if allowed is None or not members[i]:
                return (1.0 + jitter(), -load[i])
            legal = sum(1 for u in members[i] if allowed[v, u])
            return (legal / len(members[i]) + jitter(), -load[i])
        i = max(open_, key=score)
        blocks[v] = i
        members[i].append(v)
        load[i] += G.degree(v)
    return blocks


def _q_max_planted(args):
    """Best Q reachable at this degree sequence, for ONE target block count.

    Rather than SEARCHING for a modular rewiring by trialling Q (the old
    hill-climb: one full community detection per proposed edge swap, and it never
    converged), CONSTRUCT one: plant a k-block partition and rewire toward it
    using the O(1) within-block test in `_rewire`. Two detections at the end
    instead of thousands, and the result does not depend on a step budget.

    Reports the larger of Q at the planted partition and Q at a detected one --
    detection may find a better split of the rewired graph, and q_max is a max,
    so taking it can only tighten the bound.
    """
    G, k, seed, sweeps, method, qseed, reps, iters = args
    m = G.number_of_edges()
    best = 0.0
    for r in range(max(reps, 1)):
        rng = np.random.default_rng(seed + 7919 * r)
        H = nx.Graph(G)
        blk = _plant_blocks(H, k, rng if r else None)   # rep 0 = deterministic
        for _ in range(max(iters, 1)):
            _rewire(H, rng, sweeps * m, block=blk)
            best = max(best, float(nx.community.modularity(
                H, [c for c in ({v for v, b in blk.items() if b == i}
                                for i in range(k)) if c])))
            # Re-plant on what the graph ACTUALLY grew into. The initial blocks
            # are a guess: on a constrained architecture (layered KA nets) many
            # of their internal edges are illegal, so the rewiring can only get
            # part way there. Detection says which partition the result really
            # has; targeting that next round lets the two converge on each other.
            q, comms = newman_q(H, method=method, seed=qseed)
            best = max(best, q)
            blk = {v: i for i, c in enumerate(comms) for v in c}
            k = len(comms)
    return best


@functools.lru_cache(maxsize=1)
def _spawn_is_safe() -> bool:
    """Would starting worker processes re-run the caller's script?

    Under the `spawn` start method (the only one on Windows) each worker imports
    the parent's `__main__`. If that is a script with no
    `if __name__ == "__main__":` guard, the worker re-executes the whole script
    instead of just running its task -- silently, with no exception, so a
    try/except fallback does NOT cover it (measured: an unguarded script ran
    itself 4x). Hence we check for the guard before spawning anything.

    The check parses the AST rather than searching the text: a first attempt
    grepped for the string and was fooled by a DOCSTRING that merely mentioned
    the guard, which ran that script 13x.
    """
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=False) != "spawn":
        return True                       # fork: child never re-imports __main__
    if multiprocessing.parent_process() is not None:
        return False                      # already a worker; never nest pools
    import __main__
    path = getattr(__main__, "__file__", None)
    if path is None:
        return True                       # REPL/notebook: no script to re-run
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return False                      # can't verify -> assume unsafe
    return any(isinstance(node, ast.If)                       # module level only
               and any(isinstance(x, ast.Name) and x.id == "__name__"
                       for x in ast.walk(node.test))
               for node in tree.body)


_POOL_OVERHEAD_S = 1.5   # measured: 22 spawn workers re-importing numpy/networkx


def _pmap(fn, args, n_jobs):
    """Parallel map over independent tasks, serial when that would not pay.

    Starting the pool costs a flat ~1.5s (Windows `spawn`: every worker re-imports
    numpy + networkx + this package). Sending the graph is NOT the expensive part
    -- measured 1.44s for 200 empty tasks vs 1.55s for 200 each carrying a
    251-edge graph and its mask -- so the only question is whether there is
    enough work to amortise the startup.

    So: run the first task (whose result we need anyway), extrapolate, and only
    spawn if the estimate clears the overhead with room to spare. Measured on the
    four real workloads, this picks correctly every time -- the two that gain
    1.8-3.7x go parallel, and the 2.0s one that was a wash (1.01x) stays serial
    instead of paying 1.5s for nothing.
    """
    args = list(args)
    if n_jobs == 1 or len(args) < 2 or not _spawn_is_safe():
        return [fn(a) for a in args]
    t0 = time.perf_counter()
    first = fn(args[0])
    if (time.perf_counter() - t0) * len(args) < 2 * _POOL_OVERHEAD_S:
        return [first] + [fn(a) for a in args[1:]]
    try:
        from concurrent.futures import ProcessPoolExecutor
        rest = args[1:]
        workers = min(len(rest), n_jobs if n_jobs > 0 else (os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return [first] + list(ex.map(fn, rest, chunksize=1))
    except Exception:
        return [first] + [fn(a) for a in args[1:]]


def normalized_qm(G: nx.Graph, n_rand: int = 200, method: str = "greedy",
                  seed: int = 0, swaps: int = 2, n_jobs: int = -1,
                  ks=(2, 3, 4, 6, 8, 12), sweeps: int = 5, reps: int = 2,
                  iters: int = 3, q_max: str = "plant",
                  restarts: int = 6, steps: int = 250):
    """METRIC 2 -- Kashtan-Alon's normalized modularity Q_m (their Eq. 2):

        Q_m = (Q_real - Q_rand) / (Q_max - Q_rand)

    Q_rand is the mean Q over degree-preserving randomizations, Q_max the highest
    Q reachable AT THE SAME DEGREE SEQUENCE. Subtracting Q_rand and dividing by
    the achievable range removes the density/size confound that makes raw Q
    incomparable across sparsities -- the whole problem with dense grown brains.

    -> (Q_m, dict). Always computed on the UNWEIGHTED, undirected graph: the null
    model is a rewiring of the topology. 0 = no better than chance, 1 = as modular
    as these degrees allow. The dict also carries `z` and `p`, which compare
    q_real to the null DISTRIBUTION rather than its mean -- they need no q_max, so
    when you only want "is this more modular than chance?" read those instead and
    ignore the rest.

    `q_max='plant'` (default) CONSTRUCTS the modular rewiring: for each block
    count in `ks`, plant a degree-balanced k-partition and rewire toward it using
    the O(1) within-block test in `_rewire` (exact for fixed-partition Q, see
    there), then detect once. `q_max='climb'` is the old hill-climb on Q itself
    -- one full community detection per proposed swap, `restarts` x `steps` of
    them. Keep it only to check 'plant' against; it is ~1000x more expensive, it
    never converged (measured on a 251-edge brain: Q_m 0.451/0.288/0.254/0.218 at
    1k/2k/4k/8k detections, still falling), and because bigger graphs exhaust a
    fixed budget sooner it inflates them -- reintroducing the very size bias Q_m
    exists to remove.

    + kills the density confound, so different-sparsity brains compare;
      answers the Guimera critique by construction; reorders what raw Q gets
      wrong (KA fg_seed3: Q=0.40 but Q_m=-0.07, i.e. random).
    - q_max is an NP-hard maximum and every estimator of it is a different lower
      bound, so Q_m is an UPPER bound whose VALUE MOVES whenever the maximiser
      changes: measured on the 10 KA Run-5 champions, MVG-vs-FG came out
      +0.229 (kashtan_alon/modularity.py, matching its log) / +0.257 ('climb')
      / +0.415 ('plant'). The DIRECTION was stable across all three; the
      magnitude was not. Quote Q_m for direction, not for scale, and prefer `z`
      when a stable number is what you need. Not bounded to [0,1]; nan when
      q_max ~ q_rand (degrees admit no modular arrangement); see `saturated`
      below; unweighted and undirected only.

    `sweeps`/`reps`/`iters` size the plant budget; the defaults were measured, not
    guessed. Raising them to 20/4/3 bought +5% q_max on a 251-edge brain and +2.4%
    on a 793-edge one for 8x and 4.5x the time -- past the defaults it is nearly
    flat. At the defaults the remaining cost is the `n_rand` nulls, not the plant.

    `n_jobs` spreads the independent work over processes (-1 = all cores).
    """
    U = _unweighted(G)
    rng = np.random.default_rng(seed)
    q_real = _q(U, method, seed)

    # Seeds are drawn up front, in the original order, so the result depends on
    # `seed` alone and not on how the work happened to be scheduled.
    rand_seeds = [int(rng.integers(1 << 31)) for _ in range(n_rand)]
    climb_seeds = [int(rng.integers(1 << 31)) for _ in range(restarts)]

    rands = np.array(_pmap(_rand_q, [(U, s, swaps, method, seed)
                                     for s in rand_seeds], n_jobs))
    q_rand = float(rands.mean()) if len(rands) else 0.0
    # The null's SPREAD, not just its mean. Free (the samples are already paid
    # for) and it needs no q_max, so it has no optimiser in it to fail.
    sd = float(rands.std(ddof=1)) if len(rands) > 1 else 0.0
    z = (q_real - q_rand) / sd if sd > 0 else float("nan")
    p = float((rands >= q_real).sum() + 1) / (len(rands) + 1) if len(rands) else float("nan")

    qmax = q_real
    if U.number_of_edges() >= 2:
        if q_max == "plant":
            ks = [k for k in ks if 2 <= k <= U.number_of_nodes()]
            got = _pmap(_q_max_planted,
                        [(U, k, seed, sweeps, method, seed, reps, iters)
                         for k in ks], n_jobs)
        elif q_max == "climb":
            # Restart 0 climbs from G itself (-> >= q_real always); the rest from
            # random rewirings, so q_max isn't just G's own local basin.
            got = _pmap(_climb_q, [(U, s, steps, swaps, method, seed, i == 0)
                                   for i, s in enumerate(climb_seeds)], n_jobs)
        else:
            raise ValueError(f"unknown q_max {q_max!r} (want 'plant' or 'climb')")
        qmax = max([qmax] + list(got))

    # `saturated`: the maximiser found nothing more modular than G itself, so
    # q_max == q_real and the formula gives exactly 1.0. That is CORRECT for a
    # graph that really is optimal (the two-clique calibration graph is, and
    # scores 1.0), and MEANINGLESS when the maximiser merely failed. There is no
    # way to tell from the number alone -- hence the flag. Distrust a saturated
    # 1.0 on anything but a small graph; raise `reps`/`ks` and see if it moves.
    gain = qmax - q_real
    denom = qmax - q_rand
    q_m = (q_real - q_rand) / denom if abs(denom) > 1e-9 else float("nan")
    return float(q_m), {"q_real": q_real, "q_rand": q_rand, "q_max": qmax,
                        "gain": float(gain), "saturated": gain <= 1e-9,
                        "sd": sd, "z": float(z), "p": p}


def _parts_of(gid: dict) -> list:
    """node->group dict -> list of non-empty node sets, in group-id order."""
    out = {}
    for v, g in gid.items():
        out.setdefault(g, set()).add(v)
    return [out[g] for g in sorted(out)]


def _q_at(G: nx.Graph, gid: dict) -> float:
    """Newman Q of G at the partition `gid`. No search -- the partition is given."""
    return float(nx.community.modularity(G, _parts_of(gid)))


def _refine(G: nx.Graph, gid: dict, free: list, passes: int = 20) -> None:
    """Move the unpinned nodes to whichever group maximises Q. In place.

    Greedy single-node moves to a local optimum (Kernighan-Lin without the swap
    step). Deltas are computed incrementally: recomputing Q per trial move is
    O(m) and there are passes x |free| x |groups| of them.
    """
    ids = sorted(set(gid.values()))
    m = G.number_of_edges()
    if not free or m == 0:
        return
    e = dict.fromkeys(ids, 0)            # internal edges per group
    for u, v in G.edges():
        if gid[u] == gid[v]:
            e[gid[u]] += 1
    a = dict.fromkeys(ids, 0)            # degree sum per group
    for v, g in gid.items():
        a[g] += G.degree(v)
    pen = lambda x: (x / (2 * m)) ** 2
    for _ in range(passes):
        moved = False
        for v in free:
            kv = G.degree(v)
            cnt = dict.fromkeys(ids, 0)  # v's neighbours per group
            for u in G[v]:
                cnt[gid[u]] += 1
            src = gid[v]
            best_g, best_d = src, 0.0
            for g in ids:
                if g == src:
                    continue
                d = ((cnt[g] - cnt[src]) / m
                     - (pen(a[src] - kv) - pen(a[src]))
                     - (pen(a[g] + kv) - pen(a[g])))
                if d > best_d + 1e-12:
                    best_g, best_d = g, d
            if best_g != src:
                e[src] -= cnt[src]; e[best_g] += cnt[best_g]
                a[src] -= kv;       a[best_g] += kv
                gid[v] = best_g
                moved = True
        if not moved:
            break


def _assign(G: nx.Graph, pinned: dict, ids: list, mode: str, passes: int) -> dict:
    """Label every node: `pinned` as given, the rest by `mode`. -> node->group."""
    gid = {v: pinned[v] for v in G if v in pinned}
    free = [v for v in G if v not in gid]
    for v in free:                       # seed: majority side, ties -> smallest id
        cnt = dict.fromkeys(ids, 0)
        for u in G[v]:
            if u in gid:
                cnt[gid[u]] += 1
        gid[v] = max(ids, key=lambda g: (cnt[g], -g))
    if mode == "majority":
        for _ in range(passes):          # iterate to a fixed point
            changed = False
            for v in free:
                cnt = dict.fromkeys(ids, 0)
                for u in G[v]:
                    cnt[gid[u]] += 1
                g = max(ids, key=lambda g: (cnt[g], -g))
                changed |= g != gid[v]
                gid[v] = g
            if not changed:
                break
    elif mode == "optimal":
        _refine(G, gid, free, passes)
    elif mode != "none":
        raise ValueError(f"unknown assign {mode!r}")
    return gid


def _lr_null(args):
    """One null sample: rewire, then RE-RUN THE ASSIGNMENT on the rewired graph.

    Re-running it is the whole point. `assign='optimal'` fits the unpinned nodes
    to maximise Q, and that fitting finds structure in noise -- measured: a plain
    ER graph with 8 pinned nodes scores r = +0.25. Scoring the nulls at the
    partition fitted to the REAL graph would give the real graph the benefit of
    that fit and deny it to the null, so every graph would look significant. The
    null has to get the same opportunity to overfit.
    """
    G, pinned, ids, mode, passes, seed, swaps = args
    H = _degree_preserving_random(G, seed, swaps)
    return _q_at(H, _assign(H, pinned, ids, mode, passes))


def left_right_q(G: nx.Graph, pinned: dict, *, exclude=(), assign: str = "optimal",
                 n_rand: int = 200, sweeps: int = 5, swaps: int = 2, seed: int = 0,
                 n_jobs: int = -1, passes: int = 20):
    """METRIC 3 (EXPERIMENTAL) -- modularity at a partition you SPECIFY.

    Named for the question it was built for (retina: is the brain split LEFT vs
    RIGHT?), but it is not limited to two groups or to inputs -- `pinned` may name
    any number of groups over any nodes. In the literature this is a PLANTED (or
    prescribed) partition score; `r` below is Newman's discrete assortativity.

    ⚠️ EXPERIMENTAL: added 2026-08-07, not yet used for any logged result and not
    yet calibrated against a published number. The pieces are individually
    verified (see the `r == 1 - crosstalk` identity below, which the test suite
    checks), but treat the composite score as provisional.

    For the retina, "is this brain split into a LEFT and a RIGHT module?" is not a
    discovery question -- the task names the two groups. So skip community
    detection entirely: pin the nodes you know, label the rest, and evaluate Q
    there. That removes the NP-hard maximisation, the step budget and the
    non-convergence that make `normalized_qm` unstable. Typical call:

        left_right_q(G, {i: i // 4 for i in range(8)}, exclude=[8])  # 8 in, 1 out

    pinned  -- node -> group id for the nodes whose side is known (the inputs).
    exclude -- nodes dropped entirely. Pass the OUTPUT: a readout must connect to
               both halves by construction, so forcing it onto a side charges a
               fixed cross-edge penalty that has nothing to do with modularity.
    assign  -- how the unpinned (hidden) nodes get a side.
               'optimal'  greedy Q-maximising moves -> the BEST CASE for the
                          modularity hypothesis. If this is at chance, done.
               'majority' each node joins where it has the most edges, iterated
                          -- the naive reading. Report both as a bracket.
               'none'     every node must already be in `pinned`.

    -> (score, dict). Keys:
       q          Newman Q at the partition (no search, exact).
       r          Newman's discrete ASSORTATIVITY: q / (1 - sum a_i^2), i.e. q
                  over its analytic ceiling (every edge internal). Published,
                  named, needs no null. 1 = perfectly split, 0 = chance, <0 =
                  anti-associated.
       crosstalk  cross-group edges / cross-group edges expected at these
                  degrees. The one-line version for prose. Identity: r == 1 -
                  crosstalk, so these are one number in two dresses.
       q_rand     mean q over degree-preserving rewirings AT THE SAME PARTITION
                  (not re-detected -- that would be a different question).
       q_max      q of a rewiring that maximises within-group edges. The ceiling
                  r assumes is often unreachable under the degree sequence and
                  the `allowed` mask; this one is achievable, hence tighter.
       score      (q - q_rand) / (q_max - q_rand), the returned value.
       z, p       q against the null DISTRIBUTION, not just its mean.
       groups     node -> group id, the exact partition scored (excluded nodes
                  absent). Feed it to `plot.plot_left_right` to see it.

    + no community detection anywhere, so no optimiser to fail and no budget to
      under-spend; fast; the null and the ceiling use the same fixed partition,
      so the density confound divides out cleanly.
    - only answers the question you asked: a brain modular along some OTHER axis
      scores ~0 here, by design (use newman_q to discover). `assign='optimal'`
      is a greedy local optimum, so `score` is a lower bound. Undirected,
      unweighted.
    """
    U = _unweighted(G)
    U.remove_nodes_from(list(exclude))
    if U.number_of_edges() == 0:
        return float("nan"), {"q": 0.0, "r": float("nan"), "crosstalk": float("nan"),
                              "q_rand": 0.0, "q_max": 0.0, "z": float("nan"),
                              "p": float("nan"), "n_free": 0, "groups": {}}

    ids = sorted({pinned[v] for v in U if v in pinned}) or [0]
    n_free = sum(1 for v in U if v not in pinned)
    if n_free and assign == "none":
        raise ValueError(f"{n_free} nodes are not in `pinned`; "
                         "set assign='optimal' or 'majority'")
    gid = _assign(U, pinned, ids, assign, passes)

    m = U.number_of_edges()
    q = _q_at(U, gid)
    a = {g: 0 for g in ids}
    for v, g in gid.items():
        a[g] += U.degree(v)
    ceiling = 1.0 - sum((x / (2 * m)) ** 2 for x in a.values())   # all edges internal
    r = q / ceiling if ceiling > 1e-12 else float("nan")
    cross = sum(1 for u, v in U.edges() if gid[u] != gid[v]) / m
    crosstalk = cross / ceiling if ceiling > 1e-12 else float("nan")

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(1 << 31)) for _ in range(n_rand)]
    rands = np.array(_pmap(_lr_null,
                           [(U, pinned, ids, assign, passes, s, swaps)
                            for s in seeds], n_jobs))
    q_rand = float(rands.mean()) if len(rands) else 0.0
    sd = float(rands.std(ddof=1)) if len(rands) > 1 else 0.0
    z = (q - q_rand) / sd if sd > 0 else float("nan")
    p = float((rands >= q).sum() + 1) / (len(rands) + 1) if len(rands) else float("nan")

    H = nx.Graph(U)
    _rewire(H, np.random.default_rng(seed), sweeps * m, block=gid)
    q_max = max(_q_at(H, _assign(H, pinned, ids, assign, passes)), q)
    denom = q_max - q_rand
    score = (q - q_rand) / denom if abs(denom) > 1e-9 else float("nan")
    return float(score), {"q": q, "r": float(r), "crosstalk": float(crosstalk),
                          "q_rand": q_rand, "q_max": float(q_max), "sd": sd,
                          "z": float(z), "p": p, "n_free": n_free,
                          "sizes": [len(c) for c in _parts_of(gid)],
                          "groups": dict(gid)}


def circuit_purity(G: nx.DiGraph, pinned: dict, *, exclude=()):
    """METRIC 4 -- left/right purity of a LOGIC CIRCUIT's gates.

    Built for experiment 4's Boolean circuits (CGP/ECGP), where every node is a
    gate and its in-edges are its arguments -- the one graph shape the other three
    metrics fit badly, because a circuit is a DAG with tiny fixed in-degree rather
    than a community-structured network. Give the program inputs a side (left = 0,
    right = 1) and let every gate take the MEAN of its parents. A gate's value is
    then the probability that a backward random walk from it, choosing uniformly
    among parents at each step, lands on a right-hand input.

        purity(v) = 2 * |x_v - 0.5|     1 = all ancestors one side, 0 = balanced

    Circuit purity is the mean over the gates, excluding the inputs (`pinned`) and
    the program output (`exclude`) -- a single readout must see both halves by
    construction, so scoring it charges a fixed penalty unrelated to modularity,
    exactly as in `left_right_q`. The ideal `L AND R` circuit -- a left module, a
    right module, one gate joining them -- scores exactly 1.

    x_v is 0 or 1 exactly when every ancestor input is on one side (every parent
    carries positive weight), so the EXTREMES say only what an ancestor-set check
    already says. What this adds is the grading in between: a continuous reading of
    *how* mixed the impure gates are, which a left/right/mixed classification
    cannot give.

    Added 2026-08-19, calibrated against a size-matched random baseline 2026-08-20
    (38k unevolved circuits, active counts 2..50: mean falls 0.50 -> 0.39 and the SD
    tightens 6x, so read a score against its OWN size bucket). It is deliberately a
    WIRING statistic that ignores what the gates compute, so it is a proxy and not a
    ground truth -- see the minus list.

    ECGP: always measure the FLATTENED circuit. The averaging is not invariant
    under module compression (a 3-input module reads 1/3 per parent, while its
    arity-2 internals read 1/4, 1/4, 1/2), so a module-level graph and its own
    expansion score differently and CGP/ECGP would not be comparable.

    pinned  -- node -> side for the nodes whose side is known (the program
               inputs). Exactly two distinct ids; the smaller one maps to 0.0.
    exclude -- nodes dropped from the AVERAGE (pass the program output). They
               still propagate, which for an output node changes nothing.

    -> (purity, dict). Keys:
       purity     mean gate purity, the returned value.
       median     median gate purity. Prefer it when a circuit carries a long
                  pure-but-useless tail, which pulls the mean up.
       n_gates    gates that entered the average.
       n_orphan   unpinned gates with no parents (no side; skipped).
       values     node -> purity, for colouring a circuit diagram.
       flow       node -> x, the raw 0..1 side mixture behind `values`.

    + O(V+E), one topological sweep: no community detection, no null model and no
      truth tables, so it survives tasks far past the 2^n_in wall that stops any
      behavioural measure. Continuous, so it can be logged every generation and
      regressed against fitness. Per-node, so a circuit diagram can be coloured by
      it. Works at any arity, so ECGP modules need no special case.
    - Wiring only: a functionally dead argument still counts. Contamination DECAYS
      geometrically with depth, so a gate ten levels below a merge reads as nearly
      pure even though it depends on both halves -- right for OR-like gates, wrong
      for AND-like ones. That is the price of abstracting the gates away. It is a
      plain mean, so pure filler raises it, and its denominator is the gate count,
      so it is size-confounded like raw Q: compare only at matched circuit size.
    """
    if not G.is_directed():
        raise ValueError("circuit_purity needs a DiGraph -- it reads parents")
    ids = sorted({pinned[v] for v in G if v in pinned})
    if len(ids) != 2:
        raise ValueError(f"`pinned` must name exactly two sides, got {ids}")
    lo = ids[0]
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        raise ValueError("circuit_purity needs an acyclic graph") from None

    x, n_orphan = {}, 0
    for v in order:
        if v in pinned:
            x[v] = 0.0 if pinned[v] == lo else 1.0
            continue
        par = [x[u] for u in G.predecessors(v) if u in x]
        if not par:                       # a source that is not a program input
            n_orphan += 1
            continue
        x[v] = sum(par) / len(par)

    drop = set(exclude) | set(pinned)
    vals = {v: 2.0 * abs(x[v] - 0.5) for v in x if v not in drop}
    arr = np.array(list(vals.values()))
    purity = float(arr.mean()) if len(arr) else float("nan")
    return purity, {"purity": purity,
                    "median": float(np.median(arr)) if len(arr) else float("nan"),
                    "n_gates": len(vals), "n_orphan": n_orphan,
                    "values": vals, "flow": dict(x)}


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


def role_segregation(W, n_in: int, n_hidden: int, threshold: float = 0.0,
                     order: str = "iho"):
    """Known-answer check: are hidden neurons driven by ONE half of the inputs?

    Generalises the retina proxy in experiments/experiment_1/curriculum.py:202.
    For each hidden neuron, s = (L - R) / (L + R) over incoming |w| from the left
    vs right input half. s ~ +/-1 means the neuron belongs to one input module;
    s ~ 0 means it mixes both.

    `order` -- node layout, "iho" (default, experiments/1-3) or "ioh" (NDP); see
    graph._roles_array. This used to be hard-coded to "iho", which on an NDP
    brain sliced the OUTPUT column as if it were the first hidden neuron and
    dropped a real one -- silently, with no error. n_out is inferred from W.

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
    n_out = W.shape[0] - n_in - n_hidden
    if n_out < 0:
        raise ValueError(f"n_in={n_in} + n_hidden={n_hidden} exceeds "
                         f"W's {W.shape[0]} nodes")
    roles = _roles_array(n_in, n_hidden, n_out, order)
    half = n_in // 2
    W_in_hid = W[np.ix_(np.flatnonzero(roles == 0), np.flatnonzero(roles == 1))]
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
