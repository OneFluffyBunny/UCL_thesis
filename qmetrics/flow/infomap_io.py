"""Plumbing between our graphs and the `infomap` package. NOT a metric.

Nothing here returns a modularity score, a null model, or a normalisation. It
builds an Infomap object from a networkx graph, runs it, and hands back the raw
output (partition, codelength, level counts). What number to report -- and
against what null -- is deliberately left open.

Why it is separate from `metrics.py`: infomap is an OPTIONAL dependency with a
C++ core. `import qmetrics` must keep working without it, so the import happens
inside the functions and `qmetrics/__init__.py` does not pull this in.

    pip install infomap==2.8.1

2.8.1 is the last release with a cp310 wheel; on Python 3.10 anything newer
falls back to the sdist and needs a C++ toolchain.

Conventions kept identical to `graph.py`, so a graph means the same thing here
as everywhere else in the package:

  * **|w|, not w.** Infomap needs non-negative link weights, and edge sign is a
    separate question from who is wired to whom.
  * **self-loops dropped.** A self-loop is trivially inside its own module.

Direction is the whole point of this module: Infomap runs a random walk, so
reversing an edge changes the answer. That only buys you something if the walk
has somewhere to circulate -- on a feedforward DAG the walker drains into the
sinks and the result is governed by teleportation instead of by wiring. See
`flow_model` on `build` for the escape hatch.

TODO (when these become a reported metric): run `rawdir` alongside the default
`directed` and record BOTH numbers. The two treat weights differently -- see
FLOW_MODELS below -- and they are not comparable to each other, so a codelength
is meaningless without saying which model produced it. `directed` is the right
default for our brains (a neuron splits one activation among its targets, which
is what out-edge normalisation means), but it degenerates when a thresholded
brain goes acyclic: on the exp-1 brain at thr 0.2 `directed` atomised into 21
singletons while `rawdir` found 13 modules around a 9-node core.
"""

from __future__ import annotations

import networkx as nx

# Values this build of infomap accepts for --flow-model (verified against 2.8.1;
# an unknown one raises "Unrecognized flow model"). The two that matter here
# differ in how a weight becomes flow, measured on a lopsided 4-node test graph:
#   directed -- out-edges normalised per node into transition probabilities,
#               plus PageRank teleportation. Node flow tracked nx.pagerank(0.85)
#               to 0.051 (not exact: infomap uses unrecorded teleportation) and
#               MOVED with teleportation_probability (1.7358/1.7413/1.7038 at
#               0.05/0.15/0.50).
#   rawdir   -- raw link weights used directly as flow, no normalisation, no
#               teleportation. Node flow equalled raw in-strength share to
#               0.000000 and was UNCHANGED by teleportation_probability.
# Same graph, codelength 1.741317 vs 1.291692: never compare across models.
FLOW_MODELS = ("undirected", "directed", "undirdir", "outdirdir", "rawdir",
               "precomputed")


def available() -> bool:
    """Is the optional `infomap` package importable? Never raises."""
    try:
        import infomap  # noqa: F401
        return True
    except ImportError:
        return False


def version() -> str:
    """Version string of the installed infomap, or '' if absent."""
    try:
        from infomap import Infomap
        return str(Infomap().version)
    except ImportError:
        return ""


def _require():
    try:
        from infomap import Infomap
        return Infomap
    except ImportError as exc:                     # pragma: no cover
        raise ImportError(
            "the `infomap` package is not installed in this environment; "
            "`pip install infomap==2.8.1` (see qmetrics/flow/infomap_io.py)"
        ) from exc


def build(G: nx.Graph, *, weighted: bool = True, flow_model: str | None = None,
          num_trials: int = 10, seed: int = 1, two_level: bool = False,
          silent: bool = True, **kw):
    """networkx graph -> a loaded `Infomap` object, not yet run. -> (im, ids).

    `ids` maps infomap's integer node id back to the original node, so results
    can be relabelled. Nodes are added explicitly (in sorted order) rather than
    implied by links, so isolated nodes survive and the mapping is deterministic
    even after a `remove_nodes_from`.

    weighted    -- use the "weight" edge attribute (as |w|) if present. False
                   sends topology only, every link weight 1.0.
    flow_model  -- one of FLOW_MODELS, passed straight to Infomap. Left None,
                   the flow model is inferred from the graph: DiGraph ->
                   "directed", Graph -> "undirected". Pass it explicitly to
                   override (e.g. "rawdir" on an acyclic network).
    num_trials  -- Infomap is stochastic; this is its restart count, the same
                   discipline `newman_q(restarts=...)` needs for louvain. The
                   infomap default is 1, which is not enough on small graphs.
    seed        -- must be >= 1. Infomap rejects 0 with an unhelpful parse error
                   deep in its CLI layer, so it is caught here instead.
    two_level   -- forbid a hierarchy, giving a flat partition comparable to
                   what Louvain/greedy return.
    **kw        -- any other Infomap constructor argument (markov_time,
                   teleportation_probability, regularized, ...).

    Note `directed=` is NOT forwarded: it is Infomap's shorthand for
    flow_model="directed", and setting both invites a silent conflict.
    """
    Infomap = _require()
    if flow_model is None:
        flow_model = "directed" if G.is_directed() else "undirected"
    if flow_model not in FLOW_MODELS:
        raise ValueError(f"unknown flow_model {flow_model!r}; want one of "
                         f"{', '.join(FLOW_MODELS)}")
    if seed < 1:
        raise ValueError(f"seed must be >= 1, got {seed} (infomap rejects 0)")

    im = Infomap(flow_model=flow_model, num_trials=num_trials, seed=seed,
                 two_level=two_level, silent=silent, no_file_output=True, **kw)

    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    for v in nodes:
        im.add_node(idx[v])

    use_w = weighted and bool(nx.get_edge_attributes(G, "weight"))
    for u, v, data in G.edges(data=True):
        if u == v:                                  # self-loops carry no signal
            continue
        w = abs(float(data.get("weight", 1.0))) if use_w else 1.0
        if w > 0.0:                                 # infomap needs positive flow
            im.add_link(idx[u], idx[v], w)

    return im, {i: v for v, i in idx.items()}


def run(G: nx.Graph, *, weighted: bool = True, flow_model: str | None = None,
        num_trials: int = 10, seed: int = 1, two_level: bool = False,
        silent: bool = True, **kw) -> dict:
    """Build, run, and return Infomap's raw output relabelled to `G`'s nodes.

    -> dict:
      modules       {node: top-level module id}   (module ids are 1-based)
      multilevel    {node: tuple of module ids, root -> leaf}
      communities   [set of nodes], sorted by smallest member -- the shape
                    `plot_communities(communities=...)` wants
      codelength                bits per step of the map equation, the quantity
                                Infomap MINIMISES. Lower = better compression.
      one_level_codelength      the same graph as a single module, i.e. the
                                do-nothing baseline codelength is measured
                                against. Not a normalisation -- just the pair
                                you need to see together.
      relative_savings          (one_level - codelength) / one_level, as infomap
                                reports it.
      num_modules, num_levels, n_nodes, n_links
      flow_model, num_trials, seed, weighted   -- echoed back, because every one
                                of them changes the answer.

    Arguments are `build`'s. No score is derived here on purpose.
    """
    im, ids = build(G, weighted=weighted, flow_model=flow_model,
                    num_trials=num_trials, seed=seed, two_level=two_level,
                    silent=silent, **kw)
    im.run()

    modules = {ids[i]: m for i, m in im.get_modules().items()}
    multilevel = {ids[i]: tuple(t) for i, t in im.get_multilevel_modules().items()}
    by_mod: dict = {}
    for v, m in modules.items():
        by_mod.setdefault(m, set()).add(v)
    communities = sorted(by_mod.values(), key=min)

    one = float(im.one_level_codelength)
    cl = float(im.codelength)
    return {"modules": modules, "multilevel": multilevel,
            "communities": communities,
            "codelength": cl, "one_level_codelength": one,
            "relative_savings": float(im.relative_codelength_savings),
            "num_modules": int(im.num_top_modules),
            "num_levels": int(im.num_levels),
            "n_nodes": int(im.num_nodes), "n_links": int(im.num_links),
            "flow_model": (flow_model if flow_model is not None else
                           ("directed" if G.is_directed() else "undirected")),
            "num_trials": num_trials, "seed": seed, "weighted": weighted}
