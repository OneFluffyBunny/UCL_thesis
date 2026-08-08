"""Modularity (Q) metrics, shared across NDP / experiments 1-3 / kashtan_alon.

Two layers, deliberately kept apart:

    graph.py    whatever-format brain  ->  a networkx graph   (the only place
                that knows about any codebase's storage format)
    metrics.py  a networkx graph       ->  a Q number         (format-agnostic)
    plot.py     a networkx graph       ->  a community picture

Add a new brain format? Add an adapter. Add a new metric? It works on every
format for free.

    from qmetrics import from_matrix, newman_q, plot_communities, threshold_sweep

    G = from_matrix(W, threshold=0.0)      # NDP's grown adjacency
    q, comms = newman_q(G)
    print(threshold_sweep(W))              # is structure hiding under weak edges?
    plot_communities(G, "brain.png", n_in=8, n_out=1)   # writes + opens the PNG

Never read a Q without looking at the picture: raw Q is confounded by density,
and the picture is what tells you whether the communities mean anything.

`newman_q`/`normalized_qm` DISCOVER a partition; `left_right_q` (EXPERIMENTAL)
scores one you already know -- "are the left 4 inputs and the right 4 inputs in
separate modules?" -- which needs no search and so has no optimiser to fail:

    score, e = qm.left_right_q(G, {i: i // 4 for i in range(8)}, exclude=[8])
    qm.plot_left_right(G, "lr.png", {i: i // 4 for i in range(8)},
                       exclude=[8], n_in=8)     # the same split, drawn

matplotlib is imported lazily inside plot.py, so it is not needed to use the
metrics. `normalized_qm` uses all cores by default (`n_jobs`); on Windows that
needs the calling script to have an `if __name__ == "__main__":` guard, and it
falls back to serial if not.
"""

from .graph import (describe, from_blocks, from_matrix, layered_mask, role_mask,
                    roles_for)
from .metrics import (left_right_q, newman_q, normalized_qm, partition,
                      role_segregation, threshold_sweep)
from .plot import open_file, plot_communities, plot_left_right

__all__ = ["describe", "from_blocks", "from_matrix", "layered_mask", "role_mask",
           "roles_for", "newman_q", "normalized_qm", "partition", "left_right_q",
           "role_segregation", "threshold_sweep", "plot_communities",
           "plot_left_right", "open_file"]
