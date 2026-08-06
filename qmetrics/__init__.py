"""Modularity (Q) metrics, shared across NDP / experiments 1-3 / kashtan_alon.

Two layers, deliberately kept apart:

    graph.py    whatever-format brain  ->  a networkx graph   (the only place
                that knows about any codebase's storage format)
    metrics.py  a networkx graph       ->  a Q number         (format-agnostic)

Add a new brain format? Add an adapter. Add a new metric? It works on every
format for free.

    from qmetrics import from_matrix, newman_q, threshold_sweep

    G = from_matrix(W, threshold=0.0)      # NDP's grown adjacency
    q, comms = newman_q(G)
    print(threshold_sweep(W))              # is structure hiding under weak edges?
"""

from .graph import (describe, from_blocks, from_matrix, layered_mask, role_mask,
                    roles_for)
from .metrics import (newman_q, normalized_qm, partition, role_segregation,
                      threshold_sweep)

__all__ = ["describe", "from_blocks", "from_matrix", "layered_mask", "role_mask",
           "roles_for", "newman_q", "normalized_qm", "partition",
           "role_segregation", "threshold_sweep"]
