"""Community pictures. Separate module so `import qmetrics` stays matplotlib-free.

Two panels, because neither alone is enough on a grown brain: a node-link view
(readable to ~300 edges, a hairball past that) and a community-sorted adjacency
matrix, which stays readable at any density and is the one that shows whether the
diagonal blocks are actually solid.
"""

from __future__ import annotations

import os
import subprocess
import sys

import networkx as nx
import numpy as np

from .graph import describe, from_matrix
from .metrics import _unweighted, left_right_q, newman_q

CROSS = "#9467bd"     # every between-community edge, in both panels


def open_file(path: str) -> None:
    """Open `path` in the OS default viewer. Never raises -- it's a convenience."""
    try:
        if sys.platform == "win32":
            os.startfile(path)                                  # noqa: S606
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open",
                              path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def plot_communities(G, path: str, *, title: str = "", n_in: int = 0, n_out: int = 0,
                     method: str = "greedy", restarts: int = 1, seed: int = 0,
                     show: bool = True, dpi: int = 135, allowed=None,
                     communities=None, q=None, kind: str = "communities",
                     labels=None, note: str = ""):
    """Draw `G`'s communities and write a PNG to `path`. -> (Q, communities).

    `G` may be a graph or a weight matrix. `n_in`/`n_out` mark the I/O anchors:
    NDP's node order is [inputs, outputs, hidden], so inputs are 0..n_in-1 and
    outputs immediately follow. `show` opens the file when done.

    By default the partition drawn is exactly the one `newman_q` scores -- and
    therefore the one `normalized_qm` scores too, since it reuses q_real's
    partition. Pass `communities` (list of node sets) to draw a partition you
    already have instead, e.g. the one `left_right_q` returns in `groups`; then
    NO detection is run and Q is evaluated at that partition. `kind`/`labels`/
    `note` only change wording on the figure.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not isinstance(G, nx.Graph):
        G = from_matrix(G, allowed=allowed)
    if G.is_directed():
        G = nx.Graph(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    if m == 0:
        raise ValueError("nothing to draw: the graph has no edges")

    if communities is None:
        q, comms = newman_q(G, method=method, seed=seed, restarts=restarts)
        how = f"{method}; same partition Q_m uses"
    else:
        comms = [set(c) for c in communities if c]
        if q is None:
            q = float(nx.community.modularity(G, comms))
        how = "at the given partition, no search"
    comms = sorted(comms, key=lambda c: min(c))        # stable, input-ordered
    memb = {v: i for i, c in enumerate(comms) for v in c}
    cmap = plt.get_cmap("tab10" if len(comms) <= 10 else "tab20")
    col = {v: cmap(memb[v] % cmap.N) for v in G}
    within = sum(1 for u, v in G.edges() if memb[u] == memb[v])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8.2))

    # --- panel 1: node-link. Within-community edges bold and coloured, the rest
    # faint grey, so the eye can weigh one against the other.
    # inputs pinned in index order clockwise from the top, so 0..n_in-1 read round
    # the circle and each module's inputs form one contiguous arc.
    ang = np.pi / 2 - 2 * np.pi * np.arange(n_in) / max(n_in, 1)
    fixed = {v: (float(np.cos(a)), float(np.sin(a)))
             for v, a in zip(range(n_in), ang) if v in G}
    # Seed the free nodes on an outer ring inside their own community's arc.
    # Left to spring's random init they collapse into one central blob, which is
    # what the dense hidden-hidden wiring wants but tells you nothing.
    init = dict(fixed)
    for i, c in enumerate(comms):
        arc = [np.arctan2(fixed[v][1], fixed[v][0]) for v in sorted(c) if v in fixed]
        base = (np.arctan2(np.mean(np.sin(arc)), np.mean(np.cos(arc))) if arc
                else np.pi / 2 - 2 * np.pi * i / len(comms))
        free_c = [v for v in sorted(c) if v not in fixed]
        for j, v in enumerate(free_c):
            a = base + (j - (len(free_c) - 1) / 2) * 1.7 / max(len(free_c), 1)
            rad = 2.4 + 0.5 * (j % 3)
            init[v] = (float(rad * np.cos(a)), float(rad * np.sin(a)))
    pos = nx.spring_layout(G, pos=init or None, fixed=list(fixed) or None,
                           seed=1, k=3.4 / np.sqrt(n), iterations=400)
    same = [(u, v) for u, v in G.edges() if memb[u] == memb[v]]
    diff = [(u, v) for u, v in G.edges() if memb[u] != memb[v]]
    nx.draw_networkx_edges(G, pos, edgelist=diff, ax=ax1, edge_color=CROSS,
                           width=0.5, alpha=0.55)
    if same:
        nx.draw_networkx_edges(G, pos, edgelist=same, ax=ax1, width=1.2, alpha=0.9,
                               edge_color=[col[u] for u, _ in same])
    io = list(range(n_in + n_out))
    rest = [v for v in G if v not in io]
    nx.draw_networkx_nodes(G, pos, nodelist=rest, node_color=[col[v] for v in rest],
                           node_size=90, ax=ax1, edgecolors="white", linewidths=0.6)
    handles = [Line2D([], [], marker="o", ls="", mfc="0.6", mec="w", ms=8, label="hidden")]
    if n_in:
        ins = list(range(n_in))
        nx.draw_networkx_nodes(G, pos, nodelist=ins, node_shape="s", node_size=340,
                               node_color=[col[v] for v in ins], ax=ax1,
                               edgecolors="black", linewidths=1.4)
        nx.draw_networkx_labels(G, pos, {i: str(i) for i in ins}, font_size=9,
                                font_weight="bold", ax=ax1)
        handles.insert(0, Line2D([], [], marker="s", ls="", mfc="0.6", mec="k",
                                 ms=11, label=f"input 0-{n_in-1}"))
    if n_out:
        outs = list(range(n_in, n_in + n_out))
        nx.draw_networkx_nodes(G, pos, nodelist=outs, node_shape="*", node_size=900,
                               node_color=[col[v] for v in outs], ax=ax1,
                               edgecolors="black", linewidths=1.4)
        handles.insert(1, Line2D([], [], marker="*", ls="", mfc="0.6", mec="k",
                                 ms=17, label="output"))
    ax1.set_title(f"{len(comms)} {kind}: within-{kind[:-1]} edges take that "
                  f"{kind[:-1]}'s colour, between-{kind[:-1]} edges purple",
                  fontsize=11)
    ax1.axis("off")
    ax1.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    # --- panel 2: adjacency, rows/cols reordered so each community is contiguous.
    order = [v for c in comms for v in sorted(c)]
    idx = {v: i for i, v in enumerate(order)}
    import matplotlib.colors as mcolors
    xc = mcolors.to_rgb(CROSS)
    rgb = np.ones((n, n, 3))
    for u, v in G.edges():
        c = cmap(memb[u] % cmap.N)[:3] if memb[u] == memb[v] else xc
        rgb[idx[u], idx[v]] = c
        rgb[idx[v], idx[u]] = c
    ax2.imshow(rgb, interpolation="nearest")
    b = np.cumsum([0] + [len(c) for c in comms])
    for x in b[1:-1]:
        ax2.axhline(x - 0.5, color="k", lw=0.9)
        ax2.axvline(x - 0.5, color="k", lw=0.9)
    for i in range(len(comms)):
        name = labels[i] if labels and i < len(labels) else f"c{i}"
        ax2.text(-0.028 * n, (b[i] + b[i + 1] - 1) / 2, name, va="center",
                 ha="right", fontsize=9, color=cmap(i % cmap.N), fontweight="bold")
    ax2.set_title(f"adjacency sorted by {kind[:-1]} — coloured = within, purple = between\n"
                  f"{within}/{m} edges ({100*within/m:.0f}%) fall inside a {kind[:-1]}",
                  fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])

    d = describe(G)
    head = f"{title}   |   " if title else ""
    tail = f"\n{note}" if note else ""
    fig.suptitle(f"{head}{d['n_nodes']} nodes, {d['n_edges']} edges, "
                 f"density {d['density']:.3f}   |   Newman Q = {q:.4f} ({how})"
                 f"{tail}", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    if show:
        open_file(path)
    return q, comms


def plot_left_right(G, path: str, pinned: dict, *, exclude=(), n_in: int = 0,
                    title: str = "", names=("left", "right"), show: bool = True,
                    dpi: int = 135, allowed=None, **kw):
    """Draw the split `left_right_q` scores. -> (score, extras) -- its own return.

    Scores first, then draws the EXACT partition that was scored (`extras.groups`),
    so picture and number can never disagree. `exclude`d nodes -- pass the output --
    are absent from the drawing because they are absent from the computation.

        plot_left_right(G, "lr.png", {i: i // 4 for i in range(8)},
                        exclude=[8], n_in=8)

    `**kw` goes to `left_right_q` (assign, n_rand, seed, n_jobs, ...).
    """
    if not isinstance(G, nx.Graph):
        G = from_matrix(G, allowed=allowed)
    score, e = left_right_q(G, pinned, exclude=exclude, **kw)
    gid = e["groups"]
    if not gid:
        raise ValueError("nothing to draw: no edges after `exclude`")

    U = _unweighted(G)
    U.remove_nodes_from(list(exclude))
    ids = sorted(set(gid.values()))
    comms = [{v for v, g in gid.items() if g == i} for i in ids]
    lab = [names[i] if i < len(names) else f"g{i}" for i in range(len(ids))]
    pin = sorted(v for v in gid if v in pinned)
    note = (f"{kw.get('assign', 'optimal')}: r = {e['r']:+.3f}   "
            f"crosstalk = {e['crosstalk']:.3f}   z = {e['z']:+.2f}, p = {e['p']:.3f}"
            f"   |   {len(pin)} pinned, {e['n_free']} free"
            + (f"   |   {len(exclude)} node(s) excluded" if len(exclude) else ""))
    plot_communities(U, path, title=title, n_in=n_in, n_out=0, show=show, dpi=dpi,
                     communities=comms, q=e["q"], kind="modules", labels=lab,
                     note=note)
    return score, e
