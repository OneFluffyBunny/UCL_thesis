"""Visualisation / stats for a grown brain (experiment 1).

Mirrors the LNDP visualiser style: inputs on the left, outputs on the right,
hidden neurons in a grid; excitatory edges orange, inhibitory blue, width by
|weight|. Hidden nodes are coloured by their evolved cell-TYPE (the thing that
matters for modularity here).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # no display; save to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# --- colour scheme -----------------------------------------------------------
# Node and edge colours are kept mutually distinct so nothing is ambiguous:
#   - input / output nodes have their own reserved colours
#   - excitatory / inhibitory edges have colours used by NOTHING else
#   - hidden cell-types draw from a palette that excludes all of the above
INPUT_COLOR = "#4A90D9"   # blue
OUTPUT_COLOR = "#E74C3C"  # red
EXC_EDGE = "#2ECC71"      # green  (excitatory, +)
INH_EDGE = "#34495E"      # slate  (inhibitory, -)
# hidden-type palette: avoids blue, red, green, slate (the four reserved above)
HIDDEN_PALETTE = [
    "#9B59B6",  # purple
    "#E67E22",  # orange
    "#F1C40F",  # gold
    "#E91E63",  # pink
    "#8B4513",  # brown
    "#95A5A6",  # grey
    "#FF69B4",  # hot pink
    "#00CED1",  # dark turquoise
]


def _hidden_color(t: int):
    return HIDDEN_PALETTE[t % len(HIDDEN_PALETTE)]


def brain_stats(genome, cfg, threshold: float) -> dict:
    """Edge/type statistics of the brain grown from `genome` (edges = |w| > thr)."""
    w = np.asarray(genome.build_weights(cfg)[0])
    hid_ids = np.asarray(genome.hidden_type_ids(cfg))
    present = np.abs(w) > threshold

    n_edges = int(present.sum())
    max_edges = (cfg.n_in * cfg.n_hidden
                 + cfg.n_hidden * (cfg.n_hidden - 1)
                 + cfg.n_hidden * cfg.n_out)
    density = 100.0 * n_edges / max_edges if max_edges else 0.0

    signed = w[present]
    n_exc = int((signed > 0).sum())
    n_inh = int((signed < 0).sum())
    counts = np.bincount(hid_ids, minlength=cfg.n_types).tolist()
    return dict(w=w, hid_ids=hid_ids, n_edges=n_edges, max_edges=max_edges,
                density=density, n_exc=n_exc, n_inh=n_inh, type_counts=counts)


def _auto_open(path):
    """Open an image with the OS default viewer (cross-platform, incl. WSL)."""
    import os, sys, subprocess
    abs_path = os.path.abspath(path)
    try:
        if sys.platform == "win32":
            os.startfile(abs_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", abs_path])
        else:
            try:
                win = subprocess.check_output(["wslpath", "-w", abs_path]).decode().strip()
                subprocess.run(["explorer.exe", win])
            except Exception:
                subprocess.run(["xdg-open", abs_path])
    except Exception as e:
        print(f"  (could not auto-open image: {e})")


def visualize_brain(genome, cfg, threshold, save_path, title="", open_after=False):
    """Draw the grown brain to `save_path` (PNG). Returns the stats dict."""
    st = brain_stats(genome, cfg, threshold)
    w, hid_ids = st["w"], st["hid_ids"]
    n_in, n_hid, n_out = cfg.n_in, cfg.n_hidden, cfg.n_out
    n = cfg.n_total

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if abs(w[i, j]) > threshold:
                G.add_edge(i, j)

    signed = [w[i, j] for i, j in G.edges()]
    absw = [abs(s) for s in signed]
    w_max = max(absw) if absw else 1.0
    w_mean = float(np.mean(absw)) if absw else 0.0
    edge_widths = [0.5 + 3.5 * (a / w_max) for a in absw]
    edge_colors = [EXC_EDGE if s > 0 else INH_EDGE for s in signed]
    edge_labels = {(i, j): f"{w[i, j]:.2f}" for i, j in G.edges() if abs(w[i, j]) >= w_mean}

    # node colours: inputs/outputs reserved, hidden by cell-type (distinct palette)
    colors, labels = [], {}
    for i in range(n):
        if i < n_in:
            colors.append(INPUT_COLOR); labels[i] = f"in{i}"
        elif i >= n - n_out:
            colors.append(OUTPUT_COLOR); labels[i] = f"out{i - (n - n_out)}"
        else:
            t = int(hid_ids[i - n_in])
            colors.append(_hidden_color(t)); labels[i] = f"t{t}"

    # layout: inputs left, outputs right, hidden grid centred at x=0
    pos = {}
    for i in range(n_in):
        pos[i] = (-2.0, (n_in - 1) / 2.0 - i)
    for i in range(n_out):
        pos[n - n_out + i] = (2.0, (n_out - 1) / 2.0 - i)
    cols = max(2, int(np.ceil(np.sqrt(n_hid * 13 / 8))))
    rows = int(np.ceil(n_hid / cols))
    x_span, y_span = 2.8, max(n_in - 1, 2.0)
    for idx in range(n_hid):
        c, r = idx % cols, idx // cols
        x = -x_span / 2 + x_span * c / max(cols - 1, 1)
        y = y_span / 2 - y_span * r / max(rows - 1, 1)
        pos[n_in + idx] = (x, y)

    fig, ax = plt.subplots(figsize=(13, 8))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=350, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=12,
                           alpha=0.6, ax=ax, width=edge_widths,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, alpha=0.8, ax=ax)

    handles = [
        mpatches.Patch(color=INPUT_COLOR, label=f"Input ({n_in})"),
        mpatches.Patch(color=OUTPUT_COLOR, label=f"Output ({n_out})"),
        mpatches.Patch(color=EXC_EDGE, label=f"Excitatory (+) n={st['n_exc']}"),
        mpatches.Patch(color=INH_EDGE, label=f"Inhibitory (-) n={st['n_inh']}"),
    ]
    for t in range(cfg.n_types):
        if st["type_counts"][t] > 0:
            handles.append(mpatches.Patch(color=_hidden_color(t),
                                          label=f"hidden type {t} (n={st['type_counts'][t]})"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    ax.set_title(
        f"{title}\nnodes: {n}  edges: {st['n_edges']}  density: {st['density']:.1f}%  "
        f"type counts: {st['type_counts']}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved -> {save_path}")
    if open_after:
        _auto_open(save_path)
    return st
