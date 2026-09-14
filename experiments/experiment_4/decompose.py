"""Draw an ECGP individual two ways side by side: the unflattened circuit on the
left (module calls as single boxes, as `visualize.draw_modular` already draws),
and on the right, for every REAL module TYPE actually used in the active circuit,
its own internal wiring -- the flattened primitive graph inside that box.

A "fake" module (`ecgp.is_fake_module`) has no internal gate interaction -- a
single primitive gate, or several gates that never actually feed each other.
`visualize._box_style` already greys those out on the left, so a fake module gets
no right-hand panel: the grey box already says everything there is to say about
it. Only genuinely interacting modules -- the ones actually acting as a reusable
subroutine rather than a gate wearing a box -- earn a decomposition panel.

This is the general-purpose tool: it takes any already-built `ecgp.Individual` and
draws it. Reproducing a SPECIFIC run's final individual (e.g. replaying seed 0's
RNG because its checkpoint was deleted) is the caller's job -- see
`scratch_decompose_final.py` for that one-off replay, which imports and calls
`draw_decomposition` below. Import this module from anywhere in the repo that
wants the same picture for a different individual/run.
"""

from __future__ import annotations

import pathlib
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import cgp
import ecgp
import visualize as viz_mod


def module_body_genotype(mod: ecgp.Module) -> cgp.Genotype:
    """A module's body as a plain flattened `cgp.Genotype` -- bodies are primitives
    only (no nesting), so this is a direct field relabelling, no inlining needed."""
    return cgp.Genotype(func=mod.func[:], ntype=[0] * len(mod.func),
                        conn=mod.conn[:], cout=[0] * len(mod.conn),
                        ogene=mod.out[:], ocout=[0] * len(mod.out), arity=2)


def render_module_body(ax, body: cgp.Genotype, pheno: cgp.Phenotype, gates, n_in: int,
                       title: str, box_colour: str) -> list:
    """A module's internal wiring, own small renderer rather than `visualize._render`.

    `_render` places an output's arrow immediately right of its source box, sized for
    a circuit where every output sits at the rightmost depth. A module output can tap
    a node that ALSO feeds a later node in the same body (e.g. a body where node 0 is
    both the input to node 1 and output 0) -- then that fixed-offset arrow lands right
    on top of the next node's box. Here every output instead lands on a fixed rail one
    column past the deepest node, with a curved leader back to wherever it is actually
    tapped from, so a mid-body tap draws cleanly instead of overlapping.
    """
    pos = viz_mod._layout(pheno, n_in, None)
    max_d = max((pheno.depth[j] for j in pheno.active), default=0)
    rail_x = max_d + 1.0

    for j in pheno.active:
        gate = gates[body.func[j]]
        for k in range(gate.arity):
            src = body.conn[j * body.arity + k]
            if src not in pos:
                continue
            ax.add_patch(FancyArrowPatch(
                pos[src], pos[n_in + j], arrowstyle="-|>", mutation_scale=11,
                shrinkA=16, shrinkB=16, color=viz_mod.EDGE, lw=1.1, alpha=0.75,
                connectionstyle="arc3,rad=0.06", zorder=1))

    for i in range(n_in):
        x, y = pos[i]
        ax.add_patch(FancyBboxPatch((x - 0.26, y - 0.17), 0.52, 0.34,
                                    boxstyle="round,pad=0.045", linewidth=1.3,
                                    facecolor="white", edgecolor="#7F8C8D", zorder=2))
        ax.text(x, y, f"p{i}", ha="center", va="center", fontsize=8.5,
                color="#7F8C8D", fontweight="bold", zorder=3)

    labels = []
    for j in pheno.active:
        x, y = pos[n_in + j]
        ax.add_patch(FancyBboxPatch((x - viz_mod.BOX_HW, y - 0.19), 2 * viz_mod.BOX_HW, 0.38,
                                    boxstyle="round,pad=0.05", linewidth=1.3,
                                    facecolor=box_colour, alpha=0.90,
                                    edgecolor=box_colour, zorder=2))
        labels.append(ax.text(x, y, gates[body.func[j]].name.upper(),
                              ha="center", va="center", fontsize=7.6,
                              color="white", fontweight="bold", zorder=3))

    # Each output gets its OWN row on the rail, spaced evenly, rather than landing at
    # its source node's y: two outputs tapped from nodes that happen to share a y (a
    # single-node-per-column chain centres every node at y=0, so this is common) would
    # otherwise both land on the rail at that same y and print their labels on top of
    # each other.
    n_out = len(pheno.out_nodes)
    slot_y = [(n_out - 1) / 2.0 - o for o in range(n_out)]
    for o, j in enumerate(pheno.out_nodes):
        src = pos[n_in + j] if j >= 0 else pos[body.ogene[o]]
        ty = slot_y[o]
        rad = 0.0 if abs(src[1] - ty) < 1e-9 else (0.2 if ty > src[1] else -0.2)
        ax.add_patch(FancyArrowPatch(
            (src[0] + viz_mod.BOX_HW, src[1]), (rail_x, ty), arrowstyle="-|>",
            mutation_scale=11, shrinkA=2, shrinkB=2, color="#2C3E50", lw=1.4,
            connectionstyle=f"arc3,rad={rad}", zorder=3))
        ax.text(rail_x + 0.1, ty, f"out{o}" if n_out > 1 else "out",
                ha="left", va="center", fontsize=8.5, fontweight="bold",
                color="#2C3E50", zorder=3)

    ax.set_title(title, fontsize=9.5, color=box_colour)
    ax.set_xlim(-0.9, rail_x + 1.1)
    ys = [p[1] for p in pos.values()] + slot_y
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.axis("off")
    return labels


def used_module_counts(ind: ecgp.Individual, n_in: int) -> Counter:
    """How many times each module TYPE is called by the active circuit's top-level
    nodes. (Plain ECGP module bodies are primitives-only -- no nesting -- so unlike
    necgp's `decompose.py`, there is no need to walk into a body to find further
    module calls: every call site lives at the top level.)"""
    active = ecgp.active_nodes(ind, n_in)
    return Counter(ind.func[j] for j in active if ind.ntype[j] != 0)


def draw_decomposition(ind: ecgp.Individual, n_in: int, gate_set, n_prim: int,
                       save_path, *, split: int | None = None,
                       circuit_title: str = "", caption_prefix: str = "") -> str:
    """Two-panel figure: left = unflattened circuit, right = one panel per REAL
    module type actually used, busiest first. Returns `save_path` (as a string).

    `circuit_title` captions the left panel (e.g. accuracy/seed); `caption_prefix`
    prefixes the figure's `suptitle`, ahead of the auto-generated real/fake tally.
    """
    counts = used_module_counts(ind, n_in)
    real_mids = [m for m in counts if not ecgp.is_fake_module(ind.modules[m])]
    fake_mids = [m for m in counts if m not in real_mids]
    used_mids = sorted(real_mids, key=lambda m: (-counts[m], m))

    n_right = len(used_mids)
    right_cols = 3
    right_rows = max(1, -(-n_right // right_cols))          # ceil
    panel_w, panel_h = 3.1, 2.35

    geom = viz_mod._modular_geometry(ind, n_in, split)
    left_w, left_h = viz_mod._modular_figsize(geom, n_in)
    right_w, right_h = right_cols * panel_w, right_rows * panel_h
    fig_w, fig_h = left_w + right_w + 0.3, max(left_h, right_h) + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=(left_w, right_w),
                          left=0.01, right=0.995, top=0.93, bottom=0.02, wspace=0.03)

    ax_left = fig.add_subplot(gs[0, 0])
    mod_colour: dict[str, str] = {}
    fits = [(ax_left, viz_mod._render_modular(
        ax_left, ind, n_in, gate_set, n_prim, geom, title=circuit_title,
        split=split, mod_colour=mod_colour, scale=1.0), viz_mod.MOD_BOX_HW)]

    gs_right = gs[0, 1].subgridspec(right_rows, right_cols, hspace=0.5, wspace=0.15)
    for k, mid in enumerate(used_mids):
        ax = fig.add_subplot(gs_right[k // right_cols, k % right_cols])
        mod = ind.modules[mid]
        body = module_body_genotype(mod)
        body_pheno = cgp.phenotype(body, mod.n_in, gate_set, None)
        name = ecgp.module_name(mid, n_prim)
        col = mod_colour.get(name, "#2C3E50")     # same colour as its box on the left
        title = (f"{name}  x{counts[mid]}  "
                f"({mod.n_in}-in, {mod.n_out}-out, {mod.n_nodes} nodes)")
        labels = render_module_body(ax, body, body_pheno, gate_set, mod.n_in, title, col)
        fits.append((ax, labels, viz_mod.BOX_HW))
    for k in range(n_right, right_rows * right_cols):
        fig.add_subplot(gs_right[k // right_cols, k % right_cols]).axis("off")

    fig.suptitle(
        f"{caption_prefix}grey module boxes have no internal gate interaction "
        f"({len(fake_mids)} fake of {len(counts)} module types) -- right panels, "
        "real modules only, busiest first",
        fontsize=12)
    fig.canvas.draw()                    # settles transData so `_fit_labels` can measure
    viz_mod._fit_labels(fits)
    save_path = pathlib.Path(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
