"""[necgp_pairwise copy of `necgp/decompose.py`, 2026-09-14; adds `mod_colour`.]

Draw a necgp (nested-ECGP) individual two ways side by side: the unflattened
circuit on the left (module calls as single boxes, as `visualize.draw_modular`
already draws), and on the right, one panel per REAL module TYPE actually
reachable in the active circuit -- its own body, one level unflattened (so a
nested call inside shows as its own box, same convention as the left side).

Generalises `experiment_4/decompose.py` for necgp's nesting: a module's body can
itself call another module (unlike plain ECGP, where a body is always
primitives-only), so panels are found by walking the active circuit RECURSIVELY
through module bodies, not just the top-level module calls -- a module that is
only ever reached by being nested inside another module still gets its own
panel. "Real" vs "fake" uses `ecgp.is_fake_module`, the same predicate
`visualize._box_style` already uses to grey boxes on the left, so a fake module
(no internal gate interaction, however many nesting levels it took to write it
down) gets no panel: the grey box on the left already says everything there is
to say about it.

This is the general-purpose tool: it takes any already-built `ecgp.Individual`
and draws it. Reproducing a SPECIFIC run's final individual is the caller's job
-- see `render.py`, which draws it for the final snapshot of a `train.py` run.
"""

from __future__ import annotations

import pathlib
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ecgp
import visualize as viz_mod


def _module_active_body(mod: ecgp.Module) -> set[int]:
    """Body-node indices reachable backward from `mod`'s own outputs, ONE level --
    a nested call is a stop here, not something to recurse through (recursion into
    nested bodies is `used_module_counts`'s job, over the whole individual)."""
    active: set[int] = set()
    stack = [lbl - mod.n_in for lbl in mod.out if lbl >= mod.n_in]
    while stack:
        b = stack.pop()
        if b in active:
            continue
        active.add(b)
        for lbl in mod.conn[b]:
            if lbl >= mod.n_in:
                stack.append(lbl - mod.n_in)
    return active


def used_module_counts(ind: ecgp.Individual, n_in: int) -> Counter:
    """How many times each module id is actually called, anywhere in the active
    circuit -- top level OR nested inside another active module's body -- so a
    module only ever reached by nesting still gets counted (and later, a panel)."""
    counts: Counter = Counter()

    def walk(func, ntype, active_idx, modules) -> None:
        for b in active_idx:
            if ntype[b] == 0:
                continue
            mid = func[b]
            counts[mid] += 1
            mod = modules[mid]
            walk(mod.func, mod.ntype, _module_active_body(mod), modules)

    walk(ind.func, ind.ntype, ecgp.active_nodes(ind, n_in), ind.modules)
    return counts


def draw_decomposition(ind: ecgp.Individual, n_in: int, gate_set, n_prim: int,
                       save_path, *, split: int | None = None,
                       circuit_title: str = "", caption_prefix: str = "",
                       mod_colour: dict[str, str] | None = None) -> str:
    """Two-panel figure: left = unflattened circuit, right = one panel per REAL
    module type reachable, busiest first. Returns `save_path` (as a string).

    `circuit_title` captions the left panel (e.g. accuracy/seed); `caption_prefix`
    prefixes the figure's `suptitle`, ahead of the auto-generated real/fake tally.
    """
    counts = used_module_counts(ind, n_in)
    real_mids = [m for m in counts
                if not ecgp.is_fake_module(ind.modules[m], ind.modules)]
    fake_mids = [m for m in counts if m not in real_mids]
    used_mids = sorted(real_mids, key=lambda m: (-counts[m], m))

    n_right = len(used_mids)
    right_cols = 3
    right_rows = max(1, -(-n_right // right_cols))          # ceil
    panel_w, panel_h = 3.4, 2.6

    geom = viz_mod._modular_geometry(ind, n_in, split)
    left_w, left_h = viz_mod._modular_figsize(geom, n_in)
    right_w, right_h = right_cols * panel_w, right_rows * panel_h
    fig_w, fig_h = left_w + right_w + 0.3, max(left_h, right_h) + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=(left_w, right_w),
                          left=0.01, right=0.995, top=0.93, bottom=0.02, wspace=0.03)

    ax_left = fig.add_subplot(gs[0, 0])
    mod_colour = {} if mod_colour is None else mod_colour
    fits = [(ax_left, viz_mod._render_modular(
        ax_left, ind, n_in, gate_set, n_prim, geom, title=circuit_title,
        split=split, mod_colour=mod_colour), viz_mod.MOD_BOX_HW)]

    gs_right = gs[0, 1].subgridspec(right_rows, right_cols, hspace=0.6, wspace=0.15)
    for k, mid in enumerate(used_mids):
        ax = fig.add_subplot(gs_right[k // right_cols, k % right_cols])
        mod = ind.modules[mid]
        mod_view = ecgp.Individual(func=mod.func, ntype=mod.ntype, conn=mod.conn,
                                   cout=mod.cout, ogene=mod.out, ocout=mod.ocout,
                                   modules=ind.modules, next_id=0)
        body_geom = viz_mod._modular_geometry(mod_view, mod.n_in, None)
        name = ecgp.module_name(mid, n_prim)
        col = mod_colour.get(name, "#2C3E50")
        title = (f"{name}  x{counts[mid]}  depth {mod.depth}  "
                f"({mod.n_in}-in, {mod.n_out}-out)")
        labels = viz_mod._render_modular(ax, mod_view, mod.n_in, gate_set, n_prim,
                                         body_geom, title=title, split=None,
                                         mod_colour=mod_colour, scale=0.85)
        ax.title.set_color(col)
        fits.append((ax, labels, viz_mod.MOD_BOX_HW))
    for k in range(n_right, right_rows * right_cols):
        fig.add_subplot(gs_right[k // right_cols, k % right_cols]).axis("off")

    fig.suptitle(
        f"{caption_prefix}grey boxes = no internal gate interaction "
        f"({len(fake_mids)} fake of {len(counts)} module types reachable) -- "
        "right panels, real modules only, busiest first, own body one level "
        "unflattened",
        fontsize=11.5)
    fig.canvas.draw()
    viz_mod._fit_labels(fits)
    save_path = pathlib.Path(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
