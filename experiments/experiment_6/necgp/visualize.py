"""Draw a necgp circuit, unflattened -- one box per genotype node, module calls
included -- with each box tagged by its NESTING FACTOR: `|1` for a primitive gate,
`|2` for a module built only from primitives, `|3` for a module that itself nests
a `|2` module, and so on. That is what plain ECGP's drawing (`experiment_4/visualize.py`,
which this file is forked from) cannot show: there, a module box is always exactly
one level deep by construction (`compress`'s no-nesting guard), so there was never
anything to distinguish. Here `compress` can absorb an existing module into a new
one (`ecgp.py`'s `nest_decay`-gated accept), so the same box shape -- a rectangle
with ports -- can now hide an arbitrary amount of internal structure, and the
suffix is the only thing on screen that says how much.

The nesting factor is `1 + Module.depth` for a module (`Module.depth` is stored
internally starting at 1 for a module built only from primitives, so this recovers
the user-facing "x=1 primitive, x=2 first module, x=3 second nested module, ..."
scale), and `1` for a plain gate.

Self-contained fork, `experiment_5`/`necgp`'s own convention: kept only the
UNFLATTENED (modular) renderer -- necgp has no flattened-circuit or cone-colour
use case yet -- plus the stage-grid machinery it needs. See
`experiment_4/visualize.py`'s own docstring for the full rationale behind the
"module = one box, with ports" drawing style; that reasoning is unchanged here.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INPUT_L = "#3498DB"
INPUT_R = "#E67E22"
EDGE = "#5D6D7E"
INPUT_SPLIT_GAP = 0.9      # extra y-gap between the left-4 and right-4 pixels
NEUTRAL = "#5D6D7E"        # box fill for a plain gate

# Module tags are drawn in their own colours, cycled by first appearance.
MODULE_COLOURS = ["#16A085", "#C0392B", "#8E44AD", "#D35400", "#2980B9", "#7F8C8D",
                  "#27AE60", "#B7950B"]

MOD_BOX_HW = 0.46          # module / gate box half-width in the modular drawing
PORT_GAP = 0.30            # vertical distance between two ports on the same box
X_STEP = 1.7               # horizontal distance between depth columns
PIN_LEN = 0.14             # how far a port pin sticks out past the box edge


def _fit_labels(fits) -> None:
    """Shrink any gate label that is wider than its box. Must run AFTER `tight_layout`.

    Copied unmodified from `experiment_4/visualize.py` -- see its docstring for why
    this has to run after layout and not be baked into a fixed font size.
    """
    for ax, labels, half_w in fits:
        fig = ax.figure
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        x0 = ax.transData.transform((0.0, 0.0))[0]
        x1 = ax.transData.transform((2 * half_w, 0.0))[0]
        avail = abs(x1 - x0) * 0.90
        for t in labels:
            w = t.get_window_extent(renderer).width
            if w > avail:
                t.set_fontsize(t.get_fontsize() * avail / w)


def _arity(ind, j: int) -> int:
    """How many of node `j`'s connection genes are actually read."""
    return 2 if ind.ntype[j] == 0 else ind.modules[ind.func[j]].n_in


def _n_out(ind, j: int) -> int:
    return 1 if ind.ntype[j] == 0 else ind.modules[ind.func[j]].n_out


def _nesting_factor(ind, j: int) -> int:
    """`1` for a plain gate; `1 + Module.depth` for a module call -- see module
    docstring for why this recovers the user-facing x=1/2/3/... scale."""
    if ind.ntype[j] == 0:
        return 1
    return 1 + ind.modules[ind.func[j]].depth


def _active_nodes(ind, n_in: int) -> set[int]:
    """The nodes reachable from the outputs, walking the GENOTYPE graph."""
    active: set[int] = set()
    stack = list(ind.ogene)
    while stack:
        lbl = stack.pop()
        if lbl < n_in:
            continue
        j = lbl - n_in
        if j in active:
            continue
        active.add(j)
        stack.extend(ind.conn[j][:_arity(ind, j)])
    return active


def n_hidden_nodes(ind, n_in: int) -> int:
    """How many boxes the drawing has: every active node, gate or module call alike."""
    return len(_active_nodes(ind, n_in))


def _modular_geometry(ind, n_in: int, split: int | None):
    """Where every box, pin and port of the unflattened drawing goes."""
    active = _active_nodes(ind, n_in)

    depth: dict[int, int] = {}
    for j in sorted(active):
        srcs = [s for s in ind.conn[j][:_arity(ind, j)] if s >= n_in]
        depth[j] = 1 + max((depth[s - n_in] for s in srcs), default=-1)
    max_d = max(depth.values(), default=0)

    height = {j: max(0.38, PORT_GAP * max(_arity(ind, j), _n_out(ind, j)))
              for j in active}
    pos: dict[int, tuple[float, float]] = {}
    in_pos: dict[int, tuple[float, float]] = {}
    for i in range(n_in):
        y = (n_in - 1) / 2.0 - i
        if split is not None:
            y += INPUT_SPLIT_GAP / 2 if i < split else -INPUT_SPLIT_GAP / 2
        in_pos[i] = (0.0, y)

    by_depth: dict[int, list[int]] = {}
    for j in active:
        by_depth.setdefault(depth[j], []).append(j)
    for d, nodes in by_depth.items():
        nodes.sort()
        span = sum(height[j] for j in nodes) + 0.34 * (len(nodes) - 1)
        y = span / 2
        for j in nodes:
            y -= height[j] / 2
            pos[j] = ((d + 1) * X_STEP, y)
            y -= height[j] / 2 + 0.34

    def port(j: int, k: int, n: int, side: int, pin: bool = False) -> tuple[float, float]:
        x, y = pos[j]
        h = height[j]
        return (x + side * (MOD_BOX_HW + (PIN_LEN if pin else 0.0)),
                y + h / 2 - (k + 0.5) * h / n)

    return active, max_d, height, pos, in_pos, by_depth, port


def _modular_figsize(geom, n_in: int) -> tuple[float, float]:
    active, max_d, height, _, _, by_depth, _ = geom
    return (min(26.0, max(7.0, 1.9 * (max_d + 3))),
            max(5.0, 0.62 * max(n_in, max((sum(height[j] for j in v) * 1.9
                                           for v in by_depth.values()), default=1)) + 2.0))


def _render_modular(ax, ind, n_in: int, gates, n_prim: int, geom,
                    title: str = "", split: int | None = None,
                    mod_colour: dict[str, str] | None = None,
                    scale: float = 1.0) -> list:
    """Draw the unflattened genotype into an existing axes; return its label artists.

    Same contract as `experiment_4/visualize.py`'s `_render_modular`, plus the `|x`
    nesting-factor suffix on every box label (see module docstring).
    """
    import ecgp

    active, max_d, height, pos, in_pos, _, port = geom
    mod_colour = {} if mod_colour is None else mod_colour

    for j in sorted(active):
        ar = _arity(ind, j)
        for t in range(ar):
            s, so = ind.conn[j][t], ind.cout[j][t]
            src = in_pos[s] if s < n_in else port(s - n_in, so, _n_out(ind, s - n_in),
                                                 +1, pin=True)
            ax.add_patch(FancyArrowPatch(src, port(j, t, ar, -1, pin=True),
                                         arrowstyle="-|>",
                                         mutation_scale=9 * scale, color=EDGE,
                                         lw=1.0 * scale,
                                         alpha=0.75, shrinkA=2, shrinkB=2,
                                         connectionstyle="arc3,rad=0.03", zorder=1))

    for i in range(n_in):
        x, y = in_pos[i]
        col = INPUT_L if (split is not None and i < split) else (
            INPUT_R if split is not None else "#7F8C8D")
        ax.add_patch(FancyBboxPatch((x - 0.26, y - 0.17), 0.52, 0.34,
                                    boxstyle="round,pad=0.045", linewidth=1.3,
                                    facecolor="white", edgecolor=col, zorder=2))
        ax.text(x, y, f"p{i}", ha="center", va="center", fontsize=8.5 * scale,
                color=col, fontweight="bold", zorder=3)

    labels: list = []
    out_nodes = {l - n_in for l in ind.ogene if l >= n_in}
    for j in sorted(active):
        x, y = pos[j]
        h = height[j]
        if ind.ntype[j] == 0:
            base_name, col = gates[ind.func[j]].name.upper(), NEUTRAL
        else:
            base_name = ecgp.module_name(ind.func[j], n_prim)
            col = mod_colour.setdefault(
                base_name, MODULE_COLOURS[len(mod_colour) % len(MODULE_COLOURS)])
        name = f"{base_name}|{_nesting_factor(ind, j)}"
        is_out = j in out_nodes
        ax.add_patch(FancyBboxPatch((x - MOD_BOX_HW, y - h / 2), 2 * MOD_BOX_HW, h,
                                    boxstyle="round,pad=0.04",
                                    linewidth=2.4 if is_out else 1.3,
                                    facecolor=col, alpha=0.90,
                                    edgecolor="#2C3E50" if is_out else col, zorder=2))
        labels.append(ax.text(x, y, name, ha="center", va="center",
                              fontsize=8.0 * scale,
                              color="white", fontweight="bold", zorder=3))
        for k in range(_arity(ind, j)):
            px, py = port(j, k, _arity(ind, j), -1)
            ax.plot([px - PIN_LEN, px], [py, py], color="#2C3E50",
                    lw=3.0 * scale, solid_capstyle="butt", zorder=4)
        for k in range(_n_out(ind, j)):
            px, py = port(j, k, _n_out(ind, j), +1)
            ax.plot([px, px + PIN_LEN], [py, py], color="#2C3E50",
                    lw=3.0 * scale, solid_capstyle="butt", zorder=4)

    for o, (l, c) in enumerate(zip(ind.ogene, ind.ocout)):
        src = in_pos[l] if l < n_in else port(l - n_in, c, _n_out(ind, l - n_in),
                                              +1, pin=True)
        ax.add_patch(FancyArrowPatch((src[0] + 0.02, src[1]), (src[0] + 0.75, src[1]),
                                     arrowstyle="-|>", mutation_scale=13 * scale,
                                     color="#2C3E50", lw=1.6 * scale, zorder=3))
        ax.text(src[0] + 0.82, src[1], "out", ha="left", va="center",
                fontsize=9.5 * scale, fontweight="bold", color="#2C3E50", zorder=3)

    ax.set_title(title, fontsize=11 * scale)
    ax.set_xlim(-0.9, (max_d + 1) * X_STEP + 1.9)
    ys = [p[1] for p in in_pos.values()] + \
         [pos[j][1] + s * height[j] / 2 for j in active for s in (-1, 1)]
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.7)
    ax.axis("off")
    return labels


def draw_modular(ind, n_in: int, gates, n_prim: int, save_path,
                 title: str = "", split: int | None = None) -> str:
    """Draw the genotype without inlining: one box per genome node, `|x` nesting
    factor on every label. See module docstring."""
    geom = _modular_geometry(ind, n_in, split)
    fig, ax = plt.subplots(figsize=_modular_figsize(geom, n_in))
    labels = _render_modular(ax, ind, n_in, gates, n_prim, geom, title, split)
    fig.tight_layout()
    _fit_labels([(ax, labels, MOD_BOX_HW)])
    fig.savefig(save_path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


STAGE_ROWS, STAGE_COLS = 2, 3


def stage_title(gen: int, hits: int, n_patterns: int, hidden: int | None = None) -> str:
    """The caption under one stage panel: when, how well, how big."""
    head = (f"gen {gen}  |  accuracy {100 * hits / n_patterns:.2f}% "
            f"({hits}/{n_patterns})")
    return head if hidden is None else f"{head}  |  {hidden} logic gates"


def stage_grid(panels, gates, n_in: int, save_path, title: str = "",
               split: int | None = None, n_prim: int | None = None,
               rows: int = STAGE_ROWS, cols: int = STAGE_COLS) -> str:
    """Tile snapshots of ONE seed's history into a `rows` x `cols` sheet.

    Each panel is `(individual, _, title, _)` -- necgp only ever draws the
    unflattened view, so the second and fourth slots (`pheno`, `origin` in
    `experiment_4/visualize.py`'s version) are unused here and may be `None`.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9.0, rows * 6.0))
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    shared: dict[str, str] = {}
    fits = []
    for ax, panel in zip(flat, list(panels) + [None] * (rows * cols)):
        ax.axis("off")
        if panel is None:
            continue
        g, _pheno, ptitle, _origin = panel
        geom = _modular_geometry(g, n_in, split)
        fits.append((ax, _render_modular(ax, g, n_in, gates, n_prim, geom, ptitle,
                                         split, mod_colour=shared, scale=1.55),
                     MOD_BOX_HW))
    if title:
        fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.975) if title else None)
    _fit_labels(fits)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
