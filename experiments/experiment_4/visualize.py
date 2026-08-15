"""Draw an evolved CGP circuit in the style of the CGP / Kashtan-Alon figures.

Horizontal layout, matching the CGP papers: program inputs on the LEFT, data
flowing left-to-right through gate boxes, the program output on the RIGHT. Only
the **phenotype** is drawn -- the active nodes -- which is what the CGP papers
show; the inactive genes are the (large) rest of the genotype and would swamp the
picture.

Colour carries the modularity claim, the way node colour does in the KA figures.
Each gate is coloured by where its input cone comes from:

    blue    left  -- depends only on retina pixels 0-3
    orange  right -- depends only on pixels 4-7
    purple  mixed -- depends on both halves
    grey    const -- depends on neither

A MODULAR solution is then obvious at a glance: a blue subtree and an orange
subtree running in parallel, meeting in a few purple nodes near the output. A
smeared solution is purple almost everywhere. Retina pixels are drawn with a gap
between the left and right blocks, as in `kashtan_alon/visualize.py`.

The title carries the accuracy, so a saved frame is self-describing.

Two renderers, one style. `draw` renders the FLATTENED circuit -- every node a plain
gate -- and is what the CGP arm gets. `draw_modular` renders the genotype UNFLATTENED,
one box per node, a module call a single box with one port per input and one per
output; that is the standard picture for the ECGP arm, in frames and in stage sheets
alike, because inlining is precisely what destroys the module boundaries an ECGP run is
being read for. The flattened twin is kept only for the final circuit, as
`seed<k>_final_flat.png`, since it is the graph that actually gets evaluated.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

import cgp

CLS_COLOUR = {"left": "#3498DB", "right": "#E67E22",
              "mixed": "#9B59B6", "const": "#95A5A6"}
INPUT_L = "#3498DB"
INPUT_R = "#E67E22"
EDGE = "#5D6D7E"
INPUT_SPLIT_GAP = 0.9      # extra y-gap between the left-4 and right-4 pixels


def _layout(pheno: cgp.Phenotype, n_in: int, split: int | None):
    """label -> (x, y). x = depth (inputs at 0), y = spread within a depth column."""
    pos: dict[int, tuple[float, float]] = {}

    for i in range(n_in):
        y = (n_in - 1) / 2.0 - i
        if split is not None:
            y += INPUT_SPLIT_GAP / 2 if i < split else -INPUT_SPLIT_GAP / 2
        pos[i] = (0.0, y)

    by_depth: dict[int, list[int]] = {}
    for j in pheno.active:
        by_depth.setdefault(pheno.depth[j], []).append(j)

    for d, nodes in by_depth.items():
        # order within a column by mean cone index, so left-half nodes sit near the
        # left-half inputs and the picture does not cross itself needlessly
        nodes.sort(key=lambda j: (sum(pheno.cone[j]) / len(pheno.cone[j])
                                  if pheno.cone[j] else n_in / 2))
        for r, j in enumerate(nodes):
            pos[n_in + j] = (float(d), (len(nodes) - 1) / 2.0 - r)
    return pos


# Module tags are drawn in their own colours, cycled by first appearance. Kept well
# clear of CLS_COLOUR so a module badge can never be mistaken for a cone class.
MODULE_COLOURS = ["#16A085", "#C0392B", "#8E44AD", "#D35400", "#2980B9", "#7F8C8D",
                  "#27AE60", "#B7950B"]


NEUTRAL = "#5D6D7E"        # box fill when cone colouring is off and the node is not modular

BOX_HW = 0.41              # gate box half-width, in data units. Fixed on purpose.


def _fit_labels(fits) -> None:
    """Shrink any gate label that is wider than its box.

    Boxes stay one fixed size -- equal-sized nodes are what let the eye read the graph
    as a graph -- so when a label does not fit it is the text that gives. Module tags
    grow with the run (`M12#1` early, `M2625#3` late), so a fixed font size either
    overflows late or wastes the box early.

    Must run AFTER `tight_layout`: that is what finally decides how many points one
    data unit is worth, and the whole measurement is in points.
    """
    for ax, labels, half_w in fits:
        fig = ax.figure
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:                       # non-Agg canvas: force one pass
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        x0 = ax.transData.transform((0.0, 0.0))[0]
        x1 = ax.transData.transform((2 * half_w, 0.0))[0]
        avail = abs(x1 - x0) * 0.90                  # leave the box its rounded padding
        for t in labels:
            w = t.get_window_extent(renderer).width
            if w > avail:
                t.set_fontsize(t.get_fontsize() * avail / w)


def _figsize(pheno: cgp.Phenotype, n_in: int) -> tuple[float, float]:
    max_d = max((pheno.depth[j] for j in pheno.active), default=0)
    w = min(22.0, max(6.0, 1.7 * (max_d + 2)))
    h = max(4.5, 0.62 * max(n_in, max((len(v) for v in [[j for j in pheno.active
            if pheno.depth[j] == d] for d in range(max_d + 1)]), default=1)) + 2.2)
    return w, h


def _render(ax, g, pheno: cgp.Phenotype, gates, n_in: int,
            title: str = "", split: int | None = None,
            origin: list[str] | None = None, colour_cones: bool = False,
            mod_colour: dict[str, str] | None = None, scale: float = 1.0) -> list:
    """Draw one circuit into an existing axes; return its gate-label text artists.

    The labels come back so the caller can hand them to `_fit_labels` once the layout
    is final.

    Split out from `draw` so a multi-panel figure can render each panel **directly**
    instead of saving a PNG per panel and pasting the images back in. That round trip
    resampled already-rasterised text and arrowheads, which is what made tiled frames
    look mushy; drawing into the axes keeps everything vector until the single final
    `savefig`.

    `mod_colour` is shared across panels by the caller so one module keeps one colour
    for the whole figure -- otherwise the same module is teal in one stage and red in
    the next, and the colour stops meaning anything across time.
    """
    pos = _layout(pheno, n_in, split)
    max_d = max((pheno.depth[j] for j in pheno.active), default=0)
    mod_colour = {} if mod_colour is None else mod_colour

    # ---- edges (drawn first so boxes sit on top) ----
    for j in pheno.active:
        gate = gates[g.func[j]]
        for k in range(gate.arity):
            src = g.conn[j * g.arity + k]        # conn is flat, row-major
            if src not in pos:
                continue                      # feeds an inactive node only
            ax.add_patch(FancyArrowPatch(
                pos[src], pos[n_in + j], arrowstyle="-|>",
                mutation_scale=11 * scale,
                shrinkA=20 * scale, shrinkB=20 * scale,
                color=EDGE, lw=1.1 * scale, alpha=0.75,
                connectionstyle="arc3,rad=0.06", zorder=1))

    # ---- program inputs ----
    for i in range(n_in):
        x, y = pos[i]
        col = INPUT_L if (split is not None and i < split) else (
            INPUT_R if split is not None else "#7F8C8D")
        ax.add_patch(FancyBboxPatch((x - 0.26, y - 0.17), 0.52, 0.34,
                                    boxstyle="round,pad=0.045", linewidth=1.3,
                                    facecolor="white", edgecolor=col, zorder=2))
        ax.text(x, y, f"p{i}", ha="center", va="center", fontsize=8.5 * scale,
                color=col, fontweight="bold", zorder=3)

    # ---- gate nodes ----
    labels: list = []
    out_set = set(pheno.out_nodes)
    for j in pheno.active:
        x, y = pos[n_in + j]
        is_out = j in out_set
        tag = origin[j] if origin and j < len(origin) else ""
        mc = None
        if tag:
            name = tag.split("#")[0]                # colour by MODULE, not by instance
            mc = mod_colour.setdefault(
                name, MODULE_COLOURS[len(mod_colour) % len(MODULE_COLOURS)])

        # One label per box. A node that came from a module is named by the module, not
        # by the primitive it happens to be: the primitive is an implementation detail
        # of a subroutine that was selected as a unit, and printing both made every box
        # carry a name that invited reading the module as equal to that gate.
        label = tag if tag else gates[g.func[j]].name.upper()
        col = CLS_COLOUR[pheno.cls[j]] if colour_cones else (mc or NEUTRAL)

        ax.add_patch(FancyBboxPatch((x - BOX_HW, y - 0.19), 2 * BOX_HW, 0.38,
                                    boxstyle="round,pad=0.05",
                                    linewidth=2.4 if is_out else 1.3,
                                    facecolor=col, alpha=0.90,
                                    edgecolor="#2C3E50" if is_out else col, zorder=2))
        labels.append(ax.text(x, y, label, ha="center", va="center",
                              fontsize=(7.6 if len(label) <= 5 else 6.5) * scale,
                              color="white", fontweight="bold", zorder=3))

    # ---- output marker ----
    for o, j in enumerate(pheno.out_nodes):
        src = pos[n_in + j] if j >= 0 else pos[g.ogene[o]]
        ax.add_patch(FancyArrowPatch((src[0] + 0.41, src[1]), (src[0] + 0.95, src[1]),
                                     arrowstyle="-|>", mutation_scale=13 * scale,
                                     color="#2C3E50", lw=1.6 * scale, zorder=3))
        ax.text(src[0] + 1.02, src[1], "out", ha="left", va="center",
                fontsize=9.5 * scale, fontweight="bold", color="#2C3E50", zorder=3)

    # The cone legend only means anything when cone colour is what is on screen.
    counts = pheno.counts()
    legend = [Patch(facecolor=CLS_COLOUR[k], label=f"{k} ({counts[k]})")
              for k in ("left", "right", "mixed", "const") if counts[k]] \
        if colour_cones else []
    if legend:
        ax.legend(handles=legend, loc="lower center", ncol=len(legend),
                  frameon=False, fontsize=8.5 * scale, bbox_to_anchor=(0.5, -0.035))

    ax.set_title(title, fontsize=11 * scale)
    ax.set_xlim(-0.9, max_d + 1.7)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 1.15, max(ys) + 0.8)
    ax.axis("off")
    return labels


def draw(g, pheno: cgp.Phenotype, gates, n_in: int, save_path,
         title: str = "", split: int | None = None,
         origin: list[str] | None = None,
         figsize: tuple[float, float] | None = None,
         colour_cones: bool = False) -> str:
    """Render one phenotype to `save_path`. Returns the path.

    `origin` (from `ecgp.flatten_with_origin`) tags each node with the module instance
    it was inlined from. Without it an ECGP circuit draws as an undifferentiated wall
    of gates -- flattening is precisely what discards module identity -- so a node that
    came from a module is **labelled with the module instance** (`M128287#2`) rather
    than with its primitive, and coloured per module. That is what makes reuse visible:
    the same colour twice in one picture is one acquired function used twice.

    `colour_cones` restores the left/right/mixed cone colouring documented at the top of
    this file. It is **off by default**: that colouring answers "is the solution split
    along the retina", which is the KA modularity question, not the ECGP question of
    which subroutines were acquired. With it off, colour is free to carry module
    identity, which is the thing these frames are being read for.
    """
    fig, ax = plt.subplots(figsize=figsize or _figsize(pheno, n_in))
    labels = _render(ax, g, pheno, gates, n_in, title, split, origin, colour_cones)
    fig.tight_layout()
    _fit_labels([(ax, labels, BOX_HW)])
    # `bbox_inches="tight"` crops back to the ink, which would undo a fixed canvas
    fig.savefig(save_path, dpi=135,
                **({} if figsize else {"bbox_inches": "tight"}))
    plt.close(fig)
    return str(save_path)


# ---------------------------------------------------------------------------
# the UNFLATTENED picture: a module is one box
# ---------------------------------------------------------------------------

MOD_BOX_HW = 0.46          # module / gate box half-width in the modular drawing
PORT_GAP = 0.30            # vertical distance between two ports on the same box
X_STEP = 1.7               # horizontal distance between depth columns
PIN_LEN = 0.14             # how far a port pin sticks out past the box edge


def _arity(ind, j: int) -> int:
    """How many of node `j`'s connection genes are actually read.

    A genotype row can be longer than the node needs -- a node that used to call a
    5-input module still carries those genes after it becomes a plain gate -- so the
    arity has to come from what the node currently *is*, never from `len(conn[j])`.
    """
    return 2 if ind.ntype[j] == 0 else ind.modules[ind.func[j]].n_in


def _n_out(ind, j: int) -> int:
    return 1 if ind.ntype[j] == 0 else ind.modules[ind.func[j]].n_out


def _active_nodes(ind, n_in: int) -> set[int]:
    """The nodes reachable from the outputs, walking the GENOTYPE graph.

    Not the same set as `cgp.Phenotype.active`, which is computed on the flattened
    circuit: a module output can be dead while the module box that produces it is alive.
    """
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


def n_module_calls(ind, n_in: int) -> int:
    """How many boxes in the drawing are module calls, repetitions counted.

    A module used three times counts three, because the question a stage sheet is read
    for is how much of the circuit is built from acquired subroutines -- not how many
    distinct ones the genome happens to hold.
    """
    return sum(1 for j in _active_nodes(ind, n_in) if ind.ntype[j] != 0)


def _modular_geometry(ind, n_in: int, split: int | None):
    """Where every box, pin and port of the unflattened drawing goes.

    Split out from the rendering so the figure can be sized from the layout before any
    axes exist, and so one panel of a grid and a standalone frame lay out identically.
    """
    active = _active_nodes(ind, n_in)

    depth: dict[int, int] = {}
    for j in sorted(active):           # ascending: a node only reads lower labels
        srcs = [s for s in ind.conn[j][:_arity(ind, j)] if s >= n_in]
        depth[j] = 1 + max((depth[s - n_in] for s in srcs), default=-1)
    max_d = max(depth.values(), default=0)

    # ---- geometry: a box is as tall as its widest side has ports ----
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
            # +1: depth 0 is "reads only program inputs", which is one column to the
            # RIGHT of the input pins, not on top of them
            pos[j] = ((d + 1) * X_STEP, y)
            y -= height[j] / 2 + 0.34

    def port(j: int, k: int, n: int, side: int,
             pin: bool = False) -> tuple[float, float]:
        """Position of port `k` of `n` on a box: side -1 = inputs (left), +1 = outputs.

        `pin=True` gives the *tip* of the pin rather than the point where it meets the
        box, which is where a wire should land: the pins stick out of the package and
        the wires attach to their ends, the way a chip is drawn.
        """
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

    Same contract as `_render`: geometry in, labels out, `_fit_labels` applied by the
    caller once `tight_layout` has settled how big a data unit is.
    """
    import ecgp                        # local: keeps `visualize` usable in CGP-only runs

    active, max_d, height, pos, in_pos, _, port = geom
    mod_colour = {} if mod_colour is None else mod_colour

    # ---- edges: port to port, so a 5-input module reads as five separate wires ----
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

    # ---- input pins ----
    for i in range(n_in):
        x, y = in_pos[i]
        col = INPUT_L if (split is not None and i < split) else (
            INPUT_R if split is not None else "#7F8C8D")
        ax.add_patch(FancyBboxPatch((x - 0.26, y - 0.17), 0.52, 0.34,
                                    boxstyle="round,pad=0.045", linewidth=1.3,
                                    facecolor="white", edgecolor=col, zorder=2))
        ax.text(x, y, f"p{i}", ha="center", va="center", fontsize=8.5 * scale,
                color=col, fontweight="bold", zorder=3)

    # ---- boxes ----
    labels: list = []
    out_nodes = {l - n_in for l in ind.ogene if l >= n_in}
    for j in sorted(active):
        x, y = pos[j]
        h = height[j]
        if ind.ntype[j] == 0:
            name, col = gates[ind.func[j]].name.upper(), NEUTRAL
        else:
            name = ecgp.module_name(ind.func[j], n_prim)
            col = mod_colour.setdefault(
                name, MODULE_COLOURS[len(mod_colour) % len(MODULE_COLOURS)])
        is_out = j in out_nodes
        ax.add_patch(FancyBboxPatch((x - MOD_BOX_HW, y - h / 2), 2 * MOD_BOX_HW, h,
                                    boxstyle="round,pad=0.04",
                                    linewidth=2.4 if is_out else 1.3,
                                    facecolor=col, alpha=0.90,
                                    edgecolor="#2C3E50" if is_out else col, zorder=2))
        labels.append(ax.text(x, y, name, ha="center", va="center",
                              fontsize=8.0 * scale,
                              color="white", fontweight="bold", zorder=3))
        # Port ticks. Without them a tall box with one wire arriving looks like a wide
        # box with a wire arriving anywhere, and the arity -- the thing that makes a
        # module a subroutine rather than a gate -- is invisible.
        for k in range(_arity(ind, j)):
            px, py = port(j, k, _arity(ind, j), -1)
            ax.plot([px - PIN_LEN, px], [py, py], color="#2C3E50",
                    lw=3.0 * scale, solid_capstyle="butt", zorder=4)
        for k in range(_n_out(ind, j)):
            px, py = port(j, k, _n_out(ind, j), +1)
            ax.plot([px, px + PIN_LEN], [py, py], color="#2C3E50",
                    lw=3.0 * scale, solid_capstyle="butt", zorder=4)

    # ---- program outputs ----
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
    """Draw the genotype **without inlining**: one box per genome node. Returns the path.

    This is the standard picture of an ECGP circuit. `draw` renders the *flattened*
    circuit, where a module has already been copied out into ordinary gates -- so a
    module with two outputs becomes two boxes carrying the same tag, and the module
    boundary survives only as a label. Here the boundary is the box: a module call is a
    single rectangle with one input port per parameter and one output port per module
    output, and the arrows land on the ports. That is the picture that shows what was
    acquired and where it is re-used; the flattened one shows what actually gets
    evaluated, and is kept alongside it only for the final circuit.

    The two are different views of the same individual, not different circuits. Node
    counts will not agree -- that is the point of the pair.
    """
    geom = _modular_geometry(ind, n_in, split)
    fig, ax = plt.subplots(figsize=_modular_figsize(geom, n_in))
    labels = _render_modular(ax, ind, n_in, gates, n_prim, geom, title, split)
    fig.tight_layout()
    _fit_labels([(ax, labels, MOD_BOX_HW)])
    fig.savefig(save_path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def frame_title(gen: int, goal: str, hits: int, n_patterns: int,
                pheno: cgp.Phenotype, seed: int | None = None,
                alg: str = "CGP", mods: str = "") -> str:
    """The one-line caption every saved frame carries.

    `mods` is the module histogram of the ECGP individual behind this circuit (empty
    for CGP). It cannot be recovered from the drawing: the picture is the *flattened*
    graph, where a module has already been inlined into ordinary gates.
    """
    c = pheno.counts()
    head = f"{alg}  gen {gen}" + (f"  seed {seed}" if seed is not None else "")
    tail = (f"{pheno.n_active} active nodes  |  "
            f"left {c['left']}  right {c['right']}  mixed {c['mixed']}")
    return (f"{head}  |  goal {goal.upper()}  |  "
            f"accuracy {hits / n_patterns:.4f}  ({hits}/{n_patterns})\n"
            + tail + (f"  |  {mods}" if mods else ""))


STAGE_ROWS, STAGE_COLS = 2, 3


def stage_title(gen: int, hits: int, n_patterns: int,
                mod_calls: int | None = None) -> str:
    """The caption under one stage panel: when, how well, and how modular. Nothing else.

    Deliberately much shorter than `frame_title`. On a sheet, everything constant
    across panels (algorithm, seed, goal) belongs in the figure title, and the node
    census is already visible in the drawing -- repeating it six times only competes
    with the circuits for attention.

    `mod_calls` is the number of module BOXES in this panel, repetitions counted, so the
    panels can be compared as a series without counting coloured boxes by eye. `None`
    for CGP, which has no modules to count.
    """
    head = (f"gen {gen}  |  accuracy {100 * hits / n_patterns:.2f}% "
            f"({hits}/{n_patterns})")
    return head if mod_calls is None else f"{head}  |  {mod_calls} module calls"


def stage_grid(panels, gates, n_in: int, save_path, title: str = "",
               split: int | None = None, colour_cones: bool = False,
               n_prim: int | None = None,
               rows: int = STAGE_ROWS, cols: int = STAGE_COLS) -> str:
    """Tile snapshots of ONE seed's history into a `rows` x `cols` sheet.

    Each panel is `(individual, phenotype, title, origin)` and is drawn **into its own
    axes**. The earlier version saved a PNG per panel and pasted the images back with
    `imshow`, which resampled text and arrowheads that had already been rasterised --
    no dpi setting fixes that, because the damage is done before the paste. Rendering
    into the axes keeps every glyph and arrow vector until one final `savefig`, so the
    sheet is as sharp as a single-circuit frame.

    `n_prim` set means the panels are ECGP individuals and get the **unflattened**
    drawing, one box per module -- the same picture as `draw_modular`, so a stage sheet
    and a final frame are read the same way. Left `None` (the CGP arm) the panels are
    plain genotypes and get the flattened one; for CGP the two coincide anyway, since
    there are no modules to inline.

    Module colours are assigned once and shared across panels, so a module that
    survives several stages keeps its colour and the eye can follow it through time.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9.0, rows * 6.0))
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    shared: dict[str, str] = {}
    fits = []
    for ax, panel in zip(flat, list(panels) + [None] * (rows * cols)):
        ax.axis("off")
        if panel is None:
            continue
        g, pheno, ptitle, origin = panel
        if n_prim is not None:
            geom = _modular_geometry(g, n_in, split)
            fits.append((ax, _render_modular(ax, g, n_in, gates, n_prim, geom, ptitle,
                                             split, mod_colour=shared, scale=1.55),
                         MOD_BOX_HW))
            continue
        fits.append((ax, _render(ax, g, pheno, gates, n_in, ptitle, split, origin,
                                 colour_cones, mod_colour=shared, scale=1.55), BOX_HW))
    if title:
        fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.975) if title else None)
    _fit_labels(fits)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
