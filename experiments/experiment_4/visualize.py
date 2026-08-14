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


def draw(g, pheno: cgp.Phenotype, gates, n_in: int, save_path,
         title: str = "", split: int | None = None,
         origin: list[str] | None = None,
         figsize: tuple[float, float] | None = None) -> str:
    """Render the phenotype to `save_path`. Returns the path.

    `origin` (from `ecgp.flatten_with_origin`) tags each node with the module instance
    it was inlined from. Without it an ECGP circuit draws as an undifferentiated wall
    of gates -- flattening is precisely what discards module identity -- so the tag is
    printed under each gate that came from a module, coloured per module, and repeated
    instances of one module share a colour. That is what makes reuse visible: the same
    colour appearing twice in one picture is one acquired function used twice.
    """
    pos = _layout(pheno, n_in, split)
    max_d = max((pheno.depth[j] for j in pheno.active), default=0)

    w = min(22.0, max(6.0, 1.7 * (max_d + 2)))
    h = max(4.5, 0.62 * max(n_in, max((len(v) for v in [[j for j in pheno.active
            if pheno.depth[j] == d] for d in range(max_d + 1)]), default=1)) + 2.2)
    # A caller tiling several of these needs them to share a canvas: sized per circuit,
    # a 4-node stage and a 35-node one come out with different aspect ratios and the
    # grid renders them at wildly different scales.
    fig, ax = plt.subplots(figsize=figsize or (w, h))

    # ---- edges (drawn first so boxes sit on top) ----
    for j in pheno.active:
        gate = gates[g.func[j]]
        for k in range(gate.arity):
            src = g.conn[j * g.arity + k]        # conn is flat, row-major
            if src not in pos:
                continue                      # feeds an inactive node only
            ax.add_patch(FancyArrowPatch(
                pos[src], pos[n_in + j], arrowstyle="-|>", mutation_scale=11,
                shrinkA=17, shrinkB=17, color=EDGE, lw=1.1, alpha=0.75,
                connectionstyle="arc3,rad=0.06", zorder=1))

    # ---- program inputs ----
    for i in range(n_in):
        x, y = pos[i]
        col = INPUT_L if (split is not None and i < split) else (
            INPUT_R if split is not None else "#7F8C8D")
        ax.add_patch(FancyBboxPatch((x - 0.26, y - 0.17), 0.52, 0.34,
                                    boxstyle="round,pad=0.045", linewidth=1.3,
                                    facecolor="white", edgecolor=col, zorder=2))
        ax.text(x, y, f"p{i}", ha="center", va="center", fontsize=8.5,
                color=col, fontweight="bold", zorder=3)

    # ---- gate nodes ----
    out_set = set(pheno.out_nodes)
    mod_colour: dict[str, str] = {}
    for j in pheno.active:
        x, y = pos[n_in + j]
        col = CLS_COLOUR[pheno.cls[j]]
        is_out = j in out_set
        ax.add_patch(FancyBboxPatch((x - 0.34, y - 0.19), 0.68, 0.38,
                                    boxstyle="round,pad=0.05",
                                    linewidth=2.4 if is_out else 1.3,
                                    facecolor=col, alpha=0.90,
                                    edgecolor="#2C3E50" if is_out else col, zorder=2))
        ax.text(x, y, gates[g.func[j]].name.upper(), ha="center", va="center",
                fontsize=7.6, color="white", fontweight="bold", zorder=3)
        tag = origin[j] if origin and j < len(origin) else ""
        if tag:
            name = tag.split("#")[0]                # colour by MODULE, not by instance
            mc = mod_colour.setdefault(
                name, MODULE_COLOURS[len(mod_colour) % len(MODULE_COLOURS)])
            ax.text(x, y - 0.30, tag, ha="center", va="center", fontsize=6.0,
                    color=mc, fontweight="bold", zorder=3)

    # ---- output marker ----
    for o, j in enumerate(pheno.out_nodes):
        src = pos[n_in + j] if j >= 0 else pos[g.ogene[o]]
        ax.add_patch(FancyArrowPatch((src[0] + 0.34, src[1]), (src[0] + 0.95, src[1]),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#2C3E50", lw=1.6, zorder=3))
        ax.text(src[0] + 1.02, src[1], "out", ha="left", va="center",
                fontsize=9.5, fontweight="bold", color="#2C3E50", zorder=3)

    counts = pheno.counts()
    legend = [Patch(facecolor=CLS_COLOUR[k], label=f"{k} ({counts[k]})")
              for k in ("left", "right", "mixed", "const") if counts[k]]
    if legend:
        ax.legend(handles=legend, loc="lower center", ncol=len(legend),
                  frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.035))

    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.9, max_d + 1.7)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 1.15, max(ys) + 0.8)
    ax.axis("off")
    fig.tight_layout()
    # `bbox_inches="tight"` crops back to the ink, which would undo a fixed canvas
    fig.savefig(save_path, dpi=135,
                **({} if figsize else {"bbox_inches": "tight"}))
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


def grid(paths, save_path, title: str = "", rows: int = 3, cols: int = 3) -> str:
    """Compose already-rendered circuit PNGs into a `rows` x `cols` contact sheet.

    Re-reads the saved per-seed frames instead of re-drawing them, so every panel is
    byte-identical to the diagram that seed already produced -- same layout, same
    caption carrying the accuracy -- and a panel can never disagree with the file it
    came from. Missing paths leave a blank cell, which is what a run of fewer than
    `rows*cols` seeds should look like.
    """
    import os
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.4, rows * 3.8))
    flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, p in zip(flat, list(paths) + [None] * (rows * cols)):
        ax.axis("off")
        if p is not None and os.path.exists(str(p)):
            ax.imshow(mpimg.imread(str(p)))
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
