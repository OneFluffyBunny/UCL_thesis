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


def draw(g, pheno: cgp.Phenotype, gates, n_in: int, save_path,
         title: str = "", split: int | None = None) -> str:
    """Render the phenotype to `save_path`. Returns the path."""
    pos = _layout(pheno, n_in, split)
    max_d = max((pheno.depth[j] for j in pheno.active), default=0)

    w = min(22.0, max(6.0, 1.7 * (max_d + 2)))
    h = max(4.5, 0.62 * max(n_in, max((len(v) for v in [[j for j in pheno.active
            if pheno.depth[j] == d] for d in range(max_d + 1)]), default=1)) + 2.2)
    fig, ax = plt.subplots(figsize=(w, h))

    # ---- edges (drawn first so boxes sit on top) ----
    for j in pheno.active:
        gate = gates[int(g.func[j])]
        for k in range(gate.arity):
            src = int(g.conn[j, k])
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
    for j in pheno.active:
        x, y = pos[n_in + j]
        col = CLS_COLOUR[pheno.cls[j]]
        is_out = j in out_set
        ax.add_patch(FancyBboxPatch((x - 0.34, y - 0.19), 0.68, 0.38,
                                    boxstyle="round,pad=0.05",
                                    linewidth=2.4 if is_out else 1.3,
                                    facecolor=col, alpha=0.90,
                                    edgecolor="#2C3E50" if is_out else col, zorder=2))
        ax.text(x, y, gates[int(g.func[j])].name.upper(), ha="center", va="center",
                fontsize=7.6, color="white", fontweight="bold", zorder=3)

    # ---- output marker ----
    for o, j in enumerate(pheno.out_nodes):
        src = pos[n_in + j] if j >= 0 else pos[int(g.ogene[o])]
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
    fig.savefig(save_path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def frame_title(gen: int, goal: str, hits: int, n_patterns: int,
                pheno: cgp.Phenotype, seed: int | None = None) -> str:
    """The one-line caption every saved frame carries."""
    c = pheno.counts()
    head = f"CGP  gen {gen}" + (f"  seed {seed}" if seed is not None else "")
    return (f"{head}  |  goal {goal.upper()}  |  "
            f"accuracy {hits / n_patterns:.4f}  ({hits}/{n_patterns})\n"
            f"{pheno.n_active} active nodes  |  "
            f"left {c['left']}  right {c['right']}  mixed {c['mixed']}")
