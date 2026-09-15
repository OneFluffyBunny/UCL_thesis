"""Draw a `train.py` run's history. CPython only (matplotlib).

    conda run -n lndp python render.py runs/<tag>            # every seed in the run
    conda run -n lndp python render.py runs/<tag>/seed0      # one seed

Per seed, into that seed's directory:

  gate_shares.png     how the gate mix changes over evolution. Top: share of the
                      circuit's top-level gates (what a stage panel shows) that are
                      NAND vs each module. Middle: share of the flattened primitive
                      gates that sit inside each module. Bottom: accuracy.
  stages.png          six snapshots of the circuit, first to last, fake modules grey
  final_decomposition.png
                      the last snapshot, plus each real module's own body

A module has the SAME colour in all three figures. The seven most-used modules (by
peak top-level share) get a colour each; the rest are pooled as "other modules".
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import census
import decompose
import gates as gates_mod
import tasks as tasks_mod
import visualize as viz

# visualize's cycle minus its grey, which would read as "primitive" or "other"
COLOURS = [c for c in viz.MODULE_COLOURS if c != "#7F8C8D"]
N_COLOURED = len(COLOURS)
NAND_COLOUR = "#D5D8DC"        # primitives: recessive light grey
OTHER_COLOUR = "#A6ACAF"       # pooled low-use modules: mid grey, same in every figure
ACC_COLOUR = "#2C3E50"


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _load(seed_dir: pathlib.Path):
    cfg = json.loads((seed_dir / "config.json").read_text())
    log = _read_csv(seed_dir / "log.csv")
    gates_rows = _read_csv(seed_dir / "gates.csv")
    with open(seed_dir / "snapshots.jsonl") as fh:
        snaps = [json.loads(line) for line in fh]
    return cfg, log, gates_rows, snaps


def _colour_map(gates_rows: list[dict]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """(module -> colour, module -> legend label, coloured modules in draw order)."""
    peak: dict[str, float] = defaultdict(float)
    label: dict[str, str] = {}
    born: dict[str, int] = {}
    for r in gates_rows:
        if r["kind"] != "module" or r["fake"] == "1":
            continue
        name = r["gate"]
        peak[name] = max(peak[name], float(r["share_top"]), float(r["share_flat"]))
        label[name] = r["label"] or r["signature"]
        born[name] = int(r["born_gen"]) if r["born_gen"] else 0
    top = sorted((m for m in peak if peak[m] > 0), key=lambda m: -peak[m])[:N_COLOURED]
    top.sort(key=lambda m: (born[m], int(m[1:])))          # fixed order: by birth
    colours = {m: COLOURS[i] for i, m in enumerate(top)}
    legend = {m: f"{m} = {label[m]}" if label[m] else m for m in top}
    return colours, legend, top


def plot_gate_shares(seed_dir: pathlib.Path, cfg, log, gates_rows, colours, legend, order):
    gens = [int(r["gen"]) for r in log]
    idx = {g: i for i, g in enumerate(gens)}
    n = len(gens)
    series = {key: {"top": [0.0] * n, "flat": [0.0] * n}
              for key in ["NAND", *order, "other modules"]}
    for r in gates_rows:
        i = idx[int(r["gen"])]
        if r["kind"] == "primitive":
            key = "NAND" if r["gate"] == "NAND" else r["gate"]
            series.setdefault(key, {"top": [0.0] * n, "flat": [0.0] * n})
        else:
            key = r["gate"] if r["gate"] in colours else "other modules"
        series[key]["top"][i] += float(r["share_top"])
        series[key]["flat"][i] += float(r["share_flat"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                             gridspec_kw=dict(height_ratios=(3, 3, 1.4)))
    keys = [k for k in series if any(series[k]["top"]) or any(series[k]["flat"])]
    for ax, part in zip(axes[:2], ("top", "flat")):
        base = [0.0] * n
        for key in keys:
            ys = series[key][part]
            top = [b + y for b, y in zip(base, ys)]
            col = colours.get(key, NAND_COLOUR if key == "NAND" else OTHER_COLOUR)
            ax.fill_between(gens, base, top, step="post", facecolor=col,
                            edgecolor="white", linewidth=0.6,
                            label=legend.get(key, key))
            base = top
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#E5E7E9", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("share of top-level gates")
    axes[1].set_ylabel("share of flattened gates")

    acc = [float(r["acc"]) for r in log]
    axes[2].step(gens, acc, where="post", color=ACC_COLOUR, linewidth=2)
    axes[2].set_ylabel("accuracy")
    axes[2].set_ylim(min(0.5, min(acc)), 1.0)
    axes[2].set_xlabel("generation")
    axes[2].grid(axis="y", color="#E5E7E9", linewidth=0.6)
    for s in ("top", "right"):
        axes[2].spines[s].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    solved = next((int(r["gen"]) for r in log if "solved" in r["reason"]), None)
    fig.suptitle(f"gate mix over evolution, seed {cfg['seed']}\n"
                 f"{cfg['task']}/{cfg['operation']}, {cfg['gates']} only, "
                 f"{'solved at gen ' + str(solved) if solved else 'not solved'}",
                 fontsize=12)
    fig.tight_layout()
    out = seed_dir / "gate_shares.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _pick_stages(snaps: list[dict], k: int = 6) -> list[dict]:
    """First, last, and the snapshots nearest to evenly spaced generations between."""
    last = snaps[-1]["gen"]
    picks, used = [], set()
    for t in range(k):
        want = last * t / (k - 1)
        best = min((s for s in snaps if s["gen"] not in used),
                   key=lambda s: abs(s["gen"] - want), default=None)
        if best is not None:
            used.add(best["gen"])
            picks.append(best)
    return sorted(picks, key=lambda s: s["gen"])


def render_seed(seed_dir: pathlib.Path) -> list[pathlib.Path]:
    cfg, log, gates_rows, snaps = _load(seed_dir)
    gate_set = gates_mod.build_set(cfg["gates"])
    n_prim = len(gate_set)
    n_in = tasks_mod.n_inputs(cfg["task"])
    n_pat = tasks_mod.n_patterns(cfg["task"])
    split = tasks_mod.split_index(cfg["task"])
    colours, legend, order = _colour_map(gates_rows)
    # drawings: every module outside the coloured set is the pooled grey, never a
    # recycled colour that would make two different modules look like one
    draw_colours = {r["gate"]: colours.get(r["gate"], OTHER_COLOUR)
                    for r in gates_rows if r["kind"] == "module"}

    outs = [plot_gate_shares(seed_dir, cfg, log, gates_rows, colours, legend, order)]

    panels = []
    for s in _pick_stages(snaps):
        ind = census.from_json(s["ind"])
        panels.append((ind, None, viz.stage_title(s["gen"], s["hits"], n_pat,
                                                  viz.n_hidden_nodes(ind, n_in)), None))
    outs.append(pathlib.Path(viz.stage_grid(
        panels, gate_set, n_in, seed_dir / "stages.png", split=split, n_prim=n_prim,
        title=f"circuit stages, seed {cfg['seed']} ({cfg['task']}/{cfg['operation']})",
        mod_colour=dict(draw_colours))))

    final = snaps[-1]
    ind = census.from_json(final["ind"])
    outs.append(pathlib.Path(decompose.draw_decomposition(
        ind, n_in, gate_set, n_prim, seed_dir / "final_decomposition.png", split=split,
        circuit_title=viz.stage_title(final["gen"], final["hits"], n_pat,
                                      viz.n_hidden_nodes(ind, n_in)),
        caption_prefix=f"seed {cfg['seed']}, gen {final['gen']}: ",
        mod_colour=dict(draw_colours))))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    args = ap.parse_args()
    root = pathlib.Path(args.run_dir)
    seed_dirs = [root] if (root / "log.csv").exists() else \
        sorted(p for p in root.glob("seed*") if (p / "log.csv").exists())
    if not seed_dirs:
        raise SystemExit(f"no seed directories with log.csv under {root}")
    for d in seed_dirs:
        for out in render_seed(d):
            print(out)


if __name__ == "__main__":
    main()
