"""Circuit purity of UNEVOLVED random circuits, as a function of active-gate count.

REGENERATES `latex_figures/purity_metric/random_circuit_purity_by_size.png` and the
by-size table quoted in `add_to_latex.md` and in `qmetrics.metrics.circuit_purity`.

The baseline any evolved purity has to be read against. Bucket by the EXACT number
of active gates, not by a range -- the point is the shape of the drift, and ranges
hide it. Only ACTIVE nodes exist in the graph at all: a 10-node genome with 5 gates
on a path to the output is filed under 5, and the 5 drifted-off gates are neither
scored nor counted in the denominator.

STRATIFIED sampling. Covering active counts 2..50 needs genome sizes from 8 to 512
(a 16-node genome averages 3.8 active gates, a 512-node one averages 39.3), so a
first-come sampler fills the small buckets with small genomes and the large buckets
with large ones -- genome size would be confounded with active count. Capping each
(active, genome) cell at the same CELL makes every bucket an equal blend of the
genome sizes that reach it. `purity_bias.py` measures the residual: purity at fixed
active count varies by <=0.02 across genome sizes against a per-circuit SD of ~0.10,
and mean depth is flat to two decimals, so the blend is harmless.

Runs from any working directory. ~1 minute.
"""
from __future__ import annotations

import pathlib
import random
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # experiment_4/: cgp, gates
sys.path.insert(0, str(_HERE.parents[3]))      # repo root: qmetrics

import cgp
import gates as gates_mod
from qmetrics import circuit_purity, open_file

import networkx as nx

N_IN, SPLIT = 8, 4
MAX_SIZE = 50
GENOMES = (8, 16, 32, 64, 128, 256, 512)
DRAWS = 16000          # per genome size
CELL = 220             # cap per (active count, genome size) cell
MIN_TOTAL = 150        # a bucket needs this many circuits to be plotted
OUT = (_HERE.parents[3] / "latex_figures" / "purity_metric"
       / "random_circuit_purity_by_size.png")


def to_digraph(g: cgp.Genotype, gate_set):
    """Program inputs + ACTIVE gates only, edges parent -> child."""
    ph = cgp.phenotype(g, N_IN, gate_set, split=SPLIT)
    act = set(ph.active)
    G = nx.DiGraph()
    G.add_nodes_from(range(N_IN))
    G.add_nodes_from(N_IN + j for j in act)
    for j in ph.active:
        for k in range(gate_set[g.func[j]].arity):
            s = g.conn[j * g.arity + k]
            if s < N_IN or (s - N_IN) in act:
                G.add_edge(s, N_IN + j)
    return G, ph


def sample() -> dict[int, list[float]]:
    gs = gates_mod.build_set("nand")
    pinned = {i: (0 if i < SPLIT else 1) for i in range(N_IN)}
    cell: dict[tuple[int, int], list[float]] = {}
    seed = 0
    for ng in GENOMES:
        for _ in range(DRAWS):
            seed += 1
            g = cgp.random_genotype(random.Random(seed), ng, N_IN, 1, 1, 2)
            G, ph = to_digraph(g, gs)
            n = ph.n_active
            if not 2 <= n <= MAX_SIZE or len(cell.get((n, ng), ())) >= CELL:
                continue
            p, _ = circuit_purity(G, pinned, exclude=[lbl for lbl in g.ogene])
            if not np.isnan(p):
                cell.setdefault((n, ng), []).append(p)
    out: dict[int, list[float]] = {}
    for (n, _), v in cell.items():
        out.setdefault(n, []).extend(v)
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_size = sample()
    sizes = sorted(k for k in by_size if len(by_size[k]) >= MIN_TOTAL)
    mean = np.array([np.mean(by_size[k]) for k in sizes])
    sd = np.array([np.std(by_size[k], ddof=1) for k in sizes])
    cnt = [len(by_size[k]) for k in sizes]

    print(f"{'active':>7}{'gates scored':>14}{'n':>7}{'mean':>9}{'sd':>8}")
    for k, m, s, c in zip(sizes, mean, sd, cnt):
        print(f"{k:>7}{k - 1:>14}{c:>7}{m:>9.4f}{s:>8.4f}")
    miss = [k for k in range(2, MAX_SIZE + 1) if k not in sizes]
    if miss:
        print(f"under {MIN_TOTAL} samples, omitted: {miss}")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.errorbar(sizes, mean, yerr=sd, fmt="o-", ms=4.5, lw=1.7, capsize=3,
                color="#2563eb", ecolor="#93c5fd", elinewidth=1.4,
                label="mean ± 1 SD over random circuits")
    ax.axhline(0.0, color="#9ca3af", lw=1.0, ls=":")
    ax.annotate("0 = every gate perfectly mixed", (MAX_SIZE, 0.0), (-6, 7),
                textcoords="offset points", ha="right", fontsize=9,
                color="#6b7280")
    ax.annotate("1 = every gate draws from one side only",
                (MAX_SIZE, 1.0), (-6, 6), textcoords="offset points",
                ha="right", fontsize=9, color="#6b7280")

    ax.set_xlabel("active gates in the circuit  (only nodes on a path to the "
                  "output; the output itself is not scored)")
    ax.set_ylabel("circuit purity")
    ax.set_title("Circuit purity of unevolved random circuits, by size\n"
                 "retina task, 8 inputs pinned left/right · stratified over "
                 f"genome sizes {GENOMES[0]}–{GENOMES[-1]} · {sum(cnt):,} circuits",
                 fontsize=12, pad=12)
    ax.set_xlim(0, MAX_SIZE + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
    open_file(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
