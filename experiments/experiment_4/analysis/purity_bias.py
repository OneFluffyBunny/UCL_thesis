"""Is purity-by-active-size confounded with GENOME size?

Backs the "<=0.02 against a per-circuit SD of ~0.10" claim in `add_to_latex.md` and
in `qmetrics.metrics.circuit_purity`.

Only active nodes are ever scored -- inactive nodes are not in the graph. But to
cover active counts 2..50 the sampler has to sweep genome sizes, and a circuit
with 30 active gates carved out of a 512-node genome is not obviously the same
object as one that is nearly all of an 80-node genome (long-range vs local
wiring, hence different depth, hence possibly different purity).

So: hold the active count FIXED and vary the genome size. If purity is flat along
that row, the mixture is harmless and one curve is honest. If it slopes, the curve
is an artefact of which genome sizes happened to fill which bucket.

Runs from any working directory. ~1 minute.
"""
from __future__ import annotations

import pathlib
import random
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # analysis/: purity_baseline
sys.path.insert(0, str(_HERE.parents[1]))      # experiment_4/: cgp, gates
sys.path.insert(0, str(_HERE.parents[3]))      # repo root: qmetrics

import cgp
import gates as gates_mod
from qmetrics import circuit_purity
from purity_baseline import N_IN, SPLIT, to_digraph

GENOMES = (16, 32, 64, 128, 256, 512)
ROWS = (5, 10, 15, 20, 25, 30, 40, 50)
DRAWS = 14000


def main() -> int:
    gs = gates_mod.build_set("nand")
    pinned = {i: (0 if i < SPLIT else 1) for i in range(N_IN)}
    cell: dict[tuple[int, int], list[float]] = {}
    depth: dict[tuple[int, int], list[int]] = {}
    reach: dict[int, list[int]] = {}

    seed = 0
    for ng in GENOMES:
        for _ in range(DRAWS):
            seed += 1
            g = cgp.random_genotype(random.Random(seed), ng, N_IN, 1, 1, 2)
            G, ph = to_digraph(g, gs)
            n = ph.n_active
            reach.setdefault(ng, []).append(n)
            if n < 2 or n > 50:
                continue
            p, _ = circuit_purity(G, pinned, exclude=[lbl for lbl in g.ogene])
            if np.isnan(p):
                continue
            cell.setdefault((n, ng), []).append(p)
            d = {i: 0 for i in range(N_IN)}
            for j in ph.active:
                d[N_IN + j] = 1 + max(d[g.conn[j * g.arity + k]]
                                      for k in range(gs[g.func[j]].arity))
            depth.setdefault((n, ng), []).append(max(d.values()))

    print("active-count reach of each genome size (mean +/- sd, max):")
    for ng in GENOMES:
        a = np.array(reach[ng])
        print(f"  n_nodes {ng:4d}: active {a.mean():5.1f} +/- {a.std():4.1f} "
              f"(max {a.max()})")

    print(f"\nPURITY at fixed active count, by genome size "
          f"(cells with <40 samples blank)")
    print("active  " + "".join(f"{ng:>14d}" for ng in GENOMES))
    for n in ROWS:
        row = f"{n:>6}  "
        for ng in GENOMES:
            v = cell.get((n, ng), [])
            row += (f"{np.mean(v):>8.3f}({len(v):>4})" if len(v) >= 40
                    else " " * 14)
        print(row)

    print(f"\nMAX DEPTH at fixed active count, by genome size")
    print("active  " + "".join(f"{ng:>14d}" for ng in GENOMES))
    for n in ROWS:
        row = f"{n:>6}  "
        for ng in GENOMES:
            v = depth.get((n, ng), [])
            row += (f"{np.mean(v):>8.2f}({len(v):>4})" if len(v) >= 40
                    else " " * 14)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
