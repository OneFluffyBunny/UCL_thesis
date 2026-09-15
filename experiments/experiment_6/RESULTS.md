# Experiment 6 — results

Three mechanisms (see `README.md`): Self-Modifying CGP, nested ECGP (`necgp/`) and
nested ECGP with a pairwise, interaction-checked `compress` (`necgp_pairwise/`).
Task for the nested variants: `retina_ka2005`/xor, NAND only, 100 nodes, (1+4) ES.
All three searches are seeded and deterministic.

## 1. SMCGP

`smcgp.py` implements Harding, Miller & Banzhaf (CEC 2009) as specified in
`PAPER_SPEC.md`: 13 self-modification operators, relative addressing, the output-flag
rule, (1+4) ES with a bootstrap of 50. `test_smcgp.py` checks out-of-range addressing,
output selection and its fallbacks, each operator on a hand-built graph, and 25 random
genotypes through development and evaluation.

Two choices were needed to make it search at all, both documented in the code:
- the connection-gene range is tied to `--nodes` (a fixed range of 200 on a 30-node
  genotype made almost every output a constant);
- output flags start at 0, so the paper's fallback (last node is the output) applies
  initially. With fair-coin flags the leftmost flagged node was almost always too
  early to address anything.

Verification (`train.py --nodes 30 --max-inputs 2 --max-evals 500000 --bootstrap 50`,
seeds 0–5): 3 of 6 seeds improve from 2/4 to 3/4 on 2-input parity, none solves. This
is slower than the paper's 126,095 evaluations to solve; `PAPER_SPEC.md` ("Known gap")
names the likely cause, so these evaluation counts are not a reproduction of the
paper. SMCGP was not applied to the retina task.

## 2. necgp — ECGP with nesting

`compress` may wrap a window that contains module calls, accepted with probability
`nest_decay ** (depth − 1)` (`--nest-decay`, 0.5). `expand` undoes one nesting level.
`test_necgp.py` checks structural invariants under heavy mutation, that nested
evaluation equals flattened evaluation, that decay 0 blocks nesting, and the
interface-operator guard for modules used inside other modules.

**Nesting happens.** Seed 0 solves with modules of depth 2–4 alive (17 modules, 11
nested), depth 5 was reached transiently, and one module calls the same sub-module
twice in its own body.

**Nested vs flat on the same seeds** (`necgp/seed_sweep.py`, nest_decay 0.5 vs 0.0,
300,000-generation cap; data `necgp/seed_sweep.csv`):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| nested, generations to solve | 78,134 | 63,938 | 29,551 | 15,662 | 99,051 | 55,186 | 91,133 | 104,420 | 36,121 | 101,400 |
| flat | 115,095 | 55,218 | 113,118 | 14,933 | 113,179 | not solved | 100,681 | 113,746 | 99,704 | 53,594 |

Nested is faster on 6 of the 9 seeds both arms solved (median 78,134 vs 100,681);
paired Wilcoxon p = 0.16, not significant. Seed 5 (flat never solves) would only
strengthen the nested side. Nesting costs wall-clock time: evaluation recurses through
module bodies.

**Many modules are fake.** `necgp/decompose_seed0.py` replays seed 0 and walks the
solved circuit through module bodies: 6 of 15 reachable module types, and 24 of 64
module calls (37.5%), have no two primitives connected, i.e. are independent NANDs in a
box. The cause is `compress`, inherited from ECGP: it wraps genome-adjacent nodes, and
genome position says nothing about wiring.

## 3. necgp_pairwise — compress only what is wired together

`compress` always proposes one genome-adjacent pair and applies it only if the
resulting module's outputs depend on an internal connection
(`module_has_interaction`); `validate()` asserts that no fake module survives. Module
mutation and interface operators are removed, so a module's interface is fixed when it
is made. `max_module_size` (5) caps a module's flattened NAND count. Details:
`necgp_pairwise/README.md`.

**Run** (PyPy, 5 seeds, ~1 minute; figures under CPython):

```bash
cd necgp_pairwise
../../experiment_5/.venv-pypy/Scripts/pypy.exe train.py --seeds 0-4 --tag base
python render.py runs/base
```

| seed | solved at generation | module ids ever active | distinct functions | modules active at the end (size in NANDs) |
|---|---:|---:|---:|---|
| 0 | 106,622 | 5 | 3 | 5 (2–3) |
| 1 | 80,696 | 9 | 2 | 8 (2–3) |
| 2 | 47,578 | 6 | 4 | 6 (2–4) |
| 3 | not solved (242/256) | 7 | 4 | 7 (2–3) |
| 4 | 126,281 | 6 | 3 | 6 (2–3) |

- No fake module in any seed, by construction.
- Modules made early persist as called functions to the end.
- They are tiny (2–4 NANDs), and the same function recurs under different ids:
  NAND(NAND(a, b), c) accounts for 8 of 9 module ids in seed 1.
- No XOR module (4 NANDs, the obvious reusable part of `retina/xor`) appears.

**Why modules are not kept or grown** (`necgp_pairwise/analysis/`, measured on
`runs/base`, 234 snapshots; commands and outputs in that folder's README):

- *Compress rarely has anything to merge.* 36.7 of 100 nodes are active, only 6.3% of
  used gate-to-gate wires join genome-adjacent nodes, and 0.62% of `compress` attempts
  succeed (0.26% with both gates active).
- *Reordering the genotype would expose most wires.* 76.7% of the other used wires can
  be made adjacent without changing the circuit; 70.6% also fit the size cap.
- *Calls made by `compress` do not survive.* No final circuit contains one. The module
  calls in final circuits (26–50 per seed, against 1–6 plain NANDs) were all made by
  point mutation, which swaps NAND and module freely.
- *Many calls are NANDs in disguise.* 34% of active mutation-made calls are read only at
  an output equal to NAND of their first two inputs.
- *Reusable parts are large in NANDs.* Minimum NAND counts (exhaustive, ≤4 inputs): XOR,
  MUX, AND3 and NOR 4; XNOR 5; OR3 and MAJ3 6; NOR3 7; the retina's object detector
  needs more than 7, above the size cap.

Not done: a flat-CGP baseline for how often the same function recurs without modules,
a knock-out of the most-used module, and module lineage tracking.
