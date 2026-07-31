# kashtan_alon — reproducing Kashtan–Alon's MVG → modularity result

A faithful, self-contained reproduction of the **Kashtan–Alon retina** experiment
using **their kind of network** (a layered feedforward net with discrete integer
weights, evolved by a mutation GA) — deliberately *outside* the thesis's NDP /
g-encoding framework, as an external reference point. The question it answers:

> **Does changing the task (Modularly-Varying Goals) spontaneously favour a
> modular brain, versus a fixed goal?**

See [`PAPER_SPEC.md`](./PAPER_SPEC.md) for the exact paper numbers and sources.

## Why this is separate from `experiments/`
`experiments/` studies *our* model (fixed-neuron, non-spatial, evolved encoding).
This folder instead runs the **paper's own** model so we have a trustworthy
baseline for the MVG→modularity claim and for the Newman-Q metric. It shares only
the **retina task** (`../experiments/shared_tasks.py`) so numbers stay comparable.

## The setup (faithful to the paper)
- **Net:** strictly layered feedforward `8→8→4→2→1`, `tanh(λ·z)` with `λ=20`,
  weights ∈ `{−2,−1,1,2}` (0 = absent, so **topology evolves and stays sparse**),
  biases ∈ `{−2..2}`. Bipolar `{−1,+1}` inputs.
- **Search:** discrete mutation GA — add/remove connection (20% each), weight ±1
  (prob 2/n), bias ±1 (prob 1/24); tournament selection + elitism; performance-only
  (no connection-cost term — that is Clune's driver, not Kashtan–Alon's MVG).
- **Modularity:** Newman **Q** on the undirected connection graph (`modularity.py`),
  via a greedy community split. This is the project's long-missing metric; it is
  written framework-agnostically so it can be promoted to a shared module later.

## Files
- `run_paper.py` — **the canonical "exactly as in the paper" command** (MVG vs
  Fixed-Goal, every parameter locked to KA 2005). Start here.
- `model.py` — layered net, population-vectorised in numpy (`weights[l]`,`biases[l]`).
- `ga.py` — discrete mutation operators + tournament selection.
- `modularity.py` — Newman Q, community split, density (the reusable metric).
- `train.py` — the single-run GA loop `run_paper.py` drives; use it directly for
  sweeps/variations. `--mvg` vs fixed-goal; logs Q/density/fitness to `runs/*_log.csv`.
- `visualize.py` — draws the best net with **nodes coloured by module**.
- `tasks.py` — the retina/boolean tasks in **pure numpy (no JAX)**; kept identical
  to `../experiments/shared_tasks.py` by `test_tasks.py`.
- `test_tasks.py` — parity guard: asserts the numpy tasks are bit-identical to the
  shared (JAX) definitions. Run after touching either file.
- `PAPER_SPEC.md` — the paper numbers + a confirmed-vs-reconstructed source audit.

**Dependencies:** numpy, networkx, matplotlib only — **no JAX, no GPU, 0 VRAM**.
The GA is CPU/BLAS (batched `matmul`), ~0.033 s/gen at pop 1000 (~2.3 h for the
full 5-seed paper run).

## Run — exactly as in the paper
`run_paper.py` runs the paper's central comparison (MVG vs Fixed-Goal) with every
parameter locked to Kashtan–Alon 2005 (8-8-4-2-1, tanh 20, weights {−2,−1,1,2},
pop 1000, 25000 gens, add/remove 20%, weight ±1 at 2/n, bias ±1 at 1/24, and **raw
fraction-correct fitness** over all 256 patterns):
```
conda run -n lndp python run_paper.py --n-seeds 5 --viz   # the full paper run (slow)
conda run -n lndp python run_paper.py --smoke             # tiny, just checks the pipeline
```
It prints mean Newman Q for MVG vs FG at the end. **Expected (paper):** Q climbs
and stays high (~0.4+) under MVG, stays low (~0.15–0.2) under the fixed goal.

### Running a single condition / variations (train.py)
```
python train.py --mvg --switch-interval 20 --n-seeds 5 --viz      # MVG only
python train.py --operation and --n-seeds 5 --viz                 # fixed retina goal
python train.py --operation or --fitness balanced --n-seeds 5     # shortcut-free FG control
```
Key flags: `--fitness {raw,balanced}` (default `raw` = the paper; `balanced` dodges
the AND shortcut), `--mvg`/`--mvg-ops`/`--switch-interval`, `--pop`/`--generations`.

### Long runs: checkpoints & resume
Full GA state is checkpointed every `--checkpoint-interval` gens (default 1000) to
`runs/<run>_ckpt.pkl` (atomic write), `runs/<run>_best.npz` is refreshed, and the
CSV is flushed each log step so intermediary results are readable mid-run.
`--resume` (**on by default**) continues an interrupted run from its last
checkpoint and skips seeds already finished (marked by `runs/<run>_result.json`);
the checkpoint is deleted on completion. Force a clean restart with `--no-resume`
(train.py) or `--fresh` (run_paper.py). So a killed 2-hour run picks up where it
left off, not from scratch.

## Caveats
- **Run length.** Defaults mirror the paper (`--pop 1000 --generations 25000`);
  the Q separation only appears over many generations. numpy is fine for that but
  CPU-bound — do full runs in a per-experiment chat / on the GPU box, not here.
- **AND shortcut.** The paper's fixed-goal control **is** `L AND R` (`--operation
  and`), and under the paper's raw fitness a constant-false net scores ≈0.81 — so
  FG plateaus there for a long time (visible in the logs). That's the paper's
  setting and we keep it: the *point* of the control is that its **Q stays low**,
  which holds regardless of the shortcut. To probe the task without the shortcut,
  `--fitness balanced` or `--operation or`/`xor` (a variation, not the paper).
  MVG's `and,or` alternation is not fooled by a constant output.
- **Object patterns.** KA's exact retina bit-patterns are unpublished; we use the
  project's `L=(p0∧p1)∨(p2∧p3)`, `R=(p4∧p5)∨(p6∧p7)` — same left-4/right-4 modular
  structure, identical to experiments 1–3.
