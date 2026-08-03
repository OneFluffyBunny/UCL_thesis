# kashtan_alon — reproducing Kashtan–Alon's MVG → modularity result

A faithful, self-contained reproduction of the **Kashtan–Alon 2005 retina**
experiment (their neural-network system) — deliberately *outside* the thesis's
NDP / g-encoding framework, as an external reference point. The question it answers:

> **Does changing the task (Modularly-Varying Goals) spontaneously favour a
> modular brain, versus a fixed goal?**

Every value is locked to the **verified primary source** (KA 2005, PMC1236541); see
[`PAPER_SPEC.md`](./PAPER_SPEC.md) for the exact quotes and sources.

> ⚠️ **History.** Runs 1–3 (see `RESULTS.md`) accidentally ran **Clune 2013's
> reimplementation**, not KA — the spec had been reconstructed from Clune and
> mislabelled "confirmed KA", and it produced a null. On 2026-08-03 the code was
> rewritten to the actual paper. If you touch this folder, read `RESULTS.md` and
> `PAPER_SPEC.md` first.

## Why this is separate from `experiments/`
`experiments/` studies *our* model (fixed-neuron, non-spatial, evolved encoding).
This folder runs the **paper's own** model, as a trustworthy baseline for the
MVG→modularity claim and for the Newman-Q metric. It uses **KA's own retina task**
(NOT the `shared_tasks.py` stand-in) — the two are intentionally different.

## The setup (faithful to KA 2005, verified)
- **Net:** layered feedforward `retina(8) → 8 → 4 → 2 → 1` (the 8 pixels are a
  separate input layer feeding the first neuron layer). **Hard-threshold** neurons
  (fire iff weighted sum + bias(=−θ) > 0, output `{0,1}`), weights ∈ `{−1,+1}`
  (0 = absent, topology evolves), **fan-in ≤3** in neuron layers 1–3 and **≤2** at
  the output. Binary `{0,1}` pixel inputs.
- **Search:** KA's GA — **elite strategy** (top `L=150` of `S=600` replicate
  unchanged), **crossover `Pc=0.5`** (offspring recombine two elite parents by
  inheriting whole per-neuron incoming columns), **mutation `Pm=0.5`** per genome
  (one edit: add/remove edge respecting fan-in, flip a weight sign, or nudge a
  threshold). Performance-only (no connection-cost term — that is Clune's driver).
- **Modularity:** KA's **normalized `Q_m = (Q_real−Q_rand)/(Q_max−Q_rand)`**
  (`modularity.py`), `Q_rand` averaged over 1000 degree-preserving randomizations.
  Raw Newman Q is logged per generation only as a cheap live trace; the **headline
  is the end-of-run `Q_m`**. Framework-agnostic → promotable to a shared module.

## Files
- `run_paper.py` — **the canonical "exactly as in the paper" command** (MVG vs
  Fixed-Goal, every parameter locked to KA 2005). Start here.
- `model.py` — the threshold net, population-vectorised in numpy (`weights[l]`,
  `biases[l]`), with per-layer fan-in caps.
- `ga.py` — the KA GA: elite selection + crossover + mutation.
- `modularity.py` — raw Newman Q **and** normalized `Q_m`, community split, density.
- `train.py` — the single-run GA loop `run_paper.py` drives; use it directly for
  sweeps/variations. `--mvg` vs fixed-goal; logs raw Q/density/fitness to
  `runs/*_log.csv`, final `Q_m` to `runs/*_result.json`.
- `visualize.py` — draws the best net with **nodes coloured by module**.
- `tasks.py` — retina/boolean tasks in **pure numpy (no JAX)**; the retina is KA's
  real Fig. 5a definition.
- `test_tasks.py` — checks KA's exact object truth counts (left 8/16, AND 64/256,
  OR 192/256, left/right independence) and copy/and2 parity with `shared_tasks.py`.
- `PAPER_SPEC.md` — the verified paper numbers + a source audit.

**Dependencies:** numpy, networkx, matplotlib only — **no JAX, no GPU, 0 VRAM**.
The GA is CPU/BLAS (batched `matmul`); the forward pass dominates runtime.

## Run — exactly as in the paper
`run_paper.py` runs the central comparison (MVG vs Fixed-Goal) with every parameter
locked to KA 2005 (`retina(8)→8→4→2→1`, ±1 weights, threshold units, fan-in 3/3/3/2,
pop 600, elite 150, crossover Pc=0.5, mutation Pm=0.5, 25000 gens, **raw
fraction-correct fitness** over all 256 patterns, Q_m over 1000 randomizations):
```
conda run -n lndp python run_paper.py --n-seeds 5 --viz   # the full paper run (slow)
conda run -n lndp python run_paper.py --smoke             # tiny, just checks the pipeline
```
It prints mean **Q_m** for MVG vs FG at the end. **Expected (paper):** `Q_m ≈ 0.35`
under MVG, `≈ 0.15` under the fixed goal.

### Running a single condition / variations (train.py)
```
python train.py --mvg --switch-interval 20 --n-seeds 5 --viz      # MVG only
python train.py --operation and --n-seeds 5 --viz                 # fixed retina goal
python train.py --operation or --fitness balanced --n-seeds 5     # shortcut-free FG control
```
Key flags: `--fitness {raw,balanced}` (default `raw` = the paper; `balanced` dodges
the AND shortcut), `--mvg`/`--mvg-ops`/`--switch-interval`, `--pop`/`--generations`,
`--n-elite`/`--pc`/`--pm` (KA search), `--qm-nrand` (final Q_m randomizations).

### Long runs: checkpoints & resume
Full GA state is checkpointed every `--checkpoint-interval` gens (default 1000) to
`runs/<run>_ckpt.pkl` (atomic write), `runs/<run>_best.npz` is refreshed, and the
CSV is flushed each log step so intermediary results are readable mid-run.
`--resume` (**on by default**) continues an interrupted run from its last checkpoint
and skips seeds already finished (marked by `runs/<run>_result.json`); the checkpoint
is deleted on completion. Force a clean restart with `--no-resume` (train.py) or
`--fresh` (run_paper.py).

## Caveats
- **Run length.** Defaults mirror the paper (`--pop 600 --generations 25000`); the
  Q_m separation only appears over many generations. numpy is fine but CPU-bound — do
  full runs in a per-experiment chat / on the GPU box, not the hub.
- **AND shortcut.** The paper's fixed-goal control **is** `L AND R` (`--operation
  and`); under raw fitness the retina AND goal has 64/256 positives, so a
  constant-false net scores **192/256 ≈ 0.75** — FG plateaus there for a while
  (visible in the logs). That's the paper's setting; the *point* of the control is
  that its **Q_m stays low**, which holds regardless. To probe the task without the
  shortcut, `--fitness balanced` or `--operation or`/`xor` (a variation, not the
  paper). MVG's `and,or` alternation is not fooled by a constant output.
- **SI residual.** The exact mutation-operator set and threshold range live in KA's
  supplementary (not the main text); `ga.py`/`model.py` use a documented faithful
  reconstruction. Everything else is quoted from the paper.
