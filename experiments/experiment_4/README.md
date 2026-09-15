# Experiment 4 — CGP vs ECGP on the Kashtan–Alon retina

Experiments 1–3 ask whether the final network is modular. This experiment asks a
prior question: **when modularity appears under goal switching, is structure being
kept and reused, or rediscovered after every switch?**

The substrate is Cartesian Genetic Programming (CGP, Boolean circuits) and its
module-acquiring variant ECGP (Walker & Miller, IEEE TEVC 12(4), 2008). Circuits are
used because a module's birth, reuse and death are explicit, countable events, which
they never are in a weight matrix.

## Hypotheses

A high final modularity is compatible with two histories:
- **(a) rediscovery**: every switch wrecks the solution and a modular one is re-found
  because it is the fastest to reach;
- **(b) persistence**: a sub-structure found once survives the switches and is
  redeployed.

**H1.** Goal switching alone gives (a); (b) needs an encoding with a unit of
inheritance below the whole genome. ECGP adds exactly that (the module) to CGP and
changes nothing else, so

| | FG (fixed L AND R) | MVG (AND ↔ OR every E generations) |
|---|---|---|
| **CGP** | baseline | does Kashtan & Alon's result replicate? |
| **ECGP** | does acquisition help without switching? | the cell of interest |

Signature of (b): recovery time after a switch shrinks over successive switches under
ECGP and stays flat under CGP; module lifetimes span many switch periods. Under (a)
ECGP gives no benefit under MVG.

**H2/H3/H4** (logged, not tested separately): acquired modules implement the task's
decomposition; reuse protects a module from loss; a useful module proliferates under a
fixed goal. The retina's solution contains the same half-detector twice, so it has a
built-in reusable module. A proliferation claim needs a knock-out control (replace a
module with a random one of the same shape; fitness must drop), which was not run.

Results: `RESULTS.md`. In short, H1's persistence signature is not observed with
either CGP or ECGP under the (1+4) ES, while a population GA on plain CGP reproduces
Kashtan & Alon's MVG → modularity effect.

## Files

| file | what it is |
|---|---|
| `cgp.py`, `gates.py`, `tasks.py` | CGP genotype, mutation, bit-parallel evaluation; gate set; the retina task (loaded from `kashtan_alon/tasks.py`) |
| `ecgp.py` | ECGP: modules, `compress`/`expand`, module mutation operators, flattening |
| `train.py` | the (1+4) ES, fixed goal or `--mvg`, per-switch recovery log, gate census, checkpoints |
| `train_pop.py` | the same circuits under Kashtan & Alon's GA (population 600, 150 elites, crossover) |
| `visualize.py`, `decompose.py` | circuit drawings (modules as single boxes, fake modules grey) |
| `config.py` | flags; paper values tagged `[Table II]` |
| `analysis/` | scripts behind the numbers and figures in `RESULTS.md` |
| `test_*.py` | correctness tests (`test_perf.py` checks throughput and is machine-dependent) |
| `PAPER_SPEC.md` | every parameter's source, tagged `[verbatim]` / `[inferred]` / `[our choice]` |

⚠️ **The search code is frozen.** Results are identified by seed, so `train.py`,
`cgp.py` and `ecgp.py` must not change behaviour. `../experiment_5/` is the fork that
may move, and `experiment_5/test_equivalence.py` checks it still walks the same search.

## Run

```bash
python train.py --task retina_ka2005 --operation and --nodes 50 --n-seeds 12            # CGP, FG
python train.py --ecgp --task retina_ka2005 --operation and --nodes 50 --n-seeds 12     # ECGP, FG
python train.py --task retina_ka2005 --mvg --switch-interval 2000 --nodes 50 --n-seeds 5 # CGP, MVG
python train_pop.py --mvg --switch-interval 2000 --generations 100000                  # GA, MVG
```

This folder runs under CPython only (`tasks.py` imports NumPy). At 8 inputs the same
search is ~6× faster under PyPy via `../experiment_5/`, but seeds do not transfer
between the two folders (experiment 5 changed the order in which mutation slots are
written).
