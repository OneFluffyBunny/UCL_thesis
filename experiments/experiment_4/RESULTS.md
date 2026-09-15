# Experiment 4 — results

Boolean circuits on Kashtan & Alon's retina (`retina_ka2005`, 8 inputs, 256
patterns), fitness = correct patterns (raw), fixed goal L AND R (FG) or AND ↔ OR every
E generations (MVG). Search is the CGP paper's (1+4) ES (`train.py`, one parent and
four mutated offspring, offspring win ties) unless noted; `train_pop.py` is Kashtan &
Alon's population GA on the same circuits. Function set `and,nand,or,nor` unless
noted. Design and hypotheses: `README.md`; every parameter's source: `PAPER_SPEC.md`.

Runs are seeded with `random.Random(seed)` and are bit-reproducible with the current
code. Runs from before 2026-08-13 used a different RNG and do not reproduce by seed.

## 1. CGP, fixed goal: genotype size

12 seeds per size, `--mutation-rate 0.03`, stop at 256/256:

| nodes | solved | median generations | median evaluations | active gates in solved circuits (min / median / max) |
|---|---|---|---|---|
| 25 | 11/12 | 77,484 | 338,745 | 16 / 17 / 18 |
| 50 | 12/12 | 37,121 | 148,493 | 17 / 20 / 26 |
| 100 | 12/12 | 21,450 | 85,809 | 18 / 23 / 30 |
| 400 | 12/12 | 9,812 | 39,259 | 21 / 47 / 65 |

```bash
python train.py --task retina_ka2005 --operation and --nodes N --mutation-rate 0.03 --n-seeds 12
```

- **Longer genotypes search faster**, as the ECGP paper and its reference [19] (Miller
  & Smith 2006) report: inactive nodes give neutral drift room. Plateaus are not
  stagnation: while fitness sits flat for thousands of generations the active circuit
  keeps changing.
- **They also bloat.** At 400 nodes the median solution is 47 gates; at 25 nodes 17.
  50 nodes is the knee (every seed solves, near-minimal circuits), and is the size
  used in the FG-vs-MVG studies below.
- The smallest solved circuit has 16 gates.
- For calibration, Walker & Miller's computational effort for even-5 parity is 130,081
  evaluations with the same algorithm, so the retina is about as hard for CGP as
  even-5 parity. Kashtan & Alon's GA needed ~1.7M evaluations on their networks.

## 2. FG vs MVG: (1+4) ES gives a null, the population GA reproduces Kashtan & Alon

### (1+4) ES

5 FG + 5 MVG seeds, 50 nodes, E = 2,000, 800,000 generations (3.2M evaluations):

```bash
python train.py --task retina_ka2005 --operation and --gates and,nand,or,nor --nodes 50 \
    --mutation-rate 0.03 --fitness raw [--mvg --mvg-ops and,or --switch-interval 2000] \
    --popsize 5 --generations 800000 --no-stop-on-solution --n-seeds 5 --archive-interval 100 \
    --dense-archive 0:6000,86000:92000,792000:798000 --tag arch --out-dir runs/fgmvg50
```

MVG circuits are measured at the end of the last AND epoch (generation 797,999), FG
circuits at the end. Purity is `qmetrics.circuit_purity` on the active circuit, output
gate excluded (1 = every gate reads only one retina half).

| arm | acc (AND) | circuit purity | active gates |
|---|---|---|---|
| FG | 1.000 in 5/5 | 0.82–0.90 | 15–24 |
| MVG | 0.81–0.85, 0/5 solved | 0.25–0.81 | 7–18 |

Every FG seed is purer than every MVG seed (Mann-Whitney U = 0, p ≈ 0.008), but **this
is a solving effect, not an MVG effect.** Taking every champion (sampled every 100
generations) with 0.81 ≤ acc(AND) ≤ 0.85, the two arms are equally pure and equally
small: FG n = 158, median purity 0.55, 7 gates; MVG n = 414, 0.53, 8 gates
(`analysis/matched_accuracy_purity.py`). A full solution forces two pure
half-detectors; the cheaper 0.84 circuits mix the halves in either arm. MVG never
holds a solution, its recovery after a switch takes 250–420 generations and does not
speed up, and it discards partial structure every 2,000 generations while FG needs
12k–115k generations of uninterrupted search to solve.

**What was ruled out along the way** (from the per-switch `*_recovery.csv` logs of
earlier MVG sweeps, 16 seeds per arm, 800,000 generations):
- *The switch interval.* E = 2,000 at (1+4) is 10,000 evaluations per goal epoch,
  within 20% of Kashtan & Alon's 20 generations × 600. With E = 200 the median level
  before a switch stays at 212/256 for the whole run.
- *Genotype size.* At 50, 100 and 400 nodes no lineage keeps a solution across a
  switch: across 44,704 goal epochs only 7 started from a perfect circuit (all at 400
  nodes), and none of those recovered within its epoch.
- *Mutation step size* (`--wiring-weight`, making rewiring rarer): smaller steps give
  faster recovery to a lower plateau, monotonically worse.
- *No learning to switch.* Per-seed Spearman correlation of recovery time with epoch
  number is ≈ 0 in every arm.

The (1+4) ES is a single lineage, so "recovers faster" is never selected: there is no
slower rival to beat.

### Kashtan & Alon's population GA

`train_pop.py` changes only the search loop: population 600, top 150 copied unchanged,
450 children of two uniformly drawn elites, per-node crossover p = 0.5 (a node's
function and inputs from parent A or B), mutation p = 0.5 per child with `cgp.mutate`
at 3% of gene slots. 50 nodes, 100,000 generations (~33.8M evaluations per seed).

```bash
python train_pop.py --generations 100000 --log-interval 1000 --out-dir runs/fgmvg50_ga_fg
python train_pop.py --mvg --switch-interval 2000 --generations 100000 --log-interval 1000 --out-dir runs/fgmvg50_pop
python train_pop.py --mvg --switch-interval 200  --generations 100000 --log-interval 1000 --out-dir runs/fgmvg50_pop_E200
python train_pop.py --mvg --switch-interval 20   --generations 100000 --log-interval 1000 --seed 3 --n-seeds 1 --workers 1 --out-dir runs/fgmvg50_pop_E20
```

(Defaults, recorded in each `config.json`: `--nodes 50 --mutation-rate 0.03 --pop 600
--n-elite 150 --pc 0.5 --pm 0.5 --archive-interval 1`, seeds 0–4. E is not in the run
name, hence one directory per E. Rerunning seed 0 of FG and MVG E = 2,000 for 2,500
generations reproduces the stored archives row for row.)

| arm (5 seeds) | acc (AND) = 1 at the end | purity at the end | purity of solved circuits, 2nd half |
|---|---|---|---|
| FG | 3/5 | 0.60, 0.69, 0.85, 0.92, 0.92 | 0.60–0.92, never 1.00 |
| MVG, E = 2,000 | 4/5 | 1.00 in 5/5 | 1.00 |
| MVG, E = 200 | 1/5 | 1.00 in 5/5 | 1.00 |
| MVG, E = 20 (seed 3 only) | 0/1 (best 242/256) | 0.72 | — |

- **With a population, MVG produces fully modular circuits and the GA alone does
  not.** The MVG circuits are a pure left detector and a pure right detector joined
  only at the output gate, so AND ↔ OR is a one-gate change. Solved FG circuits are
  never fully pure, so the difference is not an accuracy confound. End purity MVG > FG
  in every seed pair (U = 0, p ≈ 0.008) at E = 2,000 and E = 200. MVG circuits are
  also smaller (16 vs 18.6 gates).
- **Recovery after a switch takes 1–2 generations** (median). Sometimes the champion
  does not drop at all, because the output-flipped variant of the champion is already
  in the population (MVG seed 0 at generation 98,000; the population mean still falls
  from 220.9 to 127.3 correct patterns).
- **Shorter epochs cost accuracy, not modularity.** Kashtan & Alon's E = 20 does not
  solve here (one seed).
- **Not budget-matched** to the (1+4) ES (10.5× more evaluations), but the (1+4) MVG
  arm was flat at ~0.84 for all 800,000 generations.

Figures for both searches: `latex_figures/experiment_4_fgmvg/`.

Open: crossover not ablated (pc = 0); one genotype size and task; 5 seeds per arm; E = 20
is one seed.

## 3. ECGP (module acquisition), fixed goal

ECGP (`--ecgp`, `ecgp.py`) adds `compress` (wrap a random run of adjacent genotype
nodes into a callable module) and `expand` (inline it back). The 14 cases the paper
leaves open are listed in `PAPER_SPEC.md` section 12. Verified in `test_ecgp.py`:
module evaluation equals evaluation of the flattened circuit, compress/expand are
fitness-neutral and invert each other, structural invariants hold under mutation, and
resume after a kill is exact.

**4-gate set, 50 nodes, 12 matched seeds:**

| arm | solved | median generations | median active gates |
|---|---|---|---|
| CGP | 12/12 | 37,121 | 19.5 |
| ECGP | 12/12 | 24,549 | 24.5 |

ECGP is faster on 9 of 12 seeds (per-seed ratio 0.24–1.55): suggestive, not
significant, in the direction Walker & Miller report. Its circuits are larger because
a module's body is copied at every call site. Nothing separates the candidate causes
(a larger function alphabet, protection of module bodies from mutation, a shorter
genotype, or genuine reuse); the knock-out test that would (replace a module with a
random one of the same shape) was not run.

**NAND-only function set, 12 seeds, 300,000-generation cap:**

| arm | solved | median generations |
|---|---|---|
| CGP, 50 nodes | 6/12 | 87,447 |
| CGP, 100 nodes | 12/12 | 38,073 |
| ECGP, 100 nodes | 12/12 | 56,571 |

- NAND alone can express the task; doubling the genotype restores CGP's search speed.
- **ECGP is slower than CGP here**, reversing the 4-gate result. Confound: with one
  primitive a CGP function-gene mutation does nothing, so a third of CGP's mutations
  are silent while ECGP's function genes can still switch to a module.
- **ECGP invents gates but does not keep them** (`analysis/nand_compositions.py`).
  Among 2-input, 1-output modules, 11 of 12 seeds actively use one computing
  A OR NOT B and 7 of 12 one computing AND at some point, compositions no single NAND
  gives. At the final circuit only 1 of 12 seeds still uses one.
- **Circuit size is a random walk, not a ratchet.** Running 30,000 generations past
  the first solve (`--post-solve-gens 30000`) moves the mean from 34.6 to 31.3 active
  gates, with 7 seeds smaller and 5 larger. Preferring the smaller circuit among tied
  offspring (`--parsimony-tiebreak`) leaves median size unchanged (33) and slows the
  median search by 56%.

**Most modules are not functional units.**
- *Fake modules.* `compress` wraps genome-adjacent nodes whether or not they are wired
  together. In seed 0's solved NAND-only circuit, 7 of the 9 module types used (24 of 30
  calls) contain no internal connection, i.e. are independent NANDs in a box
  (`analysis/decompose_seed0.py`). `ecgp.is_fake_module` detects these and every
  drawing greys them out.
- *Reuse under MVG* (`analysis/module_reuse.py`, final genotype of each seed, medians):

  | arm | modules per seed | called more than once in the active circuit | ever active |
  |---|---|---|---|
  | ECGP FG, 50 nodes | 5 | 78% | 100% |
  | ECGP MVG, 50 nodes | 10 | 29% | 61% |
  | ECGP MVG, 400 nodes | 68 | 33% | 71% |

## 4. ECGP under MVG

Same configuration as the CGP MVG arms: E = 2,000, 16 seeds, 800,000 generations,
4-gate set.

| | CGP 50n | ECGP 50n | CGP 400n | ECGP 400n |
|---|---|---|---|---|
| goal epochs starting from a perfect circuit | 0 / 6,384 | 0 / 6,384 | 7 / 6,384 | 0 / 6,384 |
| median recovery (generations) | 329 | 196 | 897 | 612 |
| trend of recovery time over the run | none | none | none | none |

- **Persistence and reuse (README H1) is not observed.** No recovery-time trend, and
  ECGP never starts an epoch from a perfect circuit.
- ECGP recovers faster to the same plateau at both sizes, consistent with module bodies
  mutating more slowly, not with memory.
- By the end of the 400-node runs a median 34 modules span 358 of 400 top-level nodes;
  under MVG most modules are stored rather than reused (table above).

Reading for the thesis: ECGP's module operators do create compound functions, but
under this task and search they are neither kept nor reused, and module counts
overstate reuse. Experiment 6 continues this with nesting.
