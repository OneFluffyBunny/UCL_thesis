# Experiment 1 — results

Compressed cell-type encoding (model: `../README.md`). Search is CMA-ES unless noted.
Runs are written to `runs/` (not committed); every number below names the command
or script that regenerates it.

**Reproducibility.** JAX/CMA-ES runs are not bit-identical across repeats on this
backend (float reduction order, amplified by CMA-ES), so a re-run is a new sample of
the same arm: compare distributions over seeds, not trajectories. Replays made with
`analysis/dense_replay.py` are bit-identical to each other.

## 1. FG vs MVG × synaptic budget (4 arms × 5 seeds) — the main study

**Setup.** `retina_ka2005` (Kashtan & Alon's retina, 256 patterns), reference goal
L AND R; MVG alternates AND ↔ OR every 20 generations. 8 inputs, 24 hidden, 1 output
(768 allowed edges), K = 8 cell types (443 genes), CMA-ES popsize 64, σ₀ = 0.1,
margin fitness, **raw** accuracy (`--no-balanced`), 10,000 generations, seeds 0–4.
Constrained arms: `--synaptic-budget 6 --shrink 0.9`. The other arms have no budget.

```bash
cd experiments
python run_fgmvg_study.py --experiment 1 --lanes 10        # 20 runs, resumable
python analysis/run_all.py --root experiment_1/runs/fgmvg   # tables + figures
```

Each run is `train.py --n-hidden 24 -K 8 --task retina_ka2005 --operation and
--no-balanced --fitness margin --no-early-stop --no-open --archive-interval 1
--generations 10000 [--mvg --mvg-ops and,or --switch-interval 20]
[--synaptic-budget 6 --shrink 0.9] --seed N --n-seeds 1`.

Why raw accuracy: the four (L, R) cells hold 64 patterns each, so raw accuracy is the
mean over cells and any network that reads only one half scores at most 0.750.
Balanced accuracy would give that same one-half network 0.833.

Why these settings (3 seeds, 2,000 generations each, FG): K = 8 reached 0.914 ± 0.012
and K = 6 0.906 ± 0.018, and at K = 8 a non-modular network is expressible, so the
encoding does not force modularity. Among budgets, S = 6, τ = 0.9 matched S = 4,
τ = 0.9 in accuracy (0.854 vs 0.857) with far less spread in density (±0.5 vs ±12.0
points); S = 2 left every seed at 0.812.

**Which network is measured.** Every end-of-run number is the *goal-matched*
champion: the last champion selected under AND (generation 9,999 for FG, 9,979 for
MVG). A 10,000-generation MVG run ends inside an OR epoch, so its final champion
would be an OR network.

**Results** (mean ± SD over 5 seeds; density over the 768 allowed edges, an edge being
|w| > 0.05):

| condition | arm | acc (AND) | density % | `lr_r` | Q | seeds with `lr_r` above null |
|---|---|---|---|---|---|---|
| budget | FG | 0.895 ± 0.017 | 43.0 ± 5.3 | +0.094 ± 0.068 | 0.169 ± 0.096 | 0/5 |
| budget | MVG | 0.853 ± 0.016 | 34.0 ± 7.6 | +0.199 ± 0.217 | 0.208 ± 0.154 | 2/5 |
| no budget | FG | **0.978 ± 0.022** | 93.7 ± 6.4 | −0.013 ± 0.007 | 0.048 ± 0.024 | 0/5 |
| no budget | MVG | 0.867 ± 0.042 | 96.9 ± 5.1 | −0.024 ± 0.005 | 0.010 ± 0.014 | 0/5 |

`lr_r` is Newman's assortativity at the planted left/right split; the null is 200
degree-preserving rewirings that respect the allowed-edge mask (p < 0.05). Per-seed
values, Q_m and purity are in `metrics_per_seed.csv`; see the metric notes below for
why those two are not used here.

Findings:

1. **The task is solvable under this encoding.** The unconstrained FG arm reaches
   0.978, with seed 3 at 1.000 (Kashtan & Alon's own network: 0.90 ± 0.03).
2. **The budget produces the measurable modularity; goal switching does not.**
   Constrained vs unconstrained: Q 4–20× higher, `lr_r` changes sign (8 of 10
   constrained runs positive, all 10 unconstrained negative), density 34–43% vs
   94–97%. Without the budget the true density is 100% (no allowed edge is exactly
   zero), so modularity is not low there, it is *undefined*.
3. **MVG does not beat FG.** Within the budget, MVG is less accurate (0.853 vs 0.895)
   and its higher `lr_r` is not significant (exact one-sided Mann-Whitney U = 15,
   p = 0.345). It is also confounded with density: MVG runs 9 points sparser, and
   across the 10 constrained runs `lr_r` correlates with density at r = −0.47. In
   experiment 2, where the two arms have equal density, the gap disappears.
4. **The budget costs accuracy** (0.978 → 0.895), so constrained and unconstrained
   arms are different competence regimes. FG vs MVG within one condition is the
   clean comparison.

**No network can hold both goals.** The two goals agree on 128 of the 256 patterns
and disagree on the other 128, and the network gets no goal cue, so for any network
acc(AND) + acc(OR) ≤ 1.5; a perfect AND network scores exactly 0.5 on OR. Low OR
accuracy of an AND-matched MVG champion (0.42–0.47 here) is therefore expected, and
the AND/OR antiphase in the switch-window figure is forced by arithmetic. The test of
Kashtan & Alon's claim is re-adaptation speed (section 2).

**Metric notes.**
- *`lr_r` depends on the edge cut.* Seed 0, goal-matched champion, density | `lr_r` at
  cuts 0 / 0.05 / 0.10 / 0.20 (`analysis/score_table.py --threshold t`):

  | arm | 0 | 0.05 | 0.10 | 0.20 |
  |---|---|---|---|---|
  | budget FG | 50.5% +0.067 | 45.8% +0.175 | 39.2% +0.449 | 31.6% +0.197 |
  | budget MVG | 33.1% −0.007 | 27.6% +0.004 | 27.3% −0.000 | 27.3% −0.000 |
  | no budget FG | 100% −0.026 | 99.2% −0.015 | 97.4% +0.000 | 93.5% +0.013 |
  | no budget MVG | 100% −0.026 | 88.2% −0.025 | 88.2% −0.025 | 88.2% −0.025 |

  The unconstrained null result holds at every cut; a positive `lr_r` in a budgeted
  arm does not have a cut-independent value, so quote it with its cut.
- *`lr_r` ≈ 2Q* here (its ceiling is ~0.5 in all 40 runs of this study and
  experiment 2's), so it adds a stable, bounded scale rather than new information.
- *Q_m* (Kashtan & Alon's normalised Q) is not used: it is undefined in 8 of 20 runs,
  saturates at 1.000 for both the least and the most modular constrained run, and its
  degree-preserving null is not comparable across encodings (experiment 2, section 3).
- *Purity* (`shared_brain_metrics.recurrent_purity`) is descriptive only. In a
  recurrent network the unrolled side-mixture drifts towards 0.5 with every step, so
  the score mostly tracks density and spectral gap.

**Figures:** `latex_figures/experiment_1_fgmvg/` (README there has the commands).

## 2. Re-adaptation after a goal switch — the bottleneck speeds it up

Kashtan & Alon's claim is that MVG populations re-adapt faster after each switch.
Measured on the budgeted MVG arm of both encodings, using per-generation replays of
seeds 0–4 in two windows, [100, 300] and [1000, 1200] (10 goal epochs each), and the
population mean accuracy on the active goal.

```bash
cd experiments/analysis
python dense_replay.py --root ../experiment_1/runs/fgmvg --constraint budget --goal mvg --seed N   # N = 0..4
python dense_replay.py --root ../experiment_2/runs/fgmvg --constraint budget --goal mvg --seed N
python recovery_stats.py
```

Generations to cover 90% of the climb from the trough at the switch to the epoch's
own peak, mean over epochs per seed:

| encoding | [100, 300] | [1000, 1200] | per-seed change | seeds faster |
|---|---|---|---|---|
| compressed (exp 1) | 6.30 ± 1.25 | **4.20 ± 0.24** | −2.4, −1.7, −2.1, −0.5, −3.8 | 5/5 |
| direct (exp 2) | 5.90 ± 0.52 | **8.36 ± 1.14** | +3.6, +3.8, +2.0, +1.7, +1.2 | 0/5 |

The per-seed changes do not overlap; exact two-sided Mann-Whitney p = 0.008 (the
smallest value possible at 5 vs 5). The same ordering holds for measures that do not
use the epoch's peak (early → late):

| measure | compressed | direct |
|---|---|---|
| gain in the first 3 generations | 0.196 → 0.299 (5/5 faster) | 0.237 → 0.152 (0/5) |
| generations to climb +0.20 | 6.1 → 2.3 (5/5) | 3.0 → 4.2 (0/5) |
| generations to reach 0.75 | 7.3 → 3.6 (5/5) | 4.4 → 5.0 (0/5) |
| generations to climb +0.30 (censored at 20) | 15.9 → 4.2 (5/5) | 4.6 → 5.8 (0/5) |

So the compressed encoding speeds up and the direct encoding slows down. Mean CMA-ES
σ over each window is unchanged or slightly larger late in both encodings (late/early
1.00–1.21 direct, 0.96–1.54 compressed, from `log.csv`), so the slowdown is not step
size collapse. The champion shows the same direction but noisily (compressed 4/5
faster, direct 1/5), because a maximum over 64 samples barely drops at a switch.

**Trough at the switch.** Right after a switch, accuracy on the new goal is
½ + ½(a − b), where a and b are the old-goal accuracy on the patterns where the
goals agree and disagree. Late troughs: direct 0.500 ± 0.001 (equally good on both
halves), compressed 0.438 ± 0.022 (better on the half the switch flips). That the
compressed population concentrates its competence on the goal-discriminating patterns
is a possible reason it re-adapts faster; it is not tested.

**Reading, together with section 1:** the bottleneck gives faster re-adaptation at a
cost in accuracy (0.895 vs 1.000) and without measurable left/right modularity.

Caveats: n = 5 per encoding, one task, one budget setting, windows taken from Kashtan
& Alon rather than chosen here. MVG shows no modularity effect in either encoding; the
optimiser, which both arms share, is the obvious suspect (section 3).

## 3. Kashtan-Alon GA instead of CMA-ES — pilot, one seed

CMA-ES has no per-gene locality, no recombination and no surviving elites, all of
which Kashtan & Alon's mechanism relies on. The pilot swaps the optimiser and nothing
else: population 600, top 150 kept, crossover p = 0.5 per offspring (each gene block
— a cell type, the input or output type, one hidden unit of `g`, `g`'s output bias —
from one parent), mutation p = 0.5 of one gene by N(0, 0.5²) [step size our choice],
3,000 generations (1.8M evaluations). Operators are tested in `test_ga.py`.

```bash
python train.py --strategy KA_GA --popsize 600 --ga-elite 150 --ga-pc 0.5 --ga-pm 0.5 \
    --ga-mut-sigma 0.5 --n-hidden 24 -K 8 --task retina_ka2005 --operation and --no-balanced \
    --fitness margin --no-early-stop --no-open --archive-interval 1 --generations 3000 \
    --mvg --mvg-ops and,or --switch-interval 20 --synaptic-budget 6 --shrink 0.9 \
    --seed 0 --n-seeds 1 --dense-log 100:300,1000:1200 --out-dir runs/ga_pilot
```

Rules fixed before the run: better accuracy means acc(AND) ≥ 0.869 (CMA-ES MVG mean
+ 1 SD); more modular means `lr_r` ≥ 0.416 and above its null at cuts 0.05 and 0.

| | generation | acc (AND) | density | `lr_r` (cut 0.05) |
|---|---|---|---|---|
| GA, at CMA-ES's 640k evaluations | 1,059 | 0.891 | 51.7% | +0.028, n.s. |
| GA, end | 2,979 | **0.938** | 57.3% | +0.040, n.s. |
| CMA-ES seed 0, end | 9,979 | 0.836 | 27.6% | +0.004, n.s. |

- **Accuracy: yes.** Above the threshold at matched evaluations, and above every
  CMA-ES MVG seed and the CMA-ES FG mean (0.895).
- **Modularity: no.** `lr_r` stays near zero and never beats its null; the GA brains
  are denser.
- **Re-adaptation does not speed up** (90% climb 8.0 → 8.3 generations; the
  population mean includes the unchanged elites, so this is descriptive only).

With n = 1 this licenses a full study, not a claim. Not yet separated: whether the
accuracy gain needs MVG (a GA fixed-goal run), and whether the missing modularity is
the encoding (this genome has no locus holding a left or right detector, so crossover
cannot move one) — the direct encoding under the same GA would test that.

## 4. Earlier findings that still hold

**Check tasks for shortcuts before reading a plateau.**
- On the stand-in `retina` task with AND, P(y = 1) = 0.19, and a network that
  computes only one side scores 0.848 balanced accuracy; a random population already
  sits at 0.843. The ~0.85 plateaus seen on that task (at K = 4 and K = 6, with
  accuracy or margin fitness, with or without a curriculum) are this shortcut.
  `retina/xor` and `retina_ka2005` (raw accuracy) do not have it.
- The best *monotone* Boolean function (which needs no inhibitory weight) reaches
  1.000 on the stand-in `retina` with AND or OR, 0.891 on `retina_ka2005`/AND, 0.875
  on /OR and 0.691 on /XOR (minimum flips to make the target monotone).

**`left` is reachable in principle but not found.** `oracle.py --task left`
hand-builds a perfect brain and fits `g` to it (residual 0.000; the resulting genome
scores 1.000 at `g` width 16 and 128). `reachability.py` shows CMA-ES started exactly
at that genome drifts away (to 0.857), and no perturbed start recovers it: the optimum
is narrower than the search step. `oracle.py --task retina` likewise builds a perfect
6-type brain for the stand-in retina (1.000).

**An absolute weight gate does not control density.** `--w-threshold 0.2` on
`retina_ka2005` (K = 6, 24 hidden, 2,000 generations): evolution scales `g`'s output
up 4–10× and density goes from 0% to 77% (FG) and 100% (MVG), with 77% of MVG's
synapses at |w| > 0.999. The relative synaptic budget cannot be escaped this way:
multiplying `g`'s output by 100 leaves a budgeted brain unchanged (max diff 3e−8),
and under it density falls during evolution (76–95% → 28–55%) while the budget holds
exactly.

**A discovered partition can be the wrong one.** A budgeted run scored Newman
Q = 0.20 while its planted left/right split was at chance (p = 0.87): one cell type
read the left half and the rest was a general pool. Hence `lr_r` at the planted split
rather than an unlabelled Q.

**E = 20 is justified by measured recovery.** With E = 200 (`runs/mvg_diag_E200`, 28
post-switch epochs) the champion recovers to within 0.005 of its epoch maximum in a
median of 10 generations (75% within 20). E = 20 is about twice the recovery time, so
the population adapts but never settles.

**Curriculum vs cold start: null** (`curriculum.py`, stand-in `retina`/AND, K = 4,
5 paired seeds): `left` for 200 generations then `retina/and` for 600 vs 800
generations of `retina/and` gave 0.862 vs 0.858, neither reaching 0.90 and neither
lateralised. The task's one-side shortcut makes this uninformative about modularity.

**Margin fitness changes the search, not the ceiling.** On the stand-in `retina`/AND
(K = 4, 5 seeds) margin gave 0.858 vs 0.854 for accuracy fitness, but accuracy-fitness
brains were single-sign and fully dense while margin brains mixed signs and were
sparser (39–100%).

## Open questions

- A density-matched comparison across encodings: prune each experiment-1 champion to
  experiment 2's edge count and rescore (the saved genomes make this cheap).
- The GA pilot's two missing controls (section 3).
- Matched evaluation budgets across encodings (experiment 1 ran 640k evaluations per
  seed, experiment 2 320k).
