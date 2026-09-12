# Experiment 2 — results & observations log

Running lab notebook for the direct-encoding control (see `README.md` for the
model, `../experiment_1/RESULTS.md` for the treatment this is measured against).
Newest entries at the bottom. Balanced accuracy throughout (chance = 0.5).
Bipolar inputs {-1,+1}. Search = CMA-ES over the raw weight vector (no shared
rule, one free parameter per edge) unless noted.

---

## retina/xor, n_hidden=20, 5 seeds, margin fitness

```
python train.py --task retina --operation xor --n-seeds 5 --fitness margin
```

Genome: 8→20 (IH) + 20×19 (HH, no self-loops) + 20→1 (HO) + 21 biases =
**581 free parameters**, optimized directly by CMA-ES (popsize=64, sigma_init=0.1,
default 1000 generations, early stop at target=1.0).

| seed | gens to solve | best balanced acc | edges used | density | exc(+) / inh(-) |
|---|---|---|---|---|---|
| 0 | 456 | **1.000** | 524/560 | 93.6% | 246 / 278 |
| 1 | 339 | **1.000** | 515/560 | 92.0% | 262 / 253 |
| 2 | 299 | **1.000** | 515/560 | 92.0% | 254 / 261 |
| 3 | 313 | **1.000** | 516/560 | 92.1% | 271 / 245 |
| 4 | 290 | **1.000** | 516/560 | 92.1% | 243 / 273 |

**The unstructured control solves `retina/xor` easily.** Every seed reaches
perfect balanced accuracy well inside the 1000-gen budget (290–456 gens) —
`retina/xor` is not a hard search problem for direct encoding the way `retina/and`
was a hard *representability* problem for experiment 1 at low K. No shortcut
confound here (xor is balanced, no one-side freebie), so this is a genuine
full-task solve, not a plateau artifact.

**Density stays ~92–94% (near fully-connected) in every seed.** With no
structural prior toward regular/modular wiring, CMA-ES converges on dense,
roughly balanced exc/inh solutions. This is the number to compare against a
modularity score once the metric exists (still not built — see
`../experiment_1/RESULTS.md`'s open threads): a near-fully-connected solved
network is a strong prior that the control is *not* modular, which is the
contrast the thesis needs against experiment 1's regular/compressed solutions.

**Caveat:** one fixed-goal run, no modularity metric yet, so "not modular" here
is an inference from density alone, not a measured score. `--mvg` (the known
positive-control pressure) not yet tried on this encoding.

## MVG (xor/and switching), n_hidden=20, 3 seeds, margin fitness

```
python train.py --task retina --mvg --mvg-ops xor,and --switch-interval 20 --generations 2000 --n-seeds 3 --fitness margin
```

`--mvg` originally hardcoded the switch cycle to `and`/`or` (the classic
Kashtan-Alon pairing); added a `--mvg-ops` flag (comma-separated, cycles in
order, default `and,or` so existing behaviour is unchanged) so the cycle can
include `xor` — needed to compare against the fixed-goal `retina/xor` run
above. `--mvg` disables early stopping (the target never stops moving), so
every seed runs the full 2000-generation budget.

| seed | reported "best accuracy" | edges used | density | exc(+) / inh(-) |
|---|---|---|---|---|
| 0 | 1.000 | 525/560 | 93.8% | 272 / 253 |
| 1 | 1.000 | 520/560 | 92.9% | 274 / 246 |
| 2 | 1.000 | 521/560 | 93.0% | 242 / 279 |

**The reported "best accuracy" is misleading here — it's driven entirely by
`and`, not `xor`.** The per-seed "best" is the highest single-generation
population-best accuracy seen *on whichever op was active that generation*; it
does not mean both targets were solved. Across all 300 logged xor-phase
generations (3 seeds × 100 each), best-in-population accuracy on xor ranged
from **0.327 to 0.919** and never once reached 1.0. `and`, by contrast, hit
exactly 1.000 repeatedly (generations 750, 1190, 1470, 1670, 1710, 1790, 1950,
1999 across the three seeds).

**xor does not evolve faster under xor/and MVG — if anything it never converges
at all.** Immediately after each switch *into* xor, best accuracy crashes to
0.33-0.44 (below chance), climbs to ~0.7-0.9 over the following ~10
generations, then gets yanked back to `and` before it can consolidate. This
pattern is stable from generation 0 through generation ~2000 — no sign of the
oscillation narrowing or xor plateauing solved, in all 3 seeds. Contrast with
the fixed-goal run above, which solved xor cleanly in 290-456 generations and
stayed there.

**Why xor/and is a much harsher pairing than and/or.** AND and OR agree on 3 of
4 truth-table rows (only differ when exactly one side is active) — a mild
perturbation to switch between. AND and XOR agree on only 1 of 4 (both are 0
only when left=right=0; everywhere else they're exact opposites: XOR=1&AND=0
when exactly one side fires, XOR=0&AND=1 when both fire). With a single shared
CMA-ES mean/covariance and no structural separation between the two
sub-problems, the population can't represent two near-opposite decision
surfaces at once, and a 20-generation window isn't enough to fully readapt
before being switched away again — so it seesaws instead of learning either
target robustly.

**Caveat / likely confound:** this result may say more about "xor and and are
adversarial partners" than about "does switching pressure help modularity
emerge." `xor,or` agrees on 3/4 rows (same agreement fraction as the classic
`and,or` pair — they only disagree when both sides fire) and would be the
fairer "gentle" partner to test the original speed/modularity question without
the adversarial-pairing confound.


## FG vs MVG x constraint, 4 arms x 5 seeds, 5k generations (2026-09-12)

The same 4-group design run on experiment 1 the same night, so the two encodings
can be read against each other. Preregistration, preflight evidence and recovery
notes: `../OVERNIGHT_2026-09-12.md`. Treatment side: `../experiment_1/RESULTS.md`.

```
cd experiments
python run_fgmvg_study.py --experiment 2 --lanes 10     # 20 runs, resumable
python analysis/run_all.py --root experiment_2/runs/fgmvg
```

`retina_ka2005`/AND, **raw** accuracy (`--no-balanced`), margin fitness,
n_hidden=24 (793 free weights), popsize 64, 5,000 generations, `--no-early-stop`,
seeds 0-4. Constrained arm `--synaptic-budget 4 --shrink 0.9`; MVG arm AND<->OR
every 20 generations. All numbers are the **goal-matched** champion (`matched`).

> **Fewer generations than experiment 1 (5,000 vs 10,000), on purpose.** CMA-ES is
> superlinear in dimension and this encoding searches 793 free weights against
> experiment 1's 443, so a generation costs ~5x; it also converges far sooner
> (historically 290-456 generations to solve retina/xor), and 5,000 is ~11x that.
> **Both arms within experiment 2 get the identical budget**, which is what the
> FG-vs-MVG claim needs. Across encodings the budgets are NOT matched — 320k
> evaluations here vs 640k in experiment 1 — so any exp1-vs-exp2 statement has to
> say so. Note the direction: the control gets the *smaller* budget and still
> wins on accuracy, so the accuracy gap is not a budget artifact.

> **Output layout changed 2026-09-12** from flat files in `--out-dir` to one
> directory per seed (`<task>_<arm>[_b<S>s<tau>]_seed<N>/` holding `log.csv`,
> `champions.npz`, `config.json`, `result.json` and the four champion DNAs/PNGs),
> matching experiment 1 exactly. One layout for both encodings means one analysis
> pipeline serves both. Older flat run files under `runs/` are untouched.

### 1. The direct encoding solves the task in every arm

**`acc(AND) = 1.000` in all twenty runs** — constrained and unconstrained, FG and
MVG. There is no accuracy story to tell here and no ceiling to argue about; the
control saturates the task. (Experiment 1, for contrast, gets 0.978 unconstrained
and 0.895 constrained.)

### 2. The budget costs this encoding NOTHING

| arm | acc (AND) | acc (OR) | density |
|---|---|---|---|
| budget FG | 1.000 (5/5) | n/a | 37.1-38.9% |
| budget MVG | 1.000 (5/5) | 0.500 (5/5) | 37.1-38.5% |
| ablation FG | 1.000 (5/5) | n/a | 95.1-95.8% |
| ablation MVG | 1.000 (5/5) | 0.500 (5/5) | 95.7-98.0% |

Perfect accuracy at ~38% density — Kashtan-Alon's own capped-arm density band
(34-38%), with better accuracy than KA's 0.90+-0.03. The constraint removes 60
points of density and zero points of accuracy.

That is a real asymmetry against experiment 1, where the same kind of constraint
cost ~8 accuracy points (0.978 -> 0.895). A 793-parameter direct encoding can
afford to spend its connectivity budget well; a 443-parameter compressed one
cannot. Worth saying plainly: **on this task the genomic bottleneck is a cost,
not a benefit, in accuracy-per-density terms.**

### 3. MVG never holds both goals — and here it is exact

The AND-matched MVG champion scores **exactly 0.500 on OR in all ten MVG seeds**
(experiment 1: 0.42-0.47). Chance. The population is not maintaining a shared
decomposition that serves both goals; it is fully re-specialising every 20
generations. Since this reproduces on a completely different encoding, it is a
property of goal-switching in this setup, not of the compressed genome.

### 4. The goal-matching fix earns its keep here more than anywhere

Three `nobudget_mvg` seeds (1, 2, 4) have **`best`(AND) = 0.500 while
`matched`(AND) = 1.000**. The old best-ever reporting would have announced
"accuracy 1.000" for those seeds — and that 1.000 was scored on OR, by a network
that is at chance on AND. This RESULTS.md already documented the same trap by
hand for the xor/and run ("the reported best accuracy is driven entirely by
`and`"); it is now fixed in code, and `result.json`'s `acc_by_op` makes it
impossible to repeat silently.
### 5. Modularity: the budget makes it, and here MVG edges ahead of FG

`tag=matched`, `n_rand=200` degree-preserving nulls, mean+-sd over 5 seeds.
`n/a` means the metric was undefined in every seed (see the density caveat).

| condition | arm | n | acc (AND) | acc (OR) | density % | LR (primary) | purity (primary) | Q | Q_m |
|---|---|---|---|---|---|---|---|---|---|
| budget (constrained) | FG | 5 | 1.000+-0.000 | n/a | 37.8+-0.7 | 0.102+-0.069 | 0.151+-0.011 | 0.221+-0.019 | 0.050+-0.023 |
| budget (constrained) | MVG | 5 | 1.000+-0.000 | 0.500+-0.000 | 37.8+-0.6 | 0.129+-0.117 | 0.169+-0.039 | 0.218+-0.013 | 0.043+-0.031 |
| no budget (ablation) | FG | 5 | 1.000+-0.000 | n/a | 95.6+-0.3 | 0.190+-1.620 (4/5) | 0.017+-0.006 | 0.070+-0.006 | 0.884+-0.000 (1/5) |
| no budget (ablation) | MVG | 5 | 1.000+-0.000 | 0.500+-0.000 | 97.3+-1.0 | n/a | 0.020+-0.013 | 0.085+-0.007 | n/a |

Left/right-significant seeds (permutation test on the planted split):
`budget_fg` **1/5**, `budget_mvg` **3/5**, `nobudget_fg` 0/5, `nobudget_mvg` 0/5.

Two things to read off this.

**The constraint is doing the work.** Purity is 0.151-0.169 constrained and
0.017-0.020 unconstrained — an order of magnitude, with non-overlapping
spreads. Q is 0.22 vs 0.07-0.09. And in the unconstrained arms the planted-split
score and Q_m are largely *undefined*, not low: at 95-98% density there is no
sparser degree-preserving null to compare against, so the null distribution
collapses and the ratio blows up or returns `nan` (`nobudget_mvg` gives `n/a` on
both). That is the same pattern experiment 1 showed. **Unconstrained runs are
not evidence of low modularity; they are evidence that the question cannot be
asked.** The budget is what makes it askable.

**Within the constrained condition, MVG > FG — weakly, but on every metric that
is defined.** purity 0.169 vs 0.151, LR 0.129 vs 0.102, and significant LR
structure in 3/5 seeds vs 1/5. Q is a tie (0.218 vs 0.221) and Q_m is a tie
within noise. With n=5 and those spreads none of this is significant on its own;
what makes it worth recording is that it is the *first* arm-pair in this project
where the MVG>FG direction is consistent rather than mixed — experiment 1's
constrained pair splits two metrics each way. Do not report it as a
demonstration of the Kashtan-Alon effect. Report it as: at matched density and
matched (saturated) accuracy, goal-switching buys a small, consistent, not
individually-significant increase in left/right structure.

### 6. Cross-encoding comparison: the bottleneck does not buy modularity here

This is the comparison the whole two-experiment design exists for, so it gets
stated plainly. Constrained arms only (the only ones where modularity is
measurable), `matched` champion, and note the densities happen to land close
enough to compare: 43.0+-5.3% (exp 1 FG) / 34.0+-7.6% (exp 1 MVG) against
37.8+-0.7% (both exp 2 arms).

| | exp 1 (compressed, 443 params) | exp 2 (direct, 793 params) |
|---|---|---|
| acc(AND), FG | 0.895+-0.017 | **1.000+-0.000** |
| acc(AND), MVG | 0.853+-0.016 | **1.000+-0.000** |
| density, FG | 43.0+-5.3% | 37.8+-0.7% |
| purity, FG | 0.114+-0.088 | **0.151+-0.011** |
| purity, MVG | 0.066+-0.106 | **0.169+-0.039** |
| Q, FG | 0.169+-0.096 | **0.221+-0.019** |
| LR, MVG | -0.090+-0.744 | **0.129+-0.117** |
| sig LR, MVG | 2/5 | **3/5** |
| Q_m, MVG | **0.587+-0.588** | 0.043+-0.031 |

The direct encoding is at least as modular as the compressed one on purity, Q
and the planted left/right split, **and** an order of magnitude more consistent
(purity sd 0.011 vs 0.088), **and** saturates the task the compressed encoding
cannot solve. The one metric favouring the bottleneck is Q_m — and its 0.587
comes with a +-0.588 spread and two seeds at exactly 0.000, so it is carried by
one or two runs, not by the arm.

**This is a negative result for the genomic-bottleneck hypothesis as stated**
("a compressed DNA->brain encoding encourages modularity"). On this task, at
matched density, compression bought no measurable modularity and cost 10-15
accuracy points.

Caveats that must travel with that sentence:

1. **Evaluation budgets are not matched across encodings** — 640k evaluations in
   experiment 1 (10k gens) vs 320k here (5k gens). The direction is reassuring
   (the control wins on *half* the budget, so the gap is not a budget artifact)
   but it is not a matched comparison, and the *modularity* numbers could move
   either way with more generations.
2. **Different competence regimes.** Experiment 2 is at ceiling; experiment 1 is
   not. A network that has solved the task and one that is still 10 points short
   need not be comparable objects, and "modularity of a partial solution" may
   simply be a different quantity.
3. **Experiment 1's metric variances are enormous** (purity +-0.088 on a mean of
   0.114; LR +-1.322). With n=5 that arm is closer to unmeasured than to
   measured. The honest reading is "no evidence the bottleneck helps", not
   "evidence the bottleneck does not help".
4. Both encodings share the same architecture, task, fitness, popsize and
   constraint mechanism, so the encoding really is the only deliberate
   difference. Parameter count is a *consequence* of the encoding, not a
   separate confound — but it does mean "compression" and "fewer parameters" are
   not separable in this design.

### Figures

Regenerate all of it with `python analysis/run_all.py --root experiment_2/runs/fgmvg`.
Written to `runs/fgmvg/` (gitignored; not copied into `latex_figures/`):

- `progress_fg_vs_mvg.png` — accuracy and density vs generation, 4 arms x 5
  seeds. MVG sampled at reference-goal epoch ends (sampling at even intervals
  aliases against the 20-generation switch and draws a fake sawtooth).
- `switch_window_budget_seed{0,1,2}.png`, `switch_window_nobudget_seed0.png` —
  400 generations generation-by-generation: accuracy on both goals, density and
  the four metrics across ~20 switches. This is the figure that shows the
  antiphase directly.
- `brains_grid_purity.png`, `brains_grid_community.png` — the 20 matched
  champions as weight graphs, coloured by per-neuron purity and by detected
  community.
- `metrics_per_seed.csv`, `metrics_summary.json` — every number above, per seed.

## Open threads

- Run `--mvg --mvg-ops xor,or --switch-interval 20 --generations 2000` — the
  gentler xor pairing (3/4 truth-table agreement, like the classic and/or
  pair), to separate "switching pressure helps" from "xor/and are adversarial."
- ~~Once a modularity metric exists, score these saved DNAs directly rather
  than inferring from density.~~ **DONE 2026-09-12** — `shared_brain_metrics.py`
  + `analysis/run_all.py`; the four metrics are in the 4-arm study above.
- Match the evaluation budget across encodings (rerun this study at 10k
  generations, or experiment 1 at 5k) so the cross-encoding modularity claim in
  section 6 rests on a matched comparison.
