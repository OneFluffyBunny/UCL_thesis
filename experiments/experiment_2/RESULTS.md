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

**MVG costs time, not outcome** (budget arm, progress figure; added 2026-09-14).
FG reaches 1.000 by about generation 400, MVG by about 1300. After that the arms
are indistinguishable: 1.000 in all 10 budget runs, density ~38% in both, `lr_r`
+0.19 vs +0.21, no visible left/right separation in the brains grid. In
experiment 1 MVG also costs final accuracy (0.853 vs 0.895): MVG never helps in
either encoding. (MVG's apparent head start at the left edge of the progress plot
is a sampling offset: its first point is generation 19, FG's generation 0.)

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

### 3. RETRACTED: "MVG never holds both goals" - this arm is Pareto-OPTIMAL

> **RETRACTED 2026-09-12.** The original text read: *"The AND-matched MVG
> champion scores exactly 0.500 on OR in all ten MVG seeds. Chance. The
> population is not maintaining a shared decomposition that serves both goals."*
> It was offered as the sharpest form of a negative result about MVG. **It is the
> exact opposite.** Full derivation in `../experiment_1/RESULTS.md` section 3;
> the short version:
>
> `retina_ka2005` has 2^8 = 256 patterns, 64/64/64/64 over the four
> (left-object, right-object) cases. AND is true on 64, OR on 192; the goals
> **agree on 128 patterns and disagree on 128**. The network is handed only the 8
> retina bits - no goal cue - so a single function is scored against both goals
> and each disagreeing pattern can satisfy at most one. Hence, for ANY
> context-free network:
>
>     acc(AND) + acc(OR) <= 1.500
>
> A *perfect* AND solver scores exactly **0.500** on OR. These ten seeds score
> 1.000 on AND and 0.500 on OR, i.e. **1.500 exactly - the maximum attainable on
> this goal pair, in all ten runs.** They are Pareto-optimal. The number that was
> reported as the study's most damning finding is in fact the best possible
> score, and "chance" was the wrong frame: 0.500 here is forced, not a failure.
> For contrast experiment 1 reaches only 1.276 / 1.334, purely because its AND
> accuracy falls short of 1.000.
>
> **What survives:** the per-generation *swapping* is real, and the population
> does re-specialise each epoch rather than parking on a compromise. But that is
> a description of the dynamics, not evidence against Kashtan-Alon - holding both
> goals simultaneously is arithmetically unavailable to this architecture, so no
> measurement of simultaneous accuracy can test their claim. The real test is
> **re-adaptation speed after a switch**, measurable from `champions.npz`. **DONE 2026-09-13**:
> see "Re-adaptation speed" below.

The AND-matched MVG champion scores **exactly 0.500 on OR in all ten MVG seeds**,
and **1.000 on AND** - summing to the 1.500 ceiling. Experiment 1's MVG arms sum
to 1.276 (constrained) and 1.334 (ablation). The direct encoding is not worse at
holding both goals than the compressed one; it is strictly better, because it
solves the active goal outright.

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

> **METRICS REVISED 2026-09-12** (same runs, same champions, re-read). Primary
> modularity numbers are now **`lr_r`** (Newman's discrete assortativity at the
> planted split) and **raw Newman Q**; the `lr` ratio score and `purity` are
> demoted to continuity columns and `Q_m` is reference-only. Reasons: the REVISED
> block in `../shared_brain_metrics.py`, and section 6 below for the
> cross-encoding problem specifically. **No number changed - only which column
> leads, and one finding below is downgraded as a result.**

| condition | arm | n | acc (AND) | acc (OR) | density % | `lr_r` (PRIMARY) | Q (PRIMARY) | purity (descr.) | Q_m (ref) | `lr` ratio (do not report) |
|---|---|---|---|---|---|---|---|---|---|---|
| budget (constrained) | FG | 5 | 1.000+-0.000 | n/a | 37.8+-0.7 | +0.194+-0.031 | **0.221+-0.019** | 0.151+-0.011 | 0.050+-0.023 | 0.102+-0.069 |
| budget (constrained) | MVG | 5 | 1.000+-0.000 | 0.500+-0.000 | 37.8+-0.6 | **+0.210+-0.036** | 0.218+-0.013 | 0.169+-0.039 | 0.043+-0.031 | 0.129+-0.117 |
| no budget (ablation) | FG | 5 | 1.000+-0.000 | n/a | 95.6+-0.3 | -0.012+-0.006 | 0.070+-0.006 | 0.017+-0.006 | undef (1/5) | undef (4/5) |
| no budget (ablation) | MVG | 5 | 1.000+-0.000 | 0.500+-0.000 | 97.3+-1.0 | -0.016+-0.006 | 0.085+-0.007 | 0.020+-0.013 | undef (0/5) | undef (0/5) |

Left/right-significant seeds (permutation test on the planted split, unaffected
by the `lr`-vs-`lr_r` change since both derive from the same `q`):
`budget_fg` **1/5**, `budget_mvg` **3/5**, `nobudget_fg` 0/5, `nobudget_mvg` 0/5.

**`lr_r` is defined in all 20 runs and flips sign with the constraint** - all ten
constrained runs positive (+0.168 to +0.259), all ten unconstrained negative
(-0.021 to -0.002). Where the `lr` ratio returned `n/a` for a whole arm, `lr_r`
returns a small negative number, which is the honest reading: at 95-98% density
the planted split is marginally *anti*-assortative, i.e. there is no structure
there. That is a better failure mode than `nan`, but note it is not new
information - see the caveat in section 5 below.

Two things to read off this.

**The constraint is doing the work.** Q is 0.22 constrained vs 0.07-0.09
unconstrained, and `lr_r` flips sign (+0.19/+0.21 vs -0.012/-0.016) with no
overlap between conditions in any of the 20 runs. Purity is 0.151-0.169 vs
0.017-0.020, an order of magnitude with non-overlapping spreads - **but purity is
no longer offered as evidence** (it is below KA's encoding-aware null in all 40
runs of this study and mostly tracks density; see
`../shared_brain_metrics.py`). The conclusion stands on Q and `lr_r`. And in the
unconstrained arms the `lr` ratio score and Q_m are largely *undefined*, not low: at 95-98% density there is no
sparser degree-preserving null to compare against, so the null distribution
collapses and the ratio blows up or returns `nan` (`nobudget_mvg` gives `n/a` on
both). That is the same pattern experiment 1 showed. **Unconstrained runs are
not evidence of low modularity; they are evidence that the question cannot be
asked.** The budget is what makes it askable.

**DOWNGRADED 2026-09-12: within the constrained condition, MVG > FG is not
supported.** The original claim here was "MVG > FG, weakly but on every metric
that is defined", called the *first consistent MVG>FG signal in the project*.
Re-reading it on the primary metrics, what is left is: `lr_r` +0.210 vs +0.194,
Q a tie the other way (0.218 vs 0.221), purity 0.169 vs 0.151 (and purity is no
longer evidence), significant seeds 3/5 vs 1/5. The `lr_r` gap is **0.016 on an
SD of 0.031-0.036 - half a standard deviation** - and an exact one-sided
Mann-Whitney U test gives **U = 16.0, p = 0.274 (n = 5 v 5)**. Not significant,
and the two primary metrics point in opposite directions.

This also removes the reading that experiment 1's larger 2.1x gap corroborated
it. Experiment 1's constrained MVG arm runs **9.0 density points sparser** than
its FG arm (34.0% vs 43.0%) and within those ten runs `lr_r` correlates with
density at **r = -0.471**, so its apparent MVG advantage is confounded. Here the
budget produced *matched* densities (37.8% vs 37.8%, density-`lr_r` correlation
-0.009) and the gap collapses to half an SD. **The density-matched encoding is
the one that shows no effect, which is the more trustworthy of the two.**

The honest summary: **this study gives no evidence that goal-switching increases
modularity, in either encoding.** The only surviving pro-MVG fact is the
significant-seed count (3/5 vs 1/5), which is 4 seeds out of 10 and carries no
correction for having looked at five metrics.

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
| **`lr_r`, FG** (PRIMARY) | +0.094+-0.068 | **+0.194+-0.031** |
| **`lr_r`, MVG** (PRIMARY) | +0.199+-0.217 | **+0.210+-0.036** |
| **Q, FG** (PRIMARY) | 0.169+-0.096 | **0.221+-0.019** |
| **Q, MVG** (PRIMARY) | 0.208+-0.154 | 0.218+-0.013 |
| sig LR, MVG | 2/5 | **3/5** |
| purity, FG (descr.) | 0.114+-0.088 | 0.151+-0.011 |
| purity, MVG (descr.) | 0.066+-0.106 | 0.169+-0.039 |
| Q_m, MVG (ref only) | 0.587+-0.588 | 0.043+-0.031 |
| `lr` ratio, MVG (do not report) | -0.090+-0.744 | 0.129+-0.117 |

The direct encoding is at least as modular as the compressed one on both primary
metrics, **and** far more consistent (`lr_r` sd 0.031-0.036 vs 0.068-0.217; Q sd
0.013-0.019 vs 0.096-0.154), **and** saturates the task the compressed encoding
cannot solve. On raw Q the two are close to indistinguishable in the MVG arms
(0.218 vs 0.208); the clean gap is in the FG arms and in the consistency.

> **CAVEAT ADDED 2026-09-12 - the paper's own metric would have INVERTED this
> verdict, and the reason matters.** Q_m is the one column that favours the
> bottleneck (0.587 vs 0.043, and 0.495 vs 0.046 averaged over both constrained
> arms - a **10.7x** gap). Taken at face value that is a headline result for the
> thesis. It is an artifact. Averaged over the ten constrained runs per encoding:
>
> | | exp 1 | exp 2 | ratio |
> |---|---|---|---|
> | `q_real` (raw Q) | 0.133 | 0.117 | **1.14x** |
> | `q_rand` (degree-preserving null) | 0.074 | 0.107 | 0.69x |
> | `q_max` (planted ceiling) | 0.187 | 0.324 | 0.58x |
> | `q_m = (real-rand)/(max-rand)` | **0.495** | **0.046** | **10.7x** |
>
> Two encodings whose RAW modularity differs by 14% get Q_m scores differing by
> 10.7x. The gap decomposes as ~6x from the numerator (experiment 1's `q_rand` is
> much lower) and ~1.9x from the compressed denominator. The culprit is mainly
> **`q_rand`** - which is the term this repo's memory note certifies as
> KA-faithful. The problem is not unfaithfulness: **a degree-preserving null is
> not an ENCODING-preserving null.** Experiment 1's genome emits a K x K cell-type
> graph blown up by clone counts (verified: `w[i,j]` depends only on
> `(type_i, type_j)` in all 20 runs), so clones share identical rows and the
> degree sequence is clumped; rewiring a clumped sequence scores lower, inflating
> the numerator. Q_m therefore rewards whichever encoding has the clumpier
> degrees, independently of how modular it actually is.
>
> Q_m fails locally too: it returns **exactly 1.000** for both the least modular
> (`q` = 0.030) and the most modular (`q` = 0.415) constrained run in experiment
> 1, because the `q_max` estimator cannot beat the observed graph and
> `normalized_qm` then saturates. And it is `nan` in 8/20 experiment-1 runs and
> 14/20 here.
>
> **Consequence for this section:** the negative verdict below is correct, but it
> was correct by luck - it rests on having led with raw Q and `lr_r` rather than
> with KA's published metric. Anyone reproducing this comparison with Q_m will
> get the opposite answer. Where a null is needed, use **random genomes through
> the same encoding at the same config** (KA's own second null, see memory note
> `reference_ka_modularity_metric`): under that null raw Q survives in 9/10
> constrained runs and `lr_r` in 6/10, while purity fails in all 40.

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
3. **Experiment 1's metric variances are enormous** (`lr_r` +-0.217 on a mean of
   0.199; Q +-0.154 on 0.208). With n=5 that arm is closer to unmeasured than to
   measured. The honest reading is "no evidence the bottleneck helps", not
   "evidence the bottleneck does not help".
5. **The densities are close but not matched, and that slack is not free.**
   Experiment 1's constrained arms span 34.0-43.0% against 37.8% here, and within
   experiment 1's ten constrained runs `lr_r` correlates with density at
   r = -0.471 (sparser reads as more modular). Experiment 1's MVG arm is the
   sparsest of the four, which flatters it on `lr_r`. A density-matched rescore is
   the fix and is cheap - the DNAs are saved; see Open threads in
   `../experiment_1/RESULTS.md`.
6. **Purity and the `lr` ratio are withdrawn, not merely demoted**, so any
   earlier version of this comparison that leaned on them (including the
   "order of magnitude more consistent on purity" line, now restated on `lr_r`
   and Q) should be re-read against the table above rather than quoted.
4. Both encodings share the same architecture, task, fitness, popsize and
   constraint mechanism, so the encoding really is the only deliberate
   difference. Parameter count is a *consequence* of the encoding, not a
   separate confound — but it does mean "compression" and "fewer parameters" are
   not separable in this design.

### Figures

Budget arm promoted to `latex_figures/experiment_2_fgmvg/` (2026-09-14; provenance,
commands and caveats in its `README.md`): `brains_grid_leftright_budget.png`,
`progress_fg_vs_mvg_budget.png`, `switch_window_budget_seed0.png` (dense replay).
Unconstrained-arm figures deliberately not promoted: true density is 100%, so
they show an unanswerable question, and every seed scores 1.000. Everything else
stays in `runs/fgmvg/` (`python analysis/run_all.py --root
experiment_2/runs/fgmvg`); `metrics_per_seed.csv` and `metrics_summary.json` hold
every number above, per seed.

## Re-adaptation speed: the direct encoding SLOWS DOWN (2026-09-13, updated 2026-09-14)

Cross-encoding result; the full analysis is in `../experiment_1/RESULTS.md`
("Re-adaptation speed after a goal switch"). Recorded here because the direct
encoding is one of its two arms.

Generations for the population mean to cover 90% of its post-switch climb,
budgeted MVG, 5 seeds, from `analysis/dense_replay.py`:

| | [100,300] | [1000,1200] | late - early |
|---|---|---|---|
| direct (here) | 5.90 +- 0.52 | 8.36 +- 1.14 | **+2.46, 0/5 seeds faster** |
| compressed (exp_1) | 6.30 +- 1.25 | 4.20 +- 0.24 | **-2.10, 5/5 seeds faster** |

Per-seed deltas +3.6, +3.8, +2.0, +1.7, +1.2 here against -2.4, -1.7, -2.1,
-0.5, -3.8 there: no overlap, exact Mann-Whitney p ~ 0.008 at n = 5 vs 5.

~~Part of the slowdown here is a ceiling effect~~: **superseded 2026-09-14.**
Peak-free measures (full table in `../experiment_1/RESULTS.md`) all show this arm
slowing in 5/5 seeds: gain in the first 3 gens after a switch 0.237 -> 0.152,
gens to climb +0.20 3.0 -> 4.2, gens to reach 0.75 4.4 -> 5.0. Not σ collapse
(σ late/early 1.00-1.21).

Late troughs sit at **0.500 ± 0.001** = `½ + ½(a − b)` with a = b: the population
is equally good on the L = R and L ≠ R patterns of the old goal. The compressed
arm sits at 0.438, better on the half the switch flips. Reading in the exp_1
section.


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
