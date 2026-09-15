# Experiment 2 — results

Direct encoding: one gene per allowed synapse plus one bias per non-input neuron
(`../shared_direct_model.py`), searched by CMA-ES. The control that experiment 1's
compressed encoding is measured against. Runs are written to `runs/` (not committed).

## 1. FG vs MVG × synaptic budget (4 arms × 5 seeds)

Same design as `../experiment_1/RESULTS.md` section 1: `retina_ka2005`, raw accuracy,
margin fitness, 8/24/1 neurons (768 allowed edges, 793 genes), popsize 64, AND ↔ OR
every 20 generations for MVG, goal-matched champion, seeds 0–4. Differences: **5,000
generations** and budget `S = 4, τ = 0.9`.

```bash
cd experiments
python run_fgmvg_study.py --experiment 2 --lanes 10
python analysis/run_all.py --root experiment_2/runs/fgmvg
```

5,000 generations because a generation costs ~5× experiment 1's (793 vs 443 genes)
and this encoding converges early (290–456 generations on `retina/xor`, section 4).
The two arms within this experiment are matched; the budgets across encodings are
not (320k vs 640k evaluations per seed).

| condition | arm | acc (AND) | density % | `lr_r` | Q | seeds with `lr_r` above null |
|---|---|---|---|---|---|---|
| budget | FG | 1.000 ± 0.000 | 37.8 ± 0.7 | +0.194 ± 0.031 | 0.221 ± 0.019 | 1/5 |
| budget | MVG | 1.000 ± 0.000 | 37.8 ± 0.6 | +0.210 ± 0.036 | 0.218 ± 0.013 | 3/5 |
| no budget | FG | 1.000 ± 0.000 | 95.6 ± 0.3 | −0.012 ± 0.006 | 0.070 ± 0.006 | 0/5 |
| no budget | MVG | 1.000 ± 0.000 | 97.3 ± 1.0 | −0.016 ± 0.006 | 0.085 ± 0.007 | 0/5 |

1. **The direct encoding solves the task in all 20 runs.** The AND-matched MVG
   champions score exactly 0.500 on OR, i.e. acc(AND) + acc(OR) = 1.5, the maximum any
   network without a goal cue can reach.
2. **The budget costs this encoding nothing.** Accuracy stays 1.000 at ~38% density
   (Kashtan & Alon's capped networks: 34–38%). In experiment 1 the same kind of
   constraint cost 8 points.
3. **The budget produces the modularity here too.** Q 0.22 vs 0.07–0.09, and `lr_r`
   positive in all 10 constrained runs (+0.168 to +0.259) and negative in all 10
   unconstrained ones (−0.021 to −0.002).
4. **MVG does not beat FG.** At identical density (37.8% vs 37.8%; density-`lr_r`
   correlation −0.009) the `lr_r` gap is half an SD (exact one-sided Mann-Whitney
   U = 16, p = 0.274) and Q points the other way. The only pro-MVG number is 3/5 vs
   1/5 significant seeds, uncorrected for the metrics examined.
5. **MVG costs time, not outcome.** In the budget arms FG reaches 1.000 by about
   generation 400 and MVG by about 1,300; afterwards the arms are indistinguishable.

**Figures:** `latex_figures/experiment_2_fgmvg/` (budget arms only; the unconstrained
arms are 100% dense in truth and score 1.000 everywhere).

## 2. Re-adaptation after a switch — this encoding slows down

Full analysis in `../experiment_1/RESULTS.md` section 2 (`analysis/recovery_stats.py`).
For the population mean, generations to 90% of the post-switch climb go from
5.90 ± 0.52 in [100, 300] to 8.36 ± 1.14 in [1000, 1200], slower in 5/5 seeds, while
the compressed encoding speeds up in 5/5.

## 3. Cross-encoding comparison: the bottleneck does not buy modularity

Constrained arms, goal-matched champions:

| | exp 1 (compressed, 443 genes) | exp 2 (direct, 793 genes) |
|---|---|---|
| acc (AND), FG / MVG | 0.895 / 0.853 | **1.000 / 1.000** |
| density, FG / MVG | 43.0% / 34.0% | 37.8% / 37.8% |
| `lr_r`, FG / MVG | +0.094 ± 0.068 / +0.199 ± 0.217 | +0.194 ± 0.031 / +0.210 ± 0.036 |
| Q, FG / MVG | 0.169 ± 0.096 / 0.208 ± 0.154 | 0.221 ± 0.019 / 0.218 ± 0.013 |

The direct encoding is at least as modular on both metrics, far more consistent
across seeds, and solves the task. **On this task, compression bought no measurable
modularity and cost 10–15 accuracy points.** Caveats: evaluation budgets are not
matched (the control gets half, and still wins on accuracy); the two encodings sit
in different competence regimes; experiment 1's variances are large at n = 5, so the
fair reading is "no evidence the bottleneck helps"; densities are close but not
matched, and in experiment 1 sparser reads as more modular; "compressed" and "fewer
parameters" cannot be separated in this design.

**Kashtan & Alon's Q_m would give the opposite answer, wrongly.** Averaged over the
10 constrained runs per encoding (`metrics_per_seed.csv`):

| | exp 1 | exp 2 | ratio |
|---|---|---|---|
| Q_real | 0.133 | 0.117 | 1.14 |
| Q_rand (degree-preserving null) | 0.074 | 0.107 | 0.69 |
| Q_max | 0.187 | 0.324 | 0.58 |
| Q_m = (Q_real − Q_rand) / (Q_max − Q_rand) | **0.495** | **0.046** | **10.7** |

Raw modularity differs by 14% while Q_m differs by 10.7×, mostly through Q_rand. A
degree-preserving null is not an encoding-preserving null: experiment 1's genome
produces a K × K type graph copied out by clone counts, so clones share identical rows,
the degree sequence is clumped, and rewiring it lowers Q. Q_m therefore rewards the
encoding with clumpier degrees, not the more modular one. For a cross-encoding null,
random genomes drawn through the same encoding are the right comparison.

## 4. Earlier runs (stand-in `retina` task, no budget)

**`retina/xor`, fixed goal, 20 hidden, 5 seeds** (`train.py --task retina --operation
xor --n-seeds 5 --fitness margin`): all seeds reach 1.000 in 290–456 generations, at
92–94% density with balanced excitatory and inhibitory weights.

**xor ↔ and MVG, 3 seeds, 2,000 generations** (`--mvg --mvg-ops xor,and
--switch-interval 20`): XOR never reaches 1.0 (best-in-population 0.33–0.92 across
all XOR epochs) while AND repeatedly does. After each switch into XOR accuracy falls
to 0.33–0.44 and has climbed to only 0.7–0.9 when the goal switches back. XOR and AND
agree on only a quarter of the (L, R) cells, so this pairing is much harsher than
AND/OR.

**AND FG vs AND ↔ OR MVG, 5 seeds each, 2,000 generations** (`--task retina
--operation and` and `--mvg --mvg-ops and,or`, both `--no-early-stop --n-hidden 20`):
FG solves in 200–280 generations; under MVG both goals repeatedly hit 1.000 with no
narrowing of the oscillation over the run. Both arms end dense (FG 94–96%, MVG
96–98%): without a constraint, goal switching does not make this encoding sparser.
