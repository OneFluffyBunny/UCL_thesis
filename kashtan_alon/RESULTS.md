# kashtan_alon — RESULTS (lab notebook)

The run record for the Kashtan–Alon retina MVG→modularity reproduction. Reference
docs (`README.md`, `PAPER_SPEC.md`) describe *what* the setup is; this file records
*what happened* when we ran it. Newest runs first.

**The claim under test (KA 2005):** under Modularly-Varying Goals (G_AND ↔ G_OR
every 20 gens) the normalized modularity **Q_m** rises and *stays high* (**0.35 ±
0.02**); under a Fixed Goal it stays low (**0.15 ± 0.02**). Same network, same
search, only the goal schedule differs.

> ⚠️ **Known fidelity gap, found 2026-09-09, applies to every run below:** the paper's
> retina fitness includes *"a penalty of 0.01 ... for every additional [effective]
> neuron above [13]"* (`PAPER_SPEC.md` §5) — our `model.py fitness()` has no
> complexity term at all, pure accuracy. This is a real, if small, parsimony
> pressure the paper's own runs had that ours don't. It's a plausible partial
> explanation for the sparsity/density questions raised while discussing Run 6 — the
> real experiment had *some* active pressure against unnecessary structure that our
> reproduction lacks entirely, on top of the mutation-symmetry argument already in
> Run 6's write-up. Not yet added to the code or re-run.

> 🐛 **REPORTING BUG found 2026-09-10 — every MVG "accuracy"/"best fit" number in
> this file is invalid; the modularity numbers are NOT affected. Fixed in code the
> same day; no re-training required.**
>
> **What was wrong.** `train.py` tracked `best_fit`/`best_op` as an all-time maximum
> over the whole run, against *whichever goal was live in that generation*. Under MVG
> the goal alternates AND↔OR, so this maximises across **two different tasks** and is
> not an accuracy at all. Worse, it is systematically biased: OR is true on 192/256
> patterns (75%) vs AND's 64/256, so OR is far easier to nearly-ace — and
> `generations=25000 / switch_interval=20` = 1,250 blocks means the final block is
> always odd-indexed = **OR**, for every seed. Result: **all 5 MVG runs report
> `best_op=or`, none `and`** — MVG's headline accuracy was always measured on the easy
> goal while FG's was always measured on the hard one. The "MVG 0.975 vs FG 0.904"
> comparison was therefore meaningless as stated.
>
> **What is NOT affected.** `Q`, `Q_m`, `circuit_purity` and `left_right_q`/`r` are all
> computed on `final_indiv` — the *final generation's* champion (`train.py`, the
> `_save_best_npz(npz_path, final_indiv, cfg)` call), **not** the peak-fitness genome.
> The per-generation CSVs are also correct: they log the true per-generation champion
> against the live op (they sawtooth, with an `op` column). So every modularity table
> in this file stands, as does `runs_purity/fg_vs_mvg_purity.png`.
>
> **Follow-up sweep, same day — the modularity tables survive a stricter test.** The
> paragraph above only rules out peak-vs-final. It does not address the fact that the
> final-generation champion *is an OR brain* for every MVG seed, so the modularity
> tables put MVG-on-OR beside FG-on-AND. Measured directly
> (`scratch_audit_final_goal.py`, no re-training — the archived per-generation
> champions in `<run>_brains.npz` provide the AND-epoch brains):
>
> | MVG, 5 seeds | final (OR) champion | last AND-epoch champion |
> |---|---:|---:|
> | mean Q_m | +0.259 | **+0.311** |
> | mean circuit purity | 0.929 | 0.933 |
>
> FG is identical either way (every generation is AND). Time-matched — every archived
> champion in the last 2,000 generations, split by the goal live at the time — AND and
> OR purity agree to within 0.01 in all five seeds (e.g. seed0 0.8427 vs 0.8423). So
> the goal of the sampled champion does not move the structural metrics, and where it
> moves Q_m at all it moves it **against** MVG: the tables understate the effect.
>
> **Every other site that mixes goals, audited.** ✅ = verified unaffected, ⚠️ = fixed
> or flagged 2026-09-10.
>
> | Site | Verdict |
> |---|---|
> | per-generation CSVs | ✅ carry an `op` column; correct as logged |
> | `analysis/fg_mvg_purity.py` accuracy panel | ✅ the two goals are equally hard (best constant output 0.750 on **both**; one-eye shortcut 0.750 on both) and MVG's own AND-phase vs OR-phase rows differ by <0.01 in 4/5 seeds |
> | `analysis/fg_mvg_purity.py` purity panel | ✅ see the time-matched table above |
> | `analysis/switch_window.py`, `analysis/dense_replay.py` | ✅ per-generation, goal shaded |
> | `analysis/stage_sheet.py` | ✅ already names the goal in every caption |
> | `analysis/paper_grid.py` | ⚠️ rewritten to draw the **last AND-epoch champion** (gen 24,970) for both arms, with Q/Q_m/r/purity recomputed for the brain drawn |
> | `run_paper.py`, `run_ablation_no_fanin.py` | ⚠️ the fitness column printed MVG-on-OR beside FG-on-AND; now names the goal and points at `acc_by_op` |
> | `highlight_modules.py::_sheet` | ⚠️ still draws final (OR) champions; cost measured above and documented in the docstring |
> | `visualize.py` net title via `train.py` | ⚠️ "final fit X" now reads "final fit X on OR/AND" |
> | `experiments/experiment_2/train.py` | ⚠️ **same bug class, not yet fixed**: `best` is an explicit best-EVER across goal switches under `--mvg`, `final` is scored on the last generation's goal |
> | `experiments/experiment_4/analysis/fg_mvg_quadrant.py` | ⚠️ **same bug class, not yet fixed**: "acc end" and the legend's "final" read the last logged generation, whose goal differs by arm |
> | `experiments/experiment_4/train.py`, `experiment_5/train.py` | ✅ fixed earlier the same day (per-goal `best_by_goal` / reset at switch) |
>
> **The corrected accuracy result — the direction survives.** Re-derived from the
> existing CSVs, per-goal, over the last 5,000 generations (no re-training):
>
> | arm | on AND | on OR |
> |---|---:|---:|
> | Run 5 FG, capped (5 seeds) | **0.904** | n/a |
> | Run 5 MVG, capped (5 seeds) | **0.952** | 0.965 |
> | Run 6 FG, no fan-in (5 seeds) | **0.970** | n/a |
> | Run 6 MVG, no fan-in (5 seeds) | **0.998** | 0.996 |
>
> MVG beats FG *on AND, the matched task*, in both run sets, and is equally good on
> OR — i.e. KA's
> evolvability claim, correctly measured. Note a single saved MVG champion scores
> ~0.50 on AND: that is not failure, it is an OR-phase snapshot, and the population
> re-solves AND within a few generations of each switch.
>
> **The fix (2026-09-10).** `best_fit`/`best_op` renamed to `peak_fit_any_op`/`peak_op`
> and documented as not-an-accuracy; `final_indiv` no longer initialised from the peak
> genome; mid-run checkpoints now save the current champion so the npz means one thing
> throughout; and **`result.json` now carries `acc_by_op`** — the saved champion scored
> against *every* goal the run could face, which is the only FG-comparable figure and
> makes this class of mistake impossible to repeat silently. Readers updated:
> `run_paper.py`, `run_ablation_no_fanin.py`, `analysis/fg_mvg_purity.py`.
> **No re-training needed** — the GA search is untouched, so every saved genome is
> exactly what the fixed code produces; all corrections are re-analysis of existing
> artifacts. Existing `result.json` files predate `acc_by_op` and still carry the old
> `best_fit` key; recompute per-goal accuracy from the saved `_best.npz` instead.

---

## Run 8 — all four groups re-scored goal-matched, n=5 (2026-09-11) → **the cap does a lot of the work; Q_m stops resolving once the cap is gone**

No new training. Every number below is the **last champion archived during an AND
epoch** — generation 24,999 (FG) / 24,970 (MVG) — scored on **AND**, i.e. exactly
the brain each panel of `paper_10runs_grid*.png` draws. This supersedes Run 7's
table, which read `result.json`, i.e. the *final-generation* champion: an OR-phase
brain for every MVG seed. Structural metrics are goal-blind, but the brain itself
is not the same brain, so Run 7's MVG rows described OR-tuned structures beside
FG's AND-tuned ones.

Reproduce: `four_group_table.py` logic is `analysis/paper_grid.py`'s
`last_on_goal()` + `modularity.normalized_qm(n_rand=1000, seed=<seed>)` +
`qmetrics.circuit_purity` / `left_right_q(n_rand=200, seed=0)`.

### The four groups (mean ± SD over seeds 0–4)

| condition | arm | accuracy (AND) | Q | Q_m | r | purity | edges | density |
|---|---|---|---|---|---|---|---|---|
| capped (paper) | FG | 0.90 ± 0.03 | 0.38 ± 0.04 | +0.02 ± 0.14 | +0.60 ± 0.14 | 0.56 ± 0.13 | 40 | 38% |
| capped (paper) | **MVG** | 0.97 ± 0.03 | **0.49** ± 0.03 | **+0.31** ± 0.09 | **+0.94** ± 0.09 | **0.93** ± 0.09 | 36 | 34% |
| no fan-in cap | FG | 0.97 ± 0.02 | 0.22 ± 0.03 | −0.07 ± 0.10 | +0.30 ± 0.07 | 0.31 ± 0.11 | 69 | 65% |
| no fan-in cap | MVG | **1.00** ± 0.00 | 0.29 ± 0.04 | +0.02 ± 0.03 | +0.51 ± 0.08 | 0.56 ± 0.11 | 55 | 52% |

Three readings:

1. **The cap is not a performance constraint, it is a modularity constraint.**
   Removing it makes the task *easier* — uncapped MVG is perfect on all five
   seeds, uncapped FG 0.97 vs capped FG's 0.90. So "the ablated nets are less
   modular because they are worse solutions" is not available as an explanation.
2. **MVG still beats FG without the cap, but by less.** The MVG−FG gap shrinks on
   every metric: Q_m 0.28 → 0.09, r 0.34 → 0.21, purity 0.37 → 0.25. Uncapped MVG
   (purity 0.56) lands where *capped FG* sits (0.56) — removing the cap costs MVG
   roughly as much modularity as removing MVG itself did. This **revises Run 6's
   headline** ("the MVG−FG gap barely moves"), which was n=3 and read
   final-generation champions.
3. **MVG is not sufficient; scarcity is the other ingredient.** Capped-FG and
   uncapped-MVG reach the same purity from opposite directions, and only
   capped-MVG is qualitatively different. That is a 2×2 interaction, not a main
   effect of MVG.

### Why Q_m stops discriminating once the cap is removed

| condition | arm | Q_real | Q_rand | Q_max | Q_max − Q_rand | Q_real − Q_rand | Q_m |
|---|---|---|---|---|---|---|---|
| capped | FG | 0.380 | 0.374 | 0.607 | 0.233 | +0.007 | +0.03 |
| capped | MVG | 0.486 | 0.413 | 0.653 | 0.240 | **+0.073** | +0.31 |
| no cap | FG | 0.216 | 0.229 | 0.417 | 0.188 | −0.013 | −0.07 |
| no cap | MVG | 0.287 | 0.285 | 0.478 | 0.194 | **+0.002** | +0.02 |

In the ablation `Q_real` and `Q_rand` collapse *together* (0.287 vs 0.285). The
numerator is +0.002 — smaller than the greedy partitioner's own noise, which is
why the SD across seeds (±0.03) spans zero. Three causes:

- **Density removes Q's headroom.** At 52–65% density almost every possible edge
  exists, so every partition has many crossing edges and everything scores low:
  `Q_max` itself falls 0.65 → 0.48.
- **The null is handed the answer through the degree sequence.** `Q_m`'s null
  holds degrees fixed and rewires. Under the cap degrees are near-uniform (~3
  everywhere), so degrees encode nothing and the modularity lives in *which*
  pairs are wired — rewiring destroys it. Uncapped, degrees become heterogeneous
  and dense, and "a random graph with these degrees" already looks clustered for
  free.
- **Q is label-blind; purity and r are not.** The uncapped MVG nets genuinely are
  more side-segregated than the uncapped FG nets (purity 0.56 vs 0.31, r +0.51 vs
  +0.30, non-overlapping ±1 SD) — Q's greedy partition at 52% density simply
  cannot resolve it.

**So: modularity really did fall (purity 0.93 → 0.56) AND Q_m separately lost its
resolution.** `Q_m` is calibrated for the sparse regime KA's own runs occupy
(34–38%); at 52–65% it becomes a ratio of two converged quantities. Report purity
and `r` as primary for the ablation, `Q_m` with this caveat. The degradation
tracking density is itself a finding — it is a caution for every other experiment
in this repo that plans to lean on `Q_m`.

### Three claims checked against the logs (`--` = measured, not asserted)

1. **Removing the cap finds solutions much faster — confirmed, and it is not
   close.** First generation at which the champion reaches a given accuracy
   *during an AND epoch*, per seed:

   | threshold | capped FG | capped MVG | no-cap FG | no-cap MVG |
   |---|---|---|---|---|
   | 0.90 | 2/5 seeds ever (340, 17030) | median **890** | median **60** | median **90** |
   | 0.95 | 1/5 ever (2570) | 4/5, median ~6900 | 4/5, median ~110 | median **170** |
   | 0.99 | never | 2/5 (2250, 6890) | 1/5 (440) | median **290** |
   | 1.00 | never | 2/5 (6890, 23930) | never | **all 5**, 240–400 |

   Uncapped MVG reaches a *perfect* score on every seed inside 400 generations;
   capped MVG manages it twice in 25,000 and capped FG never gets past 0.95.
   That is ~10× on the easy thresholds and a difference in kind on the hard ones.

2. **MVG runs sparser than FG in both conditions — but "parsimony" is the wrong
   word; nothing in the fitness prices an edge.** Mean density (% of the 107
   possible feedforward edges):

   | group | gen 0 | 100 | 500 | 2000 | 10000 | 24990 |
   |---|---|---|---|---|---|---|
   | capped FG | 27.4 | 28.9 | 34.3 | 37.0 | 37.4 | 37.5 |
   | capped MVG | 27.4 | 28.5 | 31.7 | 34.3 | 35.1 | 33.8 |
   | no-cap FG | 50.0 | 53.8 | 63.0 | 64.3 | 63.6 | **64.3** |
   | no-cap MVG | 50.0 | 51.7 | 50.9 | 49.4 | 51.7 | **53.4** |

   The mechanism is not MVG *removing* edges — it is FG *adding* them while MVG
   does not. Both ablation arms start at exactly 50.0% (the init formula
   `k = round(0.5 × cap)` is anchored to the cap, so the *capped-vs-uncapped*
   density comparison is confounded, per Run 6 note 5 — but the FG-vs-MVG
   comparison *within* a condition is clean, both arms starting at the same
   point). Our fitness has no complexity term at all (the missing 0.01/neuron
   penalty, top of this file), so nothing rewards fewer edges. The likelier
   reading: under MVG a newly added edge must help on **both** goals to keep
   paying off across a switch, so goal-specific edges are repeatedly de-selected
   and fixation of new wiring is slower. Non-stationarity acting as a
   regulariser, not parsimony pressure.
3. **A one-side detector scores 0.75 — on BOTH goals.** Raw fraction-correct over
   all 256 patterns:

   | predictor | on AND | on OR | mean over the MVG schedule |
   |---|---|---|---|
   | constant 0 | 0.750 | 0.250 | 0.500 |
   | constant 1 | 0.250 | 0.750 | 0.500 |
   | **LEFT only** (or RIGHT only) | **0.750** | **0.750** | **0.750** |

   LEFT and RIGHT are each true on exactly 128/256 patterns, AND on 64, OR on
   192. So the one-module solution is the unique 0.75-everywhere plateau: under
   MVG it strictly dominates a constant output (0.75 vs 0.50 averaged over the
   schedule) and pays **zero re-adaptation cost at every switch**. MVG cannot
   punish it — which is why MVG's mechanism only engages once a network already
   computes both halves. (The `retina_ka2005` stand-in used by experiments 1–3
   has the same property at 0.8333; this is the KA-faithful task's version of it.)

### Exact parameters — both conditions, for replication

Commands (repo root; `conda run` because the terminal does not persist conda):

```bash
# capped, paper-faithful: 5 FG + 5 MVG -> kashtan_alon/runs/
conda run -n lndp python kashtan_alon/run_paper.py --n-seeds 5 --viz --fresh

# the deterministic duplicate that also archives per-generation purity and the
# champion brain at every log point -> kashtan_alon/runs_purity/
# (same seeds, same parameters, bit-identical trajectory; the brains archive is
#  what makes goal-matched re-analysis possible with no re-training)
conda run -n lndp python kashtan_alon/analysis/fg_mvg_purity.py --n-seeds 5

# ABLATION: 5 FG + 5 MVG -> kashtan_alon/runs_no_fanin/
conda run -n lndp python kashtan_alon/run_ablation_no_fanin.py --n-seeds 5
```

**The ablation is one line**: `M.NetConfig(layers=layers, fan_in=())` instead of
`M.NetConfig(layers=layers)`. `model.py:_fan_in()` then returns the full previous
layer. It is read in **two** places, so both change: `model.py:91` (initial
fan-in, `k = round(init_density × cap)` → 2 edges/neuron capped vs 4 uncapped)
and `ga.py:95` (the ceiling the add-edge mutation may not exceed). Every other
parameter is identical between conditions:

| parameter | value | provenance |
|---|---|---|
| architecture | retina(8) → 8 → 4 → 2 → 1 | paper, verbatim |
| weights | ∈ {−1, +1}, magnitude never mutates | paper, verbatim |
| units | hard threshold, fires iff (Σw·x + bias) > 0; bias = −threshold | paper, verbatim |
| fan-in cap | `(3,3,3,2)` capped / `()` = unbounded ablation | capped = paper; ablation = ours |
| population | 600 | paper (circuit experiment; reused by analogy) |
| generations | 25,000 | paper, verbatim |
| elite | 150 of 600, copied unchanged | reconstruction by analogy |
| crossover `Pc` | 0.5, **per destination neuron** — a neuron's whole incoming column + its threshold is inherited from one parent | paper says "neuron-level"; the mechanism is our reconstruction |
| mutation `Pm` | 0.5 per genome, **exactly one** edit from {add edge, remove edge, flip sign, nudge threshold ±1 clamped to [−3,+3]} | `Pm` is paper; the operator set is our reconstruction (Supporting Info unavailable) |
| initial density | 0.5 (as a fraction of the fan-in cap, per node) | ours — not paper-stated |
| fitness | `raw` = fraction correct over **all 256** patterns, every generation | KA's measure; paper samples 100/gen (deviation 1) |
| goals | FG: `LEFT AND RIGHT` throughout. MVG: AND↔OR, switch every `E = 20` generations | paper, verbatim |
| `Q_m` randomisations | 1,000 | paper, verbatim |
| `Q_m` estimator | degree-preserving hill-climb, 6 restarts × 250 steps | **not** the paper's (it re-evolves 100 populations toward Q) — deviation 2 |
| complexity penalty | **absent** | paper has 0.01/neuron above 13 — deviation 3, known gap |
| log interval | every 10 generations (`purity` column only in `runs_purity/`, `runs_no_fanin/`) | ours |
| seeds | 0–4 per arm per condition, `np.random.default_rng(seed)` | ours |

Seeded and side-effect-free: re-running a seed retraces the identical
trajectory, which `analysis/dense_replay.py --verify [--no-fanin]` proves rather
than assumes (it re-runs seed 0 with per-generation logging and diffs every
shared generation against the archive).

---

## Run 7 — Run 6 ablation extended to 5 seeds/condition + purity/left_right_q scored on all 20 runs (2026-09-10, rough notes) → **direction survives at full power on all 4 metrics**

Two things, both analysis/completion of existing work, no code changes:
1. **Run 6 extended from 3 to 5 seeds/condition** (trained only the 2 missing MVG +
   2 missing FG seeds; resumed, so seeds 0-2 are untouched from Run 6). Command:
   `conda run -n lndp python run_ablation_no_fanin.py --n-seeds 5`.
2. **`qmetrics`'s circuit purity (METRIC 4) and `left_right_q` (METRIC 3, planted
   left/right partition)** — implemented in `qmetrics/` but never run on any
   kashtan_alon result before now — computed on all 20 saved `*_best.npz` genomes
   (Run 5's 10 + Run 6's 10), analysis-only, alongside the existing Q_m/raw Q.
   Scripted in `scratch_metrics_table.py` (gitignored, `scratch_` convention;
   rerun any time — no re-evolution needed).

| group | n | Q_m | raw Q | purity | left_right_q |
|---|---|---|---|---|---|
| Run 5 FG (capped fan-in) | 5 | 0.025 | 0.380 | 0.561 | 0.546 |
| Run 5 MVG (capped fan-in) | 5 | 0.245 | 0.479 | 0.929 | 0.916 |
| Run 6 FG (no fan-in) | 5 | −0.071 | 0.216 | 0.307 | 0.537 |
| Run 6 MVG (no fan-in) | 5 | 0.079 | 0.306 | 0.567 | 0.766 |

Significance on Q_m (Welch t two-sided / Mann-Whitney U one-sided MVG>FG):
- Run 5: t=3.34, p=0.021; MWU p=0.016 (matches the original Run 5 write-up below).
- **Run 6 at n=5** (was n=3, "not significant, treat as suggestive"): t=2.11,
  p=0.070 (two-sided, borderline); MWU p=0.048 (significant).

**Verdict:** MVG beats FG on **all four metrics**, both capped and uncapped — the
direction is robust to removing the fan-in cap. The Run 6 gap, previously
underpowered at n=3, is now borderline/significant at n=5. Absolute modularity
still drops in both arms when the cap is removed (as in Run 6's original
write-up) — scarcity still sets the ceiling on how modular a solution gets, it
just isn't necessary for MVG to beat FG.

Not yet done: `left_right_q`/purity logged live per-generation the way
`runs_purity/` logs purity for Run 5 (that folder's genomes are bit-identical
to Run 5's by construction, so its archived per-gen brains are usable for this
without any new training — see `analysis/fg_mvg_purity.py`). No equivalent
per-generation archive exists for Run 6 yet.

---

## Run 6 — fan-in cap ablation, 3 seeds/condition (2026-08-20) → **partial support: absolute Q_m drops, but the MVG>FG gap survives**

Direct ablation of KA's own constraint (the fan-in cap, `RETINA_FAN_IN = (3,3,3,2)`
in `model.py`), testing the constraint-necessity hypothesis in `add_to_latex.md`
("Testing whether a constraint is necessary for modularity"). Same Fig 5a retina
task, same GA hyperparameters as Run 5 (pop 600, 25 000 gens, elite 150, Pc=0.5,
Pm=0.5, `fitness=raw`, `qm_nrand=1000`) — the only change is
`NetConfig(fan_in=())`: every neuron may now receive from every node in the
previous layer instead of being capped at 3 (hidden layers) / 2 (output).

- **Command:** `conda run -n lndp python run_ablation_no_fanin.py --n-seeds 3 --fresh`
- **Where:** laptop (CPU) · 3 seeds/condition (fewer than Run 5's 5, for runtime —
  see caveat below).

| condition | mean Q_m (no cap) | per seed | acc AND / OR | mean density | Run 5 (capped) Q_m | Run 5 density |
|---|---|---|---|---|---|---|
| **MVG** | **0.119 ± 0.140** | 0.084, −0.000, 0.272 | 0.998 / 0.996 | 56.0% | 0.245 ± 0.049 | 33.4% |
| **FG (L AND R)** | **−0.100 ± 0.113** | −0.197, −0.127, 0.025 | 0.970 / n.a. | 66.7% | 0.025 ± 0.139 | 37.7% |

> Accuracy column **corrected 2026-09-10** (n=5 seeds; per-goal, last 5,000 gens —
> see the reporting-bug note at the top). Replaces "mean best fit MVG 1.000 / FG
> 0.975", where MVG's figure was a cross-goal maximum taken during an OR phase.

- **Not significant at n=3:** Welch t = 2.11 (p ≈ 0.11), Mann–Whitney U = 8 (p ≈
  0.10). Underpowered — treat magnitudes as suggestive only.
- **The MVG−FG separation survives, almost unchanged:** gap = 0.219 with the cap
  removed vs. 0.220 with it (Run 5). Every seed still ranks the way KA predicts
  (MVG's worst seed, −0.0004, still beats FG's best seed, 0.025).
- **But absolute Q_m drops for BOTH conditions**, by about the same amount
  (MVG 0.245→0.119, ∆−0.126; FG 0.025→−0.100, ∆−0.125) — not selectively on FG.
  Density roughly doubles in both conditions too (MVG 33%→56%, FG 38%→67%),
  consistent with wiring no longer being scarce, but the network never saturates
  to 100% the way `experiment_1`'s no-budget arms did (see note 3 below). ⚠️ See
  note 5 below — a chunk of this density jump is an init-seeding artifact, not
  purely evolution's doing.
- **FG solves the task *better* without the cap** (0.970 on AND vs Run 5's
  0.904) while MVG stays near-saturated either way (0.998 on AND, 0.996 on OR,
  vs Run 5's 0.952/0.965) — unlimited fan-in gives FG
  more raw capacity to just solve the task, which is the mechanism the
  hypothesis predicts, but it doesn't push FG's Q_m low enough to erase the gap.

**Verdict: does NOT cleanly confirm "removing the constraint kills modularity"
— it complicates the strong form of that prediction.** What the data actually
shows:
1. **The MVG-driven relative advantage over FG looks constraint-independent** —
   the ~0.22 gap holds whether or not the fan-in cap exists. Goal-switching
   itself, not the fan-in cap specifically, appears to be doing most of the work
   of separating MVG from FG in this model.
2. **The fan-in cap does matter for absolute modularity magnitude** — both
   conditions lose ~0.12–0.13 of Q_m without it, and FG's best fit rises. That's
   consistent with "scarcity forces reuse" as a story about *how well/cheaply*
   the task gets solved, not as the sole cause of MVG beating FG.
3. n=3/condition, high variance (MVG spans −0.0004 to 0.272 — one seed sits
   right inside Run 5's constrained range). Not proof either way at this n.
4. **This does not match `experiment_1`'s no-budget result, and that mismatch
   is itself informative.** There, removing the synaptic budget saturated both
   arms to 100% density and made the modularity metric undefined (no structure
   left to score) — the strongest possible "constraint removed → no modularity
   measurable" outcome. Here, removing the fan-in cap raised density to only
   56–67%, nowhere near saturating; KA's mutation operator (`ga.py`'s
   `_add_edge`, one edge at a time, capped implicitly by `_fan_in()` returning
   full layer width) never drives the network to the complete graph in 25 000
   gens the way an unbounded continuous weight budget does. The two ablations
   are not equivalent tests of "remove the constraint" — one hits a hard
   ceiling (complete graph), the other doesn't.
5. ⚠️ **`--init-density` is itself scaled by the cap, so removing the cap
   silently changes the starting point, not just the mutation dynamics.**
   `init_population()` seeds every genome with `k = round(init_density * cap)`
   incoming edges per destination neuron. With `init_density=0.5` (default,
   unchanged in this ablation) and the cap removed, `cap` = the full previous
   layer's width, so **every genome starts at exactly 50.0% density on gen 0**
   (confirmed in the log CSVs: gen-0 density is 50.00% for every uncapped seed,
   vs. 27.36% for the Run 5/capped baseline, since there `cap`=3 or 2). Checking
   the full per-gen trajectory: FG genuinely climbs from that 50% seed to a
   ~65–67% attractor within the first ~500 generations and holds there for the
   remaining 24 500 — real evolutionary movement. MVG mostly random-walks
   between ~41–54% around its 50% start with no clear net drift, ending at
   51.9–60.4%. **So the "density roughly doubles" framing above overstates how
   much of the capped-vs-uncapped density gap is evolution's doing** — part of
   it (the jump from 27%→50% at gen 0) is purely the init-density formula
   reinterpreting "0.5" against a much wider cap, not a result. Re-running with
   an `init_density` chosen to give a comparable *absolute* edge count at gen 0
   (or logging density-vs-generation explicitly) would be needed to cleanly
   separate "evolution converges to a different density" from "we planted a
   denser forest to begin with."

### Next steps (proposed)
1. **More seeds** (5+, matching Run 5) to get the gap-survives finding past n=3
   noise before trusting the exact magnitude of either effect.
2. If the gap really does survive at higher n, the constraint-necessity framing
   in `add_to_latex.md` needs revising for KA specifically: the fan-in cap
   is not what makes MVG > FG in this model — something about goal-switching
   itself is, and the cap only sets the absolute modularity ceiling.
3. Worth trying a bigger ablation — remove fan-in *and* increase population/task
   size so density can actually approach saturation — to get a cleaner match to
   the `experiment_1` no-budget condition and see if the gap survives even there.
4. **Fix the init-density confound (note 5) before trusting the density
   numbers**: rerun with `init_density` scaled to match Run 5's gen-0 absolute
   edge count (≈27% of 106 ≈ 29 edges), so both conditions start from the same
   seed and any divergence is attributable to evolution, not initialization.

---

## Run 5 — first KA-faithful run, 5 seeds (2026-08-03) → ✅ **MVG > FG reproduced (directional, significant)**

The real test on the rebuilt (KA-faithful) code: MVG vs Fixed-Goal, pop 600, 25 000
gens, 5 seeds, scored with normalized **Q_m**. This is the run Runs 1–3 should have
been (they ran Clune's reimplementation; see Run 4 / [[reference_ka_retina_algo]]).

- **Command:** `conda run -n lndp python run_paper.py --n-seeds 5 --viz --fresh`
- **Where:** laptop (CPU, pure-numpy, 0 VRAM) · ~0.015–0.024 s/gen · full run ≈ 50 min.

| condition | mean Q_m | per seed | acc on AND | acc on OR |
|---|---|---|---|---|
| **MVG** | **0.245 ± 0.049** | 0.186, 0.316, 0.266, 0.240, 0.217 | **0.952** | 0.965 |
| **FG (L AND R)** | **0.025 ± 0.139** | −0.143, 0.228, −0.007, −0.036, 0.083 | **0.904** | n/a |

> Accuracy columns **corrected 2026-09-10** (see the reporting-bug note at the top).
> They are the true per-generation champion fitness from the CSVs, split by the live
> goal, averaged over the last 5,000 generations — so both arms are compared on the
> *same* task. They replace a previously-tabled "mean best fit MVG 0.975 vs FG 0.904",
> where MVG's figure was a cross-goal maximum recorded during an OR phase.
> Re-derived from existing logs by `scratch_metrics_table.py`; no re-training.

- **Separation +0.22, significant:** Welch t ≈ 3.3 (p ≈ 0.02); Mann–Whitney U = 2
  (p ≈ 0.03). The gap matches KA's own (0.35 − 0.15 = 0.20).
- **Every MVG seed is positively modular (0.19–0.32); 4/5 FG seeds sit at/below 0.**
  A genuine MVG>FG modularity difference between equally-performing nets — KA's claim,
  and a clean flip from the earlier null.
- **Brains** (`runs/*_best.png`): MVG nets show the left-4 / right-4 retina feeding
  largely separate hidden modules meeting near the output; FG nets are tangled
  (cross-wiring at every layer). Visually consistent with the Q_m gap.

**Verdict: ✅ direction reproduced, statistically significant.** NOT a full match:
1. **Absolute scale shifted down** — MVG 0.25 (KA 0.35), FG 0.03 (KA 0.15); the effect
   size reproduces but the magnitudes don't. Prime suspects: the SI-reconstructed
   mutation operators/threshold range (`ga.py`) and our exact pixel→object mapping.
2. **FG seed 1 (0.228)** is an outlier as modular as MVG → FG's SD is large; the effect
   is significant but not every FG run is non-modular (n=5).
3. Even MVG nets keep some cross-wiring (not a textbook-clean split).

### Next steps (proposed)
1. **Chase KA's SI** for the exact mutation operators + threshold range — the most
   likely reason the absolute Q_m sits below the paper. Cheapest path to a full match.
2. **More seeds** (n≥10) to tighten the FG spread and the significance.
3. Wire **Q_m into per-gen logging** (currently raw Q) to see *when* in evolution the
   MVG modularity emerges vs the FG.

---

## Run 4 — code rebuilt to be KA-faithful (2026-08-03) → **pipeline verified; full run pending**

Runs 1–3 turned out to be running **Clune 2013's reimplementation**, not KA 2005 —
the spec was reconstructed from Clune and mislabelled "confirmed KA" (details:
[[reference_ka_retina_algo]]). On 2026-08-03 the actual KA methods were fetched from
the primary source (PMC1236541) and the code was **rewritten to match** the paper's
neural-network retina experiment:

| what changed | before (Clune-derived) | now (KA 2005, verified) |
|---|---|---|
| task | stand-in `(p0∧p1)∨(p2∧p3)` | **KA's real Fig. 5a objects** (≥3-black / outer-column-only) |
| weights / units | {−2,−1,1,2}, tanh(λ=20) | **±1**, hard **threshold** ({0,1} out) |
| fan-in | none | **≤3 / ≤2** per layer |
| reproduction | mutation-only | **crossover Pc=0.5** + mutation Pm=0.5 |
| selection / pop | tournament + 1 elite, pop 1000 | **elite 150 / 600** |
| headline metric | raw Newman Q | normalized **Q_m**, Q_rand over **1000** |

Architecture was **already correct at 8-8-4-2-1** (8 retina pixels are a separate
input layer; an intermediate note that mis-said "8-4-2-1" was wrong).

- **Verified:** `test_tasks.py` passes with KA's exact truth counts (left object 8/16,
  retina AND 64/256, OR 192/256, left/right provably independent); a smoke run
  (pop 60 / 300 gens) exercises the whole GA→Q_m→checkpoint→viz pipeline cleanly.
- **Not yet done:** the real MVG-vs-FG run (pop 600, 25 000 gens). Expectation if the
  reproduction is faithful: **Q_m ≈ 0.35 (MVG) vs ≈ 0.15 (FG)**. This is the run that
  actually tests KA's claim — everything before it was the wrong experiment.
- **Residual caveat:** exact mutation operators + threshold range are in KA's SI (not
  the main text); `ga.py` uses a documented faithful reconstruction.

---

> ⚠️ **CORRECTION (2026-08-03):** Run 2's "reproduced" verdict below used **raw
> Newman Q**, which is density-confounded. Re-scored with **Kashtan-Alon's actual
> normalized metric Q_m = (Q_real−Q_rand)/(Q_max−Q_rand)**, the result is a **NULL**:
> MVG Q_m = −0.04 ± 0.17, FG = −0.13 ± 0.12 (KA target: 0.35 vs 0.15). Both conditions
> are non-modular (Q_real ≈ Q_rand — no excess over degree-matched random), and MVG is
> not meaningfully above FG. The raw-Q "0.38 vs 0.27" gap was purely the density
> artifact (MVG nets sparser ⇒ higher raw Q for free). Metric validated: two-clique
> graph → Q_m 1.00, ER random → 0.22, ring lattice → 0.50. **We have NOT reproduced
> KA's MVG→modularity result.** See "Run 3" note below.

## Run 2 — corrected measurement, 5 seeds (2026-08-02) → ~~✅ MVG > FG reproduced~~ **retracted (raw-Q artifact)**

Same paper-locked config as Run 1, but with the two Run-1
measurement artifacts fixed (commit `8a60977`): **early-stop OFF** (FG evolves the
full 25000 gens, matched to MVG) and **Q measured on the final-generation champion**
(the evolved topology), not the first net to hit peak fitness. **The raw-Q numbers
below stand as computed but are the WRONG metric — see the correction banner above.**

- **Command:** `conda run --no-capture-output -n lndp python run_paper.py --n-seeds 5 --fresh --viz`
- **Where:** laptop (CPU, pure-numpy, 0 VRAM) · full run ≈ 2.3 h · all seeds fit 1.000.

### Result — the separation is clean

| Measure | MVG | FG (L AND R) | gap |
|---|---|---|---|
| **Final-gen Q** (mean) | **0.383** | 0.274 | **+0.109** |
| Final-gen Q per seed | 0.400, 0.339, 0.332, 0.470, 0.372 | 0.295, 0.270, 0.318, 0.196, 0.289 | |
| Per-gen Q, last-10% window (phase-averaged, robust) | 0.341 ± 0.014 | 0.284 ± 0.020 | +0.057 |
| edges (final net) | 34–41 (sparser) | 46–59 | |

- **No overlap:** MVG's worst seed (0.332) > FG's best (0.318). **25/25** seed-pairs MVG > FG.
- Every run solves the task (fit 1.000 both conditions) → a modularity difference
  between *equally-performing* networks, which is exactly Kashtan–Alon's claim.
- MVG nets are consistently **sparser** — modular solutions prune cross-connections.

**Verdict vs paper:** ✅ **Direction and consistency reproduced** — MVG spontaneously
favours higher modularity than a fixed goal, robustly across seeds and across two
independent Q measures. Absolute scale differs from the paper (our FG ≈ 0.27 vs
paper's ~0.15–0.2; our MVG ≈ 0.38 vs ~0.4+) — expected, since our retina object
patterns and greedy-Newman-Q metric differ from KA's, and there's no connection-cost
term pushing FG lower. What's testable — MVG > FG — holds cleanly.

### Run 3 — re-scored with KA's normalized Q_m (2026-08-03, no re-run; on Run 2's saved nets)

Implemented `modularity.normalized_qm` = `(Q_real − Q_rand)/(Q_max − Q_rand)`
(Q_rand = mean greedy-Q over 100 degree-preserving randomizations; Q_max = greedy-Q
of a modularity-maximizing rewiring at the same degree sequence). Metric validated on
known graphs: two K6 cliques+bridge → **1.00**, ER random → **0.22**, ring → **0.50**.

| condition | Q_m (mean ± sd) | per seed | raw Q (Run 2) |
|---|---|---|---|
| MVG | **−0.04 ± 0.17** | −0.07, −0.27, −0.15, +0.24, +0.04 | 0.383 |
| FG  | **−0.13 ± 0.12** | −0.06, −0.07, −0.05, −0.36, −0.11 | 0.274 |

**Verdict: NULL — not reproduced.** Every net has Q_real ≈ Q_rand (no excess modularity
over degree-matched random); both conditions are non-modular and below even a random
graph's Q_m ≈ 0.22. MVG (−0.04) vs FG (−0.13) is within noise (n=5). The Run-2 raw-Q
separation was a pure density artifact.

### Next steps (proposed)
1. **Clune connection-cost term** — promoted to #1. The follow-up literature's reliable
   modularity driver; MVG-alone is fragile/insufficient (our null is consistent with
   this). A variation, deliberately outside KA's MVG-only claim.
2. **Lower `--init-density`** (currently 0.5): sparser nets, higher Q ceiling.
3. **Faithfulness audit of task/architecture** vs KA's exact retina — our object
   patterns are a stand-in and may not create KA's modular pressure.
4. **Wire Q_m into `train.py` per-gen logging** (currently logs raw Q) so future runs
   track the correct metric live, and consider a direct functional left/right measure.

---

## Run 1 — full paper preset, 5 seeds (2026-07-31) → **near-null (measurement artifacts, superseded by Run 2)**

_Note: Run 1's seed-0 raw artifacts were later overwritten by a smoke test; the
aggregate numbers below stand, and Run 2 supersedes this run. Kept as the record of
why the naive measurement produced a false null._

- **Command:** `conda run -n lndp python run_paper.py --n-seeds 5 --viz`
- **Where:** laptop (CPU, pure-numpy, 0 VRAM) · ~30–34 ms/gen · full run ≈ 2.3 h
- **Config (locked to paper):** net 8-8-4-2-1, tanh λ=20, weights {−2,−1,1,2},
  biases {−2..2}, pop 1000, 25000 gens, mutation-only GA (add/remove 20%, weight
  ±1 @ 2/n, bias ±1 @ 1/24), tournament k=3 + 1 elite, **raw** fraction-correct
  fitness. Two conditions: **MVG** (and↔or/20) vs **FG(L AND R)**.
- **Logs:** `runs/retina_{mvg,fg}_raw_seed{0..4}_log.csv` (per-gen Q/density/fitness),
  `runs/*_best.png` (final brains, module-coloured), `runs/*_result.json` (per-seed
  summary). NB: `runs/paper_run.log` (console) was buffered by `conda run` and only
  flushed at exit — use `--no-capture-output` next time for a live console stream.

### Result — the paper's separation did NOT reproduce

**Best-net Q** (saved best-fitness individual, from `result.json`):

| Condition | mean Q | Q per seed | mean best fit |
|---|---|---|---|
| MVG (and↔or) | **0.309** | 0.339, 0.296, 0.298, 0.314, 0.299 | 1.000 |
| FG (L AND R) | **0.279** | 0.308, 0.296, 0.271, 0.219, 0.302 | 1.000 |

**Per-generation Q** (fairer — best net *each* logged gen, over the whole run):

| Condition | mean (all gens) | mean (2nd half) | max ever | last gen / seed |
|---|---|---|---|---|
| MVG | 0.335 | 0.339 | 0.549 | 25000 (all seeds) |
| FG  | 0.248 | 0.254 | 0.376 | 480, 220, 8090, 7490, 490 |

**Verdict vs paper:** ❌ **Not reproduced at the paper's magnitude.** Paper claims
MVG ~0.4+ vs FG ~0.15–0.2. We get MVG ≈ 0.31–0.34 vs FG ≈ 0.25–0.28 — *directionally*
correct (MVG more modular) but only a ~0.05–0.09 gap, and both well inside each
other's range. Every seed solved the task (fit 1.000).

### Confounds / caveats (read before trusting this)

- **FG early-stops, MVG doesn't** — the biggest one. FG hits fit 1.0 → `--target 1.0`
  early-stop fires (3/5 seeds before **gen 500**; others ~7500–8000), while MVG runs
  the full 25000 gens (its goal keeps moving, so it never early-stops). So FG is
  scored on a barely-evolved first solution and MVG on a long-evolved one — **not
  matched on evolutionary time.** *Fix for the next run:* `--target 1.1` (or a
  no-early-stop flag) so FG also evolves the full 25000 gens. Even handicapped-short,
  though, FG's separation from MVG is already weak.
- **Object patterns ≠ KA's.** We use `L=(p0∧p1)∨(p2∧p3)`, `R=(p4∧p5)∨(p6∧p7)`; KA's
  exact retina bit-patterns are unpublished. The MVG effect is known to be sensitive
  to task details.
- **Consistent with the follow-up literature.** Clune, Mouret & Lipson 2013 found MVG
  *alone* is an unreliable modularity driver without a connection-cost term — a weak
  separation here is not surprising. Adding a connection-cost variant is the obvious
  next experiment (a *variation*, not the paper).
- **Best-net Q ≈ per-gen Q** here (0.31 vs 0.34 MVG; 0.28 vs 0.25 FG), so the
  "save the best-*fitness* net, not the most-modular one" choice is not what's
  suppressing the signal — the signal is genuinely weak.

### Next steps (proposed, not yet run)
1. Re-run with FG early-stop disabled (`--target 1.1`) — the fair apples-to-apples.
2. If still weak: try a connection-cost variant (Clune's driver) and/or `xor` goal.
3. Consider the exact KA goal pair / object set if we can recover it.

---

## Conventions for this file
- One `## Run N` block per run, newest first; keep the exact command and the config
  that mattered so it's reproducible.
- Record the **Q separation** (the headline), not just final fitness.
- `runs/` is gitignored (regenerable) — conclusions live here.
