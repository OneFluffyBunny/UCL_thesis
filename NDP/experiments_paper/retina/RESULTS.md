# NDP × Kashtan-Alon retina task — results log

Lab notebook for the `KA_experiments` branch: every real run of the retina-AND
task (`ka_task.py`) inside the NDP growth framework. Raw per-run outputs
(config.yml, logger.csv, PNGs, solution .npy) live under `saved_models/<id>/`
(gitignored, regenerable) — this file is the durable record of what was found.
Training commands are given under each run below; every derived claim below
that (mechanistic explanation, error comparison, saturation table) is
reproducible from those saved outputs via `experiments_paper/retina/analysis.py`
— exact command given inline where each claim is made.

## Task

Single-output, bipolar {-1,+1}, sign-of-output classification. Target =
`left_feature(x) AND right_feature(x)`, both `(x0&x1)|(x2&x3)` mirrored onto
two disjoint 4-bit blocks (`ka_task.py`) — the project's own stand-in, not
Kashtan & Alon (2005)'s real Fig. 5a rule (see `kashtan_alon/` in the sibling
`UCL_thesis` repo for that). 256 exhaustively-enumerated patterns, evaluated
independently. Imbalanced (~19% positive) → raw accuracy is deceptive
(constant-0 scores ~81% raw); fitness switched to balanced accuracy via
`--balanced-fitness`.

## Runs

### Run 1786030050 — `size-reg-alpha 0.04`, warmup 500, 1000 gens

`--balanced-fitness --no-early-stopping --size-reg io_ratio --size-reg-alpha 0.04 --size-reg-warmup 500 --generations 1000 --popsize 128`

(I-I/O-O edge ban not implemented yet — seed graph still fully dense, 81 edges)

- Population collapsed to the minimum 9-node, zero-growth network by ~gen 10, never recovered.
- **Best: balanced accuracy 0.8432416444838805 (raw 204/256), 9 nodes.**
- 358s total.

### Run 1786032037 — `size-reg-alpha 0.01`, no warmup, 3000 gens, I-I/O-O banned

`--balanced-fitness --no-early-stopping --size-reg io_ratio --size-reg-alpha 0.01 --generations 3000 --popsize 128 --show`

(first run with `forbid_io_self_edges` default-on: seed graph now 16 edges, pure I-O connectivity, no I-I/O-O)

- Baseline (untrained x0 DNA): fitness -0.414 (raw 0.2/256), 576 nodes.
- Population collapsed to the same minimum 9-node, zero-growth network by ~gen 60-100, never recovered for the remaining ~2900 generations (sigma kept wandering 0.3-0.5 — CMA-ES was still searching, just never found anything better).
- **Best: balanced accuracy 0.8432416444838805 (raw 204/256), 9 nodes, 16 edges.**
- Bit-for-bit identical fitness to Run 1786030050, despite alpha being 4x weaker and I-I/O-O edges now structurally impossible. Suggests this specific zero-growth solution is a genuine, sharp optimum (likely global-for-zero-growth), not an artifact of over-strong regularisation.
- 899s total.

### Run 1786033855 — no size regularisation at all, 200 gens

`--balanced-fitness --no-early-stopping --generations 200 --popsize 128 --show`

(no `--size-reg` flag: nothing discourages growth)

- Baseline (untrained x0 DNA): fitness 0.241 (raw 0.2/256), 13 nodes (varies run to run — x0 is random, no fixed seed in the yaml).
- Gen 0 population mean was 314 nodes, falling to a mean of roughly 10-30 nodes by gen ~50-190 — **with zero explicit size penalty**. The live per-generation log showed the current-generation best individual at 9 nodes for most of the run.
- **Final `solution_best` actually grows to 40 nodes, 502 edges — balanced accuracy 0.8438 (raw 201/256).** Verified directly: `grow_network()` on the saved `solution_best.npy` gives 40 nodes, matching what CMA-ES's own training-time fitness closure (`fitness_functional(..., return_stats=True)`) reports for the same DNA. (Earlier version of this entry said "9 nodes, zero growth" — that was wrong, see the print-labelling bug below; corrected after investigation.)
- So growth *did* happen here, unforced — but the resulting 40-node network scores essentially the same (0.8438) as the two regularised runs' 9-node linear/majority-vote solutions (0.8432). Real evidence that, so far, growth produces bigger but not meaningfully *better* networks on this task within a couple hundred generations — consistent with, not contradicting, the plateau story below.
- **Root-caused a logging bug, not a growth bug**: the "before growth" checksum print in `fitness_functional()` (`train_backend.py:294`, now fixed) reused the label "The final grown graph has N nodes..." for the *seed* graph (always 9 nodes/16 edges for retina), printed right before the real post-growth result under the same misleading label. Both `grow_network()` and the training-time fitness closure agreed all along (40 nodes) — there was never an actual discrepancy between code paths, only a confusing duplicate print. Fixed to say "The seed graph (before growth) has...".

### Run 1786053806 — MVG (and/or, switch every 20 gens), 1500 gens

`--balanced-fitness --no-early-stopping --mvg --mvg-switch-interval 20 --generations 1500 --popsize 128 --show`

(first MVG run: no `--size-reg`; goal alternates and/or every 20 gens, 75 switches total; `solution_best`/`solution_centroid` are the final-generation champion under MVG, not best-ever — see optimizers.py)

- 3046s total (~51 min).
- **Final-gen champion (op=and, gen 1499): balanced accuracy 0.8438 (raw 201/256), 80 nodes, 1586 edges.** Peak-ever fitness across the whole run was also 0.8438 — same ~0.84 ceiling as every FG run, this time via an 80-node substrate. Numerically identical to Run 1786033855's fitness value (0.8437838903677413) to full precision.
- Node count never settled the way FG runs did: bounced 9-320+ in the first ~150 gens, then oscillated in a calmer ~20-80 range for the rest of the run without ever freezing — goal-switching kept the population from collapsing to a fixed topology, unlike every FG run so far.
- Modularity not yet measured — no Q/Q_m metric wired into NDP yet (exists in the sibling `kashtan_alon/` repro as `qmetrics/`, not connected here). Fitness parity with FG runs says nothing about whether the resulting graph is more modular, which is the actual KA hypothesis this run is meant to test. **Pending: modularity comparison script.**

### Run 1786102425 — FG, pruning threshold 0.3, 200 gens

`--balanced-fitness --no-early-stopping --pruning --pruning-threshold 0.3 --generations 200 --popsize 128 --show`

- **Identical outcome to Run 1786033855 (no pruning): 40 nodes, 502 edges, balanced accuracy 0.8438 (raw 201/256).** Pruning had zero effect.
- Cause verified directly (`python experiments_paper/retina/analysis.py weights 1786102425`): every one of the 502 edge weights is ~0.9997 (tanh-saturated), nowhere near the 0.3 threshold. `--pruning-threshold 0.5` would fail identically — not tested, since the cause was already confirmed rather than guessed.

## Mechanistic explanation of the ~0.84 plateau

`python experiments_paper/retina/analysis.py weights 1786032037` and
`python experiments_paper/retina/analysis.py popcount`

Inspected the actual evolved weight matrix of the best solution from Run
1786032037 (`saved_models/1786032037/solution_best.npy`, via `grow_network()`):
all 8 input→output weights are identical (`0.628` each). With no hidden
nodes and no bias anywhere in the propagation equation, this network is
**exactly an unweighted majority vote over the 8 raw bipolar pixels**
(predict 1 iff ≥5 of 8 bits are `1`) — hand-computing this rule reproduces
`0.8432416445` to 10 decimal places, an exact match.

Per-popcount breakdown of the true label explains why this scores so well
for something that ignores the task's actual AND/OR structure entirely:

| popcount | patterns | P(label=1) |
|---|---|---|
| 0–3 | 163 | 0.000 |
| 4 | 70 | 0.057 |
| 5 | 56 | 0.286 |
| 6 | 28 | 0.714 |
| 7–8 | 9 | 1.000 |

More 1-bits mechanically raises the odds some adjacent pair is `(1,1)`, so
popcount is a strong free proxy for the label almost everywhere — majority
vote is correct for free outside the popcount 4-6 band (154/256 patterns),
which is exactly where it's near coin-flip and all its errors live. Confirmed
via coordinate-wise perturbation that this is a local optimum (no single
weight change improves it). It's the symmetric, zero-growth point in weight
space, trivially reachable by CMA-ES — escaping it needs hidden nodes
computing genuine pairwise-AND features to resolve the popcount 4-6 band, but
any half-grown attempt at that pays size/coordination cost before it's
coherent enough to help, so selection snaps back to the free ~84% the linear
shortcut already provides. Run 1786033855 (unregularised) shows growth *can*
happen unforced (40 nodes) without this trap suppressing it, but the growth
that did happen didn't escape the ~84% plateau either — it just reached a
same-scoring solution by a bigger route.

### Error-pattern comparison: 9-node (regularised) vs 40-node (unregularised)

`python experiments_paper/retina/analysis.py compare 1786032037 1786033855`

Actually ran both grown networks over all 256 patterns and compared which
specific ones each gets wrong (not just the aggregate score):

- 9-node (Run 1786032037): 52/256 wrong. 40-node (Run 1786033855): 55/256 wrong.
- **51 of those 52 mistakes are shared** (Jaccard overlap 0.911) — only 5
  patterns differ between the two brains' error sets at all.
- Per-popcount error breakdown is essentially identical between them: both
  get **0 errors** outside the popcount 4-6 band, and both concentrate the
  bulk of their errors at **popcount=5** (40/56 wrong, for both) — i.e. both
  independently rediscovered the same "predict 1 iff popcount≥5" heuristic
  and fail on it in exactly the same place.
- The 5 differing patterns are all popcount=4 (the exact tie point of a
  uniform-weight rule, bipolar sum = 0) — i.e. the only place the two brains
  disagree is where the linear/majority heuristic is genuinely on a knife's
  edge and infinitesimal weight asymmetry decides it, not a sign of different
  computational strategies.
- **Conclusion**: growth, when it happens unforced, isn't finding a
  *different* solution to this task — it's re-deriving the same
  density/popcount proxy through a bigger, more redundant substrate. The
  ~0.84 boundary looks like a genuine property of what this architecture
  finds *easy* to discover here (a symmetric, count-based heuristic), not an
  artifact of network size or regularisation regime.

### Edge-weight saturation: universal across every run here, not universal to NDP

`python experiments_paper/retina/analysis.py saturation 1786030050 1786032037 1786033855 1786053806 1786102425 1785945244`

| Run | Edges | \|w\| min / max / std |
|---|---|---|
| 1786030050 (9n) | 81 | 0.8225 / 0.8225 / 0.000000 |
| 1786032037 (9n) | 16 | 0.6280 / 0.6280 / 0.000000 |
| 1786033855 (40n) | 502 | 0.9998 / 0.9999 / 0.000054 |
| 1786053806 (80n, MVG) | 1586 | 1.0000 / 1.0000 / 0.000000 |
| 1786102425 (40n, pruned) | 502 | 0.9997 / 0.9997 / 0.000000 |
| CartPole (`saved_models/1785945244`) | 468 | 0.0040 / 0.0548 / **0.021207** |

Every retina/KA run collapses to a single common edge magnitude (std ≈ 0),
regardless of size/regularisation/FG-vs-MVG. CartPole shows real spread and
no saturation — so this is a property of *this task's* degenerate shortcut,
not an inherent NDP tendency. Root cause for the 9-node cases is mechanical,
not evolved: `build_initial_network_state()` gives all 8 input nodes the
*same* embedding vector, so the weight-MLP is given identical input for
every input→output edge and cannot output anything but the same value.
Traced through propagation/growth and this holds for the whole run: the
architecture can only ever compute permutation-invariant functions of its
inputs (blind to *which* inputs are on), which the true task label is not —
see `add_to_latex.md`'s general-observations note on this.

## Open thread

All four runs land on essentially the same ~0.84 plateau. Two (regularised,
FG) via the exact same 9-node linear/majority-vote local optimum; the other
two (unregularised, one FG one MVG) via bigger (40- and 80-node) routes that
still don't beat it. So the plateau isn't specific to the zero-growth
solution, or to a fixed goal — it looks like a broader attractor that neither
ordinary growth nor goal-switching escapes on fitness terms, at least within
the generation counts tried so far. Untested: whether a *much* longer run,
an explicit push away from the linear optimum (higher `sigma_init`, seeding
hidden nodes at t=0), or — for MVG specifically — **whether the plateau being
identical on fitness hides a real difference in modularity** (Q/Q_m not yet
measured; comparison script pending).

## 5-seed FG vs MVG size/accuracy table (2026-09-08)

Topped up from 2 existing runs (1786033855 FG, 1786053806 MVG) to 5 completed
seeds per condition, to give the still-pending modularity comparison more than
one seed each to work with. Driver: `experiments_paper/retina/top_up_seeds.py`
(idempotent scan-and-launch over `saved_models/`, one run at a time; see its
docstring for the exact per-condition matching criteria). Note in passing: the
scan's own matching criteria turned up run 1786102425 (the pruning-threshold-0.3
run above) as an additional pre-existing FG match, since pruning uses different
config keys than the size-reg ones the filter checks — so only 3 new FG runs were
needed, not 4.

Exact commands (one per run, repeated to fill each condition to 5):

```
FG:  python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 200 --popsize 128 --balanced-fitness --no-early-stopping --operation and --snapshot
MVG: python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 1500 --popsize 128 --balanced-fitness --no-early-stopping --mvg --mvg-ops and,or --mvg-switch-interval 20 --snapshot
```

(`seed:` is `null` in the yaml, so every invocation gets an independent
`np.random.randint` seed automatically — no `--seed` flag exists.)

**MVG final-task check**: before trusting these MVG accuracies as comparable to
FG's, verified that the *last* generation (1499) of every one of the 5 MVG runs
was scored on op `and` — the same task FG is fixed to — not whatever op the
75th goal-switch happened to leave it on. Confirmed two ways: (1) each run's
saved `config.yml` has both `current_op: and` and `mvg_final_gen_op: and`; (2)
independently re-running `retina_fitness()` on the regrown network (which reads
`config["current_op"]`) reproduces each run's own stored `mvg_final_gen_fitness`
exactly. This isn't incidental to these 5 seeds — it falls out of the
generations/switch-interval combination itself: 1500 gens / 20-gen switches =
75 blocks (0-indexed 0..74), the last block (74) is even, and block parity
picks the first entry of `mvg_ops` (`and`) — so *any* run at these settings
ends on `and`.

Stats via `python experiments_paper/retina/run_stats.py <run_id> [<run_id> ...]`
(new script; loads `config.yml` + `solution_best.npy` through `grow_network()`,
same path as `analysis.py`'s `load_run()`; reports node/edge counts,
`retina_fitness()` balanced accuracy, and raw edge-weight `|w|` min/max/std —
deliberately no modularity metric, that's a separate step using `qmetrics`):

```
python experiments_paper/retina/run_stats.py 1786033855 1786102425 1788808796 1788808995 1788809505 1786053806 1788817038 1788821061 1788866451 1788871442
```

| Run ID | Seed | Cond | Nodes | Edges | Balanced Acc | \|w\| min/max/std | I/O weights uniform | Training time |
|---|---|---|---|---|---|---|---|---|
| 1786033855 | 7950840 | FG | 40 | 502 | 0.8438 | 0.9998 / 0.9999 / 0.000054 | Yes | 338s |
| 1786102425 | 1842456 | FG | 40 | 502 | 0.8438 | 0.9997 / 0.9997 / 0.000000 | Yes | 322s |
| 1788808796 | 1748523 | FG | 24 | 386 | 0.8432 | 0.5061 / 1.0000 / 0.132047 | Yes | 188s |
| 1788808995 | 1589167 | FG | 12 | 76 | **0.8540** | 0.9999 / 0.9999 / 0.000003 | Yes | 323s |
| 1788809505 | 7287042 | FG | 48 | 978 | 0.8438 | 1.0000 / 1.0000 / 0.000000 | Yes | 7494s* |
| 1786053806 | 7318332 | MVG | 80 | 1586 | 0.8438 | 1.0000 / 1.0000 / 0.000000 | Yes | 3046s |
| 1788817038 | 6550047 | MVG | 40 | 502 | 0.8438 | 1.0000 / 1.0000 / 0.000000 | Yes | 3993s |
| 1788821061 | 7766560 | MVG | 16 | 166 | **0.8540** | 0.2332 / 0.9955 / 0.176284 | Yes | 45372s* |
| 1788866451 | 6779840 | MVG | 40 | 502 | 0.8438 | 1.0000 / 1.0000 / 0.000000 | Yes | 4977s |
| 1788871442 | 9985617 | MVG | 40 | 502 | 0.8438 | 1.0000 / 1.0000 / 0.000000 | Yes | 3511s |

\* Inflated by the laptop sleeping mid-run (wall-clock `training time` counts the
sleep interval); not real compute time. Doesn't affect the trained result.

Group summary: FG nodes 12-48 (mean 32.80, median 40, stdev 14.53), edges
76-978 (mean 488.80, stdev 324.28), accuracy 0.8432-0.8540 (mean 0.8457). MVG
nodes 16-80 (mean 43.20, median 40, stdev 23.05), edges 166-1586 (mean 651.60,
stdev 542.23), accuracy 0.8438-0.8540 (mean 0.8459).

Note re: the earlier single-seed pilot (the two runs at the top of this file,
1786033855 FG=40 nodes vs 1786053806 MVG=80 nodes) — that comparison read as
MVG growing exactly 2x FG's size. Across the 5-seed samples the gap narrows a
lot: MVG's mean is +32% over FG's (43.20 vs 32.80 nodes), not +100%, and the
two conditions have identical medians (40 nodes each) — the original two pilot
runs are themselves members of these 5-seed samples, and MVG's 80-node run
remains the largest single network of all 10 (next-highest MVG run is 40).
Both distributions have large stdev relative to their means (FG: 14.53/32.80;
MVG: 23.05/43.20), i.e. within-condition spread is comparable to or larger
than the between-condition gap at n=5. No significance test run on this (size
was never the target metric here, modularity is) — noted descriptively only.

8 of 10 runs land on the exact same balanced accuracy
(`0.8437838903677413`, the popcount/majority-vote plateau established above).
Two runs — FG 1788808995 (12 nodes) and MVG 1788821061 (16 nodes) — reach a
different, higher value (`0.8539879720003943`) instead. Checked: both still
have perfectly uniform direct input→output weights (≈0.9999 and ≈0.9861
respectively, ptp < 1e-3), so neither has abandoned the linear/majority
heuristic — with a handful of extra hidden nodes beyond the 9-node minimum,
something about those nodes' indirect paths into the output shifts a few
boundary-case (popcount 4-6) predictions relative to the other 8 runs. Not
investigated further here.

> **Update 2026-09-14:** resolved — see "What the budget-matched brains compute"
> at the end. The 0.8540 is floating-point rounding at the popcount-4 tie, not a
> better brain: every brain is exactly a majority vote (true score 0.8432).

No modularity metric computed yet in this entry — node/edge counts and
accuracy alone don't distinguish FG from MVG (ranges overlap heavily, both
plateau at the same accuracy), which is expected: those stats say nothing
about *how* the edges are organized. That comparison is still pending.

## Budget-matched FG vs MVG seed set: 1500 generations each (2026-09-13)

**This supersedes the 5-seed table above as the FG/MVG comparison set.** In that
set FG ran 200 generations and MVG ran 1500. Fitness stops improving by gen
3-57 (FG) and 7-208 (MVG) in every run, so after that CMA-ES is drifting with
no fitness gradient, and MVG got ~1300 more generations of that drift than FG.
Any size or modularity difference measured on the final brains was therefore
confounded with the budget. The mismatch came from the original seed-top-up
spec, which inherited the two pilot runs' budgets (1786033855 at 200,
1786053806 at 1500). The Kashtan-Alon reproduction does not have this problem:
both arms run to generation 24,999 (`kashtan_alon/runs_purity/*_log.csv`).

Fix: 5 new FG runs at 1500 generations. The 5 MVG runs are unchanged. The five
200-generation FG runs stay in `saved_models/` but are no longer part of the
comparison.

Driver (resumable, rescans `saved_models/`, launches only what is missing):

```
conda run --no-capture-output -n ndp python -u experiments_paper/retina/top_up_seeds.py --target 5
```

which runs, one at a time:

```
FG:  python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 1500 --popsize 128 --balanced-fitness --no-early-stopping --operation and --snapshot
MVG: python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 1500 --popsize 128 --balanced-fitness --no-early-stopping --mvg --mvg-ops and,or --mvg-switch-interval 20 --snapshot
```

| Cond | Run ID | Seed | Final-gen best (bal. acc) | Training time |
|---|---|---|---|---|
| FG | 1789303257 | 3410590 | 0.8438 | 566s |
| FG | 1789303833 | 7345678 | 0.8438 | 529s |
| FG | 1789304372 | 7737269 | 0.8438 | 597s |
| FG | 1789304978 | 1231914 | 0.8540 | 644s |
| FG | 1789305630 | 5316271 | 0.8438 | 1457s |
| MVG | 1786053806 | 7318332 | 0.8438 | 3046s |
| MVG | 1788817038 | 6550047 | 0.8438 | 3993s |
| MVG | 1788821061 | 7766560 | 0.8540 | 45372s* |
| MVG | 1788866451 | 6779840 | 0.8438 | 4977s |
| MVG | 1788871442 | 9985617 | 0.8438 | 3511s |

\* includes a laptop-sleep gap; wall-clock, not compute.

Matched-parameter check: every config key was diffed across all 10 runs. The
only differences are `mvg` (False/True), MVG-only outputs
(`mvg_final_gen_op`, `mvg_final_gen_fitness`, `mvg_peak_fitness`) and
bookkeeping (`id`, `_path`, `seed`, `show`, `training time`, `baseline_x0_*`).
`generations`, `environment`, `operation`, `mvg_ops`, `mvg_switch_interval`,
`prunning_phase`, `popsize`, `balanced_fitness`, `early_stopping` and all
growth-model settings are identical. All MVG final-generation champions were
scored on `and` (1500/20 = 75 blocks, the last one even), the same task as FG.
Code (`train.py`, `train_backend.py`, `optimizers.py`, `ka_task.py`) and
`run_experiment.yaml` were last modified 2026-08-06/07, before 4 of the 5 MVG
runs and all 5 new FG runs; the first MVG run (1786053806, 2026-08-06) predates
the Aug 7 `train.py` edit, but its config matches the others on every training
parameter.

The modularity numbers in `MODULARITY_RERUN_RESULTS.md` use the old
200-generation FG runs and should not be read as a matched comparison. The
matched set is analysed in the next two entries (added 2026-09-14).

## FG vs MVG over training, budget-matched set (2026-09-14)

Per-generation champions of all 10 runs were recovered by deterministic replay:

```
conda run --no-capture-output -n ndp python -u experiments_paper/retina/replay_archive.py
```

It writes `saved_models/<run>/replay_champions.npz` and log
`logs/replay_archive.log`. All 10 archives are `VERIFIED=True`: the best-ever
fitness and population mean match `logger.csv` for every generation, and the
final genome matches `solution_best.npy`.

Figures:

```
python experiments_paper/retina/matched_figures.py grid --out-dir <dir>
python experiments_paper/retina/matched_figures.py progression --out-dir <dir>
```

The progression figure shows the mean ± 1 SD of the 5 runs per arm, for each
generation's champion. FG is sampled at 60 points, MVG at the end of each of
its 38 AND epochs. Every sampled brain, regrown, reproduces its replay fitness
exactly (`logs/progression_fig.log`). Copies are in UCL_thesis
`latex_figures/NDP/KA/` (`brains_grid_leftright_matched.png`,
`progress_fg_vs_mvg_matched.png`).

- **Accuracy (AND): a complete plateau in both arms from the first few
  generations.** Both sit at ≈ 0.844 for 1500 generations (MVG mean 0.846, FG
  0.8438).
  - Every value away from 0.8432 is rounding (next entry).
  - The FG dip to 0.834 in the last ~40 generations is one run, 1789305630: its
    last 43 generation champions score 0.78–0.80. Its saved best-ever brain is
    0.8438.
- **Density:** both arms settle by ~generation 450, FG ≈ 55%, MVG ≈ 42%, with
  ±1 SD bands (≈ 17–90%) that overlap almost completely.
- **lr_r:** both settle at 0.1–0.2 (FG ≈ 0.12, MVG ≈ 0.17), bands fully
  overlapping.
- **MVG shows no effect here.** In the Kashtan–Alon reproduction MVG gave
  clearly more modular networks than FG: Q_m 0.245 vs 0.025, p ≈ 0.02–0.03
  (UCL_thesis `kashtan_alon/RESULTS.md`, Run 5). In NDP on this task, FG and MVG
  give the same accuracy, the same function and overlapping density/lr_r. The
  next entry shows why MVG has nothing to select for.

## What the budget-matched brains compute (2026-09-14)

```
conda run -n ndp python experiments_paper/retina/counting_analysis.py all
```

Every number below comes from that command.

### 1. Every brain is exactly a majority vote

All 10 final brains, FG and MVG, 12 to 80 neurons, compute: **1 if ≥ 5 of the 8
inputs are on, 0 if ≤ 3.** At exactly 4 on, the output is 0 in exact arithmetic.
Checks on all 10:

- **Same count → same output.** Outputs for patterns with the same number of
  inputs on are identical (popcount ≠ 4; ≤ 1e-16 at step 1). This already holds at propagation step 1,
  where outputs are still ≈ ±0.93–0.96, not saturated.
- **f(−x) = −f(x)** to ≤ 7e-15.
- **At exactly 4 on, |output| ≤ 1e-12.** Flipping a 4-on pattern gives another
  4-on pattern, so count-invariance and oddness together force 0.
- **The reported scores are rounding.** Treating those ties as 0 gives **every
  brain 0.8432**. The reported 0.8438 / 0.8540 depend on floating-point
  rounding at the ties.
  - Renumbering hidden neurons, a mathematically identical network, moves
    `retina_fitness` to 0.7737–0.7936 for the 40-, 30- and 80-neuron brains.
  - The 12- and 16-neuron brains stay at 0.8540 under renumbering, but not under
    a batched matrix product (0.8274 / 0.8202, checked earlier the same day).

**Ceilings** (`ceilings`), balanced accuracy over all 256 patterns:

| Rule family | AND | OR |
|---|---|---|
| any rule on popcount alone | 0.8432 | 0.8212 |
| any sign-symmetric classifier, f(−x) = −f(x) | 0.9796 | 0.8429 |
| popcount rules NDP can express (both limits: 0 at popcount 4) | **0.8432** | **0.7657** |

The best NDP-expressible rule is the majority vote for **both** goals. Counting
cannot solve the task: `11001100` (label 1) and `10101010` (label 0) both have
4 on.

### 2. Why: NDP cannot tell its inputs apart, and its rollout has no bias

- **No input identity.** All 8 inputs share one role embedding
  (`train_backend.py:62-67`; only 2 role vectors, `:843`) and start with the
  same neighbourhood (the output). The growth, embedding-transform and weight
  MLPs are applied identically everywhere, and a child is wired to its parent
  plus the parent's whole neighbourhood (`NDP.py:407-411`). So each input grows
  an exact copy of every other input's subtree, and the brain is invariant
  under any input permutation. This is an argument from the setup, confirmed on
  the 10 brains; it is not a formal proof for every genome.
- **Growth history shows in the wiring.** Every hidden neuron is wired to exactly
  1 input (a descendant of that input) or to all 8 (a descendant of the output):
  never 2–7, in any brain. The 8 larger brains split into **8 greedy Newman
  communities, one input each**. The 12-neuron brain forms 1 community, the
  16-neuron brain 2 (4 + 4, Q = 0.006). These are input lineages, not a
  left/right split.
- **No bias.** The rollout is `s ← tanh(Wᵀs)` from zero with inputs re-clamped
  and no bias (`NDP.py:230-240`), so f(−x) = −f(x) for any weights. The target
  is not sign-symmetric (`11001100` and `00110011` are both positive).
- **Shared wirings.** The 10 brains have only 5 distinct edge sets. FG
  1789303833, 1789304372, 1789305630 and MVG 1788817038, 1788866451, 1788871442
  are one identical 40-neuron wiring.

### 3. Rounding is the only way out, and evolution does use it during OR epochs

`or-champs`: the best OR-epoch champion of every MVG run scores **0.8429**
under `retina_fitness`, against the exact NDP cap of **0.7657**. With 4-on ties
set to 0, every one scores 0.7657. Renumbering hidden neurons drops them to
0.79–0.80.

Two of the five grew input-asymmetrically, which only rounding at a growth
decision sitting on its threshold can cause:
- **1788817038, generation 112:** neurons per cycle 9 → 12 → …, input degrees
  79/79/79/79/111/111/79/79;
- **1788866451, generation 32:** input degrees 47/47/63/63/47/47/47/47.

None of this survives into a final brain.

### 4. A perfect brain exists, but NDP cannot grow it

UCL_thesis `experiments/experiment_1/oracle.py --task retina` hand-wires a
recurrent tanh brain for this same stand-in task. It has 4 AND detectors on
different input pairs, 2 OR combiners, an output AND, and biases. It scores
**1.000** (stage 1; re-run 2026-09-14). NDP cannot express it: each detector
must single out 2 specific inputs, and its threshold is a bias. **The ≈ 0.84
plateau is a representability limit of this NDP set-up, not a search
failure.**

### 5. Consequences

- Only functions of how many inputs are on are reachable. That excludes both
  retina variants, KA's task, and even `x0 AND x1`.
- **MVG has nothing to select for.** The majority vote is the best expressible
  brain for AND and for OR at once, so a goal switch never rewards a different
  brain.
- Fitness above 0.8432 (AND) or 0.7657 (OR) in any log or figure of this study is
  a rounding artefact.
- Untested ways to lift the limit:
  - per-input embeddings or positional codes;
  - a bias term in the rollout;
  - symmetry breaking in the seed graph.

### 6. Elitism

- **All retina runs were elitist.** `CMA_elitist: true` in every saved
  `KA_retina*` config (30 runs).
- **Elitism was varied only on LunarLander** (`fluffy_experiments.md`):
  - Run 6 (elitist, σ_init 1, popsize 512) froze its best at −67 from
    generation 70 while the population mean kept improving. That was diagnosed as
    elitist premature convergence.
  - Run 8 added `--no-elitism` (`train.py:149`, commit b23abb3). σ kept rising
    (0.80 → 0.90) and the best improved to −72.93 by generation 60, then froze
    again. It was killed at ~110 generations, unsolved.
- **Not retried on retina, and it could not help there.** The cap in §1–§2 holds
  for every genome, so no optimiser setting can move it.
