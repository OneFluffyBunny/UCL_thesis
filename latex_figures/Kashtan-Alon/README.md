# Kashtan-Alon figures — provenance

Every figure in this folder, and exactly how to rebuild it. Nothing here is
regenerable from the repo alone: the run archives (`kashtan_alon/runs*/`) are
gitignored, so **the training command is part of the provenance** and is listed
for each figure.

Run everything from the repo root with `conda run -n lndp python ...`
(the terminal does not persist conda activations).

Every generator below lives in `kashtan_alon/analysis/` and is tracked. They were
`scratch_*.py` until 2026-09-10, i.e. untracked by repo convention and one
`git clean` away from making these figures unrebuildable; they were promoted for
exactly that reason. Anything new that draws a thesis figure belongs there too.

---

## Approved figures

### `switch_window_seed0.png` — per-generation view of the goal switch

| | |
|---|---|
| **Shows** | FG seed 0 vs MVG seed 0, every generation logged, in three windows: `[100,300]`, `[1000,1200]`, `[10000,10200]`. Rows: champion accuracy, population-mean accuracy, champion circuit purity. |
| **Point it makes** | The champion accuracy sawtooth is visible only very early; by gen 1000 the population has split into per-goal specialists and a small off-goal reservoir pins `max()` at ceiling. The switch cost stays fully visible in the population mean. |
| **Source data** | `kashtan_alon/runs_dense/` |
| **Generator** | `kashtan_alon/analysis/switch_window.py` |

Rebuild:

```bash
# 1. replay both arms with per-generation logging (~5 min each, run in parallel)
conda run -n lndp python kashtan_alon/analysis/dense_replay.py --arm fg
conda run -n lndp python kashtan_alon/analysis/dense_replay.py --arm mvg

# 2. confirm the replay reproduces the archived run exactly (must print VERIFIED)
conda run -n lndp python kashtan_alon/analysis/dense_replay.py --verify

# 3. draw
conda run -n lndp python kashtan_alon/analysis/switch_window.py
```

Notes:
* The dense windows live in `dense_replay.py:WINDOWS`; the plotted windows in
  `switch_window.py:WINDOWS`. **Keep them in sync** — the plotter reads whatever
  the replay logged.
* Per-generation logging uses `train.py --dense-log lo:hi,lo:hi`. Logging draws
  no randomness, so a dense run is bit-identical to a normal one; step 2 checks
  this against `runs_purity/` rather than trusting it.
* Recovery statistics print to stdout (they were removed from the title).

Approved 2026-09-10. The copy in this folder came from
`kashtan_alon/runs_dense/switch_window_seed0.png`.

### `fg_vs_mvg_purity.png` — accuracy and purity across evolution

| | |
|---|---|
| **Shows** | Mean ± 1 SD over 5 seeds per arm, 25,000 generations. Left: accuracy of the champion on the goal live that generation. Right: left/right circuit purity. Bold line = 500-generation moving average (`--smooth 50` log points × `--log-interval 10`), faint line raw. |
| **Source data** | `kashtan_alon/runs_purity/` |
| **Generator** | `kashtan_alon/analysis/fg_mvg_purity.py` |

```bash
conda run -n lndp python kashtan_alon/analysis/fg_mvg_purity.py --n-seeds 5   # ~10.5 min/seed/arm
conda run -n lndp python kashtan_alon/analysis/fg_mvg_purity.py --plot-only   # redraw only
```

⚠️ **Known caveat.** The accuracy panel plots `best_fit`, a max over 600
individuals. It shows no drop at a goal switch, and that is not a smoothing
artefact: at a late switch ~4 individuals are already perfect on the incoming
goal while the ~417 perfect on the outgoing one collapse, so `max()` reaches
past the collapse. The switch cost is visible in the *population mean*
(`mean_fit`), which is the middle row of `switch_window_seed0.png`. An earlier
version split this panel by goal; the two curves coincided by construction, for
the same reason, and the split was removed.

The MVG advantage in the accuracy panel was audited (2026-09-10) and survives:
the two goals are equally hard (best constant output 0.750 on both), MVG's own
curve differs by <0.01 between AND-phase and OR-phase rows, and goal-matched
best-on-AND champions give MVG 0.975 vs FG 0.904 raw, 0.967 vs 0.829 balanced —
MVG ahead on every seed under both metrics. See `kashtan_alon/RESULTS.md`.

Approved 2026-09-10. The copy in this folder came from
`kashtan_alon/runs_purity/fg_vs_mvg_purity.png`.

### `newman_communities_mvg_seed1.png` — why the planted-partition metrics

| | |
|---|---|
| **Shows** | One capped MVG brain (seed 1, last AND-epoch champion, generation 24,970) drawn with the **old Newman-Q community colouring**: a translucent blob per greedy community, within-community edges in the community colour, between-community edges thick and red. Nodes are laid out by side (left-reading left, right-reading right) so the true split is visible; only the colouring is Q's. |
| **Point it makes** | The brain is a literal two-module network — circuit purity and `r` are both exactly 1.00, and no left–right edge exists anywhere below the output neuron. Greedy Newman Q, on the same graph, returns **four** communities: it cuts the left module in two and makes the integrator spine a module of its own, flagging 7 of 33 edges as "between-module" when almost all of them stay on one side. Its Q_m (+0.26) is also *lower* than seed 0's (+0.46), whose purity is only 0.81 — the metric ranks the perfectly split brain below the imperfect one. The case is not "Q is wrong"; it is that Q must **search** for a partition (NP-hard, budget-limited, and not aimed at the task's split), whereas the task already names the partition. |
| **Source data** | `kashtan_alon/runs_purity/` |
| **Generator** | `kashtan_alon/analysis/newman_vs_binary.py`, drawing via `highlight_modules.draw_net()` |

```bash
conda run -n lndp python kashtan_alon/analysis/newman_vs_binary.py           # seed 1
conda run -n lndp python kashtan_alon/analysis/newman_vs_binary.py --seed 3  # another
```

The caption's numbers come from the generator's own stdout, which for the
approved figure reads:

```
retina_mvg_raw_seed1: gen 24970 acc 0.9688 Q=0.502 Q_m=+0.260
  (q_rand=0.442 q_max=0.674) purity=1.000 r=+1.000 4 communities, 7/33 cross edges
  community 1: [4, 5, 6, 7, 11, 12, 13, 14, 17]
  community 2: [2, 3, 8, 10, 18, 19]
  community 3: [0, 1, 9, 15]
  community 4: [16, 20, 21, 22]
```

Nodes 0–7 are the retina (0–3 left, 4–7 right), 22 is the output. Community 1 is
the whole right half plus the output's parent; communities 2 and 3 are the left
half **cut in two**; community 4 is the integrator spine. That is the figure's
entire argument in four lines.

Notes:
* `draw_net()` gained an optional `pos=` argument for this figure; its default
  is unchanged, so every other caller draws exactly as before.
* The figure is deliberately bare — title `Newman Q decomposition`, the legend,
  and the two retina-half labels, nothing else. Every number it is *about* (Q,
  Q_m, purity, `r`, cross-edge count, where the halves meet, the community
  memberships) is **printed to stdout** by the generator and belongs in the
  caption prose. An earlier version put all of it in the title; it was
  unreadable, and the numbers are the caption's job.
* The left/right divider is drawn only as far up as it is **measured** to hold
  (the first block containing a left↔right edge), so the figure stays honest on
  a seed whose split is not perfect.

Approved 2026-09-11. The copy in this folder came from
`kashtan_alon/runs_purity/newman_communities_mvg_seed1.png`.

### `paper_10runs_grid.png` — the 10 final champion circuits

| | |
|---|---|
| **Shows** | Last-AND-epoch champion of all 10 paper runs (5 FG top, 5 MVG bottom), nodes coloured by purity flow (blue = reads left retina, red = right) and ordered within each layer by that value. Per-panel caption: arm, seed, accuracy on AND; then Q, Q_m, r, purity. Legend is a blue→red gradient bar. |
| **Point it makes** | The MVG row is visibly two-coloured and the FG row is visibly mixed, seed by seed, with no metric needed to see it. Three of five MVG seeds reach `r = +1.00, purity = 1.00` — no left↔right edge anywhere below the output. This is the **direct counterpart of `paper_10runs_grid_no_fanin.png`**: same layout, same generator, cap removed, and there the MVG row loses the colour separation. |
| **Source data** | `kashtan_alon/runs_purity/` (needs `_brains.npz`; `runs/` predates archiving — same 10 runs, same seeds and parameters) |
| **Generator** | `kashtan_alon/analysis/paper_grid.py`, drawing via `highlight_modules.draw_net_purity()` |

```bash
conda run -n lndp python kashtan_alon/run_paper.py            # the 10 runs, hours
conda run -n lndp python kashtan_alon/analysis/paper_grid.py  # the grid
```

The per-panel numbers come from the generator's own stdout, which for the
approved figure reads:

```
retina_fg_raw_seed0:  gen 24999 acc 0.8711 | Q=0.315 Q_m=-0.143 r=+0.394 purity=0.344
retina_fg_raw_seed1:  gen 24999 acc 0.9570 | Q=0.415 Q_m=+0.228 r=+0.749 purity=0.656
retina_fg_raw_seed2:  gen 24999 acc 0.8906 | Q=0.380 Q_m=-0.007 r=+0.560 purity=0.571
retina_fg_raw_seed3:  gen 24999 acc 0.8867 | Q=0.401 Q_m=-0.036 r=+0.680 purity=0.635
retina_fg_raw_seed4:  gen 24999 acc 0.9141 | Q=0.391 Q_m=+0.083 r=+0.640 purity=0.598
retina_mvg_raw_seed0: gen 24970 acc 1.0000 | Q=0.501 Q_m=+0.459 r=+0.828 purity=0.810
retina_mvg_raw_seed1: gen 24970 acc 0.9688 | Q=0.502 Q_m=+0.260 r=+1.000 purity=1.000
retina_mvg_raw_seed2: gen 24970 acc 1.0000 | Q=0.434 Q_m=+0.333 r=+0.850 purity=0.857
retina_mvg_raw_seed3: gen 24970 acc 0.9688 | Q=0.508 Q_m=+0.240 r=+1.000 purity=1.000
retina_mvg_raw_seed4: gen 24970 acc 0.9375 | Q=0.484 Q_m=+0.237 r=+1.000 purity=1.000
```

Every FG panel is generation 24,999 and every MVG panel 24,970: that is the last
generation each arm had AND as its live goal, not a different cut for each arm.

Notes:
* `SHOW_VALUES` in `paper_grid.py` prints each node's flow value inside the node
  — off for the figure, turn on to verify the colouring.
* `NODE_SCALE` multiplies marker area; it is passed to
  `highlight_modules.draw_net_purity(node_scale=...)`, whose default of 1.0
  keeps every other caller unchanged. `draw_net_purity`'s x-limits must clear a
  node *radius* past the outermost retina pixel (±4.1) or large nodes are sliced.
* Accuracy is recomputed for the brain actually drawn, not read from
  `result.json` (which describes the final-generation champion — mid-OR-epoch
  for every MVG seed). See the goal-matching note in the ablation section below.

Approved 2026-09-11. The copy in this folder came from
`kashtan_alon/runs/paper_10runs_grid.png`.

---

## The fan-in ablation — the same three figures, cap removed

The ablation is **one line**: `NetConfig(layers=layers, fan_in=())` in place of
`NetConfig(layers=layers)`. `model.py:_fan_in()` then falls back to "the whole
previous layer", so every neuron may read every neuron below it. Everything else
— task, pop 600, 25,000 generations, elite 150, `Pc`/`Pm` 0.5, switch interval
20, ±1 weights, threshold units, seeds 0–4 — is byte-identical to the capped
runs. Note the cap is read in **two** places, so removing it changes both:
`model.py:91` sets the initial fan-in (`k = round(0.5 × cap)`, 2 edges/neuron
capped vs 4 uncapped) and `ga.py:95` is the ceiling the add-edge mutation may
not exceed. It is not "cap removed at selection time only".

Training (all three figures below read from runs produced by these):

```bash
# the 10 ablation runs (5 FG + 5 MVG), ~10.5 min/seed/arm
conda run -n lndp python kashtan_alon/run_ablation_no_fanin.py --n-seeds 5

# the dense per-generation replay of seed 0, both arms
conda run -n lndp python kashtan_alon/analysis/dense_replay.py --arm fg  --no-fanin
conda run -n lndp python kashtan_alon/analysis/dense_replay.py --arm mvg --no-fanin
conda run -n lndp python kashtan_alon/analysis/dense_replay.py --verify --no-fanin
```

⚠️ `fg_mvg_purity.py --no-fanin` **refuses to train** (it exits unless
`--plot-only` is also given). Its preset builds a *default* `NetConfig`, so
training under that flag would quietly write fan-in-**capped** runs into the
ablation's directory. `run_ablation_no_fanin.py` owns the training.

### `switch_window_seed0_no_fanin.png`

Same as `switch_window_seed0.png`, ablation runs. Source data
`kashtan_alon/runs_dense_no_fanin/`; generator
`kashtan_alon/analysis/switch_window.py --no-fanin`.

**Point it makes:** the switch dynamics are unchanged by the cap. Population
mean still craters to ~0.50 at every switch, and 90% recovery takes 4.1–4.4
generations uncapped vs 4.2–4.6 capped. What the extra wiring buys is the
recovery *level* (0.97 vs 0.89 in the early window), not the recovery *speed*.

### `fg_vs_mvg_purity_no_fanin.png`

Same as `fg_vs_mvg_purity.png`, ablation runs. Source data
`kashtan_alon/runs_no_fanin/`; generator
`kashtan_alon/analysis/fg_mvg_purity.py --no-fanin --plot-only`.

**Point it makes:** MVG `best_fit` 0.763 → 1.000 and FG 0.763 → 0.970 (both
above their capped counterparts), while purity ends at 0.567 (MVG) and 0.316
(FG) against 0.929 / 0.561 capped. Better solutions, much less modular.

### `paper_10runs_grid_no_fanin.png`

Same as the capped grid, ablation runs. Source data `kashtan_alon/runs_no_fanin/`;
generator `kashtan_alon/analysis/paper_grid.py --no-fanin`.

**Point it makes:** ablated MVG scores a perfect 1.00 on AND on all five seeds
while per-brain Q_m collapses to a ~+0.02 mean (two seeds negative).

**Goal-matching:** every panel in both grids is the **last champion archived
during an AND epoch** — generation 24,999 for FG, 24,970 for MVG — scored on
AND. `25000/20 = 1250` blocks leaves generation 24,999 inside an OR epoch for
every MVG seed, so a final-generation champion would be an OR specialist
scoring 0.50 on AND (below the 0.75 a constant output gets). `last_on_goal()`
takes the *last* such champion, not the best: a maximum over ~1,250 archived
champions is a cherry-pick and lands at a different generation in every run.

Approved 2026-09-11. Copies came from
`kashtan_alon/runs_dense_no_fanin/`, `kashtan_alon/runs_no_fanin/` and
`kashtan_alon/runs/` respectively.

---

## Pending review

Nothing pending. All seven figures above are approved and filed here.

