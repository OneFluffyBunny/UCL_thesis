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

These are built and open for review but **not yet copied into this folder**.

### `paper_10runs_grid.png` — the 10 final champion circuits

| | |
|---|---|
| **Shows** | Last-AND-epoch champion of all 10 paper runs (5 FG top, 5 MVG bottom), nodes coloured by purity flow (blue = reads left retina, red = right) and ordered within each layer by that value. Per-panel caption: arm, seed, accuracy on AND; then Q, Q_m, r, purity. Legend is a blue→red gradient bar. |
| **Source data** | `kashtan_alon/runs_purity/` (needs `_brains.npz`; `runs/` predates archiving — same 10 runs, same seeds and parameters) |
| **Generator** | `kashtan_alon/analysis/paper_grid.py`, drawing via `highlight_modules.draw_net_purity()` |

```bash
conda run -n lndp python kashtan_alon/run_paper.py            # the 10 runs, hours
conda run -n lndp python kashtan_alon/analysis/paper_grid.py  # the grid
```

Notes:
* `SHOW_VALUES` in `paper_grid.py` prints each node's flow value inside the node
  — off for the figure, turn on to verify the colouring.
* `NODE_SCALE` multiplies marker area; it is passed to
  `highlight_modules.draw_net_purity(node_scale=...)`, whose default of 1.0
  keeps every other caller unchanged. `draw_net_purity`'s x-limits must clear a
  node *radius* past the outermost retina pixel (±4.1) or large nodes are sliced.
* Accuracy is recomputed for the brain actually drawn, not read from
  `result.json` (which describes the final-generation champion — mid-OR-epoch
  for every MVG seed). See the goal-matching note in the ablation section above.

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
