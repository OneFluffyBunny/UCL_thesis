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

## Pending review

These are built and open for review but **not yet copied into this folder**.

### `paper_10runs_grid.png` — the 10 final champion circuits

| | |
|---|---|
| **Shows** | Final-generation champion network of all 10 paper runs (5 FG top, 5 MVG bottom), nodes coloured by purity flow (blue = reads left retina, red = right) and ordered within each layer by that value. Per-panel caption: arm, seed, final accuracy and the goal it was scored on; then Q, Q_m, r, purity. Legend is a blue→red gradient bar. |
| **Source data** | `kashtan_alon/runs/` |
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
* Accuracy comes from `final_fit` / `final_op` in each run's `_result.json`, so
  MVG panels report the goal (OR) the schedule happened to leave live at
  generation 24,999.
