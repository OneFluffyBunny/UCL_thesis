# kashtan_alon — results

Reproduction of Kashtan & Alon (2005): layered threshold networks on their retina task,
fixed goal (FG, L AND R) vs modularly varying goals (MVG, AND ↔ OR every 20
generations). Setup and sources: `README.md` and `PAPER_SPEC.md`.

**Paper's claim:** under MVG the normalised modularity Q_m reaches 0.35 ± 0.02; under
FG it stays at 0.15 ± 0.02.

## Result

Five seeds per arm, capped (the paper's fan-in limits) and uncapped (an ablation of
our own). Every number is for the **last champion archived during an AND epoch**
(generation 24,999 for FG, 24,970 for MVG), scored on AND; mean ± SD over seeds 0–4:

| condition | arm | accuracy (AND) | Q | Q_m | r | purity | edges | density |
|---|---|---|---|---|---|---|---|---|
| capped | FG | 0.90 ± 0.03 | 0.38 ± 0.04 | +0.02 ± 0.14 | +0.60 ± 0.14 | 0.56 ± 0.13 | 40 | 38% |
| capped | **MVG** | 0.97 ± 0.03 | **0.49 ± 0.03** | **+0.31 ± 0.09** | **+0.94 ± 0.09** | **0.93 ± 0.09** | 36 | 34% |
| no cap | FG | 0.97 ± 0.02 | 0.22 ± 0.03 | −0.07 ± 0.10 | +0.30 ± 0.07 | 0.31 ± 0.11 | 69 | 65% |
| no cap | MVG | 1.00 ± 0.00 | 0.29 ± 0.04 | +0.02 ± 0.03 | +0.51 ± 0.08 | 0.56 ± 0.11 | 55 | 52% |

Q: Newman modularity of a greedy partition. Q_m = (Q − Q_rand) / (Q_max − Q_rand) with
1,000 degree-preserving randomisations. r: assortativity at the planted left/right
split. Purity: `qmetrics.circuit_purity`, how completely each neuron's input traces to
one retina half. Per-seed values are printed by `analysis/paper_grid.py` (capped) and
`analysis/paper_grid.py --no-fanin`.

1. **MVG > FG reproduces with the paper's constraint.** The Q_m gap (+0.29 here, +0.22
   on final-generation champions) matches the paper's +0.20, though both arms sit
   lower than the paper's values. On final-generation champions the Q_m difference is
   significant (one-sided Mann-Whitney p = 0.016; `analysis/ablation_table.py` prints
   the per-seed values). Three of five capped MVG champions have no left-right edge
   below the output neuron (r = purity = 1.00).
2. **The fan-in cap is a modularity constraint, not a performance constraint.** Without
   it both arms solve the task better, and MVG still beats FG, but by less on every
   metric (Q_m gap 0.28 → 0.09, r 0.34 → 0.21, purity 0.37 → 0.25). Uncapped MVG is as
   pure as capped FG. Only the combination of MVG and scarce wiring gives the
   qualitatively modular networks.
3. **MVG is also more accurate on the matched goal.** Goal-matched over the last 2,000
   generations: capped 0.959 vs 0.904, uncapped 0.999 vs 0.970 (`ablation_table.py`).
   The two goals are equally hard (a constant output scores 0.750 on both).
4. **Without the cap, Q_m loses resolution.** Q and Q_rand converge (0.287 vs 0.285 for
   uncapped MVG), because at 52–65% density every partition has many crossing edges
   and a degree-preserving null already looks clustered. Purity and r still separate
   the uncapped arms. Report purity and r for dense networks; Q_m is calibrated for the
   sparse regime (34–38%) the paper's networks occupy.

Other observations from the per-generation logs (`<run>_log.csv`, `op` column):
- **Uncapped networks solve much faster.** Uncapped MVG reaches 1.00 on AND within 400
  generations on every seed; capped MVG reaches it on 2 of 5 seeds in 25,000; capped FG
  never exceeds 0.95 on 4 of 5.
- **MVG runs sparser because FG keeps adding edges, not because MVG removes them.**
  Both uncapped arms start at 50.0% density; FG rises to 64% and MVG stays near 50–53%.
  Nothing in the fitness prices an edge, so a likelier reason is that under MVG a new
  edge must pay off on both goals to survive a switch.
- **A one-half detector scores 0.750 on both goals.** Under MVG it pays no cost at a
  switch, so MVG cannot punish it; its mechanism only engages once a network computes
  both halves.
- **The switch cost is visible in the population mean, not the champion.** After a
  late switch the mean drops to ~0.50 while a few individuals are already perfect on
  the new goal, so the champion does not drop. 90% recovery of the mean takes 4.1–4.6
  generations, capped or not (`analysis/switch_window.py`).

## Reproduce

From the repository root:

```bash
python kashtan_alon/run_paper.py --n-seeds 5 --fresh                  # capped, 10 runs -> runs/
python kashtan_alon/analysis/fg_mvg_purity.py --n-seeds 5             # same runs, archiving every champion -> runs_purity/
python kashtan_alon/run_ablation_no_fanin.py --n-seeds 5              # uncapped, 10 runs -> runs_no_fanin/
python kashtan_alon/analysis/paper_grid.py [--no-fanin]               # the table above (per seed) + figure
python kashtan_alon/analysis/ablation_table.py                        # final-champion Q_m, goal-matched accuracy
python kashtan_alon/analysis/dense_replay.py --arm fg|mvg [--no-fanin]  # per-generation replay of seed 0
python kashtan_alon/analysis/dense_replay.py --verify [--no-fanin]    # must print VERIFIED
```

Each run is ~10 minutes per seed per arm on a laptop CPU. Runs are seeded
(`np.random.default_rng(seed)`) and bit-reproducible; `dense_replay.py --verify`
checks this against the archive. Figure commands: `latex_figures/Kashtan-Alon/README.md`.

The ablation is one line, `NetConfig(layers=layers, fan_in=())`. The cap is read in two
places: the initial fan-in (`k = round(0.5 × cap)`, 2 edges per neuron capped vs 4
uncapped, hence the different starting densities) and the ceiling for the add-edge
mutation.

| parameter | value | source |
|---|---|---|
| architecture | retina(8) → 8 → 4 → 2 → 1 | paper |
| weights / units | ±1; hard threshold, fires iff Σw·x + bias > 0 | paper |
| fan-in cap | 3, 3, 3, 2 (none in the ablation) | paper |
| population / generations | 600 / 25,000 | paper |
| elite | 150 of 600 copied unchanged | reconstructed by analogy |
| crossover | p = 0.5, per destination neuron (its incoming column and threshold from one parent) | "neuron-level" in the paper; mechanism reconstructed |
| mutation | p = 0.5 per genome, one edit: add edge, remove edge, flip sign, or threshold ±1 in [−3, 3] | p from the paper; operator set reconstructed (the supplement was unavailable) |
| initial density | 0.5 of the fan-in cap per neuron | our choice |
| fitness | fraction correct over all 256 patterns | paper samples 100 per generation |
| goals | FG L AND R; MVG AND ↔ OR every 20 generations | paper |
| Q_m | 1,000 randomisations; Q_max by a degree-preserving hill climb (6 × 250 steps) | paper re-evolves populations towards Q instead |
| complexity penalty | none | paper: 0.01 per neuron above 13 (not implemented) |

## History

Runs 1–3 ran a reimplementation of Clune et al. (2013), not Kashtan & Alon (different
task, weights, units and GA), and gave a null once scored with Q_m. The code was
rewritten to the paper on 2026-08-03; everything above uses the rewritten code.
