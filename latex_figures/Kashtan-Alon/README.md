# Kashtan-Alon figures

Figures from `kashtan_alon/` (results: `kashtan_alon/RESULTS.md`). The run archives are
not committed, so each entry lists the training command as well as the generator. Run
everything from the repository root in the `lndp` environment. Generators live in
`kashtan_alon/analysis/`.

## Training

```bash
python kashtan_alon/run_paper.py --n-seeds 5                    # capped runs -> kashtan_alon/runs/
python kashtan_alon/analysis/fg_mvg_purity.py --n-seeds 5       # same runs with every champion archived -> runs_purity/
python kashtan_alon/run_ablation_no_fanin.py --n-seeds 5        # uncapped runs -> runs_no_fanin/
python kashtan_alon/analysis/dense_replay.py --arm fg           # seed 0, per-generation logging -> runs_dense/
python kashtan_alon/analysis/dense_replay.py --arm mvg
python kashtan_alon/analysis/dense_replay.py --verify           # must print VERIFIED
# ablation replays: the same three commands with --no-fanin (-> runs_dense_no_fanin/)
```

Each seed and arm takes ~10 minutes on a laptop CPU. Runs are seeded and
bit-reproducible, and logging draws no randomness, so a dense replay is identical to
the archived run (the `--verify` step checks it).

## Figures

| file | shows | generator |
|---|---|---|
| `switch_window_seed0.png` | FG seed 0 vs MVG seed 0, every generation, in windows [100, 300], [1000, 1200] and [10000, 10200]: champion accuracy, population mean accuracy, champion purity | `switch_window.py` |
| `switch_window_seed0_no_fanin.png` | the same, uncapped | `switch_window.py --no-fanin` |
| `fg_vs_mvg_purity.png` | mean ± 1 SD over 5 seeds, 25,000 generations: champion accuracy on the live goal, and circuit purity (bold: 500-generation moving average) | `fg_mvg_purity.py --plot-only` |
| `fg_vs_mvg_purity_no_fanin.png` | the same, uncapped | `fg_mvg_purity.py --no-fanin --plot-only` |
| `paper_10runs_grid.png` | the last AND-epoch champion of all 10 capped runs (FG top, MVG bottom), neurons coloured by which retina half they read, captions with Q, Q_m, r and purity | `paper_grid.py` |
| `paper_10runs_grid_no_fanin.png` | the same, uncapped | `paper_grid.py --no-fanin` |
| `newman_communities_mvg_seed1.png` | one capped MVG brain (seed 1, generation 24,970) coloured by greedy Newman-Q communities | `newman_vs_binary.py` |

Caption numbers are printed by each generator.

## Notes for captions

- **Which network.** Grids and the communities figure show the last champion archived
  during an AND epoch (FG generation 24,999, MVG 24,970), with accuracy recomputed on
  AND. A final-generation MVG champion would be an OR network scoring ~0.50 on AND.
- **No dip at the switch in the accuracy curves.** They plot the champion, a maximum
  over 600 individuals; late in a run a few individuals are already perfect on the
  incoming goal. The switch cost shows in the population mean (middle row of the
  switch-window figures).
- **Switch-window numbers.** 90% recovery of the population mean takes 4.1–4.4
  generations uncapped and 4.2–4.6 capped; uncapped recovers to a higher level (0.97 vs
  0.89 in the early window). The dense windows are set in `dense_replay.py` and the
  plotted ones in `switch_window.py`; keep them in sync.
- **Uncapped curves.** MVG champion accuracy 0.763 → 1.000 and FG 0.763 → 0.970, while
  final purity is 0.567 (MVG) and 0.316 (FG), against 0.929 and 0.561 capped.
- **The communities figure.** Seed 1 is a literal two-module network (purity and r are
  1.00, no left-right edge below the output neuron), but greedy Newman Q returns four
  communities: it splits the left half in two and makes the path to the output a module
  of its own. Its Q_m (+0.26) is also lower than that of seed 0 (+0.46), whose purity is
  only 0.81. Q has to search for a partition; the task already names the one that
  matters, which is why r and purity are reported.
- `fg_mvg_purity.py --no-fanin` refuses to train: it would write capped runs into the
  ablation folder. `run_ablation_no_fanin.py` trains the ablation.
