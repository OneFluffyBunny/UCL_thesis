# NDP × retina task — results

The Neural Developmental Program (NDP) grows a network from a small seed graph; CMA-ES
evolves the growth rules. This study runs it on a retina task, fixed goal (FG) vs
modularly varying goals (MVG), to see whether goal switching makes the grown networks
modular. Trained runs are written to `saved_models/<run id>/` (not committed).

Environment: the `ndp` conda environment (Python 3.10, torch 2.12 CPU, numpy 2.2.6,
cma 4.4.4, gymnasium 1.3.0; `NDP/requirements.txt`). Commands run from `NDP/`. The
analysis scripts import `qmetrics/` and `experiments/shared_brain_metrics.py` from the
enclosing repository.

## Task

`ka_task.py`: 8 bipolar inputs, one output, all 256 patterns. Target
`left AND right` (or OR), each side `(x0 & x1) | (x2 & x3)` on its 4 inputs. This is
the thesis's stand-in retina (`experiments/shared_tasks.py`, task `retina`), not
Kashtan & Alon's object rule. Only 19% of patterns are positive, so fitness is balanced
accuracy (`--balanced-fitness`).

## Study: 5 FG + 5 MVG runs, 1,500 generations each

```bash
FG:  python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 1500 --popsize 128 --balanced-fitness --no-early-stopping --operation and --snapshot
MVG: python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 1500 --popsize 128 --balanced-fitness --no-early-stopping --mvg --mvg-ops and,or --mvg-switch-interval 20 --snapshot
```

`experiments_paper/retina/top_up_seeds.py --target 5` launches whatever is missing,
one run at a time. Each run draws its own seed (recorded in `config.yml`), and CMA-ES
is elitist (`CMA_elitist: true`). All configuration keys except the MVG flag and
bookkeeping are identical across the 10 runs. 1,500 / 20 = 75 goal epochs, so every
MVG run ends on AND, the same goal as FG.

| arm | run id | seed | final accuracy (AND) |
|---|---|---|---|
| FG | 1789303257 | 3410590 | 0.8438 |
| FG | 1789303833 | 7345678 | 0.8438 |
| FG | 1789304372 | 7737269 | 0.8438 |
| FG | 1789304978 | 1231914 | 0.8540 |
| FG | 1789305630 | 5316271 | 0.8438 |
| MVG | 1786053806 | 7318332 | 0.8438 |
| MVG | 1788817038 | 6550047 | 0.8438 |
| MVG | 1788821061 | 7766560 | 0.8540 |
| MVG | 1788866451 | 6779840 | 0.8438 |
| MVG | 1788871442 | 9985617 | 0.8438 |

**Over training** (figures `latex_figures/NDP/KA/`). The per-generation champions of
all 10 runs were recovered by deterministic replay (`replay_archive.py`; all 10
`VERIFIED` against `logger.csv` and the saved genome, `logs/replay_archive.log`) and
drawn with `matched_figures.py grid|progression --out-dir <dir>` (every regrown brain
reproduces its replay fitness, `logs/progression_fig.log`).
- Accuracy on AND is flat at ≈ 0.844 in both arms from the first few generations.
- Density settles by ~generation 450 at ~55% (FG) and ~42% (MVG), `lr_r` at ~0.12 and
  ~0.17, with ±1 SD bands overlapping almost completely.
- **MVG has no effect here**, unlike the Kashtan-Alon reproduction
  (`kashtan_alon/RESULTS.md`). The next section shows why.

## What the brains compute

```bash
python experiments_paper/retina/counting_analysis.py all     # ceilings | final | or-champs
```

**1. Every brain is a majority vote.** All 10 final brains (12–80 neurons) output 1 if
at least 5 of the 8 inputs are on and 0 if at most 3. Patterns with the same number of
inputs on get the same output; f(−x) = −f(x) to 1e-14; at exactly 4 inputs on the
output is 0 to 1e-12. With those ties set to 0 every brain scores 0.8432; the reported
0.8438 and 0.8540 come from floating-point rounding at the ties (renumbering hidden
neurons, which gives an identical network, moves the score to 0.77–0.79).

Ceilings over all 256 patterns (balanced accuracy):

| rule family | AND | OR |
|---|---|---|
| any rule on the number of inputs on | 0.8432 | 0.8212 |
| any sign-symmetric classifier, f(−x) = −f(x) | 0.9796 | 0.8429 |
| count rules NDP can express (output 0 at 4 on) | **0.8432** | **0.7657** |

**2. Why.**
- *No input identity.* All inputs share one role embedding and one starting
  neighbourhood, and the growth and weight networks are applied identically
  everywhere, so each input grows a copy of every other input's subtree and the brain
  is invariant to input permutations. Every hidden neuron connects to exactly one
  input or to all eight. The larger brains split into eight greedy Newman communities,
  one per input lineage, not a left/right split.
- *No bias.* The rollout is `s ← tanh(Wᵀs)` with inputs re-clamped and no bias, so the
  output is an odd function of the input. The target is not (`11001100` and
  `00110011` are both positive).
- The 10 brains have only 5 distinct edge sets; six of them share one 40-neuron wiring.

**3. A perfect brain exists but cannot be grown.** `experiments/experiment_1/oracle.py
--task retina` hand-wires a recurrent tanh brain for this task (4 AND detectors on
different input pairs, 2 OR combiners, an AND output, biases) that scores 1.000. It
needs detectors that single out specific inputs, and thresholds, which NDP cannot
express. **The ≈ 0.84 plateau is a representability limit of this NDP set-up, not a
search failure.**

Consequences: only functions of the number of inputs on are reachable, which excludes
both retina variants and even `x0 AND x1`; MVG has nothing to select for, because the
majority vote is the best expressible brain for AND and for OR; any fitness above
0.8432 (AND) or 0.7657 (OR) in a log here is a rounding artefact. Untested ways out:
per-input embeddings or positional codes, a bias in the rollout, symmetry breaking in
the seed graph.

## Earlier runs, same conclusion

Before the matched study: size regularisation (`--size-reg io_ratio`, α 0.04 and 0.01),
no regularisation, pruning at threshold 0.3, and an MVG run. All reach 0.8432–0.8438:
the regularised runs as a 9-node network with all 8 input weights equal (an unweighted
majority vote), the others as 40–80-neuron networks making the same errors. Pruning
had no effect because every edge weight was tanh-saturated (≈ 0.9997).
`analysis.py weights|popcount|compare|saturation` reproduces these checks on saved
runs. A LunarLander side study on this fork is in `../lunarlander/RESULTS.md`.
