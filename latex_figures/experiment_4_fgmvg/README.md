# Experiment 4 — FG vs MVG, CGP circuits searched by Kashtan–Alon's population GA

Promoted 2026-09-15, approved by the user. Source study:
`experiments/experiment_4/runs/study_ga_E2000/` (gitignored; regenerable), a
`study.json` that pairs two run directories:

| arm | run directory (under `experiments/experiment_4/runs/`) |
|---|---|
| FG | `fgmvg50_ga_fg/cgppop_retina_ka2005_fg-and_n50_S600L150_g100000` |
| MVG | `fgmvg50_pop/cgppop_retina_ka2005_mvg-and-or_n50_S600L150_g100000` |

Repo at promotion time: branch `fluffy_experiments`, HEAD `e1ab8dc`. The search code
(`train_pop.py`, `cgp.py`) is unchanged since `7dffb55`; the runs predate that commit
(`config.json` records no commit), so they were checked against it: rerunning seed 0 of
both arms for 2,500 generations with `7dffb55`'s `train_pop.py` gives archive rows
identical to the stored runs (2026-09-15; covers the MVG switch at generation 2,000).

## The arm these figures show

**Not the (1+4) ES of the CGP paper.** Same circuits, task, gates and mutation operator
as experiment 4's `train.py`; only the search loop is KA's GA
(`kashtan_alon/ga.py::reproduce`):

| | value |
|---|---|
| task | `retina_ka2005` (KA's Fig. 5a retina, 8 inputs, 256 patterns), reference goal L AND R |
| MVG | AND <-> OR every **E = 2000** generations, starting on AND |
| genotype | CGP, 50 nodes, 1 output, gates `and,nand,or,nor` (arity 2) |
| fitness | raw accuracy (hits / 256) |
| population | S = 600; top L = 150 copied unchanged; 450 children of two uniformly drawn elites |
| crossover | p = 0.5, per node (function + inputs from parent A or B); else clone of A |
| mutation | p = 0.5 per child, `cgp.mutate` at 3% of gene slots (5 of 151) |
| generations / seeds | 100,000 / seeds 0–4 per arm |
| evaluations | ~33.8M per seed (both arms) |
| wall time | ~18–21 min per seed, 5 seeds in parallel, CPython, laptop |

Commands, replication detail and the (1+4)-vs-GA comparison: `add_to_latex.md`,
"Roadmap: from standard CGP to KA's population GA".

## The files

| file | what it is | regenerate with |
|---|---|---|
| `circuits_grid_ga_E2000.png` | one circuit per run, FG top / MVG bottom; gates coloured by circuit purity's per-gate left/right share (`bwr`, blue = LEFT p0–p3, red = RIGHT p4–p7); captions `acc | purity | gates` | `fig_fgmvg_circuits.py --root ../runs/study_ga_E2000` |
| `progress_fg_vs_mvg_ga_E2000.png` | champion accuracy on the active goal, champion circuit purity, champion active gates; 5 seed mean per arm, shaded +- 1 SD; sampled every 200 generations (500 points) | `fig_fgmvg_progress.py --root ../runs/study_ga_E2000 --every 200` |
| `switch_window_ga_E2000_seed0.png` | seed 0, windows [0, 6000), [15000, 21000) (FG seed 0 solves at 17,461) and [94000, 100000); champion accuracy on the active goal, population mean accuracy (50-gen rolling mean), champion purity; OR epochs shaded | `fig_fgmvg_windows.py --root ../runs/study_ga_E2000 --windows 0:6000,15000:21000,94000:100000 --seed 0` |

All three run from `experiments/experiment_4/analysis/` (conda env `lndp`); add `--out`
to name the file. The study files were written as `circuits_fg_vs_mvg_ga_E2000.png`,
`progress_fg_vs_mvg_ga_E2000.png` and `windows_fg_vs_mvg_seed0_ga_E2000.png`.

Numbers in the circuits figure:

| seed | FG acc / purity / gates | MVG acc / purity / gates |
|---|---|---|
| 0 | 1.000 / 0.92 / 19 | 0.906 / 1.00 / 12 |
| 1 | 0.969 / 0.69 / 19 | 1.000 / 1.00 / 20 |
| 2 | 1.000 / 0.92 / 19 | 1.000 / 1.00 / 16 |
| 3 | 0.941 / 0.85 / 18 | 1.000 / 1.00 / 16 |
| 4 | 1.000 / 0.60 / 18 | 1.000 / 1.00 / 16 |

Progress figure, last point (generation 100,000): FG acc 0.982, purity 0.794, 18.6
gates; MVG acc 0.981 (on OR, the active goal), purity 1.000, 16.0 gates.

## Caveats that must travel with these figures

* **Which circuit is drawn differs by arm.** FG: the final champion (generation
  99,999). MVG: the champion at the end of the last AND epoch (generation 97,999), so
  both rows are scored on AND. MVG seed 0 is the one unsolved MVG run (solved at
  6,181, lost it around 60k).
* **The windows figure is the archived trajectory, not a replay.** Every generation's
  champion and population mean were written by the run itself (`--archive-interval 1`).
* **A missing accuracy dip at a switch is not a bug.** At 98,000 (AND -> OR) the
  champion's score stays 232 because the population already held the champion with
  its output gate flipped to OR (AND 136 / OR 232); no single circuit can score 232 on
  both goals (hits_AND + hits_OR <= 384). The population mean does dip (220.9 ->
  127.3). At 96,000 no such variant was present: 232 -> 200, recovered in 4 gens.
* **The progress accuracy curve is on the ACTIVE goal**, which is fair only because
  recovery (1–2 generations median) is far shorter than the 200-generation sampling.
* **Circuit purity is a wiring statistic and weakly size-confounded** (random circuits:
  0.50 at 2 gates -> 0.39 at 50; `latex_figures/purity_metric/`). MVG circuits are
  slightly smaller (16 vs 18.6), which pushes the wrong way for a size artefact.
* **FG solves 3/5, MVG 4/5 at the end.** The purity contrast also holds solved-vs-solved
  (FG 0.60–0.92, never 1.00; MVG 1.00), so it is not an accuracy confound.

---

# The (1+4) ES baseline — `*_es1p4*.png`

Promoted 2026-09-15, approved by the user; drawn with the same (restyled) scripts, so
the two sets read alike. Source study: `experiments/experiment_4/runs/fgmvg50/`
(gitignored), no `study.json` — the two run directories are found by glob:
`cgp_retina_ka2005_fg-and_n50_m0.03_g800000_arch` and
`cgp_retina_ka2005_mvg-and-or_n50_m0.03_g800000_arch`. Repo at promotion time: branch
`fluffy_experiments`, HEAD `e1ab8dc`.

## The arm

Experiment 4's own frozen search, `train.py`: the CGP paper's **(1+4) ES** (one parent,
4 mutated offspring, offspring win ties = neutral drift, no crossover). Same task,
genotype (50 nodes, `and,nand,or,nor`), raw accuracy, 3% point mutation and MVG
schedule (AND <-> OR, E = 2000) as the GA set above. **800,000 generations**, 3.2M
evaluations per seed, seeds 0–4. Command and the side-by-side comparison with the GA:
`add_to_latex.md`, "(1+4) ES vs KA's GA — what actually differs".

## The files

| file | what it is | regenerate with |
|---|---|---|
| `circuits_grid_es1p4.png` | as `circuits_grid_ga_E2000.png`; FG champion at 799,999, MVG last AND-epoch champion at 797,999 | `fig_fgmvg_circuits.py --root ../runs/fgmvg50` |
| `progress_fg_vs_mvg_es1p4.png` | as the GA progress figure, but sampled **every 2000 generations** (400 points, each the last generation of a goal epoch) | `fig_fgmvg_progress.py --root ../runs/fgmvg50 --every 2000` |
| `switch_window_es1p4_seed0.png` | seed 0, windows [0, 6000), [86000, 92000) (FG seed 0 solves at 89,108) and [792000, 798000); population mean = parent + 4 offspring, 50-gen rolling mean | `fig_fgmvg_windows.py --root ../runs/fgmvg50 --seed 0` |

Numbers in the circuits figure:

| seed | FG acc / purity / gates | MVG acc / purity / gates |
|---|---|---|
| 0 | 1.000 / 0.82 / 24 | 0.848 / 0.81 / 9 |
| 1 | 1.000 / 0.90 / 16 | 0.844 / 0.55 / 11 |
| 2 | 1.000 / 0.87 / 16 | 0.844 / 0.50 / 7 |
| 3 | 1.000 / 0.90 / 16 | 0.812 / 0.25 / 14 |
| 4 | 1.000 / 0.89 / 15 | 0.844 / 0.71 / 18 |

Progress figure, last point (generation 800,000): FG acc 1.000, purity 0.875, 17.4
gates; MVG acc 0.840, purity 0.685, 10.4 gates.

## Caveats for this set

* **Why every 2000, not 200.** Under the (1+4) ES a switch takes hundreds of
  generations to recover (seed 0: 18–1922), so 200-generation samples would mostly
  trace recovery dips. Epoch-end samples are pre-switch accuracy on the goal just
  trained (AND and OR alternate).
* **The windows figure is the archived trajectory** (`--dense-archive` over those
  three windows); outside them the archive is every 100 generations.
* **FG > MVG purity here is a solving effect, not an MVG effect.** At matched accuracy
  (0.81–0.85) FG and MVG champions have the same purity (median 0.60 vs 0.62) and size
  (7 gates) — `add_to_latex.md`, "The 5 FG vs 5 MVG study".
* **Not budget-matched to the GA set:** 800k generations / 3.2M evaluations vs 100k /
  ~33.8M.
