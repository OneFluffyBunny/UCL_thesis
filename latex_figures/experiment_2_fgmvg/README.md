# Experiment 2 — FG vs MVG, direct encoding, synaptic budget

Promoted 2026-09-14, approved by the user. Source study:
`experiments/experiment_2/runs/fgmvg/` (gitignored; regenerable). Repo at
promotion time: branch `fluffy_experiments`, HEAD `26a4b97`. Runs' own commits
(from `config.json`): `15a09f8` for 9 of the 10 budget runs, `71134c7` for
`budget_fg` seed 0 (the two were not diffed).

## The arm these figures show

Direct encoding (one gene per allowed synapse plus one bias per non-input neuron,
768 + 25 = 793 genes), **synaptic budget** arm: S = 4.0, shrink tau = 0.9.
8 in / 24 hidden / 1 out, 768 allowed edges, `retina_ka2005`, reference goal AND,
MVG alternating AND <-> OR every 20 generations, evosax `CMA_ES` (not elitist)
popsize 64, raw accuracy, margin fitness, **5000 generations** (exp_1 ran 10000 —
generation counts are not matched across encodings), 5 seeds per arm. Full
replication detail: `add_to_latex.md`, "Replicating the study exactly" (exp_2
differs only as stated there).

## The files

| file | what it is | regenerate with |
|---|---|---|
| `brains_grid_leftright_budget.png` | goal-matched champion of all 10 runs; retina inputs square, hidden neurons coloured by the left/right module `lr_r` is scored at; captions `acc / density / lr_r` at 2 dp | `fig_brains.py --root <study> --grid --constraint budget` |
| `progress_fg_vs_mvg_budget.png` | champion accuracy, density and `lr_r` over 5000 generations; 5 seed mean per arm, shaded +- 1 SD; sampled at reference-goal epoch ends | `fig_progress.py --root <study> --constraint budget` |
| `switch_window_budget_seed0.png` | windows [100,300] and [1000,1200]; champion accuracy on the ACTIVE goal, per-generation population mean, champion `lr_r`; off-goal epochs shaded | `fig_switch_window.py --root <study> --constraint budget --seed 0` |

All three run from `experiments/analysis/`.

Switch-window figures restyled and re-promoted 2026-09-14, approved by the user:
titles, axis labels and legend only (same replay data, same panels).

## Caveats that must travel with these figures

* **The windows figure is a REPLAY, seed 0** (`analysis/dense_replay.py`), a
  second sample of the arm rather than the archived trajectory; all its rows come
  from the one replay. See `latex_figures/experiment_1_fgmvg/README.md`.
* **The first point of each progress curve is not at the same generation**: FG's
  is generation 0, MVG's is the end of the first AND epoch (generation 19). The
  apparent MVG head start at the left edge is that offset, not an effect.
* **Positive `lr_r` depends on `--prune-threshold 0.05`** and FG ≈ MVG here
  (+0.19 vs +0.21); do not quote a value without its cut.
* **Position in the brains figure is a spring layout, not a layer.** Density is
  over the 768 allowed edges.
