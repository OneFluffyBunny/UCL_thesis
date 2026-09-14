# Experiment 1 — FG vs MVG, compressed encoding, synaptic budget

Promoted 2026-09-13, approved by the user. Source study:
`experiments/experiment_1/runs/fgmvg/` (gitignored; regenerable). Repo at
promotion time: branch `fluffy_experiments`, HEAD `26a4b97`.

## The arms these figures show

Compressed (K cell-type) encoding, both sides of the ablation: `*_budget*` files
are the **synaptic budget** arms, `*_nobudget*` files the unconstrained ablation
(no budget, no gate; promoted 2026-09-13). 8 in / 24 hidden / 1 out, 768 allowed
edges, K = 8 types, 443 genes, budget S = 6.0, shrink tau = 0.9 (budget arms only),
`retina_ka2005`, reference goal AND, MVG alternating AND <-> OR every 20
generations, evosax `CMA_ES` (not elitist) popsize 64, raw accuracy, 10000 generations, 5 seeds per arm.

## The files

| file | what it is | regenerate with |
|---|---|---|
| `brains_grid_leftright_budget.png` | goal-matched champion of all 10 runs; retina inputs square (KA convention), hidden neurons coloured by the left/right module `lr_r` is scored at; captions `acc / density / lr_r` at 2 dp | `fig_brains.py --root <study> --grid --constraint budget` |
| `progress_fg_vs_mvg_budget.png` | champion accuracy, density and `lr_r` over 10000 generations; **5 seed mean per arm, shaded +- 1 SD**; sampled at reference-goal epoch ends | `fig_progress.py --root <study> --constraint budget` |
| `switch_window_budget_seed0.png` | two KA windows [100,300] and [1000,1200]; champion accuracy on the ACTIVE goal, per-generation population mean, champion `lr_r`; off-goal epochs shaded | `fig_switch_window.py --root <study> --constraint budget --seed 0` |

| `brains_grid_leftright_nobudget.png` | as above, unconstrained arms | `fig_brains.py --root <study> --grid --constraint nobudget` |
| `progress_fg_vs_mvg_nobudget.png` | as above, unconstrained arms | `fig_progress.py --root <study> --constraint nobudget` |
| `switch_window_nobudget_seed0.png` | as above, unconstrained arms; also a seed-0 replay | `fig_switch_window.py --root <study> --constraint nobudget --seed 0` |

All six run from `experiments/analysis/`. Full replication detail: `add_to_latex.md`,
"Replicating the study exactly".

Switch-window figures restyled and re-promoted 2026-09-14, approved by the user:
titles, axis labels and legend only (same replay data, same panels).

## Caveats that must travel with these figures

* **The windows figure is a REPLAY, seed 0**, produced by
  `analysis/dense_replay.py`, and all three of its rows come from that one
  replay. The archived study is not bit-reproducible on this backend (float
  reduction order, amplified by CMA-ES: sigma 0.097134 vs 0.097137 at generation
  0), so a replay is a *second sample of the same arm*. Replayed and archived
  rows are never mixed in one panel. Replay-vs-replay is bit-exact, so the figure
  is regenerable.
* **Density and `lr_r` use `--prune-threshold 0.05`**, an inherited convention,
  not a derived value. The *null* result is threshold-robust, but a *positive*
  `lr_r` in a budgeted arm is not (seed 0 FG reads +0.067 / +0.175 / +0.449 /
  +0.197 at cuts 0 / 0.05 / 0.10 / 0.20). Do not quote a positive `lr_r` without
  its cut. See `add_to_latex.md`, "The FG vs MVG study — parameters as actually
  run".
* **Position in the brains figure is a spring layout, not a layer** — the model
  gives neurons no position at all. Read the colours, not the distances.
* **In the no-budget figures true density is exactly 100%** — no allowed edge is
  zero — so their `lr_r` (≈ −0.02) means *unanswerable*, not *unmodular*.
* Density here is over the **768 architecturally allowed** edges (IH + HH + HO,
  no self-loops), never over N^2.
