# Overnight run — exp_1 & exp_2, FG/MVG x constraint (2026-09-12)

**READ THIS FILE FIRST** after any compaction, kill, or reboot. It carries the
user's verbatim instructions, the settled config, the exact commands, a live run
ledger, and recovery steps.

Branch: `fluffy_experiments`. Machine: local laptop, 22 CPU cores, JAX on CPU
(`[CpuDevice(id=0)]`). Python 3.10.20 in conda env `lndp`.

---

## 1. The user's instructions (verbatim, 2026-09-12, before going to sleep)

> I want to take a look again at what we did in the Kashton-Alon folder where we
> ran 5 FG runs and 5 MVG runs followed by the same 5 and 5 on the ablation of no
> fan-in. After that we generated figures of the brains, tables of the accuracies
> and modularity metrics (4 in total) and progress in aggregate and for specific
> windows. I want you to add similar code so we can do the same stuff for
> experiment 1 and experiment 2. On the actual specifics of the task and
> hyperparameters, look through the experiments we did already in here and see
> which ones is more likely to lead to be the accuracy. To me it seems like K=6 or
> K=8 on 24 hidden neurons on the the retina_ka2005 task (the original from the
> paper which I think take the fitness as the raw score out of 256 no balancing)
> is the way to go. Feel free to do preliminary experiments before running and you
> can choose really if you disagree with my read. After you decided on the exact
> task and hyperparams, run with those same params 5 runs on FG and 5 on MVG (i
> think 10k generations is good, but a different number might be chosen based on
> your experiments). After that, run the same 10 using the budgeted version
> (similar to the fan-in we used in Kashton-Alon): I think S=4, tau=.9 were good
> parameters but not sure so feel free to experiment again before running. Make
> sure all runs are properly saved and all the figures we talked about earlier can
> be generated here as well (importantly we must be able to see for a few seeds how
> accuracy/modularity changes in some windows generation by generation). Just to
> make sure the task switches should be AND/OR probably every 20 generations and
> make sure the various bugs we encountered before on what is the best brain are
> fixed (importantly the best brain is looked on the current task etc.). After
> that, produce 5 FG runs and 5 MVG runs using the direct encoding in experiment 2.
> After that propose smth to be similar to the budgeted version in experiment 1
> (don't worry if you don't find anything, I can think of smth in the morning).
> Through all this, make sure you clearly save all instructions in a separate file
> which you can easily check and that stuff is recoverable in case anything breaks
> (e.g., laptop shuts down). If ever context is above 200k, compact and read from
> the file we talked about.

Follow-ups:

> 2. do this in the usual fluffy_experiments branch. 4. yup don't add any figures
> yet in there [latex_figures/].

> If ever not sure what to do do an experiment to decide. Having some runs in the
> morning is better than no runs. Before any runs, do make the necessary code
> changes: make sure they affect no old code and do make sure they do what you
> think they do (many bugs creeped in the past). Make sure all these instructions
> are saved somewhere separately so you can read them from. You go past 200k input
> tokens/msg ... press compact.

### Standing constraints (from CLAUDE.md + memory)
- `main` is FROZEN. Work on `fluffy_experiments`. Commit (`git add`!) — do not push
  unless asked.
- `experiment_4/` is FROZEN (runs identified by seed; search must not change).
- Run Python as `conda run -n lndp python ...`. `conda run` cannot take a
  multi-line `python -c` — write a file instead.
- `runs/` and `*.png` under `experiments/` are gitignored by design. Conclusions
  go in `RESULTS.md`.
- **Do NOT copy any figure into `latex_figures/`** — that needs human approval.
- Balanced accuracy is the repo default, but see section 3: this run uses RAW.

### Operating rules the user set for tonight
- **If unsure, run an experiment to decide.** Do not block on a question.
- **Some runs in the morning beats no runs.** Bias to launching.
- **Code changes must not touch old code paths.** Everything additive and
  default-off; verify each change does what it claims (regression-test it).
- Compact at >200k input tokens, then re-read this file.

### Pre-committed decisions (so no 3am question is needed)
- If `retina_ka2005` proves not fully representable under the `g` encoding:
  **keep the task**. A representability ceiling binds FG and MVG equally, so the
  FG-vs-MVG contrast survives. Record the measured ceiling and move on.
- If K=6 and K=8 tie in the preflight: **take K=8** (non-modularity is expressible
  at K=8, which defuses the "the encoding forced modularity" objection).
- If a preflight is ambiguous or overruns its time box: take the default in
  section 3 and launch. Launching beats optimising.

---

## 2. What we are reproducing (the kashtan_alon/ template)

`kashtan_alon/` ran 4 groups x 5 seeds:

| condition | arm | acc (AND) | Q | Q_m | r | purity | edges | density |
|---|---|---|---|---|---|---|---|---|
| capped (paper) | FG | 0.90+-0.03 | 0.38+-0.04 | +0.02+-0.14 | +0.60+-0.14 | 0.56+-0.13 | 40 | 38% |
| capped | MVG | 0.97+-0.03 | 0.49+-0.03 | +0.31+-0.09 | +0.94+-0.09 | 0.93+-0.09 | 36 | 34% |
| no cap | FG | 0.97+-0.02 | 0.22+-0.03 | -0.07+-0.10 | +0.30+-0.07 | 0.31+-0.11 | 69 | 65% |
| no cap | MVG | 1.00+-0.00 | 0.29+-0.04 | +0.02+-0.03 | +0.51+-0.08 | 0.56+-0.11 | 55 | 52% |

The 4 metrics: **Q** (Newman), **Q_m** (normalized, KA's), **r** (left/right
correlation at the planted split), **purity** (circuit purity).

Figures approved from that study (`latex_figures/Kashtan-Alon/README.md`),
generators in `kashtan_alon/analysis/`:
1. `switch_window_seed0.png` (+ `_no_fanin`) — generation-by-generation accuracy
   and modularity across a *window* spanning goal switches. <- the user explicitly
   wants this one for exp_1/exp_2.
2. `fg_vs_mvg_purity.png` (+ `_no_fanin`) — aggregate progress, FG vs MVG.
3. `newman_communities_mvg_seed1.png` — brain drawing, communities coloured.
4. `paper_10runs_grid.png` (+ `_no_fanin`) — the 10-run grid.

Goal-matching rule inherited from K-A: **every reported panel/number is taken from
the last champion archived during an AND epoch** — never a best-ever across goal
switches, never an endpoint that happens to land on OR.

---

## 3. Settled configuration (see section 5 for preflight evidence)

| knob | value | why |
|---|---|---|
| task | `retina_ka2005` | KA's real Fig.5a rule; equal (L,R) cells |
| metric | **RAW accuracy** (`--no-balanced`) | equal cells => raw IS per-cell mean; every one-eye cheat caps at exactly 0.750. Balanced would hand that same cheat 0.833. |
| operation (FG) | `and` | matches K-A's reported arm |
| MVG ops | `and,or`, `--switch-interval 20` | classic KA pairing; E=20 measured right in exp_1 RESULTS.md (median t_recover = 10 gens) |
| K (`--n-types`) | **8** | preflight: 0.914 vs K=6's 0.906; and at K=8 non-modularity is expressible |
| n_hidden | 24 | user's read; matches the 2x2 study |
| fitness | `margin` | accuracy fitness is piecewise-constant => near-zero gradient signal for CMA-ES |
| generations | exp1 **10,000**; exp2 **5,000** | exp1: FG saturated ~gen 1400 historically, MVG still moving at 3000. exp2: CMA-ES is superlinear in dimension and exp2 searches 793 weights vs exp1's 443, so a generation costs ~5x; exp2 also converges in 290-456 gens historically, so 5000 is ~11x its own solve time. Arms WITHIN an experiment are always matched; across encodings they are not, and any exp1-vs-exp2 statement must say so. |
| seeds | 0..4 per arm | |
| constrained arm | exp1 **S=6 tau=0.9**; exp2 **S=4 tau=0.9** | exp_1's analogue of KA's fan-in cap; the absolute gate (`--w-threshold`) is a known failure |
| unconstrained arm | no budget, no gate | |
| always | `--no-early-stop --no-balanced --no-open` | `--no-early-stop` is REQUIRED for a fair FG-vs-MVG comparison |

**Measured accuracy (see section 5): unconstrained 0.914-0.926, constrained
~0.854.** This SUPERSEDES the planning estimate of 0.85-0.89, which was based on
a monotone-representability bound that does not apply (the g-encoding is not
monotone) and on weaker earlier task/metric/K combinations. KA's own paper net
got 0.90+-0.03, so the unconstrained arm is now slightly ABOVE the reference
reproduction.

---

## 4. Build order (code before runs)

1. [ ] Per-generation **champion archive** in exp_1 `train.py` (the `_brains.npz`
       equivalent). Without it the switch-window figure is impossible.
2. [ ] **Goal-matching fix**: per-goal bests, `acc_by_op` in `result.json`,
       `last_on_goal()` equivalent. (exp_1 `train.py` line ~200 carries the same
       best-EVER-across-switches bug K-A already fixed.)
3. [ ] `retina_ka2005` branch in exp_1 `oracle.py` (today `--task` accepts only
       `left`/`retina`) — representability preflight.
4. [ ] Recurrence-safe **purity** metric. `qmetrics.circuit_purity` calls
       `nx.topological_sort` and raises on a cycle; exp_1's HH block is recurrent.
       Fix = time-unroll the `rnn_iters` into a DAG.
5. [ ] Scoring pipeline: 4 metrics (Q, Q_m, r, purity) over an archive.
6. [ ] Brain drawing for exp_1/exp_2 (KA's `highlight_modules.py` is layer-based,
       takes a list of per-layer matrices — NOT reusable for a single (N,N)
       recurrent adjacency).
7. [ ] 4 figure generators mirroring K-A's.
8. [ ] Runner with skip-completed-seed-dir resumability.
9. [ ] Same for exp_2 (shares `shared_direct_model.py` with exp_3 — changes there
       must not break exp_3).

**Every change must be additive and default-off.** Old commands must behave
byte-identically. Regression check before launching.

---

## 5. Preflight results

All preflights: `retina_ka2005`/and, RAW accuracy, margin fitness, n_hidden=24,
FG, **3 seeds, 2000 generations**. "END" = the `matched` champion, which under a
fixed goal is the endpoint network — the one whose density is comparable across
arms. Reproduce with `experiments/experiment_1/scratch_preflight_table.py`.

| arm | END acc | END density | per-seed END acc |
|---|---|---|---|
| K=6, no budget | 0.906 +- 0.018 | 91.5 +- 8.7 | 0.891 0.902 0.926 |
| **K=8, no budget** | **0.914 +- 0.012** | 90.9 +- 8.3 | 0.914 0.926 0.902 |
| S=2, tau=0.85 | 0.812 +- 0.000 | 60.2 +- 19.7 | 0.812 0.812 0.812 |
| S=2, tau=0.9 | 0.812 +- 0.000 | 49.5 +- 12.3 | 0.812 0.812 0.812 |
| S=4, tau=0.85 | 0.837 +- 0.022 | 66.4 +- 7.3 | 0.852 0.848 0.812 |
| S=4, tau=0.9 | 0.857 +- 0.016 | 52.3 +- 12.0 | 0.852 0.875 0.844 |
| S=4, tau=0.95 | 0.842 +- 0.041 | 50.2 +- 2.9 | 0.871 0.812 |
| S=6, tau=0.85 | 0.865 +- 0.055 | 55.9 +- 7.0 | 0.812 0.922 0.859 |
| **S=6, tau=0.9** | **0.854 +- 0.018** | **49.3 +- 0.5** | 0.848 0.875 0.840 |
| S=6, tau=0.95 | 0.838 +- 0.036 | 47.9 +- 4.1 | 0.863 0.812 |
| S=8, tau=0.95 | 0.844 +- 0.044 | 47.7 +- 2.9 | 0.812 0.875 |

**K = 8** (confirmed). Ahead of K=6 on the endpoint at every seed count tried
(0.914 vs 0.906 at 3 seeds; 0.920 vs 0.896 at 2). The margin is small, so the
argument carries the rest of the weight: at K=8 a non-modular solution is
expressible, which is what stops "the encoding forced the modularity" from being
a free objection. Cost is ~12 of ~443 genome parameters — `g` dominates.

**Budget S = 6, tau = 0.9** — NOT the S=4 in the original plan. S=4 and S=6 at
tau=0.9 tie on accuracy (0.857 vs 0.854, well inside the seed spread), but S=6
lands lower in density and, more importantly, **24x tighter** (+-0.5 vs +-12.0).
For a 5-seed FG-vs-MVG contrast a consistent density is worth more than half a
point of mean accuracy, because density is the confound the whole comparison has
to hold still. Rejected: tau=0.95 buys only ~2 points of density for ~1.5 points
of accuracy and much worse variance; S=2 collapses every seed to exactly 0.812,
which is the budget being too tight to hold a solution at all.

**Accuracy is far better than expected — revise the prior.** The planning note
said "expect 0.85-0.89, nothing in exp_1 has ever exceeded 0.885". Wrong: the
unconstrained K=8 arm reaches **0.914-0.926**, clearing experiment 1's previous
all-time best by ~4 points, and one budget seed hit 0.922. The 0.891 figure
quoted as a ceiling is the *monotone-representability* bound, and the g-encoding
is not monotone, so it never bound this. `retina_ka2005` under the g-encoding is
substantially solvable; it was the earlier task/metric/K combinations that were
weak, not the encoding.

**The budget costs ~6 points of accuracy (0.914 -> 0.854).** That is the price of
the constraint and it must be reported, not buried: the constrained arm is a
different competence regime, so a modularity difference between constrained and
unconstrained is confounded with accuracy. The FG-vs-MVG contrast WITHIN each
constraint level is the clean comparison.

**Density confirms the 2x2's central finding.** The unconstrained arm converges
to 87-100% density (live runs are sitting at 98-100% by generation 6800). At that
density Q_m is undefined or meaningless and the modularity question is not
answered low — it is unanswerable. The budget is what makes it askable.

### Two environment gotchas found the hard way (cost ~25 min)
1. **`conda run` cannot be launched concurrently.** 8 parallel invocations raced
   on one activation temp file; 7 died in under a second and the launcher
   reported success. Call `C:\Users\raduc\miniconda3\envs\lndp\python.exe`
   directly for anything parallel.
2. Doing so loses `conda run`'s UTF-8 stdout, and the `sigma` character in
   experiment 1's log line then kills the run with `UnicodeEncodeError` the
   moment output is redirected. Export `PYTHONIOENCODING=utf-8`.
   `run_fgmvg_study.py` does both correctly.

---

## 6. Run ledger

Everything is driven by one script. **To see what is done:**

```
conda run -n lndp python experiments/run_fgmvg_study.py --status
```

**To run / resume everything** (skips any seed whose `result.json` says
`"complete": true`, so this is also the crash-recovery command):

```
cd experiments
python run_fgmvg_study.py --experiment 1 --lanes 10
python run_fgmvg_study.py --experiment 2 --lanes 10
```

Note `python` there must be the env interpreter directly
(`C:\Users\raduc\miniconda3\envs\lndp\python.exe`), NOT `conda run` — see the
gotchas in section 5. The runner already does this for the children it spawns.

40 runs total = 2 experiments x 4 arms x 5 seeds. Output lands in
`experiments/experiment_{1,2}/runs/fgmvg/<arm>/<run_name>_seed<N>/`.

| experiment | arm | gens | status |
|---|---|---|---|
| 1 (compressed) | nobudget_fg | 10,000 | **5/5 done, scored, figured** |
| 1 | nobudget_mvg | 10,000 | **5/5 done, scored, figured** |
| 1 | budget_fg (S=6, tau=0.9) | 10,000 | **5/5 done, scored, figured** |
| 1 | budget_mvg (S=6, tau=0.9) | 10,000 | **5/5 done, scored, figured** |
| 2 (direct) | nobudget_fg | 5,000 | **5/5 done** |
| 2 | nobudget_mvg | 5,000 | **5/5 done** |
| 2 | budget_fg (S=4, tau=0.9) | 5,000 | **5/5 done** |
| 2 | budget_mvg (S=4, tau=0.9) | 5,000 | **5/5 done** |

### Analysis

```
cd experiments
python analysis/run_all.py --root experiment_1/runs/fgmvg
python analysis/run_all.py --root experiment_2/runs/fgmvg
```

Produces, next to the runs: `metrics_per_seed.csv`, `metrics_summary.json`,
`progress_fg_vs_mvg.png`, `switch_window_*.png`, `brains_grid_*.png`.
Individual pieces: `score_table.py`, `fig_progress.py`, `fig_switch_window.py`,
`fig_brains.py`. Add `--quick` while iterating.

**Timing, so a slow pass is not mistaken for a hang:** a 20-run scoring pass at
`--n-rand 200` takes ~13 minutes. `normalized_qm` costs ~10-15s per brain and is
INDEPENDENT of `n_rand` (the cost is the fixed Q_max hill-climb); `left_right_q`
is cheap but DOES scale with `n_rand`; purity and Newman Q are near-free.
Keep `--n-jobs 1` whenever training is also running.

---

## 7. Recovery

- All runs write to disk incrementally (`log.csv` per log-interval, champion
  archive per log point, `result.json` at seed end).
- The runner **skips seed dirs that already contain a completed `result.json`**,
  so re-running the same command resumes rather than restarts.
- To see what is running: check section 6 above, then `ls experiments/*/runs/`.
- To resume everything: re-run the commands in section 6 verbatim.


---

## 8. Results so far (read this first in the morning)

Full write-up: `experiment_1/RESULTS.md`, new section at the bottom. Numbers:
`experiment_1/runs/fgmvg/metrics_summary.json` + `metrics_per_seed.csv`.

### Experiment 1 — done, 20/20

| condition | arm | acc (AND) | acc (OR) | density % | LR | purity | Q | Q_m |
|---|---|---|---|---|---|---|---|---|
| budget | FG | 0.895+-0.017 | n/a | 43.0+-5.3 | -0.569+-1.322 | **0.114+-0.088** | 0.169+-0.096 | 0.403+-0.324 |
| budget | MVG | 0.853+-0.016 | 0.423+-0.034 | 34.0+-7.6 | -0.090+-0.744 | 0.066+-0.106 | **0.208+-0.154** | 0.587+-0.588 |
| ablation | FG | **0.978+-0.022** | n/a | 93.7+-6.4 | undefined (3/5) | 0.021+-0.008 | 0.048+-0.024 | undefined (1/5) |
| ablation | MVG | 0.867+-0.042 | 0.467+-0.080 | 96.9+-5.1 | undefined (1/5) | 0.004+-0.008 | 0.010+-0.014 | undefined (1/5) |

Seeds beating their own null at the planted split: **budget MVG 2/5, all others 0/5.**

Three things to take to the thesis:

1. **`retina_ka2005` is solvable under the g-encoding — 0.978, one seed at
   1.000.** This kills the "experiment 1 tops out at 0.885" line that is
   currently in RESULTS.md, and beats KA's own 0.90+-0.03.
2. **The CONSTRAINT makes the modularity, goal-switching does not.** 5-16x on
   purity, 4-20x on Q, density halved. Removing it sends 3 of 5 MVG seeds to
   *exactly* 100.0% density, where the metrics are undefined rather than low.
   MVG does not beat FG: accuracy and purity favour FG, Q and LR-significance
   favour MVG — two of four each way, reported as a split.
3. **MVG never holds both goals.** The AND-matched champion scores 0.42-0.47 on
   OR, below the 0.750 one-eye cap. `switch_window_budget_seed0.png` shows AND
   and OR alternating in near-perfect antiphase across all 500 switches with no
   narrowing. MVG is re-specialising every epoch, not building a shared modular
   decomposition.

### Experiment 2 — the budget analogue you asked me to propose

Already built and running, so there is nothing to decide in the morning unless
you dislike it. `shared_direct_model.py` gained the SAME synaptic budget as
experiment 1: each neuron gets a fixed total incoming |weight| `S`, shared out
over its synapses, with `--shrink tau` zeroing anything below tau x that
neuron's own mean before the share-out.

It is an exact analogue rather than an approximation, and simpler than
experiment 1's, for one reason: experiment 1 evaluates `g` per *signature* and so
must weight each pair by how many clone neurons it stands for
(`_source_multiplicity`); in the direct encoding every matrix entry IS one
synapse, so that correction collapses to the role mask. Same guarantee in both:
`sum_i |w[i,v]| = S` for every non-input neuron.

Default 0.0 leaves experiment 3 byte-identical (regression-checked). Measured on
a 150-generation probe: **38.7% density at 0.883 accuracy**, the same band as
KA's capped arm (34-38%). Running at S=4, tau=0.9.

### Experiment 2 — done, 20/20

**The direct encoding solves the task outright: `acc(AND) = 1.000` in ALL TWENTY
seeds** — every arm, every seed, constrained and not. And the MVG failure is
sharper than experiment 1's: the AND-matched MVG champion scores **exactly 0.500
on OR** in all ten MVG seeds. So the antiphase trade-off is not an artifact of
the compressed encoding; it is what goal-switching does here regardless of how
the brain is encoded.

| arm | acc (AND) | acc (OR) | density |
|---|---|---|---|
| budget FG | 1.000 (5/5) | n/a | 37.1-38.9% |
| budget MVG | 1.000 (5/5) | 0.500 (5/5) | 37.1-38.5% |
| ablation FG | 1.000 (5/5) | n/a | 95.1-95.8% |
| ablation MVG | 1.000 (5/5) | 0.500 (5/5) | 95.7-98.0% |

**The budget costs the direct encoding NOTHING.** 1.000 accuracy at ~38% density
— Kashtan-Alon's capped-arm density band (34-38%) with better accuracy than KA's
own 0.90+-0.03. Contrast experiment 1, where the same kind of constraint cost 8
points (0.978 -> 0.895). A 793-parameter direct encoding can afford the
constraint; a 443-parameter compressed one cannot.

**This sets up the actual thesis question, and the comparison is clean.** The two
constrained arms land at MATCHED density (exp 2 at 37-39%, exp 1 at 34-43%), so
density is controlled for free. The direct encoding is ahead on accuracy
(1.000 vs 0.895). Whether the genomic bottleneck buys anything therefore rests
ENTIRELY on the modularity metrics at that matched density — see the two
`metrics_summary.json` files.

### What is NOT done
- No figure has been copied into `latex_figures/` — that needs your eyes first.
- `add_to_latex.md` has not been updated with the new 4-group table.
- The earlier 2x2 in `add_to_latex.md` (3 seeds, n_hidden=24, K=6, S=4 tau=0.9,
  2000 gens) is now SUPERSEDED by this 5-seed 10k-generation study, but the old
  entry has not been marked as such.
