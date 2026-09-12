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
| K (`--n-types`) | **TBD by preflight, default 8** | |
| n_hidden | 24 | user's read; matches the 2x2 study |
| fitness | `margin` | accuracy fitness is piecewise-constant => near-zero gradient signal for CMA-ES |
| generations | 10,000 | FG saturated ~gen 1400 historically; MVG still moving at 3000 (longest exp_1 run ever). 10k is ~3x that. |
| seeds | 0..4 per arm | |
| constrained arm | `--synaptic-budget S --shrink tau`, **TBD, default S=4 tau=0.9** | exp_1's analogue of KA's fan-in cap; the absolute gate (`--w-threshold`) is a known failure |
| unconstrained arm | no budget, no gate | |
| always | `--no-early-stop --no-balanced --no-open` | `--no-early-stop` is REQUIRED for a fair FG-vs-MVG comparison |

**Expected accuracy: 0.85-0.89, not a solve.** Nothing in exp_1 has ever exceeded
0.885; the monotone-representability ceiling for `retina_ka2005`/and is 0.891.
KA's own paper net got 0.90+-0.03, so this is the same regime — the story is
modularity-at-matched-competence, not "we solved it".

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

(filled in as they run)

---

## 6. Run ledger

(filled in as runs complete; each row = one seed dir under experiments/*/runs/)

---

## 7. Recovery

- All runs write to disk incrementally (`log.csv` per log-interval, champion
  archive per log point, `result.json` at seed end).
- The runner **skips seed dirs that already contain a completed `result.json`**,
  so re-running the same command resumes rather than restarts.
- To see what is running: check section 6 above, then `ls experiments/*/runs/`.
- To resume everything: re-run the commands in section 6 verbatim.
