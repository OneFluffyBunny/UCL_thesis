# UCL_thesis — project context

Read this first. It orients any session. **The remote GPU box only sees this git
repo — auto-memory does NOT travel through git — so everything a remote session
needs lives here or in the linked `.md` files.** This file *points*; the detail
lives in the linked files (don't duplicate them).

## ⚠️ Branches and what is actually saved — read before any git command (2026-08-20)
There are **exactly two** branches: `main` and `fluffy_experiments`. `CGP`,
`cgp_speedups` and `qmetrics` were **deleted**, local and remote, after being folded
into `fluffy_experiments` by fast-forward (no commit was lost — each was already an
ancestor of the tip). **If a session still believes it is on `CGP` or `cgp_speedups`,
it is not** — the checkout is shared, so its HEAD already moved. Re-read the branch
with `git rev-parse --abbrev-ref HEAD` rather than trusting remembered context, and
do not recreate the deleted names.

- **`main` is frozen — do not commit to it, do not fast-forward it.** It is a
  stable marker, not a working branch.
- **All experiments and all work go on `fluffy_experiments`**, or on a branch forked
  from it (fork freely; just never target `main`).
- **`git push` only sends COMMITTED files. An untracked file is on nobody's branch
  and no push will ever save it.** This is not theoretical: experiments 5 and 6 —
  two complete experiments with their test suites, ~11k lines — sat untracked for
  weeks, on one laptop only, and were invisible to the GPU box (which sees nothing
  but this repo). Committed 2026-08-20. **`git add` the files you create.** After
  finishing a unit of work, run `git status` and look at the `??` lines: anything
  there is unsaved.
- Deliberately NOT saved, by design: `runs/` (regenerable — conclusions belong in
  `RESULTS.md`), `*.png` under `experiments/` (put figures worth keeping in
  `latex_figures/`), `papers/`, `.venv-pypy/`, `LNDP/`, `NDP/`.
- `scratch_*.py` are untracked by convention. That is a choice, not an accident —
  but it does mean they are one `git clean` away from gone.

## What this project is
Master's thesis (UCL): the **emergence of modularity** in evolved neural networks.
Thesis: modularity is **selected for by evolution, precedes learning, and is
encouraged by a compressed DNA→brain encoding (a genomic bottleneck)**.

## Hard constraints (define the model — never violate)
- **No physical space** for neurons (no coordinates/distance) — modularity must
  come from something other than wiring-length cost.
- **Fixed neuron count** — only *connections* evolve.
- **Parsimony first** — the smallest model that can show the effect.

## The experiments (all under `experiments/`)
Two axes are varied one at a time: exp_1↔exp_2 vary the **encoding**; exp_2↔exp_3
vary the **optimiser** (exp_2 and exp_3 share the *same* direct-encoding model,
`shared_direct_model.py`).
- **experiment_1/ — treatment (compressed g-encoding).** Genome ≈ O(K): K
  cell-type identities + one shared connection rule `g`; weight of edge i→j =
  `g(feat_i, feat_j)`. Search is confined to a low-dim manifold of *regular*
  networks → structural bias toward modularity.
- **experiment_2/ — control (direct encoding, CMA-ES).** Genome IS the raw weight
  vector (one free number per edge). No rule, no sharing — a Kashtan-Alon-style
  baseline the bottleneck is measured against. Same tasks / CMA-ES loop / metrics;
  only the encoding differs from exp_1.
- **experiment_3/ — optimiser control (direct encoding, gradient descent).** Same
  network as exp_2, trained by backprop + Optax instead of CMA-ES. Loss is a
  differentiable surrogate (`margin`, the exact one CMA-ES maximises → fair
  head-to-head; or `bce` → gradient-oracle bound); accuracy stays the reported
  (non-differentiable) metric. Isolates "how much does the gradient help?".

## External reference reproduction (`kashtan_alon/`)
Not one of the three experiments — a **faithful reproduction of Kashtan–Alon 2005**
(the retina MVG→modularity result), deliberately outside our NDP/g-encoding
framework. Rebuilt 2026-08-03 to the *verified* paper spec (PMC1236541): their
network retina(8)→8→4→2→1, **±1 weights**, hard-**threshold** units, fan-in ≤3/≤2,
their **real Fig. 5a retina task** (NOT the `shared_tasks.py` stand-in), and their GA
(**elite 150/600 + crossover Pc=0.5 + mutation Pm=0.5**). Modularity is KA's
normalized **Q_m** (`kashtan_alon/modularity.py`, framework-agnostic → promotable to
shared). ⚠️ Runs 1–3 accidentally ran *Clune 2013's* reimplementation and produced a
null; see `RESULTS.md` + `PAPER_SPEC.md` + memory [[reference_ka_retina_algo]] before
touching this. Run with `conda run -n lndp python kashtan_alon/run_paper.py` (the
"exactly as in the paper" command); see `kashtan_alon/README.md`.

## Where the detail lives (read these; don't restate them here)
- `experiments/experiment_1/RESULTS.md` — the lab notebook: every run, the
  shortcut-trap caution, the representability-vs-reachability findings, open threads.
- `experiments/README.md` — full experiment-1 encoding spec.
- `experiments/experiment_2/README.md` — direct-encoding control framing + `RESULTS.md`.
- `experiments/experiment_3/README.md` — GD-vs-EC optimiser control; differentiability notes.
- `experiments/HISTORY.md` — the pre-migration git history (2 old commits).

## Established facts (don't relitigate)
- **Check every task for shortcuts/imbalance before trusting a result.** retina/AND
  has a one-side shortcut worth 0.848 balanced acc → its "0.85 plateau" was a
  trivial one-module cheat. Prefer retina/**xor** (balanced, no shortcut).
- Curriculum-vs-cold experiments so far are **nulls** (no speed or modularity gain).
- When a run stalls below the task we cannot yet separate *representability*
  (encoding can't express it) from *reachability* (search didn't find it).
- **The #1 missing tool is a modularity METRIC** (Newman Q / Infomap on the grown
  adjacency) — build it before any new architecture; it's the thing we can't yet measure.

## Conventions
- Stack: JAX, Equinox, Optax, evosax (CMA_ES). Balanced accuracy (chance = 0.5),
  bipolar inputs {-1,+1}.
- Run Python via the conda env: `conda run -n lndp python ...` (the terminal does
  not persist conda activations, so prefix every run).
- Experiment outputs go to `runs/` (gitignored — regenerable; conclusions go in
  RESULTS.md).
- Siblings `LNDP/` (the abandoned original framework) and `NDP/` are gitignored,
  not part of this repo.

## Remote (UCL GPU lab)
- GitHub: `https://github.com/OneFluffyBunny/UCL_thesis`
- Machine `shoveler-l.cs.ucl.ac.uk` via jump host `knuckles.cs.ucl.ac.uk` (user
  `araducea`). Default shell is **csh** — run `bash` first.
- Permanent store (use for everything): `/cs/student/project_msc/2025/ml/araducea/`.
- Workflow: develop locally → push → pull on remote → run experiments there.
