# UCL_thesis — project context

Read this first. It orients any session. **The remote GPU box only sees this git
repo — auto-memory does NOT travel through git — so everything a remote session
needs lives here or in the linked `.md` files.** This file *points*; the detail
lives in the linked files (don't duplicate them).

## ⚠️ Branches and what is actually saved — read before any git command (2026-09-15)
There is **exactly one** branch: **`main`**. On 2026-09-15 `main` was fast-forwarded
to `fluffy_experiments`, and `fluffy_experiments` and `spec-modularity` were then
**deleted**, local and remote (no commit was lost — both were ancestors of `main`).
Earlier, `CGP`, `cgp_speedups` and `qmetrics` went the same way into
`fluffy_experiments`. **If a session still believes it is on any of those branches, it
is not** — the checkout is shared, so its HEAD already moved. Re-read the branch with
`git rev-parse --abbrev-ref HEAD` rather than trusting remembered context, and do not
recreate the deleted names. (Older notes in this repo that say "`main` is frozen" or
"work on `fluffy_experiments`" predate this.)

- **All experiments and all work go on `main`.** Fork a short-lived branch if you want
  one, and fast-forward it back.
- The NDP subtree under `NDP/` is a *squashed* import. The standalone repo's full
  history (branch `KA_experiments`, commit `01c800b`) is saved on GitHub as the tag
  **`ndp-standalone-history`**.
- **`git push` only sends COMMITTED files. An untracked file is on nobody's branch
  and no push will ever save it.** This is not theoretical: experiments 5 and 6 —
  two complete experiments with their test suites, ~11k lines — sat untracked for
  weeks, on one laptop only, and were invisible to the GPU box (which sees nothing
  but this repo). Committed 2026-08-20. **`git add` the files you create.** After
  finishing a unit of work, run `git status` and look at the `??` lines: anything
  there is unsaved.
- Deliberately NOT saved, by design: `runs/` (regenerable — conclusions belong in
  `RESULTS.md`), `*.png` under `experiments/` (put figures worth keeping in
  `latex_figures/`), `papers/`, `.venv-pypy/`, `LNDP/`, `NDP/saved_models/`.
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
- **experiment_4/ — CGP vs ECGP on the KA retina.** Boolean circuits, pure Python,
  no JAX. Does module *re-use* (ECGP's compress/expand) actually happen, and does it
  help under MVG? ⚠️ **FROZEN**: its logged runs are identified by seed, so its
  search must not change. See its `RESULTS.md`.
- **experiment_5/ — big brains (FORK of exp_4).** Many inputs *and many outputs*,
  asking whether **behavioural** modularity emerges (which inputs actually move each
  output) as opposed to exp_4's structural cone readout. Runs headless under **PyPy**;
  `test_equivalence.py` proves it is exp_4's algorithm and that PyPy and CPython give
  byte-identical runs. Machinery done, science not started. ⚠️ One exception:
  `SPECIALISATION.md`, `ENTRENCHMENT.md` and `CALIBRATION.md` are **AI-authored
  sub-studies** (Claude Code, 2026-08-21..28, branch `spec-modularity`) — preregistered,
  run, and **not reviewed by a human**. Quarantined in those three files on purpose; do
  not cite from them. Net result if you read only one thing: `CALIBRATION.md` shows the
  `SPEC` metric is **92% determined by the task** (eta^2 = 0.917) and that a *random*
  circuit already scores 0.92 on a 2-output task — so SPEC works as a description of a
  solved circuit but has almost no room to respond to a treatment.
- **experiment_6/ — nested modules.** ECGP (exp_4) explicitly forbids a module
  containing a module; this experiment exists to build a variant that allows
  nesting *with its complexity priced*, so evolution can be tested for whether it
  keeps/reuses/builds-on modules rather than just calling flat ones. Self-Modifying
  CGP (Harding/Miller/Banzhaf 2009, a developmental graph-rewriting mechanism, NOT
  the same claim as nested named modules) is implemented and verified to search
  correctly. The actual nested-ECGP target is un-built — Modular CGP's own paper
  (Walker's thesis; Springer ch. 3) is paywalled/unreachable, so it will be an
  original extension (leading idea: nesting gated by a decaying probability), not a
  reproduction. See its `README.md` (framing), `PAPER_SPEC.md` (SMCGP spec, every
  claim tagged verbatim/inferred/our-choice), `RESULTS.md`.

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
- `experiments/experiment_4/README.md` + `PAPER_SPEC.md` — the ECGP spec and the
  CGP-vs-ECGP framing; `RESULTS.md` is the notebook.
- `experiments/experiment_5/README.md` — big-brain arm: task families, the
  structural-vs-behavioural measurement, and **which interpreter to use**.
- `experiments/experiment_6/README.md` + `PAPER_SPEC.md` — the nested-modules
  framing and the SMCGP spec; `RESULTS.md` is the notebook.
- `experiments/HISTORY.md` — the pre-migration git history (2 old commits).

## Established facts (don't relitigate)
- **Check every task for shortcuts/imbalance before trusting a result.** retina/AND
  has a one-side shortcut worth 0.848 balanced acc → its "0.85 plateau" was a
  trivial one-module cheat. Prefer retina/**xor** (balanced, no shortcut).
- Curriculum-vs-cold experiments so far are **nulls** (no speed or modularity gain).
- When a run stalls below the task we cannot yet separate *representability*
  (encoding can't express it) from *reachability* (search didn't find it).
- ~~The #1 missing tool is a modularity METRIC~~ — **BUILT.** `qmetrics/` (Newman Q,
  KA's normalized Q_m, planted left/right split, circuit purity, Infomap) plus
  `experiments/shared_brain_metrics.py`, which wraps the four of them for a
  RECURRENT (N,N) brain and adds `recurrent_purity` — `qmetrics.circuit_purity`
  raises on a cycle, so it cannot run on any brain in `experiments/`. Score a study
  with `experiments/analysis/run_all.py --root <runs dir>`.
- **Read density before reading any modularity number.** An unconstrained arm
  converges to 94-100% density, where a complete graph has no communities to find
  and no sparser null to compare against: Q_m and the left/right score come back
  `nan` and purity is 0.000 by construction. That is *unanswerable*, not
  *unmodular*. The synaptic budget is what makes the question askable.
- **PyPy is not a free win — it has a crossover.** In exp_5 (boolean circuits, one
  truth-table integer per wire) PyPy is 3-6x faster below ~14 program inputs and up to
  5x *slower* above it: CPython's big-integer bitwise ops are hand-written C, so once
  one op costs more than the interpreter overhead around it PyPy's advantage is gone.
  Measure with `experiments/experiment_5/bench.py --crossover` on any new machine
  before choosing. Wide tasks = CPython; narrow tasks = PyPy.
- **Exhaustive truth-table evaluation caps out around 20 inputs.** A wire is a
  `2**n_in`-bit int: 8 KB at 16 inputs, 2 MB at 24. No interpreter fixes that; the only
  way past it is scoring a sampled subset of patterns, which changes what fitness means.

## Conventions
- Stack: JAX, Equinox, Optax, evosax (CMA_ES). Balanced accuracy (chance = 0.5),
  bipolar inputs {-1,+1}.
- Run Python via the conda env: `conda run -n lndp python ...` (the terminal does
  not persist conda activations, so prefix every run). ⚠️ `conda run` cannot take a
  `python -c` script containing newlines — write a file instead.
- Experiment 5 additionally has a PyPy venv, built by
  `conda run -n lndp python experiments/experiment_5/setup_pypy.py` (works on the
  Linux GPU box too). It has no packages and needs none: exp_5's search is
  stdlib-only. Diagrams stay a CPython job (`render.py`).
- Experiment outputs go to `runs/` (gitignored — regenerable; conclusions go in
  RESULTS.md).
- `LNDP/` (the abandoned original framework) is gitignored, not part of this repo.
- `NDP/` IS tracked (since 2026-09-15): a `git subtree --squash` import of the standalone
  NDP repo at `C:\Users\raduc\programare\NDP` (full history, branch `KA_experiments`).
  Credit to the original authors is at the top of `NDP/readme.md`. Sync with
  `git subtree push --prefix=NDP <standalone> KA_experiments` (edits made here) or
  `git subtree pull --prefix=NDP <standalone> KA_experiments --squash` (edits made there).

## Remote (UCL GPU lab)
- GitHub: `https://github.com/OneFluffyBunny/UCL_thesis`
- Machine `shoveler-l.cs.ucl.ac.uk` via jump host `knuckles.cs.ucl.ac.uk` (user
  `araducea`). Default shell is **csh** — run `bash` first.
- Permanent store (use for everything): `/cs/student/project_msc/2025/ml/araducea/`.
- Workflow: develop locally → push → pull on remote → run experiments there.
