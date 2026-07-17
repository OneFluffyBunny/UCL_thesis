# UCL_thesis — project context

Read this first. It orients any session. **The remote GPU box only sees this git
repo — auto-memory does NOT travel through git — so everything a remote session
needs lives here or in the linked `.md` files.** This file *points*; the detail
lives in the linked files (don't duplicate them).

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
