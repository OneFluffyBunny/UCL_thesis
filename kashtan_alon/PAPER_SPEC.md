# Kashtan–Alon retina — paper specification (what we are reproducing)

Faithful notes on the *actual* Kashtan–Alon experiment, kept separate from our
implementation so the source of truth is auditable. This folder deliberately
uses the **paper's own network type** (a layered feedforward net with discrete
weights, evolved by a mutation GA) instead of the thesis's NDP/g-encoding
framework — the point is a clean external reference for "does MVG favour a
modular brain?".

## Sources
- **Kashtan & Alon (2005)**, *Spontaneous evolution of modularity and network
  motifs*, PNAS 102(39):13773–13778. doi:10.1073/pnas.0503610102 — the MVG result.
  (PNAS full text + arXiv PDF were not fetchable; spec below is the reproducible
  retina/architecture as restated in Clune et al. 2013.)
- **Clune, Mouret & Lipson (2013)**, *The evolutionary origins of modularity*,
  Proc. R. Soc. B 280:20122863. PMC3574393 / arXiv:1207.2743 — open-access, gives
  the exact retina + network numbers below. (Its own *headline* driver is
  connection-cost, not MVG; we take its architecture but Kashtan–Alon's MVG
  manipulation.)

## The task — retina / left–right object recognition
- **8-pixel retina**, split **left 4 / right 4**. Each half may or may not
  contain an "object" (a pattern of interest); the valid patterns differ slightly
  between the two halves. The exact object bit-patterns are **not published**, so
  we reuse the project's existing retina (`experiments/shared_tasks.py`):
  `L = (p0∧p1)∨(p2∧p3)`, `R = (p4∧p5)∨(p6∧p7)` — same left-4/right-4 modular
  structure, and identical to experiments 1–3 so results are comparable.
- **Two goals:** `G_AND = L ∧ R` and `G_OR = L ∨ R`. They share the *same* left
  and right sub-features; only the top-level combiner changes — this is what makes
  them "modularly varying".

## The experiment
- **MVG:** alternate `G_AND ↔ G_OR`, switching **every ~20 generations**
  (Kashtan–Alon 2005). Shared sub-goals ⇒ modularity is expected to emerge.
- **Fixed-Goal (control):** hold one goal for the whole run ⇒ expected non-modular.

## The network (the "actual type of network")
- **Strictly layered feedforward**, layer sizes **8 → 8 → 4 → 2 → 1** (max hidden
  widths 8/4/2). A node in layer *n* connects only to layer *n−1*.
- **Activation** `tanh(λ·z)` with **λ = 20** (steep ⇒ near-sign); output decision
  is `sign(output)`.
- **Discrete weights:** connection weights ∈ integers **{−2, −1, 1, 2}**; a
  connection may also be **absent** (topology evolves). Biases/thresholds ∈
  integers **{−2, −1, 0, 1, 2}** (input layer has no bias).
- Input pixels are bipolar **{−1, +1}** (consistent with the tanh net and the
  rest of the project).

## The search (mutation-based genetic algorithm)
- **Population = 1000**, **25 000 generations**, **asexual (mutation-only)** GA.
- **Mutation operators:**
  - 20% chance per network of **adding** one connection;
  - 20% chance per network of **removing** one connection;
  - each connection: prob **2/n** (n = #connections) of a weight ±1 step;
  - each node: prob **1/24 ≈ 4.16%** of a bias ±1 step.
- Selection: KA-MVG is single-objective (**performance only**). (Clune's NSGA-II +
  connection-cost is *his* modularity driver, deliberately **not** used here.) We
  use tournament (k=3) + one elite — a standard performance-only GA.

## Fitness (the paper's own performance measure)
- **Raw fraction of correct answers** over **all 256 input patterns** — Kashtan–Alon
  use plain accuracy, **no class balancing**. This is our **default** (`--fitness raw`).
- Caveat (CLAUDE.md): under raw accuracy the imbalanced `G_AND` goal has a
  constant-output **shortcut** (always-false ≈ 0.81). `--fitness balanced` (the
  thesis's chance-0.5 measure) is available to sidestep it, but is **not** the
  paper's setting.

## Confirmed vs. reconstructed (source audit)
- **Confirmed directly from Kashtan–Alon 2005:** MVG alternates two goals sharing
  sub-goals, **switching every 20 generations**; performance = fraction correct.
- **Confirmed from Clune 2013's explicit KA replication:** arch **8-8-4-2-1**,
  `tanh(λ=20)`, weights `{−2,−1,1,2}`, biases `{−2..2}`, pop **1000**, **25 000**
  gens, add/remove **20%**, weight ±1 at **2/n**, bias ±1 at **1/24**.
- **Reconstructed (unpublished in KA):** the exact retina object bit-patterns — we
  use the project's `L=(p0∧p1)∨(p2∧p3)`, `R=(p4∧p5)∨(p6∧p7)` (same modular split).

## Modularity measurement
- **Newman's Q** on the network graph: partition nodes into modules, then measure
  (edges within modules) − (edges expected at random). Modular networks score
  **Q ≈ 0.4+**; non-modular **Q ≈ 0.15–0.2** in the paper.
- We compute Q on the **undirected connection graph** over all neurons (I/O
  included), via a greedy-modularity community split (networkx).

## What "success" looks like here
Q rises and stays high under **MVG** but stays low under **Fixed-Goal**, with MVG
also reaching the task faster — reproducing Kashtan–Alon's central claim that
*changing tasks with shared sub-goals spontaneously favour a modular brain*.
