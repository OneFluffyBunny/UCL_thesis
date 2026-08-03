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
  **PRIMARY SOURCE, verified 2026-08-03 by fetching the full methods from
  PMC1236541.** Every value in the "KA 2005 (verified)" columns below is quoted from
  that text. This is the ONLY paper we are replicating.
- **Clune, Mouret & Lipson (2013)**, *The evolutionary origins of modularity*,
  Proc. R. Soc. B 280:20122863. PMC3574393 / arXiv:1207.2743 — a *later, different*
  reimplementation. Its headline driver is connection-cost, not MVG. **We do NOT
  follow it.** Our current code accidentally inherited its numbers (8-8-4-2-1, tanh,
  ±2/±1 weights, mutation-only) — that is the bug to fix, listed as "wrong" below.

## The task — retina / left–right object recognition (KA 2005, Fig. 5a — verified)
- **Retina = 4 pixels wide × 2 pixels high = 8 pixels**, each 0/1. Split into a
  **left 2×2 block** (4 pixels) and a **right 2×2 block** (4 pixels).
- **Left object** present iff the four left pixels **≥3 are black**, OR exactly
  **1–2 black pixels lying entirely in the left column** of that block. **Right
  object**: the symmetric rule with "**right column** only". (Verbatim: *"A left
  object is defined by three or more black pixels or one or two black pixels in the
  left column only"*; right *"defined in a similar way, with one or two black pixels
  in the right column only"*.) Both are exact Boolean functions of 4 inputs,
  enumerable over all 256 patterns.
- **Two goals:** `G1 = L AND R` and `G2 = L OR R` — shared L/R sub-features, only
  the top combiner changes (this is what makes them modularly varying).
- ⚠️ **Our code does NOT use this task yet.** It uses the stand-in
  `L=(p0∧p1)∨(p2∧p3)`, `R=(p4∧p5)∨(p6∧p7)` (shared with experiments 1–3). For exact
  KA replication the task must be swapped to the definition above. The pixel→input
  mapping (which of the 8 inputs are "left" vs "right") is shown only in Fig. 5a;
  we take left = the 4 left-block pixels, right = the 4 right-block pixels.

## The experiment
- **MVG:** alternate `G_AND ↔ G_OR`, switching **every ~20 generations**
  (Kashtan–Alon 2005). Shared sub-goals ⇒ modularity is expected to emerge.
- **Fixed-Goal (control):** hold one goal for the whole run ⇒ expected non-modular.

> 🛑 **MAJOR CORRECTION (2026-08-03), VERIFIED against the primary source, then the
> code was rebuilt to match.** The "network"/"search" values previously here described
> **Clune 2013's reimplementation, NOT Kashtan–Alon 2005** (verified via PMC1236541).
> The code (`model.py`, `ga.py`, `tasks.py`, `train.py`, `run_paper.py`) was rewritten
> on 2026-08-03 to the values below. NB: an intermediate edit of this file wrongly
> claimed KA's architecture was "8-4-2-1" — that was a misread; the retina's 8 pixels
> are a **separate input layer** (*"connections from the retina were to the first layer
> only"*), so the node graph is **8-8-4-2-1**, which the code already had.

## The network (KA 2005 retina — verified, now implemented)
| | **KA 2005 (verified)** | status in code |
|---|---|---|
| Node graph | **retina(8) → 8 → 4 → 2 → 1** (8 pixels feed the first neuron layer) | ✅ matches |
| Weights | **−1 or +1** (0 = absent) | ✅ `WEIGHT_VALUES=(-1,1)` |
| Units | hard **threshold**: fire iff weighted sum + bias(=−θ) > 0, output {0,1} | ✅ |
| **Fan-in limit** | **≤3 incoming** for neuron layers 1–3, **≤2** for the output (layer 4) | ✅ `fan_in=(3,3,3,2)` |
| Inputs | pixel values **{0,1}** | ✅ `--input-encoding binary` |
| Size penalty | 0.01 fitness per neuron above a base count (SI; ambiguous base) | ⚠️ not implemented (fan-in already caps size) |

## The search (KA 2005 retina GA — verified, now implemented)
- **Population S = 600.** Reproduction uses **CROSSOVER, Pc = 0.5** (per-destination-
  neuron column inheritance, `ga.reproduce`) — the mechanism that recombines modular
  building blocks (left-/right-object detectors); the mutation-only port lacked it.
- **Mutation Pm = 0.5 per genome** — one random edit: add/remove edge (respecting the
  fan-in cap), flip a weight's sign, or nudge a threshold. ⚠️ The exact operator set
  and threshold range are in KA's **SI** (not the main text); this is a documented
  faithful reconstruction, refinable against the SI.
- **Selection:** elite strategy — top **L = 150 of 600** replicate unchanged; the other
  450 are offspring of the elite.
- **Goal switch** every **E = 20** gens. Reported: MVG solves in ~2,800 gens, fixed-goal
  ~21,000. (KA's *circuit* task uses S=1000, L=300, Pm=0.7, NAND gates — a DIFFERENT
  system; do not mix its numbers into the neural-net retina.)

## Fitness (the paper's own performance measure)
- **Raw fraction of correct answers** over **all 256 input patterns** — Kashtan–Alon
  use plain accuracy, **no class balancing**. This is our **default** (`--fitness raw`).
- Caveat (CLAUDE.md): under raw accuracy the imbalanced `G_AND` goal has a
  constant-output **shortcut** (always-false ≈ 0.81). `--fitness balanced` (the
  thesis's chance-0.5 measure) is available to sidestep it, but is **not** the
  paper's setting.

## Confirmed vs. reconstructed (source audit — VERIFIED 2026-08-03 from PMC1236541)
- **Quoted directly from KA 2005, retina:** task = **4×2 retina, L/R objects per
  Fig. 5a** (see task section); goals **L AND R ↔ L OR R** switching every **20**
  gens; node graph **8-8-4-2-1** (8 retina pixels → 8→4→2→1 neurons); weights **±1**;
  threshold units; **fan-in ≤3/3/3/2**; pop **600**; **crossover Pc=0.5** + mutation
  Pm=0.5; elite **150/600**; fitness = fraction correct; modularity = normalized
  **Q_m** (Q_rand over **1000** randomizations) with MVG **0.35±0.02** vs FG **0.15±0.02**.
- **From CLUNE 2013 (a *different* reimplementation), do NOT attribute to KA:** the
  8-8-4-2-1 arch, tanh(λ=20), weights {−2,−1,1,2}, add/remove-20%/2n/1-24 mutation,
  and the connection-cost driver. Our code currently follows these — the bug.
- **Only thing NOT numerically published (given as Fig. 5a picture):** the exact
  pixel→input index mapping. The object *rules* ARE given (quoted above); we take
  left = the 4 left-block pixels, right = the 4 right-block pixels.

## Modularity measurement (KA 2005 Eqs. 1–2 — verified)
- **Normalized Q_m** = (Q_real − Q_rand) / (Q_max − Q_rand), where Q_real is Newman's
  Q of the best partition, **Q_rand = mean Q over 1000 degree-preserving randomized
  networks**, Q_max = max Q at the same degree sequence. Reported: MVG **0.35±0.02**
  vs FG **0.15±0.02** (NN); circuits MVG 0.54 vs FG 0.12.
- Implemented in `modularity.normalized_qm` (currently n_rand=100 — bump to 1000 to
  match). Raw Newman Q alone is density-confounded and must NOT be used for the
  MVG-vs-FG comparison (see RESULTS.md Run 2 retraction).

## What "success" looks like here
Q rises and stays high under **MVG** but stays low under **Fixed-Goal**, with MVG
also reaching the task faster — reproducing Kashtan–Alon's central claim that
*changing tasks with shared sub-goals spontaneously favour a modular brain*.
