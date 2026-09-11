# Content for LaTeX writeup

Scratch notes destined for the thesis. Not polished prose — just the facts/claims
to work into the final writeup, with enough detail to write them up correctly later.

---

## Biological importance / motivation

Running log of motivating facts for why modularity matters biologically —
grounds the thesis's premise that it's worth evolving toward. More to add.

- The human brain is modular (structurally and functionally segregated
  circuits), and there is evidence this modularity is what makes it more
  **adaptable** (evolvable/evolvable-to-new-tasks).
- **What we are modelling is the brain at birth — before learning.** The object
  of study is *innate* circuitry: the structure the genome specifies, prior to
  any experience-driven refinement. This is why the thesis claims modularity
  *precedes* learning rather than being produced by it, and why our brains are
  scored as-grown rather than after a training phase.
- **Real neurons never get free wiring — a biological constraint is whatever
  makes connectivity scarce, not any one specific mechanism.** Axon growth and
  maintenance cost energy and material (wiring-length cost, à la Clune et al.
  2013); a neuron has a finite amount of dendritic/somatic surface for synapses
  to land on (roughly, a fan-in-style cap, à la Kashtan-Alon's model); and
  synaptic strength is homeostatically regulated so that a neuron's *total*
  input drive is held within a range rather than growing without bound
  (synaptic scaling — Turrigiano 2008 is the canonical review — which is the
  direct biological analogue of our synaptic-strength budget). These three
  mechanisms look unrelated at the implementation level, but they share the
  same abstract role: **they make connectivity a scarce, competed-for resource
  instead of a free good.** That scarcity is plausibly the actual ingredient
  evolution needs to select for modularity — not any one of length, count, or
  strength specifically, but the fact that *some* budget forces circuitry to be
  reused/shared rather than freely duplicated when a goal changes. See "Testing
  whether a constraint is necessary for modularity" below for the evidence.

---

## High-level hypotheses and abstractions

Running log of cross-cutting hypotheses/hunches — not established results, flag
as untested until backed by a controlled comparison. More to add.

- **Hypothesis: constraints on connectivity may be critical for both sparsity
  and modularity of the grown brain.** Impression from running NDP, the
  `kashtan_alon/` reproduction, and experiment_1: left unconstrained, growth/
  search is prone to converging on a dense, tangled brain rather than a sparse
  modular one (consistent with the `qmetrics` density-confound note — a dense
  graph makes Q ≈ 0 nearly unavoidable regardless of wiring). Not yet isolated
  as a controlled variable across all three.
- ~~**Worth testing: does constraining also make search converge faster**~~
  **ANSWERED, and the answer is the other way round** (2026-09-11, KA fan-in
  ablation, 5 seeds/arm). Constraining makes search **much slower**: uncapped MVG
  reaches a perfect score on all five seeds inside 400 generations, capped MVG
  manages it twice in 25,000, and capped FG never passes 0.95. See "KA with the
  fan-in cap removed" below for the full threshold table.
- ~~**Hunch for *why*, if the speed effect holds**: constraints shrink the search
  space, and the solution happens to sit inside the constrained subspace~~ —
  **the measurement contradicts this too.** The unconstrained search is not
  wasting budget covering a larger space; it finds a dense, entangled,
  high-accuracy solution almost immediately. What the constraint does is *remove
  the easy needles*, leaving only solutions that have to economise on wiring — and
  those are the modular ones. Restated: **scarcity buys modularity and pays for it
  in both accuracy and search time.** That is a cost the thesis should state
  openly rather than a free lunch.

---

## Evolutionary computation methods

Optimisers used to search the DNA/genome across this thesis. So far: **CMA-ES**
only (NDP, experiment_1, experiment_2 all use it; experiment_3 uses gradient
descent instead, as its own optimiser-control axis).

- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** — population-based,
  derivative-free optimiser for continuous parameter vectors. Maintains a
  multivariate Gaussian (mean, covariance matrix, step size σ) over the search
  space; each generation samples `popsize` candidates, evaluates their fitness,
  and updates the mean/covariance/σ toward the better-ranked samples. The
  covariance adapts to the local curvature of the fitness landscape, letting
  the search stretch/rotate to follow narrow valleys instead of only searching
  axis-aligned. Elitist variant used here (`CMA_elitist`), so the best-found
  solution persists across generations.

---

## CMA-ES vs gradient descent

*(Placeholder — we'll come back to this later and expand it.)*

- **When the fitness function is differentiable, gradient descent is simply
  better.** CMA-ES only estimates a search direction from `popsize` fitness
  samples per generation; GD reads the exact direction off the backward pass at
  the cost of roughly one evaluation. Paying for a derivative-free optimiser is
  only justified when the objective genuinely isn't differentiable (or the
  differentiable surrogate is a poor proxy for it).
- This is exactly what **experiment_3** isolates: same direct-encoding network
  as experiment_2, backprop + Optax instead of CMA-ES, with `margin` as the
  differentiable surrogate for the accuracy CMA-ES maximises.

---

## Original NDP architecture (Najarro et al. 2023, arXiv:2307.08197)

- NDP = a small "DNA" MLP that grows a policy graph (the "brain") via iterative
  local message-passing. Weight-sharing / cellular-automaton style: the same
  MLP is applied identically at every node.
- Growth cycle (repeated `number_of_growth_cycles` times): propagate node
  embeddings → growth MLP decides spawn/no-spawn per node → new node's
  embedding = mean of its parent neighbourhood's embeddings → weight MLP sets
  edge weight from the two endpoint embeddings → optional pruning.
- Seed size: the original paper starts from **1 node**, for every environment
  including I/O tasks like CartPole (verified against original commits
  `3e4591c`/`a107e57`) — not an `obs_dim + action_dim` skeleton.
- Outer loop: CMA-ES evolves the flat DNA parameter vector. Fitness = mean
  reward over `nb_growth_evals` growth runs × `nb_episode_evals` rollouts.
- Original I/O convention: purely *positional* — first `obs_dim` / last
  `action_dim` node slots are read as input/output only at rollout time, never
  seeded or marked during growth itself. (Not confirmed whether the paper's
  own text calls this a limitation — only the code has been checked.)

---

## Our changes to the framework

Running log — add one entry per change, most recent last.

### 1. Input/output differentiation (I/O-anchor redesign)

**Why**: the original positional convention read output as
`network_state[-action_dim:]` — literally "whichever nodes are currently
last." Since growth appends new nodes to the end of the list every cycle,
every neurogenesis event silently reassigned which node counted as "the
output." The brain was brittle: output identity kept shifting mid-growth
instead of staying pinned to specific nodes.

**Change**: fixed input/output anchor nodes (indices `0..obs_dim-1` always
inputs, `obs_dim..obs_dim+action_dim-1` always outputs), enforced
structurally so growth can never displace them; separate evolved input-role
vs output-role embeddings (`has_io_roles`) so the growth MLP can condition on
I/O identity from the first growth cycle, instead of inferring it
positionally; edge sign (excitatory/inhibitory) made an explicit, plotted
property of the grown graph rather than an unsigned weight.

<!-- 2. (next change goes here) -->

### 2. Kashtan-Alon retina task support

`--balanced-fitness` (fitness + size-reg + target all scale consistently);
`forbid_io_self_edges` (default on, masks I-I/O-O edges out of the seed,
permanent and zero genome cost); MVG (`--mvg`, tracks the final generation's
champion rather than best-ever, since best-ever isn't comparable across a
goal switch); `--pruning-threshold` (existed, was never exposed). Also fixed: `io_ratio`/`io_edges` size-reg wasn't scaled by max reward like
the other size penalties were, so the same `alpha` meant very different
things depending on task reward scale; and the "unpromising run" early-stop
threshold was tuned for raw-reward scales and always fired at gen 500 on a
`[0,1]` balanced-fitness run, silently discarding the run's logs/snapshot —
`--no-early-stopping` opts out.

---

## NDP: general observations

- **The seed is complex and results are hard to control.** Even a "1-node"
  seed carries a full evolved embedding plus a growth/weight-MLP pair applied
  identically everywhere; small changes early in growth compound through every
  later cycle (propagate → grow → reweight), so final brain size/shape is
  sensitive to the seed and hard to predict or steer directly from the DNA.
- **For harder tasks the brain can explode in size, and regularisation
  doesn't always help.** Seen directly in the LunarLander runs
  (`fluffy_experiments.md`): brain-size explosion was the dominant failure
  mode before `io_ratio` size regularisation was added (Run 4+), and even
  after adding it, the regulariser could be gamed — DNA staying at the
  minimum node count to dodge the size penalty rather than improving the raw
  task score, rather than converging on a well-sized, well-performing brain.
- **Neurogenesis (as modelled here) is biologically unrealistic.** Growth
  adds whole new nodes with weights set from a global-ish learned MLP applied
  identically everywhere, all within a handful of discrete growth cycles per
  lifetime — a highly simplified stand-in for real developmental
  neurogenesis, which is local, activity-dependent, and continuous rather
  than cycle-synchronous and network-wide.
- **Growth is density-compounding by construction.** A spawning child
  inherits its *entire parent's neighbourhood*, not just the parent
  (`add_new_nodes()`) — hubs get copied onto every child spawned near them,
  compounding each cycle. With no distance/wiring cost in this project,
  growth itself is one of the few things that could push toward sparse
  structure, and by default it pushes the opposite way. Counter-levers exist
  but are off by default: `--pruning`, lower `initial_sparsity`,
  `node_pairs_based_growth`.
- **Shared per-role (not per-node) seed embeddings make some functions
  unreachable, regardless of search budget.** All input nodes get the same
  embedding vector (`build_initial_network_state`), so the network can only
  ever compute permutation-invariant functions of its inputs (blind to
  *which* inputs are on, only *how many*). No embedding-size/MLP-width/
  generations tweak fixes this — needs per-input, not per-role, embeddings.
- **Evolved edge-weight diversity is a cheap diagnostic for a degenerate
  shortcut.** Every retina/KA solution so far collapses to near-uniform edge
  magnitude (sign-only differentiation); CartPole's solver doesn't (real
  spread, no saturation). Worth checking on any new result.

---

## LunarLander: unsolved, on this fork and (per the paper's own account) originally

This architecture has not solved LunarLander-v3 (threshold reward 200) on this
fork. Logged in `fluffy_experiments.md`: 8 runs total, spanning no
regularisation, `io_ratio` size regularisation, edge regularisation,
elitism on/off, sigma_init 0.1–1.0, popsize up to 512. Best result across all
8: Run 5, **+45.0 average over 100 eval episodes**, with very high variance
(3-episode render: `134.9, 57.6, −129.3`). Every other run landed between
−150 and −37. Recurring failure modes: brain-size explosion (fixed by
`io_ratio` reg from Run 4 on), premature sigma/CMA-ES convergence under
elitism, and the regulariser being gamed (DNA staying at minimum node count
to dodge the size penalty rather than improving the raw task score).
*(The original paper is also reported to not have solved LunarLander — not
independently re-verified against the paper's text this session, only noted
per your own recollection.)*

---

## Kashtan-Alon-style retina task (NDP)

Ported into NDP (`ka_task.py`, branch `KA_experiments`): single-output,
bipolar, sign-of-output, target = `left_feature AND right_feature`.

- **Two different task variants exist** — don't conflate them: the
  `shared_tasks.py` stand-in (used by experiment_1/2/3, now NDP) has L and R
  as the **same logical expression** mirrored on two pixel blocks; the
  `kashtan_alon/` faithful reproduction uses a different formula. Different
  class balance, note which one a given result used.
- **Often use balanced accuracy, not raw accuracy** — raw accuracy is
  deceptive here (task is imbalanced, ~19% positive), so "always predict 0"
  scores ~81% for free with no gradient toward the real solution. (NDP:
  `--balanced-fitness` flag, default off; experiment_1 defaults on.)
- **Both FG and MVG converge to the same ~84% "popcount" solution and can't
  improve on it.** Every evolved network we've inspected (regularised,
  unregularised, FG, MVG) reduces to an unweighted majority vote over the 8
  raw input bits — ignores which specific bits are on, just counts how many.
  Gets the popcount-0-3 and popcount-7-8 bands perfect for free (structurally
  can't be wrong there) and is near coin-flip on the popcount-4-6 band, where
  the task's actual pairwise AND/OR structure would be required. See
  `experiments_paper/retina/RESULTS.md` in NDP for the full derivation.

---

## LNDP: general observations

Running log — brief for now, more to add.

- **Fixed node count — only connections change.** Unlike NDP, LNDP never adds or
  removes neurons; a lifetime consists of synaptogenesis and pruning over a fixed
  set of nodes. This is the **more biologically realistic** choice: the mammalian
  brain does comparatively little neurogenesis after development, and shapes
  itself overwhelmingly by making and eliminating *synapses*.
- **Like NDP, it is not a compression** — the genome is *larger* than the
  phenotype it specifies. The rules (graph transformer + node/edge GRUs + prune
  and synaptogenesis MLPs) cost ~1.9k parameters, while a 24-node brain has only
  ~322 possible edges. The genome is O(1) in node count and the phenotype is
  O(N²), so the encoding only becomes a genuine bottleneck **above a crossover
  node count** — i.e. unless we start with a very large number of nodes.
- **Even the original paper struggles on simple control tasks, at great compute
  cost.** Its whole suite is toy control (CartPole, Acrobot, Pendulum, a 5-cell
  foraging grid; observation dim ≤ 8), and reaching those scores takes 10,000
  generations × popsize 128 × 3 trials ≈ 4M episodes for problems standard RL
  solves in minutes. The paper itself concedes performance is below conventional
  RL architectures and that scaling to higher-dimensional tasks is open.
- **The architecture is clunky and hard to experiment with.** A graph transformer,
  two GRU rules, two threshold MLPs and a spontaneous-activity process are all
  entangled, so most interventions touch several coupled components at once and
  the effect of any single change is hard to isolate. Poor substrate for the
  controlled one-variable-at-a-time comparisons this thesis needs.
- **The retina task and this architecture are mismatched.** LNDP's mechanism is
  reward-modulated plasticity *during* the lifetime, but the retina target is a
  static 256-row truth table. So (i) the map is fully known at evolution time —
  anything the plastic network achieves, a frozen weight matrix could achieve;
  (ii) the reward is one bit ("was the previous pattern right?") over i.i.d.
  patterns, carrying no credit assignment; (iii) the graph keeps rewiring *while*
  accuracy is measured, so the function being scored drifts within the
  evaluation. Empirically (KA formula, fixed goal AND, balanced sampling, 24
  nodes, popsize 128, 120 patterns × 3 episodes, chance = 60/120): the population
  mean never left chance over 20 generations, the champion's honest re-test *fell
  below* chance (87–89 selected → 47.9–53.3 re-tested), and per-episode spread was
  ±14–22 points against the ~5.5 expected from pattern sampling alone. Most of the
  variance is developmental drift, and selection rewards drift luck, not function.
- **There is also no well-defined object to measure.** Modularity metrics need a
  static adjacency matrix, and in LNDP the graph never stops changing, so "the"
  grown network does not exist as a phenotype. It can be forced to exist — grow
  under spontaneous activity, then freeze plasticity for the whole lifetime — but
  that reduces LNDP to exactly NDP's protocol (genome → developmental program →
  static network → score), which we already have and can intervene on far more
  cheaply. The plasticity is what makes LNDP distinct, and it is precisely what a
  static modularity probe cannot use.
- **Verdict: biologically appealing, wrong tool for this thesis.** Its commitments
  are attractive — a fixed neuron count shaped by synaptogenesis and pruning, a
  developmental phase driven by spontaneous activity, lifetime plasticity gated by
  reward. But that same richness is the problem: the mechanics entangle
  development with learning, the phenotype is a moving target rather than a
  structure, and the many coupled components make single-variable comparisons
  impractical. We therefore do **not** pursue the modularity study in LNDP.

---

## Experiment 1 — architecture particulars

The treatment arm: a compressed DNA→brain encoding, evolved by CMA-ES.

**Biological framing.** We model the brain **at birth, not during continual
learning**. The brain is grown once from the genome and then frozen while it
solves the task — no within-life plasticity — so what is scored is its *innate*
ability, the circuitry the genome specifies before any experience. Adaptation
happens across generations, not within a lifetime.

**Neurons.** Fixed count, three roles: `n_in = 8` inputs, `n_hidden = 20`
hidden, `n_out = 1` output (defaults; N = 29). Only connections evolve — no
neurogenesis, no pruning, no physical space/coordinates.

**Connectivity.** Directed graph, allowed edges are **input→hidden**,
**hidden→hidden**, **hidden→output** only; no self-loops, no input→input,
no direct input→output. At the defaults that is 160 + 380 + 20 = **560 allowed
edges**. Inference is a synchronous recurrent pass `a ← tanh(a @ w + b)` for a
fixed 8 iterations with the inputs re-clamped each step; the decision is
`output > 0`.

**The genome does not store weights.** It stores (a) `K` evolved hidden
cell-type identity vectors plus one shared input identity and one shared output
identity, (b) per-type *abundance* logits setting how many hidden neurons are of
each type, (c) a per-type bias, and (d) one shared connection rule `g`, a small
MLP. Every weight is then `w_ij = g(feat_i, feat_j)`, where each neuron's
feature is `[type identity | positional code | role one-hot]` (dims 4 + 4 + 3 =
11). `g` is asymmetric in its two arguments (hence a *directed* graph),
`tanh`-bounded to [−1, 1], and deterministic given the genome — no
developmental noise decides function.

**Genome size: O(K), independent of the neuron count.** At the defaults
(K = 4, `g` = 22→16→1): 385 parameters in `g` + 34 in the type/abundance/bias
genes = **419 genes specifying 560 weights**. The point is the *scaling*, not
this ratio: raising `n_hidden` to 100 takes the phenotype to 10,800 edges while
the genome stays at 419. Extra neurons add no new wiring to specify.

**Why: the positional code is given to input/output neurons only.** Hidden
neurons are type-only, so any two hidden neurons of the same type have identical
features, hence identical incoming weights, outgoing weights and bias — they are
exact clones with identical activation at every timestep. Consequences worth
stating explicitly:
- Only `U = n_in + K + n_out` distinct feature signatures exist (13 at the
  defaults), so the whole brain is built from `U² = 169` distinct weight values,
  gathered into the full matrix.
- **A brain with `K` types is functionally a `K`-neuron recurrent network with
  gain-scaled edges**, whatever `n_hidden` is. `n_hidden` enters only as a
  multiplier: a downstream neuron receives `m_t · a_t · w_tj` from type `t`.
- So **`K` is the lever for distinct functional roles; `n_hidden` is a lever for
  gain, not diversity.** Abundance is softplus-*normalised*, so evolution
  controls the ratios between types, while `n_hidden` is a fixed uniform scale
  set by the experimenter.

**Abundance uses softplus, not softmax**, deliberately: the near-linear response
means small mutations move counts by ±1 and a starved type can recover, avoiding
an exponential extinction trap. `abundance = 0` is an equal split.

**Search.** CMA-ES over the flat genome (popsize 64, σ_init 0.1, elitist).
Balanced accuracy (chance = 0.5), bipolar inputs {−1, +1}.

### Assessment: what the cell-type encoding buys, and what it costs

**In its favour: cell types are a defensible abstraction.** Distinct neuron
types are a standard organising principle in neuroscience, and the division of
labour the encoding assumes — a genome specifying a *taxonomy of types plus
wiring rules*, rather than individual synapses — is closer to how a real genome
could plausibly specify a brain than a direct encoding is. A direct encoding
needs one heritable number per synapse, which no genome has room for. So the
compression here is not merely a parameter-count trick; it is the biologically
motivated part of the design. ⚠️ *Citations still needed* (cortical cell-type
taxonomies; genomic-bottleneck arguments) — do not write this paragraph up
without them.

**Cost 1 — the grown networks are dense, and structurally so.** This is the
main practical problem with the encoding as built. `w_ij = g(feat_i, feat_j)`
with `g` a smooth MLP, so an *absent* edge requires `g` to output exactly zero —
a measure-zero event. The brain is therefore fully connected (within the role
mask) at initialisation and stays that way: nothing in the fitness prefers fewer
edges, and larger |w| helps, so density only ever rises. Measured on
`retina_ka2005`, `n_hidden = 24` (492 role-allowed undirected edges):

| arm | density | Newman Q (unweighted) | Newman Q (weighted) |
|---|---:|---:|---:|
| FG, synaptic gate t = 0.2 | 72.2% | 0.053 | 0.033 |
| MVG, synaptic gate t = 0.2 | **100.0%** | **0.000** | **0.000** |

For scale, Kashtan–Alon and Clune report Q ≈ 0.35 for networks they call
modular. MVG converged to *literally* the complete role-allowed graph, and 77%
of its synapses sat at |w| > 0.999 — the `tanh` bound on `g`'s output turns into
a saturation attractor. Two consequences that matter for the writeup:
- **Q is not measurable on these brains, and this is not a metric artefact.**
  Newman Q is crushed by density by construction, and weighting made it *worse*
  (0.033 vs 0.053 for FG), because a saturated weight matrix is closer to the
  degree-product null than the topology is.
- **It is not unique to experiment 1.** The direct-encoding controls come out at
  ~92–94% (exp 2, CMA-ES) and ~87.5–90.9% (exp 3, gradient descent). Density is
  a property of "every allowed edge is free and nothing penalises it", not of the
  bottleneck. Any modularity comparison across the three arms needs a sparsity
  mechanism first.

**Cost 2 — expressiveness is capped by `K`, not by the neuron count.** Stated in
full above: same-type hidden neurons are exact clones, so a `K`-type brain is
functionally a `K`-neuron recurrent network with gain-scaled edges. Raising
`n_hidden` buys gain, not diversity. This interacts with the density problem:
because clones are indistinguishable, sparsity here can only ever be
**block-level** — switching off a type→type block silences every clone pair at
once. Fine-grained pruning is not representable. That is not purely a
limitation, since a block is what a module *is* in this model, but it does mean
the topology's real dimensionality is the `U × U` signature grid (90 legal pairs
at the defaults), not the 768 directed edges it expands to.

### Verdict: neither solved the task nor became modular — we moved on

**Density (Cost 1) turned out to be fixable.** A **synaptic budget** — fixed total
incoming `Σ|w| = S` per neuron, plus a relative shrink `τ` zeroing any synapse
below `τ ×` its target's own mean incoming magnitude — takes brains from 76–95%
density at generation 0 to 28–55% at the end. (Neither half works alone; an
*absolute* weight gate failed outright, since evolution just inflated `g`'s output
4–10× and walked through it.) So the null below is not a density artefact.

**With density controlled, FG vs MVG is flat.** `retina_ka2005`, `S = 4`, `τ = 0.9`,
E = 20, popsize 64, 2,000 generations, 3 seeds each:

| arm | balanced accuracy | density | planted L/R `r` |
|---|---:|---:|---:|
| FG (goal = AND) | 0.839 ± 0.033 | 39.4% | +0.136 ± 0.062 |
| MVG (AND↔OR) | 0.806 ± 0.005 | 37.7% | +0.115 ± 0.021 |

**A half-retina ablation says there is no module to find.** Scoring each brain's
output against the left/right object bits: only 1 of 6 (FG seed 2) builds a
half-detector (L 0.891, R 0.609; clamping left → 0.500, clamping right → 0.833 —
cleanly one-sided). The other five score *below* the 0.833 a single-half detector
gets, read both halves equally (`vs L` ≈ `vs R` ≈ 0.72), and lose the same to
either ablation — a smeared function of all 8 pixels, not two modules, not one.
All three MVG seeds are exactly symmetric (`vs L` = `vs R` = 0.719) where FG let
seed 2 lateralise: a real MVG effect, but symmetry is necessary for the modular
solution and nowhere near sufficient.

**A task property worth recording.** On `retina_ka2005` each half-object is true
for 8 of 16 half-patterns, so a perfect single-half detector scores **0.8333
under AND and 0.8333 under OR — gap exactly 0.0000**, while a two-sided AND-solver
swings 1.000 ↔ 0.667. The one-module solution is the optimal compromise across
both goals at zero re-adaptation cost, so MVG's mechanism only engages once a
network already computes both halves. This is KA's task, not our encoding.

⚠️ **Caveat.** 2,000 × 64 = 128k evaluations vs Kashtan–Alon's median 1.68M —
**13× under budget**. This is not evidence against the MVG→modularity hypothesis;
it is evidence that *this encoding at this budget* produces no half-detector for
any modularity metric to measure. Consistent with Cost 2: a `K`-type brain is
functionally a `K`-neuron recurrent net, and the left/right decomposition may be
out of reach at that width.

**Verdict.** Best accuracy 0.885 of 1.000, five of six seeds below the one-module
ceiling, and the FG/MVG contrast the encoding was built to test came out flat. We
stop the modularity study in this architecture and move to a different one.
*(⚠️ Name the successor here once settled, and what it changes relative to this.)*

---

## Experiment 4/6 — CGP, ECGP, and necgp (nesting extension)

Brief scratch section on the Boolean-circuit arm; expand later.

- **CGP vs ECGP (experiment_4, ⚠️ frozen).** Both evolve Boolean circuits on
  the KA retina task, pure Python, no gradients. ECGP adds two operators over
  plain CGP: `compress` (bundle a window of genome-adjacent nodes into a
  reusable module, callable by id) and `expand` (inline a module call back to
  primitives). Modules may **not** contain modules — a body is always
  primitives-only.
- **necgp (experiment_6) relaxes that restriction**: a module's body may
  itself call another module, gated by a decaying probability (`nest_decay`)
  so deep nesting stays comparatively rare. Built to test whether evolution
  keeps/reuses/builds-on modules given the option, rather than only calling
  flat ones.
- **Nesting's generation-count win is a real direction, not a significant
  effect.** Single-seed run: 32% fewer generations to solve, nested vs flat.
  9-seed paired sweep (same seed, nested vs flat): nested wins 6/9, mean
  paired diff −17.8k generations, but Wilcoxon p = 0.16 — does not reach
  significance at n=9.
- **A large fraction of "modules," nested or not, do no real computation.**
  Decomposing a solved circuit (drawing each real module's own internals,
  walking the active circuit recursively through nesting so a module only
  ever reached via another module's body is still counted) found 6 of 15
  reachable module types (40%) — and 24 of 64 actual module calls in the
  active circuit (37.5%) — are to modules with **zero internal gate
  interaction**: a single primitive, or several primitives that never chain,
  wrapped in module packaging. Root cause: ECGP's `compress` groups nodes
  that are **adjacent in genome position**, not nodes that are **connected in
  the phenotype's data-dependency graph** — position and data-flow are
  different things, and compress only looks at the former. Nesting doesn't
  fix this; it just gives the same failure mode a second layer to occur in.
- **Read: this is a structural problem with the compression mechanism, not a
  tuning problem.** Neither the generation-count trend nor the module
  structure supports a behavioural-modularity claim for ECGP/necgp as built.
  A phenotype-driven compression operator (select connected subgraphs of the
  *active* circuit by real data dependency, not genome position) is the
  candidate fix under discussion — write up once implemented and tested.

---

## Modularity metrics (`qmetrics/`)

Shared package at repo root: adapters turn any brain format (NDP's `W`,
exp 1–3's `w`, `kashtan_alon`'s layer blocks) into a graph; metrics see only the
graph. **Four metrics so far.**

**1. Newman Q** — (edges inside communities) − (expected if rewired at random
keeping degrees).
- Newman Q is the *objective*; Louvain and greedy/CNM are *search algorithms*
  for it. No such thing as "Louvain modularity". Maximisation is NP-hard, so
  every Q is a lower bound — state which search was used.
- Weaknesses: density confound (dense graph → Q ≈ 0 regardless of wiring);
  resolution limit (Fortunato 2007: modules below ~√(2m) edges invisible, ~9 at
  m=40); random graphs score nonzero (Guimerà 2004), so high Q isn't proof.

**2. Normalized Q_m** (Kashtan–Alon 2005, Eq. 2) — (Q_real − Q_rand)/(Q_max −
Q_rand), density held fixed across all three terms. 0 = chance, 1 = as modular
as these degrees permit. **Report this for dense grown brains**, since raw Q
isn't comparable across sparsities. Q_max is a hill-climb, so it biases Q_m up.

**3. Planted-bipartition modularity** (`left_right_q`, EXPERIMENTAL) — Q evaluated
at a partition the *task* specifies, rather than one we search for. The code is
named for the question (left vs right); the concept's published name is the one
above. See the section below.

**4. Circuit purity** (`circuit_purity`) — for **logic circuits**
specifically (experiment 4's CGP/ECGP), where the graph is a DAG of arity-2 gates
rather than a community-structured network. Label the inputs by side and average
down the DAG; a gate's purity is how far its ancestry leans one way. See the
section below.

### Graph size must be controlled for — it is not a detail

**Raw Q is not comparable between networks of different size/density.** Three
independent biases, all pushing the same way (sparser/bigger scores higher):

- **Density confound.** Q's null term is `k_i·k_j/2m`; in a dense graph every
  community already has near-expected internal edges, so Q compresses toward 0
  however it is wired. Our FG brain is density 0.322, MVG 0.251 — MVG is the
  *sparser* graph, so its higher raw Q (0.167 vs 0.127) is the expected direction
  of the artefact and cannot be read as evidence.
- **Resolution limit** (Fortunato 2007): modules below ~√(2m) edges are invisible
  — ~22 edges at FG's m=251 but ~40 at MVG's m=793. The two graphs are being
  asked *different questions* about what counts as a module.
- **Achievable range differs.** The maximum Q a graph can reach is set by its
  degree sequence, so raw Q compares points on two different scales.

**Normalising does not automatically fix it.** Two measured traps:

- *Q_m inherits a budget bias.* Q_max is estimated by a search with a fixed
  iteration budget; a bigger graph exhausts that budget sooner, so its Q_max is
  more under-estimated and its Q_m inflated. Measured at equal `steps`: the
  hill-climb's improvement over the real network was 0.035 on FG vs 0.009 on MVG
  — a 4× gap against a 3.2× edge-count ratio. Budget must scale with edges.
- *The z-score is not size-fair either.* The null's SD shrinks roughly as 1/√m,
  so z ≈ excess × √m and the bigger graph scores higher for free (FG m=251 vs MVG
  m=793 gives MVG a 1.78× head start). **Report z as significance, never as
  effect size.**

**The clean fix is experimental design, not statistics: compare networks of the
same size.** Kashtan–Alon's own FG/MVG comparison used an identical architecture
with only the goal schedule differing. Where sizes cannot be matched, normalise,
state the residual bias, and report the size alongside every Q.

### Left/right modularity for MVG — the planted-bipartition metric

MVG alternates between two goals over the *same* left/right decomposition, so the
question is not "what modules exist?" but "**is the network split into a left and
a right module?**" That is a much easier question, and worth stating why.

- **Terminology.** "Binary modularity" is not a standard term. The partition into
  two groups is a **bipartition**; scoring a partition you specify rather than
  discover is a **planted** (or prescribed) partition; and the published metric
  for "how well do edges respect a given node labelling" is Newman's **discrete
  (attribute) assortativity** (Newman 2003, *Mixing patterns in networks* —
  citation to verify). So: *planted-bipartition modularity*, or equivalently
  *assortativity with respect to the left/right labelling*.
- **We are not inventing a metric.** Newman Q is *defined* for any partition;
  community detection is only a search for a good one. Handing Q the task's own
  partition removes the NP-hard maximisation, the iteration budget and the
  non-convergence that make Q_m unstable — there is no optimiser left to fail.
- **Definition.** `r = (Σe_ii − Σa_i²)/(1 − Σa_i²)` = Q-at-the-fixed-partition
  divided by its analytic ceiling (every edge internal). 1 = perfectly split,
  0 = chance, <0 = anti-associated.
- **Crosstalk** = (fraction of edges crossing between groups) / (fraction expected
  to cross at these degrees, i.e. `1 − Σa_i²`). 1 = chance, 0 = disconnected
  halves, >1 = anti-modular. **Identity: r = 1 − crosstalk** — the formal metric
  and the prose number are one object ("23% fewer left–right crossings than these
  degrees predict" = r +0.23).
- **p** = fraction of the degree-preserving, mask-respecting nulls reaching
  `Q ≥ Q_real`, with the +1/+1 correction, so its floor is `1/(n_rand+1)`
  (0.005 at n_rand=200 — MVG sits exactly there, i.e. 0/200 nulls matched it).
  p answers *distinguishable from chance*, never *how modular*. Both p and z
  favour the larger graph (null SD shrinks ~1/√m), so **compare magnitudes with
  r and read p only within one network.**
- **The output node is excluded.** A single readout must connect to both halves by
  construction, so forcing it onto a side charges a fixed cross-edge penalty
  unrelated to modularity, and it hits small graphs hardest.
- **Hidden nodes have no a-priori side, so report a bracket.** `optimal` assigns
  them to maximise Q (the *best case for the modularity hypothesis*); `majority`
  assigns each to the side it has more edges to (the naive reading). If even
  `optimal` is at chance, the question is settled.
- **⚠️ The null must re-fit the assignment on every rewired graph.** `optimal`
  fits free nodes to the data and so finds structure in noise — a plain ER graph
  with 8 pinned nodes scores r = +0.25. Scoring nulls at the partition fitted to
  the *real* graph gives the real graph an advantage the null never gets, and
  makes everything look significant. Measured cost of getting this wrong: FG's
  z fell from **+5.66 (p=0.005) to +0.88 (p=0.199)** once the null re-fitted.
  This is a general lesson for any planted-partition statistic, not a detail of
  this implementation.

### The empirical case for the planted partition — one picture

*(Figure: `latex_figures/Kashtan-Alon/newman_communities_mvg_seed1.png`, generator
`kashtan_alon/analysis/newman_vs_binary.py`.)*

The argument above is conceptual. Here it is as a measurement, on one real brain:
the capped **MVG seed 1** last-AND-epoch champion (generation 24,970, accuracy
0.97 on AND).

- That network is a **literal two-module network**. Circuit purity and `r` are both
  exactly **1.00**, and no left–right edge exists anywhere below the output neuron
  — every hidden neuron's live ancestry traces to one retina side only (audited
  node-by-node, `scratch_purity_audit.py`). It is a direct instantiation of KA's
  own Fig. 5e prose, *"two distinct modules ... each monitoring a different side of
  the retina"*.
- Greedy Newman Q, handed the same graph, returns **four communities**: it cuts the
  left module in two and makes the integrator spine a module of its own. It flags
  **7 of 33 edges as "between-module"** when almost all of them stay on one side.
- And `Q_m` **ranks it below a less modular brain**: seed 1 (purity 1.00) scores
  Q_m +0.26, while seed 0 (purity 0.81) scores +0.46.

The point is not that Newman Q is wrong — Q is *defined* for any partition, and
the greedy split it found is a perfectly good one by its own objective. The point
is that **Q has to search for a partition**: the search is NP-hard, budget-limited,
subject to the resolution limit, and — decisively — has no reason to aim at the
split the *task* is about. These graphs are also small enough to sit near
Fortunato's resolution limit (~√(2m) ≈ 8 edges at this brain's m=33), which is the
scale of the sub-communities Q actually returned. Under MVG we already know the
partition the experiment is about, so
handing it to Q removes the optimiser entirely, along with its budget, its seed
dependence and its instability. That is the whole justification for `r` and circuit
purity, and it is why they are the **primary** metrics for every left/right result
in this thesis, with `Q_m` reported second for comparability with the paper.

### Circuit purity — a left/right measure built for logic circuits

The three metrics above were designed for weighted neural graphs. A Boolean
circuit is a different object: a DAG whose every node has in-degree 2 (the gate's
arguments), no weights, and a single output. Newman Q measured on such a circuit
rewards long-and-thin wiring rather than modularity — on hand-built references it
scored a *perfectly modular* circuit (Q_m 0.14) below a maximally entangled chain's
size-matched peers, and its detected communities were pixel pairs, never the two
halves. Hence a purpose-built metric.

- **Definition.** Pin the program inputs to a side (left = 0, right = 1). Every
  gate takes the **mean of its parents**; equivalently, a gate's value `x` is the
  probability that a backward random walk from it, uniform over parents at each
  step, ends on a right-hand input. `purity(v) = 2·|x_v − 0.5|`. Circuit purity is
  the mean over active gates, excluding the inputs and the program output (a
  readout must see both halves by construction). The ideal `L AND R` circuit — a
  left module, a right module, one gate joining them — scores exactly **1**.
- **Not new mathematics, new application.** The one-step case is Guimerà–Amaral's
  **participation coefficient** (`P = 2p(1−p)`, so purity `= √(1−2P)`); propagating
  it down the DAG is a **harmonic function / label propagation** (Zhu, Ghahramani &
  Lafferty 2003) with the inputs as boundary values. The novelty is applying it to
  evolved circuits, where no modularity metric had been proposed at all.
- **ECGP must be measured on the FLATTENED circuit.** Averaging is not invariant
  under module compression — a 3-input module reads 1/3 per parent while its
  arity-2 expansion reads 1/4, 1/4, 1/2 — so a module-level graph and its own
  expansion score differently, and CGP/ECGP would not be comparable.
- **Calibration** (retina, 8 inputs, `and`; evolved circuits all at 256/256):

  | circuit | active gates | purity |
  |---|---|---|
  | REF modular (two pure chains + 1 join) | 29 | **1.000** |
  | REF tail (1 merge, then 28 pure-left gates) | 29 | 0.929 |
  | evolved, best of 6 seeds | 23 | 0.915 |
  | evolved, worst of 6 seeds | 33 | 0.568 |
  | REF chain (one line eating all 8 pixels) | 29 | 0.530 |
  | unevolved random (n = 87) | 16.8 | **0.426 ± 0.115** |

- **Random baseline, by size** (`latex_figures/purity_metric/`). Sampling: draw
  uniform random CGP genomes of 8…512 nodes, keep only the **active** subgraph
  (nodes on a path to the output — inactive nodes are never scored and never enter
  the denominator), and bucket by the **exact** active-gate count, 2…50. Genome
  size has to be swept to reach the large buckets, so each (active count, genome
  size) cell is capped equally — otherwise small buckets would be all small genomes
  and genome size would be confounded with circuit size. 38k circuits, ≈300–1200 per
  bucket. Residual confound measured separately and null: at fixed active count,
  purity varies ≤0.02 across genome sizes against a per-circuit SD of ~0.10.

  | active gates | 2 | 5 | 10 | 20 | 30 | 40 | 50 |
  |---|---|---|---|---|---|---|---|
  | mean purity | 0.524 | 0.474 | 0.453 | 0.434 | 0.417 | 0.405 | 0.393 |
  | SD | 0.500 | 0.229 | 0.154 | 0.117 | 0.104 | 0.091 | 0.083 |

  The baseline **decays slightly as circuits grow** (0.50 → 0.39) while the SD
  tightens 6×. The decay is the metric behaving as designed, not an artefact:
  purity falls as one advances through a circuit, because each extra layer averages
  over a wider ancestry and any single side's share regresses toward ½ — bigger
  circuits are deeper, so more of their gates sit in that mixed interior. Practical
  consequence: **read a score against its own size bucket**, where the shrinking SD
  makes the comparison sharper for large circuits, not weaker.
- **+** O(V+E), one sweep: no community detection, no null model, no truth tables,
  so it scales far past the 2^n_in wall that stops any behavioural measure.
  Continuous (loggable per generation, regressable against fitness), per-node
  (colour the circuit diagram by it), and arity-agnostic. Separates evolved
  circuits from random cleanly (0.57–0.92 vs 0.43).
- **−** It is a **wiring** statistic that ignores what the gates compute, so a
  functionally dead argument still counts. Contamination **decays geometrically
  with depth**: REF tail scores 0.929 although every one of its gates depends on
  both halves. Whether that decay is right depends on the gates — an OR chain does
  dilute an early input, an AND chain does not — and the metric cannot tell them
  apart by construction. The floor is unanchored (the maximally non-modular chain,
  0.530, outscores random, 0.426), it is a plain mean so pure filler raises it, and
  its denominator is the gate count, so it is **size-confounded exactly like raw Q**
  — compare only at matched circuit size.

### Constraints on the null model — correctness, not refinement

Q_m compares against rewired versions of the network. Our models forbid certain
edges (inputs↛inputs; layered nets connect adjacent layers only; KA also caps
fan-in ≤3/≤2). Naive rewiring ignores this and generates networks the model
could never produce, so Q_rand/Q_max describe the wrong ensemble.

- **Measured on our KA runs: 54.9% of unconstrained-null edges were impossible**
  (retina→retina, layer-skipping). Fixed with an `allowed` mask the rewirer
  honours → 0.0% illegal, degrees and edge count still preserved.
- **Bias is systematic:** constraining lowers Q_max (architecture limits how
  modular anything *can* be) while Q_rand barely moves, so unconstrained Q_m
  **understates** modularity. All 10 KA runs rose; several ~doubled (fg_seed1
  +0.34→+0.70), one flipped sign (fg_seed2 −0.02→+0.13).
- **Every Q_m reported must state its constraint set** — otherwise
  uninterpretable.
- Only *pairwise* constraints reduce to a mask. *Degree caps* (KA fan-in) need
  per-node counters, **not yet implemented**: rewiring preserves total degree
  but can shift a node's in/out split, so caps don't come free with the mask.

---

## Preliminary modularity results — circuit purity across experiments

⚠️ **In the works.** First pass, one arm's worth of seeds each, no statistics —
a naive cross-experiment comparison to see whether `circuit_purity` is worth
trusting, not yet a result to cite. Expect this table to grow as more arms get
saved genotypes.

| domain | arm | n seeds | purity (median) | size (median) | task metric (median) | other structural metric (median) |
|---|---|---:|---:|---|---|---|
| exp_4 CGP | FG 50n, 4-gate | 12 | 0.777 | 19.5 active | acc 1.000 | cone-frac 0.659 |
| exp_4 ECGP | FG 50n, 4-gate | 12 | 0.805 | 24.5 active | acc 1.000 | cone-frac 0.682 |
| exp_4 CGP | FG 100n, NAND-only | 3 | 0.595 | 29.0 active | acc 1.000 | cone-frac 0.364 |
| exp_4 CGP | FG 100n, 4-gate | 3 | 0.829 | 24.0 active | acc 1.000 | cone-frac 0.625 |
| exp_4 CGP | MVG 50n (rerun, `--save-best`) | 4 | 0.843 | 9.5 active | acc 0.906 | cone-frac 0.386 |
| exp_4 ECGP | MVG 50n (rerun, `--save-best`) | 4 | 0.793 | 12.0 active | acc 0.924 | cone-frac 0.528 |
| exp_4 CGP | MVG 400n (rerun, `--save-best`) | 4 | 0.597 | 39.5 active | acc 0.984 | cone-frac 0.291 |
| exp_4 ECGP | MVG 400n (rerun, `--save-best`) | 4 | 0.676 | 28.5 active | acc 0.932 | cone-frac 0.521 |
| KA paper-faithful | FG | 5 | 0.598 | 40 edges (mean) | acc on AND 0.891 | Q_m −0.007 |
| KA paper-faithful | MVG | 5 | **1.000** | 36 edges (mean) | acc on AND 0.969 | Q_m **0.260** |
| KA no-fanin ablation | FG | 5 | 0.287 | 69 edges (mean) | acc on AND 0.977 | Q_m −0.076 |
| KA no-fanin ablation | MVG | 5 | **0.530** | 55 edges (mean) | acc on AND 1.000 | Q_m **0.016** |

*(The four KA rows were re-scored 2026-09-11 on **goal-matched** brains — the last
champion archived during an AND epoch, all four groups on AND — at n=5. They
previously read final-generation champions, which for MVG are OR-phase brains, and
the ablation rows were n=3. Means for all four groups, with SDs, are in the
ablation section below and in `kashtan_alon/RESULTS.md` Run 8.)*
| exp_3 (GD, margin loss) | retina/xor | 5 | N/A — cyclic graph | ~470/560 edges | acc 1.000 | raw Q ~0.019 (undirected) |

**The metric has real explanatory power, despite every caveat above** — but
exp_4's own MVG comparison shows exactly why it has to be read size-adjusted,
not raw. On `kashtan_alon/`, purity tracks `Q_m` in direction **twice over**:
MVG > FG at both the paper-spec fan-in and the no-fanin ablation, the same
MVG-more-modular result the field already accepts from `Q_m` alone. Inside
exp_4's FG-only arms, purity and the coarse cone-fraction proxy rank all four
identically (ECGP > CGP, 4-gate > NAND-only).

**exp_4's MVG rerun initially looked like it broke that agreement — it
didn't, once size is controlled.** At the 50-node budget (the one place FG
and MVG share a node count), *raw* purity says MVG ≥ FG (CGP: 0.843 vs 0.777;
ECGP: 0.793 vs 0.805) while cone-fraction says the opposite, sharply (CGP:
0.386 vs 0.659; ECGP: 0.528 vs 0.682) — a real disagreement between the two
structural metrics, on the same circuits. But MVG's circuits are also much
smaller (9.5–12 active gates vs 19.5–24.5 for FG), and the calibration table
above already establishes that *random* circuits get purer as they shrink. Z-scoring
each arm's purity against the random baseline **at its own active-gate count**
(`scratch_purity_zscore.py`, linear interpolation of the calibration buckets)
reverses the raw ranking:

| arm | active (median) | purity | baseline mean / SD | z |
|---|---:|---:|---|---:|
| CGP  FG  50n  | 19.5 | 0.777 | 0.435 / 0.119 | 2.88 |
| CGP  MVG 50n  |  9.5 | 0.843 | 0.455 / 0.161 | **2.40** |
| ECGP FG  50n  | 24.5 | 0.805 | 0.426 / 0.111 | 3.41 |
| ECGP MVG 50n  | 12.0 | 0.793 | 0.449 / 0.147 | **2.35** |

Both algorithms: FG sits further above chance than MVG once size is
controlled, even though MVG's raw number was equal or higher. So purity does
**not** actually contradict cone-fraction here — it *agrees* that MVG is less
modular than FG, exactly like the existing RESULTS.md finding — but a naive
reading of the raw numbers would have said the opposite, and would have been
wrong for a reason the metric's own documentation already flags (size
confound) rather than anything new. **Lesson for the rest of this table: every
row above except the size-matched KA pairs should really be read as a
z-score against the calibration baseline, not as a raw number** — this has
only been done for the 50n arms so far. The 400n MVG arms (z = 2.09 CGP,
2.42 ECGP) have no size-matched FG-400n purity to compare against yet.

⚠️ **Re-check this "resolution" before trusting it.** The z-score gap
above (FG ~0.5–1.0 higher than MVG) is medians only, no variance, and is
similar in size to the gate-set confound noted below — read the numbers
again carefully before citing this as a real agreement rather than a wash.

**One confound surfaced along the way, worth remembering:** NAND-only vs
4-gate circuits at matched size (29 vs 24 active nodes) differ almost as much
in purity (0.595 vs 0.829) as FG vs MVG differs anywhere in this table — gate
set alone moves the score. Any purity comparison has to hold the gate set
fixed, not just the node count.

**`circuit_purity` is architecturally inapplicable to exp_3 (and, by the same
argument, exp_2).** Their `role_mask` permits hidden↔hidden edges in both
directions, so the grown network is recurrent, not a DAG — every one of the 5
saved genomes fails `nx.topological_sort`. This isn't a metric weakness, it's
scope: the metric needs a feedforward circuit. Raw `newman_q` still runs there
(~0.02, near zero) but per its own density-confound caveat that's close to
uninterpretable at ~90% density — exp_3 still needs `normalized_qm`, not this.

**Idea to keep in mind, not yet tested: purity may partly be reading the
*task's* intrinsic modularity, not just the circuit's.** If the target
function itself decomposes cleanly along the pinned left/right split (as
`L AND R` does by construction), a circuit that merely *computes it correctly*
may come out purer than one solving a scrambled/entangled target of the same
size and gate budget — independent of anything evolution or the encoding did.
A cheap check: take the same task, permute which inputs count as "left" vs
"right" (or otherwise scramble the target's own decomposition) so the correct
circuit can no longer lateralise as cleanly, then compare purity on evolved
solutions to that variant against the ones above. If purity drops on the
scrambled task at matched accuracy and size, that's the task-intrinsic
component showing through, not a property of the search — and it would mean
every purity number in this table needs a task-matched null, not just a
size-matched one (the existing calibration only controls for active-gate
count, not the target function's own decomposability).

---

## Testing whether a constraint is necessary for modularity

**What "constraint" means here — three mechanisms, one abstract role.** Three
different things get called a "constraint" across this project and the
literature it engages with, and they are easy to conflate:

1. **Clune et al. 2013's connection cost** — a *continuous fitness penalty* on
   wiring length, requiring neurons to occupy physical space (cost ∝ Euclidean
   distance between them). Explicitly excluded by this thesis's hard constraints
   (no physical space).
2. **Kashtan–Alon's fan-in cap** — a *hard architectural limit*: each neuron may
   receive from at most 3 others (first hidden layer) or 2 (every layer after),
   full stop, independent of space.
3. **Our synaptic budget (`experiment_1`)** — a *conserved resource*: total
   incoming `Σ|w|` per neuron is capped (`S`) and shared among however many
   synapses exist, with a relative shrink (`τ`) pruning the weakest. Not a count
   cap — a strength budget.

What unifies them is not the mechanism but the **role**: each makes
connectivity a *scarce, competed-for resource* rather than a free good. The
hypothesis under test is that scarcity in this abstract sense — not wiring
length specifically — is the ingredient MVG needs to produce modularity. With
nothing to compete for, there is no trade-off forcing the network to reuse or
specialise circuitry when the goal switches; it can just grow a redundant,
entangled solution that already satisfies every goal at once (see the
"one-module compromise" fact under Experiment 1, above: 0.833 under *both* AND
and OR, zero re-adaptation cost, no sharing required — always available, and
cheaper to find than a modular split, once wiring is free).

**Evidence gathered so far.** All of it is consistent with the hypothesis; none
of it is yet a direct ablation of KA's own constraint.

- **`experiment_1`, no synaptic budget (Cost 1, above): the brain is fully
  dense regardless of goal.** MVG converged to *literally* the complete
  role-allowed graph (100% density, 77% of weights saturated at |w| > 0.999);
  FG was little better (72–100% across the arms measured). Newman Q could not
  register anything (0.000–0.053) — no sparse structure exists for a modularity
  metric to find, weighted or not.
- **New (2026-08-19), matched no-budget vs budget 2×2**, `retina_ka2005`,
  `n_hidden=24`, popsize 64, 2,000 generations, 3 seeds/arm, scored with
  `qmetrics.left_right_q` (primary) and `normalized_qm` (secondary), `n_rand=200`
  (so `p=0.005` is the floor — 0/200 nulls matched):

  | arm | density | left_right_q, mean (sig. seeds) | Q_m, mean (sig. seeds) |
  |---|---:|---:|---:|
  | FG, no budget | 100.0% | undefined — complete graph | undefined |
  | MVG, no budget | 100.0% | undefined — complete graph | undefined |
  | FG, budget (S=4, τ=0.9) | 48.7% | 0.225 (**2/3**, p=.020/.010) | 0.001 (1/3, p=.005) |
  | MVG, budget (S=4, τ=0.9, E=20) | 37.7% | 0.467 (1/3, p=.005) | **0.453 (2/3**, p=.005/.005) |

  Two findings, and they cut in different directions — report both, do not
  average over the disagreement:
  1. **Removing the budget doesn't lower modularity, it makes the question
     unanswerable.** Density saturates to 100% in both FG and MVG, and the
     metric is undefined on a complete graph. That is itself the strongest
     form of "no modularity possible here" — stronger than a low score, because
     there is no structure left to score.
  2. **With the budget restored, the two metrics disagree on which goal wins.**
     `Q_m` (secondary) shows a clean MVG > FG gap (0.453 vs 0.001, 2/3 vs 1/3
     seeds significant). `left_right_q` (**primary**, per the metric-choice
     guidance above) shows the opposite pattern in significance count — FG has
     *more* individually-significant seeds (2/3) than MVG (1/3), even though
     MVG's mean is nominally higher (0.467 vs 0.225); that mean is inflated by
     one large but non-significant MVG outlier (seed 2, `LR_opt=1.000`,
     `p=0.299`). So the designated primary metric does **not** cleanly support
     "MVG produces more left/right modularity than FG" here — only `Q_m` does.
     n=3/arm either way, so treat all of this as suggestive, not settled.
- **`kashtan_alon/` reproduction: Q_m ≈ 0.35 (MVG) vs ≈ 0.15 (FG)**, both *with*
  KA's fan-in cap intact (`kashtan_alon/RESULTS.md`). This is the literal system
  the field cites for "MVG produces modularity," and it has always run with the
  constraint switched on. **No one, including this project, has run it with the
  cap removed.**

### KA with the fan-in cap removed — the direct ablation of KA's own constraint

**The ablation is one line.** `NetConfig(fan_in=())` in place of the default
`(3,3,3,2)`; `model.py:_fan_in()` then falls back to "the whole previous layer",
so every neuron may read every neuron below it. Same task, same seeds, same GA
hyperparameters, same 25,000 generations. One honest qualifier on "only": the cap
is read in two places and removing it changes both — `model.py:91` sets the
initial fan-in (`k = round(0.5 × cap)`, 2 edges/neuron capped vs 4 uncapped) and
`ga.py:95` is the ceiling the add-edge mutation may not exceed. So it is "the cap
is gone wherever it acted", not "the cap is gone at selection time only".

**Result (n=5/arm/condition, 2026-09-11, `kashtan_alon/RESULTS.md` Run 8).** Every
brain below is the **last champion archived during an AND epoch** and is scored on
**AND**, so all four groups are goal-matched (this matters: `25000/20 = 1250`
blocks leaves generation 24,999 inside an *OR* epoch for every MVG seed, so a
final-generation MVG champion is an OR specialist — a different brain, not just a
differently-scored one).

| condition | arm | accuracy (AND) | Q | Q_m | r | purity | edges | density |
|---|---|---|---|---|---|---|---|---|
| capped (paper) | FG | 0.90 ± 0.03 | 0.38 ± 0.04 | +0.02 ± 0.14 | +0.60 ± 0.14 | 0.56 ± 0.13 | 40 | 38% |
| capped (paper) | **MVG** | 0.97 ± 0.03 | **0.49** ± 0.03 | **+0.31** ± 0.09 | **+0.94** ± 0.09 | **0.93** ± 0.09 | 36 | 34% |
| no fan-in cap | FG | 0.97 ± 0.02 | 0.22 ± 0.03 | −0.07 ± 0.10 | +0.30 ± 0.07 | 0.31 ± 0.11 | 69 | 65% |
| no fan-in cap | MVG | **1.00** ± 0.00 | 0.29 ± 0.04 | +0.02 ± 0.03 | +0.51 ± 0.08 | 0.56 ± 0.11 | 55 | 52% |

**⚠️ This revises what an earlier draft of this section said.** At n=3, scored on
final-generation champions, the MVG−FG gap looked unchanged by the ablation
(0.219 vs 0.220) and the write-up concluded "goal-switching is doing most of the
work, independent of whether wiring is scarce". At n=5 and goal-matched, the gap
**shrinks on every metric**: Q_m 0.28 → 0.09, r 0.34 → 0.21, purity 0.37 → 0.25.
Do not cite the old reading.

Three things to say about this table:

1. **The cap is not a performance constraint; it is a modularity constraint.**
   Removing it makes the task *easier* — uncapped MVG is perfect on all five
   seeds, uncapped FG reaches 0.97 vs capped FG's 0.90. So the ablated networks
   are not less modular because they are worse solutions. They are better
   solutions that happen to be non-modular, which is the strongest form this
   result could take.
2. **MVG is not sufficient — the constraint does a lot of the work.** Uncapped MVG
   (purity 0.56) lands exactly where *capped FG* sits (0.56): removing the cap
   costs MVG about as much modularity as removing MVG itself did. Capped-FG and
   uncapped-MVG arrive at the same number from opposite directions, and only
   capped-MVG (0.93) is qualitatively different. That is a **2×2 interaction**,
   not a main effect of MVG — and it is a better story than "MVG causes
   modularity": MVG causes modularity *when wiring is scarce enough that the
   monolithic solution is not reachable*.
3. **The MVG > FG direction still survives the ablation**, on purity (0.56 vs
   0.31) and r (+0.51 vs +0.30) with non-overlapping ±1 SD. So "necessary" is too
   strong; "does most of the work at this network size" is the defensible claim.

**Removing the cap also finds solutions far faster.** First generation at which
the champion reaches a given accuracy *during an AND epoch*:

| threshold | capped FG | capped MVG | no-cap FG | no-cap MVG |
|---|---|---|---|---|
| 0.90 | 2/5 seeds ever | median 890 | median **60** | median **90** |
| 0.99 | never | 2/5 seeds ever | 1/5 seeds ever | median **290** |
| 1.00 | never | 2/5 (gen 6,890 and 23,930) | never | **all 5 seeds**, gen 240–400 |

Uncapped MVG is perfect on every seed inside 400 generations; capped MVG manages
it twice in 25,000 and capped FG never passes 0.95. This is the first direct
evidence for the "**does constraining also make search converge faster?**"
question raised under *High-level hypotheses* above — and the answer here is the
**opposite** of the hunch recorded there. The hunch was that a constraint shrinks
the haystack around the same needle; what actually happens is that the constraint
*removes* the easy needles. Unconstrained search is not wasting its budget on a
larger space — it is finding a dense, entangled, high-accuracy solution almost
immediately, and that solution is simply not available under the cap. Scarcity
buys modularity and **pays for it in both accuracy and search time**.

**MVG runs sparser than FG in both conditions — but "parsimony" is the wrong
word.** Mean density (% of the 107 possible feedforward edges):

| group | gen 0 | 500 | 2,000 | 24,990 |
|---|---|---|---|---|
| capped FG | 27.4 | 34.3 | 37.0 | 37.5 |
| capped MVG | 27.4 | 31.7 | 34.3 | 33.8 |
| no-cap FG | 50.0 | 63.0 | 64.3 | **64.3** |
| no-cap MVG | 50.0 | 50.9 | 49.4 | **53.4** |

The mechanism is not MVG *removing* edges — it is FG *adding* them while MVG does
not. Our fitness has no complexity term at all (the paper's 0.01/neuron penalty is
a known missing piece, deviation 3 below), so nothing rewards fewer edges and
"task switching forces a parsimonious net" cannot be literally true. The likelier
reading: **under MVG a newly added edge has to pay off on *both* goals to survive
a switch**, so goal-specific wiring is repeatedly de-selected and new edges fix
more slowly. Non-stationarity acting as an implicit regulariser — the same shape
of argument as noise or dropout — rather than a parsimony pressure. Testable: put
the paper's neuron penalty back and see whether the FG/MVG density gap widens
(real parsimony pressure should hit FG harder) or stays put.

Two caveats that have to travel with the density numbers:
- *Capped-vs-uncapped density is partly an artefact.* `init_population()` seeds
  `k = round(init_density × cap)`, and `cap` is whatever the current fan-in limit
  is, so `init_density=0.5` means 27.4% under the cap and exactly **50.0%**
  without it. The two conditions did not start from a comparable point. The
  **FG-vs-MVG comparison inside one condition is clean**, though — both arms start
  identically — which is the comparison the paragraph above rests on.
- *MVG is always the sparser arm, and every metric here favours sparsity.* So some
  of the MVG advantage could in principle be density. It cannot be all of it:
  capped MVG (34%) vs capped FG (38%) is a 4-point density difference carrying a
  0.37 purity difference, and the *denser* uncapped MVG (52%) ties the *sparser*
  capped FG (38%) at purity 0.56 — density alone does not order these groups.

**Q_m stops discriminating once the cap is removed, and that is a metric result
worth reporting in its own right.**

| condition | arm | Q_real | Q_rand | Q_max | Q_max − Q_rand | Q_real − Q_rand | Q_m |
|---|---|---|---|---|---|---|---|
| capped | FG | 0.380 | 0.374 | 0.607 | 0.233 | +0.007 | +0.03 |
| capped | MVG | 0.486 | 0.413 | 0.653 | 0.240 | **+0.073** | +0.31 |
| no cap | FG | 0.216 | 0.229 | 0.417 | 0.188 | −0.013 | −0.07 |
| no cap | MVG | 0.287 | 0.285 | 0.478 | 0.194 | **+0.002** | +0.02 |

In the ablation `Q_real` and `Q_rand` collapse *together* (0.287 vs 0.285): the
numerator is +0.002, smaller than the greedy partitioner's own noise, which is why
the across-seed SD (±0.03) spans zero. Three causes, all consequences of density:
(i) **no headroom** — at 52–65% density almost every possible edge exists, so every
partition has many crossing edges and everything scores low; `Q_max` itself falls
0.65 → 0.48; (ii) **the null is handed the answer through the degree sequence** —
`Q_m` holds degrees fixed and rewires, and under the cap degrees are near-uniform
(~3 everywhere) so the modularity lives in *which* pairs are wired and rewiring
destroys it, whereas uncapped the degrees are heterogeneous and dense and a random
graph with those degrees already looks clustered for free; (iii) **Q is
label-blind** — purity and `r` know the left/right labelling and still separate
the uncapped arms cleanly, Q's greedy partition cannot.

The correct statement is therefore **not** "Q_m says modularity vanished". It is:
modularity genuinely fell (purity 0.93 → 0.56) **and, separately, Q_m lost its
resolution.** `Q_m` is calibrated for the sparse regime KA's own runs occupy
(34–38% density); pushed to 52–65% it is a ratio of two converged quantities.
Ironically it was introduced precisely to remove the density confound from raw Q —
and it removes it by subtracting a null that gets *closer* to the real value as
density rises and dividing by a range that *shrinks*. Read purity and `r` as
primary in the ablation, `Q_m` with this caveat, and note that the degradation
tracking density is a caution for every other arm of this thesis that leans on
`Q_m` (experiment_1's dense brains above all).

### The task's own 0.75 shortcut — why MVG cannot punish a one-module solution

On the KA-faithful retina, raw fraction-correct over all 256 patterns:

| predictor | on AND | on OR | mean over the MVG schedule |
|---|---|---|---|
| constant 0 | 0.750 | 0.250 | 0.500 |
| constant 1 | 0.250 | 0.750 | 0.500 |
| **LEFT only** (equivalently RIGHT only) | **0.750** | **0.750** | **0.750** |

`LEFT` and `RIGHT` are each true on exactly 128/256 patterns; AND on 64, OR on
192. So a detector that solves **one half of the retina and ignores the other**
scores 0.75 under *both* goals, strictly dominates a constant output under MVG
(0.75 vs 0.50 across the schedule), and pays **zero re-adaptation cost at every
switch**. The one-module solution is exactly the thing MVG's mechanism cannot
punish — which is why MVG only starts producing modularity once a network already
computes both halves, and why the interesting evolutionary action is all above
0.75. This is the KA-faithful task's version of the "one-module compromise" fact
already recorded for the `retina_ka2005` stand-in under Experiment 1 (0.8333 under
both goals there); same phenomenon, different number, and it is a property of
**KA's task**, not of any encoding we built.

### Methodology note to carry into the writeup

Our reproduction **deviates from the paper in five documented ways** — exhaustive
256-pattern fitness instead of a 100-pattern sample; a hill-climb `Q_max` instead
of re-evolving 100 populations toward Q; no complexity penalty; elite 150/600
reused by analogy from the circuit experiment; and a reconstructed mutation
operator set / crossover mechanism / threshold range (the Supporting Information
is unavailable to us). All five are listed with their evidence in the
Kashtan-Alon section below and in `kashtan_alon/PAPER_SPEC.md`, and deviations 1–3
all push the same way, which is consistent with our absolute `Q_m` (0.31 MVG /
0.02 FG) sitting below the paper's (0.35 / 0.15) while the **gap** (0.29 vs 0.20)
and its direction reproduce. Every KA number in this thesis must be introduced as
a *reproduction of the effect*, never as a replication of the magnitudes.

---

## Kashtan-Alon reproduction (`kashtan_alon/`) — task, fitness, exact commands, and deviations from the paper

> 🐛 **Reporting bug found 2026-09-10 — do not quote any MVG accuracy number from
> an older draft of this section. Modularity numbers are unaffected. Fixed in code
> the same day; no re-training needed.**
>
> `train.py` reported `best_fit` as an all-time maximum across the run, scored
> against *whichever goal was live that generation*. Under MVG the goal alternates
> AND↔OR every 20 generations, so that scalar maximises over **two different
> tasks** — it is not an accuracy. It is also systematically biased toward OR (true
> on 192/256 patterns vs AND's 64/256, hence far easier to nearly-ace) and, because
> `25000/20 = 1250` blocks makes the final block always odd-indexed, every MVG seed
> ends mid-OR: **all 5 MVG runs recorded `best_op=or`, none `and`.** So the
> previously-drafted "MVG 0.975 vs FG 0.904" compared MVG-on-the-easy-goal against
> FG-on-the-hard-goal.
>
> **Unaffected:** `Q`, `Q_m`, `circuit_purity`, `left_right_q`/`r` — all computed on
> the *final generation's* champion, not the peak-fitness genome — and the
> per-generation CSVs, which log the true per-generation champion with an `op`
> column. The `r`/purity tables below and `runs_purity/fg_vs_mvg_purity.png` stand.
>
> **Corrected accuracy (re-derived from existing logs, per goal, last 5,000
> generations, `scratch_metrics_table.py`; no re-training):**
>
> | arm | acc on AND | acc on OR |
> |---|---:|---:|
> | Run 5 FG, capped fan-in | **0.904** | n/a |
> | Run 5 MVG, capped fan-in | **0.952** | 0.965 |
> | Run 6 FG, no fan-in | **0.970** | n/a |
> | Run 6 MVG, no fan-in | **0.998** | 0.996 |
>
> The MVG>FG direction survives in both run sets when measured on the matched task —
> this is KA's evolvability claim, not a new one. A single saved MVG champion scores
> ~0.50 on AND; that is an OR-phase snapshot, not a failure, since the population
> re-solves AND within a few generations of each switch.
>
> **Fix:** `best_fit`→`peak_fit_any_op` (documented as not-an-accuracy), and
> `result.json` now records `acc_by_op` — the saved champion against *every* goal,
> the only FG-comparable figure. See `kashtan_alon/RESULTS.md` for the full note.

Detail for the "our own experiments" part of the Kashtan-Alon writeup — the raw
reproduction (`kashtan_alon/`), distinct from the NDP port covered above. Covers
all 20 saved runs: `runs/` (paper-spec fan-in, 5 FG + 5 MVG) and
`runs_no_fanin/` (fan-in-cap-removed ablation, 5 FG + 5 MVG).

**Exact commands run.**
- Initial 10 (paper-spec fan-in), 2026-08-03:
  `conda run -n lndp python run_paper.py --n-seeds 5 --viz --fresh`
- Ablation 10 (fan-in cap removed): `conda run -n lndp python run_ablation_no_fanin.py --n-seeds 5`.
  Originally run 2026-08-20 at `--n-seeds 3`, extended to 5 on 2026-09-10 by
  re-running the same command (resume logic skipped the 3 already-complete
  seeds/condition and trained only the 2 missing ones). Every GA parameter is
  identical to `run_paper.py`'s — the only code difference is
  `NetConfig(fan_in=())` in place of the default `(3,3,3,2)`, plus a separate
  `--out-dir ./runs_no_fanin`. Verified by diffing the two scripts directly,
  not just their docstrings.

**Genome representation and how it's evolved (`model.py`, `ga.py`).** The
paper's genome is *"a fixed size of 15 genes each encoding a neuron"* — one
gene per non-retina neuron (8+4+2+1 = 15 neurons; the 8 retina pixels are
inputs, not genome). Our genome is the same unit of heredity in a different
storage layout: instead of an explicit gene list, each neuron's gene is *its
column of incoming weights plus its own threshold*, stored as one slice of a
per-block weight tensor (`weights[l]`, shape `(pop, layer_l, layer_{l+1})`,
`int8 ∈ {−1,0,+1}`, `0` = no edge) and one slice of a per-block bias vector
(`biases[l]`, shape `(pop, layer_{l+1})`, `int8` = `−threshold`). Both are
vectorised across the whole population on axis 0 (no per-individual Python
loop during the forward pass). Weight *magnitude* is fixed at 1 (KA: *"each
connection had weight −1 or 1"*), so there is no "weight mutation" in the
gradient-descent sense — only sign flips and edge add/remove.

- **Initialization.** Every destination neuron starts with `k = round(0.5 ×
  cap)` incoming edges (its fan-in cap, or the full previous layer if
  unconstrained), each drawn to a random source with a random `±1` weight;
  all thresholds start at 0. `init_density=0.5` is our own choice — not
  paper-stated.
- **Elitism.** The top `L=150` of `S=600` genomes (by raw fitness) are copied
  to the next generation **unchanged**. `S`, `L` are paper-verified for the
  *circuit* experiment (main text); reused here for the neural-net experiment
  by analogy, since the paper doesn't restate them there (flagged deviation).
- **Crossover (`Pc=0.5` per offspring).** Two parents are drawn from the
  elite pool. With probability `Pc` the offspring is built by, **for every
  destination neuron independently**, inheriting that neuron's *entire gene*
  (its full incoming-weight column **and** its threshold, together) from one
  parent or the other, chosen with 50/50 odds per neuron. This is exactly the
  paper's *"neuron-level"* recombination — crossover swaps whole genes, never
  splits a single neuron's incoming connections across both parents. An
  offspring that skips crossover just clones parent A outright.
- **Mutation (`Pm=0.5` per genome).** A genome selected for mutation gets
  **exactly one** random edit, drawn uniformly from four operators: add one
  edge (into a destination neuron under its fan-in cap, random source, random
  `±1` weight), remove one existing edge, flip one existing edge's sign
  (`+1↔−1`, magnitude never changes), or nudge one neuron's threshold by `±1`
  (clamped to `[−3,+3]`). `Pm`, the elite strategy, and `±1` weights are
  paper-verified; **the specific 4-operator mutation set and the `[−3,+3]`
  threshold range are our own reconstruction** — the paper's Supporting
  Information (where these would be specified) is not available to us (see
  `PAPER_SPEC.md`). `[−3,+3]` is chosen as the full non-trivial range: with
  `{0,1}` activations, `±1` weights and fan-in ≤3, the raw weighted sum lies
  in `[−3,3]`, so a threshold outside that band would just pin the neuron
  permanently on or off.

**GA hyperparameters (`run_paper.py`'s "paper-locked" preset, identical for
every one of the 20 runs — only `fan_in` and the FG/MVG combiner differ):**

| parameter | value | note |
|---|---|---|
| population size | 600 | matches paper exactly |
| generations | 25,000 | matches paper exactly |
| architecture | retina(8)→8→4→2→1, weights ∈ {−1,+1}, threshold units | matches paper exactly |
| fan-in cap | (3,3,3,2) — capped runs; unbounded — ablation | capped matches paper; ablation is our own test |
| initial edge density | 0.5 (fraction of the fan-in cap, per node) | our choice — not stated in paper |
| elite count | 150 of 600 | reconstruction by analogy — **not** paper-stated for the neural-net experiment (deviation 4 below) |
| crossover probability `Pc` | 0.5 | reconstruction — Supporting Info unavailable (deviation 5) |
| mutation probability `Pm` | 0.5 | reconstruction — Supporting Info unavailable (deviation 5) |
| fitness measure | raw fraction-correct over all 256 patterns | KA's own measure; see deviation 1 below (paper samples 100 patterns/gen, we use all 256 every gen) |
| MVG goal-switch interval `E` | 20 generations | matches paper exactly |
| `Q_m` randomizations | 1,000 | matches paper exactly (`--qm-nrand`) |
| `Q_m` estimation method | degree-preserving hill-climb, 6 restarts × 250 steps | **not** the paper's method (deviation 2 below: paper re-evolves 100 populations toward Q) |
| per-generation log interval | every 10 generations | our own choice, for the CSV trajectory |
| seeds per condition | 5 (both the capped runs and the ablation) | |

**The task (identical for all 20 runs; only the top-level combiner differs by
condition).** Architecture: retina(8) → 8 → 4 → 2 → 1, hard-threshold neurons,
weights ∈ {−1,+1}, fan-in ≤3/≤3/≤3/≤2 (capped runs) or unbounded (ablation).
Evaluated **exhaustively over all 2⁸ = 256 input patterns** (not a sample).
Pixels 0-3 are the left 2×2 retina block (0,1 = outer/left column; 2,3 = inner
column), pixels 4-7 the right block (mirrored: 6,7 = outer/right column; 4,5 =
inner column).

```
LEFT(x)  = 1  iff  (x0+x1+x2+x3 ≥ 3)  OR  (x2=0 AND x3=0 AND x0+x1 ≥ 1)
RIGHT(x) = 1  iff  (x4+x5+x6+x7 ≥ 3)  OR  (x4=0 AND x5=0 AND x6+x7 ≥ 1)
```
(the right rule mirrors the left rule over its own outer column x6,x7 and
inner column x4,x5). Each half-rule is true for exactly 8 of 16 half-patterns.

- **FG**: `y = LEFT(x) AND RIGHT(x)`, fixed for all 25,000 generations
  (true for 64/256 patterns — 25%).
- **MVG**: alternates every 20 generations — `y = LEFT AND RIGHT` for 20 gens,
  then `y = LEFT OR RIGHT` for 20 gens, repeating (OR true for 192/256 — 75%).

**Fitness: raw fraction-correct, not balanced accuracy.** `model.py`'s
`raw_accuracy` = `(pred == y).mean()` over all 256 patterns, no class
weighting — this is Kashtan-Alon's own measure, locked into `run_paper.py`'s
preset (`fitness="raw"`) and confirmed by every result filename
(`retina_fg_raw_seed*`, `retina_mvg_raw_seed*`). **Not shortcut-safe**: since
AND is only true 25% of the time, "always predict 0" already scores 0.75 for
free — a known trap (see the CLAUDE.md shortcut warning). `train.py` also
implements `--fitness balanced` (chance = 0.5 regardless of class imbalance,
the thesis's own shortcut-aware convention used elsewhere) but it was **never
invoked** for any of these 20 runs — `raw` only, matching the paper.

**Where our implementation differs from the paper (applies to all 20 runs,
not only the ablation):**

1. **Fitness evaluation set.** Paper: *"The environment contained 100
   different randomly chosen retina patterns"* — fitness is fraction-correct
   over a 100-pattern sample, redrawn (or not — unstated) each generation.
   Ours: exhaustive over all 256 patterns, every generation. Removes the
   paper's sampling noise entirely; an easier optimisation problem than the
   one KA actually ran.
2. **Q_max estimator.** Paper: Q_max is obtained by **re-evolving** the
   population with Q itself as the fitness, then averaging the best network's
   Q over 100 such simulations. Ours (`modularity.py:normalized_qm`): a
   **degree-preserving hill-climb** (6 restarts × 250 steps) directly on the
   already-evolved network — a cheap proxy for the paper's method, not a
   reproduction of it, and a likely direct contributor to the absolute-Q_m
   gap below (a hill-climb from a real network is not guaranteed to reach the
   same optimum as 100 independent re-evolutions toward Q).
3. **Missing neuron-count penalty.** Paper: *"A penalty of 0.01 was applied
   for every additional neuron above ... 13 neurons"* (applies to both MVG
   and FG). Absent from `model.py`'s `fitness()`, which is pure accuracy with
   no complexity term. Confirmed fidelity gap (see `kashtan_alon/PAPER_SPEC.md`
   §5), present in every run.
4. **Elite count L = 150/600.** Not stated in the paper's main text for the
   neural-network experiment (only the separate circuit experiment states
   300/1000); ours is a reconstruction by analogy.
5. **Mutation operator set, crossover mechanism, and threshold/bias range.**
   None of these appear in the paper's main text — they live in the
   Supporting Information, which is not available to us (see
   `kashtan_alon/PAPER_SPEC.md`). `ga.py`'s implementations are documented
   reconstructions, not verified against the source.

**Q_m: paper vs. ours.**

| | Q_rand: n randomizations | Q_max method | n seeds | Q_m |
|---|---|---|---|---|
| Paper, MVG | 1,000 | re-evolution, avg of 100 sims | not stated | **0.35 ± 0.02** |
| Paper, FG | 1,000 | same | not stated | **0.15 ± 0.02** |
| Ours, MVG (`runs/`) | 1,000 | degree-preserving hill-climb | 5 | **0.245 ± 0.049** |
| Ours, FG (`runs/`) | 1,000 | degree-preserving hill-climb | 5 | **0.025 ± 0.139** |

`--qm-nrand` (1,000) matches the paper exactly. Direction and significance
reproduce (Welch t≈3.34, p≈0.02; Mann-Whitney p≈0.03; gap ≈0.22 vs. paper's
0.20), but absolute magnitude sits below the paper on both arms — consistent
with deviations 1-3 above all pushing the same way (easier fitness signal,
different Q_max method, no complexity penalty). FG's variance is also much
higher than the paper's ±0.02 (one FG seed, 0.228, is as modular as MVG).

**`left_right_q`'s `r` (planted left/right partition, no KA-paper equivalent —
scored 2026-09-10, `scratch_metrics_table.py`/`RESULTS.md` Run 7, `n_rand=200`,
analysis-only on the saved final genomes, no re-evolution):**

| run | condition | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Run 5 (capped fan-in) | FG | 0.064 | 0.777 | 0.674 | 0.645 | 0.570 | **0.546** |
| Run 5 (capped fan-in) | MVG | 0.765 | 1.000 | 0.815 | 1.000 | 1.000 | **0.916** |
| Run 6 (no fan-in, n=5) | FG | 0.516 | 1.000 | −0.119 | 1.000 | 0.287 | **0.537** |
| Run 6 (no fan-in, n=5) | MVG | 0.743 | 0.695 | 1.000 | 1.000 | 0.394 | **0.766** |

Direction (MVG > FG) survives on `r`, capped and uncapped, the same as it does
on `Q_m` (Run 7 above) and on `circuit_purity` (table below) — all four
metrics agree on direction even though they disagree on absolute scale and,
in the no-budget `experiment_1` side-study above, occasionally on
significance. `r` has no direct paper equivalent to compare magnitude
against — KA never computed a planted-bipartition score, only Newman `Q_m`
with a greedy-detected partition.

**`circuit_purity` (same runs, same table source), for completeness:**

| run | condition | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Run 5 (capped fan-in) | FG | 0.344 | 0.656 | 0.571 | 0.635 | 0.598 | **0.561** |
| Run 5 (capped fan-in) | MVG | 0.857 | 1.000 | 0.833 | 1.000 | 0.952 | **0.929** |

Two of the five Run 5 MVG seeds (1 and 3) score **exactly 1.000** — verified
node-by-node (`scratch_purity_audit.py`) to be a genuine complete left/right
split (every hidden neuron's live parents, including 2- and 3-way
convergences, trace back to only one retina side), not a degenerate/trivial
score from single-input pass-through wiring. This is a literal instantiation
of the paper's own qualitative Fig. 5e description — *"Two distinct
(nonidentical) modules evolved spontaneously in the network, each monitoring
a different side of the retina"* — but `circuit_purity` itself is a metric
built for this project's exp_4 Boolean-circuit work and applied here as a
diagnostic; KA never computed it, so there is no paper number to compare
the magnitude against, only the qualitative match.

**Per-generation progress is logged, not just the final state.** Every run
writes `<run_name>_log.csv`, sampled every 10 generations
(`gen,op,best_fit,mean_fit,Q,density,edges`; 2502 rows, gen 0→24990) — so raw
Newman **Q** (not Q_m, which is only computed once, at the end) has a full
trajectory for every one of the 20 runs, in both `runs/` and
`runs_no_fanin/`. **Purity per-generation** only exists for the separate
`runs_purity/` archive (`analysis/fg_mvg_purity.py`, its CSVs add a `purity`
column) — a deterministic duplicate of Run 5's 10 genomes, built specifically
to add live purity/accuracy/champion-brain logging on top of the paper
preset. No equivalent per-generation archive exists for the ablation runs.
Already plotted: `runs_purity/fg_vs_mvg_purity.png` — champion-brain circuit
purity (left panel) beside champion-brain accuracy (right panel), mean ± 1 SD
across 5 seeds/arm, generation 0→25,000, MVG vs FG overlaid on both panels
(figure-ready for the thesis appendix as-is).
`left_right_q` has never been logged per-generation anywhere — only computed
post-hoc on saved final genomes (`kashtan_alon/scratch_lr_table.py`,
`scratch_metrics_table.py`).

### `left_right_q`'s own chain of numbers (`qmetrics/metrics.py:596-700`)

It is not one number — every call produces this whole sequence, in the order
computed:

| step | quantity | formula / how it's obtained | what it tells you |
|---|---|---|---|
| 1 | `q` | plain Newman Q, evaluated **at the fixed left/right partition** you hand it (no search) | raw fraction-internal-edges minus expected — exact, no simulation |
| 2 | `r` | `q / ceiling`, `ceiling = 1 − Σaᵢ²` (`aᵢ` = fraction of total degree in group *i*) | rescales `q` by the best it could **structurally** ever be if every edge were internal, given these group sizes — deterministic, no null needed. 1 = perfectly split, 0 = chance, <0 = anti-associated |
| 3 | `crosstalk` | `(actual cross-group edge fraction) / ceiling` | same information as `r` in a different dress — `ceiling` is also the *expected* cross-edge fraction under the null, so this reads as "how much crosstalk relative to chance." **Identity: `r = 1 − crosstalk`** (checked by the test suite) |
| 4 | `q_rand` | mean `q` over 200 degree-preserving rewirings, scored **at the same fixed partition** (not re-detected) | Monte Carlo null — KA's own "control 1," done at a planted rather than searched partition |
| 5 | `q_max` | `q` of a rewiring that greedily maximizes within-group edges, floored at `≥ q` | the **achievable** ceiling given the real degree sequence/edge mask — tighter than step 2's `ceiling`, which assumes every edge *could* be internal even when the topology can't reach that |
| 6 | `score` (the function's actual return value) | `(q − q_rand) / (q_max − q_rand)` | a Kashtan-Alon-style **normalized** modularity, evaluated at the planted partition instead of a searched one — distinct from `r`, don't conflate the two |
| 7 | `z`, `p` | `z = (q − q_rand)/sd(nulls)`; `p` = fraction of null samples `≥ q` | significance only — is `q` distinguishable from the null *distribution*? Biased by graph size (bigger `m` → smaller null SD → bigger `z` for the same effect) — never use for cross-network magnitude comparisons, only within one network |

*⭐ `r` (equivalently `crosstalk`) is probably the easiest of these to actually
use: it's a closed-form ratio with no Monte-Carlo noise, needs no
random-seed-dependent null, and has a clean plain-English reading ("X% fewer
left-right crossings than chance"). `score` is the one to reach for when a
KA-comparable normalized number is wanted instead (built the same way as
`Q_m`), but it inherits sampling noise from `q_rand`/`q_max` that `r` doesn't
have. `z`/`p` answer a different question (significance, not magnitude) and
should not be reported as if they were effect sizes.*

### Does any of this actually respect the network's real structure?

Checked directly in the code, per metric — the answer is not the same for
all four:

- **Raw Newman `Q`** (`modularity.py:newman_q`): no null model at all, so the
  question doesn't apply — it only ever measures the real evolved graph.
- **`Q_m`, the version actually used everywhere in `RESULTS.md`/`result.json`/
  the picture above** (`kashtan_alon/modularity.py:normalized_qm`, imported by
  `train.py`): its null (`_degree_preserving_random`) calls plain
  `nx.double_edge_swap` with **no mask at all** — it does not know layers are
  adjacent-only, let alone that fan-in is capped at 3/3/3/2. This is the exact
  gap the "Constraints on the null model" section above already measured
  (54.9% of unconstrained-null edges illegal on this architecture) — but that
  fix lives only in `qmetrics`'s own `normalized_qm`/`_q_max_planted`, used
  for a one-off correctness check, and was **never ported into
  `kashtan_alon/modularity.py`**. So **every officially reported Q_m number
  for all 20 runs — Run 5 through 7, and the brain-grid figure's subtitles —
  is computed against a structurally-invalid null**, and per that section's
  own finding this makes them **understate** true modularity (the corrected
  numbers came out higher, several roughly doubled).
- **`left_right_q`**: its null (`_rewire`/`_lr_null`) *does* honour
  `G.graph['allowed']` — but only because `qm.from_blocks(..., constrain=True)`
  (the default, used by every `left_right_q` call in this repo) auto-attaches
  a `layered_mask` (adjacent-layers-only). So `q_rand`/`q_max`/`score`/`z`/`p`
  correctly refuse to invent retina-to-retina or layer-skipping edges. It
  still does **not** know about the fan-in cap (≤3/≤3/≤3/≤2) — that's a
  per-node degree limit, not a pairwise mask, and per-node counters in the
  rewirer are "not yet implemented" anywhere in this codebase (same caveat as
  `Q_m`). `r`/`crosstalk` are unaffected either way — they use no null at all.
- **`circuit_purity`**: no null model, no random comparison network, so the
  layer/fan-in question doesn't arise — it's one deterministic pass over the
  *actual* DAG's real parent edges. It **deliberately** excludes the pinned
  inputs and the program output from the averaged score (`drop = exclude |
  pinned` in the code) — not an oversight: inputs are trivially "pure" by
  definition (they *are* the label), and a single output node must see both
  halves by construction, so scoring it would charge a fixed penalty
  unrelated to modularity, exactly mirroring `left_right_q`'s own `exclude`
  argument for the same reason.

---

## General observations (across all experiments)

Running log — synthesis across NDP/experiment_1/2/3/kashtan_alon, not tied to
any one of them. More to add as we go.

- **We need a DNA→brain framework that can build sparse brains — sparsity is
  a precondition for measuring modularity at all**, not just a nice-to-have.
  Dense graphs compress Newman Q toward 0 regardless of wiring (see
  `qmetrics` density-confound note above), so a framework that only produces
  dense brains gives us nothing to measure modularity *on*.
- **Recurring problem: at the small network sizes we work with, our
  compressed encodings end up more complex than direct encoding, not less.**
  The genome overhead of a shared rule (`g` in experiment_1, the growth/weight
  MLPs in NDP) is roughly fixed regardless of network size, so at tens of
  neurons that fixed cost isn't yet amortised — direct encoding (one number
  per edge) is smaller and simpler at this scale.
- **Ironic inversion: the compression argument only pays off at the scale we
  are deliberately avoiding.** A brain with millions of neurons is exactly
  where an O(K) or O(1)-in-node-count genome would be a genuine win over an
  O(N²) direct encoding — but a network that large makes the shared-rule
  genome itself unwieldy and computationally expensive to search (bigger
  rule networks, more expensive forward passes per genome evaluation), which
  is why this thesis stays at small N by design (parsimony-first constraint).

### Supporting tools (not metrics)

- **Threshold sweep** — Q across prune levels (cutoffs = quantiles of non-zero
  |w|). Separates "structure buried under weak edges" (Q climbs as tail is cut)
  from "one blob" (flat, low). Caveat: pruning manufactures modularity — any
  graph → Q ≈ 1 once fragmented — so read Q next to the edge count.
- **Role segregation** — per-hidden-neuron s = (L−R)/(L+R) over incoming |w|.
  Retina-only ground-truth check to validate the real metrics; not comparable to
  published Q values.
