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
- **Worth testing: does constraining also make search converge faster**, not
  just sparser/more modular? Open question, not yet run as its own comparison.
- **Hunch for *why*, if the speed effect holds**: constraints shrink the
  search space, and the solution happens to sit inside the constrained
  subspace — so a constrained search is effectively searching a smaller
  haystack containing the same needle, while an unconstrained search wastes
  budget covering the larger space outside it.

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
