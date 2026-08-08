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

---

## Modularity metrics (`qmetrics/`)

Shared package at repo root: adapters turn any brain format (NDP's `W`,
exp 1–3's `w`, `kashtan_alon`'s layer blocks) into a graph; metrics see only the
graph. **Three metrics so far.**

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
  0 = chance, <0 = anti-associated. Identity worth quoting: **r = 1 − crosstalk**,
  where crosstalk = (cross-group edges)/(cross-group edges expected at these
  degrees). So the formal metric and the one-line prose number are the same object.
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

### Supporting tools (not metrics)

- **Threshold sweep** — Q across prune levels (cutoffs = quantiles of non-zero
  |w|). Separates "structure buried under weak edges" (Q climbs as tail is cut)
  from "one blob" (flat, low). Caveat: pruning manufactures modularity — any
  graph → Q ≈ 1 once fragmented — so read Q next to the edge count.
- **Role segregation** — per-hidden-neuron s = (L−R)/(L+R) over incoming |w|.
  Retina-only ground-truth check to validate the real metrics; not comparable to
  published Q values.
