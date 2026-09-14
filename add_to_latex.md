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
  axis-aligned. ⚠️ Earlier notes said an elitist variant (`CMA_elitist`) was used.
  That is **wrong for experiments 1 and 2**: they run evosax 0.1.6 `CMA_ES`, which
  has no elitism (μ = 32 of λ = 64 recombined, nothing carried over). Not
  re-checked for the NDP runs.

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
- **NDP cannot tell one input neuron from another (confirmed 2026-09-14).**
  - **What it does distinguish:** input from output (separate role embeddings),
    and hidden neurons by where they grew.
  - **What it cannot distinguish:** input 0 from input 5. They share one
    embedding and one neighbourhood, and every rule is applied identically. So
    each input grows an exact copy of what every other input grows, and the
    brain can only react to *how many* inputs are on.
  - **The exception is rounding.** In floating point, rounding occasionally
    breaks the tie. Evolution can exploit that briefly, but it never survived
    into a final brain.
  - On the retina task this caps NDP at a majority vote, whatever the size or
    budget. Full evidence, the perfect brain NDP cannot grow, and what it means:
    "Result: every brain plateaus at a majority vote" and the sections after it,
    under the retina study below.
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

### The task we actually used, and why (NDP FG vs MVG study)

**A slightly different task from Kashtan & Alon (2005).** We did not use KA's
Fig. 5a object rule. We used the project's stand-in retina task
(`ka_task.py`, a copy of `shared_tasks.py`, shared with experiments 1-3):

    L = (x0 AND x1) OR (x2 AND x3)        -- left 4 pixels
    R = (x4 AND x5) OR (x6 AND x7)        -- right 4 pixels
    target = L AND R   (FG);   L AND R  <->  L OR R   (MVG)

evaluated on all 2^8 = 256 patterns, inputs bipolar (±1). In KA's task the left
and right objects are mirror images under a more involved rule ("three or more
black pixels, or one or two black pixels in the left column only";
`kashtan_alon/PAPER_SPEC.md`). In ours, L and R are the **same Boolean
expression applied to two disjoint blocks of inputs**.

- **Motivation: symmetry should raise the potential for modularity.** The task
  decomposes into two identical sub-problems joined by one gate. A solution
  that builds one L/R sub-circuit and repeats it is available by construction,
  and a two-module brain (one module per side, joined at the output) is the
  natural decomposition. If modularity is ever going to appear, it should be
  on a task where the modules are identical and cleanly separable. This is the
  design rationale; it was not tested against KA's own rule in NDP.

**Balanced fitness.** The dataset is imbalanced: under AND only 49/256 patterns
(19%) are positive, and under OR 175/256 (68%).

- **Motivation:** with raw accuracy, "always predict 0" already scores 81% on
  AND, with no fitness gradient toward the real solution. So an evolved brain
  can look good without computing anything. Fitness is therefore **balanced
  accuracy**, the mean of the accuracy on positive and on negative patterns
  (`--balanced-fitness`, `train_backend.py::retina_fitness`). Chance is 0.5
  whatever the class balance, and the two MVG goals score on the same scale
  despite opposite imbalance.

### Study: 5 FG + 5 MVG brains, budget-matched (2026-09-13)

We trained **5 Fixed-Goal (FG) and 5 Modularly-Varying-Goal (MVG) brains** with
identical settings except the goal schedule. Every `config.yml` key was diffed
across all 10 runs. The only differences are `mvg`, seed and bookkeeping
(NDP `experiments_paper/retina/RESULTS.md`, "Budget-matched FG vs MVG seed set").

**Goals.**
- FG: `L AND R` for all generations.
- MVG: goal alternates `L AND R` / `L OR R` every **20 generations**, starting on
  AND. That gives 75 epochs, the last one on AND, so both arms' final
  champions are scored on the same goal.

**Evaluation.**
- Each pattern is scored from a zeroed network state. The 8 inputs are clamped
  to the pattern at every propagation step.
- The network is recurrent with `tanh` activation, and propagates for (graph
  diameter + 3) steps.
- The brain has a **single output neuron**, and the prediction is its sign.
- Evaluation is deterministic: one growth run and one rollout per genome
  (`nb_growth_evals = nb_episode_evals = 1`).

**Brain / developmental program.**
- Seed graph: 9 neurons, the 8 inputs each connected to the single output (8
  undirected edges). Undirected, no self-loops.
- **No input–input edges** (and no output–output edges), in the seed or in any
  grown brain. Inputs are re-clamped to the pattern at every propagation step,
  so an edge into an input could never change anything (NDP `train.py:159`,
  `forbid_io_self_edges`).
- Evolved role embeddings (size 3): one shared by all inputs, one for the
  output.
- Node-based growth, **6 growth cycles**. In each cycle:
  1. Embeddings propagate for (diameter + 2) steps, each followed by the
     embedding-transform MLP.
  2. The growth MLP decides, per neuron, whether it spawns a child (output
     > 0). A child is wired to its parent's whole neighbourhood and starts at
     the mean of those neighbours' embeddings.
  3. The weight MLP sets every edge weight from its two endpoint embeddings.
- MLPs, all tanh hidden layers with bias:
  - growth 3→5→1 (26 params);
  - embedding transform 3→5→3, tanh-bounded output (38);
  - edge weight 6→5→1, tanh-bounded, so weights lie in [−1, 1] (41).
- Genome = **111 parameters** (6 role-embedding + 26 + 38 + 41).
- **No pruning.** An earlier FG run with pruning at threshold 0.3 (run
  1786102425) grew exactly the same brain as the unpruned run: 40 neurons, 502
  edges, balanced accuracy 0.8438. Every edge weight had saturated near 0.9997
  under the tanh output, far above 0.3, so nothing was pruned. We therefore run
  without pruning (NDP `experiments_paper/retina/RESULTS.md`, "Run 1786102425").
- No size or edge regularisation, no cap on neuron count. The largest possible
  brain is 9·2^6 = 576 neurons.

**Optimiser.** CMA-ES (`pycma`):
- population **128**, **1500 generations**;
- σ_init = 0.5, minimum std 0.005, x0 ~ U[−1, 1];
- **elitist** (`CMA_elitist: True`);
- **no early stopping**: every run goes the full 1500 generations, so the
  final brains of all 10 runs are compared at the same point, after the same
  budget.

The saved FG brain is the best-ever genome. The saved MVG brain is the
final-generation champion, because best-ever is not comparable across goal
switches.

**Seeds.** Drawn independently per run (`np.random.randint(10**7)`, NDP
`train.py:28`).

| Arm | Run ID | Seed |
|---|---|---|
| FG | 1789303257 | 3410590 |
| FG | 1789303833 | 7345678 |
| FG | 1789304372 | 7737269 |
| FG | 1789304978 | 1231914 |
| FG | 1789305630 | 5316271 |
| MVG | 1786053806 | 7318332 |
| MVG | 1788817038 | 6550047 |
| MVG | 1788821061 | 7766560 |
| MVG | 1788866451 | 6779840 |
| MVG | 1788871442 | 9985617 |

**Exact commands** (NDP repo, `conda` env `ndp`; config
`experiments_paper/retina/run_experiment.yaml`):

    FG:  python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 1500 --popsize 128 --balanced-fitness --no-early-stopping --operation and --snapshot
    MVG: python train.py --conf experiments_paper/retina/run_experiment.yaml --generations 1500 --popsize 128 --balanced-fitness --no-early-stopping --mvg --mvg-ops and,or --mvg-switch-interval 20 --snapshot

⚠️ An earlier 5 + 5 set ran FG for only 200 generations against MVG's 1500. That
confounds any size or modularity difference with the budget. The set above
replaces it.

### Result: FG vs MVG over training: MVG does not change anything here

Figure: `latex_figures/NDP/KA/progress_fg_vs_mvg_matched.png`. It shows the mean
of the 5 runs per arm, shaded ±1 SD, for the best brain of each generation. The
FG runs are sampled at 60 points; the MVG runs at the end of each of their 38 AND
epochs, so both arms are scored on the same goal. Made by NDP
`experiments_paper/retina/matched_figures.py progression` from per-generation
replays of all 10 runs. Every replay reproduced its run's logged best-ever
fitness, population mean and saved genome exactly (`replay_archive.py`).

- **Accuracy: a complete plateau from the first few generations, identical in
  both arms.** Both lines sit at ≈ 0.844 for 1500 generations. MVG's mean is
  0.846 against FG's 0.8438, only because of its one 0.8540 run.
  - Everything away from 0.8432 is floating-point rounding (see below), so
    there is no real accuracy difference.
  - The FG dip to 0.834 in the last ~40 generations is one run (1789305630). Its
    best brain of each of those generations scores 0.78–0.80: the same counting
    brain with its 4-inputs-on answers rounding the other way.
- **Density: both arms settle by ~generation 450.** FG ≈ 55%, MVG ≈ 42%, but the
  ±1 SD bands (≈ 17–90%) overlap almost completely. They mix the 12- and
  16-neuron brains with the 40–80-neuron ones.
- **L/R modularity (lr_r): both settle between 0.1 and 0.2.** FG ≈ 0.12,
  MVG ≈ 0.17, with fully overlapping bands.
- **The final brains agree.** All 10 compute the same function (a majority vote,
  below). 3 FG and 3 MVG brains share one identical 40-neuron wiring.

**So, where MVG helped in the Kashtan–Alon case, here it does not seem to do
anything.**
- In Kashtan & Alon (2005), and in our faithful reproduction of it, MVG produces
  clearly more modular networks than FG. The reproduction gives Q_m
  0.245 ± 0.049 against 0.025 ± 0.139 over 5 seeds each; Welch p ≈ 0.02,
  Mann–Whitney p ≈ 0.03 (UCL_thesis `kashtan_alon/RESULTS.md`, Run 5).
  That network and task differ from ours (±1 weights, threshold units, KA's
  own object rule).
- In NDP on the retina task, FG and MVG give the same accuracy, the same kind of
  brain and overlapping density and modularity.
- The sections below explain why MVG has nothing to work with here. NDP can
  only build counting brains, and the same counting brain is the best
  available for both AND and OR, so switching goals never rewards a different
  brain.

### Result: every brain plateaus at a majority vote (checked 2026-09-14)

**All 10 brains, FG and MVG, from 12 to 80 neurons, compute the same function:
count how many of the 8 inputs are on.** 5 or more on → 1, 3 or fewer → 0.
With exactly 4 on, the output is exactly 0 in exact arithmetic, so it predicts 0.
However complicated a grown brain looks, the hidden neurons change nothing
about this.

- **Reported scores.** 8 brains score 0.8438, 2 score 0.8540. Both are rounding
  artefacts. The true score of every brain is **0.8432**. That is the best any
  count-based rule can do: all 2^9 rules were checked, and the best is "1 iff ≥ 5
  on".
- **Why counting can't solve the task.** `11001100` (label 1, x0∧x1 and x4∧x5)
  and `10101010` (label 0, no pair on) both have 4 inputs on. Only *which*
  inputs are on separates them, and a counter gives both the same answer.

Checks, on all 10 brains regrown from their saved genomes (scripts
`counting_proof.py`, `counting_proof2.py`, `odd_ceiling.py`):
1. **Same count → same output.** Patterns with the same number of inputs on give
   identical outputs, up to 1e-16. This already holds after the first
   propagation step, while outputs are still ≈ ±0.93–0.96, so it is not tanh
   saturation hiding differences.
2. **Flipping every input flips the output exactly:** f(−x) = −f(x), to 7e-15.
3. **Exactly 4 on → output 0.** Flipping a 4-on pattern gives another 4-on
   pattern. Check 1 says the output stays the same; check 2 says it flips sign.
   Only 0 satisfies both. Measured: |output| ≤ 1e-12, against ±1 elsewhere.
4. **The 4-on answers are floating-point noise.** Renumbering the hidden neurons
   leaves the network mathematically identical, but changes the project's own
   score (`retina_fitness`): FG 1789303257 gives 0.7839–0.8438, MVG 1788866451
   0.7737–0.8438, MVG 1786053806 0.7839–0.8438. The two 0.8540 brains are stable
   under renumbering but give 0.8274 / 0.8202 under a batched matrix product.
5. **Treating the 4-on zeros as 0 gives every brain 0.8432.**

This settles the open point in NDP `experiments_paper/retina/RESULTS.md:260-269`
(why two runs reach 0.8540): they are not better brains.

### Why: NDP cannot tell its inputs apart

**What NDP does differentiate.**
- **Inputs from the output:** they get separate evolved role embeddings
  (change 1 above).
- **Hidden neurons by where they grew:** a child starts at the mean of its
  neighbours' embeddings. In practice hidden neurons come in two kinds, those
  wired to exactly 1 input (descendants of that input) and those wired to all 8
  (descendants of the output). None is wired to 2–7 inputs, in any of the 10
  brains. Their weights can differ, e.g. 0.81–1.00 in FG 1789303257.

**What it cannot differentiate: one input from another.**
- All 8 inputs share one embedding. There are only 2 evolved role vectors, input
  and output (NDP `train_backend.py:62-67`, `:843`).
- All 8 inputs start with the same neighbourhood: just the output.
- The growth, embedding-transform and weight MLPs are applied identically
  everywhere. A child is wired to its parent and the parent's whole
  neighbourhood (NDP `NDP.py:407-411`).
- So whatever input 0 grows, input 5 grows an exact copy of it, with the same
  weights. The brain is unchanged by any relabelling of its inputs, so its
  output can only depend on how many inputs are on.
- This is an argument from the setup, confirmed on all 10 brains. It is not a
  formal proof for every genome, but nothing in it depends on what evolution
  found.

**A second, independent limit: no biases.**
- The rollout is `s ← tanh(Wᵀs)` from a zero state, with the inputs re-clamped
  each step and no bias term (NDP `NDP.py:230-240`). So f(−x) = −f(x) exactly,
  for *any* weights.
- The target is not sign-symmetric: `11001100` and its flip `00110011` are both
  positive.
- The best any sign-symmetric classifier can do is **0.980 on AND** and **0.843
  on OR**, computed exactly over the 128 pattern/flipped-pattern pairs.
- On AND, the input-identity limit is the one that binds. Combining both limits
  forces output 0 at 4 on, hence 0.8432.

**The one escape is floating-point rounding, and evolution does find it, briefly.**
In exact arithmetic the 8 inputs are identical. In floating point, sums done in
a different order differ by ~1e-16, and sometimes that difference gets amplified:
- **At a growth decision sitting on its threshold.** The best OR-epoch champion
  of MVG 1788817038 (generation 112, 256 neurons) grew asymmetrically: inputs 4
  and 5 have 111 connections, the other six 79. Its first growth cycle went 9 → 12
  neurons, which a symmetric growth cannot produce.
- **Through the recurrent dynamics.** In the same brain the 4-on output is
  3e-9 after one step and ±1 after two.

Such brains score above the exact caps during OR epochs:
- the best OR champion of each of the 5 MVG runs scores **0.8429**,
  against **0.7657** for the best majority-style rule on OR;
- the gain is fragile. Renumbering the hidden neurons of MVG 1786053806's OR
  champion (generation 233, a perfectly symmetric brain) drops it to 0.7935,
  and relabelling its inputs gives 0.78–0.81.

None of this survived: all 10 *final* brains are exact counters (checks above).

### What this means

- **Only functions of how many inputs are on are reachable.** Any task where it
  matters *which* input is active is out of reach for this NDP set-up: both
  retina variants, KA's task, even "x0 AND x1". More neurons, generations or
  population cannot help. The limit is in the encoding, not the search.
- **NDP cannot build a left and a right module in the functional sense.** A
  left module is a group of neurons that responds to x0–x3 and not to x4–x7,
  and no NDP brain can treat x0 differently from x4. The wiring does have
  structure, but it comes from growth history: each input and its own
  descendants, identical across inputs. It is not a left/right split.
- **MVG has nothing to select for here.** The majority vote is at once the best
  expressible brain for AND (0.8432) and for OR (0.7657). Every other
  count-based rule NDP can express scores lower on both. So switching goals
  never rewards a different brain: FG and MVG are pushed toward the same
  function. The FG-vs-MVG comparison in NDP therefore cannot show a Kashtan–Alon
  style effect on this task, whatever the modularity numbers say.
- **Read NDP fitness curves with care.** Scores above 0.8432 (AND) or 0.7657
  (OR) come from floating-point artefacts, not from computing anything new.
  This includes the OR-epoch values, and the 0.8438/0.8540 of the final brains.
- **What would lift the limit (untested in NDP):**
  - a distinct embedding or positional code per input;
  - a bias term in the rollout;
  - deliberate symmetry breaking in the seed graph.
  The first two are exactly what experiment 1's cell-type model has (next
  section).

### A perfect brain exists, but NDP cannot grow it

- **Oracle.** Experiment 1's oracle hand-wires a recurrent tanh brain for this
  same stand-in retina AND task (UCL_thesis `experiments/experiment_1/oracle.py`,
  `build_oracle_retina`, lines 92-127). The wiring:
  - 4 AND detectors, each reading a different input pair (x0∧x1, x2∧x3, x4∧x5,
    x6∧x7);
  - 2 OR combiners, one per side;
  - an AND at the output;
  - per-neuron biases.
- It scores **balanced accuracy 1.000** (`experiment_1/RESULTS.md:94-107`;
  re-run 2026-09-14 with `python oracle.py --task retina`, stage 1 = 1.000).
  The retina task does have a perfect solution in a small recurrent tanh network.
- **NDP cannot express that brain, for both reasons above.**
  1. Detector x0∧x1 must respond to x0 and x1 and ignore x2..x7. That needs
     inputs to be told apart.
  2. The detectors' thresholds are biases, and NDP's rollout has none.
- **So the 0.84 plateau in NDP is representability, not search.** No genome,
  population size or generation budget reaches a perfect brain in this
  framework. The perfect brain is simply not in the set of brains NDP can grow.
- **Consequence for the modularity question.** The function every brain
  computes treats all 8 inputs alike, so no brain in this study computes
  separate left and right sub-functions. Any left/right structure measured in
  the wiring does not correspond to a left/right *computation*.

### Motivation for the next model: K neuron types

- **The perfect brain needs neurons that differ in identity**, not just in
  position:
  - detectors that each read *different* inputs;
  - combiners that do a *different* job from detectors;
  - an output doing a third job.
  The oracle uses 6 distinct hidden roles.
- **NDP has exactly 2 identities, input and output.** Every other difference
  between neurons must come from position in the graph, and the seed graph is
  perfectly symmetric across inputs, so that symmetry is never broken.
- **This is what the cell-type model (experiment 1) adds**
  (`experiments/experiment_1/model.py:11-17`):
  - each input and output neuron gets a fixed positional code, so inputs are
    distinguishable;
  - hidden neurons carry one of **K evolved cell types**;
  - each type has its own bias.
  With K ≥ 6, the oracle's perfect brain is representable in that family
  (`experiment_1/RESULTS.md:94-107`). Fewer than 6 is not ruled out; the oracle
  only shows 6 suffice.

### What we tried in NDP, and why we moved on

We tried many settings on this task. Every one ended at the same ≈ 0.84
plateau, or failed before reaching it. Run IDs and details are in NDP
`experiments_paper/retina/RESULTS.md` and `saved_models/*/config.yml`.

- **Raw vs balanced accuracy.** The first runs scored raw accuracy; these were
  short runs of 1–3 generations (e.g. 1786027807, 1786031757). We switched to
  balanced accuracy because "always predict 0" already scores 81% raw.
- **Size regularisation** (`io_ratio`, penalising neurons beyond the 9 I/O
  anchors):
  - α = 1.0 with a 500-generation warm-up (20-generation test);
  - α = 0.04 over 1000 generations (run 1786030050);
  - α = 0.01 over 3000 generations (run 1786032037).
  With α = 0.04 and α = 0.01 the population collapsed to the 9-neuron seed with
  no growth: a majority vote over the direct input→output weights, balanced
  accuracy 0.8432. Weakening α 4× changed nothing.
- **Banning input–input and output–output edges:** seed graph 81 → 16 edges.
  The regularised runs ended at the same 0.8432.
- **No regularisation at all.** The brain grew freely, to 40 neurons, and scored
  the same (0.8438). It made the same mistakes as the 9-neuron brain: 51 of 52
  errors shared, all where 4–6 inputs are on (run 1786033855).
- **Pruning at threshold 0.3.** No effect: every weight had saturated at
  ≈ 0.9997, so the same 40-neuron brain came out (run 1786102425).
- **Elitism.** All retina runs used elitist CMA-ES. We varied elitism on
  LunarLander: Run 6 (elitist) froze its best score at −67 from generation 70
  while the population mean kept improving. Run 8 turned elitism off
  (`--no-elitism`). Its σ kept rising and the best improved to −72.93, then
  froze again; it was stopped at ~110 generations, unsolved
  (NDP `fluffy_experiments.md`). We did not retry it on retina, because the
  counting limit holds for every genome, whatever the optimiser does.
- **Short vs long budgets:** 200 generations, then 1500, for FG.
- **FG vs MVG:** 5 + 5 runs, twice — first budget-mismatched (200 vs 1500),
  then budget-matched (1500 each).

**The framework struggles with this task in general.**
- **Retina.** No setting got past the counting plateau. The sections above show
  why: NDP cannot tell its inputs apart, and its rollout has no bias, so the
  perfect brain is not in the set it can grow.
- **LunarLander.** The same pattern of being hard to control showed up there:
  8 runs, none solved it (see the LunarLander section above).

This is why we moved on from NDP to the cell-type model (experiment 1): each
input has its own positional code, and hidden neurons carry one of K evolved
types with their own biases.

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

### Overview — the framework on one page

The genome does not list synapses. It lists **cell types plus one wiring rule**.
The brain is grown from them once, frozen, and scored.

1. **Genome.** It holds `K` hidden cell-type identity vectors, one input identity
   and one output identity. It also holds `K` abundance genes, `K + 2` type
   biases, and the weights of a small MLP `g`. That is 443 real numbers at the
   study's settings.
2. **Development.** The abundance genes split the fixed pool of hidden neurons
   among the `K` types. Each neuron gets a feature vector
   `[type identity | positional code | role]`, and every allowed synapse i→j
   gets the weight `g(feat_i, feat_j)`.
3. **Constraint (budgeted arm only).** A synaptic budget shares a fixed total
   incoming |weight| among each neuron's synapses and zeroes the weak ones. This
   is where sparsity comes from.
4. **Behaviour.** The frozen recurrent network runs 8 synchronous steps on each
   of the 256 retina patterns. The sign of the output is the decision.
5. **Evolution.** CMA-ES proposes genomes and selects on a smooth margin
   surrogate. The goal is either fixed (FG) or alternates AND ↔ OR every 20
   generations (MVG).

The idea to carry into the thesis: **the number of free parameters scales with
the number of cell types, not with the number of synapses.** Here 443 genes grow
768 synapses, and the genome would still be 443 genes at 10,800 synapses. This
is the genomic bottleneck the thesis claims encourages modularity. Experiment 2
is the same brain, task and optimiser with the bottleneck removed: one gene per
synapse plus one bias per non-input neuron, 768 + 25 = 793 genes.

### Why this abstraction — the justification

Each design choice answers either to one of the thesis's hard constraints or to
a biological fact. ⚠️ The sources are the ones believed to support each point.
**None has been checked against the original in this repo, so verify each
before citing.** The earlier "cell types are a defensible abstraction"
paragraph (under *Assessment*) makes the same argument and carries the same
warning.

| design choice | justification | source to verify |
|---|---|---|
| genome specifies wiring *rules*, not synapses | An information budget. The genome has ~2×10⁴ genes (~3×10⁹ bp) against ~10¹⁴–10¹⁵ synapses. A per-synapse blueprint cannot fit, so innate circuitry must be specified in compressed form. | Zador 2019, *Nat. Commun.*, "A critique of pure learning…"; Koulakov, Shuvaev, Lachi & Zador 2022, "Encoding innate ability through a genomic bottleneck" |
| cell types are the unit the genome specifies | Neurons fall into genetically defined types, and whether two neurons connect depends strongly on their pre- and post-synaptic types | Zeng & Sanes 2017, *Nat. Rev. Neurosci.*; Kovács, Barabási et al. 2020, *PNAS* (genetic model of the *C. elegans* connectome) |
| one shared rule `w_ij = g(feat_i, feat_j)` | The indirect-encoding idea of HyperNEAT's CPPN, which computes a weight from two neurons' descriptors. Here the descriptor is a type identity rather than a coordinate, because the model has no space. | Stanley, D'Ambrosio & Gauci 2009, *Artif. Life* |
| no coordinates, no distance | Hard constraint. Wiring-length cost is the established alternative explanation for modularity, so it must not be available here. | Clune, Mouret & Lipson 2013, *Proc. R. Soc. B* |
| fixed neuron count, only connections evolve | Hard constraint (parsimony). It isolates wiring from neurogenesis. | [our choice] |
| grown once, frozen, then scored | We model innate circuitry at birth. The claim is that modularity *precedes* learning. | see *Biological importance* |
| synaptic **in**-budget with relative shrink | Homeostatic synaptic scaling: a neuron scales all its incoming synapses together to hold total drive near a set point. It is also this framework's analogue of Kashtan–Alon's per-neuron fan-in cap. | Turrigiano 2008, *Cell*, "The self-tuning neuron"; Kashtan & Alon 2005, *PNAS* |
| positional code on inputs and outputs only | A retina pixel or a motor output has a fixed identity: where it sits in the sensor array. Hidden neurons are interchangeable members of their type. | [our choice] |
| softplus abundance | A near-linear response means small mutations move type counts by ±1, so a starved type can recover (no extinction trap) | [our choice, measured] |

The price of the abstraction is stated under *Cost 2* below. Same-type neurons
are exact clones, so expressiveness is capped by `K`, not by the neuron count.

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

**Search.** CMA-ES over the flat genome (popsize 64, σ_init 0.1; evosax `CMA_ES`, **not** elitist — see *Replicating the study exactly*).
Balanced accuracy (chance = 0.5), bipolar inputs {−1, +1}.


### The FG vs MVG study — parameters as actually run

⚠️ The paragraphs above give the code **defaults**. The study reported in this
thesis used a larger brain and a sparsity mechanism that did not exist when they
were written. Use these numbers when writing up results.

| | value | note |
|---|---|---|
| neurons | 8 in / 24 hidden / 1 out (N = 33) | |
| allowed edges | **768** | IH 192 + HH 552 (no self-loops) + HO 24, of 33² = 1089 |
| cell types `K` | 8 | `type_dim` 4, `pos_dim` 4, feature dim 11 |
| rule `g` | 22 → 16 → 1 (depth 1) | 385 of the 443 genes |
| genome | **443 genes for 768 weights** | types 32 + in 4 + out 4 + abundance 8 + bias 10 + `g` 385 |
| `rnn_iters` | 8 | synchronous recurrent pass, inputs re-clamped |
| synaptic budget `S` | 6.0 | exp_2 (direct) uses 4.0 |
| shrink `τ` | 0.9 | |
| task | `retina_ka2005` | Kashtan–Alon's real Fig. 5a retina problem |
| reference goal | AND | MVG alternates AND ↔ OR every **20** generations |
| optimiser | evosax 0.1.6 `CMA_ES` (non-elitist, μ = 32), popsize 64, σ_init 0.1 | fitness `margin`, metric **raw** accuracy (`--no-balanced`) |
| generations | 10000 (exp_1), 5000 (exp_2) | 5 seeds per arm, 4 arms |
| archiving | champion genome **every** generation | `log.csv` every 10 gens / every epoch under MVG |

**Sparsity: the synaptic budget.** `g` is a smooth MLP, so an *absent* edge
requires `g` to output exactly zero — a measure-zero event. The encoding
therefore grows **fully connected** graphs by construction (measured: 100.0% of
allowed edges non-zero in every unconstrained arm, both encodings), which makes
every modularity metric undefined or trivially zero. Two mechanisms were tried:

- *Absolute gate* (`--w-threshold`): **measured to fail.** Evolution inflates `g`
  4–10× and walks through it — a 10× inflation takes a gated brain from 36 to
  744 edges.
- *Synaptic budget* (used here): every non-input neuron receives a fixed total
  incoming |weight| `S` to share among its synapses, and any synapse weaker than
  `τ ×` its target's own mean incoming |g| is zeroed **before** the share-out.
  Being *relative* is what makes it inescapable: nothing can inflate above its
  own mean, and multiplying `g`'s output by 100 leaves a budgeted brain
  bit-identical (max diff 3e−8).

Normalising over the **target** (in-budget, not out-budget) is deliberate on
three grounds: it is homeostatic synaptic scaling, the better-attested biology;
it bounds each neuron's pre-activation by `S`, keeping the recurrent pass inside
`tanh`'s useful range; and it asks the question the task cares about — it forces
each hidden neuron to choose **which inputs to listen to**. Under a budget `g`
loses its `tanh` output activation (redundant after renormalisation, and a
saturation attractor: 77% of synapses had been sitting at |w| > 0.999). ⚠️ A
genome saved with a budget must be reloaded with the same setting.

**Density — how it is computed, and a caveat that must be written up.** Density
is `edges / 768`, i.e. over the **architecturally allowed** edges, never over
N²; a structurally impossible edge is never counted as an absent one, in density
or in any null model. An edge is `|w| > 0.05` (`--prune-threshold`). That
threshold is **inherited convention, not a derived value** — it entered in the
first commit — so its influence was measured (seed 0, goal-matched champion,
each cell density | `lr_r`):

| arm | `t` = 0 (true) | `t` = 0.05 | `t` = 0.10 | `t` = 0.20 |
|---|---|---|---|---|
| exp_1 budget FG | 50.5% +0.067 | 45.8% +0.175 | 39.2% +0.449 | 31.6% +0.197 |
| exp_1 budget MVG | 33.1% −0.007 | 27.6% +0.004 | 27.3% −0.000 | 27.3% −0.000 |
| exp_1 no-budget FG | 100% −0.026 | 99.2% −0.015 | 97.4% +0.000 | 93.5% +0.013 |
| exp_1 no-budget MVG | 100% −0.026 | 88.2% −0.025 | 88.2% −0.025 | 88.2% −0.025 |

Three things follow. (1) In the unconstrained arms the **true** density is
exactly 1 — not one allowed edge is zero — so anything below 100% there is an
artefact of where the cut is placed, and the modularity question is unanswerable
rather than answered negatively. (2) The *null* result is threshold-robust:
every unconstrained `lr_r` sits at −0.02 ± 0.01 across the whole range. (3) A
*positive* `lr_r` in a budgeted arm is **not** threshold-stable (FG reads +0.067
/ +0.175 / +0.449 / +0.197 across the four cuts), so no specific positive value
should be quoted without its cut. Under a budget the zeros are structural — they
come from `τ`, not from the cut — so the defensible choice for budgeted arms is
to count exact zeros (`t = 0`) and reserve the threshold for the unconstrained
arms, where nothing is exactly zero.

**Per-generation logging.** `champions.npz` archives the champion every
generation, so champion accuracy and `lr_r` are per-generation with no re-run. A
*population* statistic cannot be recovered from it, and `log.csv` under MVG has
one row per goal epoch (10 points per 200-generation window), so
`--dense-log lo:hi` was ported from `kashtan_alon/train.py` and
`analysis/dense_replay.py` re-runs a seed with it. Measured, not assumed: two
replays are bit-identical to each other (max |Δgenome| = 0.0, with and without
`--dense-log`), but a replay is **not** bit-identical to the archived run — it
diverges from generation 0 (σ 0.097134 vs 0.097137) through float
reduction-order nondeterminism, amplified chaotically by CMA-ES. A replay is
therefore a *second sample of the same arm*, so replayed and archived rows are
never mixed in one panel.

### Replicating the study exactly

This is everything needed to re-run the 20 experiment-1 runs (and the 20
experiment-2 control runs) and re-derive every reported number. Wherever the
code does something a reader would not guess, it is stated here. Verified
against the code and the runs' own `config.json` on 2026-09-13.

**Software and hardware.**
- Python 3.10.20, JAX 0.6.2 on the CPU backend, evosax 0.1.6, Equinox 0.13.8,
  NumPy 2.2.6, on a Windows 11 laptop.
- 10 runs ran in parallel, each with `OMP_NUM_THREADS = MKL_NUM_THREADS = 2`.
  Under that load a run took 0.18–0.22 s per generation, so a 10,000-generation
  run is 30–36 min of wall-clock time.
- **Bit-exact reproduction is not expected**, even on this machine. See the
  replay note above. Replicate the distribution over seeds, not the trajectory.

**Code version.** Each run's `config.json` records it.
- Experiment 1: the budgeted arms ran at `040e952` and the unbudgeted arms at
  `71fbacd`, an ancestor of it. `model.py` and `shared_tasks.py` are identical
  between the two. `train.py` differs only in reporting (the per-generation
  archive, the goal-matched champion and `log.csv`), and the search loop is
  unchanged.
- Experiment 2: `040e952` (10 runs), `15a09f8` (9) and `71134c7` (1).

**Launch.** From `experiments/`, with `python` being the env interpreter called
directly, never `conda run`, which is not parallel-safe:

```
python run_fgmvg_study.py --experiment 1 --lanes 10
python run_fgmvg_study.py --experiment 2 --lanes 10
```

Each of the 20 runs is this call, made from `experiments/experiment_1/`:

```
python train.py --n-hidden 24 -K 8 --task retina_ka2005 --operation and --no-balanced \
    --fitness margin --no-early-stop --no-open --archive-interval 1 --resume \
    --generations 10000 \
    [--mvg --mvg-ops and,or --switch-interval 20]      # MVG arms
    [--synaptic-budget 6 --shrink 0.9]                 # budgeted arms
    --seed {0..4} --n-seeds 1 --out-dir runs/fgmvg/{budget,nobudget}_{fg,mvg}
```

Experiment 2 uses the same call from `experiments/experiment_2/` with no `-K`,
`--generations 5000` and `--synaptic-budget 4 --shrink 0.9`. Every other value
is a code default, recorded in `config.json` and listed in the table above.

**Task.** `retina_ka2005`.
- All 2⁸ = 256 patterns are enumerated (no sampling), with bits encoded bipolar
  {−1, +1}.
- Pixel layout: `0 2 | 4 6` over `1 3 | 5 7`.
- Left object: `L = [x0+x1+x2+x3 ≥ 3] ∨ [x2 = x3 = 0 ∧ (x0 ∨ x1)]`, whose outer
  column is x0, x1. The right object is the mirror image: outer column x6, x7,
  inner column x4, x5.
- Each object is true for 8 of its 16 half-patterns, so the four (L, R) cells
  hold 64 patterns each. The target is `L AND R` or `L OR R`.

**Genome → weights.** Let `S` = synaptic budget and `τ` = shrink.
- *Features* (dimension 11):
  - input `i`: `[e_in | PE(i) | 1 0 0]`
  - hidden neuron of type `t`: `[e_t | 0 0 0 0 | 0 1 0]`
  - output: `[e_out | PE(0) | 0 0 1]`

  Here `PE(p)_k = sin(p / 10000^(2⌊k/2⌋/4))` for even `k` and `cos(·)` for odd
  `k`, with `k = 0..3`, so the output's code is `[0 1 0 1]`.
- *Type assignment.* `p_t = softplus(a_t) / Σ softplus(a)`, with boundaries
  `B = 24 · cumsum(p)`. Hidden slot `s ∈ 0..23` gets type `#{t : s ≥ B_t}`,
  clipped to `K − 1`. This gives `U = 8 + 8 + 1 = 17` distinct signatures.
- *Rule.* `g` is an Equinox MLP 22 → 16 → 1 with a **ReLU** hidden layer (the
  Equinox default). Its output activation is the identity under a budget and
  `tanh` without one. The raw weight for source signature `u` and target
  signature `v` is `ĝ_uv = g([f_u, f_v])`.
- *Budget* (budgeted arm only), applied to the 17 × 17 signature matrix.
  - `m_uv` is how many neurons of signature `u` feed one neuron of signature
    `v`: 1 for an input, the clone count `c_t` for hidden type `t`, and `c_t − 1`
    on the hidden diagonal because there are no self-loops. It is 0 on
    disallowed pairs. Then:

    ```
    μ_v  = Σ_u m_uv |ĝ_uv| / Σ_u m_uv                    target v's mean incoming |g|
    g̃_uv = sign(ĝ_uv) · max(|ĝ_uv| − τ·μ_v, 0)          soft threshold
    w_uv = g̃_uv · S / Σ_u m_uv |g̃_uv|                   0 if the column is all zero; ε = 1e−8
    ```

  - The shrink is a **soft** threshold. Synapses below `τ·μ_v` go to exactly 0,
    **and** every survivor is shortened by `τ·μ_v` before the share-out. Summed
    over its real incoming synapses, every hidden and output neuron then has
    `Σ|w| = S` exactly.
- *Gather.* `W` (33 × 33) is `w` gathered through the neuron → signature map,
  times the role mask (IH, HH and HO allowed; no self-loops).
- *Bias.* `b_j` is the `type_bias` of neuron `j`'s type: index `K` for inputs,
  `K + 1` for the output.

**Inference.** Start from `a = 0 ∈ ℝ³³`. For 8 steps, clamp `a[0:8] ← x`, then
set `a_j ← tanh(Σ_i a_i W_ij + b_j)`, where row `i` is the source. The output is
`a_32` after step 8, and the decision is `a_32 > 0`. An input needs 2 steps to
reach the output, so 6 of the 8 steps are recurrence.

**Fitness and metric.**
- *Selection fitness* (what CMA-ES maximises). With `s = 2y − 1` and `out` the
  raw `tanh` output, `F = mean over 256 patterns of (min(s·out, 0.5) + 1) / 1.5`,
  which lies in [0, 1]. A pattern stops paying once its output is 0.5 onto the
  correct side, so selection pressure flows to the cases still wrong.
- *Reported metric.* **Raw** accuracy, the fraction of the 256 patterns with
  `(out > 0) = y`. The four (L, R) cells are equal in size, so raw accuracy is
  the mean over cells, and every predictor that reads only one half is capped at
  0.750. Balanced accuracy would hand that same one-half cheat 0.833, which is
  why it was **not** used.

**Search.**
- evosax 0.1.6 `CMA_ES` with popsize λ = 64 and σ₀ = 0.1. Everything else is
  the evosax default: μ = 32 parents (`elite_ratio` 0.5) with log-rank
  recombination weights, and initial covariance `C₀ = I`.
- Fitness goes through `FitnessShaper(maximize=True)`, with no rank or z-score
  shaping.
- ⚠️ **This CMA-ES is not elitist.** No individual survives into the next
  generation.
- ⚠️ **The search starts from the all-zero genome.** evosax draws the initial
  mean uniformly from `[init_min, init_max] = [0, 0]`. The random values in
  `Genome.init` (identities ~ 0.1·N(0, 1)) serve only as a shape template and
  never enter the search.
- RNG: `key = PRNGKey(seed)`, split once for the template and once for CMA-ES
  initialisation, then once per generation for `ask`.

**Goal schedule.** `goal(g) = [AND, OR][⌊g / 20⌋ mod 2]` for `g = 0 … 9999`, so
generations 0–19 are AND. A 10,000-generation MVG run therefore **ends inside an
OR epoch** (generations 9980–9999).

**Which network every number is measured on.**
- A generation's champion is the argmax of **raw accuracy on the active goal**
  over its 64 samples (first index on ties), not the argmax of the margin.
- The **goal-matched champion** is the last champion archived under AND:
  generation 9999 for FG and 9979 for MVG.
- Every end-of-run accuracy, density and `lr_r` is measured on the goal-matched
  champion.
- `champions.npz` holds every generation's champion genome and its accuracy on
  both goals.

**Modularity (`lr_r`).**
1. Keep `|W_ij| > 0.05` over the allowed edges only.
2. Symmetrise: `i–j` exists if either direction survives. Drop weights (the
   graph is unweighted) and remove the output neuron.
3. Pin inputs 0–3 to the left and inputs 4–7 to the right.
4. Seed each hidden neuron to the side with more pinned neighbours (ties go
   left). Then refine with up to 20 passes of greedy single-node moves that
   maximise Q.
5. `r = Q / (1 − Σ_g a_g²)`, where `a_g` is side `g`'s share of degree. Its
   meaning and its p-value (200 degree-preserving, mask-respecting rewirings)
   are under *Left/right modularity for MVG* below.

**Recovery statistic** (the facilitated-variation result).
- Data: the per-generation replays of the budgeted MVG arm, seeds 0–4 in each
  encoding.
- Statistic: `phase_stats` in `analysis/fig_switch_window.py`, applied to the
  **population mean accuracy on the active goal** (all 64 samples).
- Windows: `[100, 300]` and `[1000, 1200]`. Only full 20-generation epochs
  count, which gives 10 epochs per window.
- Per epoch:
  - trough = the value at the epoch's first generation
  - peak = the epoch's maximum
  - time = the first generation offset at which the value reaches
    `trough + 0.9·(peak − trough)`
- A seed's number is its mean time over the window's epochs. Seeds are then
  compared early vs late, and late − early is compared across encodings with a
  two-sided exact Mann–Whitney test.
- ⚠️ The per-seed aggregation and the test were run from a throwaway script, not
  from a file in the repo.

**Analysis pipeline.** From `experiments/analysis/`, with `<root>` =
`../experiment_1/runs/fgmvg`:

| output | command |
|---|---|
| per-seed tables, `metrics_summary.json` | `python run_all.py --root <root>` |
| brains grid | `python fig_brains.py --root <root> --grid --constraint {budget,nobudget}` |
| progress figure | `python fig_progress.py --root <root> --constraint {budget,nobudget}` |
| per-generation replay | `python dense_replay.py --root <root> --constraint {budget,nobudget} --seed N [--goal mvg]` (windows `100:300,1000:1200`; writes to the sibling `runs/fgmvg_dense/`) |
| windows figure | `python fig_switch_window.py --root <root> --constraint {budget,nobudget} --seed 0` |

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

> ## ⚠️ SUPERSEDED IN PART (2026-09-12) — the "did not solve the task" half is WRONG
>
> A 4-arm x 5-seed study at **10,000** generations (`experiments/experiment_1/RESULTS.md`,
> bottom section; runs in `experiment_1/runs/fgmvg/`) re-ran this design with
> **K=8**, **raw** accuracy and **S=6, τ=0.9**, and reaches **0.978 ± 0.022 on
> `retina_ka2005`/AND unconstrained, with seed 3 at exactly 1.000** — above
> Kashtan–Alon's own network (0.90 ± 0.03). So:
>
> * **"Best accuracy 0.885 of 1.000" and "neither solved the task" no longer hold.**
>   The encoding solves it. What was weak was this section's task/metric/K
>   combination (K=6, *balanced* accuracy, 2,000 generations), not the encoding.
> * **The ⚠️ "13× under budget" caveat below was the right instinct and is now
>   largely discharged.** 10,000 × 64 = 640k evaluations is ~2.6× under KA's 1.68M
>   median rather than 13×, and the extra budget is exactly where the accuracy
>   came from.
> * **The recommendation to abandon this architecture should be revisited before
>   it is acted on.** It rests on the accuracy claim above.
>
> **What SURVIVES, and is now stronger, is the FG-vs-MVG null.** At 5 seeds and 5×
> the generations the contrast is still not in MVG's favour: accuracy and purity
> both favour FG, Q and left/right significance favour MVG (2/5 seeds vs 0/5),
> two of four cuts each way. And the new study adds a mechanism for *why*: the
> AND-matched champion of an MVG run scores only **0.42–0.47 on OR**, and the
> per-generation trace shows AND and OR alternating in near-perfect antiphase
> across all 500 switches. MVG is re-specialising every epoch rather than building
> a shared decomposition — consistent with this section's "MVG's mechanism only
> engages once a network already computes both halves".
>
> Also superseded: the 3-seed table below is replaced by the 5-seed one in
> `experiment_1/RESULTS.md`. Note the two are not directly comparable — that table
> is *balanced* accuracy at K=6, the new one is *raw* at K=8.

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

## FG vs MVG × synaptic budget — results, both encodings (2026-09-12 → 14)

The Kashtan–Alon 4-group design (fixed vs modularly-varying goal × constrained vs
not), 5 seeds per arm, run on the compressed (exp 1) and direct (exp 2)
encodings. Parameters: *The FG vs MVG study — parameters as actually run*; every
command and definition: *Replicating the study exactly*. All numbers are measured
on the **goal-matched champion**, the last champion selected under AND. Lab
notebooks: `experiments/experiment_{1,2}/RESULTS.md`.

| encoding | constraint | arm | acc (AND) | density % | `lr_r` | Q | seeds beating null |
|---|---|---|---|---|---|---|---|
| compressed | budget | FG | 0.895 ± 0.017 | 43.0 ± 5.3 | +0.094 ± 0.068 | 0.169 ± 0.096 | 0/5 |
| compressed | budget | MVG | 0.853 ± 0.016 | 34.0 ± 7.6 | +0.199 ± 0.217 | 0.208 ± 0.154 | 2/5 |
| compressed | none | FG | 0.978 ± 0.022 | 93.7 ± 6.4 | −0.013 ± 0.007 | 0.048 ± 0.024 | 0/5 |
| compressed | none | MVG | 0.867 ± 0.042 | 96.9 ± 5.1 | −0.024 ± 0.005 | 0.010 ± 0.014 | 0/5 |
| direct | budget | FG | 1.000 | 37.8 ± 0.7 | +0.194 ± 0.031 | 0.221 ± 0.019 | 1/5 |
| direct | budget | MVG | 1.000 | 37.8 ± 0.6 | +0.210 ± 0.036 | 0.218 ± 0.013 | 3/5 |
| direct | none | FG | 1.000 | 95.6 ± 0.3 | −0.012 ± 0.006 | 0.070 ± 0.006 | 0/5 |
| direct | none | MVG | 1.000 | 97.3 ± 1.0 | −0.016 ± 0.006 | 0.085 ± 0.007 | 0/5 |

"Seeds beating null": `lr_r`'s degree-preserving permutation test at the planted
left/right split, p < 0.05. Density uses the 0.05 cut, so "none" rows below 100%
are an artefact of the cut: their true density is exactly 100%.

**What the study shows.**

1. **The compressed encoding can solve the task.** Unconstrained FG reaches
   0.978 ± 0.022, seed 3 exactly 1.000, above KA's own network (0.90 ± 0.03). The
   earlier "~0.885 ceiling" came from a weaker task/metric/K combination.
2. **The budget taxes compression, not the direct encoding.** It costs the
   compressed encoding ~8 accuracy points (0.978 → 0.895) and the direct encoding
   none (1.000 at ~38% density, inside KA's capped-arm band of 34–38%).
3. **The budget is what makes modularity askable.** `lr_r` flips sign with the
   constraint: all 20 unconstrained runs are negative, 18 of 20 constrained runs
   positive. Unconstrained brains are complete graphs, so there the question is
   undefined, not answered "unmodular".
4. **MVG does not increase modularity in either encoding.**
   - Compressed: `lr_r` +0.094 vs +0.199, not significant (exact one-sided
     Mann–Whitney p = 0.345), and confounded. The MVG arm is 9 points sparser,
     and `lr_r` correlates with density at r = −0.471 across the 10 constrained
     runs.
   - Direct: `lr_r` +0.194 vs +0.210, p = 0.274. Densities are matched
     (37.8 vs 37.8, density–`lr_r` r = −0.009) and the gap is half an SD.
   - The density-matched encoding is the one that shows no effect.
5. **MVG costs something in both encodings.**
   - Compressed: final accuracy drops (0.895 → 0.853 budgeted, 0.978 → 0.867
     unconstrained).
   - Direct: only time is lost. FG reaches 1.000 by generation ~400, MVG by
     ~1300, and the two are indistinguishable afterwards.
6. **The bottleneck buys no modularity.** At similar density the direct encoding
   is at least as modular on `lr_r` and Q, and far more consistent (`lr_r` SD
   0.03 vs 0.07–0.22). It also solves the task outright.
   Caveats that travel with this:
   - Evaluation budgets are not matched: 640k evaluations for compressed vs 320k
     for direct. The control gets the smaller budget and still wins on accuracy.
   - Competence regimes differ (at ceiling vs 10 points short).
   - The compressed arm's variances are large at n = 5, so read this as "no
     evidence it helps".
   - "Compression" and "fewer parameters" cannot be separated in this design.
7. **Accuracy on the other goal is not a test of MVG.** The network gets no goal
   cue, and AND and OR disagree on half the patterns, so a perfect AND solver
   necessarily scores 0.500 on OR. Low OR accuracy in an AND-matched champion
   marks a *good* AND solver, not a failure. An earlier "MVG never holds both
   goals" claim was withdrawn on this ground.
   - What does survive is the dynamics: the population re-specialises every
     epoch, in both constraint conditions and both encodings.
   - The valid test of KA's claim is re-adaptation speed, next section.

**Methodology lessons from this study** (each would have changed a conclusion):

- **Goal matching.** 10,000 / 20 is even, so every MVG run ends mid-OR, and the
  "final" champion is OR-selected. Measured on an 80-generation self-test, the
  final and the goal-matched champion differ 4× in density. In 3 unconstrained
  direct MVG seeds the best-ever champion reported "1.000", which turned out to
  be scored on OR by a network at 0.500 on AND.
- **KA's Q_m inverts the cross-encoding verdict.** Compressed vs direct Q_m is
  0.495 vs 0.046 (10.7×) at raw Q 0.133 vs 0.117 (1.14×). A degree-preserving
  null is not an encoding-preserving one: cell-type clones share identical rows,
  which clumps the degree sequence and lowers `q_rand`.
  - Q_m also saturates at exactly 1.000 for both the least and the most modular
    run, and is `nan` in 22 of 40 runs.
  - The fitting null is random genomes through the same encoding. Against it,
    raw Q survives in 9/10 constrained runs and `lr_r` in 6/10.
- **Purity is withdrawn as evidence.** It sits below that encoding-aware null in
  all 40 runs. On recurrent brains it tracks density and mixing, not
  modularity.
- **The `lr` ratio score is degenerate.** It is `nan` in 15 of 40 runs and
  returns exactly 1.000 for an anti-assortative graph. `lr_r` replaces it. With a
  ceiling of ~0.5, `lr_r` ≈ 2Q, so it adds a stable, named scale but no signal
  beyond raw Q.
- **Positive `lr_r` depends on the pruning cut** (see the threshold sweep
  above).

**Figures** (promoted, each folder with a provenance README):
- `latex_figures/experiment_1_fgmvg/`: brains grid, progress and switch
  windows, for both the budget and no-budget arms.
- `latex_figures/experiment_2_fgmvg/`: the same three figures, budget arm
  only. The unconstrained direct arm is 100% dense and 1.000 in every seed, so a
  table row covers it.

---

## Facilitated variation: the bottleneck buys re-adaptation speed ⭐

**The headline result of the FG/MVG study, and not the one it was built to
find.** Claim: *a compressed DNA→brain encoding makes a population re-adapt to a
returning goal faster as evolution proceeds; a direct encoding gets slower.* This is
facilitated variation measured directly — it routes through no modularity metric,
no null model and no pruning threshold, so it is immune to both the threshold
sensitivity and the `lr_r` resampling instability documented elsewhere.

**Measurement.** Budgeted MVG arms, 5 seeds per encoding. Goals alternate
AND ↔ OR every 20 generations, so a 200-generation window contains 10 goal
epochs. Per epoch: the population mean accuracy at the switch (trough), the peak
reached inside that epoch, and the generations needed to cover 90% of that climb
— measured against the epoch's **own** peak, because early in training the
population never reaches any fixed threshold and a fixed cut would censor every
early epoch at the epoch length. (Kashtan–Alon's reasoning, retained.) Each
seed's number is already a mean over 10 switches. Windows [100,300] and
[1000,1200], one order of magnitude apart, are KA's.

| | [100,300] | [1000,1200] | late − early |
|---|---|---|---|
| compressed (exp_1) | 6.30 ± 1.25 gens | **4.20 ± 0.24** | **−2.10 — 5/5 seeds faster** |
| direct (exp_2) | 5.90 ± 0.52 | **8.36 ± 1.14** | **+2.46 — 0/5 seeds faster** |

Per-seed deltas −2.4, −1.7, −2.1, −0.5, −3.8 against +3.6, +3.8, +2.0, +1.7,
+1.2. The two sets do not overlap; exact Mann–Whitney at n = 5 vs 5 returns the
smallest value the test can produce, **p ≈ 0.008**.

**The effect does not depend on how recovery is measured.** The statistic above
is relative to each epoch's own peak, and peaks move between windows (direct
0.84 → 0.97), so it was checked against three measures that use no peak
(population mean, 5 seeds, early → late window):

| measure | compressed | seeds faster | direct | seeds faster |
|---|---|---|---|---|
| gain in the first 3 gens after a switch | 0.196 → **0.299** | 5/5 | 0.237 → **0.152** | 0/5 |
| gens to climb +0.20 above the trough | 6.1 → **2.3** | 5/5 | 3.0 → **4.2** | 0/5 |
| gens to reach 0.75 | 7.3 → **3.6** | 5/5 | 4.4 → **5.0** | 0/5 |

Every measure agrees in every seed. The compressed encoding accelerates; the
direct encoding genuinely **decelerates**, modestly. This supersedes an earlier
note here that the direct slowdown was "partly a ceiling effect" and should not
be claimed. The slowdown is not CMA-ES step-size collapse: σ is unchanged or
slightly larger late (ratio 1.0–1.2 in all 5 seeds). ⚠️ These three measures
came from a scratch script, not a file in the repo.

**Champion vs population.** The champion curve agrees directionally but is noisy
(4/5 vs 1/5) because the champion is a max over the population and barely dents
at a switch. The population mean is the right statistic, and was KA's too.

**What the population carries across a switch.** AND and OR agree on the half
of the patterns where L = R and disagree on the half where L ≠ R. So a network's
accuracy on the new goal right after a switch is exactly `½ + ½(a − b)`, with
`a` and `b` its old-goal accuracy on the L = R and L ≠ R patterns respectively.
The trough therefore reads directly as where the population's old-goal
competence sits. Population troughs, early → late: direct 0.464 → **0.500**,
compressed 0.510 → **0.438**. Late in the run the direct population is **equally
good on both halves** (a = b), so nothing about the switch structure is
represented. The compressed population sits below 0.5, so it is better on the
L ≠ R patterns, **exactly the half that the switch flips** (b − a ≈ 0.12), and
yet it is the one that re-adapts faster. A population whose competence is
concentrated on the goal-discriminating patterns may have less to rebuild.
Hypothesis, not tested.

**How this sits with the rest of the study.** In the compressed budgeted arm:
modularity (`lr_r`) has **no support** (0/5 FG and 2/5 MVG seeds beat their own
degree-preserving null, Fisher p ≈ 0.44); accuracy is **worse** than the direct
encoding (0.895 vs 1.000); evolvability is **supported in 5/5 seeds**. So the
thesis's defensible claim becomes: *the genomic bottleneck buys facilitated
variation, at a cost in raw competence, and without producing measurable
left/right modularity.* Narrower than the original hypothesis. It also refutes
the natural objection that a shared `g` — one gene moving many synapses at once —
should make re-adaptation harder: the entangled encoding is the one that wins.

**In the direct encoding, MVG costs time, not outcome** (budget arm,
`latex_figures/experiment_2_fgmvg/`). FG reaches 1.000 by about generation 400
and MVG by about 1300. After that the arms are indistinguishable: all 10
champions score 1.000, density is ~38% in both, and `lr_r` is +0.19 (FG) against
+0.21 (MVG), with no visible left/right separation in the brains grid. In the
compressed encoding MVG also costs final accuracy (0.853 vs 0.895). **MVG never
helps, in either encoding.** (In the progress figure MVG's apparent head start
at the left edge is a sampling offset: its first point is generation 19, FG's is
generation 0.)

⚠️ **Caveats.** n = 5 per encoding, one task, one budget setting, two windows
inherited rather than chosen. The measurements come from **replays**: a replay is
a second sample of the arm, not a re-reading of the archived run, so these
numbers will not reconcile edge-for-edge with the end-of-run tables.
Replay-vs-replay is bit-exact, so they are regenerable.

### Why MVG produced no modularity here — the operator gap (open thread)

MVG shows **no** modularity effect in either encoding (compressed +0.094 FG →
+0.199 MVG; direct +0.194 → +0.210), so the absence cannot be blamed on the
compressed encoding. What both arms share is the **optimiser**. KA's
MVG→modularity result runs on a GA with per-gene mutation (Pm = 0.5) and
crossover (Pc = 0.5, elite 150 of 600): modularity is selected there because a
modular genome can swap one module in few mutations, and because crossover can
recombine intact modules between individuals. CMA-ES offers neither — it samples
from a single multivariate Gaussian, so there is no per-gene locality and no
recombination of parts — and at n = 443 its covariance adapts on a timescale far
longer than the 20-generation switch period, so the alternation may simply sit
below the resolution at which it can adapt.

**We ported KA's goal protocol but not KA's variation operators, while their
result is fundamentally a claim about how variation is generated.** Stated
positively, this is a finding rather than a null: *MVG's modularity effect
appears to require recombination, a variable KA never varied.* Testing it needs a
GA arm on the same task and encoding, and that is the single most valuable
follow-up the study points to.

**GA pilot on the cell-type model (2026-09-14) — mention in the write-up, even
though it is one seed.**
- *Design.* The optimiser was swapped for a Kashtan–Alon-style GA and nothing
  else changed (`experiments/experiment_1/ga.py`, `--strategy KA_GA`). The GA
  uses:
  - population 600, with the best 150 copied unchanged;
  - crossover P = 0.5, each gene block from one parent (block = a cell type, the
    input or output type, or one hidden unit of `g`);
  - mutation P = 0.5 of one gene, N(0, 0.5²), where the step size is our choice;
  - 3000 generations = 1.8M evaluations.

  The run was one MVG seed on the budget arm, with pass/fail thresholds
  preregistered in `experiment_1/RESULTS.md`.
- *Accuracy improved.* The goal-matched champion reached acc(AND) **0.938**. At
  CMA-ES's evaluation count (640k) it was already 0.891, above every CMA-ES MVG
  seed (0.853 ± 0.016). The population mean was also higher late in the run
  (peak 0.865 vs 0.806), so this is not only "the best of 600 draws".
  - Likely source: the GA keeps many parents plus crossover, so it searches many
    regions at once. CMA-ES shrinks search to one Gaussian.
  - Not yet controlled: CMA-ES at population 600, and a GA fixed-goal run.
- *Modularity did not appear.* `lr_r` was +0.04 to +0.05 and did not beat its
  null at either cut. The brain was denser (57% vs ~30%). `lr_r` peaked at
  +0.14 near generation 600, then faded as accuracy rose.
- *Recovery.* No speed-up across windows (8.0 → 8.3 gens), unlike CMA-ES on this
  encoding (descriptive only).
- *How to write it up.* Changing the optimiser to KA's operators improved
  competence but did not recover modularity in the cell-type model. This is
  expected if the genome has no locus where a left or right detector can live:
  crossing whole cell types or `g` units cannot move a module. It therefore
  narrows the operator-gap explanation rather than refuting it.
- *Missing tests.* The decisive test is the same GA on the direct encoding,
  where a neuron's incoming column *is* KA's crossover unit. A GA fixed-goal arm
  is also needed before any MVG-vs-FG statement. ⚠️ n = 1, and the mutation step
  was not tuned.

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

## Experiment 4 (CGP): why MVG does poorly where it helped Kashtan–Alon

**The observation.** KA's retina, raw accuracy, 4-gate set `and,nand,or,nor`.
FG solves every seed (1.000) at every genotype size. MVG (and↔or, E=2000) never
holds a solution: at 50 nodes the last AND-epoch champion scores **0.81–0.85**
(0/20 seeds ever perfect); the best arm (400 nodes) reaches 256/256 in 7/20 seeds
but **≤1% of AND epochs end perfect**. Constant output and any one-sided circuit
both cap at 0.750, so MVG circuits do use both halves — just not correctly. In
KA's own GA, MVG *beats* FG (0.975 vs 0.904).

**Not the explanation: the switch itself.** A modular circuit (L detector + R
detector + combiner) needs one gate change for AND→OR, and both gates are in the
set: ~200 generations to propose it at 50 nodes, well inside a 2000-generation
epoch. MVG never *finds* the modular circuit; failure is upstream of the switch.
Also already ruled out: E (evaluation-matched to KA), genotype size, mutation
step size (`experiment_4/RESULTS.md`).

**Candidate explanations** (hypotheses, strongest first):

1. **No population → nothing for MVG to select between.** KA's mechanism is
   selection *among variants*: modular nets recover faster after each switch and
   take over; crossover recombines modules. (1+4) CGP is a single lineage with no
   memory beyond one parent, so "recovers faster" is never rewarded — there is no
   slower rival to beat. Consistent with: per-seed recovery shows no trend
   (Spearman ρ ≈ 0 in every arm).
2. **Switches too slow relative to adaptation.** KA flips every 20 generations, too
   fast for a population to re-specialise, so selection acts on the *time-averaged*
   fitness, which rewards both-goal (modular) solutions. We matched *evaluations*
   per epoch, but the lineage re-adapts in ~330 of 2000 generations: each epoch is
   FG on alternating tasks. The relevant quantity may be E / recovery time, not
   evaluations. Untested below recovery time (E=200 still re-specialised in ~70).
3. **Switching disrupts CGP's neutral drift.** CGP finds solutions by long neutral
   walks (one seed flat at 234/256 for 18,000 generations before solving; FG median
   23–37k generations = 12–18 MVG epochs). A switch exposes hidden variation to
   selection on the other goal before it pays off. Consistent with: more nodes help
   MVG (0.84 → 0.88, 50 → 400) but not FG; pre-switch accuracy flat all run.
4. **Modules are expensive in 2-input Boolean gates.** A KA threshold unit computes
   "≥3 of 4 black" in one neuron; with AND/OR/NAND/NOR the object detector is a
   multi-gate subcircuit (smallest full solution found: 16 gates). Long assembly
   time, interrupted every 2000 generations — feeds (3).

**The 5 FG vs 5 MVG study (50 nodes, 800k generations, `runs/fgmvg50`, 2026-09-14).**
End state (MVG = last AND-epoch champion): FG acc 1.000 in 5/5, purity 0.82–0.90,
15–24 gates; MVG acc 0.81–0.85, purity 0.25–0.81, 7–18 gates. Every FG seed is
purer than every MVG seed (Mann–Whitney U=0, p≈0.008) — **but this is a solving
effect, not an MVG effect:**

- **At matched accuracy the arms are identical.** All sparse-archive champions
  with acc(AND) 0.81–0.85: FG (pre-solve) purity median 0.60, 7 gates (n=205);
  MVG 0.62, 7 gates (n=791). A full solution essentially forces two pure
  half-detectors (→ high purity); the cheap 0.84 approximations mix L and R early.
- **MVG circuits are small because they never solve**, not because switching
  prunes: a 0.84 circuit needs ~7 gates in either arm, while FG carries 15–20 gates
  just before solving. Across a switch MVG's size is erratic rather than shrinking
  (median 8.5 → 7 → 8; single switches 17→2 and 2→21) — re-adaptation is often
  a rewiring near the output that disconnects or reconnects a large subcircuit.
- **Recovery does not speed up** (MVG seed 0: 18/666 gens early, 1269/1710 late).
- **Stepping stones outlive the epoch.** FG needs 12k–115k generations to solve
  under a *fixed* goal; MVG discards the partial structure every 2,000.

So with a single-lineage (1+4) ES, varying goals neither builds modularity nor
removes it at equal accuracy. This is a valid null for CGP-as-published, but **not
a test of KA's mechanism**, which needs a population whose variants differ in
recovery speed (explanation 1).

**Cheap tests** (CGP, 50 nodes, ~1–2 min/seed): short epochs E=20/50 (tests 2;
existing flag); seed MVG from a solved FG circuit (separates "can't find" from
"can't hold"; needs `--init-from`); a real population with tournament selection
(tests 1; new code, departs from the CGP paper's Table II).

### Roadmap: from standard CGP to KA's population GA (2026-09-14)

**Step 1 — standard CGP, (1+4) ES** (`train.py`, CGP paper's Table II; `runs/fgmvg50`).
5 FG + 5 MVG, 50 nodes, E=2000, 800k generations. FG solves (1.000, purity
0.82–0.90); MVG plateaus at 0.84 for the whole run (purity 0.25–0.81, 0/200 AND
epochs perfect, recovery 250–420 gens). At matched accuracy the arms have equal
purity → a null, attributed to the single lineage (explanation 1 above).

**Step 2 — same circuits, KA's GA** (`train_pop.py`, mirrors `kashtan_alon/ga.py`;
`runs/fgmvg50_pop`). Only the search loop changes: population 600, top 150 copied
unchanged, 450 children of two random elite parents, crossover p=0.5 (per node:
each node's function + inputs from parent A or B — our CGP translation of KA's
per-neuron crossover), mutation p=0.5 (CGP's own 3% point mutation). MVG, E=2000,
100k generations, 5 seeds: **acc(AND) 1.000 in 4/5 (seed 0: 0.906, solved then
collapsed at ~60k), purity 1.00 in 5/5, median 16 gates; recovery after a switch
2–4 generations**; perfect AND epochs 22/20/8/1/12 of 25. The circuits are
textbook-modular: a pure left detector and a pure right detector joined only at
the output gate, so AND↔OR is a one-gate change.

**Not yet attributable to MVG.** (a) FG was run with the (1+4) ES only — the GA
alone might yield purity 1.00; (b) the GA used ~34M evaluations/seed vs 3.2M for
(1+4) MVG (though (1+4) MVG was flat at 0.84 for all 800k generations, so budget
alone is an unlikely explanation for the accuracy gap).

**Next:** FG with the GA (the missing control, ~19 min); then E (KA's 20 vs 2000)
and pc=0 (does crossover matter?).

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
