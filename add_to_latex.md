# Content for LaTeX writeup

Scratch notes destined for the thesis. Not polished prose — just the facts/claims
to work into the final writeup, with enough detail to write them up correctly later.

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

---

## Modularity metrics (`qmetrics/`)

Shared package at repo root: adapters turn any brain format (NDP's `W`,
exp 1–3's `w`, `kashtan_alon`'s layer blocks) into a graph; metrics see only the
graph. **Two metrics so far.**

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
