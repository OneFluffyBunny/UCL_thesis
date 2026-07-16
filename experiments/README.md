# Experiments

Minimal models for studying the **emergence of modularity** in evolved neural networks.

Core question: what makes a brain modular? Working hypothesis: modularity is **selected for** (not learned) because a modular DNA→brain encoding is more *evolvable* — when the task changes slightly, it re-adapts in few generations (facilitated variation).

Guiding constraints (carried across experiments):
- **No physical space** for neurons — modularity must come from something other than wiring-length cost.
- **Fixed number of neurons**; only connections are determined by the DNA.
- **Parsimony first** — as few rules/knobs as possible.

---

## Experiment 1

Reuses much of the NDP/LNDP implementation, stripped down.

- **Brain type:** *static* (NDP-style) — the DNA builds the brain, then it is frozen while solving the task. No within-lifetime plasticity. Adaptation happens across generations (evolution), not within a life.
- **Neurons:** fixed count, three types — **input**, **hidden**, **output**. No physical space; the brain is a pure graph.
- **Connections:** only **IH** (input→hidden), **HH** (hidden→hidden), and **HO** (hidden→output) are allowed.
- **Output:** a **single output neuron**, thresholded to a binary decision (squash with `tanh`, threshold at 0). The multi-output + `argmax` head from NDP/LNDP may be revisited later.
- **Activation:** `σ`, defaulting to `tanh`. Other activations may be tried later.
- **Inference:** treat the brain as a directed graph and reuse LNDP's mechanics — a synchronous recurrent pass over the weighted adjacency matrix (see below), run for a fixed number of iterations.
- **Tasks:** logical-gate problems in the style of Kashtan–Alon (modularly-varying goals).
- **Search:** **CMA-ES** over the genome for now. Other evolutionary approaches (genetic algorithms, NES, MAP-Elites / quality-diversity, …) may be used later.

### DNA → brain encoding (cell-type identities)

The DNA does **not** specify connection weights directly, and its size is **independent of the number of neurons** (`O(K)`). It stores:

- a small set of evolved **cell-type identity vectors** — `K` for hidden neurons, plus 1 shared identity for inputs and 1 for outputs;
- per-type **abundance** genes deciding how many hidden neurons are of each type (these evolve; total hidden count stays fixed; a type can grow, shrink, or go extinct);
- one shared connection rule `g` (a small MLP) mapping a pair of neuron features to a weight.

Each neuron's feature vector is `[type identity | positional code | role one-hot]`:

- **Type identity** is evolved and *shared* within a type — this is the compression / modularity bottleneck. Few shared types ⇒ block-structured wiring.
- **Positional code** is fixed (sinusoidal, zero evolved params) and given to **input/output neurons only**; hidden neurons are type-only, so same-type hidden neurons are interchangeable (keeps the encoding compressed; effective hidden capacity ≈ K).
- The weight is `w_ij = g(feat_i, feat_j)`, with `g` **asymmetric** in its arguments ⇒ a **directed** graph. Output bounded to `[-1, 1]` (`tanh`). `g` is **deterministic** given the genome (no developmental noise deciding function — a lesson from past LNDP failures).

**Abundance → counts:** `softplus(abundance)` normalised gives the per-type fraction; cumulative boundaries bucket each hidden slot to a type. softplus (not softmax) keeps the response near-linear so counts mutate gradually (±1) and starved types can recover — no exponential extinction trap. Starts as an equal split.

**Efficiency:** `g` is evaluated once per *distinct feature signature pair*, not per neuron pair. Hidden neurons of the same type share one signature, so the brain is built from `U = n_in + K + n_out` distinct signatures (`U²` evaluations of `g`) and the full `N×N` weight matrix is produced by gathering — no extra `g` calls.

This encoding can express both modular and non-modular brains, so the modularity result is not rigged.

> **Other encodings to try later:** direct weight encoding (control arm), genomic bottleneck (Zador/Koulakov), generative grammar / L-system, developmental GRN (Kouvaris 2017). See project notes.

*(More details to be filled in.)*

---

## Future ideas / levers to try

- **Evolve `rnn_iters` as a gene.** Currently the number of synchronous recurrent passes is a fixed hyperparameter (default 8), hand-set and unrelated to the actual graph. Make it a per-genome gene so each DNA evolves its own "settling time" / thinking window. Alternative principled options: run the recurrent pass **to a fixed point** (`‖aₜ₊₁ − aₜ‖ < ε`) instead of a fixed step count, or scale iterations to the realised graph depth. Caveat: variable-length loops are awkward to `jit`/`vmap`, and fixed-point iteration isn't guaranteed to converge (can oscillate).
- **Alternative encodings** (see callout above): direct weight encoding as a control arm, genomic bottleneck, L-system, developmental GRN.
- **Other search methods:** genetic algorithms, NES, MAP-Elites / quality-diversity.
- **Adam as a diagnostic baseline** (not in the fitness loop): gradient "oracle" to decompose why CMA-ES fails — brain-ceiling / encoding-ceiling / evolvability ladder. Caveat: the `>0` output is non-differentiable, so it needs a soft surrogate loss.
- **Real sparsity** (vs cosmetic prune-threshold): L1/L0 penalty on weights or hard-thresholding inside `forward`, so a modular brain genuinely has few cross-module edges.
- **Sensory adaptation:** input clamp is currently exact and noiseless; real receptors attenuate sustained input.
