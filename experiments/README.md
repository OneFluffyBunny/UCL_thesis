# Experiments

Six models, each in its own folder with a `RESULTS.md` notebook. Experiments 1-3
vary one thing at a time on the same recurrent network: 1 vs 2 changes the
**encoding**, 2 vs 3 changes the **optimiser**. Experiments 4-6 move to Boolean
circuits, where modules are explicit, countable objects.

| | model | search |
|---|---|---|
| 1 | recurrent tanh network, compressed cell-type encoding (below) | CMA-ES; a Kashtan-Alon-style GA as a pilot |
| 2 | the same network, one gene per synapse (`shared_direct_model.py`) | CMA-ES |
| 3 | the same network as 2 | gradient descent (Optax), `README.md` there |
| 4 | CGP and ECGP Boolean circuits | (1+4) ES, and a population GA (`train_pop.py`) |
| 5 | fork of 4 for many outputs, PyPy-capable | (1+4) ES |
| 6 | self-modifying CGP; nested ECGP (`necgp/`, `necgp_pairwise/`) | (1+4) ES |

## Shared code

| file | used by | what it is |
|---|---|---|
| `shared_tasks.py` | 1, 2, 3 | the Boolean tasks; `retina_ka2005` is Kashtan & Alon's real object rule, `retina` an older stand-in (see its docstring) |
| `shared_direct_model.py`, `shared_direct_viz.py` | 2, 3 | the direct-encoding network, its synaptic budget, and its drawing |
| `shared_brain_metrics.py` | 1, 2, NDP | modularity scores for a recurrent (N, N) weight matrix (wraps `qmetrics/`) |
| `run_fgmvg_study.py` | 1, 2 | launches the 4 arms x 5 seeds FG-vs-MVG study, resumable |
| `analysis/` | 1, 2 | tables and figures for that study (`run_all.py` regenerates all of them) |

Commands run from inside the experiment folder unless a notebook says otherwise.

## Experiment 1 — the compressed encoding

**Network.** A fixed number of input, hidden and output neurons with no positions.
Allowed connections are input→hidden, hidden→hidden (no self-loops) and
hidden→output. Inference is a synchronous recurrent pass run for `--rnn-iters`
steps (default 8) with the inputs re-clamped each step:
`a_j ← tanh(Σ_i a_i W_ij + b_j)`. The decision is the sign of the output neuron.

**Genome.** The genome does not store weights, and its size does not depend on the
number of neurons. It holds:

- `K` evolved **cell-type identity vectors** for hidden neurons, plus one for inputs
  and one for outputs;
- per-type **abundance** genes: `softplus(a)` normalised gives each type's share of
  the hidden neurons (the total stays fixed);
- one per-type bias;
- one shared **connection rule** `g`, a small MLP.

Each neuron's feature is `[type identity | positional code | role one-hot]`. Inputs
and outputs get a fixed sinusoidal positional code; hidden neurons get none, so all
hidden neurons of one type are identical. The weight of edge i→j is
`w_ij = g(feat_i, feat_j)` (directed, deterministic). `g` is evaluated once per pair
of distinct feature signatures (`K + n_in + n_out` of them), and the full matrix is
gathered from that.

Because same-type hidden neurons are exact clones, a network with `K` types behaves
like a `K`-unit recurrent network whose edges are scaled by the clone counts:
abundance acts as a gain, and `K` sets how many distinct roles exist.

**Synaptic budget** (`--synaptic-budget S --shrink τ`). Without it `g` never outputs
an exact zero, so every allowed edge exists. With it, each non-input neuron's incoming
|weights| are soft-thresholded at `τ ×` their own mean and rescaled to sum to `S`.
Experiment 2 has the same mechanism. Exact equations: `experiment_1/model.py`.

Files in `experiment_1/`: `model.py` (genome → weights), `train.py` (CMA-ES or
`--strategy KA_GA`), `ga.py` (the GA), `config.py` (flags), `oracle.py` and
`reachability.py` (hand-built solutions and basin probes), `curriculum.py`
(curriculum vs cold start), `visualize.py`, `visualize_ckpt.py`, `smoke_test.py`,
`test_ga.py`.
