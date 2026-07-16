# Experiment 2 — direct encoding (the control)

The Kashtan-Alon-style baseline that experiment 1's compressed encoding is
measured against. Same tasks, same CMA-ES loop, same metrics — the **only**
difference is how the network is encoded.

|  | experiment 1 (`../experiment_1`) | experiment 2 (here) |
|---|---|---|
| genome | ~O(K): cell-type identities + shared rule `g` | the raw weight vector (~E free weights) |
| weight of edge i→j | `g(feat_i, feat_j)` (shared across edges) | its own independent parameter |
| search space | low-dim manifold of *regular* networks (structural bias toward modularity) | the *whole* space of networks the topology allows (no bias) |
| role | the treatment (bottleneck hypothesis) | the control / null model |

**Why both.** The thesis — "a compressed DNA→brain encoding encourages
modularity" — is only meaningful relative to a no-compression baseline. If
modularity emerges *here* too (from the selection pressure alone), the encoding
isn't doing the work; if it emerges here only under strong pressure (MVG /
connection cost) but in experiment 1 under weaker pressure (e.g. curriculum),
that difference is the contribution.

## Files
The directory deliberately mirrors `../experiment_1` — same file names, same
flags (via `config.py`), same outputs — minus the encoding-specific pieces
(`oracle.py`, `reachability.py`, and the `-K`/`--type-dim`/`--g-*` knobs), which
have no meaning for a direct encoding.
- `model.py` — `DirectGenome`: genome = weights + biases; fixed topology
  (`[inputs, hidden, outputs]`, edges IH|HH|HO, no self-loops), synchronous
  recurrent inference — dynamics identical to experiment 1.
- `config.py` — CLI → `BrainConfig` + `RunConfig`. Trimmed mirror of experiment
  1's config (no encoding knobs; drops exp 1's dead `--eval-reps`/`--test-reps`/
  `--n-examples`/`--no-elitism`). Shared defaults are kept **identical** so the
  two experiments are comparable out of the box (e.g. `--n-hidden 20`).
- `tasks.py` — thin shim that re-exports `../shared_tasks.py` (the single shared
  task definitions, imported identically by every experiment).
- `visualize.py` — brain image + `brain_stats`, adapted from experiment 1 (hidden
  neurons in one neutral colour — direct encoding has no cell-types).
- `train.py` — CMA-ES loop; balanced accuracy, optional `--fitness margin`,
  fixed-goal or `--mvg` (AND↔OR switching), `--viz-interval` live rendering.
  Mirrors experiment 1.

## Run
```
# fixed-goal retina/xor, 5 seeds  (defaults match exp 1: n_hidden=20)
python train.py --task retina --operation xor --n-seeds 5 --fitness margin

# modularly-varying goal (the known positive control for emergent modularity)
python train.py --task retina --mvg --switch-interval 20 --generations 2000
```

## Not built yet (next steps)
- A modularity metric (Newman `Q` / task-aware left-right block score) — the
  prerequisite for actually *measuring* modularity in either experiment. Belongs
  in `../shared_tasks.py`'s sibling (a shared analysis module) so both encodings
  score identically; `visualize.py` can then recolour hidden nodes by community.
- Curriculum runner (port from `../experiment_1/curriculum.py`).
