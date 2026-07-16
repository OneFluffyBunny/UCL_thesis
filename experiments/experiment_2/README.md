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
- `model.py` — `DirectGenome`: genome = weights + biases; fixed topology
  (`[inputs, hidden, outputs]`, edges IH|HH|HO, no self-loops), synchronous
  recurrent inference — dynamics identical to experiment 1.
- `tasks.py` — thin shim that re-exports `../shared_tasks.py` (the single shared
  task definitions, imported identically by every experiment).
- `train.py` — CMA-ES loop; balanced accuracy, optional `--fitness margin`,
  fixed-goal or `--mvg` (AND↔OR switching). Mirrors experiment 1.

## Run
```
# fixed-goal retina/xor, 5 seeds
python train.py --task retina --operation xor --n-hidden 10 --n-seeds 5 --fitness margin

# modularly-varying goal (the known positive control for emergent modularity)
python train.py --task retina --mvg --switch-interval 20 --n-hidden 10 --generations 2000
```

## Not built yet (next steps)
- A modularity metric (Newman `Q` / task-aware left-right block score) — the
  prerequisite for actually *measuring* modularity in either experiment.
- Curriculum runner (port from `../experiment_1/curriculum.py`).
- Brain visualiser (adapt `../experiment_1/visualize.py`).
