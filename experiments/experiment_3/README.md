# Experiment 3 — direct encoding + gradient descent (the optimiser control)

Where experiment 2 evolves the direct-encoding weight vector with **CMA-ES**,
experiment 3 trains the **same network** with **gradient descent**. Experiments
1↔2 vary the *encoding*; experiments 2↔3 vary the *optimiser* — the same model
(`../shared_direct_model.py`) under both, so the comparison is clean.

|  | search style | what it uses |
|---|---|---|
| experiment 2 (CMA-ES) | population sampling; move the distribution toward fitter members | fitness values only (black-box; no gradients) |
| experiment 3 (GD) | compute d(loss)/d(weights) and step straight downhill | full gradient of a differentiable loss |

**Why both.** CMA-ES treats fitness as a black box and explores semi-randomly;
gradient descent exploits the exact slope. Comparing them on the *same*
direct-encoded network isolates "how much does having the gradient help here?" —
and gives a strong-local-optimiser reference point for the representability-vs-
reachability question (if even GD can't fit a task the topology allows, that
points at a genuine limit, not just weak search).

## Differentiability (the one thing that makes this possible)
- **forward pass** — `scan` of `tanh(a @ w + b)`: smooth in the weights ⇒ backprops fine.
- **accuracy** — the `out > 0` threshold has zero gradient ⇒ can never be the loss;
  it stays the *reported* metric only.
- **loss** (`--loss`) — a differentiable stand-in:
  - `margin` (default): the negative of the exact hinged signed-margin surrogate
    exp 2's CMA-ES *maximises* → **fair optimiser-only comparison** (same objective).
  - `bce`: balanced logistic loss on `p = (tanh_out+1)/2` → **gradient-oracle**
    upper bound (GD's best shot; objective differs from exp 2, so not head-to-head).

## Files
Mirrors experiment 2; only `config.py` (optimisation group) and `train.py` (the
loop) differ. `model.py`, `visualize.py`, `tasks.py` are thin shims onto the
shared modules experiment 2 also uses — so the two experiments cannot diverge.
- `model.py` → `../shared_direct_model.py` (`DirectGenome`).
- `visualize.py` → `../shared_direct_viz.py`.
- `tasks.py` → `../shared_tasks.py`.
- `config.py` — CLI → `BrainConfig` + `RunConfig`; the `optimisation` group
  (`--optimizer` adam/sgd, `--lr`, `--steps`, `--loss`, `--grad-clip`) replaces
  exp 2's `evolution` group. Architecture / task / analysis groups are identical.
- `train.py` — Optax loop: `eqx.filter_value_and_grad` on the loss, `optim.update`,
  `eqx.apply_updates`. Full-batch (all `2**n_in` inputs), deterministic.

## Run
```
# fair head-to-head with exp 2 (same objective it maximises), 5 seeds
python train.py --task retina --operation xor --loss margin --n-seeds 5

# gradient-oracle bound: GD's best shot with a proper logistic loss
python train.py --task retina --operation xor --loss bce --n-seeds 5
```

## Comparing to experiment 2 — mind the x-axis
One GD **step** = one forward+backward over the full input set (1 function
evaluation). One CMA-ES **generation** = `popsize` (default 64) forward evals.
So step-count and generation-count are **not** the same unit; the fair
cross-experiment budget axis is **total function evaluations**. Wall-clock also
differs (GD steps here are ~0.001 s once compiled).

## Not built yet / open threads
- A modularity metric (Newman `Q` / left–right block score) — shared with exp 1/2;
  needed to ask whether GD's dense solutions are any more/less modular than
  CMA-ES's. Belongs next to `../shared_tasks.py`.
- Whether GD gets *stuck* where CMA-ES eventually solves (local minima), or
  solves faster — the actual experiment; run it in a per-experiment chat.
- `--optimizer sgd` and learning-rate sweeps left for the run phase.
