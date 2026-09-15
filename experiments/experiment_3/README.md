# Experiment 3 — direct encoding + gradient descent (the optimiser control)

Experiment 2's network (`../shared_direct_model.py`) trained with gradient descent
instead of CMA-ES. Experiments 1 vs 2 vary the encoding; 2 vs 3 vary the optimiser.
The comparison asks how much having the exact gradient helps, and gives a strong
local optimiser as a reference for "can this topology fit the task at all".

## Differentiability

- The forward pass (`tanh(a @ w + b)`, scanned) is smooth in the weights.
- Accuracy (`out > 0`) has zero gradient, so it is only the reported metric.
- `--loss margin` (default) is the hinged signed-margin surrogate that experiment 2's
  CMA-ES maximises, so the comparison changes only the optimiser.
- `--loss bce` is a balanced logistic loss on `(tanh_out + 1) / 2`: gradient descent's
  best case, with a different objective.

## Files

`model.py`, `visualize.py` and `tasks.py` re-export the shared modules experiment 2
uses. `config.py` replaces experiment 2's evolution flags with `--optimizer`
(adam/sgd), `--lr`, `--steps`, `--loss` and `--grad-clip`. `train.py` is the Optax
loop, full batch over all `2**n_in` patterns, deterministic.

## Run

```bash
python train.py --task retina --operation xor --loss margin --n-seeds 5 --steps 5000
python train.py --task retina --operation xor --loss bce --n-seeds 5 --steps 5000
```

One gradient step evaluates the network once on every pattern; one CMA-ES generation
evaluates it `popsize` (64) times. Compare budgets in evaluations, not steps vs
generations.
