# Experiment 3 — results

Experiment 2's direct-encoding network (`../shared_direct_model.py`) trained by
gradient descent instead of CMA-ES: stand-in `retina/xor`, 20 hidden neurons, Adam
(lr = 1e-2), full batch, 5 seeds. Accuracy is balanced; density counts |w| > 0.05.

Reference point, experiment 2 on the same task and network: 5/5 seeds solved in
290–456 generations = 18,560–29,184 evaluations (popsize 64), density 92–94%.

## `--loss margin` (the objective CMA-ES maximises)

```bash
python train.py --task retina --operation xor --loss margin --n-seeds 5 --steps 5000 --no-open
```

| seed | steps to solve | acc | density |
|---|---|---|---|
| 0 | 1680 | 1.000 | 89.3% |
| 1 | 3984 | 1.000 | 90.9% |
| 2 | 1635 | 1.000 | 89.5% |
| 3 | 2181 | 1.000 | 90.7% |
| 4 | 200 | 1.000 | 87.5% |

## `--loss bce` (logistic loss, gradient descent's best case)

```bash
python train.py --task retina --operation xor --loss bce --n-seeds 5 --steps 5000 --no-open
```

| seed | steps to solve | acc | density |
|---|---|---|---|
| 0 | 143 | 1.000 | 82.9% |
| 1 | 88 | 1.000 | 87.9% |
| 2 | 72 | 1.000 | 83.9% |
| 3 | 204 | 1.000 | 85.9% |
| 4 | 89 | 1.000 | 82.1% |

## Reading

One gradient step is one evaluation of all 256 patterns, so steps compare directly
with CMA-ES evaluations. Gradient descent needs ~7–150× fewer evaluations than CMA-ES
with the same objective, and ~100–400× fewer with BCE. Its solutions are slightly
sparser (82–91% vs 92–94%) but still near fully connected: the gradient finds a dense
solution faster, not a more structured one.
