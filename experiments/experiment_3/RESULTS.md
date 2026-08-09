# Experiment 3 — results log

Direct encoding (`../shared_direct_model.py`), retina/xor, n_hidden=20, 5 seeds,
Adam (lr=1e-2). Compare against `../experiment_2/RESULTS.md` (same task/net,
CMA-ES): 5/5 seeds solved in 290–456 generations = 18,560–29,184 evals
(popsize=64), density ~92–94%.

## margin loss (fair, same objective CMA-ES maximises)
```
python train.py --task retina --operation xor --loss margin --n-seeds 5 --steps 5000 --no-open
```
| seed | steps to solve | acc | density |
|---|---|---|---|
| 0 | 1680 | 1.000 | 89.3% |
| 1 | 3984 | 1.000 | 90.9% |
| 2 | 1635 | 1.000 | 89.5% |
| 3 | 2181 | 1.000 | 90.7% |
| 4 | 200  | 1.000 | 87.5% |

## bce loss (gradient-oracle bound)
```
python train.py --task retina --operation xor --loss bce --n-seeds 5 --steps 5000 --no-open
```
| seed | steps to solve | acc | density |
|---|---|---|---|
| 0 | 143 | 1.000 | 82.9% |
| 1 | 88  | 1.000 | 87.9% |
| 2 | 72  | 1.000 | 83.9% |
| 3 | 204 | 1.000 | 85.9% |
| 4 | 89  | 1.000 | 82.1% |

## Notes
- 5/5 seeds solved under both losses — as expected, GD beats CMA-ES on
  evals-to-solve by a wide margin: ~7–150x fewer evals under margin (same
  objective), ~100–400x fewer under bce. Exact gradients help a lot on a
  network this small.
- Density is a bit lower than CMA-ES's ~92–94% (margin ~87.5–90.9%, bce
  ~82.1–87.9%) but still dense/unstructured — GD isn't finding anything more
  modular, just finding a dense solution faster. No modularity metric yet to
  say more than that.
