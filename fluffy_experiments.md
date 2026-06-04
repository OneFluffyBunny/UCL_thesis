# Fluffy Branch — LunarLander Experiments Log

All runs use `LunarLander-v3`, the `fluffy_redesign` branch (I/O-anchor redesign),
16 physical cores, CMA-ES with elitism, 868 trainable parameters.

---

## Run 1 — Baseline (no regularisation, original sigma)

**Command:**
```
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml
    --generations 300 --popsize 96 --nb-episode-evals 3 --sigma-init 0.1
    --save-dna --snapshot
```

| Param | Value |
|---|---|
| sigma_init | 0.1 |
| popsize | 96 |
| growth_cycles | 5 |
| nb_episode_evals | 3 |
| size_regularisation | none |
| pruning | off |

**Results:**

| Gen | Best (reg) | Pop mean | Sigma | s/gen |
|---|---|---|---|---|
| 0 | −83.5 | −245 | 0.098 | 2.7 |
| 100 | −4.2 | −167 | 0.037 | ~20 |
| 200 | +10.3 | −149 | 0.019 | ~60 |
| 300 | +10.3 | −137 | 0.008 | ~120 |

- **100-episode eval of best DNA: −127**
- **Final brain: 296 nodes, 25k edges**

**Lessons:**
- `sigma_init=0.1` (paper default) causes premature convergence — sigma collapsed to 0.008 by gen 300.
- Brain exploded to 296 nodes; gen time climbed from 2s → 120s over 300 gens.
- +10.3 training score was noise — proper evaluation gave −127.
- 3 episode evals far too noisy for reliable fitness signal.

---

## Run 2 — Larger sigma, more evals (killed early)

**Command:**
```
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml
    --generations 200 --popsize 256 --nb-episode-evals 10 --sigma-init 0.5
    --save-dna --show
```

| Param | Value |
|---|---|
| sigma_init | 0.5 |
| popsize | 256 |
| growth_cycles | 5 |
| nb_episode_evals | 10 |
| size_regularisation | none |
| pruning | off |

**Results (killed at gen ~220):**

| Gen | Best (reg) | Pop mean | Sigma | s/gen |
|---|---|---|---|---|
| 0 | −93.8 | −213 | 0.479 | 2.7 |
| 20 | −79.0 | −160 | 0.441 | 22.6 |
| 130 | +10.3 | −150 | 0.019 | ~50 |

**Lessons:**
- `sigma_init=0.5` kept sigma alive longer but brain still exploded (50s/gen by gen 130).
- Best only marginally better than Run 1.
- Brain size explosion is the primary bottleneck — must be addressed.

---

## Run 3 — Fewer growth cycles (killed early)

**Command:**
```
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml
    --generations 200 --popsize 256 --nb-episode-evals 10 --sigma-init 0.3
    --growth-cycles 3 --save-dna --show
```

| Param | Value |
|---|---|
| sigma_init | 0.3 |
| popsize | 256 |
| growth_cycles | 3 |
| nb_episode_evals | 10 |
| size_regularisation | none |
| pruning | off |

**Results (killed at gen ~150):**

| Gen | Best (reg) | Pop mean | Sigma | s/gen |
|---|---|---|---|---|
| 0 | −91.2 | −429 | 0.287 | 2.5 |
| 20 | −67.3 | −257 | 0.273 | 4.3 |
| 90 | −62.9 | −173 | 0.204 | 4.6 |
| 150 | −62.9 | −144 | 0.153 | 20.5 |

**Lessons:**
- Reducing growth cycles from 5→3 helped initially (4s/gen vs 50s), but gen time still climbed to 20s by gen 150.
- 3 cycles insufficient to prevent brain explosion without explicit size pressure.
- Marginal improvement over Run 2 in best score.

---

## Run 4 — io_ratio size regularisation ✓

**Command:**
```
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml
    --generations 200 --popsize 256 --nb-episode-evals 10 --sigma-init 0.3
    --growth-cycles 3 --size-reg io_ratio --size-reg-alpha 5
    --save-dna --show
```

| Param | Value |
|---|---|
| sigma_init | 0.3 |
| popsize | 256 |
| growth_cycles | 3 |
| nb_episode_evals | 10 |
| size_regularisation | io_ratio, alpha=5 |
| pruning | off |

**Results:**

| Gen | Best (reg) | Pop mean | Sigma | s/gen |
|---|---|---|---|---|
| 0 | −103.6 | −383 | 0.288 | 0.9 |
| 10 | −65.6 | −349 | 0.254 | 2.0 |
| 20 | −37.0 | −300 | 0.261 | 2.0 |
| 200 | −37.0 | −398 | 0.064 | 2.4 |

- **100-episode eval of best DNA: −132**
- **Final brain: 16 nodes** (vs 296 in Run 1)
- **Gen time: flat 2.3s for all 200 gens** ✓

**Lessons:**
- `io_ratio` regularisation solved the brain explosion completely — 16 nodes, flat gen time throughout.
- Best jumped from −104 → −37 in just 20 gens (better early progress than any previous run).
- Sigma still collapsed (0.29 → 0.06) — CMA-ES locked into a local optimum at gen 20 and never escaped.
- Training best (−37) vs 100-eval score (−132): 10 episode evals still too noisy.
- Pruning was off — could help further compactify the brain.

---

## Run 5 — Large sigma, full cycles, pruning

**Command:**
```
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml
    --generations 200 --popsize 512 --nb-episode-evals 10 --sigma-init 1
    --growth-cycles 5 --size-reg io_ratio --size-reg-alpha 5 --pruning
    --save-dna --show
```

| Param | Value |
|---|---|
| sigma_init | 1.0 |
| popsize | 512 |
| growth_cycles | 5 |
| nb_episode_evals | 10 |
| size_regularisation | io_ratio, alpha=5 |
| pruning | on |

*Results pending.*
