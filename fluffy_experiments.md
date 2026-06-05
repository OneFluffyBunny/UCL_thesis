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

**Results:**

| Gen | Best (reg) | Pop mean | Sigma | Raw (best) | Nodes (best/mean) | s/gen |
|---|---|---|---|---|---|---|
| 0 | −72.48 | −309.23 | 0.957 | −70.0 | 18 / 86.2 | 1.4 |
| 10 | −72.48 | −313.78 | 0.930 | −80.5 | 23 / 52.4 | 7.9 |
| 20 | −68.38 | −271.34 | 1.062 | −84.6 | 14 / 50.5 | 7.4 |
| 30 | +14.43 | −233.08 | 1.156 | −86.9 | 13 / 26.1 | 5.4 |
| 100 | +14.43 | −158.89 | 1.104 | −81.6 | 12 / 14.1 | 4.2 |
| 190 | +14.43 | −150.49 | 0.596 | −85.9 | 12 / 13.2 | 4.1 |

- **100-episode eval of best DNA: +45.0** — best result across all runs
- **3-episode render: 134.9, 57.6, −129.3 → mean +21.1** — extremely high variance
- **Final brain: 13 nodes, 84 edges** (1 hidden node above seed)
- **Gen time: flat ~4-5s for all 200 gens** ✓
- **Sigma: peaked at 1.36, ended 0.60** — much healthier than previous runs

**Lessons:**
- Large sigma (1.0) + large popsize (512) kept sigma alive and expanding until gen 60 — best sigma behaviour so far.
- **alpha=5 too strong**: DNA gamed the regularisation by staying at exactly 12-13 nodes. Raw reward stuck at ~−80 throughout training, but the 100-eval gave +45 — surprisingly good given how tiny the brain is.
- The +14.43 regularised best came entirely from penalty reduction (tiny brain), not from raw task improvement. The regularisation is creating a perverse incentive to grow zero hidden nodes.
- **Key question for next run**: reduce alpha to let the brain grow a few more useful hidden nodes while still preventing explosion. Try alpha=1 or 2.
- Pop mean improved steadily (−309 → −150) suggesting the population as a whole is learning, but the elitist best gets stuck because it locked in a penalty-gaming solution early.

> **Note on edge regularisation baseline:** Fixed in run 9 — baseline changed from `obs×act=32` to `seed_size²=144`. This only penalises edges added beyond the fully-connected seed, rather than penalising the seed itself.

---

## Run 6 — Lower alpha, more generations (killed at gen ~190)

**Command:**
```
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml
    --generations 500 --popsize 512 --nb-episode-evals 20 --sigma-init 1
    --growth-cycles 5 --size-reg io_ratio --size-reg-alpha 1 --pruning
    --save-dna --show
```

| Param | Value |
|---|---|
| sigma_init | 1.0 |
| popsize | 512 |
| growth_cycles | 5 |
| nb_episode_evals | 20 |
| size_regularisation | io_ratio, alpha=1 |
| pruning | on |

**Results (killed at gen ~190):**

| Gen | Best (reg) | Pop mean | Sigma | Raw (best) | Nodes (best/mean) | s/gen |
|---|---|---|---|---|---|---|
| 0 | −114.10 | −259.23 | 0.956 | −112.4 | 32 / 125.9 | 3.7 |
| 10 | −99.73 | −263.48 | 0.918 | −99.1 | 20 / 97.9 | 28.3 |
| 20 | −89.22 | −268.35 | 0.980 | −90.4 | 69 / 52.7 | 19.0 |
| 40 | −79.89 | −212.19 | 1.134 | −89.0 | 14 / 47.0 | 15.0 |
| 70 | −67.07 | −202.63 | 1.414 | −104.1 | 14 / 35.9 | 10.0 |
| 100 | −67.07 | −179.47 | 1.649 | −87.7 | 30 / 28.2 | 8.5 |
| 180 | −67.07 | −178.01 | 1.010 | −97.3 | 12 / 18.9 | 8.0 |

**Lessons:**
- **Best sigma behaviour yet**: peaked at 1.67 around gen 100, still above 1.0 at gen 180. Large sigma + large popsize is the right approach.
- **alpha=1 better than alpha=5**: raw ≈ regularised in early gens, confirming genuine task improvement rather than penalty gaming. But the 30-point gap reappeared by gen 50+ — best candidate still has a large brain being penalised.
- **Best stuck at −67 from gen 70**: same premature convergence pattern. The elitist mechanism locks in a solution too early.
- **Pop mean improving** (−259 → −168) even when best is frozen — the population keeps learning but the elitist best blocks it.
- **CMA-ES elitism is a problem**: `CMA_elitist: True` means the best solution is always kept, which prevents escape from local optima. Disabling it or using a restart strategy is the most impactful remaining change.
- Gen time settled ~8s and stayed flat — regularisation working well at alpha=1.

---

## Run 7 — Edge-only regularisation (killed at gen 0, too slow)

Edge regularisation without node regularisation — brains exploded to mean 71 nodes immediately, making each generation too slow. Killed after gen 0. Lesson: node regularisation is essential for gen time control; edge reg alone is insufficient.

---

## Run 8 — Both regularisation, no elitism, growth threshold 0.5 (killed at gen ~110)

**Command:**
```
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml
    --generations 500 --popsize 256 --nb-episode-evals 30 --sigma-init 1
    --growth-cycles 5 --growth-threshold 0.5 --size-reg both
    --size-reg-alpha 1 --size-reg-alpha-edges 2 --pruning --no-elitism
    --save-dna --show
```

| Param | Value |
|---|---|
| sigma_init | 1.0 |
| popsize | 256 |
| growth_cycles | 5 |
| nb_episode_evals | 30 |
| size_regularisation | both (alpha_nodes=1, alpha_edges=2) |
| growth_threshold | 0.5 |
| pruning | on |
| elitism | off |

**Results (killed at gen ~110):**

| Gen | Best (reg) | Pop mean | Sigma | Raw (best gen) | Nodes (best/mean) | s/gen |
|---|---|---|---|---|---|---|
| 0 | −121.30 | −512.81 | 0.957 | −114.2 | 18 / 86.8 | 3.9 |
| 10 | −91.55 | −420.54 | 0.807 | −109.5 | 12 / 39.3 | 18.2 |
| 30 | −88.00 | −301.00 | 0.799 | −107.0 | 13 / 19.1 | 11.6 |
| 60 | −72.93 | −196.71 | 0.861 | −112.0 | 12 / 13.1 | 10.8 |
| 110 | −72.93 | −169.64 | 0.899 | −109.5 | 12 / 12.4 | 9.7 |

**Lessons:**
- No-elitism kept sigma rising (0.80 → 0.90) for 110 gens — good exploration.
- Best improved to −72.93 at gen 60 then froze — better than elitist runs but still stalling.
- Mean nodes settled at seed size (12) — edge regularisation combined with growth threshold 0.5 is suppressing all node growth.
- **~40-point raw/reg gap** persists throughout — the 144-edge seed is being heavily penalised by alpha_edges=2 relative to baseline=32. Confirms the note: baseline=144 (seed edges) would be more appropriate than 32.
- Growth threshold 0.5 too conservative — combined with regularisation, zero hidden nodes are growing.
- **Key changes for next run**: use baseline=144 for edge reg (only penalise edges beyond seed), lower growth threshold, or drop edge reg entirely and focus on no-elitism + node reg.
