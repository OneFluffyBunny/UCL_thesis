# NDP × LunarLander — results (unsolved)

`LunarLander-v3` on this fork of NDP (with the input/output-anchor redesign), CMA-ES,
868 trainable parameters, 16 cores. Every run starts from

```bash
python train.py --conf experiments_paper/lunarlander/run_experiment.yaml --save-dna ...
```

with the flags below. "Best" is the training fitness after the size penalty; the
100-episode evaluation of the saved best genome is the reliable number.

| run | flags | generations | best (training) | 100-episode eval | brain at the end | notes |
|---|---|---|---|---|---|---|
| 1 | `--popsize 96 --nb-episode-evals 3 --sigma-init 0.1` | 300 | +10.3 | −127 | 296 nodes, 25k edges | σ collapsed to 0.008; generation time grew 2 s → 120 s |
| 2 | `--popsize 256 --nb-episode-evals 10 --sigma-init 0.5` | ~220 (stopped) | +10.3 | — | growing | 50 s per generation by generation 130 |
| 3 | as 2, `--sigma-init 0.3 --growth-cycles 3` | ~150 (stopped) | −62.9 | — | growing | 20 s per generation by generation 150 |
| 4 | as 3, `--size-reg io_ratio --size-reg-alpha 5` | 200 | −37.0 | −132 | 16 nodes | size stayed flat (2.3 s per generation); σ 0.29 → 0.06 |
| 5 | `--popsize 512 --nb-episode-evals 10 --sigma-init 1 --growth-cycles 5 --size-reg io_ratio --size-reg-alpha 5 --pruning` | 200 | +14.4 | **+45** | 13 nodes, 84 edges | best result; the training score came from the size penalty (raw reward stayed ~−80) |
| 6 | as 5, `--nb-episode-evals 20 --size-reg-alpha 1` | ~190 (stopped) | −67.1 | — | 12–30 nodes | σ stayed above 1 until generation 180; best frozen from generation 70 |
| 7 | edge penalty only | 0 (stopped) | — | — | mean 71 nodes | too slow to run |
| 8 | `--popsize 256 --nb-episode-evals 30 --sigma-init 1 --growth-threshold 0.5 --size-reg both --size-reg-alpha 1 --size-reg-alpha-edges 2 --pruning --no-elitism` | ~110 (stopped) | −72.9 | — | 12 nodes (no growth) | without elitism σ kept rising, but the best froze from generation 60 |

What was learned:
- Without a size penalty the grown networks explode and generation time grows without
  bound; a node penalty (`io_ratio`) fixes this, an edge penalty alone does not.
- `sigma_init` 0.1 (the paper's default) collapses early; σ 1.0 with a population of
  512 keeps exploring longest.
- A strong node penalty (α 5) is gamed: the network stays at the seed size and the
  training score improves only through the penalty. With α 1 the best still froze
  early while the population mean kept improving (elitist premature convergence, which
  `--no-elitism` did not cure either).
- Few episodes per evaluation (3–10) make the training fitness far more optimistic than
  a 100-episode evaluation.

The task was not solved on this fork within these budgets.
