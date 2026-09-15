# Experiment 2 — direct encoding (the control)

The baseline that experiment 1's compressed encoding is measured against. Same
network, tasks, CMA-ES loop and metrics; only the encoding differs.

|  | experiment 1 | experiment 2 (here) |
|---|---|---|
| genome | K cell-type identities + one shared rule `g` | one weight per allowed edge + one bias per non-input neuron |
| weight of edge i→j | `g(feat_i, feat_j)`, shared across edges | its own parameter |
| role | treatment (genomic bottleneck) | control |

If modularity appears here under the same selection pressure, the encoding is not
what produces it.

## Files

The folder mirrors `../experiment_1` (same flags and outputs) without the
encoding-specific parts.

- `model.py`, `visualize.py`, `tasks.py` — re-export `../shared_direct_model.py`,
  `../shared_direct_viz.py` and `../shared_tasks.py`, which experiment 3 also uses.
- `config.py` — command-line flags → `BrainConfig` + `RunConfig`.
- `train.py` — CMA-ES loop: fixed goal or `--mvg`, accuracy or `--fitness margin`,
  optional `--synaptic-budget S --shrink τ`, per-generation champion archive.

## Run

```bash
# the FG-vs-MVG study (all four arms): see RESULTS.md section 1
python train.py --task retina_ka2005 --operation and --no-balanced --fitness margin \
    --n-hidden 24 --generations 5000 --no-early-stop --archive-interval 1 \
    [--mvg --mvg-ops and,or --switch-interval 20] [--synaptic-budget 4 --shrink 0.9]
```
