# Emergence of modularity in evolved networks

Code and results for a UCL MSc thesis. The question is **what makes an evolved
network modular**. The working hypothesis is that modularity is selected for by
evolution, precedes learning, and is encouraged by a compressed genome-to-network
encoding (a genomic bottleneck).

Constraints shared by the models: neurons have no position in space (so modularity
cannot come from wiring-length cost), the neuron count is fixed and only connections
evolve, and each model is the smallest one that can show the effect.

## Layout

| folder | what it is | results |
|---|---|---|
| `kashtan_alon/` | Reproduction of Kashtan & Alon (2005): threshold networks on their retina task, fixed vs modularly varying goals (FG vs MVG), plus an ablation of their fan-in cap | `kashtan_alon/RESULTS.md` |
| `experiments/experiment_1/` | Compressed encoding: a few evolved cell types plus one shared connection rule `g`, searched by CMA-ES | `RESULTS.md` there |
| `experiments/experiment_2/` | Control: the same recurrent network with one gene per synapse, CMA-ES | `RESULTS.md` there |
| `experiments/experiment_3/` | Optimiser control: experiment 2's network trained by gradient descent | `RESULTS.md` there |
| `experiments/experiment_4/` | Boolean circuits (CGP) and their module-acquiring variant (ECGP) on the Kashtan-Alon retina, (1+4) ES and a population GA | `RESULTS.md` there |
| `experiments/experiment_5/` | Fork of experiment 4 for many-input, many-output circuits, runnable under PyPy (infrastructure) | `RESULTS.md` there |
| `experiments/experiment_6/` | Self-modifying CGP and two nested-module extensions of ECGP | `RESULTS.md` there |
| `qmetrics/` | Modularity metrics: Newman Q, Kashtan-Alon's normalised Q_m, the planted left/right score, circuit purity | docstrings |
| `NDP/` | Neural Developmental Programs (Najarro et al. 2023), imported with credit, plus a retina study run on it | `NDP/experiments_paper/retina/RESULTS.md` |
| `latex_figures/` | The figures used in the thesis, each folder with a README giving the regenerating command | |

`experiments/README.md` describes experiments 1-3 and the shared code in
`experiments/`.

## Setup

```bash
conda create -n lndp python=3.10
conda activate lndp
pip install -r requirements.txt
python run_tests.py          # every test suite; add --perf for the timing tests
```

Everything except `NDP/` runs on CPU in this environment. Experiments 5 and 6 can
also run under PyPy (`experiments/experiment_5/setup_pypy.py` builds a venv); the
search code there is standard library only.

Run outputs go to `runs/` folders, which are not committed. Each `RESULTS.md` gives
the command that produced every reported number, so results are regenerated rather
than stored. Where a run is seeded, the notebook says whether it is bit-reproducible
(the pure-Python and NumPy searches are; JAX/CMA-ES runs are reproducible in
distribution, not bit for bit).
