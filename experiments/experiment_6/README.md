# Experiment 6 — nested modules

Experiment 4's ECGP lets a circuit wrap nodes into a callable module, but a module can
never contain another module. So evolution can reuse a small part but cannot build a
larger part out of smaller ones. This experiment builds mechanisms that allow that and
asks whether evolution keeps, reuses and builds on modules. Results: `RESULTS.md`.

## The three mechanisms

- **SMCGP** (`smcgp.py`): Self-Modifying CGP, Harding, Miller & Banzhaf, CEC 2009. The
  genotype holds ordinary gates and 13 graph-rewriting operators that regrow the
  phenotype before each evaluation. It has no named, callable modules, so it answers a
  related question (does self-modifying growth produce repeated sub-graphs) rather than
  this experiment's. It was built first because its paper was fully available.
  `PAPER_SPEC.md` tags every detail `[verbatim]`, `[inferred]` or `[our choice]`.
- **necgp** (`necgp/`): ECGP with nesting allowed, priced by accepting a nested
  `compress` with probability `nest_decay ** (depth − 1)`. Our own extension: Modular
  CGP (Walker & Miller's nesting variant) is described only in a thesis and a book
  chapter we could not obtain.
- **necgp_pairwise** (`necgp_pairwise/`): necgp with `compress` restricted to one
  wired-together pair, so a module that does nothing a loose NAND could not do can
  never be created. Module mutation and interface operators are removed.

Each is a self-contained copy (no imports across experiment folders), so changing one
cannot move another's results.

## Files

| file | what it is |
|---|---|
| `smcgp.py`, `gates.py`, `tasks.py`, `config.py`, `train.py` | SMCGP, its function sets, the even-parity curriculum, flags and (1+4) ES loop |
| `test_smcgp.py` | SMCGP correctness tests |
| `necgp/ecgp.py` | experiment 4's `ecgp.py` with nesting; changes are tagged `EXTENDED`, the module docstring is the map |
| `necgp/cgp.py`, `necgp/gates.py`, `necgp/tasks.py` | copies of experiment 4's (tasks adjusted for the folder depth) |
| `necgp/smoke.py` | a plain (1+4) ES for a fixed generation budget |
| `necgp/stage_run.py` | run to solution with the given `--nest-decay` and with 0.0 on the same seed, and draw a stage sheet |
| `necgp/seed_sweep.py`, `necgp/seed_sweep.csv` | nested vs flat over seeds 0–9, with a paired Wilcoxon test |
| `necgp/decompose_seed0.py`, `necgp/decompose.py`, `necgp/visualize.py` | the fake-module count and drawings (box labels `|x` give nesting depth) |
| `necgp/test_necgp.py`, `necgp/test_visualize.py` | tests |
| `necgp_pairwise/` | see its `README.md` |

## Run

```bash
python train.py --max-inputs 6 --max-evals 300000                 # SMCGP, even parity
cd necgp && python stage_run.py --gates nand --nest-decay 0.5     # nested vs flat, seed 0
cd necgp && python seed_sweep.py                                  # seeds 0-9
```

Everything is pure Python with one big integer per wire (all patterns evaluated in one
bitwise operation), which caps tasks at about 20 inputs.
