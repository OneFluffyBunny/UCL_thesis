# Experiment 5 — results

Infrastructure for many-input, many-output circuits (see `README.md`). No modularity
hypothesis has been tested here yet. What is established is that the fork is the
same search as experiment 4, where PyPy helps and where it does not, and that the
behavioural readout works.

## The fork is experiment 4's algorithm

`test_equivalence.py` runs experiment 4 and experiment 5 in separate processes and
compares a rolling hash of the per-generation fitness trace and the final genotype,
for CGP and ECGP:

```
ok  CGP  n=100 seed=0 gens=400: identical to experiment 4 (hits 212, trace 5f7c67ad7335)
ok  CGP  n=400 seed=0 gens=200: identical to experiment 4 (hits 208, trace 7647a18a6f11)
ok  ECGP n=100 seed=0 gens=300: identical to experiment 4 (hits 221, trace 640ff1b6e941)
```

One difference is equalised explicitly, and a control checks that without it the two
really diverge: `cgp._draw_slots` returns mutation slots in draw order instead of as a
`set`. CPython and PyPy iterate small-integer sets in different orders, so the same
seed gave different runs on the two interpreters. The draws, the chosen slots and the
RNG state are unchanged; only the order the slots are written in changes, so a given
experiment-4 seed does not reproduce here. The same test shows PyPy and CPython give
identical runs of experiment 5.

## PyPy has a crossover at 14–15 inputs

`bench.py --crossover`: CGP, 200 nodes, `parityN` (inputs vary one at a time, one
output), best of 3 after a warm-up; Core Ultra 7 155H, Windows 11, CPython 3.10.20 vs
PyPy 7.3.23 (Python 3.11.15). Milliseconds per generation:

| inputs | bytes per wire | CPython | PyPy | PyPy speed-up |
|---|---|---|---|---|
| 8 | 32 | 0.201 | 0.025 | 8.18× |
| 10 | 128 | 0.269 | 0.042 | 6.38× |
| 12 | 512 | 0.215 | 0.071 | 3.03× |
| 13 | 1,024 | 0.142 | 0.082 | 1.73× |
| 14 | 2,048 | 0.111 | 0.096 | 1.15× |
| 15 | 4,096 | 0.126 | 0.191 | **0.66×** |
| 16 | 8,192 | 0.150 | 0.419 | **0.36×** |
| 17 | 16,384 | 0.250 | 0.822 | **0.30×** |
| 18 | 32,768 | 0.449 | 2.195 | **0.20×** |

A wire is a `2**n_in`-bit integer. CPython's big-integer bitwise operations are
hand-written C; PyPy removes the interpreter overhead around them, and once a single
operation costs more than that overhead its slower big integers dominate. The
crossover is a property of truth-table width, so wide ("big brain") tasks should run
on CPython. `train.py` prints a note when a run is on the slow side.

`bench.py` on real configurations (ms per generation; `code` = experiment 5 vs 4 on
CPython, `PyPy` = experiment 5 on PyPy vs CPython; raw data `bench_results.csv`):

| config | in/out | exp 4 CPython | exp 5 CPython | exp 5 PyPy | code | PyPy | total |
|---|---|---|---|---|---|---|---|
| CGP `retina_ka2005` n=100 | 8/1 | 0.152 | 0.129 | 0.011 | 1.2× | 11.5× | **13.6×** |
| CGP `retina_ka2005` n=400 | 8/1 | 0.301 | 0.347 | 0.043 | 0.9× | 8.0× | **7.0×** |
| CGP `retina_ka2005` n=800 | 8/1 | 0.492 | 0.535 | 0.080 | 0.9× | 6.7× | **6.1×** |
| ECGP `retina_ka2005` n=100 | 8/1 | 1.008 | 0.860 | 0.155 | 1.2× | 5.6× | **6.5×** |
| ECGP `retina_ka2005` n=400 | 8/1 | 3.927 | 4.479 | 0.847 | 0.9× | 5.3× | **4.6×** |
| CGP `mult3` n=200 | 6/6 | — | 0.299 | 0.032 | — | 9.3× | — |
| CGP `add4` n=200 | 9/5 | — | 0.198 | 0.040 | — | 4.9× | — |
| CGP `retina_x2` n=200 | 16/2 | — | 0.232 | 0.736 | — | **0.3×** | — |

On 8-input tasks all of the speed-up comes from the interpreter (up to 13.6×). On the
one 16-input task PyPy is 3× slower.

**`x ^ mask` instead of `~x & mask`.** Equal for any `x` inside the mask, with one
full-width allocation instead of two. `bench_ab.py` (a copy with the change reverted,
same hits): no effect at 8 inputs, 1.15× at 14, **2.0× on CPython and 1.7× on PyPy at
18 inputs**. It is the only optimisation found that helps where PyPy cannot.
`test_equivalence.py`'s trace hashes are unchanged by it.

**Chunked wires do not help.** Splitting a wire into a list of 64-bit words is 10–80×
slower than one big integer on both interpreters at every width tested (256 bits to
256 kbits): each gate allocates a list of boxed integers. NumPy `uint64` arrays beat
big integers only above ~64k patterns (1.2× at 64k, 3.9× at 1M).

## Many outputs and the behavioural readout

A program carries `O` output genes; the score is correct (output, pattern) pairs,
kept an integer so the (1+4) ES's tie rule stays exact. Task families with known
decompositions are defined in mask algebra and checked on every pattern by
`test_tasks.py`: `retina_xN` (independent sub-problems), `addN`, `multN`, and
`parityN` (no decomposition).

`cgp.behavioural_deps` gives the inputs each output actually depends on (flipping the
input alone changes the output for some pattern), logged as `beh_pure` next to the
structural `out_pure`. They differ when a wire sits in an output's cone without
influencing it; `test_tasks.py` checks this on `x AND (NOT x)` (cone `{0}`, support
`{}`).

## Open

- **No null model for `beh_pure`.** `retina_xN`'s outputs are independent by
  construction, so a high score there means nothing until compared with random
  circuits of the same size (or random genomes through the same encoding).
- **~20 inputs is the ceiling** for exhaustive evaluation (a wire is 2 MB at 24
  inputs). Going further means scoring a sample of patterns, which makes fitness
  noisy and needs a decision about the tie rule.
- `multN`'s input groups are not a decomposition; use it only as a negative control.
