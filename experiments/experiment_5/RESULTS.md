# Experiment 5 — results log

The lab notebook for the **big-brain** arm: many-input / many-output CGP/ECGP, and
whether **behavioural modularity** shows up there. Same conventions as
`../experiment_4/RESULTS.md` — newest section at the bottom, every entry dated, every
claim tied to the command and seed count that produced it, and nulls written up as
carefully as positives.

---

## 2026-08-18 — §1 the fork, the PyPy port, and where PyPy stops helping

No science yet. This entry is the infrastructure, and one measured result that
constrains every experiment that follows.

### Why a fork rather than a shared core

Experiment 4's `RESULTS.md` identifies runs by seed, so any change to its search
invalidates them. Experiment 5 needed a change to its search (many outputs) and a
change to its randomness (below), so `cgp.py`, `ecgp.py`, `gates.py`, `tasks.py`,
`config.py`, `train.py`, `visualize.py` and the tests were copied into
`experiment_5/` and experiment 4 was left alone.

The cost of a fork is silent drift, so it is tested rather than promised.
`test_equivalence.py` drives **both** implementations end to end in separate
subprocesses (they both own modules named `cgp`, so they cannot share an interpreter)
and compares a rolling hash of the per-generation fitness trace plus the final
genotype:

```
ok  CGP  n=100 seed=0 gens=400: identical to experiment 4 (hits 212, trace 5f7c67ad7335)
ok  CGP  n=400 seed=0 gens=200: identical to experiment 4 (hits 208, trace 7647a18a6f11)
ok  ECGP n=100 seed=0 gens=300: identical to experiment 4 (hits 221, trace 640ff1b6e941)
```

...with one difference equalised by an explicit monkey-patch, and a control asserting
that without the patch the two really do diverge (otherwise the match above would be
vacuous).

### The one algorithmic change: draw order

`cgp._draw_slots` picks which gene slots a mutation hits. It returned a `set`, and
`mutate` iterated it. **Set iteration order for small ints is an implementation
detail**, and CPython's and PyPy's differ — so the same seed produced different runs
under the two interpreters, and "PyPy gives the same answers" could never have been
demonstrated. It now returns the picks in draw order.

What that does and does not change:

- The draw loop is untouched: same `random()` calls, same acceptance test. From any
  RNG state the **same slots** come out and the generator is left in the **same
  state** — asserted over 1000 draws across five (total, k) shapes, and separately for
  the wiring-weighted sampler.
- What changes is the order those slots are *written* in, and therefore which random
  value lands in which slot. The distribution over mutations is identical; a given
  seed's trajectory is not.

⚠️ **An experiment-4 seed does not reproduce in experiment 5.** Same class of break as
the numpy→`random` switch already recorded in `experiment_4/cgp.py`.

### PyPy: it works, and it is bit-exact

`setup_pypy.py` pins **PyPy 7.3.23 / Python 3.11** and builds `.venv-pypy` with *no
packages* — experiment 5's search is stdlib-only, which is what made the port
possible. Getting there needed:

- `tasks.py` re-expressed in mask algebra instead of loading `kashtan_alon/tasks.py`
  through numpy. `test_tasks.py` asserts the two agree bit for bit on every pattern of
  every shared task, so the duplication cannot silently drift from the KA rule.
- numpy and matplotlib made lazy in `train.py`. numpy was only doing `savez` and three
  summary statistics; genotypes are now JSON, the statistics are a dozen lines of
  arithmetic. `--viz` under PyPy fails immediately with the fix in the message, and
  `render.py` redraws under CPython from the saved circuits.

`test_equivalence.py` then shows the pay-off — not "PyPy is close", the identical run:

```
ok  CGP  n=100 seed=0 gens=800: PyPy == CPython (hits 212, trace 367c08dd0fff)
ok  CGP  n=800 seed=7 gens=200: PyPy == CPython (hits 212, trace 4c87bd95a331)
ok  ECGP n=100 seed=0 gens=500: PyPy == CPython (hits 222, trace 3c050eec7854)
```

### ⚠️ THE RESULT THAT MATTERS: PyPy has a crossover, and big brains are on the wrong side of it

Measured with `bench.py --crossover` (CGP, 200 nodes, `parityN` — the one family whose
input count moves one at a time with the output count fixed at 1; best of 3 timed
repeats after an untimed warm-up pass). Machine: Core Ultra 7 155H, Windows 11,
CPython 3.10.20 vs PyPy 7.3.23 / 3.11.15.

| inputs | patterns | bytes/wire | CPython ms/gen | PyPy ms/gen | PyPy speedup |
|---|---|---|---|---|---|
| 8 | 256 | 32 | 0.201 | 0.025 | 8.18x |
| 10 | 1 024 | 128 | 0.269 | 0.042 | 6.38x |
| 12 | 4 096 | 512 | 0.215 | 0.071 | 3.03x |
| 13 | 8 192 | 1 024 | 0.142 | 0.082 | 1.73x |
| 14 | 16 384 | 2 048 | 0.111 | 0.096 | 1.15x |
| 15 | 32 768 | 4 096 | 0.126 | 0.191 | **0.66x** |
| 16 | 65 536 | 8 192 | 0.150 | 0.419 | **0.36x** |
| 17 | 131 072 | 16 384 | 0.250 | 0.822 | **0.30x** |
| 18 | 262 144 | 32 768 | 0.449 | 2.195 | **0.20x** |

**Crossover at 14–15 inputs.** PyPy wins by 8x at 8 inputs and loses by 5x at 18.

The branch was started on the premise that PyPy is the obvious speed-up. It is — for
**narrow** tasks. It is the opposite for wide ones, by a factor of five.

The cause is not the search. A wire is a `2**n_in`-bit integer, and CPython's
big-integer bitwise ops are hand-written C. PyPy's advantage is removing interpreter
overhead *around* those ops; once one op costs more than the overhead around it, the
advantage is gone and PyPy's slower `rbigint` is all that is left. The crossover is
therefore a property of truth-table width, not of the algorithm.

**Since "big brains" means "wider truth tables", the regime this experiment is named
after is the regime where PyPy loses.** `train.py` prints a one-line note when a run
is on the wrong side of the line; `bench.py --crossover` re-measures it on any machine.

### The headline sweep

Measured with `bench.py` (best of 3 timed repeats after an untimed warm-up; same
machine). `exp4 CPython` is the frozen baseline, `code` isolates the fork's changes
at fixed interpreter, `PyPy vs exp5` isolates the interpreter.

| config | in/out | exp4 CPython | exp5 CPython | exp5 PyPy | code | PyPy vs exp5 | **total** |
|---|---|---|---|---|---|---|---|
| CGP retina_ka2005 n=100 | 8/1 | 0.152 | 0.129 | 0.011 | 1.2x | 11.5x | **13.6x** |
| CGP retina_ka2005 n=400 | 8/1 | 0.301 | 0.347 | 0.043 | 0.9x | 8.0x | **7.0x** |
| CGP retina_ka2005 n=800 | 8/1 | 0.492 | 0.535 | 0.080 | 0.9x | 6.7x | **6.1x** |
| ECGP retina_ka2005 n=100 | 8/1 | 1.008 | 0.860 | 0.155 | 1.2x | 5.6x | **6.5x** |
| ECGP retina_ka2005 n=400 | 8/1 | 3.927 | 4.479 | 0.847 | 0.9x | 5.3x | **4.6x** |
| CGP mult3 n=200 | 6/6 | — | 0.299 | 0.032 | — | 9.3x | — |
| CGP add4 n=200 | 9/5 | — | 0.198 | 0.040 | — | 4.9x | — |
| CGP retina_x2 n=200 | 16/2 | — | 0.232 | 0.736 | — | **0.3x** | — |

ms per generation (mutate 4 offspring, evaluate, select). Raw data:
`bench_results.csv`. The last three rows have no experiment-4 column because
experiment 4 cannot express a multi-output task at all.

**Read it as two separate levers.** The `code` column is 0.9–1.2x — i.e. **the fork
bought no speed on an 8-input task**, which is the honest answer: the complement
rewrite is worth nothing when a wire is 32 bytes (its payoff is in the next section),
and the ±10% either way is scheduling noise on a hybrid CPU. All of the headline
speed-up on narrow tasks is the interpreter, and it is up to **13.6x** end to end.

**And the last row is the warning.** On the one genuinely wide task in the sweep,
`retina_x2` at 16 inputs, PyPy is 3x *slower*, so the "total" speed-up there is a
slow-down. That is the crossover above, showing up in a real configuration rather
than in a sweep designed to find it.

### One optimisation that helps both interpreters

Every complement in the hot path was `~x & mask`. On an arbitrary-precision integer
that allocates two full-width temporaries — the negative complement, then the masked
result — where `x ^ mask` allocates one, and the two are equal for any `x` inside the
mask. Same for the fitness count: `n_patterns - popcount(x ^ target)` avoids the
full-width complement that `popcount(~(x ^ target) & mask)` needs.

Measured back-to-back at equal effort (`bench_ab.py`: a copy of experiment 5 with the
rewrite reverted, both driven through `bench_driver.py`, best of 3, `parityN`,
200 nodes). Both copies reach identical `hits`, so they are the same search.

| inputs | CPython `~x & mask` | CPython `x ^ mask` | gain | PyPy `~x & mask` | PyPy `x ^ mask` | gain |
|---|---|---|---|---|---|---|
| 14 | 0.125 | 0.109 | 1.15x | 0.126 | 0.100 | 1.26x |
| 15 | 0.152 | 0.126 | 1.21x | 0.279 | 0.214 | 1.31x |
| 16 | 0.248 | 0.174 | 1.43x | 0.670 | 0.407 | 1.64x |
| 17 | 0.360 | 0.256 | 1.41x | 1.122 | 0.855 | 1.31x |
| 18 | 0.791 | 0.394 | **2.01x** | 3.860 | 2.254 | 1.71x |

The gain grows with width because that is what it is buying: nothing at 8 inputs
(where a wire is 32 bytes and the loop is interpreter-bound), **2x at 18** (where a
wire is 32 KB and the loop is memory-bound). It is the only optimisation found here
that helps in the regime PyPy cannot.

`test_equivalence.py`'s trace hashes are **unchanged** by this edit, which is the
strongest available evidence that it is a pure rewrite: the same seeds still walk the
same generations to the same genotypes.

### A measurement bug this exposed, in the parallelism test

`test_perf.py::test_cpu_utilisation` asserts the worker pool stays above 70%
efficiency. Under PyPy it measured 64%, then **49%** when the workload was tripled —
efficiency falling as the job grew, which is the wrong direction for a spawn-cost
problem.

The cause was not the pool. Seeds stop early when they solve, so a seed that solved at
generation 3 000 finished while its neighbour ran to the end and the pool waited on the
slowest: **load imbalance in the search, being reported as idleness in the pool.** It
stayed hidden at experiment 4's 6 000 CPython generations (2 of 12 seeds solved, small
spread) and only became dominant once PyPy made a long budget cheap enough to run.

Fixed by giving the test `--no-stop-on-solution`, so every seed does identical work.
Both interpreters now pass on a job that is genuinely worth parallelising: CPython 72%
(4.3x over serial), PyPy 81% (4.9x).

Two smaller consequences, both now in `train.py`:

- `_SPAWN_BUDGET_SECS` is per interpreter (8 s CPython, 25 s PyPy). A worker's JIT
  warms from cold — compiled traces do not survive a process spawn — so PyPy needs a
  bigger job before a pool pays for itself.
- `_GEN_COST_MS` likewise. Using the CPython anchors under PyPy overestimates every
  job by ~4x and sends short runs to a pool that cannot amortise.

### Many outputs

A program now carries `O` output genes, `O` coming from the task. Score is correct
`(output, pattern)` pairs summed over all outputs; a perfect score is
`O * n_patterns`. Summed rather than averaged so the score stays an **integer** — the
(1+4) ES tests `==` on it and that branch is the neutral drift the search actually runs
on, which float equality would make fire erratically.

New parametric task families, all defined in mask algebra and all verified on every
pattern by `test_tasks.py`: `retina_xN` (8N in, N out — the positive control, an exact
known decomposition), `addN`, `multN` (Walker & Miller's own ECGP benchmarks), and
`parityN` (the negative control — no decomposition exists).

### The behavioural readout

`cgp.behavioural_deps` returns the inputs an output's behaviour *actually* depends on:
input `i` counts iff some pattern exists where flipping `i` alone changes the output.
Exact, and computed with one shift and three masks per input regardless of circuit
size, by exploiting the fact that patterns `r` and `r + 2**(n_in-1-i)` differ in input
`i` alone. Logged as `beh_pure` at every log row, next to the structural `out_pure`.

The two disagreeing is the point: a wire can sit in an output's cone and carry no
influence at all. `test_tasks.py` pins that down on `x AND (NOT x)` — structural cone
`{0}`, behavioural support `{}`.

First sighting, on a 3 000-generation `retina_x2` run (2 seeds, 300 nodes, not an
experiment — a smoke test): seed 0 reached 1/2 behaviourally pure outputs, seed 1
reached 2/2. Reported only to show the instrument produces a moving number; it means
nothing until there is a null model for it (see `PAPER_SPEC.md` §7).

### Open threads

- **No null model for `beh_pure`.** Counting behaviourally pure outputs is a
  description, not a test. `retina_xN`'s outputs are independent *by construction*, so
  a high score there is unsurprising; the question is how high a random circuit of the
  same size scores. Candidates: degree-preserving rewire, or random genotypes through
  the same encoding (Kashtan & Alon's own second null — memory
  `reference_ka_modularity_metric`).
- **~20 inputs is the ceiling.** Exhaustive evaluation makes a wire `2**n_in` bits;
  2 MB at 24 inputs. No interpreter fixes that. Getting past it means scoring a
  *sampled* subset of patterns, which makes fitness stochastic and needs an explicit
  decision about selection and the neutral tie-break under noise. `tasks.py` is written
  in terms of `n_patterns()` rather than `1 << n_inputs()` throughout so it stays a
  one-file change when that decision is made.
- ~~**Chunked masks, untested.**~~ **CLOSED 2026-08-19 — falsified, do not retry.**
  The idea was that splitting a wire into 64-bit words would turn PyPy's weakness
  (slow big integers) into its strength (many cheap word-sized ops) and might invert
  the crossover. Measured directly (`(a & b) ^ mask`, one gate, constant-folding
  defeated): a list of 64-bit chunks is **10-80x SLOWER than a single big int on both
  interpreters**, at every width from 256 bits to 256 Kb. PyPy does not rescue it —
  it is the least-bad of the two, and still 9-20x down. The cost is not the arithmetic
  but the per-gate allocation of a list of boxed ints, which the JIT cannot elide
  because the result escapes into the wire array. Big integers stay the representation.
  The only representation that ever beats them is **numpy uint64 arrays, and only
  above ~64k patterns** (1.2x at 64k, 3.9x at 1M, where big-int cost also turns
  superlinear); below that numpy's per-call dispatch (~1.7 us, flat) loses by ~2x.
- **The multiplier's input groups are not a decomposition** and are labelled as such.
  If `multN` is used, it is as the negative control, not as a task expected to
  decompose.
