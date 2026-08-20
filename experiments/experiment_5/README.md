# Experiment 5 — big brains: does *behavioural* modularity appear at scale?

Experiment 5 takes the CGP/ECGP framework built in `../experiment_4/` and pushes it to
**big brains** — circuits with many program inputs *and many program outputs*, rather
than experiment 4's 8-in / 1-out retina. The question it exists to answer is whether
**behavioural modularity** emerges there: not "is the wiring separable?" (experiment
4's input-cone readout) but "does the circuit decompose into parts that each *do a
job*, and can those parts be identified from behaviour alone?"

Two things make this a separate experiment rather than a bigger run of experiment 4:

1. **Scale is a different regime.** More inputs and more outputs change what the search
   can represent, what it costs to evaluate, and what a "module" can be.
2. **The measurement has to change.** With one output, modularity can only be read off
   the structure. With many outputs there is a behavioural handle: which inputs does
   each output actually depend on, and do those dependency sets factorise?

> **Status.** The *machinery* is built, tested and benchmarked (this file, plus
> `RESULTS.md` §1). The *science* is not started: no hypothesis has been run, and
> `PAPER_SPEC.md` is still mostly TODO. What exists is a many-output search that runs
> **4.6–13.6× faster** than experiment 4 in the regime where that is possible (and
> knows to tell you when it is not), an exact behavioural-dependency readout, and a
> proof that none of it changed the algorithm.

---

## Relationship to experiment 4

Experiment 5 **forked** experiment 4's code rather than sharing it. Experiment 4's
`RESULTS.md` is a lab notebook of runs identified by seed, and any change to its
search would invalidate them, so that copy is frozen and this one moves.

The fork is not a licence to drift. `test_equivalence.py` runs both implementations
end to end and asserts their generation-by-generation fitness traces and final
genotypes hash **equal**, for CGP and for ECGP. Exactly one algorithmic difference is
allowed, and the test equalises it explicitly to prove nothing else is hiding behind
it:

| | experiment 4 | experiment 5 | why |
|---|---|---|---|
| slot sampling | `_draw_slots` returns a `set`; `mutate` iterates it | returns the picks in **draw order** | CPython and PyPy iterate a set of ints differently, so the same seed diverged between interpreters. Same draws, same slots, same RNG state — only the write order changes |
| program outputs | one | `O`, from the task | behavioural modularity needs several jobs to be modular about |
| structural classes | `left` / `right` against one split index | K groups from `tasks.input_groups` | a big-brain task has K sub-problems, not two halves. At K=2 the names are still `left` / `right`, so the CSV columns and diagrams are unchanged |
| task definitions | loads `kashtan_alon/tasks.py` (numpy) | re-expressed in mask algebra (no numpy) | numpy does not exist under PyPy. `test_tasks.py` asserts the two agree bit for bit |
| complements | `~x & mask` | `x ^ mask` | identical for `x ⊆ mask`, but one full-width allocation instead of two — worth **2.0× at 18 inputs**, nothing at 8 |
| numpy / matplotlib | imported at module load | lazy, never in the search | the search must import under PyPy |

⚠️ **An experiment-4 seed does not reproduce here**, because of the draw-order change.
This is the same class of break as the numpy→`random` switch already recorded in
`experiment_4/cgp.py`, and it buys the ability to run the identical experiment under
either interpreter.

---

## Tasks

A task name carries its **size**, so scale is a flag rather than a code edit:
`--task retina_x3`, `--task add6`, `--task mult4`.

| family | inputs | outputs | ground-truth decomposition |
|---|---|---|---|
| `retina_ka2005` (= `retina_x1`) | 8 | 1 | the two 2×2 blocks — identical to experiment 4 |
| `retina_xN` | 8N | N | one group per retina; the N sub-problems are genuinely independent |
| `addN` | 2N+1 | N+1 | one group per bit position `{a_i, b_i}` — a ripple-carry adder's modules |
| `multN` | 2N | 2N | the two operand words. **Not** a modular decomposition; expected to read as almost entirely "mixed", and useful as the negative control |
| `parityN` | N | 1 | none — parity does not decompose at all. The other control, and the sweep axis for `bench.py --crossover` |
| `left`, `and2`, `copy` | 4, 2, 1 | 1 | sanity tasks, inherited |

Every target is built as **mask algebra** — the adder is the textbook full-adder
recurrence over whole-truth-table integers — so a 20-input task builds in
milliseconds rather than a million Python iterations. `test_tasks.py` checks the
retina against `kashtan_alon/tasks.py`, and the adder and multiplier against
arithmetic, on every pattern.

### The 2ⁿ wall

Evaluation is **exhaustive**: one wire is a `2**n_in`-bit integer. That is 8 KB at 16
inputs, 128 KB at 20, 2 MB at 24. Cost per gate grows linearly in it, so **~20 inputs
is the ceiling**, and no interpreter changes that. `config.py` warns above 16 inputs.
Getting past it means scoring a *sampled* subset of patterns instead of all of them —
deliberately not built (it changes what fitness *means*), and `tasks.py` is written in
terms of `n_patterns()` rather than `1 << n_inputs()` throughout so that it stays a
one-file change.

---

## The measurement

Two readouts, deliberately kept apart, because their disagreement is informative.

**Structural** (`cgp.phenotype`) — where a node's input *cone* comes from. A node
whose cone lies inside one group is that group's; one spanning several is `mixed`;
one with an empty cone is `const`. Logged as `left`/`right` (K=2), `pure`, `mixed`,
`const`, and `out_pure` = program outputs whose cone stays inside one group.

**Behavioural** (`cgp.behavioural_deps`) — which inputs actually *move* the output.
Input `i` counts iff some pattern exists where flipping `i` alone changes the output:
the standard support of a boolean function. Logged as `beh_pure`.

They are not the same thing. A wire can sit in an output's cone and carry no influence
at all — `x & ~x`, a gate masked off downstream, a module input the body ignores. A
circuit that is structurally smeared but behaviourally clean is *behaviourally
modular*, and that distinction is the whole point of experiment 5.

Computing it costs one shift and three masks per input, whatever the circuit is (see
`cgp.behavioural_deps` for the pattern-ordering identity that makes that work), so it
runs at every log row rather than only at the end.

---

## Which interpreter — this is measured, not a preference

The branch was started to put the search on **PyPy**. It does help, and it also
*hurts*, and which one depends entirely on how wide the truth tables are:

    PyPy is 8.2× FASTER at 8 inputs, 3.0× at 12, level (1.15×) at 14
    PyPy is 1.5× SLOWER at 15 inputs, and 5× SLOWER at 18

The crossover sits **between 14 and 15 inputs** on this machine. The cause is not the search: a
wire is a big integer, CPython's big-integer bitwise ops are hand-written C, and once
one bitwise op costs more than the interpreter overhead around it, PyPy's only
advantage is gone and its slower `rbigint` shows through. Since "big brains" means
"wider truth tables", **the regime this experiment is named after is the regime where
PyPy loses.** `RESULTS.md` §1 has the table.

`train.py` prints a one-line note at startup when you are on the wrong side of the
line. Re-measure on any new machine with:

```
conda run -n lndp python bench.py --crossover     # where is the line on this machine?
conda run -n lndp python bench.py                # the headline sweep
conda run -n lndp python bench_ab.py <tmpdir>    # A/B one optimisation at equal effort
```

---

## Running it

```bash
# one-time: install PyPy and build the venv (Windows / Linux / macOS)
conda run -n lndp python setup_pypy.py

# narrow tasks (< 14 inputs): PyPy, headless
.venv-pypy/Scripts/python train.py --task retina_ka2005 --no-viz --save-best

# wide tasks (>= 14 inputs): CPython
conda run -n lndp python train.py --task retina_x2 --no-viz --save-best

# diagrams afterwards, always under CPython
conda run -n lndp python render.py runs/<run-dir> --colour-cones
```

`--viz` under PyPy fails immediately with that instruction rather than dying inside
matplotlib's import.

### Tests

```bash
conda run -n lndp python test_tasks.py         # task definitions vs kashtan_alon + arithmetic
conda run -n lndp python test_cgp.py           # fast evaluator vs the reference evaluator
conda run -n lndp python test_ecgp.py          # module operators, flatten, invariants
conda run -n lndp python test_equivalence.py   # == experiment 4, and PyPy == CPython
conda run -n lndp python test_perf.py          # throughput ceilings + pool utilisation
.venv-pypy/Scripts/python test_perf.py         # ...and the same under PyPy
```

`test_perf.py` carries a separate ceiling per interpreter: reusing CPython's would
make the PyPy run unable to fail.

---

## What this does NOT prove

Nothing about modularity yet. This is infrastructure. In particular:

- No hypothesis has been stated (`PAPER_SPEC.md` is a skeleton) and no experiment run.
- `beh_pure` being high on a task whose outputs are independent **by construction**
  (`retina_xN`) is not evidence of anything on its own — the control is `multN` and
  `parityN`, where it should stay low.
- The exhaustive-evaluation ceiling means "big brain" currently tops out around 20
  inputs. That is wide enough for `retina_x2`, `add8`, `mult8`; it is not wide enough
  for anything one would casually call big.

## Status

Machinery complete and verified, 2026-08-18. Science not started. See `RESULTS.md`.
