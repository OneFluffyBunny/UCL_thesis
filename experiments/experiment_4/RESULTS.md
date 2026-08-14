# Experiment 4 — results log

Lab notebook for the CGP/ECGP module-reuse experiment. Hypothesis and design in
`README.md`; every parameter's provenance in `PAPER_SPEC.md`.

---

## 2026-08-13 — CGP plumbing built; genotype-size sweep on the FG retina

Task `retina_ka2005` (KA 2005's verified object rule, loaded from
`kashtan_alon/tasks.py`), fixed goal `L AND R`, raw correct-count fitness over all
256 patterns, function set AND/NAND/OR/NOR, (1+4) ES.

### The generation axis is NOT comparable to Kashtan–Alon's

Worth stating before any number below is read. KA's median was ~1.68M evaluations
on a network of ±1-weight threshold units. CGP searches **boolean circuits with
exact gates** — a different and much more direct substrate for a boolean task — and
solves the same goal in **~34k evaluations**, roughly 50× fewer. That is a
statement about the substrate, not about search quality, and KA's figure must not
be used as a budget target for this experiment.

### How small *can* the circuit be? 19 gates

Hand-built and verified at **256/256** (`scratchpad/min_circuit.py`). For one 2×2
block with `(a,b)` the outer column and `(c,d)` the inner, KA's object rule is
`[a+b+c+d ≥ 3] ∨ [¬c ∧ ¬d ∧ (a∨b)]`, which with `v=a&b, u=a|b, s=c|d, t=c&d`
collapses to

    object = (v & s) | (u & XNOR(c,d))          -- 9 two-input gates

so the whole task is **9 + 9 + 1 = 19 gates**: two half-detectors plus one
combiner. ⚠️ This is an *upper bound* on the minimum — no claim that 19 is
optimal; general circuit-size lower bounds are not settled here.

Two consequences worth keeping in view:

- **The two halves compute the *same function* on disjoint inputs**
  (`obj(p0,p1,p2,p3)` and `obj(p6,p7,p4,p5)`). The task has a literal built-in
  reusable module. This is the basis of H4 in `README.md`.
- **Genotype length is not circuit size.** Everything in the sweep below is a
  *search-space* parameter. A 50-node genotype can represent the 19-gate circuit
  perfectly well; it simply never finds it.

### Sweep: genotype size × mutation rate

6 seeds per cell, 25 000-generation cap, stop-on-solution.

| nodes | mut rate | mutations/offspring | solved | median gens | fastest | mean acc | mean active | median evals |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.03 | 5 | 0/6 | – | – | 0.9206 | 16.8 | 100 005 |
| 100 | 0.03 | 9 *(Table II)* | 4/6 | 16 973 | 7 965 | 0.9850 | 27.7 | 70 211 |
| 200 | 0.01 | 6 | 1/6 | 20 559 | 20 559 | 0.9583 | 26.3 | 100 005 |
| 200 | 0.02 | 12 | 5/6 | 17 640 | 7 515 | 0.9857 | 28.7 | 71 169 |
| 200 | 0.03 | 18 | 3/6 | 14 481 | 12 691 | 0.9798 | 27.7 | 93 035 |
| 400 | 0.03 | 36 | 6/6 | 19 870 | 7 939 | 1.0000 | 34.5 | 79 489 |
| **800** | **0.01** | **24** | **6/6** | **8 404** | **5 841** | **1.0000** | **59.5** | **33 625** |
| 800 | 0.03 | 72 | 6/6 | 10 563 | 6 110 | 1.0000 | 58.8 | 42 261 |
| 1600 | 0.03 | 144 | 4/6 | 8 053 | – | 0.9792 | – | – |

**Findings.**

1. **Genotype size is the dominant knob.** Table II's 100 nodes / 3% solves only
   4/6 within 25k generations; 800 nodes solves 6/6 in half the generations and
   half the evaluations; 50 nodes never solves. Note this is *reachability*, not
   representability — 50 nodes can encode the 19-gate circuit.
2. **It plateaus at ~800.** 1600 nodes is not better (4/6, and ~6× the wall clock
   for the same median), so "bigger is better" runs out rather than continuing.
3. **The effective knob is mutations *per offspring*, not the rate.** 0.01 is bad
   at 200 nodes (6 mutations, 1/6) and best at 800 (24 mutations, 6/6) — the rate
   multiplies a genotype length that is itself changing. Report the count, not
   just the percentage.
4. **The extra nodes are neutral scaffolding, not a bigger circuit.** At 800 nodes
   the phenotype is still only ~59 active nodes out of 800. This is CGP's
   neutrality doing the work — consistent with the paper's own remark that larger
   modules help because they "contain more inactive nodes, so that neutrality …
   could be having an impact".
5. **The solved circuits look modular.** The first perfect solution (100 nodes,
   seed 0, gen 16 570) decomposes as 6 left-only nodes, 5 right-only, 12 mixed —
   a left subtree and a right subtree merging in a chain near the output. See
   `frames/seed0_final.png`. This is an eyeball observation on n=1, not a result.

### ⚠️ The literature already answered the genotype-length question — check it first

The sweep above was run **before** checking the paper on this point, and 800 was
an invented value (my own doubling ladder 50→1600), not a published one. Both
passages below are in the PDF already in `papers/`:

> *(p. 2)* "the very short genotypes hindered performance, as the amount of effort
> required to evolve solutions in CGP is **strongly dependent on the chosen
> genotype length**. This has recently been investigated in detail **[19]**, where
> it was found that **large genotypes required markedly less effort to evolve
> solutions**."

> *(§X)* "ECGP was rerun on the even parity problem … with a maximum module size of
> 20 nodes and a **genotype size of 400 nodes** … The extra resources provided a
> significant performance boost … computational effort for the even 4 parity was
> 32 641, a **speedup of 6.18**"

> *(§X)* "all of the computational effort figures for CGP and ECGP are **heavily
> dependent on the number of nodes encoded in the genotype [19]**. This can cause
> vast fluctuations in the performance of both techniques"

So the "bigger genotype solves faster" result is **expected**, not a discovery,
and the paper itself went 100 → 400 for exactly this reason.

**Missing source.** `[19] = J. F. Miller and S. L. Smith, "Redundancy and
computational efficiency in Cartesian genetic programming," IEEE Trans. Evol.
Comput. 10(2):167–174, Apr. 2006.` We do **not** have it. It is the paper that
closes the "how much redundancy is optimal?" question quantitatively; no number
for that ratio is recorded here because none is sourced. Download it before
treating any genotype length as principled.

### Chosen FG configuration

    --nodes 400 --mutation-rate 0.03 --task retina_ka2005 --operation and

**400 because the paper used 400**, and it solved 6/6 in the sweep at ~35 active
nodes. It still deviates from Table II's 100 (recorded in `PAPER_SPEC.md` §9 —
Table II's values were tuned on parity/adders/multipliers, never on this task),
but it is a value taken from the source rather than invented.

**800 is deprecated** as the headline configuration. It was never sourced, and at
~70 active nodes for a 19-gate problem it is 3.7× bloated — which contaminates the
structural readout (see the confound below). Keep it only as a speed tool. CLI
defaults remain at Table II's values so nothing deviates silently.

### Calibration: is ~36 000 generations bad? No.

Walker & Miller's computational-effort figures, same algorithm and function set:
**even-4 parity 32 641**, **even-5 parity 130 081**. Our retina costs ~41 k
evaluations (800 nodes) to ~128 k (100 nodes) — i.e. for CGP the KA retina is
about as hard as **even-5 parity**. Entirely normal. (⚠️ Koza computational effort
is the evaluations for 99 % success probability, not a median-over-seeds, so these
are same-order-of-magnitude calibration, not a like-for-like comparison.) For
contrast KA's own GA needed ~1.68 M evaluations on this task.

The human intuition that "this is easy, I can draw it in 19 gates" does not
transfer: a human **decomposes**, evolution only gets a scalar hit-count that is
nearly blind to the decomposition.

### What the frames show: neutral drift is the mechanism

`runs/_viz100/` — 100 nodes, seed 1, a frame every 2000 generations
(`seed1_evolution.png`, 20 frames; `seed1_evolution_3x3.png`, 9 representative).
`runs/` is gitignored; regenerate with

    conda run -n lndp python train.py --nodes 100 --mutation-rate 0.03 \
      --task retina_ka2005 --operation and --seed 1 --n-seeds 1 \
      --generations 40000 --log-interval 2000 --viz --viz-seeds 2 \
      --save-best --out-dir runs/_viz100 --tag seed1story

Note the figures draw **only the phenotype** — the nodes on a path to the output.
The other ~77 of 100 are invisible, which is exactly why the plateau frames look
like different circuits at identical fitness.

| gen | 0 | 2 000 | 4 000 → 20 000 | 22 000 | 28 000 → 36 000 | 36 132 |
|---|---|---|---|---|---|---|
| hits/256 | 190 | 228 | **234 (flat)** | 241 | **248 (flat)** | 256 |

Seed 1 sits at 234/256 for **18 000 generations** — yet across those frames the
circuit changes continuously (active nodes 16→24, right-only nodes 3→14). Fitness
flat, structure churning. That is CGP's neutrality doing the work: ~77 of 100
nodes are inactive, mutations landing on them are free, and the genotype wanders
along the plateau until an improvement is one mutation away. The plateaus are not
stagnation — they are relocation.

This is also why bigger genotypes help: they add inactive nodes (drift room), not
capacity. The phenotype stays ~23 active either way.

### Population size and parallelism — leave (1+4) alone

Asked whether raising λ above 4 would help, given parallel hardware. It would not:

- **Wrong currency.** λ=4 costs 4 evaluations/generation (seed 1: 36 132 gens =
  144 537 evals). Doubling λ roughly halves generations and doubles their cost;
  computational effort — the measure the paper reports — stays flat at best.
- **The work is too small to parallelise.** One evaluation is ~0.18 ms of pure
  Python bigint bitwise ops; process dispatch costs more, and the GIL blocks
  thread-level gain. Coordination would exceed the work.
- **The loop is inherently serial.** Generation *t+1*'s parent is generation *t*'s
  outcome, so intra-generation parallelism caps at λ× and ~36 000 sequential steps
  remain regardless.
- **The parallelism that pays is across seeds** — 50 independent runs, no
  coordination, near-linear. The 28-min FG baseline was 50 sequential ~30 s runs.

⚠️ No source quantifies λ's effect on CGP search quality; Table II fixes (1+4)
`[verbatim]` and that is all that is claimed. If we ever want the answer it is a
measurement, not a citation.

### Infrastructure notes

- Checkpoint/resume verified by killing a run at generation 1000 and resuming: it
  continued from gen 1000 with the RNG state restored and solved at 16 570.
- Evaluating only active nodes gave 4× (1.2 → 0.30 ms/gen at 100 nodes) with
  identical trajectories.
- **Density is not logged** — a CGP genotype's edge count is fixed by construction,
  so density is a constant. The structural columns are the input-cone
  decomposition (left / right / mixed / const) instead.

---

## 2026-08-13 — FG baseline, 50 seeds (`--nodes 800 --mutation-rate 0.01`)

`retina_ka2005`, fixed goal `L AND R`, 40 000-generation cap, 28 min wall clock.

| | |
|---|---|
| solved | **46 / 50 (92%)** |
| generations to solve | median **10 318**, IQR [6 033, 17 300], min 2 588, max 28 122 |
| evaluations to solve | median **41 283** |
| the 4 failures | best accuracy 0.906, 0.938, 0.969, 0.969 |
| active nodes | median 70 of 800 (range 27–116) |

This is the reference every later arm is measured against.

### Are the solutions modular? Partly — and it depends on genotype size

Cone statistic: the fraction of active nodes that read **only one retina half**
(left-only + right-only, over all active). Compared against a null of random
unevolved 800-node genotypes.

| config | solved | median gens | median active | half-specific fraction | IQR |
|---|---|---|---|---|---|
| random genotypes (null) | – | – | 51 | **0.096** | – |
| 100 nodes / 0.03 | 10/12 | 32 028 | 23 | **0.629** | [0.522, 0.737] |
| 800 nodes / 0.01 | 46/50 | 10 318 | 70 | **0.298** | [0.222, 0.371] |

1. **Evolution builds half-specific structure well above chance under FG alone** —
   0.298 against a 0.096 null at 800 nodes. Every one of the 46 solved circuits has
   both a left-only and a right-only node; none is fully smeared. (The binary
   version of that is close to trivial — solving the task requires reading both
   halves — so the *fraction* is the number to quote, not the count.)
2. ⚠️ **The size difference may be a BLOAT ARTEFACT — do not report it as a
   structural result.** 800 nodes yields 0.298, 100 nodes 0.629, and the IQRs do
   not overlap. But the cone statistic is a *fraction of active nodes*, and the
   800-node circuits carry **70 active nodes for a 19-gate problem** against the
   100-node runs' 23. Pile ~50 redundant mixed nodes onto a modular core and the
   fraction collapses **without the core being any less modular**. The measured
   difference is therefore confounded with circuit size and is not yet evidence
   that big genotypes evolve less modular solutions.
3. **What would decide it.** Normalise against the 19-gate optimum rather than
   against the active count, or compare only the sub-circuit reachable from the
   output with redundant nodes pruned. Until one of those is done, treat the
   0.298-vs-0.629 gap as *unresolved*.
4. **Genotype size is nevertheless not a neutral knob.** ⚠️ Any FG-vs-MVG
   comparison must hold it fixed; 100-node runs are cheap (12 seeds in 178 s) so
   running both sizes is affordable.

---

## 2026-08-13 — optimisation: 3–5× serial, 4.7× parallel, and a measured null

Everything below was measured, not estimated, before being implemented.

### Where the time actually went

Profiling `run_seed` (400 nodes, 3000 gens, `cProfile`):

| | share |
|---|---|
| `cgp.mutate` | **59 %** — and almost all of it RNG: 877 621 `_randbelow_with_getrandbits` calls |
| `cgp.evaluate` | 36 % — of which the active-node walk was 31–42 % and `.tolist()` 8–27 % |

`.tolist()` was the quiet one: `evaluate` converted the *whole* genotype to Python
lists on every call — an 800×2 array to use ~65 of its rows — so cost per active
node rose 1.39 → 3.51 µs as genotypes grew 100 → 800.

### What was changed

1. **Genotype is plain Python lists, not numpy** (`conn`/`cout` flat, row-major),
   and the search RNG is `random.Random`, not `np.random.Generator`. Numpy is kept
   for I/O and summary stats only.
2. **The backward walk is fused into `evaluate`**, with gate dispatch inlined on an
   opcode (`gates.OP_*`) instead of `gate.fn(args, mask)`. The opcode keeps the
   fast path general across any `--gates` selection.
3. **`mutate` draws slots by rejection sampling into a set** using `random()`
   rather than `random.sample` + `randrange`.
4. **`phenotype()` walks active nodes only**, not all `n_nodes`. It was 1.5 ms at
   800 nodes; harmless under FG (log-interval only) but an MVG run logs every 20
   generations, where it was 9–16 % overhead.
5. **`--workers`: seeds run in a process pool**, with `train.plan_workers` choosing
   the count adaptively.

### Verified speedup (A/B against the previous commit, same harness)

4000 generations per point, (1+4), retina/AND, logging and viz off:

| nodes | HEAD (numpy) | now | speedup |
|---|---|---|---|
| 100 | 0.547 ms/gen | 0.175 | **3.1×** |
| 400 | 1.570 ms/gen | 0.437 | **3.6×** |
| 800 | 3.229 ms/gen | 0.596 | **5.4×** |

Parallel: 12 seeds × 6 workers = 124.8 core-s of work in 26.3 s wall — **79 %
efficiency, 4.7× over serial**. Combined with the serial gain that is **~20×** end
to end.

⚠️ Timing on this machine spreads ~1.4× run to run (hybrid CPU; Windows migrates
the process between P- and E-cores), so single measurements are not trustworthy —
`test_perf.py` reports best-of-3.

### Measured null: incremental fitness caching is NOT worth doing

The intuition is sound — most mutations are phenotypically silent, so why
re-evaluate? Measured, per *offspring*:

| genotype | mutations/offspring | phenotype unchanged |
|---|---|---|
| 100 nodes | 9 | 7.1 % |
| 400 nodes | 36 | 2.1 % |
| 800 nodes | 72 | **0.3 %** |

`n_mutations = rate × genotype_length`, so a bigger genotype mutates
*proportionally more genes* and the chance all of them miss the active set
collapses. Detecting "unchanged" also costs the active walk (31–42 % of an
evaluation), so the cache is near net-zero even at 100 nodes. Recomputing only
downstream of the earliest change fails for the same reason: with 9–72 scattered
mutations the earliest changed active node is near the front of the graph.

(An earlier note in this session said "~¾ of mutations are silent". That was true
per *gene*; per offspring — the unit that would let you skip an evaluation —
silence is rare.)

### GPU: no

Whole dataset = 256 patterns = **32 bytes** (4 uint64 words); one evaluation is
~250 word-ops, and a CUDA kernel launch (~5–10 µs) costs more than the entire
evaluation. The hot loop is a data-dependent DAG walk — divergent control flow,
the worst case for SIMT — and batching across seeds would force abandoning
active-only pruning (evaluate all 400 nodes instead of ~45) to buy the
parallelism. Latency-bound on 32 bytes; GPUs are throughput machines.

### Parallelism: seeds are the only axis

- **Generations are strictly serial** — generation *t+1*'s parent is generation
  *t*'s outcome.
- **Offspring (λ=4) is too small** — ~0.1 ms of work per generation, below process
  dispatch cost. Raising λ also buys nothing in the right currency: it roughly
  halves generations while doubling their cost, leaving computational effort flat.
- **Seeds are embarrassingly parallel**, capped at physical cores. SMT siblings are
  excluded deliberately: the loop is integer/branch-bound with a few-kilobyte
  working set, so there are no memory stalls for a second thread to fill.

`plan_workers` caps at seed count and physical cores, obeys an explicit
`--workers`, and falls back to serial for jobs estimated under 8 s (Windows spawns
a fresh interpreter per worker, ~0.5–1 s each). Its cost model is anchored to
*real loop* measurements — a first version fitted to a microbenchmark
underestimated 4× and silently sent a 25 s job down the serial path.

### Tests

`test_perf.py` (new) covers the three things `test_cgp.py` cannot: a throughput
ceiling per genotype size, parallel efficiency ≥ 70 %, and a table-driven check
that `plan_workers` responds to seeds/cores/job size rather than returning a
constant. It also asserts **pooled runs give byte-identical results to serial
runs**, which is the real risk in the multiprocessing change.

`test_cgp.py` gained a second `fast == slow` case: the 9-gate set exercises the
generic mixed-arity walk, but every real run uses the all-arity-2 set and its
unrolled path, which would otherwise have been untested.

Checkpoint/resume re-verified after the RNG state format changed
(`bit_generator.state` → `getstate()`): killed at gen 12 500 of 30 000, resumed,
and the final state matched the uninterrupted run exactly.

⚠️ **Seeds no longer reproduce pre-2026-08-13 runs.** The random stream changed
with the RNG. Reproducibility within the current code is unaffected.

**Not taken: PyPy.** The module is now almost entirely Python-integer bitwise work
— PyPy's best case — and experiment 4 needs no JAX, so it could plausibly gain
another 5–10× with no code change. It needs a separate environment, so it is
recorded as an option in `cgp.py`'s docstring rather than adopted. Numba is a
poorer fit: the hot loop is a data-dependent graph walk, not an array kernel.

### Open threads

- **Resolve the bloat confound** (finding 2 above) before any structural claim
  about genotype size is written down.
- **Download Miller & Smith 2006 (TEVC 10(2):167–174)** — the only source that
  closes "how much redundancy is optimal?" quantitatively.
- No ECGP yet. The genotype is already in the gene-pair encoding it needs. Both
  H1's recovery-time signature and H4's module-proliferation signature are blocked
  on it.
- The cone statistic has a null (random genotypes) but no *degree-preserving* null;
  it is a structural readout, not a modularity metric, and should not be reported
  as `Q`.

---

## 2026-08-13 — MVG arm: recovery instrumentation, the E sweep, and a step-size problem

The first MVG runs. Headline: **MVG is far worse than FG here, not better** — the
opposite of Kashtan–Alon — and the evidence points at CGP's mutation operator taking
steps far too large for a lineage to *adapt* an existing circuit to a switched goal.

### Recovery is recorded as an EVENT, not sampled

To ask "does recovery get faster with each switch?" the log interval is the wrong
instrument: it samples on a fixed grid while recoveries happen on the switch grid.
`train.py` now emits `*_seed{N}_recovery.csv`, one row per goal epoch:

```
seed,epoch,gen,goal,hits_before,hits_after,drop,gens_to_recover,censored
```

At each switch the parent is re-scored on the new goal (mandatory — fitness is
goal-dependent), the drop is recorded, and the epoch closes the generation `hits`
returns to `hits_before`, its level on the goal it was just moved off. Comparing
against `hits_before` rather than a fixed threshold measures *re-adaptation* rather
than the raw difficulty of the new goal.

**Censoring is explicit.** An epoch that ends before recovering is flagged
`censored=1` and excluded from every median and correlation. Both shortcuts are
traps that manufacture the hypothesis: dropping censored epochs biases early means
downward (early epochs fail most often), and recording them as `E` understates how
much slower they were. The result JSON carries `n_epochs`, `n_censored`,
`mean_drop`, `recovery_median`, `recovery_first_half`/`second_half` and
`recovery_rho` — Spearman of recovery time against epoch index, **negative =
getting faster**, which is the MVG prediction.

⚠️ **Bug found by the resume test, not by the feature.** Resume restored `goal` as
the goal at generation 0 rather than the goal in force at the checkpoint, so a
resumed MVG run scored the saved parent against the wrong target — diverging the
*search*, not just the log (`active_nodes` 47 vs 22 on the same seed). Pre-existing
and silent, and it would have hit every MVG run, since `--checkpoint-interval`
defaults to 1000. The goal is now checkpointed explicitly. Note it cannot be
recomputed from `gen`: the checkpoint is written at the *end* of generation
`gen - 1`, so deriving it would be off by one epoch exactly at the boundaries.

### Genotype size, FG: the full sweep (12 seeds each, `--operation and`)

| nodes | solved | median gens | median evals | median wall s | active (solved): min / median / max |
|---|---|---|---|---|---|
| 25 | **11/12** | 77 484 | 338 745 | 28.10 | **16** / 17 / 18 |
| 50 | 12/12 | 37 121 | 148 493 | 11.02 | 17 / 20 / 26 |
| 100 | 12/12 | 21 450 | 85 809 | 6.70 | 18 / 23 / 30 |
| 400 | 12/12 | **9 812** | **39 259** | **2.48** | 21 / **47** / 65 |

Two monotone trends, in opposite directions, over a 16× range of genotype size:

- **Search speed rises with genotype length.** 400 nodes solves in 8× fewer
  generations than 25 and 2.2× fewer than 100. This is the neutral-drift effect the
  ECGP paper describes, and it is now measured at four sizes rather than asserted.
- **Circuit bloat rises with it too.** Median solved circuit goes 17 → 20 → 23 → 47
  active nodes. At 400 nodes the median solution is **2.5× the size of the
  hand-derived 19-gate construction**; at 25 nodes it is *below* it.

25 nodes is where it starts to break: 11/12, and 3× slower than 50 for circuits only
1 gate smaller. **50 is the knee** — full solve rate, near-minimal circuits.

⚠️ **The 19-gate construction is not optimal.** The smallest *solved* circuit found
was **16 active nodes** (`--nodes 25`), with 17 the median at that size — three
gates below the construction. That construction was only ever an upper bound on the
minimum, and it is now known to be a loose one. (Beware the raw `active_nodes`
minimum: the unsolved 25-node seed ended at 13 active nodes and is not evidence
about anything. Only `final_hits == 256` circuits count.)

**Choosing on wall time was a mistake.** 400 nodes is fastest and was recommended on
that basis; it is also the worst on parsimony (a hard project constraint), the worst
on step size (below), and it is `[our choice]` — `--nodes 100` is the paper's
`[Table II]` value.

### E=200 does not work, and the failure is diagnostic

4 seeds × 200 000 generations at 400 nodes — 20× what FG needs to solve:

| epoch band | n | censored | median recovery | median `hits_before` |
|---|---|---|---|---|
| 0–198 | 796 | 38% | 71 | 212 |
| 199–398 | 800 | 37% | 72 | 212 |
| 399–598 | 800 | 39% | 74 | 212 |
| 599–798 | 800 | 39% | 70 | 212 |
| 799–998 | 800 | 37% | 75 | 212 |

**0 of 3996 epochs were entered from a perfect circuit.** `hits_before` is pinned at
**212/256 = 0.828** — the one-side shortcut plateau — for the entire run, and
censoring is *flat*, not front-loaded, so a longer budget is not the fix. MVG at
E=200 never solves the task at all, while FG at the same size solves 12/12.

### E=2000 half works (16 seeds × 800 000 gens, 400 nodes)

E=2000 is the evaluation-matched KA anchor: their E=20 generations × 600
individuals ÷ 5 evals per (1+4) generation ≈ 2400 `[our choice, derived]`.

|  | E=200 | E=2000 | FG (same size) |
|---|---|---|---|
| best hits, median over seeds | 212 (plateau) | **250** | 256 |
| seeds solved | 0/4 | **5/16** | 12/12 |
| median gens to first solve | never | 325 217 | **9 812** |
| censored epochs | 38% | 46% | — |

E was the binding constraint: the lineage escapes the shortcut plateau and five
seeds reach 256/256. But:

- **MVG is 33× slower than FG** and solves 5/16 against 12/12. Opposite of KA.
- **A solution never survives a switch.** 7 of 6384 epochs began perfect, and
  **100% of those 7 were censored** — every seed that reached 256/256 was knocked
  off by the switch and never climbed back within the epoch.
- **45% of every epoch is spent clawing back to where it already was** (median
  recovery 897 of 2000 gens). The drop is structural: a perfect AND circuit scores
  exactly 128/256 on OR, so and↔or is near-maximally adversarial at the output.
- **No recovery trend.** Per-seed ρ median −0.021, 11/16 negative — p ≈ 0.11 under
  a coin-flip null, i.e. noise. `hits_before` flat at 224–225 across all five epoch
  bands: a higher plateau, but still a plateau.

### Smaller genotypes do NOT rescue MVG (16 seeds × 800 000 gens each)

The step-size argument below predicts that 100 nodes — 9 mutated genes/offspring
instead of 36 — should adapt across a switch better than 400. It does not.

| nodes | E | solved | best hits (median) | epochs entered perfect | censored | median recovery | ρ (median) |
|---|---|---|---|---|---|---|---|
| 400 | 2000 | **5/16** | 250 | 7/6384, all censored | 46% | 897 (45% of E) | −0.021, 11/16 neg |
| 100 | 1000 | **0/16** | 237 | **0/12784** | 40% | 224 (22% of E) | −0.022, 12/16 neg |
| 100 | 2000 | **0/16** | 242 | **0/6384** | 43% | 558 (28% of E) | +0.001, 8/16 neg |

Holding E=2000 and dropping 400 → 100 nodes takes the solve rate from 5/16 to
**0/16**, even though FG at 100 nodes solves 12/12. So the smaller step *does* what
it was predicted to do locally — recovery takes 22–28% of an epoch instead of 45% —
and still never assembles a full solution. `hits_before` is flat at 215–217 across
all epoch bands in both runs: another plateau, slightly above E=200's 212.

**No recovery trend anywhere.** Across all three arms the per-seed Spearman ρ hovers
at zero (−0.022, +0.001, −0.021) with 8–12 of 16 seeds negative — indistinguishable
from a coin flip. The H1 signature is absent in plain CGP, which is what it should
be if module retention is the thing that produces it.

**`0` epochs entered from a perfect circuit, in 25 552 epochs across all arms.** That
is the cleanest control statistic we have: plain CGP *never* carries a solution
across a goal switch. It is a much stronger baseline than a recovery-time
distribution, and it is precisely what ECGP's compress/expand is supposed to change.

Interpretation: genotype length is not the binding constraint on MVG. Smaller
genotypes buy compact circuits and faster local recovery but lose the neutral-drift
reservoir that FG search depends on, and the switch arrives before the lineage has
converted either advantage into a solution.

### The likely cause: the mutation operator cannot make small moves

> ⚠️ **Superseded 2026-08-14 — this hypothesis was tested directly and falsified.**
> Biasing mutation away from rewiring (`--wiring-weight`) shrinks the step and makes
> MVG *worse*, monotonically. See "The step-size hypothesis, tested and falsified"
> below. The measurements in this section stand; the causal reading does not.

`--mutation-rate` is a *fraction of the genotype*, so absolute step size scales with
node count:

| nodes | genes | genes mutated/offspring | active nodes | ≈ active genes | ≈ active hits |
|---|---|---|---|---|---|
| 100 | 301 | 9 | 22 | ~67 | ~2 |
| 400 | 1201 | 36 | 47 | ~142 | ~4 |

Measured per-offspring phenotypic silence was **7.1% / 2.1% / 0.3%** at 100/400/800
nodes: at 400 nodes **98% of offspring are a different circuit**. There is no way to
propose "flip one gate". Poisson estimate of the chance an offspring changes exactly
one specific active gene and nothing else in the live circuit: ~1 in 2000 at 400
nodes vs ~1 in 190 at 100. With 4 offspring/gen that is ~500 generations merely to
*propose* the surgical edit an and→or switch mostly needs.

The E=200 logs show the consequence directly — active-node count over consecutive
epochs on one seed went **22 → 13 → 16 → 15 → 13 → 10 → 8** while hits sat at
208–212. That is not a circuit adapting its combiner; it is a circuit being rebuilt
from scratch each epoch and drifting toward smaller degenerate shortcuts.

Two knobs are conflated here: **node count controls both compactness and step
size.** Holding nodes at 400 while dropping `--mutation-rate` to ~0.0075 would give
the same ~9 genes/offspring with the bloat intact, separating them — at the cost of
taking mutation rate off `[Table II]`.

### On penalising circuit size — a tension to respect

The obvious parsimony fix (among ties, prefer fewer active nodes) attacks the thing
that makes CGP work. The tie-break is `rnd.choice(ties)`, i.e. neutral drift, and
the frame-by-frame analysis above identified it as **the escape mechanism** (fitness
flat at 234 for 18 000 generations, then 241 → 248 → 256). Replacing a random walk
along the neutral plateau with a directed one may remove the escape. Ranked, least
invasive first: (1) cap nodes structurally — free, and back on `[Table II]`;
(2) weak *stochastic* parsimony, keeping the random tie-break but biasing it to the
smaller circuit with small probability *p*; (3) `hits − λ·active` in the fitness —
avoid: it changes the reported metric and this repo already has a cautionary case,
the `balanced` fitness that "manufactures a gradient into the one-module solution".

### Open threads from this section

- **Is the MVG failure step size or bloat?** Genotype size is now ruled out as the
  lever (100 nodes is *worse*, 0/16). The remaining decomposition run is 400 nodes
  with `--mutation-rate 0.0075` — same ~9 genes/offspring as 100 nodes, but with the
  drift reservoir kept. That separates step size from genotype length, at the cost
  of taking mutation rate off `[Table II]`.
- **Does anything ever retain a solution across a switch?** **0 of 25 552 epochs**
  across all three MVG arms. That zero is the number ECGP has to beat, and it is a
  better control statistic than any recovery-time distribution.
- **Choose the FG reference config on parsimony, not wall time.** 50 nodes is the
  knee: 12/12 solved, median 20 active nodes, 11 s per 12 seeds.
- `--popsize` is the next suspect after E and step size: KA ran 600 individuals with
  crossover, so a switch cost them *diversity*; our (1+4) is a single lineage, so a
  switch hits the whole population. Changing it departs from `[Table II]` and should
  wait until the E and step-size responses are mapped.

## 2026-08-14 — MVG never solves the retina; the step-size hypothesis is falsified

**Headline.** Across every configuration tried (nodes 50/100/400, E=200/1000/2000,
three mutation mixes) **MVG never solved the Kashtan–Alon retina under (1+4) CGP** —
best arm 5/16 seeds (400 nodes, E=2000), every other arm 0/16. The *same code* under
FG solves it **12/12 at every genotype size**, median 37 000 generations (11 s for
12 seeds) at 50 nodes.

So MVG is not encoding-agnostic. The protocol that *improves* KA's networks
(`kashtan_alon/RESULTS.md` Run 5: MVG 0.975 vs FG 0.904) strictly degrades ours:
**not all frameworks are amenable to a KA-type approach**, and the KA result is not
portable across encodings without checking.

### The switch is NOT "too disruptive" — that reading is wrong

`d(and, or) = 128` of 256 patterns (computed from `tasks.target_mask`): the goals
differ on **exactly half the truth table**, so a *perfect* AND circuit scores chance
the instant the goal becomes OR. That maximum disruption is paid by the ideal
modular solution too — KA's own networks take the identical hit every 20 generations
and still beat their FG control. The measured median drop was **112/256 in all three
mutation arms, identical**: a property of the goal pair, not of the encoding.

MVG's claim is not that a switch is survivable but that it is *repairable by one
gate if the representation is modular*. Ours never become modular, so the mechanism
never engages. **The failure is upstream of the switch.**

### E is already evaluation-matched to the paper — stop sweeping it

| | gens per epoch | evals per gen | **evals per epoch** |
|---|---|---|---|
| Kashtan–Alon 2005 `[verbatim]` | 20 | 600 | **12 000** |
| this experiment, E=2000 `[our choice, derived]` | 2000 | 5 | **10 000** |

Within 20% of the paper. E is not the explanation and needs no further sweeps.

### The step-size hypothesis, tested and falsified

Hypothesis (previous section): mutation is too coarse to propose "flip one gate", so
the circuit is rebuilt each epoch. Test: make rewiring rarer, leave gate swaps alone.
New flag **`--wiring-weight w`** `[our choice, experimental]` — relative sampling
weight of a wiring gene (node input, or the output gene) against 1 for a function
gene. `w = 1.0` is the paper and takes the original code path, verified
bit-identical; sequential rejection in `cgp._draw_slots_biased`.

50 nodes, and↔or, E=2000, 16 seeds × 800 000 generations, identical seeds; `w=1.0`
doubles as our first **50-node MVG control**.

| arm | % of mutations hitting a function gene | solved | best hits med (max) | best acc | active med | epochs entered perfect | censored | med recovery |
|---|---|---|---|---|---|---|---|---|
| **w=1.0** (paper) | 33% | 0/16 | **234** (240) | 0.915 | 10 | 0/6384 | 38% | 329 |
| **w=0.25** | 66% | 0/16 | **232** (240) | 0.910 | 7 | 0/6384 | 34% | 210 |
| **w=0.0625** | 89% | 0/16 | **228** (234) | 0.889 | 8 | 0/6384 | 28% | 108 |

Shrinking the step 16-fold (~0.5 rewires per offspring) made it **worse,
monotonically, with no optimum in between**. Step size is not the binding constraint.

⚠️ **Faster recovery here is not a win.** Recovery is measured against `hits_before`,
so a lineage at 228 has less to climb back to than one at 234 — it recovers faster
*to a lower bar*, while the ceiling falls with it. Check any recovery speed-up
against its ceiling before believing it.

### Under MVG, bigger genotypes are better — the opposite of FG

With the new 50-node control (all E=2000, 16 seeds):

| nodes | best hits med | solved |
|---|---|---|
| 400 | 250 | 5/16 |
| 100 | 242 | 0/16 |
| 50 | 234 | 0/16 |

**Monotone, and the opposite direction from FG**, where 50 is the knee and 400 is
pure bloat. Consistent with neutral drift being the only escape available: more
inactive material to drift through matters when the goal keeps moving and not at all
when it does not. This puts parsimony in **direct tension with MVG performance** —
state it in the writeup whichever way ECGP goes.

### What is still open after this

- **Retention has never been observed.** Across all six MVG arms, **7 of 44 704
  epochs** entered from a perfect circuit — all in the 400/E2000 arm, **all
  right-censored**, none ever re-solved. Decisive test: seed MVG from a solved FG
  circuit (needs `--init-from` reading a `_best.npz`; ~10 min). Splits "cannot find
  a solution under MVG" from "cannot hold one".
- **`--popsize` is now the largest untested confound.** (1+4) is a single lineage
  with no archive — `best_geno` is passive, never fed back into selection — so a
  switch wipes everything; KA's 600 + crossover cost them only diversity. "CGP fails
  under MVG" is entangled with "(1+4) fails under MVG". Departs from `[Table II]`.
- Closed, do not revisit: E sweeps (evaluation-matched), genotype-size sweeps
  (monotone), mutation-operator tweaks (falsified).

### Measurement caveat

The `w=0.25` arm reported 4288 s wall against ~400 s for the others. **Not
algorithmic**: per-generation cost is the same in all three arms (p50 353/372/431 µs)
and one logging interval in every seed absorbed ~3700 s — the machine stalled for
about an hour mid-arm. Noted because the raw summary line is misleading on its own.

## 2026-08-14 — ECGP implemented and verified (no experiment run yet)

`ecgp.py` + `--ecgp`. Everything the paper states is implemented as stated; the 14
cases its prose leaves open are listed in **PAPER_SPEC section 12**, each tagged
`[our choice]` and commented at its site. No `[inferred]` entries exist, so the
"no code while anything is inferred" rule still holds.

**Why a separate module from `cgp.py`.** A CGP node has fixed arity, so `conn` is one
flat row-major list — the layout the hot loop depends on. An ECGP node's arity is
whatever its module needs (2..2·ms). Rather than make `cgp.py` variable-arity and
re-validate every number already in this file, ECGP got its own representation and
`cgp.py` was left alone. The two meet in `flatten()`, which inlines every module into
an ordinary CGP genotype — so **both arms report active nodes, cone classes and depth
through the same `cgp.phenotype`**, and the comparison is not confounded by
measurement.

### Verification

| check | result |
|---|---|
| CGP arm unchanged by the backend refactor | **bit-identical** on 4 seeds — same `solved_gen`, `evals`, `active_nodes`, cone counts |
| `evaluate` == `cgp.evaluate(flatten(·))` | 60 evolved individuals, 881 modules |
| `compress` / `expand` fitness-neutral | 48 / 187 applications, no behaviour change |
| `expand(compress(g)) == g` as a graph | 100 round trips, identical func/conn/ogene |
| type I function gene immune | 30 rounds at mutation rate 1.0 |
| structural invariants (section 12 `validate`) | 400 mutation rounds × 4 values of `ms` |
| module bounds hit their ceilings | input ceiling and output ceiling both reached |
| prune-on-promotion | list == exactly the modules the winner uses |
| ECGP kill-and-resume | exact, including the module dict and the id counter |

⚠️ The refactor A/B had to be built by hand: `git HEAD` still carries the numpy
`default_rng`, a *pre-existing uncommitted* rewrite, so comparing against HEAD
measures that switch and not the change under test. The valid comparison reverts only
the four backend call sites in a copy of the current tree.

Two bugs the tests caught, both in cases the paper does not discuss (now section 12
entries 4 and 5): a node whose **output count shrinks** leaves later references
pointing at an output that no longer exists, and a drawn input slot can be **retired
mid-application** when an earlier drawn function gene shrinks that node's arity.

### Smoke runs only — not results

| run | seeds | outcome |
|---|---|---|
| FG, 50 nodes, `--operation and` | 4 | **4/4 solved**, median 25 028 gens, 5-6 modules in the final individual |
| MVG, 50 nodes, E=2000, 60 000 gens | 4 | 0/4, best 232/256, 6 modules, 29 epochs logged, recovery CSV written |

⚠️ Neither is a result. n=4, the FG figure sits against a CGP median of 37 121 gens
measured on **12** seeds, and the MVG run is 60 000 generations against the 800 000
the CGP arms had. Both exist to show the paths execute end to end. The real
comparison needs matched seeds and budgets.

### Process note

Three things in this file were settled by sweeping when the answer was already in
the paper on disk (genotype length), or by asserting when nothing had been
verified (the 800 recommendation). The habit to keep: **grep the PDF before
running the sweep**, and tag any number that came from a sweep rather than a
source as `[our choice]` at the moment it is written, not afterwards.

## 2026-08-14 — ECGP vs CGP on FG50: the first real comparison

12 matched seeds, identical budget and configuration, the *only* difference being
`--ecgp`. Data in `runs/_fg50cmp/`.

| | `--nodes 50 --mutation-rate 0.03 --generations 300000 --operation and`, fixed goal |
|---|---|

| arm | solved | gens (median) | evals (median) | active nodes (median) | core-seconds (total) |
|---|---|---|---|---|---|
| CGP  | 12/12 | 37 121 | 148 493 | 19.5 | 131.1 |
| ECGP | 12/12 | **24 549** | **98 207** | 24.5 | 159.6 (340.8 before the speed work) |

**ECGP solves in fewer generations on 9 of 12 seeds.** Per-seed ratio spans
0.24 (seed 5: 20 332 vs 84 986) to 1.55 (seed 7: 52 880 vs 34 057), so the effect is
a shifted distribution, not a uniform win — and n=12 with that spread is suggestive,
not significant. It reproduces the *direction* Walker & Miller report; no claim
beyond that.

ECGP's circuits are consistently **larger** (24.5 vs 19.5 active nodes). Inlining a
module copies its body at every call site, so a reused module is cheap in the genotype
and expensive in the flattened phenotype. Any later comparison of solution *size*
between the arms has to say which of the two it is counting.

### Why ECGP might be ahead — four mechanisms, weights unknown

1. **A wider function alphabet.** A function gene can take a module id, so one point
   mutation substitutes a whole sub-circuit where CGP needs several coordinated ones.
2. **Protection.** `compress` is fitness-neutral but changes the *mutational
   neighbourhood*: a type I node's function gene is immune and its body only mutates at
   0.04. Found structure is frozen while the rest keeps searching.
3. **A shorter genotype.** `compress` shrinks it, and 3% of a shorter genome is fewer
   mutations per offspring — a step-size change, not a representation change.
4. **Genuine reuse.** The retina optimum contains the same half-detector twice; a
   module pays for it once.

The module census below argues (2) and (3) over (4). Separating them needs the H4
knock-out control in the README (swap a module instance for a random one of the same
shape; fitness must drop) — **not yet run**.

### Module census: most "acquired modules" are degenerate

11 172 modules observed across the 12 ECGP seeds, 17 936 module-rows at log points
(`*_gates.csv`, one row per gate per log point, with a canonical signature).

| observation | count |
|---|---|
| modules computing exactly one primitive | 3 437 (31%) |
| trivial (every output constant, a wire, or NOT) | 2 476 (22%) |
| distinct canonical functions among all 11 172 | 1 254 |
| modules seen at only one log point (lifetime 0) | 7 078 (63%) |
| log points where a module appears >1× in the active circuit | 10 549 / 17 936 |
| peak reuse | `M964` ×10 (an AND) |

So the module list is mostly churn. It is **not** evidence of an acquired functional
vocabulary on its own. Real examples exist — seed 7's solved circuit contains an XOR
module, `(AND (OR i0 i1) (NAND i0 i1))`, used 3× — but they are the minority.

⚠️ Lifetime is a **lower bound and partly an artefact**: `add_input` / `add_output`
renumber a module by the paper's rule, so a modified module looks like a death plus a
birth. Distinguishing them needs a lineage field on `Module`; until then do not quote
the lifetime figure as a decay rate.

### Instrumentation added

- **Per-log gate census** → `<run>_seed<k>_gates.csv`: one row per gate (primitive or
  module) with active count, genotype copies, node/input/output counts, a human label,
  a **canonical signature**, and an s-expression. Module ids come from a
  never-rewound counter, so a name is bound to one act of creation, as asked.
- **Canonical signature** = truth table with dummy inputs dropped and inputs relabelled
  to the lexicographic minimum ⇒ equal signature iff same function up to input
  relabelling. Two modules that both behave like XOR are detectable as such
  (`test_ecgp.py` builds XOR three ways and all give `2:6`). Made cheap by permuting
  only inputs of **equal influence** (exact, usually leaves one candidate).
- **Module provenance in the drawing.** `flatten_with_origin` tags each flattened node
  with the module instance it came from (`M2625#1`); `visualize.draw(origin=...)`
  prints it under the gate, coloured **per module** — so the same colour twice in one
  picture is one acquired function used twice. Flattening is exactly what erases module
  identity, which is why the plain drawing could never show it.
- **Stage grid** (`--grid-seed`, default on): 9 evenly spaced snapshots of *one* seed
  through generational time, decimated during the run (halve the resolution, keep the
  span), so any run length costs ~9 renders at the end.
- `--viz-seeds` **default changed 1 → 0.** A per-log PNG over a 300 000-generation
  budget is ~3000 matplotlib renders and dominated the run; the stage grid replaces the
  need.

Cost of the census: 2.15 ms cold / 0.55 ms warm per log point against 0.80 ms per
generation ⇒ ≤2.7% at `--log-interval 100`, ~27% at interval 10. Negligible as used.

### ECGP speed: 2.14× on the real run, bit-identical

cProfile put `point_mutate` at 1.014 s of 3.859 s with 4.29 M `len()` calls: the
mutable-slot table was being built **twice per offspring** (once to count slots, once
to index them). Four changes — build it once with `itertools.accumulate` and pass it,
`sum(map(len, ·))` for the count, `n_outputs_of` inlined at its two hot sites,
`copy` via `map(list, ·)` — gave **0.804 → 0.273 ms/gen (2.9×)** and **340.8 → 159.6
core-seconds** on the 12-seed run.

No random draw was added, removed or reordered, and the verification is that claim
made testable: re-running all 12 seeds reproduces `solved_gen`, `best_hits`, `evals`,
`active_nodes`, cone counts, `n_modules`, `module_nodes`, `genotype_nodes`,
`module_sizes` and the gate string **exactly**.

I had earlier blamed ECGP's wall-clock loss on the module evaluator. That was wrong:
`evaluate` is **0.5×** CGP's cost (7.9 vs 14.5 µs) because a compressed genotype has
fewer top-level nodes. The entire gap was `mutate` (38.6 vs 4.7 µs).

### Open threads from this section

- **H4 knock-out control** — the one measurement that would separate "modules are
  functional units" from "compress is a step-size trick". Unbuilt.
- **NAND-only function set** (`--gates nand`) — would remove the 31% re-encapsulated-
  primitive confound outright, since with one gate any non-trivial module is genuine
  composition. Caveat to control first: with one primitive a **CGP** function-gene
  mutation is a no-op, so ~1/3 of its mutations do nothing and all mutation becomes
  rewiring — the arms stop being step-size-matched by construction (ECGP's function
  gene can still swap to a module id). Report non-null mutations per offspring
  alongside, and note the 19-gate minimal circuit above no longer applies.
- **Module lineage field**, to make the lifetime figure mean what it appears to mean.
- **MVG comparison** — the actual point of experiment 4. Nothing here is that; FG50 is
  the sanity check that ECGP works at all before asking whether it *inherits* rather
  than rediscovers modularity across goal switches.

---

## 2026-08-14 — how an ECGP circuit is drawn (convention, not a result)

The standard picture of an ECGP individual is now the **unflattened genotype**:
`visualize.draw_modular`, one box per genome node, a module call a **single box** with
one input port per parameter and one output port per module output, arrows landing on
the individual ports and short tick marks marking them. Colour is module identity, so
the same colour twice in one picture is one acquired function re-used twice.

This replaces the flattened drawing as the default because flattening is precisely what
destroys the thing the run is being read for. In the flattened view a 2-output module
is inlined into its body gates, so **one call site becomes several boxes carrying the
same tag** (`M687#6` twice is not two uses — it is one use, opened up). That is
unreadable as a module map and generated three rounds of confusion before it was
replaced.

Where it applies: per-log frames, the final frame (`frames/seed<k>_final.png`) and the
2×3 stage sheet (`seed<k>_stages.png`) — all unflattened for the ECGP arm. The
flattened twin survives for the final circuit only, as `frames/seed<k>_final_flat.png`,
because it is the graph that actually gets evaluated. The CGP arm is unchanged: with no
modules the two views coincide, and cone colouring (`--colour-cones`) still applies
there.

Two things the modular view deliberately does not do. It draws the nodes active in the
**genotype** graph, which is not the flattened active set (a module output can be dead
while its box is alive), so the node counts in the two views will not agree. And it
offers no cone colouring: a module box has several outputs with possibly different
cones, so there is no single left/right/mixed class to paint it with.

Regression check for any change here: ECGP seed 7, FG-and, 50 nodes, m=0.03 must still
report `solved_gen = 52880`.
