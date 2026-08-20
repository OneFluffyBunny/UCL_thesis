# Experiment 5 — specification

> **Status: skeleton.** The *representation* sections below are written, because code
> exists and a spec that does not describe the code is worse than none. Every section
> concerning the **hypothesis, the task choice and the protocol** is still TODO — that
> is the science, and it has not been decided.

Experiment 5 is the **big-brain** arm: many-input / many-output circuits, evolved with
the same CGP and ECGP machinery as `../experiment_4/`, asked whether **behavioural
modularity** emerges. Provenance tags follow the project convention and are used the
same way as in `../experiment_4/PAPER_SPEC.md`:

* `[verbatim]` — stated in a source, quoted or paraphrased faithfully.
* `[inferred]` — not stated, but forced by something that is.
* `[our choice]` — free parameter, chosen here, with the reason.

Everything inherited from experiment 4 keeps the tag it had there. Only the deltas
are restated below.

---

## 1. Representation `[inherited]`

Unchanged from `../experiment_4/PAPER_SPEC.md` §1 and §4: one row of nodes, no
levels-back, node `j` labelled `n_in + j` may reference any label below its own; genes
stored as pairs so ECGP's module nodes fit the same genotype; modules hold primitives
only, no nesting.

**Delta — program outputs.** `[our choice]` A genotype carries `O` output genes rather
than one, where `O` is fixed by the task. They are initialised, per Walker & Miller, to
the last `O` nodes of the genotype, and each is a mutable gene slot, so the mutation
budget `round(rate * n_gene_slots)` now includes them: `n_nodes * (1 + arity) + O`.

## 2. Parameters

Table II defaults carry over unchanged (`[verbatim]`): mutation rate 0.03, genotype
point mutation probability 1, (1+4) ES, module list starts empty, `compress` 0.1,
`expand` 0.2, module point 0.04, add/remove input 0.01/0.02, add/remove output
0.01/0.02. `ms = 5` stays `[our choice]`.

TODO — the parameters experiment 5 actually sweeps, and their justification.

## 3. Evolutionary strategy `[inherited]`

(1+4) ES with the neutral tie-break, exactly as `../experiment_4/PAPER_SPEC.md` §3.
Not re-derived here; `test_equivalence.py` asserts the two implementations run it
identically.

## 4. Task family

Implemented (see `tasks.py` and the README's table): `retina_xN`, `addN`, `multN`,
`parityN`, plus experiment 4's `retina_ka2005` / `left` / `and2` / `copy`.

* The retina rule is `[verbatim]` Kashtan & Alon 2005 Fig. 5a, and `test_tasks.py`
  asserts the mask-algebra re-expression equals `kashtan_alon/tasks.py` bit for bit.
* `retina_xN` — `[our choice]` N independent copies, one program output each. The
  positive control: the task's decomposition is exact and known.
* `addN` / `multN` — `[our choice]`, but not arbitrary: adders and multipliers are
  Walker & Miller's own ECGP benchmarks, which makes any module the search discovers
  comparable against a literature that already describes what modules those problems
  contain.
* `parityN` — `[our choice]` the negative control: no decomposition exists, so a
  behavioural-modularity score that rises here is measuring an artefact.

TODO — **which** of these is the experiment, at what size, and why. Nothing below
should be read as settled until this section is.

## 5. Fitness

`[our choice]` Score = correct `(output, pattern)` pairs summed over all `O` outputs
and all patterns; a perfect score is `O * n_patterns`. Summed rather than averaged so
the score stays an **integer**: the (1+4) ES tests `>` and `==` on it, and the `==`
branch is the neutral drift CGP actually searches with, so a float score would make
that mechanism fire erratically.

`[inherited]` Evaluation is exhaustive over every input pattern, bit-parallel, exact.

TODO — whether per-output weighting is ever wanted (e.g. an adder's carry bit is
"harder" than its low bit), and the argument either way. Currently every output counts
equally, which is the assumption-free default and is what makes `hits` comparable
across tasks.

⚠️ **Open, and load-bearing.** Exhaustive evaluation caps the experiment at ~20 inputs
(`RESULTS.md` §1). Going wider requires scoring a sampled subset of patterns, which
makes fitness stochastic and needs an explicit decision about how selection and the
neutral tie-break behave under noise. Not built; not decided.

## 6. Function set `[inherited]`

AND / NAND / OR / NOR, Table II `[verbatim]`. `--gates` accepts others and each is a
deviation, for the reason `../experiment_4/gates.py` records: adding `xor` hands the
retina its combiner for free.

## 7. The measurement `[our choice]`

Two readouts, both defined on the flattened circuit so CGP and ECGP are measured by
identical code:

* **structural** — a node's class is the input group its whole cone lies in, else
  `mixed`, else `const` (empty cone). Reported per class, plus `out_pure`, the number
  of program outputs whose cone stays in one group.
* **behavioural** — `beh_pure`, the number of program outputs whose *support* (inputs
  that provably change it) lies in one group. Exact, not sampled.

TODO — the statistic that the hypothesis is actually tested on, and its null. Counting
pure outputs is a description, not yet a test: a null model is needed for "how many
pure outputs would a random circuit of this size have?", and the honest options are a
degree-preserving rewire or random genotypes through the same encoding (which is what
Kashtan & Alon's second null did — see memory `reference_ka_modularity_metric`).

## 8. Deliberate deviations from experiment 4

Listed with reasons in `README.md`'s comparison table; all six are asserted equivalent
or documented-and-tested. The one that changes results is the slot draw order, and
`test_equivalence.py` demonstrates it is the only one.

## 9. Implementation decisions not fixed by any source — all `[our choice]`

* Input groups per task (`tasks.input_groups`) — stated per family in `README.md`.
  `multN`'s two-operand grouping is explicitly *not* a modular decomposition and is
  labelled as such at the point of use.
* Group class names: `left`/`right` at K=2 for experiment-4 compatibility, `g0..gK`
  above. Cosmetic, but it keeps two experiments' CSVs readable by one script.
* Genotypes saved as JSON rather than `.npz`, because numpy cannot be imported under
  PyPy and a genotype is six lists of small integers.
* PyPy 7.3.23 / Python 3.11 pinned in `setup_pypy.py`, so the "PyPy == CPython" claim
  is a claim about a specific build.
