# SPEC metric calibration — does the statistic measure what it claims?

> ## ⚠️ READ THIS FIRST — provenance
>
> **Designed, run and interpreted by Claude Code (Opus 5) on its own initiative**, at
> the user's invitation to do research autonomously. **Not reviewed by a human.**
> Quarantined here on purpose; do not cite from it. Third file in a quarantined set with
> [`SPECIALISATION.md`](SPECIALISATION.md) (where SPEC was defined and first used) and
> [`ENTRENCHMENT.md`](ENTRENCHMENT.md).
>
> Run 2026-08-28 on branch `spec-modularity`.

---

## 0. Why this exists, and why it should have come first

`SPECIALISATION.md` ran a hypothesis test on SPEC. `ENTRENCHMENT.md` ran a second study
downstream of it. **Neither ever checked that SPEC has room to move.** That is the wrong
order, and this file is the correction.

The methodological error, stated plainly: *a hypothesis test on an outcome variable whose
dynamic range is unknown cannot be interpreted, because a null is indistinguishable from
a metric that was pinned to its endpoint the whole time.*

This is **not a hypothesis test.** It is a measurement of the instrument. Both possible
outcomes are useful: either SPEC discriminates known-different structure, or it does not
and everything built on it needs re-reading.

## 1. What SPEC is

From `cgp.specialisation`: perturb each active node (invert its truth-table mask,
repropagate) and record which program outputs move.

* **specialised** — the node moves exactly one output
* **pleiotropic** — it moves more than one
* **excluded** — it moves none (behaviourally inert whatever the wiring says)

`SPEC = specialised / (specialised + pleiotropic)`.

Because inert nodes are *excluded*, SPEC is immune to the bloat confound that
`experiment_4/RESULTS.md` flags against its own cone statistic — padding a modular core
with redundant nodes cannot dilute it. That property is what made SPEC look worth
calibrating rather than discarding.

## 2. The two legs

**Leg 1 — the random-genotype null.** SPEC on *unevolved* 100-node genotypes, 300 per
task. No search involved; this is the substrate's own baseline. It answers: *what range
can SPEC move within at all?*

**Leg 2 — ground truth.** Three tasks with an **identical shape** — 8 inputs, 2 outputs,
same node budget, same gate set, same random null — differing *only* in how much their
two outputs are forced to share:

| task | target | forced sharing | SPEC should be |
|---|---|---|---|
| `pair_zero` | `O1 = L`, `O2 = R` | none | **highest** |
| `pair_partial` | `O1 = L`, `O2 = L XOR R` | partial | middle |
| `pair_full` | `O1 = O2 = L` | total | **lowest** |

Holding shape fixed is what makes this a test of the *metric* rather than of the tasks —
the cross-task incomparability warning in `cgp.specialisation`'s docstring applies to
tasks of different shape, which is why the ordering claim is confined to these three.

`add4`, `mult4` and `retina_x2` are run alongside to see whether the picture survives at
more outputs and more inputs. `retina_x3` was **excluded on feasibility**: 24 inputs is
2 MB per wire, past the ~20-input truth-table cap recorded in `CLAUDE.md`.

**Criterion, fixed before leg 2 ran:** SPEC passes if the three medians are ordered
`pair_zero > pair_partial > pair_full` on solved circuits, with the extremes separated.

---

## 3. Leg 1 result — the null is near the CEILING

300 random 100-node genotypes per task. This leg is deterministic and needed no search.

| task | in | out | median SPEC | p25 | p75 | median influencing nodes |
|---|---|---|---|---|---|---|
| `pair_full` | 8 | 2 | **0.9167** | 0.833 | 1.000 | 15.5 |
| `pair_partial` | 8 | 2 | **0.9167** | 0.833 | 1.000 | 15.5 |
| `pair_zero` | 8 | 2 | **0.9167** | 0.833 | 1.000 | 15.5 |
| `add4` | 9 | 5 | 0.7222 | 0.647 | 0.795 | 31.0 |
| `mult4` | 8 | 8 | 0.6373 | 0.560 | 0.697 | 42.0 |
| `retina_x2` | 16 | 2 | **0.9474** | 0.867 | 1.000 | 14.0 |

**This is the headline, and it is bad news for how SPEC was used.**

1. **An unevolved random circuit scores 0.92.** There is almost no headroom above chance.
   `SPECIALISATION.md`'s hypothesis was that staging would *raise* SPEC — it was looking
   for an increase in a statistic already pinned near its maximum by the substrate. That
   study was close to unfalsifiable in the direction it was testing, and the calibration
   that would have revealed this was never run.
2. **The null is set by output count, not by organisation.** 0.92 at 2 outputs, 0.72 at
   5, 0.64 at 8. The mechanism is visible in the last column: a random genotype's outputs
   hang off small, mostly-disjoint cones, so nearly every node moves exactly one output.
   Fewer outputs → less cone overlap by chance → higher SPEC. **Nothing about modular
   organisation is involved.**
3. **The null is identical for all three `pair_*` tasks (0.9167 to four figures).** It has
   to be — a random genotype never looks at the target. That is a feature for leg 2: the
   three arms share a baseline exactly, so any ordering among them is attributable to
   what evolution did.
4. **Evolution moves SPEC DOWN, not up.** Solved `pair_partial` circuits scored ~0.63 in
   `SPECIALISATION.md` against this 0.92 null. Solving a task *couples* outputs, because
   shared sub-circuitry is re-use and re-use reads as pleiotropy.

**Consequence for the sign convention.** SPEC is, in practice, an inverse measure of
*sharing between outputs*. "High SPEC" is the state of a random circuit; the informative
quantity is **how far below the null a solved circuit sits**. Leg 2 tests whether that
distance tracks ground truth.

---

## 4. Leg 2 result — the metric is VALID, and that is the problem

100 seeds per task, **100/100 solved in all three**, seeds 5000–5099.

| task | forced sharing | median SPEC | IQR | mean ± sd | range |
|---|---|---|---|---|---|
| `pair_zero` | none | **1.0000** | [1.000, 1.000] | 0.987 ± 0.031 | 0.826–1.000 |
| `pair_partial` | partial | **0.6809** | [0.615, 0.720] | 0.664 ± 0.086 | 0.323–0.821 |
| `pair_full` | total | **0.0000** | [0.000, 0.000] | 0.084 ± 0.171 | 0.000–0.680 |

Pairwise, one-sided Mann–Whitney:

| contrast | difference | U | p | |
|---|---|---|---|---|
| `pair_zero` > `pair_partial` | +0.3191 | 10 000 / 10 000 | 6.1 × 10⁻³⁷ | **ORDERED** |
| `pair_partial` > `pair_full` | +0.6809 | 9 862 | 1.2 × 10⁻³⁴ | **ORDERED** |
| `pair_zero` > `pair_full` | +1.0000 | 10 000 / 10 000 | 4.0 × 10⁻³⁹ | **ORDERED** |

`U = 10 000` out of a possible 10 000 is **complete separation** — not one `pair_partial`
circuit out of 100 scored as high as any `pair_zero` circuit.

**SPEC passes the calibration.** It reads ground-truth behavioural decomposition, in the
right order, with the extremes perfectly separated. The instrument is not broken.

### But: the target fixes the reading

> **η² = 0.917.** Task identity explains **92 % of the variance** in SPEC.

The two extreme arms have essentially **no spread at all** — `pair_zero` IQR
[1.000, 1.000], `pair_full` IQR [0.000, 0.000]. Once a circuit solves, SPEC is very
nearly a function of *what was asked for*, not of *how evolution got there*.

That is the finding, and it cuts against the way SPEC has been used in this quarantined
set:

* **As a descriptive measure of a solved circuit's decomposition, SPEC works.** It would
  answer "is this circuit organised the way the task decomposes?" accurately.
* **As a dependent variable for comparing two treatments on one task, it has almost no
  room to respond.** Only the partial-overlap regime leaves any variance, and even there
  the free band is about 0.1 wide (IQR [0.615, 0.720]) around a centre the target pins.

## 4b. ⚠️ Correction to `SPECIALISATION.md`

`SPECIALISATION.md` concluded **"H-S1 refuted"**. That is stronger than the evidence
supports, and this calibration is why.

A refutation requires an instrument able to detect the effect had it been there. On
`pair_partial` — the only arm with any headroom — SPEC's free band is ~0.1 wide against a
target-fixed centre. The correct statement is:

> **H-S1 is not refuted; it is largely untested.** What was shown is that staging does not
> move SPEC *within the narrow band the target leaves free* (replication: r = +0.006, 95 %
> CI [−0.107, +0.116], n ≈ 400). That is a real null about SPEC, and it is *not* a null
> about modularity, because 92 % of what SPEC reports is the task.

The saturation worry recorded in `SPECIALISATION.md` §5 before the run was therefore
correct, and worse than it was written: it applies to `pair_partial` too, not only to the
`full` and `zero` endpoints.

`ENTRENCHMENT.md` is unaffected — its outcome variable is generations-to-solve, not SPEC.

## 4c. Secondary tasks

⚠️ **The secondary run was stopped before it finished.** What actually exists:

| task | seeds run | of planned | solved | median SPEC (unsolved) |
|---|---|---|---|---|
| `add4` (9 in, 5 out) | 50 | 50 — **complete** | **0/50** | 0.4520 |
| `mult4` (8 in, 8 out) | 40 | 50 — partial | **0/40** | 0.5588 |
| `retina_x2` (16 in, 2 out) | 0 | 30 — **never started** | — | — |

So `add4` genuinely failed to solve in 500 000 generations across a full 50 seeds;
`mult4` failed across the 40 seeds it got; and **`retina_x2` was never run at all** —
the only `retina_x2` numbers anywhere in this file are the 4-seed pilot in §5
limitation 3, which is not part of this run.

They contribute only to the confound check, where they confirm limitation 3: unsolved
circuits sit at a mid-range SPEC (0.45, 0.56) with no relation to organisation, purely
because their outputs hang off partly-disjoint junk cones.

**Consequence: the ground-truth claim is established at 2 outputs only.** Whether SPEC's
ordering survives at 5 or 8 outputs is untested, and would need tasks that actually solve
in budget — `add4` and `mult4` are both too hard for (1+4) CGP at 100 nodes, which is
itself worth knowing before anyone designs a many-output study on this substrate.


---

## 5. Known limitations, recorded before leg 2 landed

1. **The null uses one genotype size** (100 nodes) and one arity. It is a baseline for
   the runs in this file, not a universal constant.
2. **`pair_full` has a legitimate two-copy solution.** Its outputs are identical, so a
   circuit that evolves *two independent copies* of `L` scores high SPEC while being the
   opposite of parsimonious. Observed in the pilot (0.55, 0.435 alongside 0.0, 0.0). This
   is a real property of the metric, not noise: SPEC cannot tell "modular" from
   "duplicated".
3. **Unsolved circuits score high.** Pilot `retina_x2` read SPEC = 1.0 on four seeds that
   never solved. Leg 2 therefore reports solved circuits only, and the confound check
   tabulates solved against unsolved explicitly.
4. **This calibrates SPEC, not modularity.** A well-behaved SPEC would still be one
   operationalisation among several; agreement with Newman Q is not tested here.
