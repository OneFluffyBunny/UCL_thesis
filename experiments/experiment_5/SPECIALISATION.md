# Specialisation sub-study — PREREGISTRATION

> ## ⚠️ READ THIS FIRST — provenance
>
> **This sub-study was designed, run and interpreted by Claude Code (Opus 5), on its
> own initiative, at the user's invitation to "do research on your own".** The
> hypothesis, the arms, the metric, the predictions and the conclusions below are the
> model's, not the author's and not a supervisor's. **Nothing here has been reviewed
> by a human.** It is a demonstration of autonomous experimental work and should be
> read with a spoonful of salt. Do not promote any claim from this file into the
> thesis without re-deriving it by hand.
>
> Started 2026-08-21 on branch `spec-modularity`, forked from `fluffy_experiments`.

---

## 0. Motivation

Espinosa-Soto & Wagner (2010), *Specialization Can Drive the Evolution of Modularity*,
PLoS Comput Biol 6(3):e1000719, report that in a gene-regulatory model, modularity
rises when a **new demand is added while the old one stays under selection**, and that
this happens **only when the two demands partially overlap** — not when they are
identical, and not when they are disjoint (their Figure S6D).

Their mechanism is **pleiotropy reduction**: interactions between shared machinery and
demand-specific machinery obstruct adaptation and are selected against. They can only
*proxy* that mechanism with Newman modularity on the interaction matrix. In a Boolean
circuit, pleiotropy is **directly and exactly computable**. That is the reason to try
the translation at all.

This is positioned against Kashtan & Alon's modularly-varying goals — which is what
`experiment_4` already runs — as a mechanism *not contingent on environmental change*.

## 1. Hypothesis

**H-S1.** Adding a second demand that *partially* overlaps an existing one, while the
first stays under selection, drives node specialisation **beyond what the task's own
structure forces**.

## 2. ⚠️ The confound that killed the first version of this design

The obvious metric — "fraction of active nodes that influence exactly one output" —
compared **across overlap levels** is worthless, because the task construction fixes
the answer before evolution starts:

| overlap | outputs | forced specialisation |
|---|---|---|
| full | `L`, `L` | ≈ **0** — one detector, both outputs read it |
| partial | `L`, `L XOR R` | ≈ **0.5** — `L` shared, `R` specific |
| zero | `L`, `R` | ≈ **1.0** — disjoint inputs, disjoint circuits |

The predicted ordering that was pitched to the user on 2026-08-20 (full low, partial
**high**, zero low) is **backwards from what the construction alone produces**. Any
across-overlap comparison measures the task, not evolution. Caught before running;
recorded because it is the kind of error that otherwise survives into a result.

**The fix: compare *within* an overlap level, across schedule.** The task is then
byte-identical in both arms, so the construction-forced component cancels exactly, and
any difference is attributable to the selection schedule. Overlap becomes a
**moderator**, not a treatment — which is also a closer match to what E-S&W actually
claim.

## 3. Design

Task substrate: the 8-pixel Kashtan–Alon retina. `L` = left object (pixels 0,1,2,3),
`R` = right object (pixels 6,7,4,5), both by the Fig. 5a rule already verified in
`tasks.py`. All 8 inputs exist from generation 0 in every arm; early-stage targets
simply ignore the ones they do not need.

**Factor 1 — schedule** (the treatment):

| level | protocol |
|---|---|
| **staged** | evolve `O1` alone for `G1` generations; then add `O2`, with `O1` still scored, for `G2` more |
| **cold** | evolve `O1` and `O2` together from generation 0, for `G1 + G2` |

Equal total generation budget. `cold` is what separates *"two demands"* from *"two
demands, arrived at in that order"* — without it the result is uninterpretable.

**Factor 2 — overlap** (the moderator):

| level | `O1` | `O2` | shares |
|---|---|---|---|
| full | `L` | `L` | everything |
| **partial** | `L` | `L XOR R` | the `L` detector |
| zero | `L` | `R` | nothing |

**Factor 3 — encoding**:

| level | encoding | gate set |
|---|---|---|
| 1 | CGP | AND/NAND/OR/NOR |
| 2 | ECGP | AND/NAND/OR/NOR |
| 3 | CGP | NAND only |
| 4 | ECGP | NAND only |

Level 3 exists only as the control for level 4: NAND-only lengthens every sub-function,
which lengthens the search, which gives neutral drift more time — a confound that has
nothing to do with modules. Rationale for including NAND-only at all is in §6.

**2 × 3 × 4 = 24 cells, 30 seeds each = 720 runs.** Seeds `0..29`, identical seed set
in every cell.

## 4. Metric

For each **active node** `n` and each **program output** `o`: invert `n`'s output mask,
repropagate, and ask whether `o`'s mask changed. This is exact — no sampling, no
clustering heuristic, no rewire null — and it is the mechanism E-S&W propose, measured
directly rather than proxied.

```
specialised  = active nodes that change exactly 1 output
pleiotropic  = active nodes that change more than 1 output
SPEC         = specialised / (specialised + pleiotropic)
```

Nodes that change no output are excluded (they are inactive by definition).

Measured on the **flattened** circuit, so CGP and ECGP go through identical code.

## 5. Preregistered predictions

**PRIMARY** — one contrast, declared before any run:

> `staged` > `cold`, within **overlap = partial**, **encoding = CGP/4-gate**, on `SPEC`.
> One-sided Mann–Whitney U, n = 30 per arm, α = 0.05.

**SECONDARY — the interaction** (this is the actual E-S&W claim):

> The `staged − cold` gap in `SPEC` is **positive at partial**, and **absent at full
> and at zero**.

**EXPLORATORY** — everything else, including all three non-baseline encodings.
Labelled as exploratory in the writeup, with no significance claimed. With 24 cells I
will otherwise find something, and it will be noise.

**What falsifies H-S1:** `staged ≈ cold` at partial. That is a real negative result
about whether E-S&W's mechanism crosses from gene networks to Boolean circuits, and
will be written up as one rather than explained away.

## 6. Why NAND-only, given that ECGP's modules are mostly fake

`experiment_4/RESULTS.md` (2026-08-14 module census, 11 172 modules over 12 seeds)
found that ECGP's acquired modules are largely degenerate: **31% compute exactly one
primitive**, **63% appear at only one log point**, and many are a fundamental gate
wearing extra inputs that do nothing. So "does ECGP acquire modules?" is already
answered, mostly in the negative, and simply counting modules here would repeat that
mistake.

Two consequences for this design:

1. **NAND-only is a manipulation of how much reuse is worth.** With four gates a
   useful sub-function is 1–2 gates, so encapsulating it buys nothing and the module
   list is churn. Under NAND-only, `AND` is 2 gates and `XOR` is ~4. If compress/expand
   can ever do real work, that is the regime.
2. **Module counts are not a metric here.** Any module claim in this sub-study must be
   filtered by the canonical behavioural signature machinery already in
   `experiment_4/ecgp.py` (two circuits that both behave like XOR share a signature),
   and a module that is signature-identical to a single primitive is **not counted**.

## 7. Phases, and the one that can kill it

| # | what | gate |
|---|---|---|
| 0 | this file, committed before any run | — |
| 1 | build: staged schedule, 3 paired tasks, `pleiotropy()` | `test_equivalence.py` stays green (exp_4 ↔ exp_5 equality must not break) |
| 2 | **pilot**: partial/CGP-4, both schedules, 5 seeds | sets `G1`, `G2`. See below. |
| 3 | full 720 | — |
| 4 | analysis, one figure, conclusions appended here | — |
| 5 | *if and only if* phase 3 is positive: the co-option test (E-S&W Fig. 5) | — |

**Phase 2 is the kill switch.** Two ways the design dies there, both cheap to detect:

- If `L` alone solves in a few hundred generations, `G1` is mostly neutral drift and
  "staged" is not really staged.
- If `L XOR R` is never solved by either schedule, `SPEC` is being measured on failed
  circuits and means nothing.

`G1` and `G2` will be **set from the pilot and then frozen**, before the 720 run.

---

## Results

*(Nothing yet — phase 0 complete, phase 1 in progress.)*
