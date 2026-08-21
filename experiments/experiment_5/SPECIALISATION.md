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

### ⚠️ The secondary test is much weaker than it looks (noted 2026-08-21, before results)

`full` and `zero` are **saturated by construction**, so the interaction prediction is
close to unfalsifiable. Measured in the first completed cell (`staged-full-cgp4`,
n=50): **43 of 50 seeds sit at exactly `SPEC = 0.000`**, and pilot P6's `pair_zero`
runs sit at exactly `SPEC = 1.000`. A floored arm cannot show a gap whether the
mechanism is real or not, so "no gap at full and zero" will be confirmed by the
*ceiling*, not by the absence of the effect.

This is recorded here rather than in the discussion because it is a design flaw, not
a finding. The secondary test as written contributes almost no evidence, and the
honest reading of the whole study rests on the **primary contrast alone**. A future
version would need overlap levels that are graded rather than extremal — e.g. `O2 =
L AND R` and `L OR R`, which share the left detector to differing degrees without
forcing `SPEC` to an endpoint.

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

## Phase 2 — the pilot, and the budgets it froze (2026-08-21)

Run under PyPy, `runs/_spec_pilot/`. Compute turned out to be a non-issue: 20 seeds ×
200 000 generations of CGP is **5.7 s** wall on 8 workers.

| pilot | question | answer |
|---|---|---|
| P1 | how hard is stage 1 (`L` alone)? | **20/20 solved**, median **3 250** gens, max 17 250, p90 11 500 |
| P2 | how hard is the full `partial` task, cold? | **20/20 solved**, median **46 921** gens, max ~86 000 |
| P3 | does SPEC drift after the task is solved? | 0.654 at solve → ~0.62 by +20 000, then **flat to +100 000** |
| P4 | does the staged arm still solve? | 12/12 at 300 000 gens |
| P5 | does NAND-only still solve? | 11/12 at 300 000 gens → budget raised |
| P6 | do `full` / `zero` solve? | 12/12 each; medians 2 855 and 18 520 |
| P7/P8 | does ECGP solve, with both gate sets? | 12/12 each; ECGP is ~15× slower in wall clock |

**Both kill-switch conditions cleared.** Stage 1 is solved but not instantly (so
staging is real, not drift), and the full task is solved by every arm (so `SPEC` is
measured on working circuits, not failures).

### Frozen parameters

| | value | why |
|---|---|---|
| `--stage1-gens` (staged arm) | **20 000** | covers P1's slowest seed (17 250) with headroom |
| `--generations` | **500 000** | P5 censored 1/12 at 300 000; the primary analysis conditions on solving, so censoring is bias |
| `--post-solve-gens` | **20 000** | clears P3's transient and equalises post-solution drift across arms |
| `--nodes` | 100 | experiment 4's default, unchanged |
| seeds | **50** | see below |

**Measurement point:** the final logged row of each seed, i.e. exactly
`solved_gen + 20 000`. Every measured circuit therefore has **identical fitness
(perfect) and identical post-solution age**, so a `SPEC` difference cannot be a
fitness difference in disguise. Seeds that never solve are excluded and the **solve
rate is reported per cell** — if it ever differs between the staged and cold arms of
one overlap level, the conditioning is itself a confound and will be said so.

### Two deviations from section 5, both recorded before any confirmatory run

1. **n = 50, not 30.** Raised solely because the pilot showed compute is not the
   binding constraint. Fixed now; **no seeds will be added after seeing results.**
2. **`gens-to-solve` was seen during piloting** — `train.py` prints it in its summary
   line, so P2/P4/P5 exposed it before the confirmatory run. It is a *secondary*
   outcome and is now demoted to exploratory-only; no claim will rest on it. `SPEC`
   itself was **not** compared between staged and cold at any point during piloting.

### One prediction already confirmed by construction, not by evolution

Pilot P6's `pair_zero` runs return `SPEC = 1.000` with `n_pleio = 0` — every active
node influences exactly one output, in every seed. That is section 2's confound
appearing exactly as predicted: with disjoint demands the task alone forces
`SPEC = 1`, and no evolutionary process is being measured. It is reported here as
evidence the metric behaves as analysed, **not** as a result.

---

## Phase 3.5 — a REPLICATION, preregistered before it ran (2026-08-21)

Phase 3 returned a **null** on the primary contrast (p = 0.0558, α = 0.05; full
numbers below). Two facts about that null, both computed before deciding anything:

* the 95% bootstrap CI on the effect size is **[−0.052, +0.404]** — it includes zero;
* resampling these data says n = 50 per arm had only **49% power** for an effect this
  size. The study was a coin flip. n = 200 per arm reaches 94%.

**This is NOT a licence to add seeds to phase 3.** Extending a dataset after seeing
p = 0.056 is optional stopping, and it would invalidate the p-value that has already
been reported. The phase-3 null stands exactly as reported, permanently.

What is legitimate is an **independent replication**, fixed in advance and run on
**disjoint seeds**. Declared here before execution:

| | |
|---|---|
| seeds | **1000–1199** (phase 3 used 0–49; no overlap) |
| n | **200 per arm**, fixed now, final |
| contrasts | exactly **two**, both one-sided Mann–Whitney U for `staged > cold` |
| | (R1) overlap=partial, encoding=**cgp4** — the phase-3 primary |
| | (R2) overlap=partial, encoding=**cgpnand** — phase 3's largest exploratory effect (r = +0.308), which needs confirmation before it means anything |
| α | **0.025 each** (Bonferroni over the two) |
| everything else | identical to phase 3: 500 000 gens, 20 000 stage-1, 20 000 post-solve, 100 nodes |

No other cell is replicated and no other test will be run on these seeds. If R1 and R2
disagree with phase 3 in either direction, **both results get reported**, and the
combined reading is stated as such rather than the more convenient one being kept.

---

## Results

### Phase 3 — the confirmatory run (2026-08-21)

24 cells x 50 seeds = **1 200 runs**, 4 223 s wall under PyPy on 8 workers.
`runs/_spec/tidy.csv`, figure `runs/_spec/specialisation.png`.

#### The primary contrast is a NULL

> `staged` vs `cold`, overlap = partial, encoding = cgp4, one-sided Mann-Whitney U:
> **U = 1481, p = 0.0558**, against a preregistered alpha of 0.05.
> median SPEC **0.6340** (staged, n=50) vs **0.6111** (cold, n=50).
> rank-biserial **r = +0.185**, 95% bootstrap CI **[-0.052, +0.404]**.
> median difference **+0.0229**, 95% CI **[-0.0104, +0.0556]**.

**H-S1 is not supported.** The direction is the predicted one and the p-value is close
to the line, which is exactly the situation in which a result gets talked up; it is
recorded here as a null because that is what the preregistered rule says it is. The
effect-size interval includes zero, and the median shift is **2.3 percentage points on
a 0-1 scale**, which would be a weak effect even if it were real.

An honest caveat that cuts the other way: resampling these data says n = 50 per arm
had only **49% power** for an effect this size. Phase 3 was close to a coin flip, so
this null is weak evidence of absence. That is what phase 3.5 exists to fix.

#### The secondary contrast is uninformative, as predicted

| overlap | staged | cold | gap | p | r |
|---|---|---|---|---|---|
| full | 0.0000 | 0.0000 | +0.0000 | 0.267 | +0.041 |
| **partial** | 0.6340 | 0.6111 | +0.0229 | 0.0558 | +0.185 |
| zero | 1.0000 | 1.0000 | +0.0000 | 0.393 | +0.020 |

The "no gap at full and zero" prediction is confirmed, and means nothing: both arms
sit on their construction-forced endpoint in 43/50 and 50/50 seeds respectively, as
flagged in section 5 before the run. **No weight is placed on this row.**

#### No artefact explains the primary result

* **Solve rate**: 50/50 in 23 of 24 cells, 49/50 in the last. Conditioning on solving
  removed essentially nobody, so it is not a hidden selection step.
* **Circuit size**: median active nodes 23 (staged) vs 22 (cold) in the primary cell.
  SPEC is a ratio over influencing nodes, and the denominators match.
* **Run duration**: Spearman rho(SPEC, solved_gen) within cells is near zero and
  inconsistently signed; 2 of 24 cells reach p < 0.05, which is what chance gives.
  So SPEC is not tracking how long a run took.

#### Exploratory — the interesting result is NOT about SPEC

`SPEC` barely moves anywhere. **Generations-to-solve does**, and in opposite
directions depending on the gate set (medians, overlap = partial):

| encoding | staged, total | staged, stage 2 only | cold | stage 2 vs cold |
|---|---|---|---|---|
| cgp4 | 80 592 | 60 592 | 69 609 | **-9 018** |
| ecgp4 | 73 559 | 53 559 | 60 592 | **-7 033** |
| **cgpnand** | 118 936 | 98 936 | 65 086 | **+33 850** |
| ecgpnand | 88 131 | 68 131 | 64 902 | +3 229 |

With the four-gate set, a solved stage-1 left-detector **transfers**: stage 2 reaches
the full task ~9 000 generations faster than a cold start. It does not repay the
20 000-generation staging tax, but the transfer is real and in the direction
Espinosa-Soto & Wagner's co-option argument predicts.

With **NAND only it reverses hard** — stage 2 takes ~34 000 generations *longer* than
starting cold. A committed stage-1 solution appears to be an obstacle rather than a
head start when every sub-function is expensive to rewire. If anything in this
sub-study is worth following up, it is this: it is a large effect, it is the opposite
of the intended one, and "curriculum learning can entrench a bad basis" is a more
interesting claim than the one the study set out to test.

⚠️ **This is exploratory and was visible during piloting** (`train.py` prints
gens-to-solve in its summary line), so it is a hypothesis for a future preregistered
run, not a finding. It has had no multiplicity correction and no mechanism test.

#### Exploratory — SPEC by encoding

| encoding | staged | cold | gap | p | r |
|---|---|---|---|---|---|
| cgp4 | 0.6340 | 0.6111 | +0.0229 | 0.0558 | +0.185 |
| ecgp4 | 0.6190 | 0.6325 | **-0.0134** | 0.741 | -0.074 |
| cgpnand | 0.6087 | 0.5886 | +0.0201 | 0.0041 | +0.308 |
| ecgpnand | 0.5963 | 0.6030 | -0.0067 | 0.402 | +0.029 |

Both **CGP** arms lean positive and both **ECGP** arms lean negative. Twelve contrasts
were computed, so one at p = 0.004 is not surprising on its own; `cgpnand` is
nevertheless the largest effect in the study and is the reason it gets replicated in
phase 3.5 rather than merely mentioned.

The `ecgpnand` / `zero` cell reaches p = 0.0006 with both medians at exactly 1.000.
That is a tail effect at a hard ceiling and is noise, not signal — it is listed only
so that the one impressive-looking p-value in the table is not quietly dropped.

