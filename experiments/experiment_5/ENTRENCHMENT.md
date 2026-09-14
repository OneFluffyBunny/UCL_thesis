# Entrenchment sub-study — PREREGISTRATION

> ## ⚠️ READ THIS FIRST — provenance
>
> **Designed, run and interpreted by Claude Code (Opus 5) on its own initiative**, at
> the user's invitation to do research autonomously. Hypothesis, arms, metric,
> predictions and conclusions are the model's. **Not reviewed by a human.** Quarantined
> here on purpose; do not cite from it. Companion to [`SPECIALISATION.md`](SPECIALISATION.md),
> which is where the effect tested here was found.
>
> Started 2026-08-24 on branch `spec-modularity`.

---

## 0. Where this came from

`SPECIALISATION.md` set out to show that a staged second demand *increases* modularity.
It does not — that hypothesis is refuted at n≈400. What survived was the **opposite**
effect, on a different outcome:

> Under a NAND-only gate set, a circuit that has already solved the left-object
> detector needs **+28 357 generations** to then solve the full two-output task,
> compared with a circuit starting cold. 95% CI [+11 860, +45 079], p = 0.00044, on
> seeds the effect was not discovered in. The four-gate set points the same way
> (+8 464, p = 0.070).

Staging is **negative transfer**. That is interesting on its own — it is a caution
against the "incrementally increasing problem complexity" strategy — but the mechanism
is unidentified, and the statistic was chosen after seeing the data. This study fixes
both.

## 1. The two candidate mechanisms

| | claim | prediction |
|---|---|---|
| **ENTRENCHMENT** | the damage is caused by *acquiring* a committed stage-1 solution, which subsequent search must dismantle | damage appears as soon as stage 1 is typically **solved**, then **plateaus** |
| **DRIFT-TIME** | the damage is caused by *time spent under single-output selection* — narrowed diversity, a drifting unselected output gene | damage grows **monotonically** with stage-1 length, no plateau |

These make opposite predictions about the *shape* of the dose–response curve, which is
why the design is a ladder rather than a two-point comparison.

**E-H1.** The cost of staging is caused by acquiring the stage-1 solution, not by time
spent in stage 1.

## 2. Design

Same substrate as `SPECIALISATION.md`: `pair_partial` — 8 KA retina pixels, `O1 = L`
(left object), `O2 = L XOR R`. `O1` stays scored forever once introduced.

**One factor of interest: stage-1 length `G1`.**

| G1 | what fraction of seeds have solved stage 1 by then |
|---|---|
| **0** (cold, no staging) | — |
| 1 000 | ~25% |
| 3 000 | ~50% (median is 2 284 / 2 694) |
| 10 000 | ~85% |
| 20 000 | ~100%, plus drift — **the level tested in `SPECIALISATION.md`** |
| 100 000 | ~100%, plus 5× more drift |

Roughly log-spaced, and pinned to the measured stage-1 solve distribution rather than
chosen for roundness (pilot below).

**Second factor: gate set** — `and,nand,or,nor` and `nand` only. The effect was ~3×
larger under NAND-only, so it is retained as the axis the effect is expected to scale
along.

**6 × 2 = 12 cells × 200 seeds = 2 400 runs.** Seeds **2000–2199**, fresh — disjoint
from phase 3 (0–49) and phase 3.5 (1000–1199).

### The budget is equalised on stage 2, not on the total

Every arm gets **`G1 + 400 000`** generations, so each has the *same 400 000
generations of stage-2 search*. Equalising the total instead would give the long-`G1`
arms less stage-2 budget and manufacture the very slowdown this study measures, as a
censoring artefact.

`--post-solve-gens 0` (the outcome is time-to-solve, not the circuit's final state) and
`--checkpoint-interval 0` (`stage1_active` is not checkpointed — see §3).

### Outcome

**stage-2 generations to solve = `solved_gen − G1`.** For the cold arm that is just
`solved_gen`. This charges the staged arms only for stage 2, so the question is *"did
having a solved detector help or hurt?"* — not *"was staging cheaper overall?"*, which
`SPECIALISATION.md` already answered (it was not).

## 3. What the run records about the cause

`train.py` captures, at the moment the second demand is added:

* `stage1_solved_gen` — when stage 1 was first solved, or −1
* `stage1_active` — **active nodes in the stage-1 circuit**, i.e. how much structure
  was committed
* `stage1_hits` — its score, so unsolved stage-1 seeds are identifiable

`stage1_active` is the entrenchment hypothesis's *cause*, measured before stage 2
rewrites anything. Reading it off the final circuit would be useless — stage 2 has by
then modified it.

⚠️ These three fields are **not written to the checkpoint**, so these runs use
`--checkpoint-interval 0` and cannot be resumed. Deliberate: adding three fields to the
checkpoint format would invalidate every existing `.pkl` in `runs/`.

## 4. Preregistered tests

Three, α = **0.0167** each (Bonferroni over 3).

> **E-P1 (anchor).** NAND-only, `G1 = 20 000` vs `G1 = 0`: stage-2 generations are
> **higher** with staging. One-sided Mann–Whitney U.

> **E-P2 (the discriminating test).** Define
> `Δ_early = median(G1=3 000) − median(G1=0)` and
> `Δ_late = median(G1=100 000) − median(G1=20 000)`.
> **Prediction: `Δ_late < Δ_early`.** Bootstrap 95% CI on `Δ_late − Δ_early` excludes
> zero from above. NAND-only.

> **E-P3 (mechanism).** Within the NAND-only `G1 = 20 000` cell, the size of the
> committed stage-1 circuit predicts the stage-2 slowdown:
> **Spearman ρ(`stage1_active`, stage-2 generations) > 0**, one-sided, n = 200.

**Why E-P2 is the one that matters.** `Δ_late` spans **80 000** generations of extra
stage-1 time; `Δ_early` spans **3 000**. If the damage were about time under restricted
selection, `Δ_late` would dwarf `Δ_early`. Entrenchment predicts `Δ_late ≈ 0` because
by `G1 = 20 000` essentially every seed has already committed its solution and there is
nothing further to entrench.

**Honest status of each test.** E-P1 is the **third look at the same effect** — it is
an anchor for the curve, not new evidence, and it is expected to pass. **E-P2 and E-P3
are the genuinely novel tests**, and neither has been looked at in any form.

## 5. What would falsify E-H1

* **E-P2 fails** (`Δ_late ≥ Δ_early`) → the cost is time under single-output selection,
  not the acquired solution. Entrenchment is wrong and the finding becomes far less
  interesting: "restricting selection for a long time is bad" is not news.
* **E-P3 null** → the *amount* of committed structure does not predict the damage, so
  whatever staging breaks, it is not proportional to what stage 1 built.
* **E-P1 fails** → the phase-3.5 effect was a fluke after all, on its third look, and
  the whole line of enquiry closes.

## 6. Known limitations, recorded before the run

1. **Output gene 1 drifts unselected during stage 1**, and drifts more the longer `G1`
   is. That is a *drift-time* sub-mechanism this design does not separate from
   diversity loss — E-P2 lumps them together as "time". If E-P2 says "time", a follow-up
   would need to freeze the unselected output gene to tell them apart.
2. **One task.** Every claim here is about `pair_partial` on the KA retina. Nothing
   licenses generalising to other decompositions.
3. **`stage1_active` is a size proxy, not an entrenchment measure.** A large circuit is
   assumed harder to dismantle; that assumption is not itself tested.
4. **Stage 1 takes the same time under both gate sets** (pilot: median 2 284 vs 2 694,
   40 seeds each), so the NAND-only effect is *not* "stage 1 was harder there". This
   confound is already ruled out, before the run.

---

## Pilot (2026-08-24)

`runs/_ent_pilot/`, 40 seeds per gate set, stage 1 only, 150 000 generation cap.

| gate set | solved | median | p25 | p75 | p90 | max |
|---|---|---|---|---|---|---|
| `and,nand,or,nor` | 40/40 | 2 284 | 1 005 | 4 629 | 11 453 | 17 193 |
| `nand` | 40/40 | 2 694 | 892 | 4 139 | 8 837 | 61 258 |

This is what the `G1` ladder is pinned to, and it is the basis for limitation 4.

---

## Results (2026-08-27) — 2 400 runs, `runs/_ent/`

**E-H1 is not supported. Neither is its alternative.** The damage is real and
replicates a third time, but its dose–response shape matches *neither* preregistered
mechanism.

### The dose–response curve

Stage-2 generations to solve (`solved_gen − G1`), median over solved seeds:

| G1 | `and,nand,or,nor` | vs cold | `nand` only | vs cold |
|---|---|---|---|---|
| **0** (cold) | 60 008 | — | 70 046 | — |
| 1 000 | 79 506 | **+19 498** | 105 376 | **+35 330** |
| 3 000 | 65 560 | +5 552 | 88 019 | +17 973 |
| 10 000 | 72 314 | +12 306 | 88 554 | +18 508 |
| 20 000 | 69 145 | +9 138 | 92 818 | +22 772 |
| 100 000 | 71 399 | +11 392 | 95 901 | +25 855 |

Solve rates 189–200 / 200 in every cell, so this is not censoring.

**The shape is a step, not a ramp.** The full cost is already paid at `G1 = 1 000` —
the *smallest* dose, at which only ~25 % of seeds have solved stage 1 — and a further
**hundredfold** increase in stage-1 length adds nothing beyond it. Median
`stage1_active` is flat at 15–18 nodes across the whole ladder.

### The three preregistered tests

| test | result | |
|---|---|---|
| **E-P1** (anchor) | **SUPPORTED** | NAND, G1=20 000 vs cold: +22 772, U = 22 133, one-sided p = 0.00068 |
| **E-P2** (discriminating) | **NOT SUPPORTED** | Δ_early = +17 973 (over 3 000 gens of dose); Δ_late = +3 083 (over **80 000**); difference −14 890, 95 % CI [−42 468, +7 024] |
| **E-P3** (mechanism) | **NOT SUPPORTED** | Spearman ρ(`stage1_active`, stage-2 gens) = **+0.036**, one-sided p = 0.31 |

**E-P1** was declared in advance to be the third look at a known effect and therefore
an anchor rather than new evidence. It behaved as expected. The effect is now seen on
three disjoint seed sets (0–49, 1000–1199, 2000–2199); **negative transfer from staging
is the one solid finding of this whole line.**

⚠️ **E-P2 is inconclusive, not a refutation.** The CI includes zero, so the
preregistered criterion fails — but the point estimate is firmly in the *predicted*
direction (80 000 generations of extra dose buy +3 083, while the first 3 000 buy
+17 973). Failing to exclude zero is not evidence for the alternative, and this write-up
does not claim drift-time won. The honest reading is that the design lacked the power to
separate a plateau from a shallow ramp at this n.

**E-P3 is a clean null**, and it is the informative one. ρ ≈ 0 in the preregistered cell
and in all ten cells descriptively (range −0.117 to +0.049, no consistent sign). **How
much structure stage 1 committed does not predict the cost at all.**

### What the shape actually says

The cost is not proportional to what stage 1 *built* (E-P3), and not proportional to how
long stage 1 *ran* (the ladder is flat from 1 000 to 100 000). It appears in full at the
first dose. That leaves a third mechanism, not preregistered here:

> **The toll is the switch itself.** Under (1+4) the population *is* one lineage. When
> the fitness denominator changes, that single lineage is wherever the previous goal put
> it, and it pays a roughly fixed re-entry cost regardless of how good or how large its
> previous solution was.

This converges with an independent result already in this repo: `experiment_4/RESULTS.md`
records that under MVG — *repeated* goal switching — (1+4) CGP never solved the retina at
all, and that **0 of 25 552 epochs retained a solution across a switch**. This sub-study is
the single-switch version of the same phenomenon, and it prices it: ~20 000–35 000
generations, ~2× worse under NAND-only.

`experiment_4/RESULTS.md` already names the suspect and it is *not* on the list of
mechanisms preregistered above: **`--popsize`**. Kashtan–Alon ran 600 individuals with
crossover, so a goal switch cost them diversity; a (1+4) elitist lineage takes the switch
across its entire population at once. That is the test this result points at.

### Status

* **E-H1 (entrenchment): not supported.** The acquired stage-1 solution is not the cause —
  its size predicts nothing.
* **Drift-time: not supported either.** 100× more stage-1 time adds nothing.
* **Negative transfer itself: confirmed, three times, on disjoint seeds.**
* Limitation 1 of §6 (the unselected output gene drifting) is now moot as an explanation:
  it would have to grow with `G1`, and the curve is flat.
