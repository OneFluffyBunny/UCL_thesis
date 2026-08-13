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

1. **Genotype size is the dominant knob, and the paper's 100 nodes is too small
   here.** Table II's 100 nodes / 3% solves only 4/6 within 25k generations. 800
   nodes solves 6/6 in half the generations and half the evaluations. 50 nodes
   never solves.
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

### Chosen FG configuration

    --nodes 800 --mutation-rate 0.01 --task retina_ka2005 --operation and

⚠️ **This deviates from Table II** (100 nodes, 3%) and is recorded as such in
`PAPER_SPEC.md` §9. The paper's values were tuned on parity/adders/multipliers,
never on this task. The CLI defaults are left at Table II's values so nothing
deviates silently — the deviation has to be typed.

### Infrastructure notes

- Checkpoint/resume verified by killing a run at generation 1000 and resuming: it
  continued from gen 1000 with the RNG state restored and solved at 16 570.
- Evaluating only active nodes gave 4× (1.2 → 0.30 ms/gen at 100 nodes) with
  identical trajectories.
- **Density is not logged** — a CGP genotype's edge count is fixed by construction,
  so density is a constant. The structural columns are the input-cone
  decomposition (left / right / mixed / const) instead.

### Open threads

- The 50-seed FG run at the chosen configuration (in progress) is the baseline
  every later arm is measured against.
- No ECGP yet. The genotype is already in the gene-pair encoding it needs.
- Whether solved circuits are *reliably* modular (finding 5 generalised across
  seeds) is unanswered and needs the 50-seed run's cone statistics.
