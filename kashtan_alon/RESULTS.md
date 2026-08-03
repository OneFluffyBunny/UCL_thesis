# kashtan_alon — RESULTS (lab notebook)

The run record for the Kashtan–Alon retina MVG→modularity reproduction. Reference
docs (`README.md`, `PAPER_SPEC.md`) describe *what* the setup is; this file records
*what happened* when we ran it. Newest runs first.

**The claim under test (KA 2005):** under Modularly-Varying Goals (G_AND ↔ G_OR
every 20 gens) the normalized modularity **Q_m** rises and *stays high* (**0.35 ±
0.02**); under a Fixed Goal it stays low (**0.15 ± 0.02**). Same network, same
search, only the goal schedule differs.

---

## Run 4 — code rebuilt to be KA-faithful (2026-08-03) → **pipeline verified; full run pending**

Runs 1–3 turned out to be running **Clune 2013's reimplementation**, not KA 2005 —
the spec was reconstructed from Clune and mislabelled "confirmed KA" (details:
[[reference_ka_retina_algo]]). On 2026-08-03 the actual KA methods were fetched from
the primary source (PMC1236541) and the code was **rewritten to match** the paper's
neural-network retina experiment:

| what changed | before (Clune-derived) | now (KA 2005, verified) |
|---|---|---|
| task | stand-in `(p0∧p1)∨(p2∧p3)` | **KA's real Fig. 5a objects** (≥3-black / outer-column-only) |
| weights / units | {−2,−1,1,2}, tanh(λ=20) | **±1**, hard **threshold** ({0,1} out) |
| fan-in | none | **≤3 / ≤2** per layer |
| reproduction | mutation-only | **crossover Pc=0.5** + mutation Pm=0.5 |
| selection / pop | tournament + 1 elite, pop 1000 | **elite 150 / 600** |
| headline metric | raw Newman Q | normalized **Q_m**, Q_rand over **1000** |

Architecture was **already correct at 8-8-4-2-1** (8 retina pixels are a separate
input layer; an intermediate note that mis-said "8-4-2-1" was wrong).

- **Verified:** `test_tasks.py` passes with KA's exact truth counts (left object 8/16,
  retina AND 64/256, OR 192/256, left/right provably independent); a smoke run
  (pop 60 / 300 gens) exercises the whole GA→Q_m→checkpoint→viz pipeline cleanly.
- **Not yet done:** the real MVG-vs-FG run (pop 600, 25 000 gens). Expectation if the
  reproduction is faithful: **Q_m ≈ 0.35 (MVG) vs ≈ 0.15 (FG)**. This is the run that
  actually tests KA's claim — everything before it was the wrong experiment.
- **Residual caveat:** exact mutation operators + threshold range are in KA's SI (not
  the main text); `ga.py` uses a documented faithful reconstruction.

---

> ⚠️ **CORRECTION (2026-08-03):** Run 2's "reproduced" verdict below used **raw
> Newman Q**, which is density-confounded. Re-scored with **Kashtan-Alon's actual
> normalized metric Q_m = (Q_real−Q_rand)/(Q_max−Q_rand)**, the result is a **NULL**:
> MVG Q_m = −0.04 ± 0.17, FG = −0.13 ± 0.12 (KA target: 0.35 vs 0.15). Both conditions
> are non-modular (Q_real ≈ Q_rand — no excess over degree-matched random), and MVG is
> not meaningfully above FG. The raw-Q "0.38 vs 0.27" gap was purely the density
> artifact (MVG nets sparser ⇒ higher raw Q for free). Metric validated: two-clique
> graph → Q_m 1.00, ER random → 0.22, ring lattice → 0.50. **We have NOT reproduced
> KA's MVG→modularity result.** See "Run 3" note below.

## Run 2 — corrected measurement, 5 seeds (2026-08-02) → ~~✅ MVG > FG reproduced~~ **retracted (raw-Q artifact)**

Same paper-locked config as Run 1, but with the two Run-1
measurement artifacts fixed (commit `8a60977`): **early-stop OFF** (FG evolves the
full 25000 gens, matched to MVG) and **Q measured on the final-generation champion**
(the evolved topology), not the first net to hit peak fitness. **The raw-Q numbers
below stand as computed but are the WRONG metric — see the correction banner above.**

- **Command:** `conda run --no-capture-output -n lndp python run_paper.py --n-seeds 5 --fresh --viz`
- **Where:** laptop (CPU, pure-numpy, 0 VRAM) · full run ≈ 2.3 h · all seeds fit 1.000.

### Result — the separation is clean

| Measure | MVG | FG (L AND R) | gap |
|---|---|---|---|
| **Final-gen Q** (mean) | **0.383** | 0.274 | **+0.109** |
| Final-gen Q per seed | 0.400, 0.339, 0.332, 0.470, 0.372 | 0.295, 0.270, 0.318, 0.196, 0.289 | |
| Per-gen Q, last-10% window (phase-averaged, robust) | 0.341 ± 0.014 | 0.284 ± 0.020 | +0.057 |
| edges (final net) | 34–41 (sparser) | 46–59 | |

- **No overlap:** MVG's worst seed (0.332) > FG's best (0.318). **25/25** seed-pairs MVG > FG.
- Every run solves the task (fit 1.000 both conditions) → a modularity difference
  between *equally-performing* networks, which is exactly Kashtan–Alon's claim.
- MVG nets are consistently **sparser** — modular solutions prune cross-connections.

**Verdict vs paper:** ✅ **Direction and consistency reproduced** — MVG spontaneously
favours higher modularity than a fixed goal, robustly across seeds and across two
independent Q measures. Absolute scale differs from the paper (our FG ≈ 0.27 vs
paper's ~0.15–0.2; our MVG ≈ 0.38 vs ~0.4+) — expected, since our retina object
patterns and greedy-Newman-Q metric differ from KA's, and there's no connection-cost
term pushing FG lower. What's testable — MVG > FG — holds cleanly.

### Run 3 — re-scored with KA's normalized Q_m (2026-08-03, no re-run; on Run 2's saved nets)

Implemented `modularity.normalized_qm` = `(Q_real − Q_rand)/(Q_max − Q_rand)`
(Q_rand = mean greedy-Q over 100 degree-preserving randomizations; Q_max = greedy-Q
of a modularity-maximizing rewiring at the same degree sequence). Metric validated on
known graphs: two K6 cliques+bridge → **1.00**, ER random → **0.22**, ring → **0.50**.

| condition | Q_m (mean ± sd) | per seed | raw Q (Run 2) |
|---|---|---|---|
| MVG | **−0.04 ± 0.17** | −0.07, −0.27, −0.15, +0.24, +0.04 | 0.383 |
| FG  | **−0.13 ± 0.12** | −0.06, −0.07, −0.05, −0.36, −0.11 | 0.274 |

**Verdict: NULL — not reproduced.** Every net has Q_real ≈ Q_rand (no excess modularity
over degree-matched random); both conditions are non-modular and below even a random
graph's Q_m ≈ 0.22. MVG (−0.04) vs FG (−0.13) is within noise (n=5). The Run-2 raw-Q
separation was a pure density artifact.

### Next steps (proposed)
1. **Clune connection-cost term** — promoted to #1. The follow-up literature's reliable
   modularity driver; MVG-alone is fragile/insufficient (our null is consistent with
   this). A variation, deliberately outside KA's MVG-only claim.
2. **Lower `--init-density`** (currently 0.5): sparser nets, higher Q ceiling.
3. **Faithfulness audit of task/architecture** vs KA's exact retina — our object
   patterns are a stand-in and may not create KA's modular pressure.
4. **Wire Q_m into `train.py` per-gen logging** (currently logs raw Q) so future runs
   track the correct metric live, and consider a direct functional left/right measure.

---

## Run 1 — full paper preset, 5 seeds (2026-07-31) → **near-null (measurement artifacts, superseded by Run 2)**

_Note: Run 1's seed-0 raw artifacts were later overwritten by a smoke test; the
aggregate numbers below stand, and Run 2 supersedes this run. Kept as the record of
why the naive measurement produced a false null._

- **Command:** `conda run -n lndp python run_paper.py --n-seeds 5 --viz`
- **Where:** laptop (CPU, pure-numpy, 0 VRAM) · ~30–34 ms/gen · full run ≈ 2.3 h
- **Config (locked to paper):** net 8-8-4-2-1, tanh λ=20, weights {−2,−1,1,2},
  biases {−2..2}, pop 1000, 25000 gens, mutation-only GA (add/remove 20%, weight
  ±1 @ 2/n, bias ±1 @ 1/24), tournament k=3 + 1 elite, **raw** fraction-correct
  fitness. Two conditions: **MVG** (and↔or/20) vs **FG(L AND R)**.
- **Logs:** `runs/retina_{mvg,fg}_raw_seed{0..4}_log.csv` (per-gen Q/density/fitness),
  `runs/*_best.png` (final brains, module-coloured), `runs/*_result.json` (per-seed
  summary). NB: `runs/paper_run.log` (console) was buffered by `conda run` and only
  flushed at exit — use `--no-capture-output` next time for a live console stream.

### Result — the paper's separation did NOT reproduce

**Best-net Q** (saved best-fitness individual, from `result.json`):

| Condition | mean Q | Q per seed | mean best fit |
|---|---|---|---|
| MVG (and↔or) | **0.309** | 0.339, 0.296, 0.298, 0.314, 0.299 | 1.000 |
| FG (L AND R) | **0.279** | 0.308, 0.296, 0.271, 0.219, 0.302 | 1.000 |

**Per-generation Q** (fairer — best net *each* logged gen, over the whole run):

| Condition | mean (all gens) | mean (2nd half) | max ever | last gen / seed |
|---|---|---|---|---|
| MVG | 0.335 | 0.339 | 0.549 | 25000 (all seeds) |
| FG  | 0.248 | 0.254 | 0.376 | 480, 220, 8090, 7490, 490 |

**Verdict vs paper:** ❌ **Not reproduced at the paper's magnitude.** Paper claims
MVG ~0.4+ vs FG ~0.15–0.2. We get MVG ≈ 0.31–0.34 vs FG ≈ 0.25–0.28 — *directionally*
correct (MVG more modular) but only a ~0.05–0.09 gap, and both well inside each
other's range. Every seed solved the task (fit 1.000).

### Confounds / caveats (read before trusting this)

- **FG early-stops, MVG doesn't** — the biggest one. FG hits fit 1.0 → `--target 1.0`
  early-stop fires (3/5 seeds before **gen 500**; others ~7500–8000), while MVG runs
  the full 25000 gens (its goal keeps moving, so it never early-stops). So FG is
  scored on a barely-evolved first solution and MVG on a long-evolved one — **not
  matched on evolutionary time.** *Fix for the next run:* `--target 1.1` (or a
  no-early-stop flag) so FG also evolves the full 25000 gens. Even handicapped-short,
  though, FG's separation from MVG is already weak.
- **Object patterns ≠ KA's.** We use `L=(p0∧p1)∨(p2∧p3)`, `R=(p4∧p5)∨(p6∧p7)`; KA's
  exact retina bit-patterns are unpublished. The MVG effect is known to be sensitive
  to task details.
- **Consistent with the follow-up literature.** Clune, Mouret & Lipson 2013 found MVG
  *alone* is an unreliable modularity driver without a connection-cost term — a weak
  separation here is not surprising. Adding a connection-cost variant is the obvious
  next experiment (a *variation*, not the paper).
- **Best-net Q ≈ per-gen Q** here (0.31 vs 0.34 MVG; 0.28 vs 0.25 FG), so the
  "save the best-*fitness* net, not the most-modular one" choice is not what's
  suppressing the signal — the signal is genuinely weak.

### Next steps (proposed, not yet run)
1. Re-run with FG early-stop disabled (`--target 1.1`) — the fair apples-to-apples.
2. If still weak: try a connection-cost variant (Clune's driver) and/or `xor` goal.
3. Consider the exact KA goal pair / object set if we can recover it.

---

## Conventions for this file
- One `## Run N` block per run, newest first; keep the exact command and the config
  that mattered so it's reproducible.
- Record the **Q separation** (the headline), not just final fitness.
- `runs/` is gitignored (regenerable) — conclusions live here.
