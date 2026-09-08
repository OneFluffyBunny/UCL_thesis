# Experiment 2 — results & observations log

Running lab notebook for the direct-encoding control (see `README.md` for the
model, `../experiment_1/RESULTS.md` for the treatment this is measured against).
Newest entries at the bottom. Balanced accuracy throughout (chance = 0.5).
Bipolar inputs {-1,+1}. Search = CMA-ES over the raw weight vector (no shared
rule, one free parameter per edge) unless noted.

---

## retina/xor, n_hidden=20, 5 seeds, margin fitness

```
python train.py --task retina --operation xor --n-seeds 5 --fitness margin
```

Genome: 8→20 (IH) + 20×19 (HH, no self-loops) + 20→1 (HO) + 21 biases =
**581 free parameters**, optimized directly by CMA-ES (popsize=64, sigma_init=0.1,
default 1000 generations, early stop at target=1.0).

| seed | gens to solve | best balanced acc | edges used | density | exc(+) / inh(-) |
|---|---|---|---|---|---|
| 0 | 456 | **1.000** | 524/560 | 93.6% | 246 / 278 |
| 1 | 339 | **1.000** | 515/560 | 92.0% | 262 / 253 |
| 2 | 299 | **1.000** | 515/560 | 92.0% | 254 / 261 |
| 3 | 313 | **1.000** | 516/560 | 92.1% | 271 / 245 |
| 4 | 290 | **1.000** | 516/560 | 92.1% | 243 / 273 |

**The unstructured control solves `retina/xor` easily.** Every seed reaches
perfect balanced accuracy well inside the 1000-gen budget (290–456 gens) —
`retina/xor` is not a hard search problem for direct encoding the way `retina/and`
was a hard *representability* problem for experiment 1 at low K. No shortcut
confound here (xor is balanced, no one-side freebie), so this is a genuine
full-task solve, not a plateau artifact.

**Density stays ~92–94% (near fully-connected) in every seed.** With no
structural prior toward regular/modular wiring, CMA-ES converges on dense,
roughly balanced exc/inh solutions. This is the number to compare against a
modularity score once the metric exists (still not built — see
`../experiment_1/RESULTS.md`'s open threads): a near-fully-connected solved
network is a strong prior that the control is *not* modular, which is the
contrast the thesis needs against experiment 1's regular/compressed solutions.

**Caveat:** one fixed-goal run, no modularity metric yet, so "not modular" here
is an inference from density alone, not a measured score. `--mvg` (the known
positive-control pressure) not yet tried on this encoding.

## MVG (xor/and switching), n_hidden=20, 3 seeds, margin fitness

```
python train.py --task retina --mvg --mvg-ops xor,and --switch-interval 20 --generations 2000 --n-seeds 3 --fitness margin
```

`--mvg` originally hardcoded the switch cycle to `and`/`or` (the classic
Kashtan-Alon pairing); added a `--mvg-ops` flag (comma-separated, cycles in
order, default `and,or` so existing behaviour is unchanged) so the cycle can
include `xor` — needed to compare against the fixed-goal `retina/xor` run
above. `--mvg` disables early stopping (the target never stops moving), so
every seed runs the full 2000-generation budget.

| seed | reported "best accuracy" | edges used | density | exc(+) / inh(-) |
|---|---|---|---|---|
| 0 | 1.000 | 525/560 | 93.8% | 272 / 253 |
| 1 | 1.000 | 520/560 | 92.9% | 274 / 246 |
| 2 | 1.000 | 521/560 | 93.0% | 242 / 279 |

**The reported "best accuracy" is misleading here — it's driven entirely by
`and`, not `xor`.** The per-seed "best" is the highest single-generation
population-best accuracy seen *on whichever op was active that generation*; it
does not mean both targets were solved. Across all 300 logged xor-phase
generations (3 seeds × 100 each), best-in-population accuracy on xor ranged
from **0.327 to 0.919** and never once reached 1.0. `and`, by contrast, hit
exactly 1.000 repeatedly (generations 750, 1190, 1470, 1670, 1710, 1790, 1950,
1999 across the three seeds).

**xor does not evolve faster under xor/and MVG — if anything it never converges
at all.** Immediately after each switch *into* xor, best accuracy crashes to
0.33-0.44 (below chance), climbs to ~0.7-0.9 over the following ~10
generations, then gets yanked back to `and` before it can consolidate. This
pattern is stable from generation 0 through generation ~2000 — no sign of the
oscillation narrowing or xor plateauing solved, in all 3 seeds. Contrast with
the fixed-goal run above, which solved xor cleanly in 290-456 generations and
stayed there.

**Why xor/and is a much harsher pairing than and/or.** AND and OR agree on 3 of
4 truth-table rows (only differ when exactly one side is active) — a mild
perturbation to switch between. AND and XOR agree on only 1 of 4 (both are 0
only when left=right=0; everywhere else they're exact opposites: XOR=1&AND=0
when exactly one side fires, XOR=0&AND=1 when both fire). With a single shared
CMA-ES mean/covariance and no structural separation between the two
sub-problems, the population can't represent two near-opposite decision
surfaces at once, and a 20-generation window isn't enough to fully readapt
before being switched away again — so it seesaws instead of learning either
target robustly.

**Caveat / likely confound:** this result may say more about "xor and and are
adversarial partners" than about "does switching pressure help modularity
emerge." `xor,or` agrees on 3/4 rows (same agreement fraction as the classic
`and,or` pair — they only disagree when both sides fire) and would be the
fairer "gentle" partner to test the original speed/modularity question without
the adversarial-pairing confound.

## retina/and FG vs and/or MVG, n_hidden=20, 5 seeds each — apples-to-apples with NDP (2026-09-08)

Run to directly compare against 5 FG + 5 MVG runs just done on NDP (`../../NDP`,
`environment: KA_retina`, which duplicates this same stand-in `retina` task —
see `../shared_tasks.py`). NDP's configs confirmed `balanced_fitness: true` and
`operation: and` (FG) / `mvg_ops: [and, or]` (MVG), so this reproduces the same
task/op/metric here, with CMA-ES direct encoding instead of NDP's developmental
growth. `--no-early-stop` used on **both** arms (required for a fair
FG-vs-MVG comparison — otherwise FG exits the moment it solves while MVG runs
the full budget, confounding any density/generations comparison).

```
python train.py --task retina --operation and --n-hidden 20 --n-seeds 5 --fitness margin --no-early-stop --generations 2000 --no-open

python train.py --task retina --mvg --mvg-ops and,or --switch-interval 20 --n-hidden 20 --n-seeds 5 --fitness margin --no-early-stop --generations 2000 --no-open
```

### FG (`and` only)

| seed | gen solved | best-DNA density | final-DNA density |
|---|---|---|---|
| 0 | 250 | 90.4% | 94.5% |
| 1 | 200 | 87.5% | 94.5% |
| 2 | 280 | 88.8% | 95.7% |
| 3 | 270 | 88.4% | 93.8% |
| 4 | 210 | 89.5% | 95.5% |

5/5 solved cleanly — same dense/unstructured pattern as the earlier retina/xor FG run.

### MVG (`and` ↔ `or`, switch every 20 gens)

Per-op best-in-population accuracy across all 2000 logged generations (the
single reported "best accuracy" is misleading on its own, same caveat as the
xor/and run above — it only reflects whichever op happened to be active):

| seed | AND: min/mean/max | OR: min/mean/max | % gens at 1.000 (AND / OR) |
|---|---|---|---|
| 0 | 0.693 / 0.858 / 1.000 | 0.643 / 0.835 / 1.000 | 31% / 27% |
| 1 | 0.698 / 0.856 / 1.000 | 0.643 / 0.833 / 1.000 | 38% / 40% |
| 2 | 0.698 / 0.855 / 1.000 | 0.646 / 0.837 / 1.000 | 27% / 38% |
| 3 | 0.705 / 0.854 / 1.000 | 0.654 / 0.838 / 1.000 | 2% / 2% |
| 4 | 0.699 / 0.858 / 1.000 | 0.643 / 0.832 / 1.000 | 19% / 17% |

**Confirms the "and/or is the gentle pairing" hypothesis from the xor/and run
above.** Unlike xor/and (xor never once hit 1.0 in 300 logged generations,
crashing below chance on every switch), here **both AND and OR repeatedly hit
exactly 1.000** in most seeds, and the post-switch dip (~0.64–0.70) stays
above chance rather than crashing below it.

**But still no real convergence over the run:** early-half (gen<1000) vs.
late-half (gen≥1000) per-op means are essentially flat (e.g. seed 0 AND:
0.862 early → 0.855 late) — 2000 generations of switching doesn't narrow the
oscillation into a stable joint solution; it's a steady-state wobble from
early on.

**Density — MVG did not go sparser than FG. If anything, slightly denser:**

| | FG (best-DNA) | FG (final-DNA) | MVG (best-DNA) | MVG (final-DNA) |
|---|---|---|---|---|
| density range | 87.5–90.4% | 93.8–95.7% | 87.1–93.6% | 96.1–97.7% |

**Conclusion: no evidence that goal-switching pressure alone pushes an
unconstrained direct encoding toward a sparser/modular solution.** Both arms
converge dense; MVG's final density is if anything the highest number in the
table. Consistent with there being no structural bias toward economizing
connections in this encoding — see "Open threads" below for the fan-in-cap /
edge-budget constraint this motivates.

### Weight-magnitude structure of the (dense) FG/xor solutions — checked directly

Loaded the 5 saved `retina_seed{0..4}_best_dna.eqx` (the earlier retina/xor FG
run) and histogrammed `|weight|` across all 560 possible edges per seed. **Not
a bimodal "some strong, rest dead" split, and not a uniform blob of saturated
weights either — a graded continuum:**

| \|w\| bucket | avg. fraction of edges (5 seeds) |
|---|---|
| < 0.05 (below prune threshold) | ~7% |
| 0.05–0.2 | ~24% |
| 0.2–0.5 | ~38% |
| 0.5–1.0 | ~29% |
| 1.0–2.0 | ~5% |
| > 2.0 | ~0% |

The ~92–94% "density" figure is real (only ~7% of edges are near-zero), and
the rest spread smoothly from small to moderate magnitude with a peak around
0.2–0.5 — nothing in the magnitude distribution itself hints at module
boundaries (no small set of dominant edges standing out against a sea of
near-zero noise).

## Open threads

- Run `--mvg --mvg-ops xor,or --switch-interval 20 --generations 2000` — the
  gentler xor pairing (3/4 truth-table agreement, like the classic and/or
  pair), to separate "switching pressure helps" from "xor/and are adversarial."
  **Superseded by the and/or run above** (same 3/4-agreement gentleness,
  already run) — xor/or itself is now lower priority unless AND/OR-specific
  behavior needs ruling out.
- **No connectivity constraint exists in this model at all** — every one of
  the ~560–581 possible edges is free. `kashtan_alon/` (the faithful KA 2005
  reproduction) shows removing its fan-in cap drives density to the complete
  graph the same way exp 2 does here, and that the cap is likely load-bearing
  for its one clean MVG>FG (Q_m) result. Adding a KA-style per-neuron fan-in
  cap (or a flat total-edge budget) to `shared_direct_model.py` is the natural
  next step before concluding direct encoding + MVG "doesn't do modularity" —
  right now it hasn't been tested under any constraint that would let it.
- Once a modularity metric exists, score these saved DNAs
  (`runs/retina_seed{0..4}_best_dna.eqx`) directly rather than inferring from
  density.
