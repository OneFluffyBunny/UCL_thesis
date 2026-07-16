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

## Open threads

- Run `--mvg --mvg-ops xor,or --switch-interval 20 --generations 2000` — the
  gentler xor pairing (3/4 truth-table agreement, like the classic and/or
  pair), to separate "switching pressure helps" from "xor/and are adversarial."
- Once a modularity metric exists, score these saved DNAs
  (`runs/retina_seed{0..4}_best_dna.eqx`) directly rather than inferring from
  density.
