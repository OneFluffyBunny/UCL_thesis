# Experiment 1 — results & observations log

Running lab notebook for the cell-type-encoding framework (see `../README.md` for
the model). Newest entries at the bottom. Balanced accuracy throughout (chance =
0.5). Bipolar inputs {-1,+1}. Search = CMA-ES over the genome unless noted.

---

## ⚠️ Methodological caution: check every task for shortcuts & imbalance

Before trusting *any* result on a task, characterise the task itself — a plateau
may be an artifact of the metric/task, not of the model or the search:

1. **Class balance.** Print `P(y=1)`. If it's far from 0.5 the task is imbalanced;
   always use *balanced* accuracy (chance = 0.5) and remember a "high" score may
   still be a degenerate one-class-ish predictor.
2. **Partial-solution shortcuts.** Enumerate what *trivial / partial* predictors
   score (constant, a single input, one sub-feature, one module). If some shortcut
   already scores near the observed plateau, the model is probably just finding
   THAT, not solving the task. (Concrete bite: `retina/and` gives **0.848 balanced
   for computing only ONE side** — see the curriculum section — so its "0.85 wall"
   was the trivial one-module solution, and the whole curriculum comparison was
   confounded.)
3. **Gen-0 best-of-population.** If a random population already sits at the
   plateau, no learning is being measured — the plateau is a free shelf.
4. **Monotone-representability.** Ask whether a *monotone* boolean function can
   already solve the task. If it can, the task never forces an inhibitory weight,
   and a network can score perfectly while using only half the sign space. The
   exact ceiling is computable — minimum flips to make the target monotone is a
   min-cut on the boolean lattice (isotonic regression on a partial order):

   | task / op | best constant | **monotone ceiling** | flips needed |
   |---|---|---|---|
   | `retina_ka2005` / and | 0.750 | 0.891 | 28/256 |
   | `retina_ka2005` / or | 0.750 | 0.875 | 32/256 |
   | `retina_ka2005` / xor | 0.500 | 0.691 | 79/256 |
   | `retina` (stand-in) / and | 0.809 | **1.000** | **0/256** |
   | `retina` (stand-in) / or | 0.684 | **1.000** | **0/256** |

   **The stand-in `retina` task under AND and OR is exactly monotone** — a second,
   independent shortcut on top of the 0.848 one-side freebie. Another reason to
   prefer `retina_ka2005`. (Method validated on `majority` → 1.000 and `x0 XOR x1`
   → 0.750.) ⚠️ Do NOT read the ceiling as an explanation of observed plateaus:
   the 0.750 plateau sits well *below* 0.891, so monotonicity is not what binds
   there — see the sign audit below.

Rule of thumb: a task is only a good modularity probe if reaching high accuracy
*requires* the modular structure you're trying to study. Prefer tasks with no
single-module shortcut and near-balanced classes (e.g. `retina/xor`).

## Difficulty staircase (single fixed goal)

| task | target | best balanced acc | notes |
|---|---|---|---|
| `copy` | bit0 | **1.000** (gen 1) | linearly separable, trivial |
| `and2` | bit0 ∧ bit1 | **1.000** (gen 1) | linearly separable, trivial |
| `left` | (p0∧p1)∨(p2∧p3) | **0.929** (plateau) | fully converged, wall below |
| `retina` AND | [(p0∧p1)∨(p2∧p3)] ∧ [(p4∧p5)∨(p6∧p7)] | **~0.85** (plateau) | K=4 training config |

`copy`/`and2` are trivial: deterministic exact fitness (no winner's-curse), static
brain, linearly separable. The interesting failures are `left` and `retina`.

---

## Two distinct walls (the main finding)

The plateaus on `left` and `retina` have **different causes**. Don't conflate them.

### `left` → REACHABILITY wall (perfect brain exists, search can't find it)

- **Oracle** (`oracle.py --task left`): a perfect genome EXISTS even at the training
  width — hand-wired ideal brain scores 1.000, OLS-fit of `g` reproduces it with
  residual **0.000**, and the resulting "oracle DNA" run through real inference
  scores **1.000**, at both `g_width=128` and `g_width=16`. So representability is
  fine.
- **Basin probe** (`reachability.py`): CMA-ES seeded *exactly on* the optimum (r=0)
  does **not** stay — it drifts to 0.857 (worse than a random start's 0.929), and
  **0% of perturbed starts recover** at any radius tested. The optimum is a needle
  narrower than the training mutation scale (`sigma_init=0.1`).
- **Why:** fitness is piecewise-constant (staircase), so the peak has no
  surrounding gradient/basin; CMA-ES never evaluates its mean and gets no inward
  signal, so it wanders off. Adam-on-accuracy stalls identically → not a CMA-ES
  artifact, it's the landscape.

### `retina` AND at K=4 → plateau, cause NOT yet proven (suspected representability)

- A natural decomposition uses **6 hidden cell-types**: 4 AND-detectors
  (A=p0∧p1, B=p2∧p3, C=p4∧p5, D=p6∧p7), each watching a *different* input pair,
  plus 2 OR-combiners (Lft=A∨B, Rgt=C∨D) feeding the output's final AND.
- Same-type hidden neurons are **position-blind clones** (identical wiring), so the
  four disjoint detectors force ≥4 distinct types; and `(A∨B)∧(C∨D)` is not linearly
  separable, so the output neuron alone can't combine them → an *argument* that <6
  is hard. **This is reasoning, not a proof.**
- **What the oracle actually shows:** `build_oracle_retina` hand-wires a **6-type**
  solution and asserts K≥6, so it only proves **6 is *sufficient*** (stage-1 = 1.000
  at K≥6). It does **NOT** prove 4 is insufficient — it refuses to build below 6, so
  we have **no oracle evidence at K=4/5**. The K=4 plateau is so far only *empirical*
  (CMA-ES + Adam both cap ~0.85); "hard to find" ≠ "not representable".
- ⚠️ Earlier notes here said "K=4 = representability wall / not constructible". That
  overclaimed: "not constructible" meant *my* 6-type wiring can't be built at K=4,
  not that *no* brain exists. Downgraded to **suspected, untested**. A curriculum or
  any search reaching >0.85 at K=4 would constructively disprove it.

| oracle config | stage 1 (ideal wiring) | stage 3 (oracle DNA via `g`) |
|---|---|---|
| retina K=4 | not tested (oracle asserts K≥6; **not** a proof of impossibility) | — |
| retina K=6, g_width=16 | **1.000** | 0.801 |
| retina K=6, g_width=128 | **1.000** | 0.882 |

> Caveat: stage-3 uses OLS (linear last-layer fit over *fixed random* `g` features),
> which **underestimates** `g`'s true capacity — 0.80–0.88 is a lower bound, not
> `g`'s ceiling. It rises with width, and a full nonlinear `g` fit would do better.

---

## Margin surrogate fitness (`--fitness margin`)

Added a `--fitness {accuracy,margin}` flag (default `accuracy`, unchanged). `margin`
= balanced, hinged (cap 0.5) signed-margin on the raw tanh output for CMA-ES
*selection*; the decision is still `sign(output)` and accuracy is still what gets
logged / early-stopped / reported. Motivation: raw balanced accuracy is
piecewise-constant (16 inputs → ~17 discrete values) = a staircase with flat
plateaus and needle optima → both gradient-free and gradient methods stall. Margin
restores a smooth ramp (standard surrogate-loss move).

### retina AND, K=4, n_hidden=20, 5 seeds — margin vs accuracy

| | accuracy (control) | margin |
|---|---|---|
| best-acc per seed | 0.853, 0.853, 0.853, 0.853, 0.860 | 0.867, 0.858, 0.853, 0.860, 0.853 |
| **mean** | **0.854** | **0.858** |
| max | 0.860 | 0.867 |
| weight signs | every seed mono-signed (all-exc *or* all-inh) | mixed exc/inh on 3/5 seeds |
| density | 100% every seed | 39–100% (sparser when mixed) |

**Margin does not raise the K=4 ceiling** (+0.004 mean = noise) — expected, since
nothing representable exists to reach. But it visibly **changes search character**:
raw accuracy collapses to a single-sign, fully dense brain; margin explores mixed
excitatory/inhibitory, sparser structures. Smoothing frees exploration; it just
needs a representable target. Worth keeping.

---

## retina AND at K=6 — capacity lifted, wall stays (key result)

The oracle proved retina-AND is representable at K≥6 (ideal brain scores 1.000).
So we retrained at **K=6, n_hidden=30** (≈5 neurons/type), 3 seeds, 600 gens, both
fitness modes, expecting the 0.85 wall to break.

| retina AND, n_hidden=30, 3 seeds | K=4 (prior) | **K=6** |
|---|---|---|
| accuracy — per seed | 0.853, 0.853, 0.853, 0.853, 0.860 | 0.853, 0.853, 0.853 |
| accuracy — mean | 0.854 | **0.853** |
| margin — per seed | 0.867, 0.858, 0.853, 0.860, 0.853 | 0.853, 0.859, 0.865 |
| margin — mean | 0.858 | **0.859** (max 0.865) |

**Lifting K from 4→6 changed nothing.** Same ~0.85 plateau, same behaviour (accuracy
seeds mono-signed & mostly dense; margin seeds sometimes sparse/mixed, e.g. one
margin seed at 29% density, all-inhibitory, 0.865).

**Revised interpretation — retina-AND has *both* walls, stacked:**
- At **K=4** the perfect brain is *suspected* not representable (see caveat above —
  argued, not proven). Going to K=6 makes it *provably* representable…
- …but **not sufficient**: at K=6 the perfect brain now exists in the family, yet
  CMA-ES still can't reach it. Underneath the representability wall sits a
  **reachability wall** — the *same* kind that pins `left`, but deeper, because
  retina-AND requires coordinated discovery of a 3-layer modular structure (4
  detectors + 2 combiners + output AND) via one shared `g` and the cell-type
  abundances simultaneously.

This is the cleanest motivation yet for the evolvability hypothesis: the compressed
encoding *can represent* modular solutions, but plain CMA-ES on a *static* goal
cannot *reach* them.

> Not yet disambiguated: the K=6 cap could also be `g`-capacity at width 16 (the
> oracle's OLS lower bound was 0.80@w16 / 0.88@w128). To separate reachability from
> `g`-capacity: retrain K=6 with wider `g` (64/128); run the basin probe at K=6; do
> a full nonlinear `g` fit in the oracle. If wider `g` still caps at 0.85 → it's
> reachability.

---

## Notes / levers

### The K-neuron equivalence (what the encoding actually does) ✅ agreed

Hidden neurons carry **no positional code** (their feature is
`[type_embedding | zeros(pos_dim) | role_onehot]`), so two hidden neurons of the
same type have *byte-identical* features. Since `w_ij = g(feat_i, feat_j)`, they
get identical incoming weights, identical outgoing weights and identical bias
(`type_bias[t]`) ⇒ **identical activation at every timestep, forever.** They are
exact clones, not merely similar.

So every type collapses to ONE state variable and the whole brain reduces to `K`
of them:

```
a_t ← σ( Σ_i x_i·w_it  +  Σ_s n_s·a_s·w_st  +  b_t )    n_s = m_s (s≠t), m_t−1 (s=t)
```

> **Experiment 1 with `K` types and `n_hidden` neurons is functionally equivalent
> to a `K`-neuron recurrent network with gain-scaled edges.** K=4 with n_hidden=1000
> is still, functionally, a 4-hidden-unit network. `n_hidden` enters *only* through
> the multipliers `m_t`.

This is also the *mechanism* behind the compression: the genome stays O(K) no
matter how many neurons, precisely because extra neurons add no new distinct
wiring to specify.

### Neuron counts act as GAIN — and evolution only controls the RATIOS ⭐

Clones are **not** free/no-ops (an earlier note overstated this). If type `t` has
`m_t` clones all holding activation `a_t`, a downstream neuron receives
`m_t · a_t · w_tj` — so **the clone count is a gain multiplier** on that type's
contribution. That is exactly why `abundance` is an evolved gene: it is a real
lever.

But note *what* evolution actually controls. `abundance` is softplus-**normalised**
(`p = softplus(a)/Σ softplus(a)`, then `m_t ≈ p_t · n_hidden`), so:

- **evolution's lever = the ratios `p_t`** (relative gains between types);
- **`n_hidden` = a fixed, uniform scale** on all gains, set by the experimenter,
  not evolved. It buys headroom on effective coupling strength (each `w` is
  tanh-bounded to [-1,1], so `m_t` is the only way to exceed unit coupling), but
  it adds **no new distinct roles**.

**Upshot:** `n_hidden` is a lever for **gain**, not for **functional diversity**.
`K` is the lever for distinct roles. For tasks like retina that need many distinct
sub-computations (≥6 for AND, ~8 for XOR), *roles* are the binding constraint —
which is why K=4 failed no matter how many neurons were thrown at it. The normal
ML intuition "more neurons ⇒ harder tasks" holds in experiment 2 (direct encoding)
and is precisely what this bottleneck deliberately severs.
- **Compute:** local machine is CPU-only JAX; the code is GPU-ready (jit + vmap over
  population × inputs) for the remote UCL lab box, but the problem is small so GPU
  gain is modest until pop / `n_hidden` / input count scale up.

## Discussion: K vs modularity (hypothesis, not yet measured)

`K` (number of hidden cell-types) is the size of the **wiring vocabulary**:
same-type hidden neurons are position-blind clones, so `K` caps how many distinct
connectivity roles the brain can have. Its relationship to modularity is a
double-edged knob:

- **Low K = tight bottleneck ⇒ modularity is *forced*.** With few types the wiring
  must be block-structured (a handful of reusable roles), so the brain is modular
  "for free" — but there's a capacity ceiling: if `K` < the number of functional
  roles the task needs, the solution isn't representable at all (cf. retina-AND at
  K=4). Too low and modularity is trivial *and* the task fails.
- **High K → n_hidden = loose bottleneck ⇒ modularity is merely *permitted*.** As
  `K` approaches the neuron count, every neuron can have its own identity and the
  encoding approaches a direct/unstructured one — an arbitrary, non-modular graph
  becomes expressible. The structural prior toward modularity vanishes; modularity
  now has to come from *selection*, not the encoding. (This matches the intuition
  that high `K` can "devolve into a non-modular mess" — high `K` *allows* the mess;
  whether it happens depends on the pressure.)
- **Sweet spot: `K` ≳ the task's functional-role count.** For retina-AND that's ~6
  (4 detectors + 2 combiners). Just enough to represent the modules and no more =
  the most compressed representable point.

**Design implication (important for the thesis, not rigging the result):** the
README notes the encoding must be able to express *both* modular and non-modular
brains, or the modularity finding is rigged. That means the scientifically
meaningful regime is `K` **high enough that non-modularity is expressible**, where
we then test whether selection (e.g. modularly-varying goals) *chooses* modularity
anyway. Very low `K` proves nothing (modularity is imposed by the encoding).

**Possible trade-off:** higher `K` may also *aid reachability* (more redundant
paths / degenerate solutions for search to stumble into) at the cost of the found
solutions being less modular when there's no pressure for it. So `K` may trade
modularity against evolvability — precisely what the `--mvg` experiment should probe.

**Gap:** we currently have no modularity *metric*. To study any of this we need to
compute a structural score on the grown weight matrix (e.g. graph modularity `Q` /
block structure, or a task-aware left/right module separation). Not yet built.

## Curriculum vs cold-start (`curriculum.py`, K=4)

Question: does an incremental curriculum (`left` → `retina/and`) reach the hard
task *faster* and produce *more modular* brains than cold-starting on the hard
task at equal budget? Paired design: both arms of a seed share the same seed
(same init genome + init CMA state), so within-seed differences are attributable
to the curriculum, not init luck.

Setup: K=4, n_hidden=12, `left:200 → retina/and:600` vs cold `retina/and:800`,
`--fitness margin`, `--sigma-restart 0.1`, 5 seeds.

| metric (mean over 5 seeds) | curriculum | cold |
|---|---|---|
| best balanced acc on retina | 0.862 | 0.858 |
| gens to 0.90 | never | never |
| L/R lateralization `frac_lat` | 0.00 (all seeds) | 0.00 (all seeds) |
| weight sign | mixed exc/inh (4/5) | single-sign all-exc/all-inh (4/5) |

**Clean null.** No advantage on speed, ceiling, or modularity. Paired acc diff
+0.004 (noise), 2–2–1 seed split. Neither arm made a modular brain. Only robust
difference is orthogonal to the hypothesis: curriculum lands on mixed exc/inh
wiring, cold collapses to a single sign. Curriculum even *starts* the retina
stage slightly worse (~0.80) than a fresh net (0.843) — a whiff of negative
transfer at the handoff, not a head start.

### Why the null: retina/AND has a "one-side shortcut" (key caveat)

`retina/and` is **class-imbalanced** (P(y=1) = 0.191, only 49/256 inputs) and
factorises as `left_feat ∧ right_feat`. Because every positive requires
`left_feat = 1`, a brain that computes **only one side and ignores the other**
already scores **0.848 balanced accuracy** (TPR = 1.0; errs only on the rare
left-on/right-off negatives):

| predictor on retina/and | balanced acc |
|---|---|
| constant 0 / 1 | 0.500 |
| compute only `left_feat` (or only `right_feat`) | **0.848** |
| single detector `p0∧p1` | 0.699 |

A random population reliably finds this one-side attractor: best-of-64 at gen 0 =
**0.843 for every seed** (evosax's CMA initial mean is zeros, so all seeds search
around the same point; the initial *populations* do differ — verified, no leak).
By contrast `left` (P(y=1)=0.44, balanced, no shortcut) starts ~0.77 and climbs
gradually.

**So the ~0.85 "plateau" is the trivial one-module solution.** Both arms were
stuck at *half the task*; the real difficulty (forming the SECOND module and
combining them) yields almost no balanced-accuracy reward under `/and`, so search
never pays to leave 0.85. The null is uninformative about modularity — the task
never demanded it.

**Next:** switch to `retina/xor` (P(y=1)≈0.49, no one-side shortcut — neither side
alone predicts the label), which genuinely forces both modules and makes any
break past chance-of-one-side a real modularity signal.

## Synaptic gate (`--w-threshold`) — an absolute gate does NOT control density ⭐

Until 2026-08 nothing in exp 1 could produce a zero weight: `g` is a continuous
MLP, the only zeros came from the fixed role mask, and the logged "density" was
an analysis-time count of `|w| > --prune-threshold` that never touched the
forward pass. `--w-threshold` added a real gate — `|g(feat_i,feat_j)| < t` → 0,
applied to the U×U signature block, with the network **evaluated** on the gated
matrix, so sparsity became part of the phenotype.

**It does not work.** First FG-vs-MVG pair on `retina_ka2005` (seed 0 both arms,
K=6, n_hidden=24, pop 64, 2000 gens, raw accuracy, `--no-early-stop`,
`--w-threshold 0.2`, ~4 min/arm):

| | FG (`and`) | MVG (`and,or`, interval 20) |
|---|---|---|
| best / final accuracy | 0.875 / 0.875 | 0.883 / 0.848 |
| density, gen 0 → final | 0.0% → **77.1%** | 0.0% → **100.0%** |
| ungated \|w\| median | 0.530 | **1.000** (saturated) |
| ungated \|w\| min | 0.007 | **0.511** |
| final σ | 0.042 (converged) | 0.268 (still exploring) |
| % positive weights | 78.6% | 34.4% |

At init this seed's max `|w|` is 0.234, so both arms start at ~0% density — then
evolution inflates `g`'s output scale 4–10× and walks straight through the gate.

**The threshold just forces evolution to move to stronger connections; it does
nothing about the density explosion.** The gate is a *one-time hurdle*: nothing
in the fitness penalises density, larger weights mean stronger signal, so
crossing is paid for once and the gate never binds again. Raising `t` buys delay,
not sparsity. Applying every threshold to the evolved final brains:

| t | FG density | MVG density |
|---|---|---|
| 0.2 | 77.1% | 100.0% |
| 0.6 | 38.5% | 82.8% |
| 0.9 | 22.3% | 77.3% |
| 0.99 | 9.6% | 77.3% |
| 0.999 | 6.2% | 77.3% |
| 1.0 | 0.0% | 60.9% |

MVG is **flat from 0.6 to 0.999** because 77.3% of its synapses have `|w| > 0.999`
— tanh-saturated, 60.9% at exactly 1.0 in float. The distribution is bimodal
(saturated, or well under 0.5), so no threshold has anything left to cut. There
is no useful window: below ~0.9 the gate is outrun, at ≥1.0 the brain is
permanently empty (tanh never reaches 1), and in between it is flat.

⚠️ **The gate is confounded with the arm.** MVG defeats it *harder* than FG (100%
vs 77%): FG converges (σ 0.042) and stops inflating, while MVG's moving goal keeps
pushing weights to saturation. So an absolute gate systematically yields denser
MVG brains — and since unweighted `Q` falls with density, this biases any Q
comparison **against** MVG, i.e. in the exact direction that manufactures a false
null. Both brains here are 77–100% dense, so `Q ≈ 0` by construction and **this
pair cannot answer the modularity question at all.**

**Next:** a *budgeted* (top-k) gate — keep the k largest-|w| signature pairs.
Density becomes a controlled constant, identical across arms by construction,
which removes it as a confound instead of merely resisting inflation. A relative
gate (`frac × max|w|`) is scale-free but useless here: with MVG's median at 1.000
it would still keep nearly everything.

## Synaptic budget (`--synaptic-budget` / `--shrink`) — a density control that WORKS ⭐

*(implemented 2026-08-10, commit `b1140df`; supersedes the "budgeted top-k gate"
plan above, which was never built — an in-budget normalisation turned out to be
the better shape.)*

Each neuron gets a fixed total *incoming* |weight| `S`, shared out over its
synapses, so `sum_i |w_iv| = S` for every non-input neuron `v`. Density stops
being free: an extra connection dilutes the ones already there. `--shrink τ`
zeroes any synapse below `τ ×` its target's **own** mean incoming `|g|`, applied
*before* the share-out. The two are a pincer — shrink alone is escapable by
flattening (make everything equal, nothing falls below the mean), and flat under
a fixed budget means no signal at all, so the only way out is contrast.

**Why relative, not absolute.** The gate above failed because evolution inflated
`g` 4–10× and walked through it. Nothing can inflate above its own mean:
multiplying `g`'s output layer by 100 leaves the budgeted brain **bit-identical**
(max |diff| 3e-8), while the same 10× inflation takes a *gated* brain from
**36 → 744 edges**. Verified in the 40-check suite, along with the budget
invariant under skewed clone counts (the multiplicity trap: `g` is per-*signature*
but a budget is spent on *synapses*, and the hidden diagonal carries `count − 1`).

⚠️ Under a budget, `g` loses its `tanh` output activation (redundant after
renormalisation, and a saturation attractor). **A genome saved with a budget must
be reloaded with the same setting** — `g`'s shape depends on it.

⚠️ `brain_stats` counts exact zeros under budget/gate instead of applying
`--prune-threshold`: at fan-in 30 a budget of 1 puts every weight near 0.03, so
the default 0.05 reported **80 edges for a brain that actually has 768**.

### Two FG runs (`retina_ka2005`/and, K=6, n_hidden=24, pop 64, 2000 gens, accuracy fitness, `--no-early-stop`, 3 seeds; 2–5 min/seed)

`runs/budget_fg` (S=2, τ=0.8) and `runs/budget_fg_b4s0.9` (S=4, τ=0.9):

| | S=2, τ=0.8 | S=4, τ=0.9 |
|---|---|---|
| final accuracy | 0.812, 0.812, 0.815 | 0.812, 0.820, **0.885** |
| density gen 0 | 87.5 / 75.7 / 90.1% | 85.4 / 94.9 / 79.3% |
| density final | 55.1 / 44.9 / 28.1% | 39.1 / 41.4 / 37.6% |
| density min | 25.8 / 21.4 / 28.1% | 20.1 / 20.1 / 29.6% |
| sign split (final) | +423/−0, +44/−301, +216/−0 | +234/−66, **+0/−318**, +89/−200 |
| median \|w\| | 0.055 / 0.049 / 0.156 | 0.260 / 0.119 / 0.305 |
| budget invariant | exact (2.000–2.000) | exact (4.000–4.000) |

**Density falls instead of exploding.** The gate went 0% → 77–100%; the budget
goes 76–95% → 28–55% in every seed. The invariant holds *exactly* after 2000
generations of CMA-ES, which is the real test — evolution cannot drift off it.

**The weight distribution is finally graded**, which the gate never achieved.
S=4 seed 0: p5 0.025, p25 0.129, p50 0.260, p75 0.520, p95 0.792. Compare the
gated MVG arm, where 77% of synapses sat at |w| > 0.999 and no threshold had
anything left to cut.

**Sign collapse is budget-dependent (new).** At S=2 all three seeds are
single-sign (2 all-excitatory, 1 all-inhibitory); at S=4 two of three are mixed.
Hypothesis: with a fixed total |incoming|, opposite-sign synapses cancel at the
target, so mixed signs spend budget to produce less net drive — i.e. the budget
*penalises the sign diversity that non-monotone functions need*. Consistent with
the accuracy: the only seed to break the 0.812 plateau (0.885, above the gate
run's 0.875 and well clear of the 0.750 one-module cap) is a mixed-sign seed.
Not established — n=3, and S=4 seed 0 is mixed but still stuck at 0.812, so sign
diversity looks necessary rather than sufficient. Distinct from the accuracy-vs-
margin sign audit below, which is about *fitness*, not the budget.

### ⚠️ Newman `Q` found "modularity" that is NOT the retina's

S=4 seed 2 scores `Q_weighted = 0.20` (highest exp 1 has produced) and
`role_segregation` reports 8/24 neurons lateralized, mean |s| = 0.361. Both are
misleading. Scoring the **planted** left/right split — the question the task
actually asks — gives `r = +0.085`, crosstalk 0.915, **p = 0.87**: chance.

Per-type left/right drive `s = (L−R)/(L+R)` explains it:

| type 0 (n=3) | type 1 (n=2) | type 2 (n=5) | type 3 (n=2) | type 4 (n=5) | type 5 (n=7) |
|---|---|---|---|---|---|
| +0.79 | −0.07 | **+0.99** | +0.21 | +0.05 | +0.07 |

There is a genuine **left-specialised cell type** (type 2's five clones read the
left half almost exclusively) and *nothing on the right*. One module plus a
general pool — which inflates mean |s| while the planted split sits at chance.
Detected communities cut across the halves as always (community 0 holds inputs
{0,2,3,6}, community 1 holds {1,7}).

➡️ **Use `left_right_q` at the planted split as the PRIMARY metric for retina
claims; report `newman_q` as secondary only.** An unlabelled `Q` will let us
announce modularity along the wrong axis.
*(Also: `left_right_q`'s `assign='majority'` score is unusable on these brains —
denominator `q_max − q_rand` = 0.0075 produced a meaningless −1.17. Quote `r` and
`p`. See the loose `abs(denom) > 1e-9` guard at `qmetrics/metrics.py:695`.)*

### Was 2000 generations enough? Marginally — and NOT for MVG

| run | seed | last improvement | gens flat after | σ end |
|---|---|---|---|---|
| S=2 | 0 / 1 / 2 | 125 / 75 / 975 | 1874 / 1924 / 1024 | 0.051 / 0.078 / 0.068 |
| S=4 | 0 / 1 / 2 | 225 / **1375** / **1150** | 1774 / 624 / 849 | 0.069 / 0.058 / 0.048 |

Three of six seeds made their last gain after gen 975, including **both** seeds
that beat the plateau. σ falls only 0.097 → 0.048–0.078 (a 2× reduction), so
CMA-ES is still exploring at the end; density still oscillates ±3–9 points over
the final 500 generations.

### ⭐ We are 13× under Kashtan-Alon's evaluation budget

From `kashtan_alon/PAPER_SPEC.md` (verified quotes): population **S = 600**
(line 57); MVG solves in **2,800 generations** (+9,500/−600) and FG in **21,000**
(+29,000/−3,600) (line 65).

| | KA gens | × pop 600 = evals | our equivalent at pop 64 |
|---|---|---|---|
| MVG solved | 2,800 | 1.68M | **26,000 gens** |
| FG solved | 21,000 | 12.6M | 197,000 gens |

Our runs so far: 2,000 × 64 = **128k evals** — 13× under KA's *median* MVG budget.
**This reframes the 0.812 plateau as probably reachability, not representability**
(cf. the two-walls section): a run 13× under the reference budget stalling is what
under-budgeting looks like.

**The goal-switch epoch `E` has the same problem.** KA's E=20 at pop 600 is 12,000
evals per goal; ours at pop 64 is **1,280** — 9× less search per goal. Matching
their effort needs **E ≈ 190**. Note `PAPER_SPEC.md:51` flags E=20 as ⚠️ AMBIGUOUS:
it is quoted for KA's *circuit* experiment and was **not** found restated for the
neural network, so our E=20 on the retina was always a reconstruction, not a
quoted value — and exp 1 is not a KA reproduction anyway (g-encoding, CMA-ES, 431
continuous params vs their ±1 direct genome).

➡️ **Do not pick `E` by which value maximises modularity** — that selects the
parameter to produce the result being tested. Set it from an independent
criterion: the time the population takes to re-adapt after a switch. Diagnostic
run for this: `runs/mvg_diag_E200` (E=200, 3000 gens, 2 seeds, log-interval 2 →
15 switches with full post-switch recovery curves).

### Why the brain starts as "one global weight"

`g`'s output at init is dominated by a *global offset*, not by the pair. For a
hidden→hidden pair (the bulk of synapses) the input to `g` is
`[type_i(4) | 0,0,0,0 | 0,1,0 ‖ type_j(4) | 0,0,0,0 | 0,1,0]` — hidden neurons get
no positional code (`model.py`, zeros) and share a role one-hot, so **only 8 of 22
input dims vary between pairs**, and those are the type embeddings initialised at
`0.1 * jr.normal`. Meanwhile the role one-hot contributes a full 1.0 in two slots
and both Linear layers carry Equinox-default biases. Measured over 12 seeds
(|mean| of `w` vs sd across pairs):

| variant | \|mean\| | sd | offset/spread |
|---|---|---|---|
| as-is | 0.124 | 0.055 | **2.3×** |
| `g` output bias → 0 | 0.085 | 0.056 | 1.5× |
| type vectors ×10 (0.1 → 1.0) | 0.165 | 0.113 | 1.5× |
| both | 0.118 | 0.115 | 1.0× |

Neither cause dominates — the `0.1` init scale matters at least as much as the
output bias. Consequence: the brain starts maximally *regular* (every synapse
near-identical), early search moves the offset rather than differentiating types,
and an absolute gate thresholds **the offset**, behaving as a global on/off switch
for the whole brain rather than as a pruning rule.

### Sign audit: accuracy-fitness runs get stuck single-sign

41 saved evolved brains: 20 single-sign, 21 mixed — and the split tracks the
fitness, not the task.

| run group | fitness | % positive per seed |
|---|---|---|
| `retina_acc5` | accuracy | 0, 100, 100, 0, 0 |
| `retina_K6_acc` | accuracy | 100, 93.6, 100 |
| `retina_K6_margin` | margin | 100, 50.2, 41.2 |
| `retina_margin5` | margin | 47.4, 100, 0, 43.3, 0 |
| `curric_k8_xor` | curriculum | all mixed (23–67) |

Every `--fitness accuracy` retina seed is single-sign or near it, with small
weights (max |w| ~0.05–0.45); margin/curriculum runs mix signs and reach ~0.99.
Same mechanism as everywhere else here: raw accuracy is piecewise-constant, CMA-ES
gets little signal, and the genome stays near its initialisation — where the
offset dominates. (⚠️ genome shape doesn't depend on `n_in`/`n_hidden`, so those
older files reload under assumed dims; the exact percentages would shift, the
single-sign-vs-mixed split would not.)

Note this did **not** bind in the gated run above: both arms ended mixed-sign
(FG 78.6%, MVG 34.4% positive). And per the monotone ceilings in the caution
section, all-positive weights cap `retina_ka2005/and` at 0.891 — above the 0.750
plateau — so single-sign weights do not explain that plateau.

## Open threads

- **Re-run curriculum on `retina/xor`** (no one-side shortcut) — the honest
  modularity test; `/and` is confounded by the 0.848 one-side freebie.
- **Disambiguate the K=6 wall: reachability vs `g`-capacity.** Retrain K=6 with
  wider `g` (64/128); run the basin probe at K=6; full nonlinear `g` fit in the
  oracle. If wider `g` still caps at 0.85 → reachability, not `g`.
- Re-run the basin probe under margin selection (does the `left` needle become a
  recoverable basin?).
- **Build a modularity metric** (graph `Q` / block structure on the grown weight
  matrix) — prerequisite for studying the K-vs-modularity relationship above.
  Prior metric decision (carry over): was leaning **Infomap** over **Louvain/Newman
  `Q`** because it handles the *directed* weight graph better; an `infomap-env`
  conda env already exists. A task-aware left/right block score is the cheap first
  cut before a general community-detection metric.
- **Test the evolvability hypothesis directly:** modularly-varying goal (`--mvg`,
  AND↔OR switching) — does a switching pressure carve reachable modular solutions
  that a static goal cannot? This is the actual thesis question. *(First pair run
  2026-08-09 on `retina_ka2005`; uninformative because both arms ended 77–100%
  dense — see the synaptic-gate section. Needs a density-controlled gate first.)*
  **Density is now controlled** (synaptic budget, 2026-08-10) — the blocker is
  cleared. Remaining prerequisites: pick `E` from the recovery-time diagnostic,
  and budget ~20–26k generations/seed (see the KA evaluation-budget arithmetic).
- ~~**Budgeted (top-k) gate**~~ — DONE differently: an in-budget normalisation
  (`--synaptic-budget`), not top-k. See the synaptic-budget section.
- **`train.py` has NO checkpoint/resume** (only `config.py` / `visualize_ckpt.py`
  mention checkpoints). Required before the ~4-hour FG/MVG pair at KA-matched
  budget — a crash currently loses the whole seed.
- **Logging gap:** the per-seed header line prints `w_threshold=` but not
  `synaptic_budget`/`shrink`. The run *directory* name carries them (`_b4s0.9`)
  and `config.json` is written, but the log line alone is ambiguous.
- **Decide whether `g`'s init should be re-centred** (zero the output bias and/or
  raise the `0.1` identity scale) so the brain doesn't start as one global weight.
  Both are one-liners and could be flags rather than default changes.
- **Log the `curric_k8_xor` null** — run exists on disk, conclusions never written
  up here.
- **Wire `qmetrics` into exp 1** — still not connected; `Q` is not computed at any
  point in the training loop. Cheap raw `Q` per log interval + normalised `Q_m`
  once at the end (the kashtan_alon split).
