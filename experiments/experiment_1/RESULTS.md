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

### E measured directly: re-adaptation costs ~10 generations, so **E=20 is right**

`runs/mvg_diag_E200`, 28 post-switch epochs (epoch 0 excluded as a cold start).
`t_recover` = generations from the switch until best accuracy comes within 0.005
of the maximum reached in that epoch.

| | value |
|---|---|
| accuracy drop at switch | mean +0.032, median +0.040, max +0.060 |
| `t_recover` | **median 10**, mean 27, p75 16, p90 73, max 176 |
| recovered within 20 / 50 / 100 gens | 75% / 89% / 89% |

❌ **The evaluation-matching argument above (E ≈ 190) is WRONG and should not be
used.** Matching KA's *evaluations per epoch* assumes re-adaptation cost scales
with population size; measured directly it does not. Re-adaptation here costs
~10 generations, so E=20 is ~2× the observed recovery time — enough to adapt,
short enough that the population never settles. E=200 is 20× recovery, which
turns each epoch into a mini-FG run: ~190 dead generations sitting at plateau,
which is exactly where FG-like specialisation happens. **Keep E=20.** It now has
two independent justifications: KA's reconstruction, and our own dynamics.

⚠️ `t_recover`'s long tail is **not** slow re-adaptation — it is genuine
innovation. Every epoch with `t_recover` > 100 (seed 1 epochs 8, 11, 14: 132,
160, 176) is one where `A_max` *exceeded* the previous plateau (0.818, 0.820,
0.828). The metric conflates "get back to where we were" with "find something
new"; only the former should inform E.

Two other things this run establishes:
- **The budget holds under MVG.** Density stayed 28–56% across all 30 epochs, so
  the FG/MVG density confound that killed the gated pair is gone.
- **MVG was still improving at 3000 generations** (seed 1 hit 0.828 in its final
  epoch, above every FG seed except the 0.885 outlier), and σ *rose* to ~0.094.
  Independent support for the long-budget recommendation.

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


## Re-adaptation speed after a goal switch — the bottleneck buys EVOLVABILITY ⭐ (2026-09-13)

The headline result of the FG-vs-MVG study, and **not** the one the study was
built to find. Measured on the budgeted MVG arms, 5 seeds per encoding, via
`analysis/dense_replay.py` (per-generation population logging in the two KA
windows) and `phase_stats` in `analysis/fig_switch_window.py`.

**The statistic.** Within a 200-generation window there are 10 goal epochs. For
each epoch: the population mean accuracy at the switch (trough), the peak it
reaches inside that epoch, and the generations needed to cover 90% of that
climb. Against the epoch's OWN peak, not a fixed threshold — early in training
the population never reaches a fixed 0.9, so a fixed cut censors every early
epoch at the epoch length and destroys the comparison. (KA's reasoning.) Each
seed's number is therefore already a mean over 10 switches.

| | s0 | s1 | s2 | s3 | s4 | mean | late − early |
|---|---|---|---|---|---|---|---|
| compressed [100,300] | 6.3 | 5.8 | 6.2 | 4.9 | 8.3 | 6.30 ± 1.25 | |
| compressed [1000,1200] | 3.9 | 4.1 | 4.1 | 4.4 | 4.5 | **4.20 ± 0.24** | **−2.10, 5/5 faster** |
| direct [100,300] | 5.2 | 6.3 | 6.1 | 5.5 | 6.4 | 5.90 ± 0.52 | |
| direct [1000,1200] | 8.8 | 10.1 | 8.1 | 7.2 | 7.6 | **8.36 ± 1.14** | **+2.46, 0/5 faster** |

**The dissociation is total.** Per-seed deltas are −2.4, −1.7, −2.1, −0.5, −3.8
(compressed) against +3.6, +3.8, +2.0, +1.7, +1.2 (direct). The two sets do not
overlap; exact Mann-Whitney at n = 5 vs 5 returns the smallest value the test can
produce, p ≈ 0.008. Seed 0 — the only seed available before this run — was the
*least* impressive compressed seed, not a lucky one.

**The result does not depend on how recovery is measured** (updated 2026-09-14).
The 90%-of-own-climb statistic is relative to each epoch's peak, and peaks move
between windows (direct 0.84 -> 0.97), so it was checked against measures that
use no peak. Population mean, 5 seeds per encoding, early -> late window:

| measure | compressed | seeds faster | direct | seeds faster |
|---|---|---|---|---|
| gain in the first 3 gens after a switch | 0.196 -> **0.299** | 5/5 | 0.237 -> **0.152** | 0/5 |
| gens to climb +0.20 above the trough | 6.1 -> **2.3** | 5/5 | 3.0 -> **4.2** | 0/5 |
| gens to reach 0.75 | 7.3 -> **3.6** | 5/5 | 4.4 -> **5.0** | 0/5 |
| gens to climb +0.30 above the trough | 15.9 -> **4.2** | 5/5 | 4.6 -> **5.8** | 0/5 |

(+0.30: the compressed arm never got there in 70% of its early epochs; those are
censored at 20.) Every measure agrees in every seed. **The compressed encoding
accelerates and the direct encoding genuinely decelerates, modestly.** This
SUPERSEDES the earlier note here that the direct slowdown was "partly a ceiling
effect" (-13% per generation, normalised) and was the weaker half: a gain over
the first 3 generations cannot be capped by a peak reached ~10 generations later.
It is also not CMA-ES step-size collapse: σ averaged over each window is unchanged
or slightly larger late (late/early ratio 1.00-1.21 in all 5 direct seeds,
0.96-1.54 compressed; from the archived runs' `log.csv`). ⚠️ These measures came
from a scratch script (per-epoch segmentation of `population_window` from
`analysis/fig_switch_window.py`), not a file in the repo.

**The champion row agrees but is noisy** (compressed 4/5 faster, direct 1/5).
Expected: the champion is a max over the population, so it barely dents at a
switch. KA's headline was the population mean for this reason.

**What the population carries across a switch** (updated 2026-09-14). AND and OR
agree on the 128 patterns with L = R and disagree on the 128 with L ≠ R. So a
network's accuracy on the new goal right after a switch is exactly
`½ + ½(a − b)`, with `a`, `b` its old-goal accuracy on the L = R and L ≠ R
patterns: the trough reads directly as where old-goal competence sits.
Population troughs, early -> late: direct 0.464 -> **0.500 ± 0.001**, compressed
0.510 -> **0.438 ± 0.022**. Late on, the direct population is equally good on both
halves (a = b). The compressed one is better on the L ≠ R patterns (b − a ≈ 0.12),
**exactly the half the switch flips**, and it is the one that re-adapts faster.
A population whose competence is concentrated on the goal-discriminating patterns
may have less to rebuild. Hypothesis, not tested. (Replaces an earlier "0.500 vs
0.465, chance vs carrying structure" note: 0.465 was not the population mean,
and "chance" was the wrong frame for 0.500.)

**Why this matters.** Re-adaptation speed is facilitated variation measured
directly. It routes through no modularity metric, no null model and no pruning
threshold — so it is immune to both the threshold sensitivity and the
`lr_r` resampling instability recorded above. Read with the rest of the study,
the compressed budgeted arm gives:

* modularity (`lr_r`): **no support** — 0/5 FG and 2/5 MVG seeds beat their own
  degree-preserving null, Fisher p ≈ 0.44;
* accuracy: the bottleneck **costs** competence — 0.895 against the direct
  encoding's 1.000;
* evolvability: **supported**, 5/5 seeds, and it is the entangled encoding that
  wins.

i.e. *the genomic bottleneck buys facilitated variation, at a cost in raw
competence, and without producing measurable left/right modularity.* Narrower
than the original hypothesis, and it contradicts the natural worry that a shared
`g` (one gene moving many synapses) would make re-adaptation harder — the
opposite holds.

⚠️ **Caveats.** n = 5 per encoding; one task; one budget setting; two windows
inherited from KA rather than chosen here. These are **replays**, i.e. a second
sample of each arm (the archived study is not bit-reproducible on this backend —
float reduction order, amplified by CMA-ES), so they will not reconcile
edge-for-edge with the end-of-run tables above. Replay-vs-replay is bit-exact, so
the numbers are regenerable.

⚠️ **What this does NOT test.** MVG shows no modularity effect in *either*
encoding (compressed +0.094 FG → +0.199 MVG; direct +0.194 → +0.210), so the
absence is not attributable to the encoding. The variable both arms share is the
optimiser: KA's MVG→modularity result runs on a GA with per-gene mutation
(Pm = 0.5) and crossover (Pc = 0.5, elite 150/600), and modularity is selected
there because a modular genome swaps one module in few mutations and because
crossover recombines intact modules. CMA-ES has neither — one multivariate
Gaussian, no per-gene locality, no recombination — and at n = 443 its covariance
adapts far slower than the 20-generation switch period. **We ported KA's goal
protocol but not KA's variation operators, and their result is a claim about how
variation is generated.** That is the open thread, not a footnote.

## GA pilot: does a Kashtan-Alon-style GA give MVG more modularity or accuracy? (2026-09-14)

**Preregistered before the run.** The FG-vs-MVG study found no MVG modularity
effect under CMA-ES in either encoding. KA's result used a GA, and their mechanism
depends on how variation is generated: local mutations, crossover of whole parts,
and surviving elites. CMA-ES has none of the three. This pilot swaps the optimiser
and nothing else (`ga.py`, `--strategy KA_GA`; operator tests in `test_ga.py`).

**Scope: ONE MVG run, seed 0.** It is a pilot. n = 1 cannot establish an effect;
it can only say whether a full FG-vs-MVG GA study is worth running.

| | value | note |
|---|---|---|
| model, task, budget, goal schedule | identical to the `budget_mvg` arm above | K = 8, n_hidden 24, S = 6, τ = 0.9, AND ↔ OR every 20 gens, raw accuracy, margin fitness |
| population / elite | 600 / 150 | KA |
| crossover | P = 0.5 per offspring; each gene block from one parent, 50/50 | KA's Pc; blocks = 8 cell types, input type, output type, 16 units of `g`, `g`'s output bias [our mapping] |
| mutation | P = 0.5 per offspring; ONE uniformly chosen gene + N(0, 0.5²) | KA's Pm and one-edit rule; step 0.5 [our choice] |
| initial population | N(0, 0.1²) per gene | the same distribution as CMA-ES's first generation |
| generations | 3000 = 1.8M evaluations | KA's median budget. CMA-ES ran 640k evaluations, so the GA is also read at generation 1066 (640k) |
| logging | champion archive every gen; population mean every gen in [100,300] and [1000,1200] | |

```
cd experiments/experiment_1
python train.py --strategy KA_GA --popsize 600 --ga-elite 150 --ga-pc 0.5 --ga-pm 0.5 \
    --ga-mut-sigma 0.5 --n-hidden 24 -K 8 --task retina_ka2005 --operation and --no-balanced \
    --fitness margin --no-early-stop --no-open --archive-interval 1 --generations 3000 \
    --mvg --mvg-ops and,or --switch-interval 20 --synaptic-budget 6 --shrink 0.9 \
    --seed 0 --n-seeds 1 --dense-log 100:300,1000:1200 --out-dir runs/ga_pilot
```

**Baseline:** the CMA-ES `budget_mvg` arm, 5 seeds. Goal-matched acc(AND)
0.853 ± 0.016; density 34.0 ± 7.6%; `lr_r` +0.199 ± 0.217 (seed 0: +0.004);
2/5 seeds beat their null.

**Reading rules, fixed now** (goal-matched champion, at generation 3000 and at the
evaluation-matched generation 1066):
- *Better accuracy:* acc(AND) ≥ 0.869, i.e. above the CMA-ES MVG mean + 1 SD.
- *Higher modularity:* `lr_r` ≥ 0.416 (CMA-ES mean + 1 SD) **and** beats its
  degree-preserving null (p < 0.05, 200 rewirings) at both cut 0.05 and cut 0
  (exact zeros).
- Anything less: "no sign at n = 1". That argues against, but does not rule out,
  the GA hypothesis for this encoding.
- ⚠️ A positive result would still be one seed. It licenses the full 5 + 5
  FG-vs-MVG GA study, not a claim.
- ⚠️ The GA's population mean includes 150 unchanged elites, so its recovery
  curves are not like-for-like with CMA-ES's samples. Recovery is reported as
  descriptive only.

### Result (run 2026-09-14, ~5 min; `runs/ga_pilot/`)

Goal-matched champion. `lr_r` p from 200 degree-preserving, mask-respecting rewirings.

| | gen | acc(AND) | density (cut 0.05) | `lr_r` (cut 0.05), p | `lr_r` (cut 0), p |
|---|---|---|---|---|---|
| **GA**, 640k evals | 1059 | **0.891** | 51.7% | +0.028, p = 1.00 | +0.035, p = 1.00 |
| **GA**, 1.8M evals (end) | 2979 | **0.938** | 57.3% | +0.040, p = 1.00 | +0.052, p = 1.00 |
| CMA-ES seed 0, 640k evals (end) | 9979 | 0.836 | 27.6% | +0.004, p = 1.00 | −0.007, p = 1.00 |
| CMA-ES seed 0, same generation | 2979 | 0.812 | 28.4% | +0.008, p = 1.00 | +0.008, p = 1.00 |

Against the preregistered rules:
- **Better accuracy: YES.** 0.938 at the end and 0.891 at matched evaluations,
  against a threshold of 0.869. At matched evaluations it beats every CMA-ES MVG
  seed (0.853 ± 0.016) and even the CMA-ES *fixed-goal* budget arm (0.895 ± 0.017).
- **Higher modularity: NO.** `lr_r` is +0.04 to +0.05, far below the 0.416
  threshold and below the CMA-ES MVG mean. It does not beat its null at either
  cut. The GA champion is also much DENSER (57% vs 28–34%), and in this model
  sparser brains read as more modular (r = −0.47). So the GA found better
  networks that are less sparse, not more modular.
- Trajectory at AND-epoch ends: `lr_r` peaks at +0.14 around generation 600
  (density 52%), then falls back to +0.02 to +0.04 as accuracy climbs from 0.85
  to 0.94.
- Recovery (descriptive, population mean includes elites): 90% of the post-switch
  climb in 8.0 gens in [100,300] vs 8.3 in [1000,1200]. There is **no speed-up**,
  unlike CMA-ES on this encoding (6.3 -> 4.2).

**Reading.** At n = 1, the GA helps competence under MVG but shows no sign of
KA's modularity mechanism in the K-type genome. That is consistent with the
caveat stated before the run: this genome has no locus that holds a left or right
detector, so crossover of cell-type blocks or `g` units cannot move a module.
Not yet separated:
- (a) the accuracy gain is a GA effect independent of MVG. This needs a GA
  fixed-goal run.
- (b) the absent modularity is due to the encoding, not the operator. This needs
  the same GA on the direct encoding, where a neuron's column IS KA's crossover
  unit.
- The mutation step (0.5) was not tuned.

## Open threads

- **DONE 2026-09-13**, see "Re-adaptation speed after a goal switch" (measured
  on the population mean from dense replays; the champion barely dents at a switch).
  ~~MEASURE RE-ADAPTATION SPEED AFTER A GOAL SWITCH~~ (added 2026-09-12, the
  replacement for the withdrawn "MVG never holds both goals" finding in section
  3 above). `acc(AND) + acc(OR) <= 1.500` is forced for any network that gets no
  goal cue, so simultaneous accuracy can never test the Kashtan-Alon claim. What
  KA actually claim is that MVG populations RE-ADAPT FASTER after each switch.
  The measurement: for every switch in an MVG run, count generations from the
  switch until the champion's accuracy on the newly-active goal returns to
  within epsilon of its pre-switch level, then ask whether that count SHRINKS
  over the run (learning to switch) or stays flat (re-specialising from scratch
  each time). Compare constrained vs unconstrained, and compare against the FG
  arm's time-to-recover after an equivalent perturbation. **Everything needed is
  already on disk** - `champions.npz` holds the per-generation champion plus its
  accuracy on every goal in play, so this is an analysis pass over existing runs
  with no retraining. Cheap, and it is the one test of the thesis's evolvability
  question that this study can still answer.
- **Density-matched cross-encoding rescore** (added 2026-09-12). Section 6 of
  `../experiment_2/RESULTS.md` compares the two encodings at "close enough"
  density (34-43% vs 37.8%), but within experiment 1's constrained runs `lr_r`
  correlates with density at r = -0.471, so that slack is not free. Prune each
  experiment-1 champion to experiment 2's edge count and rescore. It would also
  settle whether the Q_m cross-encoding gap (0.495 vs 0.046 at equal raw Q) is
  driven by clumped degrees or merely by density. Saved DNAs make it cheap.
- **Re-run curriculum on `retina/xor`** (no one-side shortcut) - the honest
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
  ⚠️ (2026-09-13) Largely moot for CMA-ES runs: evosax starts the search mean at
  the ALL-ZERO genome (`init_min = init_max = 0`), so `Genome.init`'s values are a
  shape template only and never enter the search.
- **Log the `curric_k8_xor` null** — run exists on disk, conclusions never written
  up here.
- **Wire `qmetrics` into exp 1** — still not connected; `Q` is not computed at any
  point in the training loop. Cheap raw `Q` per log interval + normalised `Q_m`
  once at the end (the kashtan_alon split).

---

## FG vs MVG x constraint, 4 arms x 5 seeds, 10k generations (2026-09-12)

The `kashtan_alon/` 4-group design, run on this encoding. Preregistration, exact
commands, preflight evidence and recovery notes: `../OVERNIGHT_2026-09-12.md`.
Reproduce the analysis with `python analysis/run_all.py --root
experiment_1/runs/fgmvg`.

```
cd experiments
python run_fgmvg_study.py --experiment 1 --lanes 10      # 20 runs, resumable
```

`retina_ka2005`/AND, **raw** accuracy (`--no-balanced`), margin fitness, K=8,
n_hidden=24, popsize 64, 10,000 generations, `--no-early-stop`, seeds 0-4.
Constrained arm `--synaptic-budget 6 --shrink 0.9`; MVG arm AND<->OR every 20
generations. Every number below is the **goal-matched** champion (`matched`: the
last champion selected under AND), so FG and MVG are the same measurement --
see the goal-matching note further down.

> **METRICS REVISED 2026-09-12** (same runs, same champions, re-read). The
> primary modularity numbers are now **`lr_r`** (Newman's discrete assortativity
> at the planted left/right split) and **raw Newman Q**. The `lr` ratio score and
> `purity` are demoted and kept only for continuity; `Q_m` is reference-only. The
> reasons are in the REVISED block in `../shared_brain_metrics.py` and in the
> caveats below - in one line each: the `lr` ratio is `nan` in 15 of 40 runs and
> returns exactly **1.000** for an *anti*-assortative graph; `purity` sits below
> KA's own encoding-aware null in all 40 runs; `Q_m` is not comparable across
> encodings. **No number in the table changed - only which column leads.**

| condition | arm | n | acc (AND) | acc (OR) | density % | `lr_r` (PRIMARY) | Q (PRIMARY) | purity (descr.) | Q_m (ref) | `lr` ratio (do not report) |
|---|---|---|---|---|---|---|---|---|---|---|
| budget (S=6, tau=0.9) | FG | 5 | 0.895+-0.017 | n/a | 43.0+-5.3 | +0.094+-0.068 | 0.169+-0.096 | 0.114+-0.088 | 0.403+-0.324 | -0.569+-1.322 |
| budget | MVG | 5 | 0.853+-0.016 | 0.423+-0.034 | 34.0+-7.6 | **+0.199+-0.217** | **0.208+-0.154** | 0.066+-0.106 | 0.587+-0.588 | -0.090+-0.744 |
| no budget (ablation) | FG | 5 | **0.978+-0.022** | n/a | 93.7+-6.4 | -0.013+-0.007 | 0.048+-0.024 | 0.021+-0.008 | undef (1/5) | undef (3/5) |
| no budget | MVG | 5 | 0.867+-0.042 | 0.467+-0.080 | 96.9+-5.1 | -0.024+-0.005 | 0.010+-0.014 | 0.004+-0.008 | undef (1/5) | undef (1/5) |

Seeds beating their own degree-preserving null at the planted split (p < 0.05):
**budget MVG 2/5; every other arm 0/5.** (The p-value is unaffected by the
`lr`-vs-`lr_r` change - both are functions of the same `q` at the same partition,
and the test compares that `q` to the null *distribution*.)

**`lr_r` flips sign with the constraint.** All 10 unconstrained runs are negative
(-0.026 to -0.005); 8 of 10 constrained are positive. At 94-100% density the
planted split is very slightly *anti*-assortative - marginally fewer
within-hemisphere edges than the degree sequence predicts. Read that as "no
structure", consistent with the undefined-not-unmodular point below. The two
exceptions are `budget_fg` seed 4 (-0.006) and `budget_mvg` seed 4 (-0.007),
which fall inside the unconstrained band, so the separation is near-complete but
not clean.

### 1. `retina_ka2005` is solvable under the g-encoding — the old ceiling was wrong

The unconstrained fixed-goal arm reaches **0.978 +- 0.022, with seed 3 at exactly
1.000**. This supersedes this notebook's standing claim that experiment 1 tops
out around 0.885 and the planning estimate of "0.85-0.89, not a solve". The 0.891
figure quoted as a ceiling is the *monotone-representability* bound; the
g-encoding is not monotone, so it never applied. What was weak was the earlier
task/metric/K combinations, not the encoding. For reference KA's own network gets
0.90+-0.03 on this task, so the unconstrained arm is now above the reference
reproduction.

### 2. The CONSTRAINT produces the modularity. Goal-switching does not.

This is the finding, and it is the same one `kashtan_alon/` reports for its
fan-in cap:

* **Q 0.169 / 0.208 (constrained) vs 0.048 / 0.010 (ablation)** - 4x to 20x,
  and Q survives the encoding-aware null (>=92nd percentile) in 9 of the 10
  constrained runs against 2 of 10 unconstrained.
* **`lr_r` +0.094 / +0.199 vs -0.013 / -0.024** - a sign flip, above the
  encoding-aware null in 6 of 10 constrained runs and 0 of 10 unconstrained.
* density 34-43% vs 94-97%.
* purity 0.114 / 0.066 vs 0.021 / 0.004 - 5x to 16x, **but this is no longer
  offered as evidence.** Purity is below the random-genome null in all 40 runs of
  this study (see caveats), and most of this gap is the density gap. The finding
  stands on Q and `lr_r`; only its evidence changed.

**Removing the budget does not answer the modularity question low; it makes the
question unanswerable.** Three of the five `nobudget_mvg` seeds converge to
*exactly* 100.0% density: a complete graph has no communities to find and no
sparser degree-preserving null to compare against, so LR and Q_m come back `nan`
and purity is 0.000 by construction (every neuron is fed by every input). Say
"undefined", not "unmodular". This replicates the 2x2 in `add_to_latex.md`.

MVG, meanwhile, does **not** beat FG:

| metric | direction | constrained | ablation |
|---|---|---|---|
| accuracy | **FG > MVG** | 0.895 vs 0.853 | 0.978 vs 0.867 |
| `lr_r` (PRIMARY) | MVG > FG, **n.s.** | +0.094 vs +0.199 (p = 0.345) | -0.013 vs -0.024 |
| Q (PRIMARY) | MVG > FG | 0.169 vs 0.208 | 0.048 vs 0.010 |
| LR seeds significant | **MVG > FG** | 0/5 vs 2/5 | 0/5 vs 0/5 |
| purity (descriptive) | **FG > MVG** | 0.114 vs 0.066 | 0.021 vs 0.004 |

Two of the four cut for MVG, two against, and the two that favour MVG disagree
with each other about the ablation. The single cleanest pro-MVG fact is that
`budget_mvg` is the ONLY arm with any seed beating its null at the planted split
(seeds 1 and 2, p = 0.005 each, `lr_r` +0.347 and +0.489, Q 0.195 and 0.462) -
but 3/5 of its seeds do not, and seed 0 is at `lr_r` +0.004. Report the split, do
not average it away.

**And the MVG>FG gap on `lr_r` is confounded with density.** The constrained MVG
arm converges 9.0 density points SPARSER than the constrained FG arm (34.0+-7.6%
vs 43.0+-5.3%), and within these 10 constrained runs `lr_r` correlates with
density at **r = -0.471**: sparser scores as more modular. So the 2.1x `lr_r` gap
is partly a density artifact and partly (possibly) real, and this design cannot
separate them. Two facts say the artifact is most of it: the gap is **not
significant** (exact one-sided Mann-Whitney U = 15.0, p = 0.345, n = 5 v 5), and
in experiment 2 - where the same budget mechanism happens to produce *matched*
densities (37.8% vs 37.8%, and density-`lr_r` correlation -0.009) - the same gap
collapses to 0.194 vs 0.210, well inside one SD. **Do not claim MVG > FG from
this study.**

### 3. MVG swaps between goals every epoch - but "never holds both" was a VACUOUS claim

> **CORRECTED 2026-09-12.** The original heading here was "MVG never holds both
> goals", offered as a substantive negative result about MVG. **It is an
> arithmetic identity, not a result**, and the correction runs the other way from
> what was written. Kept visible rather than deleted because the same claim was
> propagated into `../experiment_2/RESULTS.md` and `../OVERNIGHT_2026-09-12.md`.
>
> On `retina_ka2005` there are 2^8 = 256 input patterns, split 64/64/64/64 across
> the four (left-object, right-object) combinations. AND is true on 64 patterns,
> OR on 192; the two goals **agree on 128 patterns** (both-true 64, both-false 64)
> and **disagree on the other 128**. The brain receives only the 8 retina bits -
> `train.py` passes no goal cue - so one and the same function is scored against
> both goals. Every pattern in the disagreeing half can therefore be correct for
> at most one goal, which forces
>
>     acc(AND) + acc(OR) <= 1.500
>
> with equality **iff** the network is perfect on all 128 agreeing patterns. A
> *perfect* AND solver scores exactly **0.500** on OR. Low OR accuracy in an
> AND-matched champion is thus the signature of a GOOD AND solver, not of a
> failure to generalise, and the "antiphase" is forced by the arithmetic.
>
> This makes the original reporting exactly backwards. Experiment 2's MVG seeds
> were written up as the *sharpest* form of the failure because they score
> "exactly 0.500 on OR" - but they also score 1.000 on AND, i.e. **1.500 exactly,
> Pareto-optimal, the best attainable score on this pair.** Experiment 1's arms
> reach 1.276 (constrained) and 1.334 (ablation), i.e. they are the ones leaving
> something on the table, and only because their AND accuracy is short of 1.000.
>
> **What survives.** The per-generation *swapping* is real and worth reporting as
> mechanism (the figures below show it), and so is the observation that the
> population re-specialises rather than parking on a both-goals compromise. What
> does NOT survive is any claim that low cross-goal accuracy measures a failure of
> MVG, or that 0.42-0.50 on OR is "below chance". **The correct test of the
> Kashtan-Alon claim is RE-ADAPTATION SPEED after a switch** - how many
> generations to recover the active goal, and whether that shrinks over the run -
> which is measurable from `champions.npz`. **DONE 2026-09-13**, see "Re-adaptation
> speed after a goal switch" above: the compressed encoding accelerates, the direct
> one slows down.

The AND-matched champion of an MVG run scores **0.423 +- 0.034 (constrained) and
0.467 +- 0.080 (ablation) on OR**, against the 0.500 that a perfect AND solver
would score and the 1.500 ceiling on the summed pair.

`runs/fgmvg/switch_window_budget_seed0.png` shows the mechanism generation by
generation: AND and OR accuracy alternate in near-perfect antiphase, each rising
to ~0.83 while it is the active goal and falling to ~0.40-0.50 the moment the
goal switches - which, per the bound above, is what a network specialising hard
on the active goal MUST look like. Over 10,000 generations and 500 switches there is no sign of the
oscillation narrowing. Seed 2 (`..._budget_seed2.png`) is the cleanest case: two
near-perfect square waves in exact antiphase.

**And it is not an artifact of the budget.** `switch_window_nobudget_seed0.png`
shows the same antiphase in the unconstrained arm — AND and OR alternating
0.90 / 0.40-0.60 — at 85-100% density with Newman Q pinned under 0.01. So the
trade-off is what goal-switching does in this framework, in both constraint
conditions, and (see experiment 2) under the direct encoding as well.

So MVG here re-specialises every 20 generations rather than parking on a
compromise network. Note what this does and does not establish: it is a real
description of the *dynamics*, but it is **not** evidence against the
Kashtan-Alon mechanism, because the 1.500 bound means no context-free network
could hold both goals in the first place. Acting on both goals at once is not
available to this architecture at all. Testing KA's actual claim needs
re-adaptation speed (Open threads).

### 4. The goal-matching bug was NOT cosmetic here

`train.py` previously reported `final` (the last generation's champion) under
`--mvg`. The switch schedule is deterministic and 10,000/20 is even, so every MVG
run ends mid-OR and `final` is an OR-selected network being compared against the
FG arm's AND-selected one. Measured on an 80-generation self-test the two differ
by **4x in density** (AND epoch 23.2%, OR epoch 91.0%), and the best-EVER
champion reported 0.840 which `acc_by_op` reveals to be 0.840 on OR and **0.512
on AND**. Fixed 2026-09-12: `matched` is the last champion selected under
`--operation`, it is now the headline under `--mvg`, and every saved champion is
scored against every goal into `result.json`'s `acc_by_op`.

### 5. Caveats

* **The `lr` RATIO score is degenerate - superseded by `lr_r`.** Its spread is
  +-1.322 to +-6.474, with single-seed values of -10.214 and +1.538 and 2/5 to
  4/5 seeds undefined in the ablation arms. Its denominator `(q_max - q_rand)`
  collapses as the graph approaches complete, and `left_right_q` floors `q_max`
  at `q`, so a graph that rewiring cannot improve on returns exactly **1.000** -
  which is what `nobudget_fg` seed 0 reports while its `q` is *negative*. The
  ratio hands its top score to the least modular graph in the study. `lr_r`
  divides the same `q` by a closed-form ceiling instead (0.38-0.50 in all 40
  runs, never collapses), is bounded, is never `nan`, and needs no null.
  Caveat on `lr_r` in turn: the ceiling is ~0.5 throughout, so **`lr_r` ~ 2q and
  carries no information raw Q does not** - it buys a stable scale and a named
  published quantity, not extra signal.
* **`purity` is withdrawn as evidence of modularity** (kept as a descriptive
  column). Against KA's own second null - 60 random genomes per run through this
  same encoding at this same config - observed purity is BELOW the null in all
  40 runs of this study, in both constraint conditions, and the null genomes are
  *denser* (58% vs the evolved 27-48%), which should have lowered their purity.
  `recurrent_purity` unrolls a hidden block that is reciprocal nearly everywhere,
  so side-mixture diffuses back and equilibrates toward 0.5; what it reports is
  distance from mixing equilibrium after `rnn_iters` steps, a function of density
  and spectral gap. `qmetrics.circuit_purity` raises on a cycle for a reason.
* **Q_m is reference-only** (+-0.324 to +-0.588, several arms with one usable
  seed). Consistent with KA's own observation that Q_m stops discriminating above
  ~50% density - and see section 6 of `../experiment_2/RESULTS.md` for the worse
  problem: Q_m is **not comparable across encodings**, because a
  degree-preserving null is not an encoding-preserving one. It also saturates at
  exactly 1.000 for both the least (`q` = 0.030) and the most (`q` = 0.415)
  modular constrained run here.
* **The budget costs ~8 points of accuracy** (0.978 -> 0.895 under FG), so
  constrained and unconstrained arms are different competence regimes and any
  constrained-vs-unconstrained modularity difference is confounded with that.
  The FG-vs-MVG contrast *within* a constraint level is the clean comparison.
* 5 seeds. Every mean here has a spread that overlaps its neighbour on at least
  one metric.
* **purity is 0.000 in 3 of 5 `budget_mvg` seeds**, which drags that mean down
  and is why its SD (+-0.106) exceeds its mean. A zero there means no hidden
  neuron had one-sided ancestry, not that the metric failed.

### Figures (promoted to `latex_figures/experiment_1_fgmvg/`, 2026-09-13)

Both constraint conditions, as `*_budget*` / `*_nobudget*`; provenance,
regeneration commands and caveats in that folder's `README.md`.

* `brains_grid_leftright_*.png`: all 10 goal-matched champions, hidden neurons
  coloured by the left/right side `lr_r` is scored at.
* `progress_fg_vs_mvg_*.png`: champion accuracy, density and `lr_r` over 10,000
  generations, 5-seed mean ± 1 SD, sampled at reference-goal epoch ends.
* `switch_window_*_seed0.png`: windows [100,300] and [1000,1200] from a dense
  REPLAY of seed 0 (champion accuracy on the active goal, population mean, `lr_r`).
* The older `brains_grid_purity.png` / `_community.png` and the 4-arm
  `progress_fg_vs_mvg.png` stay in `runs/fgmvg/` (purity is withdrawn as evidence).
* `metrics_per_seed.csv`, `metrics_summary.json`: the numbers above.
