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
- **Test the evolvability hypothesis directly:** modularly-varying goal (`--mvg`,
  AND↔OR switching) — does a switching pressure carve reachable modular solutions
  that a static goal cannot? This is the actual thesis question.
