# RESULTS.md — experiment 6 lab notebook

## 2026-08-19 — SMCGP built, tested, and verified to search (not yet tuned)

**What was built.** `smcgp.py`/`gates.py`/`tasks.py`/`config.py`/`train.py` — a
full SMCGP implementation from the CEC 2009 paper (`PAPER_SPEC.md`), including
all 13 self-modification operators, relative addressing, the output-flag
selection rule, and the (1+4) ES / bootstrap-50 / mutation-rate-0.1 evolutionary
loop, paper-verbatim where stated.

**Correctness (`test_smcgp.py`, all passing):** out-of-range relative addressing
resolves to constant 0 and does not extend the backward walk; output-selection
falls back correctly (leftmost-flagged, else last-`n_out`, else corrupt);
DUP/DEL/CHF/ADD each checked against a hand-built tiny graph; 25 random
genotypes x 5 mutations x 5 curriculum sizes develop+evaluate with no crash.

**Two real bugs found and fixed during smoke-testing** (both now documented
inline in `smcgp.py`/`config.py`, not just here):

1. **Connection-gene domain too loose.** First draft fixed `addr_max=200`
   regardless of genotype size. On a 30-node genotype that means most
   addresses resolve out-of-range by construction, so nearly every random
   genotype's output collapses to a constant — a 300k-evaluation run never
   left the "any constant" plateau (score frozen at exactly 2/4 for every one
   of 50 bootstrap individuals and every offspring after). Fixed by tying
   `addr_max` to `--nodes` by default.
2. **Output-flag init biased the population into a structural trap.** Drawing
   each node's output-flag as a fair coin means, on N nodes, the *leftmost*
   flagged node (the paper's own selection rule) lands within the first
   couple of positions almost every time — and a node that early has nowhere
   legal to address backward to, so its value is a constant regardless of
   every other gene. This alone reproduced the same frozen-at-2/4 plateau
   even after fix #1. Fixed by initialising every output-flag to 0, so the
   population starts from the paper's *other* documented fallback (last
   node = plain CGP), which has full reach; flags are then introduced
   gradually by mutation.

**Verification run** (both fixes applied): 6 seeds, 30-node genotypes,
2-input-only curriculum, 500,000-evaluation budget each
(`--nodes 30 --max-inputs 2 --max-evals 500000 --bootstrap 50`, seeds 0-5).
Bootstrap populations start homogeneous at exactly 2/4 (the "no combining
computation reached yet" score — see `PAPER_SPEC.md`'s per-function-score
table), and **3 of 6 seeds climbed to 3/4** within budget (seeds 2, 3, 5);
none reached 4/4 (fully solved). This confirms the search mechanism genuinely
explores and improves — it is not stuck/broken — but it is empirically slower
than the paper's own reported average of 126,095 evaluations to solve 2-input
parity with the restricted function set (Table II). `PAPER_SPEC.md`'s "Known
gap" section records the likely cause (the first-input pass-through
convention applied to all 13 self-modifying operators, an `[inferred]` choice
the paper only demonstrates for one operator) and states plainly: do not
quote this implementation's evaluation counts as reproducing Table II.

**Not yet done:** no run has been taken to a full curriculum solve; no
retina/KA task wiring (SMCGP's development iterations are keyed to a growing
`n_in`, which only makes sense for a task family like parity — the fixed
8-input retina task doesn't have a natural curriculum axis, see `README.md`);
no comparison against experiment_4's plain CGP or ECGP.

## 2026-08-19 — necgp built: ECGP's own nesting extension, first smoke run

**What was built** (`necgp/` — a self-contained fork of `experiment_4/ecgp.py`,
same convention `experiment_5` uses): plain ECGP with section 6's no-nesting abort
replaced by a **depth-gated accept** — `compress` may now absorb a window
containing type I/II nodes (so the new module nests whatever they call), gated by
`nest_decay ** (depth - 1)` — a new `--nest-decay` CLI flag, `[our choice]`,
starting value 0.5 (agreed with the user: compress's own base rate plus the
window having to land on an existing module call already throttles depth hard, so
an aggressive decay on top would make depth ≥2 practically unreachable). `expand`
now undoes exactly **one** level of nesting per call (not a full recursive unroll
to primitives) — this fell out almost for free once `Module.ntype` is carried
through unchanged during both operators, rather than forced to 0 on inlining; see
`necgp/ecgp.py`'s module docstring for the reasoning and the "one expand = one
level down" argument.

**The structural refactor this actually required**, beyond the abort/decay swap:
plain ECGP's `Module` stores its body as a FLAT, fixed-arity-2 `conn` list with no
`cout`/`ocout` at all, because a body was guaranteed primitives-only (arity always
2, one output). A nested body node's callee can have any arity up to `2*ms` and
more than one output, so `Module` now carries the same shape `Individual` already
has: `ntype` (0/1/2), `conn`/`cout` as list-of-lists, `out`/`ocout` paired, plus a
stored `depth`. Every function that assumed "body = flat primitives" was rewritten
(`compress`, `expand`, `module_point_mutate`, `_run_module`, `flatten_with_origin`,
`add_input`/`remove_output`/etc., `prune_modules` — the last needed a genuinely
different fix: liveness has to walk INTO module bodies now, or a module alive only
because another module's body calls it gets pruned out from under its parent).

**`[v1 scope]` decision:** `add_input`/`remove_input`/`add_output`/`remove_output`
repair every node that calls the module they resize, but that repair
(`_nodes_using`) only looks at the top-level genotype — repairing a call site
sitting inside another module's body is representable but not implemented yet, so
these four operators simply refuse to act on a module that is **nested-into** by
another module (`_is_nested_into`), the same way they already refuse past their
size bounds. `compress`/`expand`/`module_point_mutate` are unaffected.

**A real bug caught by `test_necgp.py` before any smoke run was trusted:**
`compress`'s `m_in <= 2*n_nodes` / `n_out <= n_nodes` bounds held automatically in
plain ECGP (every absorbed node was arity-2), but a single absorbed nested call can
carry up to `2*ms` inputs or several outputs **by itself**, which can blow either
bound alone. Fixed by aborting the compress (no retry, same as every other failure
mode) when either bound would be violated — `necgp/ecgp.py`'s `compress`, both
inline-commented.

**Test suite** (`necgp/test_necgp.py`, all passing): structural invariants
(`validate`, itself extended to recompute and check every module's `depth`
recursively) survive 5x1000 generations of deliberately aggressive nested mutation
pressure; `evaluate()` agrees exactly with `flatten()` + `cgp.evaluate()` including
at depth > 1; `nest_decay=0.0` provably blocks all nesting beyond depth 1;
`nest_decay=1.0` reaches depth 2 within 3000 generations; one `expand` on a
depth > 1 module exposes a still-nested body, not raw primitives; a nested-into
module is refused by all four interface operators; `prune_modules` keeps a module
alive only via nesting.

**First smoke run** (`necgp/smoke.py` — a plain (1+4) ES, no checkpointing/CSV,
not the frozen `experiment_4/train.py`): `--gates nand --nest-decay 0.5 --seed 0`,
100 nodes, retina_ka2005/xor, 100,000 generations, 176s (567 gen/s):

```
gen 100000  score 256.0000  hits 256/256  modules 14  depth_hist {1: 6, 2: 6, 3: 2}
done -- final score 256.0000 (SOLVED), 14 modules alive, deepest module ever seen: depth 5
```

**Nesting is real, not just theoretically reachable:** the final population holds
genuine depth-2 and depth-3 modules, e.g. `M4360` (depth 3, nests `M654`, `M220`,
`M2811`) and `M4367` (depth 3, nests `M3025` **twice**) — and `M3025` (depth 2)
itself nests `M405` **twice** in its own body. That is module re-use happening
*inside* a module, not just at the top level, which is the exact phenomenon this
experiment exists to look for. Depth 5 was reached transiently (`max_depth_ever`)
before selection pruned it back down — consistent with the decay pricing deeper
nesting without making it unreachable, as intended. `evaluate()` vs.
`flatten()`+`cgp.evaluate()` cross-check passed on the final individual.

**Not yet done:** only one seed, one gate set, one decay value run so far — no
sweep over `--nest-decay`, no comparison against plain ECGP (experiment_4) or a
CGP baseline on the same task/budget, no check of whether deep modules are reused
across independent runs or are a one-off, no work on the introspection/naming
tools plain ECGP has (`module_expr`/`module_table`) that necgp deliberately
dropped for this first pass. See `README.md`'s "Where this goes next".

## 2026-08-20 — run-to-solution, NAND-only: nesting cuts generations, costs wall-clock

**What was built.** `necgp/visualize.py`, a fork of `experiment_4/visualize.py`
kept to just the unflattened (module = one box) renderer, with every box label
carrying a `|x` **nesting-factor suffix**: `|1` for a plain gate, `|2` for a
module built only from primitives, `|3` for a module that itself nests a `|2`
module, and so on (`x = 1 + Module.depth`, `Module.depth` starting at 1 for a
depth-1 module — see the file's own docstring). `necgp/stage_run.py`, a run-to-
solution driver (stops the instant `hits == n_patterns`, not at a fixed
generation budget, so "generations to solve" is a real measurement) that runs
**two twin searches on the same seed** — `--nest-decay` as given (default 0.5)
and `--nest-decay 0.0` (nesting structurally impossible, proven by
`test_decay_zero_blocks_all_nesting`) — back to back through the identical
driver, so the only variable between the two runs is whether `compress` may fold
an existing module into a new one. It then tiles six evenly-spaced snapshots of
the nested run's own history into the same 2x3 stage sheet `experiment_4` uses
(`visualize.stage_grid`, ported unchanged).

**The run** (`stage_run.py --seed 0`, defaults otherwise: `--gates nand`, 100
nodes, retina_ka2005/xor, `--nest-decay 0.5` vs `0.0`, `--max-generations
300000` safety cap):

```
nested  (nest_decay=0.50): SOLVED at gen 78134   (256/256 hits, 229.5s, 340 gen/s)
flat    (nest_decay=0.0) : SOLVED at gen 115095  (256/256 hits, 136.3s, 844 gen/s)
nested vs flat: -36961 generations, +93.2s (nested faster in generations, slower in wall-clock)
```

**Nesting cuts generations by 32% (78 134 vs 115 095) but is 1.7x slower in wall-
clock (229.5s vs 136.3s)**, because a nested generation costs more to *evaluate*:
`ecgp.evaluate`/`_run_module` recurse through every level of nesting on every
fitness call, so a depth-4 module (present in the final nested individual, see
below) is several recursive calls deep per node, where a depth-1-only individual
is one flat call. So the mechanism does what it was built for — folding an
existing module into a new one lets evolution reach the target in meaningfully
fewer generations, i.e. nesting is doing real search work, not just adding
churn — but on a per-generation-cost basis (or on wall-clock at fixed hardware)
it does not currently pay for itself in *this pure-Python, unoptimised*
implementation. Whether it would win on wall-clock too under a faster evaluator,
or whether 32% fewer generations is itself an artifact of one seed, is open —
see "Not yet done" below.

**The final nested individual keeps real, deep nesting, not just depth-1
modules that happened to survive:** 17 modules alive at the solution, **11 of
them depth > 1** (module ids `654, 2811, 3025, 3097, 3337, 3386, 3391, 3408,
3409, 3410, 3411`). The stage sheet (`necgp/stages_seed0_nand.png`) shows this
directly — by gen 78134 (the solved panel) the circuit carries `M3411|4`, a
module nesting a module nesting a module nesting primitives, sitting right next
to plain `NAND|1` gates in the same picture. Depth climbs steadily across the
six panels (gen 0: all `|1` NAND; gen 15000: `|2`/`|3`; gen 30000-65000: mostly
`|2`/`|3` with some reduction; gen 78134/solved: `|2` through `|4` coexisting),
which reads as evolution using nesting throughout the run rather than only
right at the end.

**Not yet done:** still one seed. No sweep over `--nest-decay` values, no repeat
across seeds to check whether the generation-count win and the depth-4 result
are reproducible or a one-seed artifact, no profiling of *where* the nested
evaluator's extra wall-clock time actually goes (recursion overhead vs. simply
evaluating more effective nodes), no comparison against experiment_4's own
NAND-only ECGP numbers (56 571 median gens, 12 seeds, but on goal `and` with its
known shortcut and a different codebase — not a fair like-for-like without
re-running on `xor`).
