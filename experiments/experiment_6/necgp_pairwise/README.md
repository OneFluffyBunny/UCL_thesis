# necgp_pairwise — EXPERIMENTAL, third attempt at a NECGP framework (2026-08-21)

⚠️ **Not `../necgp/`. Not validated. A separate, still-untested variant, kept in
its own directory on purpose so it is never accidentally cited alongside
`necgp/`'s numbers.** See `ecgp.py`'s module docstring for the full mechanism.

## Why this exists

`../necgp/`'s own decomposition of a solved circuit (`../RESULTS.md`,
2026-08-21 entry, "decomposing a solved circuit") found that 37.5% of module
*calls* in a solved circuit were to modules with **zero internal gate
interaction** — a single gate, or several gates that never chain, wrapped in
module packaging. Root cause: `necgp/`'s `compress` (inherited from plain
ECGP) grabs a window of **genome-adjacent** nodes and accepts unconditionally.
Genome position carries no information about which nodes actually feed each
other, so most of what gets bundled is gunk, not reusable computation.

## The mechanism under test

`compress` here always proposes exactly **two** genome-adjacent nodes (never a
wider window), and **rejects, as a no-op, unless the second node actually
reads the first node's output** — i.e., unless they communicate. This should
make a fake module structurally impossible for this operator to create, not
something detected after the fact. Nesting still happens (an existing module
can stand in for one half of a later pair, gated by `nest_decay` exactly as in
`necgp/`), so bigger modules build up incrementally through successive
communicating merges, rather than one `compress` grabbing an arbitrary run at
once.

**Deliberately stripped down**, to keep this attributable to the one change
under test: no `module_point_mutate`, no `add_input`/`remove_input`/
`add_output`/`remove_output`. A module's interface is fixed forever at
whatever `compress` gave it; the only way to change one is `expand` then
re-`compress`. Four knobs total: `mutation_rate`, `compress_prob`,
`expand_prob`, `nest_decay`.

`max_module_size` (default 5, unchanged number from `necgp/`, **different
meaning**) now bounds the resulting module's own recursively-flattened
primitive count, not body-slot count — every module body here is always
exactly 2 slots, so a slot-count bound would do nothing. Reminder: **increase
this if 5 turns out too small** to let evolution build anything useful.

## Hypothesis (stated before the full run — the smoke test above doesn't count)

If the mechanism works as intended: the population should still be able to
solve `retina_ka2005`/xor (NAND-only), and its final module composition should
show **0% fake modules by construction** (checked twice over — a `validate()`
assertion that fires if one ever survives, and the same transitive-call-count
report `necgp/scratch_decompose_final.py` used, for a direct side-by-side
against `necgp/`'s 37.5%). What's genuinely unknown going in: whether the
stricter accept criterion makes `compress` fire so rarely that little to no
modular structure forms at all within a reasonable generation budget — that
would be a refutation of "this is a usable replacement," even if the
zero-fake-modules guarantee itself holds.

## Status

First real run in progress/complete — see `../RESULTS.md` for the dated entry
and numbers. Single seed(s), NAND-only, `retina_ka2005`/xor, same task/budget
`necgp/` used, for direct comparability.
