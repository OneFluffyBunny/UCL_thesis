# PAPER_SPEC.md -- Self Modifying CGP (SMCGP)

Source: S. Harding, J. F. Miller, W. Banzhaf, **"Self Modifying Cartesian Genetic
Programming: Parity,"** CEC 2009. Local copy:
`papers/Harding_Miller_Banzhaf_2009_Self_Modifying_CGP_Parity.pdf` (gitignored,
copyrighted -- see `papers/` in the repo root). Obtained as a free, author-hosted
copy, read in full; every `[verbatim]` item below is a direct paraphrase/quote
checked against that text, not a secondary summary.

Every design point is tagged:
- **[verbatim]** -- the paper states this explicitly.
- **[inferred]** -- the paper implies this but does not spell it out fully; the
  choice made here is the most literal reading, with the ambiguity noted.
- **[our choice]** -- the paper is silent (this representation-level detail
  doesn't exist in the paper at all, e.g. an unbounded domain needing SOME
  bound to sample from); a specific, defensible choice was made and is not
  claimed to be what the original authors used.

Do not read an untagged claim elsewhere in this experiment's code as paper fact
-- check here first, per the project's `feedback_no_guessing` convention.

## Representation

**[verbatim]** A genotype is a list of nodes. Each node has: one function gene
(indexing a *single, unified* id space covering the INPUT terminal, the
computational functions, and the 13 self-modification operators -- evolution
picks whichever kind of node a position becomes); connection genes ("as in
CGP"); three real-valued parameter genes P0, P1, P2; one binary output-flag
gene. "In this paper all nodes take two inputs, hence each node is specified by
7 genes."

**[verbatim]** Connections are **relative, backward-only, and positive**: gene
value `c` means "`c` nodes back." A gene must be `> 0`. An address landing
before the start of the graph resolves to a **constant 0**, not an error --
this is explicitly what lets a self-modification operator relocate or
duplicate a sub-graph while it "retains semantic validity."

**[our choice]** The mutation/init domain for a connection gene is `[1,
addr_max]`. `addr_max` defaults to `--nodes` (the genotype length) -- see
`config.py`'s docstring for why: an early draft fixed it at a flat 200 against
a 30-node genotype, and nearly every address landed out of range by
construction, collapsing almost every genotype to a constant regardless of its
other genes (verified: a 300k-evaluation run of that draft never left the
"any constant" plateau). Tying it to genotype size was the fix.

**[our choice]** The mutation/init domain for a real-valued parameter gene
(P0..P2) is `[-param_range, param_range]`, `param_range` defaulting to 10.0.
The paper never bounds these; a domain of this rough scale keeps DU4/MOV/DUP/etc's
`int(Pi) + x` arithmetic producing indices that land inside or just outside a
graph of a few dozen nodes, rather than always wildly out of range.

## Inputs and outputs

**[verbatim]** An INPUT node returns the next program input each time it is
*called*, wrapping around to the first input once it has been called more
times than there are inputs. All negative-resolving addresses return 0 (not an
input) -- this replaced a flawed earlier scheme (their own prior SMCGP paper)
where negative addresses meant "connect to an input," which broke once
self-modification moved a sub-graph away from the start of the genotype.

**[inferred]** The order INPUT nodes are *called* in, within one evaluation
pass, is not stated. Implemented as ascending active-node position ("leftmost
first"), consistent with every other left-to-right rule the paper does state
(the To-Do list order, "leftmost node of the graph").

**[verbatim]** Output selection: a node is an output candidate if its
output-flag gene is 1. If there are at least `n_out` flagged nodes, the
**leftmost** `n_out` of them are used. If **none** are flagged, the system
falls back to the **last `n_out` nodes** ("essentially, this reverts the
system back to the previous [plain CGP] approach"). If the graph has fewer
nodes than `n_out` at all, the individual is **corrupt** and scored as a
failure without being evaluated.

**[inferred]** The paper covers ">= n_out flagged" and "0 flagged" explicitly
but not "1..n_out-1 flagged." Implemented as: fall back to the last-`n_out`
rule in that case too, rather than inventing an unaddressed hybrid.

**[our choice, load-bearing]** Every node's output-flag gene is **initialised
to 0**, not drawn as a fair coin. See `smcgp.random_node`'s docstring: with a
Bernoulli(0.5) flag on N nodes, the *leftmost* flagged node lands within the
first couple of positions almost every time, and a node that early has no
legal backward address at all -- every connection resolves to the constant-0
fallback -- trapping the whole population on a fixed-constant plateau no
mutation can climb out of. Starting unflagged means the population begins from
the paper's OTHER documented fallback (last node = plain CGP), which has full
reach; flags are then introduced gradually by mutation, at the normal rate.

## Development

**[verbatim]** Per test case of `n_in` inputs, the phenotype starts as a copy
of the genotype (the genotype itself stays fixed for the whole evaluation) and
is iterated `n_in - 2` times. Each iteration: a backward walk from the output
node(s) finds the active nodes; every active self-modifying node, in
left-to-right (ascending position) order, is appended to a shared "To Do"
list, capped at `todo_cap` entries (paper default **2**, "manipulations are
relatively computationally expensive"); the list is then applied, **in order**
("first in is first executed"). "Redundant operations (e.g. copying 0 nodes)
are ignored, however they are taken into account in the To Do list length."

**[inferred]** Whether the To-Do list persists *across* development
iterations or is collected-then-drained fresh each iteration is not fully
pinned down by the text ("there is a single To Do list for evaluation of each
individual" vs. "after each iteration, the list is parsed"). Implemented as:
collect (up to `todo_cap`) and apply once per iteration -- i.e. each
development step is a complete collect-then-apply cycle, matching the
"after each iteration, parsed" sentence literally.

**[verbatim, shared operator rules]** `Pi + x` is truncated to an integer.
Address indexes below 0 are clamped to 0; beyond the graph, clamped to the
graph length. Start/end pairs are sorted ascending. `x` is "the integer
position of the node in the phenotype graph that contains the self modifying
function" -- **[inferred]** re-resolved at the moment each queued operator is
actually applied (not frozen when it was collected), since earlier operators
in the same batch can shift positions; a node deleted by an earlier operator
in the batch is simply skipped ("redundant... ignored").

## Operator table (`smcgp.py`'s `_apply_*` functions)

All target ranges below are `[verbatim]`. One easy-to-miss distinction,
checked directly against the text: **DU4/MOV/DUP/DU3/OVR/ADD address a node
relative to the operator's OWN position (`P0 + x`)**, but **CHF/CHC/CHP
address node `P0` directly** -- `x` is not added for those three.

| Op | Definition [verbatim] |
|---|---|
| DU4 | Copy `P1` nodes starting at `P0+x`; insert after `P0+x+P1`; copies' connection genes are multiplied by `P2`. |
| SHIFTCONNECTION | Add `P2` to the connection genes of the `P1` nodes starting at `P0+x`. |
| MULTCONNECTION | Multiply the connection genes of the `P1` nodes starting at `P0+x` by `P2`. |
| MOV | Move the nodes between `P0+x` and `P0+x+P1`; insert after `P0+x+P2`. |
| DUP | Copy the nodes between `P0+x` and `P0+x+P1`; insert after `P0+x+P2`. |
| DU3 | As DUP, but copies' connections are adjusted to keep pointing at the ORIGINAL absolute targets. |
| DEL | Delete the nodes between `P0+x` and `P0+x+P1`. |
| ADD | Insert `P1` new random nodes after `P0+x`. |
| CHF | Change the function of node `P0` to the function indexed `P1`. |
| CHC | Change the `(P1 mod 3)`th connection of node `P0` to `P2`. |
| CHP | Change the `(P1 mod 3)`th parameter of node `P0` to `P2`. |
| OVR | Copy the nodes between `P0+x` and `P0+x+P1` **to** position `P0+x+P2`, overwriting what was there. |
| COPYTOSTOP | Copy from `x` to the next COPYTOSTOP node, "STOP" node, or end of graph; insert where the scan stopped. |

Two more `[inferred]`/`[our choice]` points, checked against the operator
table and not elsewhere in the paper:

- **"insert after position K" means index `K+1`**, not index `K`
  (`DU4`/`MOV`/`DUP`/`DU3` all use this wording). OVR says "**to** position K"
  (no "after"), so it is NOT offset by one -- this distinction is easy to miss
  and was caught by a failing hand-built test (`test_smcgp.py`), not by
  reading alone.
- **CHC uses `P1 mod 2`, not the paper's `mod 3`.** The paper's SMCGP is
  general enough to support variable node arity elsewhere in that body of
  work; `mod 3` only makes sense there. This representation fixes arity at 2
  (matching this paper's own "all nodes take two inputs"), so `mod 3` would
  sometimes pick a third connection slot that does not exist here. CHP still
  uses `mod 3` verbatim -- three parameter genes really do exist per node.
- **DU3's recomputed connection gene**, when the new position would need a
  forward or self address (new position `<=` original target): set to
  guarantee an out-of-range result (resolves to constant 0), since this
  representation's connection genes have no non-positive domain to fall back
  to.
- **OVR does not grow the graph.** The paper does not say whether an overwrite
  targeting near the end of the graph may extend it; the copy is clipped to
  fit inside the existing node count (DUP/MOV/ADD already cover deliberate
  growth).
- **COPYTOSTOP has no standalone "STOP" node type** in this implementation --
  Table I defines no such function -- so the only terminator besides
  end-of-graph is another COPYTOSTOP node.

## Function set (Table I)

**[verbatim]** `BF0`..`BF15` are the sixteen 2-input boolean functions,
individually spelled out in the paper (e.g. BF6 reduces to XOR, BF7 to OR,
BF8 to NOR, BF9 to XNOR, BF14 to NAND, BF15 to TRUE -- checked by hand against
`gates.bf`'s minterm-index derivation, which reproduces the whole table from
one formula instead of hand-transcribing sixteen lambdas). Two experiments are
reported: the **restricted set** (AND, OR, NAND, NOR -- this repo's default,
matching experiment_4's own `DEFAULT_GATES`) and the **full set** (all sixteen).

**[inferred, load-bearing on a worked example]** A self-modifying node, when
it participates in the graph as an ordinary computational node (most of them
do, since most nodes ARE reachable), needs SOME output value. The paper gives
exactly one example: "The DUP function... is defined to return the first
input." This implementation applies "pass through the first input (`c0`)" to
**all 13** self-modifying operators uniformly, since the paper gives no
per-operator table for this and the one worked example is consistent with a
single shared rule.

## Evolutionary algorithm and parameters (section IV)

All **[verbatim]**, this repo's paper-matching defaults in `config.py`:

| Parameter | Value |
|---|---|
| Strategy | (1+4) evolution strategy |
| Bootstrap population | 50 random individuals |
| Mutation rate | 0.1 per gene, independently |
| Function/connection gene mutation | uniform resample over the legal range ("unbiased") |
| Real-valued gene mutation | 10% chance of a fresh uniform draw, else additive `N(0, sigma=20)` noise |
| To-Do list length | 2 |
| Evaluation budget | 10,000,000 (paper's own ceiling; this repo's CLI default is far smaller for a fast smoke test -- see `config.py`) |

**[verbatim, explicitly stated by the authors]**: "The evolutionary parameter
values have not been optimized, and we would expect performance increases if
more suitable values were used." Do not treat 0.1/20/2/50 as tuned for
anything but the authors' own even-parity task -- reusing them for a different
task (e.g. this repo's `experiment_4`/`experiment_5` retina/KA tasks, should
this be extended there) is itself an `[our choice]`, not a validated transfer.

**[our choice, not in the paper]** Tie-break on equal offspring/parent
fitness: promote a uniformly random tied offspring (neutral drift), the same
convention `experiment_4`'s plain-CGP (1+4) loop uses. The paper's (1+4) ES
description does not state a tie-break rule at all.

## Fitness function (section VI-A)

**[verbatim]** For each test case, in increasing order (2 to `max_inputs`
inputs): develop the genotype for `n_in - 2` iterations; evaluate the
resulting phenotype exhaustively; add its hit count to a running score. The
**first** test case size a genotype fails to solve completely ends scoring for
that individual -- larger sizes are never attempted. This is the paper's own
growth-forcing curriculum, and also why scoring stays cheap even as
`max_inputs` approaches 20 (most individuals fail early, long before the
2^20-pattern top of the curriculum is ever reached).

**[inferred]** "Parity" here is implemented as plain n-input XOR
(`tasks.target_parity`); which labelling ("even" vs "odd" parity) the paper
means is immaterial to search difficulty (see `tasks.py`'s docstring).

## Known gap: search speed vs. the paper (honest, not yet closed)

Verified (see `RESULTS.md`): the implementation is not stuck -- given enough
evaluation budget, offspring scores do improve past the initial 2-input
plateau (2/4 -> 3/4 correct, in roughly half of a 6-seed, 500k-evaluation-per-
seed batch). But no seed had solved even the 2-input case within 500k
evaluations, against the paper's own reported *average* of 126,095 for the
restricted function set (Table II). The gap is plausibly explained by the
several `[inferred]`/`[our choice]` points above (the uniform first-input
pass-through convention especially, which makes the large majority of the
function-gene space -- 13 of 18 ids -- act as simple relays rather than
combining computation) compounding against the paper's own tuned-for-parity
numbers, which the authors themselves flag as unoptimised. This is recorded
as an open efficiency gap, not silently smoothed over -- do not quote this
implementation's evaluation counts as reproducing Table II.
