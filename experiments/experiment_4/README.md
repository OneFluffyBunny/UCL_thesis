# Experiment 4 — CGP vs ECGP on the Kashtan–Alon retina (does reuse actually happen?)

**Not** a fourth point on the encoding/optimiser axes. Experiments 1–3 all ask
*"is the final network modular?"*. Experiment 4 asks a different, prior question:
**when modularity appears under goal switching, is any structure actually being
kept and reused — or is it being rediscovered from scratch every time?**

The substrate is Cartesian Genetic Programming (CGP) and its module-acquiring
variant ECGP (Walker & Miller, IEEE TEVC 12(4), 2008, `papers/`). CGP is a
discrete boolean-circuit representation, *not* our neural model — it is used here
because module birth, reuse and death are **explicit, named, countable objects**,
which they can never be in a weight matrix.

---

## The gap

Kashtan–Alon measure `Q` on the *final* network and show MVG > FG. A high final
`Q` is compatible with two completely different evolutionary histories:

- **(a) Repeated rediscovery.** Each goal switch wrecks the solution; the lineage
  re-evolves a modular one from scratch every period, because modular solutions
  happen to be the ones reachable fastest. Modularity is a property of the
  *attractor*, refound each time. Nothing persists.
- **(b) Persistence and reuse.** A substructure is found once, survives the
  switches, and is redeployed against each new goal. Modularity is the *fossil*
  of a lineage-level history.

A snapshot of `Q` cannot tell these apart. Neither KA's paper, nor
`kashtan_alon/`, nor exp 1–3 distinguish them. This is the same dichotomy as
convergent-vs-duplication origin for network motifs (Conant & Wagner 2003).

"Behavioural modularity" — evolution finding a structure at some point in
history and then reusing/adapting it — is claim **(b)**, specifically. The
recurring observation that *"structures that solve a subproblem quickly get
lost"* is an observation of **(a)**.

## H1 — the hypothesis

> Goal switching alone produces modularity by **repeated rediscovery**, not by
> persistence and reuse. Reuse requires the encoding to supply a **unit of
> inheritance below the whole genome**; given one, evolution will accumulate and
> redeploy substructures across goal switches.

ECGP is a clean manipulation of exactly one variable. CGP and ECGP share the
representation, the mutation operator, the (1+4) ES and the fitness function —
ECGP only adds an inheritable, reusable sub-unit (the module). No comparison
available in exp 1–3 isolates "does having a unit of reuse matter?" this cleanly.

## The design

2×2, on the KA Fig. 5a retina (`kashtan_alon/tasks.py`, `task=retina`), boolean
fitness over all 256 input patterns:

| | FG (fixed `L AND R`) | MVG (`and` ↔ `or`, every E=20 gens) |
|---|---|---|
| **CGP** | baseline | KA's result should replicate |
| **ECGP** | does acquisition help without switching? | **the cell of interest** |

## The signature measurement

> **Time-to-recovery after each goal switch shrinks over successive switches
> under ECGP, and stays flat under CGP.**

A learning curve at the *lineage* level — the module list acting as an
accumulated vocabulary. Flat curves under both conditions falsify the reuse
story outright. This is visible even when `Q` differences are a null.

Predicted signatures:

| Observable | under **rediscovery** | under **persistence** |
|---|---|---|
| module lifetimes (MVG) | all < one switch period | bimodal; tail spanning many periods |
| recovery time vs switch number | flat | decreasing |
| ECGP advantage over CGP under MVG | **none** — nothing to carry | grows with number of switches |
| module list at switch *n* vs *n+1* | flushed and rebuilt | core survivors + new arrivals |

Row 3 is the sharp one: under pure rediscovery, ECGP should confer **no benefit
at all** under MVG.

## Secondary hypotheses (logged for free)

- **H2 — do modules match sub-goals?** Acquired modules implement the task's
  natural decomposition (a left-retina detector, a right-retina detector) rather
  than arbitrary chunks. Testable because a module has few inputs: dump its
  truth table and compare against `_left_feature` / `_right_feature`. This is
  ground truth on *"is this a module in the functional sense, or just a connected
  blob?"* — impossible to get in a neural net.
- **H3 — does reuse itself confer protection?** Module lifetime scales with peak
  call-site count. ECGP's protection is *imposed* (contents are immune to point
  mutation) but its **survival** is not — a module dies the moment it is absent
  from the fittest individual. If H3 holds, it is empirical warrant for the
  thesis claim that a compressed encoding manufactures conservation for free by
  giving each parameter many dependents.

## A null is still a result

If MVG produces no persistent modules even where the representation makes
persistence maximally easy, the conclusion is: **modularity under goal switching
is convergent, not inherited.** That reframes the bottleneck's job — to make the
modular attractor *reachable*, not to make found structure *heritable*.

## What this does NOT prove

Nothing directly about neural networks, real-valued weights, or the exp-1
g-encoding; `compress`/`expand` do not port mechanically to a fixed-neuron weight
vector under CMA-ES. Its role is an **existence proof about the mechanism**, in a
substrate where acquisition, reuse and death are directly observable. Scope: a
two-week side experiment, not a second thesis.

## Status

Spec only — no code yet. Blocked on `PAPER_SPEC.md` (see below).

⚠️ **Before writing any code**: `kashtan_alon/` runs 1–3 produced a null because
the algorithm was reimplemented from the wrong source. A verbatim `PAPER_SPEC.md`
must be written and checked before implementation — same discipline that fixed
`kashtan_alon/`. Every entry is marked `[verbatim]` / `[inferred]` / `[our
choice]`, and **no code is written while anything reads `[inferred]`**.

Note that the PDF's tables and inline math are **images** that text extraction
drops silently — a missing number means "screenshot that page", never "pick
something reasonable".

Recovered so far (all `[verbatim]`): Table II in full (pop 5; 100 nodes / 300
genes; 3% point mutation = 9 genes, probability 1; compress/expand 0.1/0.2;
module point mutation 0.04; add/remove input 0.01/0.02; add/remove output
0.01/0.02; module list initially empty; 50 runs); one-row topology with a node
connectable to *any* previous node (⇒ levels-back unrestricted — not a parameter
in this formulation); function set AND/NAND/OR/NOR. **Outstanding:** the maximum
number of module inputs (p. 7, dropped inline symbol).
