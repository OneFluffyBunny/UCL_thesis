# ECGP — paper specification

Source: J. A. Walker and J. F. Miller, *"The Automatic Acquisition, Evolution and
Re-use of Modules in Cartesian Genetic Programming"*, IEEE Trans. Evol. Comput.,
vol. 12, no. 4, pp. 397–417, Aug. 2008. Local copy in `papers/` (gitignored).

Every entry is tagged:

- **`[verbatim]`** — stated in the paper. Implement exactly.
- **`[inferred]`** — derived from the paper but not stated. **None currently. No
  code is written while any `[inferred]` entry exists.**
- **`[our choice]`** — the paper never ran this task, so no source can supply it.
  Declared and justified here, never invented silently.

⚠️ The PDF's tables and inline math are **images**; text extraction drops them
without warning. Values below marked `[verbatim]` from Table II and from the
p. 5 bounds paragraph were read off the rendered page, not extracted.

---

## 1. Representation

| Item | Value | Tag |
|---|---|---|
| Topology | 1 row (`"This one-dimensional topology is used throughout the work we report in this paper"`, p. 3, citing [15]) | `[verbatim]` |
| Connectivity | a node may connect to the output of **any previous node** — no levels-back restriction. The term never appears in the paper and Table II omits it: it is not a free parameter in this formulation. | `[verbatim]` |
| Node arity | 2 | `[verbatim]` |
| Genes per node | 3 gene-slots: 1 function + 2 inputs | `[verbatim]` |
| A "gene" in ECGP | a **pair of integers**. Function gene = `(function_or_module_id, node_type)`. Input gene = `(node_index, node_output)`. `node_output` selects *which* output of the referenced node, since ECGP nodes may have several. | `[verbatim]` |
| Program outputs | `O` extra integers appended to the genotype, each naming the node an output is taken from; **initialised to the last `O` nodes**. | `[verbatim]` |
| Node types | `0` = primitive; `I` = module created by `compress`; `II` = module created by point mutation (a reuse). Encoded as the **second integer of the function gene pair**. | `[verbatim]` |

## 2. Table II — common parameters (all test problems)

All `[verbatim]`. `*` = ECGP only.

| Parameter | Value |
|---|---|
| Population size | 5 |
| Initial genotype size | 100 nodes (300 genes) |
| Genotype point mutation rate | 3% (9 genes) |
| Genotype point mutation probability | 1 |
| Compress / Expand probability `*` | 0.1 / 0.2 |
| Module point mutation probability `*` | 0.04 |
| Add / Remove input probability `*` | 0.01 / 0.02 |
| Add / Remove output probability `*` | 0.01 / 0.02 |
| Module list initial contents `*` | Empty |
| Number of independent runs | 50 |

**Reading of the mutation rows:** probability 1 ⇒ the operator is applied every
time; 3% (9 genes) ⇒ exactly 9 gene-slots are mutated, chosen at random. For an
input gene both integers of the pair are mutated together — *"Both of these are
mutated at the same time, to ensure that every connection in the graph is still
valid."* `[verbatim]`

## 3. Evolutionary strategy — (1+4) ES

`[verbatim]`, Section V-A:

1. Randomly generate 5 genotypes; select the fittest.
2. Mutate the winner to make 4 offspring.
3. New generation = winner + its 4 offspring.
4. Select the winner:
   a. if any offspring has **better** fitness, the best becomes winner;
   b. **otherwise, an offspring with fitness equal to the best is chosen at random**;
   c. otherwise the parent remains.
5. Repeat until max generations or a solution is found.

Step 4b is the neutral-drift tie-break and is **not optional** — it is the
mechanism CGP's efficiency rests on. Mutation only; no crossover.

## 4. Module representation

`[verbatim]`. Module genotype = **header** (4 integers: module id, number of
module inputs, number of nodes, number of module outputs) + **body** (nodes and
module outputs, encoded as any ECGP genotype).

| Bound | Value |
|---|---|
| Nodes per module | min 2, max `ms` (user-set) |
| Module outputs | min 1, max `n` (= nodes in the module; one output per node) |
| Module inputs | min 2, max `2n` |

Module inputs are **not** encoded in the module genotype, so their count does not
affect its size. Not every node output need be connected ⇒ neutrality exists
inside modules too, identical to CGP's.

**No nesting.** Modules may contain only primitive-function nodes. Prevents
multi-level nesting, which caused code growth, stack overflows and out-of-memory
errors when decoding. Named as future work.

## 5. The global module list

`[verbatim]`:

- Stored globally, **an extension of the primitive function list**, shared by all
  individuals in the population.
- Any node may be mutated to represent any module or primitive in either list
  (subject to node-type rules).
- Dynamic, **no maximum size**.
- When the fittest individual is promoted, the list is **pruned to exactly the
  modules present in that individual** — everything found only in less fit
  individuals is deleted. This is the second-level "copy-or-die" selection and is
  the paper's stated reason the list does not grow without bound.

## 6. Genotype operators

### compress `[verbatim]`
- Pick two random points, respecting the module size limits; encapsulate all
  **type 0** nodes between them into a new module.
- **Aborts if any type I or type II node lies between the points** (no nesting).
- Initial module **inputs** = the connections from the encapsulated nodes' inputs
  to outputs of earlier nodes / program inputs. **Repeated connections to the same
  earlier node each get their own module input.**
- Initial module **outputs** = the connections from later nodes (right of the
  right-hand boundary) into the encapsulated nodes.
- The new module appears in the genotype as a **type I** node.
- All later nodes/inputs/outputs are relabelled so connections stay intact.

### expand `[verbatim]`
- Picks a random **type I** node and replaces it with the module's nodes.
- Applies to type I only. **Type II nodes are immune** — added to stop runaway
  genotype growth from replicate-then-expand cycles.
- Runs at **twice** the probability of compress; it is a module *destruction*
  operator.

Both are **fitness-neutral**: the genotype before and after represents the same
directed graph.

### genotype point mutation `[verbatim]`
- As CGP's, except it may set a node's function to any primitive **or any module
  in the module list**.
- Type 0 → module makes it **type II**. Type 0 and type II are treated
  identically by this operator; a type II node can mutate back to a primitive.
- **The function gene of a type I node is immune** to it. A type I node can only
  be removed by `expand`.
- On any type change, the node **keeps as many of its original inputs as it needs
  and randomly generates any extra**. Same when a type II node switches from
  module A to module B with a different input count.
- Input genes of type 0, I and II nodes are all mutable.

**This is the only route to module replication** — a module duplicates when a
point mutation writes its id into some node's function gene.

## 7. Module operators

`[verbatim]`. Five, all of which must respect the §4 input/output bounds at all
times.

1. **module point mutation** — a restricted genotype point mutation over the
   module body: may change input and function genes, but **may not introduce type
   II nodes** (primitives only), and **a module output may never be mutated to
   connect directly to a module input** (that would bypass the module's
   computation — a "junk" module).
2. **add-input** — increment the header's input count; insert one extra,
   randomly-valued input gene into *every* type I and type II node representing
   the module. **The module is renumbered to the next available id.**
3. **add-output** — increment the header's output count; append one extra,
   randomly-valued output gene to the module genotype. **Also renumbered.**
4. **remove-input** — decrement the count; drop the chosen input gene from every
   node representing the module.
5. **remove-output** — decrement the count; drop the chosen output gene from the
   module genotype.

> ⚠️ The paper's text for remove-input says *"the gene representing the number of
> module **inputs** ... is decremented"* under the **remove-output** heading — an
> evident typo in the source. Implemented as the symmetric reading: remove-output
> decrements outputs.

## 8. Function set

`AND, NAND, OR, NOR` `[verbatim]` — the set used for even-parity and for the
adders/comparators (*"The function set used is identical to the even parity
problem (AND, NAND, OR, NOR)"*).

## 9. Task-specific — `[our choice]`

The paper never ran the Kashtan–Alon retina, so nothing below has a source. Each
is a declared decision, not a recovered value.

| Item | Choice | Justification |
|---|---|---|
| Task | KA Fig. 5a retina, `kashtan_alon/tasks.py` (`task=retina`) | already verified against PMC1236541 in this repo |
| Program inputs / outputs | 8 / 1 | the task |
| Fitness | number of correct outputs over all **256** input patterns | exhaustive ⇒ no sampling noise; the natural analogue of the paper's boolean fitness |
| Function set | as §8 | keeps effort figures comparable with the paper's boolean problems |
| `ms` (max module size) | 5 | the paper's multiplier value; its own sweep found *no* correlation between `ms` and performance, so this is not worth sweeping |
| MVG | `and` ↔ `or`, switch every **E = 20** generations | matches `kashtan_alon/`; E=20 was measured directly in this repo (commit `74714f7`) |
| Termination | fixed generation budget, **not** run-until-solved | the paper runs uncapped until a solution is found; under MVG the target keeps moving, so "solved" is not terminal. Budget set once timing is measured. |
| Seeds | 50 | the paper's `[verbatim]` run count; kept for comparability |

## 10. Deliberate deviations

**Genotype size and mutation rate, for runs on this task.** Table II's 100 nodes /
3% is `[verbatim]` and remains the CLI default, so nothing deviates unless it is
typed explicitly. But it was tuned on parity, adders and multipliers, never on the
retina, and it solves this task in only 4 of 6 seeds within 25 000 generations.
The measured configuration (`RESULTS.md`, 2026-08-13) is:

    --nodes 800 --mutation-rate 0.01        # 6/6 solved, median 8 404 generations

Justification is empirical, not a preference: a sweep over nodes ∈ {50 … 1600} ×
rate ∈ {0.01, 0.02, 0.03} found 800 the plateau — 400 and 800 solve 6/6 where 100
and 200 do not, and 1600 regresses to 4/6 at ~6× the wall clock. Every run must
record which configuration it used.

Nothing else deviates. Where the paper specifies, we follow it exactly; §9 covers
what it does not specify.

## 11. Open cross-check (not a gap)

**Table I** — *"The Effect of the Operators on Each Node Type"* — is image-only
and was not read. The paper introduces it with *"To summarize, the properties of
type 0, I, and II nodes are shown in Table I"*, and every rule it summarises is
stated in the prose captured in §6. So it is a **cross-check, not a missing
value**. Worth one screenshot to confirm §6 against it before trusting the
implementation.

## 12. Implementation decisions the paper does not specify — all `[our choice]`

Written when `ecgp.py` was implemented (2026-08-14). These are **not** parameters and
none is `[inferred]`: each is a place where the paper's prose leaves a case open that
code must nevertheless decide. Every one is commented at its site in `ecgp.py`.

| # | Situation the paper leaves open | Decision | Why |
|---|---|---|---|
| 1 | "Pick two random points, respecting the module size limits" — two points, then reject illegal gaps? | Draw a legal length 2..`ms` uniformly, then a uniform position | Same support, no rejection loop, and independent points would bias toward short modules |
| 2 | `compress` on a run **nothing outside reads** (entirely inactive code) — it would have 0 outputs, below the min of 1 | Give it one output: the last encapsulated node | Refusing instead would make compress silently rarer on exactly the inactive regions CGP drifts in |
| 3 | Program outputs pointing *into* a compressed run — are they "later nodes"? | Yes; they get a module output like any other reference | The graph is otherwise left dangling |
| 4 | A node's **output count shrinks** (type II holding a 3-output module mutates to a primitive), while later nodes still ask for output 2 | Clamp those references to the highest surviving output | Smallest repair; re-drawing them would inject mutation into genes the operator never selected |
| 5 | A drawn input slot is **retired mid-application** because an earlier drawn function gene shrank that node's arity | The slot is spent, not re-drawn | Re-drawing raises the effective rate on every other gene |
| 6 | A drawn slot lands on a **type I function gene**, which is immune | Spent, not re-drawn | Same reason as 5; otherwise the rate would scale with how many type I nodes an individual carries |
| 7 | How many slots does **module point mutation** mutate? Called "a restricted genotype point mutation", given no rate of its own | The genotype rule, `max(1, round(rate × slots))` | With 2-5 node bodies the `max(1, …)` floor decides it either way |
| 8 | **remove-input**: what happens to body genes that referenced the dropped input? | Re-pointed at a surviving module input chosen at random | A body node still needs two inputs; deleting is not representable |
| 9 | **remove-output**: what happens to outer references carrying the dropped index? | Re-pointed at a surviving output at random; higher indices shift down | Same; the alternative is a dangling reference |
| 10 | **Order** of the operators within one offspring | compress → expand → module operators → genotype point mutation | Point mutation then sees the module list the offspring is actually evaluated with, so a module made this generation can be re-used immediately |
| 11 | Are the five module rates per offspring or per module? | Per module, per offspring | A module is the object they act on; the genotype operators stay per offspring |
| 12 | "Set a node's function to any primitive or any module" — with what distribution? | Uniform over primitives ∪ module list | The unqualified reading; nothing in the paper weights them |
| 13 | The second integer of a reference, when the source has several outputs | Drawn uniformly over the source's outputs, together with the label | The paper mutates the pair together "to ensure that every connection is still valid" |
| 14 | The module list is "global and shared", but a (1+4) ES mutates 4 offspring independently | Each individual owns its dict; the **winner's** list, pruned to what it uses, becomes the next generation's | A literally shared list would let one offspring's module operator change another's fitness. Pruning-on-promotion is the paper's own rule (section 5), so this is the same semantics one generation at a time |
