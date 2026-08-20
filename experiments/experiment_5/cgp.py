"""CGP genotype, mutation and evaluation -- pure Python, no numpy anywhere.

FORKED FROM `experiment_4/cgp.py`. Experiment 4's copy is frozen: its logged runs and
their seeds have to stay reproducible, so this file -- not that one -- is where the
big-brain work happens. Three things differ, all of them marked `[exp5]` below:

  1. MANY OUTPUTS. A program has `O` outputs, not one. `fitness` scores every output
     over every pattern, so the perfect score is `O * n_patterns`.
  2. INTERPRETER-INDEPENDENT RANDOMNESS. `_draw_slots` used to return a `set`, and
     `mutate` iterated it. Set iteration order for ints is an implementation detail:
     CPython and PyPy disagree, so the same seed would have produced different runs
     under the two interpreters and "PyPy gives the same answers" would have been
     unprovable. The picks are now returned in DRAW order. The *set* of slots drawn
     from a given RNG state is bit-for-bit what experiment 4 draws (`test_cgp.py`
     asserts exactly this against `experiment_4/cgp.py`); only the order in which they
     are then written changes, and with it the stream of values written into them.
     ⚠️ So an experiment-4 seed does not reproduce here. That is the same class of
     break as the numpy->`random` switch documented below, and it buys the ability to
     run the identical experiment under either interpreter.
  3. GROUPS, NOT A SPLIT. The structural readout classified nodes as left/right/mixed
     against one `split` index. A big-brain task has K sub-problems, so it now
     classifies against a partition of the inputs (`tasks.input_groups`). At K=2 the
     class names are still `left` / `right`, so experiment 4's columns and diagrams
     are unchanged.

Spec: `PAPER_SPEC.md` (Walker & Miller, IEEE TEVC 12(4), 2008). Sections cited below
refer to it.

REPRESENTATION (spec section 1). One row; a node may connect to *any* previous
node -- there is no levels-back parameter in this formulation. Program inputs are
labelled `0 .. n_in-1`; node `j` is labelled `n_in + j` and may reference any
label strictly below its own.

Genes are stored as **pairs**, which is the ECGP encoding:

    func[j], ntype[j]              function gene:  (function-or-module id, node type)
    conn[j*a+k], cout[j*a+k]       input gene k:   (source label, which output of it)
    ogene[o], ocout[o]             output gene o:  (source label, which output of it)

For pure CGP every `ntype` is 0 and every `*cout` is 0, exactly as the paper
states ("every node encoded in the ECGP genotype is of node type 0, and the second
integer of each pair encoding the node inputs is always 0"). Keeping the pairs now
means ECGP slots in later without rewriting the genotype.

WHY PLAIN LISTS AND `random.Random`, NOT NUMPY. The genotype used to be int32
arrays. That cost more than it bought: `evaluate` called `.tolist()` on the whole
genotype on every call -- converting an 800x2 array to use ~65 of its rows -- which
was 8-27% of runtime and *grew* with genotype size, and `rng.integers()` costs
~1-2 us per scalar draw against ~0.1 us for `random.randrange`. Switching the
genotype to Python lists and the search RNG to `random.Random` measured **2.2-3.2x
faster per generation** at 100-800 nodes with identical semantics. The absence of
numpy is now also a hard requirement rather than a preference: this module has to
import under PyPy, where numpy is not available.

EVALUATION -- truth-table masks. Each wire's behaviour over *all* input patterns is a
single Python int: bit `r` is the wire's value on pattern `r`. A gate is then one
bitwise op over every pattern simultaneously, and fitness is a population count. This
is exact, not approximate -- `evaluate_slow` below is the obvious per-pattern
implementation and `test_cgp.py` asserts the two agree.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from gates import Gate


@dataclass
class Genotype:
    """One individual, held as plain Python lists (see the module docstring).

    `conn` and `cout` are **flat**, row-major: gene `k` of node `j` is at index
    `j * arity + k`. Flat beats a list-of-lists here because the evaluator indexes
    them millions of times and one indirection per access is measurable.
    """
    func: list[int]     # (n_nodes,)             function id, indexes the gate set
    ntype: list[int]    # (n_nodes,)             node type; always 0 in CGP
    conn: list[int]     # (n_nodes * arity,)     source label per input gene
    cout: list[int]     # (n_nodes * arity,)     which output of that source; always 0 in CGP
    ogene: list[int]    # (n_outputs,)           source label per program output
    ocout: list[int]    # (n_outputs,)           which output of it; always 0 in CGP
    arity: int          # input genes per node -- needed to index the flat lists

    def copy(self) -> "Genotype":
        return Genotype(self.func[:], self.ntype[:], self.conn[:], self.cout[:],
                        self.ogene[:], self.ocout[:], self.arity)

    @property
    def n_nodes(self) -> int:
        return len(self.func)

    @property
    def n_outputs(self) -> int:
        return len(self.ogene)

    def inputs_of(self, j: int) -> list[int]:
        """Source labels of node `j`'s input genes -- the readable accessor for the
        flat layout. Not used in the hot loop, which indexes `conn` directly."""
        b = j * self.arity
        return self.conn[b:b + self.arity]


def n_gene_slots(n_nodes: int, arity: int, n_outputs: int) -> int:
    """Total mutable gene slots.

    `n_nodes * (1 + arity)` is the paper's count -- 100 nodes x 3 = "300 genes"
    (Table II) -- plus the `O` output integers appended to the genotype. Both are
    mutable, so both are counted. For the retina (O = 1) this is 301, and 3% of it
    still rounds to the 9 genes Table II names.
    """
    return n_nodes * (1 + arity) + n_outputs


def n_mutations(n_nodes: int, arity: int, n_outputs: int, rate: float) -> int:
    """Number of gene slots mutated per application. At least 1."""
    return max(1, int(round(rate * n_gene_slots(n_nodes, arity, n_outputs))))


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------

def _draw_slots(rnd: random.Random, total: int, k: int) -> list[int]:
    """`k` distinct slot indices drawn uniformly from `range(total)`, IN DRAW ORDER.

    `random.sample` is correct but heavy: it builds a set/pool and routes every draw
    through `_randbelow_with_getrandbits`, and profiling showed `mutate` (dominated
    by exactly this) taking 59% of the search loop. When `k` is small relative to
    `total` -- always true here, e.g. 36 of 1601 -- rejection sampling into a set is
    the same distribution for ~`k` cheap draws: expected collisions are
    `k^2 / 2*total`, under one for our sizes. `random()` is a single C call against
    `randrange`'s Python-level bit_length loop.

    The `int(random() * total)` bias is bounded by `total / 2**53` and is irrelevant
    at these magnitudes. Falls back to `sample` when `k` is a large fraction of
    `total`, where rejection would thrash.

    `[exp5]` RETURNS A LIST, IN THE ORDER THE SLOTS WERE DRAWN. Experiment 4 returned
    the `set` and let `mutate` iterate it, which made the run depend on CPython's set
    layout for small ints -- PyPy's differs, so the same seed diverged between
    interpreters. The draw loop below is unchanged (same `random()` calls, same
    acceptance test, so the same slots come out of the same RNG state -- `test_cgp.py`
    asserts that against experiment 4's function); only the container changes, which
    fixes the order the caller writes them in and makes a run interpreter-independent.
    """
    if k >= total:
        return list(range(total))
    if k * 2 > total:
        return rnd.sample(range(total), k)
    rand = rnd.random
    picked: set[int] = set()
    order: list[int] = []
    add = picked.add
    while len(picked) < k:
        s = int(rand() * total)
        if s not in picked:
            add(s)
            order.append(s)
    return order


def _draw_slots_biased(rnd: random.Random, total: int, n_func: int, k: int,
                       w: float) -> list[int]:
    """`k` distinct slot indices, IN DRAW ORDER, with WIRING slots down-weighted by `w`.

    ⚠️ [our choice -- NOT in the paper]. Walker & Miller mutate every gene slot with
    equal probability, so with arity 2 a node's function gene is picked 1/3 of the
    time and its two input genes 2/3. This samples a function slot with weight 1 and
    a wiring slot (an input gene, or an output gene -- both change *what connects to
    what*) with weight `w < 1`, testing whether the operator's step size is what
    breaks MVG: a rewire relocates a whole subtree, a function swap edits one gate
    in place.

    Sequential rejection rather than a cumulative-weight table: draw uniformly, keep
    a wiring slot with probability `w`. That is exactly sampling without replacement
    with weights 1 : w, and stays a couple of cheap `random()` calls per pick, which
    matters because `mutate` is ~59% of the search loop.

    `[exp5]` Draw-ordered list, for the reason given in `_draw_slots`.
    """
    rand = rnd.random
    picked: set[int] = set()
    order: list[int] = []
    add = picked.add
    while len(picked) < k:
        s = int(rand() * total)
        if s >= n_func and rand() >= w:
            continue
        if s not in picked:
            add(s)
            order.append(s)
    return order


def random_genotype(rnd: random.Random, n_nodes: int, n_in: int,
                    n_outputs: int, n_funcs: int, arity: int) -> Genotype:
    """A uniformly random valid genotype.

    Output genes are initialised to the last `n_outputs` nodes, per the paper:
    "These integers are initially chosen so that the program outputs are given by
    the outputs of the last O nodes in the genotype."
    """
    rand = rnd.random
    # A single-gate function set (e.g. --gates nand) has only one legal value, so
    # drawing it is pure overhead -- every node's function gene is 0 regardless.
    func = [0] * n_nodes if n_funcs == 1 else [int(rand() * n_funcs) for _ in range(n_nodes)]
    conn: list[int] = []
    for j in range(n_nodes):
        # node j is labelled n_in + j and may reference any label below that
        lim = n_in + j
        for _ in range(arity):
            conn.append(int(rand() * lim))
    return Genotype(func=func, ntype=[0] * n_nodes, conn=conn,
                    cout=[0] * (n_nodes * arity),
                    ogene=list(range(n_in + n_nodes - n_outputs, n_in + n_nodes)),
                    ocout=[0] * n_outputs, arity=arity)


# ---------------------------------------------------------------------------
# mutation (spec section 6, "genotype point mutation")
# ---------------------------------------------------------------------------

def mutate(g: Genotype, rnd: random.Random, n_mut: int, n_in: int,
           n_funcs: int, wire_w: float = 1.0) -> Genotype:
    """Return a mutated copy with exactly `n_mut` gene slots resampled.

    Table II gives "genotype point mutation probability 1" -- the operator is
    applied every time, mutating a fixed *count* of slots (3% = 9), not each slot
    with probability 0.03.

    A resampled gene is drawn uniformly from its legal range and may land on its
    current value; the paper does not require a mutation to change anything, and
    forcing a change would alter the neutral-drift behaviour the (1+4) ES depends
    on.

    `n_funcs == 1` (a single-gate function set, e.g. `--gates nand`) is a special
    case worth noting: every function gene is provably always 0 -- init sets it,
    and this operator can only resample it to `int(rand() * 1) == 0` -- so a
    function-gene draw is a no-op *by construction*, not just in expectation. The
    slot is still counted in `n_mut` / `total` (the mutation budget and its
    distribution over slot kinds are unchanged -- this is a speed fix, not an
    effective-rate change), but the wasted draw-and-write is skipped below.

    `wire_w` is a deviation from the paper and defaults to 1.0, which is the paper:
    every slot equally likely. See `_draw_slots_biased`.
    """
    out = g.copy()
    n, a, n_out = g.n_nodes, g.arity, g.n_outputs
    func, conn, ogene = out.func, out.conn, out.ogene
    rand = rnd.random

    # Flat slot indexing: [0, n)          -> function gene of node j
    #                     [n, n + n*a)    -> input gene (j, k), i.e. conn[s - n]
    #                     [n + n*a, ...)  -> output gene o
    total = n_gene_slots(n, a, n_out)
    n_a = n + n * a
    k = min(n_mut, total)
    slots = (_draw_slots(rnd, total, k) if wire_w == 1.0
             else _draw_slots_biased(rnd, total, n, k, wire_w))
    single_func = n_funcs == 1
    for s in slots:
        if s < n:
            if not single_func:
                func[s] = int(rand() * n_funcs)
        elif s < n_a:
            t = s - n                       # flat conn index == j * a + k
            conn[t] = int(rand() * (n_in + t // a))
            # the paired output-index gene is mutated at the same time -- "Both of
            # these are mutated at the same time, to ensure that every connection
            # in the graph is still valid". Single-output nodes in CGP => always 0.
            out.cout[t] = 0
        else:
            o = s - n_a
            ogene[o] = int(rand() * (n_in + n))
            out.ocout[o] = 0
    return out


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def _uniform_binary(gates: Sequence[Gate], arity: int) -> bool:
    """True when every gate takes exactly 2 inputs and nodes have 2 input genes.

    That is the paper's function set (AND/NAND/OR/NOR) and every default run, so it
    gets an unrolled walk. The generic path below stays correct for `--gates`
    selections containing `not`/`const0`/`const1`.
    """
    return arity == 2 and all(gt.arity == 2 for gt in gates)


def active_nodes(g: Genotype, n_in: int, gates: Sequence[Gate]) -> list[int]:
    """Node indices (0-based, not labels) reachable from the program outputs.

    The genotype-phenotype map leaves the rest inactive -- CGP's neutrality. Only
    the arity actually used by a node's gate is followed, so surplus input genes
    do not drag nodes into the phenotype.

    Returned ascending, which is already a valid topological order because node `j`
    may only reference labels below `n_in + j`.
    """
    n, a, conn, func = g.n_nodes, g.arity, g.conn, g.func
    seen = bytearray(n)                 # bytearray beats a set: no hashing, no resize
    active: list[int] = []
    stack = [lbl - n_in for lbl in g.ogene if lbl >= n_in]

    if _uniform_binary(gates, a):
        while stack:
            j = stack.pop()
            if seen[j]:
                continue
            seen[j] = 1
            active.append(j)
            b = j + j
            lbl = conn[b]
            if lbl >= n_in:
                stack.append(lbl - n_in)
            lbl = conn[b + 1]
            if lbl >= n_in:
                stack.append(lbl - n_in)
    else:
        arities = [gt.arity for gt in gates]
        while stack:
            j = stack.pop()
            if seen[j]:
                continue
            seen[j] = 1
            active.append(j)
            b = j * a
            for k in range(arities[func[j]]):
                lbl = conn[b + k]
                if lbl >= n_in:
                    stack.append(lbl - n_in)
    active.sort()
    return active


@dataclass
class Phenotype:
    """The decoded circuit: active nodes plus the structure we actually report.

    Density is NOT reported for CGP. It is a flawed metric here for the same reason
    it was in experiment 1, and worse: a CGP genotype's edge count is fixed by
    construction (`n_nodes * arity`), so "density" would be a constant. The
    informative structural readout is instead **where each node's input cone comes
    from**, measured against the task's own decomposition (`tasks.input_groups`):

      <group name>  -- the node depends only on inputs inside ONE group
      mixed         -- it depends on inputs from several groups
      const         -- on none (a constant gate)

    A modular solution shows up directly as large pure groups meeting only in a few
    mixed nodes near the outputs; a smeared one is mixed almost everywhere. This needs
    no community detection and no null model.

    `[exp5]` Experiment 4 had exactly two groups and hard-coded their names as
    `left` / `right`. Here the count is the task's (K retinas, K adder positions,
    ...), so the names come from `group_names`, which still yields `left` / `right`
    at K=2 -- experiment 4's CSV columns and diagrams are byte-compatible.

    ⚠️ These classes are STRUCTURAL: they say what a node is *wired to*, not what it
    is *sensitive to*. A node can sit in the cone of an output and have no influence
    on it whatsoever. `behavioural_deps` below is the exact behavioural counterpart,
    and the two disagreeing is a finding rather than a bug.
    """
    active: list[int]                 # node indices in the phenotype
    depth: dict[int, int]             # node index -> longest path from a program input
    cone: dict[int, frozenset[int]]   # node index -> program inputs it depends on
    cls: dict[int, str]               # node index -> group name / "mixed" / "const"
    out_nodes: list[int]              # node index feeding each program output (-1 if an input)
    names: tuple[str, ...]            # the group names, in group order
    out_cls: list[str]                # class of each program output's own cone

    @property
    def n_active(self) -> int:
        return len(self.active)

    def counts(self) -> dict[str, int]:
        """Active nodes per class. Keys: every group name, then `mixed`, `const`."""
        c = {n: 0 for n in self.names}
        c["mixed"] = 0
        c["const"] = 0
        for j in self.active:
            c[self.cls[j]] += 1
        return c

    def n_pure_outputs(self) -> int:
        """Program outputs whose whole input cone sits inside a single group.

        The headline structural number for a many-output task: if the circuit really
        did decompose, every output is wired only to its own sub-problem's inputs.
        """
        return sum(1 for c in self.out_cls if c not in ("mixed", "const"))


def group_names(groups) -> tuple[str, ...]:
    """Display names for the input groups.

    Two groups keep experiment 4's `left` / `right`, so its log columns, its diagram
    colours and any comparison against it all continue to work unchanged. More than
    two get `g0 .. g(K-1)`, which is the honest generic answer.
    """
    if not groups:
        return ()
    if len(groups) == 2:
        return ("left", "right")
    return tuple(f"g{i}" for i in range(len(groups)))


def phenotype(g: Genotype, n_in: int, gates: Sequence[Gate],
              groups: Sequence[frozenset[int]] | None = None) -> Phenotype:
    """Decode the genotype's structure. `groups` partitions the program inputs.

    `groups` comes from `tasks.input_groups(task)` -- the task's own decomposition
    into sub-problems. With `groups=None` every node with a non-empty cone is classed
    "mixed", which is the honest answer for a task that has no known decomposition.

    Cones and depths are accumulated over the **active nodes only**. That is both
    correct and much cheaper than the whole genotype: every predecessor of an active
    node is itself active (the backward walk followed it), and `active` is ascending
    so a node's predecessors are always already resolved. Sweeping all `n_nodes`
    instead cost 1.5 ms at 800 nodes, which matters because an MVG run logs every
    `--switch-interval` (20) generations.
    """
    active = active_nodes(g, n_in, gates)
    a, conn, func = g.arity, g.conn, g.func

    cone: dict[int, frozenset[int]] = {}
    depth: dict[int, int] = {}
    for j in active:
        acc: set[int] = set()
        d = 0
        b = j * a
        for k in range(gates[func[j]].arity):
            lbl = conn[b + k]
            if lbl < n_in:
                acc.add(lbl)
                if d < 1:
                    d = 1
            else:
                acc |= cone[lbl - n_in]
                dp = depth[lbl - n_in] + 1
                if dp > d:
                    d = dp
        cone[j] = frozenset(acc)
        depth[j] = d

    names = group_names(groups)
    # input -> its group index, so classifying a cone is one pass over the cone
    # rather than K set intersections per node
    owner = [0] * n_in
    if groups:
        for gi, grp in enumerate(groups):
            for i in grp:
                owner[i] = gi

    def classify(c: frozenset[int]) -> str:
        if not c:
            return "const"
        if not groups:
            return "mixed"
        first = owner[next(iter(c))]
        for i in c:
            if owner[i] != first:
                return "mixed"
        return names[first]

    cls = {j: classify(cone[j]) for j in active}

    out_nodes = [lbl - n_in if lbl >= n_in else -1 for lbl in g.ogene]
    # a program output wired straight to input `lbl` has that single input as its cone
    out_cls = [classify(cone[j]) if j >= 0 else classify(frozenset({lbl}))
               for j, lbl in zip(out_nodes, g.ogene)]
    return Phenotype(active=active, depth=depth, cone=cone, cls=cls,
                     out_nodes=out_nodes, names=names, out_cls=out_cls)


def evaluate(g: Genotype, gates: Sequence[Gate], in_masks: Sequence[int],
             mask: int, n_in: int) -> list[int]:
    """Output truth-table masks, one per program output. Bit-parallel over patterns.

    The hot path of the whole experiment. Three things make it fast:

    1. Only the ACTIVE nodes are computed -- at 800 nodes that is ~65 gate
       evaluations rather than 800.
    2. The backward walk is **fused** into this function rather than calling
       `active_nodes`, so `conn`/`func` are looked up once and the arity test is
       hoisted out of the loop.
    3. Gate dispatch is inlined on the opcode (see `gates.OP_*`) instead of calling
       `gate.fn(args, mask)`, which would build a list and make a Python call per
       node -- both more expensive than the bitwise op they wrap.
    4. `[exp5]` Every complement is `x ^ mask`, never `~x & mask`. Equal for any `x`
       inside the mask, but one full-width allocation instead of two -- see
       `gates.py`. Worth nothing at 8 inputs and worth ~25% at 18.

    `evaluate_slow` is the readable reference; `test_cgp.py` asserts they agree.
    """
    n, a, conn, func, ogene = g.n_nodes, g.arity, g.conn, g.func, g.ogene

    # --- backward pass: which nodes are in the phenotype (see `active_nodes`)
    seen = bytearray(n)
    active: list[int] = []
    stack = [lbl - n_in for lbl in ogene if lbl >= n_in]
    if _uniform_binary(gates, a):
        while stack:
            j = stack.pop()
            if seen[j]:
                continue
            seen[j] = 1
            active.append(j)
            b = j + j
            lbl = conn[b]
            if lbl >= n_in:
                stack.append(lbl - n_in)
            lbl = conn[b + 1]
            if lbl >= n_in:
                stack.append(lbl - n_in)
    else:
        arities = [gt.arity for gt in gates]
        while stack:
            j = stack.pop()
            if seen[j]:
                continue
            seen[j] = 1
            active.append(j)
            b = j * a
            for k in range(arities[func[j]]):
                lbl = conn[b + k]
                if lbl >= n_in:
                    stack.append(lbl - n_in)
    active.sort()

    # --- forward pass, ascending == topological order
    ops = [gt.op for gt in gates]
    vals = list(in_masks) + [0] * n
    for j in active:
        b = j * a
        op = ops[func[j]]
        x = vals[conn[b]]
        if op == 0:                                   # and
            vals[n_in + j] = x & vals[conn[b + 1]]
        elif op == 2:                                 # nand
            vals[n_in + j] = (x & vals[conn[b + 1]]) ^ mask
        elif op == 1:                                 # or
            vals[n_in + j] = x | vals[conn[b + 1]]
        elif op == 3:                                 # nor
            vals[n_in + j] = (x | vals[conn[b + 1]]) ^ mask
        elif op == 4:                                 # xor
            vals[n_in + j] = x ^ vals[conn[b + 1]]
        elif op == 5:                                 # xnor
            vals[n_in + j] = x ^ vals[conn[b + 1]] ^ mask
        elif op == 6:                                 # not
            vals[n_in + j] = x ^ mask
        elif op == 7:                                 # const0
            vals[n_in + j] = 0
        else:                                         # const1
            vals[n_in + j] = mask
    return [vals[lbl] for lbl in ogene]


def evaluate_slow(g: Genotype, gates: Sequence[Gate], n_in: int,
                  n_patterns: int) -> list[list[bool]]:
    """Reference implementation: one pattern at a time, plain bools.

    Exists purely so `test_cgp.py` can assert the fast path agrees with it. Never
    used in the search loop.
    """
    a = g.arity
    outs: list[list[bool]] = [[] for _ in range(g.n_outputs)]
    for r in range(n_patterns):
        # pattern r's bit for input i. Must match tasks.input_masks' packing: that
        # takes column i of itertools.product([0,1], repeat=n_in), whose first
        # element varies slowest, so input i is bit (n_in-1-i) of r.
        vals = [bool((r >> (n_in - 1 - i)) & 1) for i in range(n_in)]
        vals = vals + [False] * g.n_nodes
        for j in range(g.n_nodes):
            gate = gates[g.func[j]]
            args = [vals[g.conn[j * a + k]] for k in range(gate.arity)]
            vals[n_in + j] = bool(gate.slow(args))
        for o, lbl in enumerate(g.ogene):
            outs[o].append(bool(vals[lbl]))
    return outs


# ---------------------------------------------------------------------------
# fitness
# ---------------------------------------------------------------------------

def hits(out_masks: Sequence[int], targets: Sequence[int], mask: int) -> int:
    """Correct (output, pattern) pairs. Perfect score is `n_out * n_patterns`.

    `[exp5]` Experiment 4 scored one output; here every program output is scored over
    every pattern and the counts are summed. Summing rather than averaging keeps the
    score an INTEGER, which the (1+4) ES depends on more than it looks: selection
    rules 4a/4b test `>` and `==` on it, and floating-point equality would make the
    neutral tie-break -- the mechanism CGP actually searches with -- fire erratically.
    """
    n_pat = mask.bit_count()
    t = 0
    for o, tg in zip(out_masks, targets):
        # `n_pat - popcount(disagreements)` rather than `popcount(agreements)`: the
        # latter needs a second full-width integer to hold the complement, and at 18
        # inputs that is 32 KB allocated and touched per output per evaluation.
        t += n_pat - (o ^ tg).bit_count()
    return t


def max_hits(n_out: int, n_patterns: int) -> int:
    """The perfect score: what `hits` returns for a circuit that is entirely right."""
    return n_out * n_patterns


def balanced_score(out_masks: Sequence[int], targets: Sequence[int],
                   mask: int) -> float:
    """Mean over outputs of balanced accuracy in [0, 1].

    NOT the default. On the retina, raw hits rate a one-half predictor exactly level
    with a constant predictor (KA's equal-cell property), whereas balanced accuracy
    rates the one-half predictor at 0.833 and thereby manufactures a gradient into
    the one-module solution -- the trap that stalled experiment 1. Offered so that
    contrast can be demonstrated, not so it can be used.
    """
    total = 0.0
    n_pat = mask.bit_count()
    for out_mask, target in zip(out_masks, targets):
        pos = target & mask
        neg = target ^ mask
        n_pos, n_neg = pos.bit_count(), neg.bit_count()
        if n_pos == 0 or n_neg == 0:
            total += (n_pat - (out_mask ^ target).bit_count()) / max(1, n_pat)
            continue
        tp = (out_mask & pos).bit_count()
        tn = ((out_mask ^ mask) & neg).bit_count()
        total += 0.5 * (tp / n_pos + tn / n_neg)
    return total / max(1, len(targets))


def fitness(g: Genotype, gates: Sequence[Gate], in_masks: Sequence[int],
            targets: Sequence[int], mask: int, n_in: int,
            kind: str = "raw") -> tuple[float, int]:
    """(selection score, raw hits).

    `hits` is always the raw count, so "solved" (`hits == max_hits`) means the same
    thing under either scoring; only the score driving selection changes.
    """
    outs = evaluate(g, gates, in_masks, mask, n_in)
    h = hits(outs, targets, mask)
    if kind == "raw":
        return float(h), h
    if kind == "balanced":
        return balanced_score(outs, targets, mask), h
    raise ValueError(f"unknown fitness kind: {kind!r}")


# ---------------------------------------------------------------------------
# behavioural dependency  `[exp5]`
# ---------------------------------------------------------------------------

def behavioural_deps(out_mask: int, in_masks: Sequence[int], mask: int,
                     n_in: int) -> frozenset[int]:
    """The program inputs this output's BEHAVIOUR actually depends on.

    Input `i` is in the set iff there exists at least one input pattern where flipping
    `i` alone changes the output -- the standard definition of a boolean function's
    support, and the exact behavioural counterpart of `Phenotype.cone`'s structural
    one. A wire can be in the cone and carry no influence at all (`x & ~x`, a gate
    whose output is masked off downstream, a module input the body ignores), so the
    two answers differ and the gap between them is itself worth logging.

    Computed with no enumeration. Under the exhaustive pattern ordering, patterns `r`
    and `r + 2**(n_in-1-i)` differ in input `i` alone whenever input `i` is 0 on `r`.
    So shifting the output mask down by that stride lines every pattern up against its
    flipped partner, XOR marks the disagreements, and `in_masks[i] ^ mask` keeps only the
    pairs counted once. One shift and three masks per input, whatever the circuit is.

    ⚠️ Tied to EXHAUSTIVE patterns: the stride identity is a fact about
    `itertools.product` ordering, not about masks in general. A sampled-pattern task
    would need this rewritten (`tasks.py` is where that decision lives).
    """
    deps = []
    for i in range(n_in):
        stride = 1 << (n_in - 1 - i)
        if (out_mask ^ (out_mask >> stride)) & (in_masks[i] ^ mask):
            deps.append(i)
    return frozenset(deps)


def behavioural_classes(out_masks: Sequence[int], in_masks: Sequence[int],
                        mask: int, n_in: int,
                        groups: Sequence[frozenset[int]] | None) -> list[str]:
    """Class of each program output by BEHAVIOURAL dependency, not by wiring.

    Same vocabulary as `Phenotype.out_cls` -- a group name, `mixed`, or `const` -- so
    the structural and behavioural readouts can be put side by side. `const` here
    means the output is genuinely a constant function, not merely that it is wired to
    nothing.
    """
    names = group_names(groups)
    owner = [0] * n_in
    if groups:
        for gi, grp in enumerate(groups):
            for i in grp:
                owner[i] = gi
    out = []
    for m in out_masks:
        d = behavioural_deps(m, in_masks, mask, n_in)
        if not d:
            out.append("const")
        elif not groups:
            out.append("mixed")
        else:
            gs = {owner[i] for i in d}
            out.append(names[gs.pop()] if len(gs) == 1 else "mixed")
    return out
