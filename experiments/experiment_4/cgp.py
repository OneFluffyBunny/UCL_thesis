"""CGP genotype, mutation and evaluation -- pure Python, no numpy in the hot loop.

Spec: `PAPER_SPEC.md` (Walker & Miller, IEEE TEVC 12(4), 2008). Sections cited
below refer to it.

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
faster per generation** at 100-800 nodes with identical semantics. Arrays are still
used for I/O (`np.savez`) and for summary statistics, never inside the loop.

⚠️ The random stream therefore differs from the numpy-era code: a given seed does
not reproduce a pre-2026-08-13 run. Reproducibility *within* this implementation is
unaffected (same seed -> same run, and checkpoint/resume restores `getstate()`).

EVALUATION -- truth-table masks. Each wire's behaviour over *all* 2**n_in input
patterns is a single Python int: bit `r` is the wire's value on pattern `r`. A
gate is then one bitwise op over every pattern simultaneously, and fitness is a
population count. This is exact, not approximate -- `evaluate_slow` below is the
obvious per-pattern implementation and `test_cgp.py` asserts the two agree.

FURTHER SPEED, NOT TAKEN (yet). This module is now almost entirely Python-integer
bitwise work, which is **PyPy's best case**; experiment 4 needs no JAX (numpy is
only used for I/O and stats), so it could run under PyPy for plausibly another
5-10x with no code change. That needs a separate environment, so it is recorded
here as an option rather than a dependency. Numba is a poorer fit: the hot loop is
a data-dependent graph walk, not an array kernel.
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

def _draw_slots(rnd: random.Random, total: int, k: int) -> list[int] | set[int]:
    """`k` distinct slot indices drawn uniformly from `range(total)`.

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
    """
    if k >= total:
        return list(range(total))
    if k * 2 > total:
        return rnd.sample(range(total), k)
    rand = rnd.random
    picked: set[int] = set()
    add = picked.add
    while len(picked) < k:
        add(int(rand() * total))
    return picked


def _draw_slots_biased(rnd: random.Random, total: int, n_func: int, k: int,
                       w: float) -> set[int]:
    """`k` distinct slot indices with WIRING slots down-weighted by `w`.

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
    """
    rand = rnd.random
    picked: set[int] = set()
    add = picked.add
    while len(picked) < k:
        s = int(rand() * total)
        if s >= n_func and rand() >= w:
            continue
        add(s)
    return picked


def random_genotype(rnd: random.Random, n_nodes: int, n_in: int,
                    n_outputs: int, n_funcs: int, arity: int) -> Genotype:
    """A uniformly random valid genotype.

    Output genes are initialised to the last `n_outputs` nodes, per the paper:
    "These integers are initially chosen so that the program outputs are given by
    the outputs of the last O nodes in the genotype."
    """
    rand = rnd.random
    func = [int(rand() * n_funcs) for _ in range(n_nodes)]
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
    for s in slots:
        if s < n:
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
    informative structural readout on a left/right retina is instead **where each
    node's input cone comes from**:

      left   -- the node depends only on retina pixels 0-3
      right  -- only on pixels 4-7
      mixed  -- on both halves
      const  -- on neither (a constant gate)

    A modular solution shows up directly as a large left group and a large right
    group that meet only in a few mixed nodes near the output. A smeared,
    non-modular solution is mixed almost everywhere. This needs no community
    detection and no null model, so it is available now and is not hostage to the
    modularity-metric work.
    """
    active: list[int]                 # node indices in the phenotype
    depth: dict[int, int]             # node index -> longest path from a program input
    cone: dict[int, frozenset[int]]   # node index -> program inputs it depends on
    cls: dict[int, str]               # node index -> "left" / "right" / "mixed" / "const"
    out_nodes: list[int]              # node index feeding each program output (-1 if an input)

    @property
    def n_active(self) -> int:
        return len(self.active)

    def counts(self) -> dict[str, int]:
        c = {"left": 0, "right": 0, "mixed": 0, "const": 0}
        for j in self.active:
            c[self.cls[j]] += 1
        return c


def phenotype(g: Genotype, n_in: int, gates: Sequence[Gate],
              split: int | None = None) -> Phenotype:
    """Decode the genotype's structure. `split` = first input of the right half.

    For the 8-pixel retina `split=4`: inputs 0-3 are the left 2x2 block and 4-7 the
    right one (see kashtan_alon/tasks.py's pixel map). With `split=None` every node
    with a non-empty cone is classed "mixed", which is the honest answer for tasks
    that have no left/right structure.

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

    cls: dict[int, str] = {}
    for j in active:
        c = cone[j]
        if not c:
            cls[j] = "const"
        elif split is None:
            cls[j] = "mixed"
        elif max(c) < split:
            cls[j] = "left"
        elif min(c) >= split:
            cls[j] = "right"
        else:
            cls[j] = "mixed"

    out_nodes = [lbl - n_in if lbl >= n_in else -1 for lbl in g.ogene]
    return Phenotype(active=active, depth=depth, cone=cone, cls=cls,
                     out_nodes=out_nodes)


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
            vals[n_in + j] = ~(x & vals[conn[b + 1]]) & mask
        elif op == 1:                                 # or
            vals[n_in + j] = x | vals[conn[b + 1]]
        elif op == 3:                                 # nor
            vals[n_in + j] = ~(x | vals[conn[b + 1]]) & mask
        elif op == 4:                                 # xor
            vals[n_in + j] = x ^ vals[conn[b + 1]]
        elif op == 5:                                 # xnor
            vals[n_in + j] = ~(x ^ vals[conn[b + 1]]) & mask
        elif op == 6:                                 # not
            vals[n_in + j] = ~x & mask
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

def hits(out_mask: int, target: int, mask: int) -> int:
    """Number of input patterns the circuit gets right (0 .. n_patterns)."""
    return int((~(out_mask ^ target) & mask).bit_count())


def balanced_score(out_mask: int, target: int, mask: int) -> float:
    """Balanced accuracy in [0, 1]: mean of true-positive and true-negative rates.

    NOT the default. On this task raw hits rate a one-half predictor exactly level
    with a constant predictor (KA's equal-cell property), whereas balanced accuracy
    rates the one-half predictor at 0.833 and thereby manufactures a gradient into
    the one-module solution -- the trap that stalled experiment 1. Offered so that
    contrast can be demonstrated, not so it can be used.
    """
    pos = target & mask
    neg = ~target & mask
    n_pos, n_neg = pos.bit_count(), neg.bit_count()
    if n_pos == 0 or n_neg == 0:
        return float(hits(out_mask, target, mask)) / max(1, mask.bit_count())
    tp = (out_mask & pos).bit_count()
    tn = (~out_mask & neg & mask).bit_count()
    return 0.5 * (tp / n_pos + tn / n_neg)


def fitness(g: Genotype, gates: Sequence[Gate], in_masks: Sequence[int],
            target: int, mask: int, n_in: int, kind: str = "raw") -> tuple[float, int]:
    """(selection score, raw hits).

    `hits` is always the raw count so "solved" (hits == n_patterns) means the same
    thing under either scoring; only the score driving selection changes.
    """
    out = evaluate(g, gates, in_masks, mask, n_in)[0]
    h = hits(out, target, mask)
    if kind == "raw":
        return float(h), h
    if kind == "balanced":
        return balanced_score(out, target, mask), h
    raise ValueError(f"unknown fitness kind: {kind!r}")
