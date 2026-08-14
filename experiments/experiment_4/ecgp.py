"""ECGP -- Embedded Cartesian Genetic Programming: modules that evolve and get re-used.

Spec: `PAPER_SPEC.md` (Walker & Miller, IEEE TEVC 12(4), 2008); section numbers below
refer to it. Everything the paper states is implemented as stated; everything it does
not state is listed in PAPER_SPEC section 12 and marked `[our choice]` at the point of
use in this file.

WHY A SEPARATE MODULE FROM `cgp.py`. A CGP node has a fixed arity, so `cgp.Genotype`
stores `conn` as one flat row-major list -- the layout the hot loop depends on. An
ECGP node's arity is whatever its module needs (2 .. 2*ms), so the flat layout cannot
hold it. Rather than make `cgp.py` variable-arity and re-validate every FG/MVG number
already in `RESULTS.md`, ECGP gets its own representation here and `cgp.py` is left
exactly as it was. The two meet in `flatten()`.

`flatten()` IS THE BRIDGE, AND THE TEST. An ECGP individual inlines to an ordinary
CGP genotype (modules hold primitives only -- no nesting, section 4 -- so one pass
suffices). That buys three things: the structural readout (`cgp.phenotype`: cone
classes, depth, active nodes) is computed on the *effective circuit*, so ECGP and CGP
runs are measured by identical code; `compress`/`expand` can be asserted
fitness-neutral by flattening either side; and `evaluate` below can be checked against
`cgp.evaluate(flatten(...))`, which `test_ecgp.py` does on random individuals.

LABELS. As in CGP: program inputs are `0 .. n_in-1`, node `j` is `n_in + j`, and a
node may reference any label strictly below its own. A *reference* is a pair
`(label, output_index)` -- ECGP nodes can have several outputs, so the second integer
picks one (section 1). Inside a module body the same scheme is offset instead by the
module's input count: `0 .. m_in-1` are module inputs, `m_in + j` is body node `j`.

MODULE IDENTITY. Module ids live in the same space as primitive function ids and start
at `n_prim`, so `func[j] < n_prim` means "primitive" and anything else is a module id
-- the paper's "the module list is an extension of the primitive function list"
(section 5). Ids are handed out by a per-individual counter and never reused.

WHY MODULES ARE OWNED BY THE INDIVIDUAL. The paper calls the module list global and
shared. Under a (1+4) ES that cannot be implemented literally: the four offspring are
mutated independently, so a module operator applied for one offspring would silently
change the fitness of the other three. Each individual therefore carries its own dict
and the *winner's* dict becomes the list for the next generation, pruned to the
modules it actually uses -- which is exactly the paper's promotion rule (section 5)
and gives the same semantics one generation at a time.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from typing import Sequence

import cgp
from gates import Gate


# ---------------------------------------------------------------------------
# representation
# ---------------------------------------------------------------------------

@dataclass
class Module:
    """A module: 4-integer header + body (section 4).

    `mid` / `n_in` / `len(func)` / `len(out)` are the header's four integers -- id,
    input count, node count, output count. The body is `func` + `conn` (arity 2, flat
    row-major like CGP's) plus the output genes `out`.

    Body nodes are primitives only -- no nesting, which the paper adopted after
    multi-level nesting caused code growth and stack overflows when decoding.
    """
    mid: int
    n_in: int
    func: list[int]     # (n_nodes,)      primitive ids only
    conn: list[int]     # (n_nodes * 2,)  labels: [0, n_in) = module input, n_in + j = body node j
    out: list[int]      # (n_out,)        labels of BODY NODES only, never a module input

    def copy(self) -> "Module":
        return Module(self.mid, self.n_in, self.func[:], self.conn[:], self.out[:])

    @property
    def n_nodes(self) -> int:
        return len(self.func)

    @property
    def n_out(self) -> int:
        return len(self.out)


@dataclass
class Individual:
    """One ECGP genotype plus the module list it owns.

    `conn`/`cout` are lists-of-lists, not the flat layout `cgp.Genotype` uses, because
    node arity varies with the module a node represents.
    """
    func: list[int]           # (n_nodes,)  primitive id (type 0) or module id (types I/II)
    ntype: list[int]          # (n_nodes,)  0 = primitive, 1 = type I (compress), 2 = type II (reuse)
    conn: list[list[int]]     # per node: source label per input gene
    cout: list[list[int]]     # per node: which output of that source
    ogene: list[int]          # (n_outputs,) source label per program output
    ocout: list[int]          # (n_outputs,) which output of it
    modules: dict[int, Module]
    next_id: int              # next unused module id

    def copy(self) -> "Individual":
        # `modules` is copied shallowly: a Module is only ever *replaced* by a mutated
        # copy (see `_replace_module`), never edited in place, so a parent can share
        # Module objects with its offspring without either being able to corrupt the
        # other.
        return Individual(self.func[:], self.ntype[:],
                          list(map(list, self.conn)), list(map(list, self.cout)),
                          self.ogene[:], self.ocout[:],
                          dict(self.modules), self.next_id)

    @property
    def n_nodes(self) -> int:
        return len(self.func)

    @property
    def n_outputs(self) -> int:
        return len(self.ogene)


@dataclass
class Params:
    """Operator probabilities. Defaults are Table II `[verbatim]`; see `config.py`."""
    compress: float = 0.1
    expand: float = 0.2
    module_point: float = 0.04
    add_input: float = 0.01
    remove_input: float = 0.02
    add_output: float = 0.01
    remove_output: float = 0.02
    max_module_size: int = 5          # `ms`, PAPER_SPEC section 9 `[our choice]`
    mutation_rate: float = 0.03       # Table II


def n_outputs_of(ind: Individual, label: int, n_in: int) -> int:
    """How many outputs the thing at `label` has -- the legal range of a `cout` gene."""
    if label < n_in:
        return 1                                  # a program input
    j = label - n_in
    if ind.ntype[j] == 0:
        return 1                                  # a primitive node
    return ind.modules[ind.func[j]].n_out


def arity_of(ind: Individual, j: int) -> int:
    """Input genes node `j` must carry: 2 for a primitive, else the module's input count."""
    if ind.ntype[j] == 0:
        return 2
    return ind.modules[ind.func[j]].n_in


def n_gene_slots(ind: Individual) -> int:
    """Mutable gene slots: one function gene + its input genes per node, plus outputs.

    Unlike CGP this is NOT a constant -- `compress` shrinks the genotype and `expand`
    grows it -- so the "3% of the genotype" mutation count is recomputed from the
    current individual every time the operator is applied (section 6).
    """
    return len(ind.func) + sum(map(len, ind.conn)) + len(ind.ogene)


def _slot_table(ind: Individual) -> tuple[list[int], int, int]:
    """(per-node slot start, node-slot count, total slots) -- the mutable-gene layout.

    Node slots are variable-width (a node carries one function gene plus one input gene
    per module input), so a flat slot index has to be resolved through a prefix table
    rather than CGP's constant stride. Built with `accumulate` at C speed and returned
    to the caller, because `mutate` needs the total (to size the mutation) and
    `point_mutate` needs the table -- computing it twice per offspring was over half of
    the operator's cost.
    """
    acc = list(itertools.accumulate((1 + n for n in map(len, ind.conn)), initial=0))
    n_node_slots = acc.pop()
    return acc, n_node_slots, n_node_slots + len(ind.ogene)


def n_mutations(ind: Individual, rate: float) -> int:
    """Gene slots mutated per application; at least 1. Same rule as `cgp.n_mutations`."""
    return max(1, int(round(rate * n_gene_slots(ind))))


def random_individual(rnd: random.Random, n_nodes: int, n_in: int, n_outputs: int,
                      n_prim: int) -> Individual:
    """A random all-primitive individual with an EMPTY module list (Table II).

    Identical in distribution to `cgp.random_genotype`: every node is type 0 with
    arity 2, and the program outputs start on the last `n_outputs` nodes.
    """
    rand = rnd.random
    func = [int(rand() * n_prim) for _ in range(n_nodes)]
    conn, cout = [], []
    for j in range(n_nodes):
        lim = n_in + j
        conn.append([int(rand() * lim), int(rand() * lim)])
        cout.append([0, 0])
    return Individual(func=func, ntype=[0] * n_nodes, conn=conn, cout=cout,
                      ogene=list(range(n_in + n_nodes - n_outputs, n_in + n_nodes)),
                      ocout=[0] * n_outputs, modules={}, next_id=n_prim)


# ---------------------------------------------------------------------------
# genotype point mutation (section 6)
# ---------------------------------------------------------------------------

def point_mutate(ind: Individual, rnd: random.Random, n_mut: int, n_in: int,
                 n_prim: int, table: tuple[list[int], int, int] | None = None) -> None:
    """CGP's point mutation, extended to modules. Mutates `ind` in place.

    Differences from `cgp.mutate`, all `[verbatim]` (section 6):

    * a function gene may be set to any primitive **or any module in the list**;
    * doing so from type 0 makes the node **type II** (a reuse) -- and this is the
      only route by which a module replicates;
    * **the function gene of a type I node is immune**; only `expand` removes one.
      A slot drawn on such a gene is spent, not redrawn: redrawing would quietly
      raise the effective mutation rate on every other gene in proportion to how
      many type I nodes the individual carries;
    * on a type change the node "keeps as many of its original inputs as it needs and
      randomly generates any extra";
    * both integers of a reference are mutated together, "to ensure that every
      connection in the graph is still valid" -- for ECGP the second integer is a real
      choice, so it is drawn uniformly over the source's outputs `[our choice]`.
    """
    n = len(ind.func)
    # Flat slot index -> (node, which gene), via a prefix table plus a bisect.
    starts, n_node_slots, total = table if table is not None else _slot_table(ind)

    choices = list(range(n_prim)) + list(ind.modules)      # primitives + module list
    rand = rnd.random
    ntype, func, modules = ind.ntype, ind.func, ind.modules

    def n_outs(lbl: int) -> int:
        """`n_outputs_of` inlined: this ran ~9 times per offspring and showed up hot."""
        if lbl < n_in:
            return 1
        j = lbl - n_in
        return 1 if ntype[j] == 0 else modules[func[j]].n_out

    for s in cgp._draw_slots(rnd, total, min(n_mut, total)):
        if s >= n_node_slots:                              # a program output gene
            o = s - n_node_slots
            lbl = int(rand() * (n_in + n))
            ind.ogene[o] = lbl
            ind.ocout[o] = int(rand() * n_outs(lbl))
            continue

        j = bisect.bisect_right(starts, s) - 1
        off = s - starts[j]

        if off > 0:                                        # an input gene
            k = off - 1
            # The slot table above names slots in the genotype as it was when the
            # operator was applied. A function gene drawn earlier in this same
            # application can have shrunk this node's arity, retiring the slot; it is
            # then spent rather than re-drawn, for the same reason a type I function
            # gene is (re-drawing would raise the effective rate on everything else).
            if k >= len(ind.conn[j]):
                continue
            lbl = int(rand() * (n_in + j))
            ind.conn[j][k] = lbl
            ind.cout[j][k] = int(rand() * n_outs(lbl))
            continue

        if ntype[j] == 1:                                  # type I function gene: immune
            continue

        was = n_outs(n_in + j)
        f = choices[int(rand() * len(choices))]
        func[j] = f
        ntype[j] = 0 if f < n_prim else 2                  # type 0 <-> type II
        _fit_arity(ind, j, n_in, rnd)
        if n_outs(n_in + j) < was:
            _clamp_refs_to(ind, j, n_in)


def _clamp_refs_to(ind: Individual, j: int, n_in: int) -> None:
    """Repair references into node `j` after its output count shrank.

    `[our choice]` -- PAPER_SPEC section 12. The paper is explicit that a node
    changing type keeps or generates the *inputs* it needs, and that both integers of
    an input gene are mutated together "to ensure that every connection in the graph
    is still valid". It never says what happens to the references pointing *at* a
    node whose output count just fell -- e.g. a type II node holding a 3-output module
    that mutates to a primitive, leaving some later node asking for its output 2.
    Those references are clamped to the highest surviving output: the smallest repair
    that keeps the graph valid, and unlike re-drawing them at random it does not
    inject mutation into parts of the genotype the operator never selected.
    """
    lbl = n_in + j
    k = n_outputs_of(ind, lbl, n_in)
    for x in range(j + 1, len(ind.func)):
        cx, ox = ind.conn[x], ind.cout[x]
        for t in range(len(cx)):
            if cx[t] == lbl and ox[t] >= k:
                ox[t] = k - 1
    for o in range(len(ind.ogene)):
        if ind.ogene[o] == lbl and ind.ocout[o] >= k:
            ind.ocout[o] = k - 1


def _fit_arity(ind: Individual, j: int, n_in: int, rnd: random.Random) -> None:
    """Resize node `j`'s input genes to its (possibly new) arity.

    "The node keeps as many of its original inputs as it needs and randomly generates
    any extra" `[verbatim]` -- so this truncates from the right and appends fresh
    valid references, never reshuffles what it keeps.
    """
    need = arity_of(ind, j)
    cur = ind.conn[j]
    if len(cur) > need:
        del cur[need:]
        del ind.cout[j][need:]
    else:
        rand = rnd.random
        while len(cur) < need:
            lbl = int(rand() * (n_in + j))
            cur.append(lbl)
            ind.cout[j].append(int(rand() * n_outputs_of(ind, lbl, n_in)))


# ---------------------------------------------------------------------------
# compress / expand (section 6) -- both are fitness-neutral by construction
# ---------------------------------------------------------------------------

def compress(ind: Individual, rnd: random.Random, ms: int, n_in: int) -> bool:
    """Encapsulate a random run of type 0 nodes into a new module. Returns success.

    Section 6: "pick two random points, respecting the module size limits; encapsulate
    all type 0 nodes between them", and **abort if any type I or type II node lies
    between the points** -- that abort is what forbids nesting.

    `[our choice]` The two points are drawn as a window of a uniformly-chosen legal
    length (2 .. ms) at a uniformly-chosen position, rather than two independent
    points that are then rejected unless their gap happens to be legal. Same support,
    no rejection loop, and it does not bias toward short modules the way independent
    points would.
    """
    n = len(ind.func)
    hi_len = min(ms, n)
    if hi_len < 2:
        return False
    ln = 2 + int(rnd.random() * (hi_len - 1))
    i = int(rnd.random() * (n - ln + 1))
    k = i + ln - 1
    for j in range(i, k + 1):
        if ind.ntype[j] != 0:                     # no nesting: abort, do not retry
            return False

    lo_lbl, hi_lbl = n_in + i, n_in + k

    # --- body, and the module's inputs.
    # "Initial module inputs = the connections from the encapsulated nodes' inputs to
    # outputs of earlier nodes / program inputs. Repeated connections to the same
    # earlier node each get their own module input" -- hence a fresh input per
    # external *gene occurrence*, with no de-duplication.
    body_func = [ind.func[j] for j in range(i, k + 1)]
    ext_lbl: list[int] = []
    ext_cout: list[int] = []
    slots: list[tuple[bool, int]] = []            # (is_internal, index)
    for j in range(i, k + 1):
        for t in range(2):
            lbl, c = ind.conn[j][t], ind.cout[j][t]
            if lo_lbl <= lbl <= hi_lbl:
                slots.append((True, lbl - lo_lbl))          # body node index
            else:
                slots.append((False, len(ext_lbl)))         # module input index
                ext_lbl.append(lbl)
                ext_cout.append(c)
    m_in = len(ext_lbl)
    body_conn = [(m_in + idx) if internal else idx for internal, idx in slots]

    # --- the module's outputs: "the connections from later nodes into the
    # encapsulated nodes". Program outputs count as later references too -- they are
    # connections into the encapsulated nodes and the graph breaks if they are not
    # given somewhere to point.
    used: set[int] = set()
    for j in range(k + 1, n):
        for lbl in ind.conn[j]:
            if lo_lbl <= lbl <= hi_lbl:
                used.add(lbl - lo_lbl)
    for lbl in ind.ogene:
        if lo_lbl <= lbl <= hi_lbl:
            used.add(lbl - lo_lbl)
    # `[our choice]` Nothing outside reads the run (it was entirely inactive code):
    # the paper's bound is "min 1" output, so give it the last encapsulated node.
    # Refusing to compress instead would make the operator silently rarer on exactly
    # the inactive regions CGP does most of its drifting in.
    refs = sorted(used) if used else [ln - 1]
    out_index = {r: o for o, r in enumerate(refs)}

    mid = ind.next_id
    ind.next_id += 1
    ind.modules[mid] = Module(mid=mid, n_in=m_in, func=body_func,
                              conn=body_conn, out=[m_in + r for r in refs])

    # --- splice: the run becomes ONE type I node, so everything after it shifts left.
    shift = ln - 1

    def remap(lbl: int, c: int) -> tuple[int, int]:
        if lbl < lo_lbl:
            return lbl, c
        if lbl <= hi_lbl:
            return lo_lbl, out_index[lbl - lo_lbl]         # now an output of the module
        return lbl - shift, c

    func = ind.func[:i] + [mid] + ind.func[k + 1:]
    ntype = ind.ntype[:i] + [1] + ind.ntype[k + 1:]
    conn = [c[:] for c in ind.conn[:i]] + [ext_lbl]
    cout = [c[:] for c in ind.cout[:i]] + [ext_cout]
    for j in range(k + 1, n):
        pairs = [remap(l, c) for l, c in zip(ind.conn[j], ind.cout[j])]
        conn.append([p[0] for p in pairs])
        cout.append([p[1] for p in pairs])
    pairs = [remap(l, c) for l, c in zip(ind.ogene, ind.ocout)]

    ind.func, ind.ntype, ind.conn, ind.cout = func, ntype, conn, cout
    ind.ogene = [p[0] for p in pairs]
    ind.ocout = [p[1] for p in pairs]
    return True


def expand(ind: Individual, rnd: random.Random, n_in: int) -> bool:
    """Replace a random type I node with its module's nodes. Returns success.

    Section 6: applies to **type I only** -- type II nodes are immune, which is what
    stops replicate-then-expand cycles from growing the genotype without bound. Runs
    at twice compress's probability because it is the destruction operator.
    """
    type1 = [j for j, t in enumerate(ind.ntype) if t == 1]
    if not type1:
        return False
    p = type1[int(rnd.random() * len(type1))]
    mod = ind.modules[ind.func[p]]
    m, base = mod.n_nodes, n_in + p
    shift = m - 1

    def from_module(lbl: int) -> tuple[int, int]:
        """A label inside the module body -> the outer label it becomes."""
        if lbl < mod.n_in:
            return ind.conn[p][lbl], ind.cout[p][lbl]      # the module's own input
        return base + (lbl - mod.n_in), 0                  # an inlined body node

    body_conn: list[list[int]] = []
    body_cout: list[list[int]] = []
    for j in range(m):
        pairs = [from_module(mod.conn[2 * j]), from_module(mod.conn[2 * j + 1])]
        body_conn.append([pairs[0][0], pairs[1][0]])
        body_cout.append([pairs[0][1], pairs[1][1]])

    def remap(lbl: int, c: int) -> tuple[int, int]:
        if lbl < base:
            return lbl, c
        if lbl == base:                                    # was a reference to the module
            return base + (mod.out[c] - mod.n_in), 0       # now to the body node behind it
        return lbl + shift, c

    func = ind.func[:p] + mod.func[:] + ind.func[p + 1:]
    ntype = ind.ntype[:p] + [0] * m + ind.ntype[p + 1:]
    conn = [c[:] for c in ind.conn[:p]] + body_conn
    cout = [c[:] for c in ind.cout[:p]] + body_cout
    for j in range(p + 1, len(ind.func)):
        pairs = [remap(l, c) for l, c in zip(ind.conn[j], ind.cout[j])]
        conn.append([q[0] for q in pairs])
        cout.append([q[1] for q in pairs])
    pairs = [remap(l, c) for l, c in zip(ind.ogene, ind.ocout)]

    ind.func, ind.ntype, ind.conn, ind.cout = func, ntype, conn, cout
    ind.ogene = [q[0] for q in pairs]
    ind.ocout = [q[1] for q in pairs]
    return True


# ---------------------------------------------------------------------------
# module operators (section 7)
# ---------------------------------------------------------------------------

def _nodes_using(ind: Individual, mid: int) -> list[int]:
    """Indices of the nodes (type I or II) that represent module `mid`."""
    return [j for j in range(len(ind.func))
            if ind.ntype[j] != 0 and ind.func[j] == mid]


def _replace_module(ind: Individual, old: int, mod: Module, renumber: bool) -> int:
    """Install `mod` in place of module `old`, optionally under a fresh id.

    Copy-on-write: the old Module object is never edited, so a parent sharing it with
    this offspring is unaffected. `renumber` implements "the module is renumbered to
    the next available id", which the paper requires of add-input and add-output.
    """
    if renumber:
        mod.mid = ind.next_id
        ind.next_id += 1
        for j in _nodes_using(ind, old):
            ind.func[j] = mod.mid
        del ind.modules[old]
    ind.modules[mod.mid] = mod
    return mod.mid


def module_point_mutate(ind: Individual, mid: int, rnd: random.Random,
                        n_prim: int, rate: float) -> None:
    """Restricted point mutation over a module's body (section 7, operator 1).

    Two restrictions, both `[verbatim]`: it **may not introduce type II nodes** (so
    functions are drawn from the primitives only, preserving "no nesting"), and **a
    module output may never be mutated to connect directly to a module input** --
    that would bypass the module's computation entirely and leave a junk module.

    `[our choice]` The number of slots mutated follows the genotype operator's rule,
    `max(1, round(rate * slots))`. The paper calls this "a restricted genotype point
    mutation" but gives it no separate rate; with bodies of 2-5 nodes the max(1, ...)
    floor decides it in practice either way.
    """
    mod = ind.modules[mid].copy()
    n = mod.n_nodes
    total = n + 2 * n + mod.n_out                 # function, input and output genes
    rand = rnd.random
    for s in cgp._draw_slots(rnd, total, max(1, int(round(rate * total)))):
        if s < n:                                 # function gene: primitives only
            mod.func[s] = int(rand() * n_prim)
        elif s < 3 * n:                           # input gene
            t = s - n
            mod.conn[t] = int(rand() * (mod.n_in + t // 2))
        else:                                     # output gene: body nodes only
            o = s - 3 * n
            mod.out[o] = mod.n_in + int(rand() * n)
    _replace_module(ind, mid, mod, renumber=False)


def add_input(ind: Individual, mid: int, rnd: random.Random, n_in: int) -> int | None:
    """Give the module one more input (section 7, operator 2). Bound: max `2n`.

    Every node representing the module gains one extra randomly-valued input gene,
    and the module is renumbered -- so the NEW id is returned (None if the bound
    blocked the operator).
    """
    mod = ind.modules[mid]
    if mod.n_in >= 2 * mod.n_nodes:
        return None
    new = mod.copy()
    # Appending rather than inserting keeps every body reference valid untouched:
    # module inputs occupy [0, n_in) and body nodes n_in + j, so inserting anywhere
    # else would shift the body labels for no gain.
    new.n_in += 1
    for j in range(new.n_nodes):
        for t in range(2):
            if new.conn[2 * j + t] >= mod.n_in:
                new.conn[2 * j + t] += 1
    new.out = [o + 1 for o in new.out]
    users = _nodes_using(ind, mid)
    new_id = _replace_module(ind, mid, new, renumber=True)
    rand = rnd.random
    for j in users:
        lbl = int(rand() * (n_in + j))
        ind.conn[j].append(lbl)
        ind.cout[j].append(int(rand() * n_outputs_of(ind, lbl, n_in)))
    return new_id


def remove_input(ind: Individual, mid: int, rnd: random.Random) -> bool:
    """Drop one module input (section 7, operator 4). Bound: min 2.

    `[our choice]` The paper does not say what happens to body genes that referenced
    the dropped input -- they cannot simply be deleted, since the body node still
    needs two inputs. They are re-pointed at a surviving module input, chosen at
    random, which is the smallest repair that keeps the module valid and matches how
    the genotype operator repairs an arity change.
    """
    mod = ind.modules[mid]
    if mod.n_in <= 2:
        return False
    idx = int(rnd.random() * mod.n_in)
    new = mod.copy()
    new.n_in -= 1
    survivors = [x for x in range(mod.n_in) if x != idx]
    for t in range(len(new.conn)):
        lbl = new.conn[t]
        if lbl == idx:
            lbl = survivors[int(rnd.random() * len(survivors))]
        new.conn[t] = (lbl - 1) if lbl > idx else lbl
    new.out = [o - 1 for o in new.out]
    users = _nodes_using(ind, mid)
    _replace_module(ind, mid, new, renumber=False)
    for j in users:
        del ind.conn[j][idx]
        del ind.cout[j][idx]
    return True


def add_output(ind: Individual, mid: int, rnd: random.Random) -> int | None:
    """Give the module one more output (section 7, operator 3). Bound: max `n`.

    Appending keeps every existing `cout` in the outer genotype valid, so no node
    referencing the module needs repair; the module is renumbered, so the NEW id is
    returned (None if the bound blocked the operator).
    """
    mod = ind.modules[mid]
    if mod.n_out >= mod.n_nodes:
        return None
    new = mod.copy()
    new.out.append(new.n_in + int(rnd.random() * new.n_nodes))
    return _replace_module(ind, mid, new, renumber=True)


def remove_output(ind: Individual, mid: int, rnd: random.Random, n_in: int) -> bool:
    """Drop one module output (section 7, operator 5). Bound: min 1.

    ⚠️ The paper's prose for this operator says the *input* count is decremented --
    an evident typo (PAPER_SPEC section 7); implemented as the symmetric reading.

    `[our choice]` References carrying the dropped output index are re-pointed at a
    surviving output chosen at random; higher indices shift down. As with
    `remove_input`, the paper is silent and the alternative -- leaving a dangling
    reference -- is not representable.
    """
    mod = ind.modules[mid]
    if mod.n_out <= 1:
        return False
    idx = int(rnd.random() * mod.n_out)
    new = mod.copy()
    del new.out[idx]
    _replace_module(ind, mid, new, renumber=False)

    users = set(_nodes_using(ind, mid))
    n_left = len(new.out)

    def fix(lbl: int, c: int) -> int:
        if lbl < n_in or (lbl - n_in) not in users:
            return c
        if c == idx:
            return int(rnd.random() * n_left)
        return c - 1 if c > idx else c

    for j in range(len(ind.func)):
        for t in range(len(ind.conn[j])):
            ind.cout[j][t] = fix(ind.conn[j][t], ind.cout[j][t])
    for o in range(len(ind.ogene)):
        ind.ocout[o] = fix(ind.ogene[o], ind.ocout[o])
    return True


def module_name(mid: int, n_prim: int) -> str:
    """Display name for a module: `M1`, `M2`, ... in order of creation.

    Ids come from a counter that is never rewound (`next_id`), so a name is bound to
    one **act of creation**, not to a function: a module that dies and an identical
    one that is compressed later get different names, and `add_input`/`add_output`
    renumber (section 7), so a module whose interface changes becomes a new one. That
    is what makes a lifetime measurable -- `M7` appearing in 300 consecutive log rows
    is one module surviving, never a coincidence of shape.

    Uniqueness holds *along a lineage*, which is the only place it is read: two
    offspring of the same parent both draw the parent's `next_id`, so the same name
    can name different modules in siblings -- but at most one sibling is promoted.
    """
    return f"M{mid - n_prim + 1}"


def active_nodes(ind: Individual, n_in: int) -> list[int]:
    """Node indices reachable backwards from the program outputs, topologically ordered.

    The same walk `evaluate` opens with, factored out for instrumentation: a gate
    histogram has to count what the circuit *uses*, and a CGP/ECGP genotype is mostly
    inactive by design -- that inactive bulk is the reservoir neutral drift moves
    through. Kept as its own function rather than returned from `evaluate` so the hot
    path allocates nothing extra.
    """
    seen = bytearray(len(ind.func))
    active: list[int] = []
    stack = [l - n_in for l in ind.ogene if l >= n_in]
    while stack:
        j = stack.pop()
        if seen[j]:
            continue
        seen[j] = 1
        active.append(j)
        for lbl in ind.conn[j]:
            if lbl >= n_in:
                stack.append(lbl - n_in)
    active.sort()
    return active


def prune_modules(ind: Individual) -> None:
    """Delete every module the individual does not use (section 5).

    The paper's second-level "copy-or-die": when the fittest individual is promoted
    the list is pruned to exactly the modules present in it. This is the stated reason
    the list cannot grow without bound, so it is not an optimisation -- dropping it
    changes the algorithm.
    """
    live = {ind.func[j] for j in range(len(ind.func)) if ind.ntype[j] != 0}
    for mid in [m for m in ind.modules if m not in live]:
        del ind.modules[mid]


# ---------------------------------------------------------------------------
# the full offspring operator
# ---------------------------------------------------------------------------

def mutate(ind: Individual, rnd: random.Random, n_in: int, n_prim: int,
           p: Params) -> Individual:
    """One offspring: compress, expand, the five module operators, point mutation.

    `[our choice]` The paper lists the operators and their probabilities but never
    fixes an order. This one goes structure-first (compress/expand), then module
    bodies, then the genotype -- so the point mutation at the end always sees the
    module list the offspring will actually be evaluated with, and a module created
    this generation is immediately available for re-use.

    Table II's rates are per *application of the operator*, i.e. per offspring; the
    five module operators are drawn per module, since that is the object they act on.
    """
    out = ind.copy()
    rand = rnd.random
    if rand() < p.compress:
        compress(out, rnd, p.max_module_size, n_in)
    if rand() < p.expand:
        expand(out, rnd, n_in)
    for mid in list(out.modules):
        if rand() < p.module_point:
            module_point_mutate(out, mid, rnd, n_prim, p.mutation_rate)
        # add-input and add-output renumber, so the id has to be carried forward or
        # the operators after them would act on a module that no longer exists.
        if rand() < p.add_input:
            mid = add_input(out, mid, rnd, n_in) or mid
        if rand() < p.remove_input:
            remove_input(out, mid, rnd)
        if rand() < p.add_output:
            mid = add_output(out, mid, rnd) or mid
        if rand() < p.remove_output:
            remove_output(out, mid, rnd, n_in)
    # one pass for both the slot count and the slot table: `n_mutations` needs the
    # total, `point_mutate` needs the layout, and they are the same walk
    table = _slot_table(out)
    n_mut = max(1, int(round(p.mutation_rate * table[2])))
    point_mutate(out, rnd, n_mut, n_in, n_prim, table)
    return out


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def flatten(ind: Individual, n_in: int) -> cgp.Genotype:
    """Inline every module -> an ordinary CGP genotype computing the same function."""
    return flatten_with_origin(ind, n_in, 0)[0]


def flatten_with_origin(ind: Individual, n_in: int,
                        n_prim: int) -> tuple[cgp.Genotype, list[str]]:
    """`flatten`, plus which module instance each resulting node came from.

    One pass suffices: modules contain primitives only (no nesting). The result feeds
    `cgp.phenotype` and `cgp.evaluate`, so ECGP runs report active-node counts and
    cone classes computed by exactly the same code as the CGP baseline.

    Flattening is what makes the two arms comparable and is also what ERASES modules
    from the picture -- a diagram of the flattened graph is a wall of ordinary gates
    with no sign of which came from where. The `origin` list is that lost information,
    one tag per node: `""` for a node written directly in the genotype, else
    `"M12#2"`, meaning "the second call site of module M12". Two nodes sharing a tag
    prefix are the same acquired function used at different places, which is exactly
    the reuse the experiment is about.
    """
    func: list[int] = []
    conn: list[int] = []
    origin: list[str] = []
    seen_calls: dict[int, int] = {}
    # old (label, output) -> new label, resolved left to right
    remap: dict[tuple[int, int], int] = {(l, 0): l for l in range(n_in)}

    for j in range(len(ind.func)):
        lbl = n_in + j
        if ind.ntype[j] == 0:
            src = [remap[(ind.conn[j][t], ind.cout[j][t])] for t in range(2)]
            remap[(lbl, 0)] = n_in + len(func)
            func.append(ind.func[j])
            conn.extend(src)
            origin.append("")
            continue
        mid = ind.func[j]
        mod = ind.modules[mid]
        seen_calls[mid] = seen_calls.get(mid, 0) + 1
        tag = f"{module_name(mid, n_prim)}#{seen_calls[mid]}"
        args = [remap[(ind.conn[j][t], ind.cout[j][t])] for t in range(mod.n_in)]
        body_lbl: list[int] = []                  # module body node -> new label
        for b in range(mod.n_nodes):
            src = []
            for t in range(2):
                ml = mod.conn[2 * b + t]
                src.append(args[ml] if ml < mod.n_in else body_lbl[ml - mod.n_in])
            body_lbl.append(n_in + len(func))
            func.append(mod.func[b])
            conn.extend(src)
            origin.append(tag)
        for o, ml in enumerate(mod.out):
            remap[(lbl, o)] = body_lbl[ml - mod.n_in]

    n = len(func)
    return cgp.Genotype(func=func, ntype=[0] * n, conn=conn, cout=[0] * (2 * n),
                        ogene=[remap[(l, c)] for l, c in zip(ind.ogene, ind.ocout)],
                        ocout=[0] * len(ind.ogene), arity=2), origin


def _apply(op: int, x: int, y: int, mask: int) -> int:
    """One gate over truth-table masks. Opcodes as in `gates.py`."""
    if op == 0:
        return x & y
    if op == 2:
        return ~(x & y) & mask
    if op == 1:
        return x | y
    if op == 3:
        return ~(x | y) & mask
    if op == 4:
        return x ^ y
    if op == 5:
        return ~(x ^ y) & mask
    if op == 6:
        return ~x & mask
    if op == 7:
        return 0
    return mask


def _run_module(mod: Module, args: list[int], ops: Sequence[int], mask: int) -> list[int]:
    """Evaluate a module body on `args`, returning one mask per module output.

    Every body node is computed, not just the active ones: bodies are at most `ms`
    (5) nodes, so the backward walk would cost more than it saves.
    """
    v = args + [0] * mod.n_nodes
    b = mod.n_in
    for j in range(mod.n_nodes):
        v[b + j] = _apply(ops[mod.func[j]], v[mod.conn[2 * j]], v[mod.conn[2 * j + 1]],
                          mask)
    return [v[o] for o in mod.out]


def evaluate(ind: Individual, gates: Sequence[Gate], in_masks: Sequence[int],
             mask: int, n_in: int) -> list[int]:
    """Output truth-table masks, one per program output. Bit-parallel over patterns.

    Direct evaluation rather than `cgp.evaluate(flatten(...))`: flattening allocates
    the whole inlined genotype on every fitness call, which is the hot path. The two
    are asserted equal on random individuals in `test_ecgp.py`.

    A node's value is a LIST of masks -- one per output -- because a module node has
    several. Primitives produce a one-element list.
    """
    n = len(ind.func)
    conn, cout, ntype, func = ind.conn, ind.cout, ind.ntype, ind.func

    seen = bytearray(n)
    active: list[int] = []
    stack = [l - n_in for l in ind.ogene if l >= n_in]
    while stack:
        j = stack.pop()
        if seen[j]:
            continue
        seen[j] = 1
        active.append(j)
        for lbl in conn[j]:
            if lbl >= n_in:
                stack.append(lbl - n_in)
    active.sort()                                  # == topological order

    ops = [g.op for g in gates]
    # The `[[0]] * n` tail shares one list object across every inactive node. That is
    # safe because an active node is only ever ASSIGNED a fresh list, and no active
    # node can read an inactive one (the backward walk above marked everything
    # reachable), so the shared object is never written to.
    vals: list[list[int]] = [[m] for m in in_masks] + [[0]] * n
    for j in active:
        cj, oj = conn[j], cout[j]
        if ntype[j] == 0:
            vals[n_in + j] = [_apply(ops[func[j]], vals[cj[0]][oj[0]],
                                     vals[cj[1]][oj[1]], mask)]
        else:
            mod = ind.modules[func[j]]
            args = [vals[cj[t]][oj[t]] for t in range(mod.n_in)]
            vals[n_in + j] = _run_module(mod, args, ops, mask)
    return [vals[l][c] for l, c in zip(ind.ogene, ind.ocout)]


# ---------------------------------------------------------------------------
# inspection -- what is a module actually COMPUTING? (read by the run log)
# ---------------------------------------------------------------------------
#
# A module is a closed function of its own inputs: its body references nothing but
# module inputs and earlier body nodes, and every call site supplies its own
# arguments. So its truth table over the 2**n_in input patterns is a COMPLETE
# description of it, independent of where it is plugged in -- which makes "what does
# M73 do?" and "are M73 and M91 the same thing?" exactly answerable, in a way the
# equivalent question about a sub-network of weights never is.
#
# Two modules are called the same function up to (a) inputs the outputs ignore and
# (b) a relabelling of the inputs. Both are wanted: `compress` grabs a random window,
# so it routinely encapsulates a dead input, and it has no reason to present the
# arguments of two rediscoveries of XOR in the same order.

MAX_CANON_PERMS = 720     # above this the permutation search is skipped (see below)

# label -> (n_inputs, predicate over the input bits). Only functions worth
# recognising by name; anything else is reported by signature alone.
_KNOWN_FUNCS = [
    ("0",     0, lambda b: 0),
    ("1",     0, lambda b: 1),
    ("WIRE",  1, lambda b: b[0]),
    ("NOT",   1, lambda b: 1 - b[0]),
    ("AND",   2, lambda b: b[0] & b[1]),
    ("OR",    2, lambda b: b[0] | b[1]),
    ("XOR",   2, lambda b: b[0] ^ b[1]),
    ("NAND",  2, lambda b: 1 - (b[0] & b[1])),
    ("NOR",   2, lambda b: 1 - (b[0] | b[1])),
    ("XNOR",  2, lambda b: 1 - (b[0] ^ b[1])),
    ("A&~B",  2, lambda b: b[0] & (1 - b[1])),
    ("A|~B",  2, lambda b: b[0] | (1 - b[1])),
    ("AND3",  3, lambda b: b[0] & b[1] & b[2]),
    ("OR3",   3, lambda b: b[0] | b[1] | b[2]),
    ("XOR3",  3, lambda b: b[0] ^ b[1] ^ b[2]),
    ("MAJ3",  3, lambda b: int(b[0] + b[1] + b[2] >= 2)),
]

_SIG_CACHE: dict = {}     # body content -> (label, signature, expression)


def module_table(mod: Module, ops: Sequence[int]) -> list[int]:
    """One truth-table mask per module output, over all `2**n_in` input patterns.

    Bit `p` of output `o` is what output `o` returns when the module's inputs are set
    to the bits of `p`. Same bit-parallel trick the task evaluation uses, so the whole
    table costs one pass over the body.
    """
    n = mod.n_in
    npat = 1 << n
    args = [sum(1 << p for p in range(npat) if (p >> i) & 1) for i in range(n)]
    return _run_module(mod, args, ops, (1 << npat) - 1)


def _support(tables: Sequence[int], n: int) -> list[int]:
    """The inputs the outputs actually depend on -- flipping any other changes nothing."""
    dep = []
    for i in range(n):
        bit = 1 << i
        if any(((t >> p) & 1) != ((t >> (p ^ bit)) & 1)
               for t in tables for p in range(1 << n)):
            dep.append(i)
    return dep


def _restrict(t: int, sup: Sequence[int], perm: Sequence[int]) -> int:
    """Re-index a truth table onto `sup`, in the order given by `perm`."""
    out = 0
    for p in range(1 << len(sup)):
        q = 0
        for k, src in enumerate(perm):
            if (p >> k) & 1:
                q |= 1 << sup[src]
        if (t >> q) & 1:
            out |= 1 << p
    return out


def _influence(tables: Sequence[int], n: int, i: int) -> tuple[int, ...]:
    """Per output, on how many input patterns does flipping input `i` change it?

    A permutation-invariant fingerprint of one input's role, and the reason the search
    below is cheap: two inputs with different influence can never be exchanged, so only
    inputs that tie need permuting against each other. For an asymmetric function that
    leaves exactly one candidate ordering instead of s!.
    """
    bit = 1 << i
    return tuple(sum(1 for p in range(1 << n)
                     if ((t >> p) & 1) != ((t >> (p ^ bit)) & 1))
                 for t in tables)


def _canonical(tables: Sequence[int], n: int) -> tuple[int, tuple[int, ...], bool]:
    """(support size, canonical tables, approximate?) -- dummies dropped, inputs relabelled.

    The relabelling is the lexicographically smallest one. The SAME permutation is
    applied to every output, so a multi-output module is canonicalised jointly rather
    than output by output.

    Candidate orderings are restricted to permutations WITHIN groups of equal-influence
    inputs, which is exact (a permutation across groups cannot map the function onto
    itself) and usually leaves one candidate. If even that exceeds `MAX_CANON_PERMS`
    the influence order alone is used and the result is flagged approximate: such a
    signature still compares exactly -- equal signatures still mean the same function
    -- it only stops seeing through a relabelling of tied inputs.
    """
    sup = _support(tables, n)
    s = len(sup)
    prof = [_influence(tables, n, i) for i in sup]
    order = sorted(range(s), key=lambda k: (prof[k], k))

    groups, k = [], 0
    while k < s:                                       # runs of equal influence
        j = k
        while j + 1 < s and prof[order[j + 1]] == prof[order[k]]:
            j += 1
        groups.append(order[k:j + 1])
        k = j + 1

    total = 1
    for g in groups:
        total *= math.factorial(len(g))
    if total > MAX_CANON_PERMS:
        return s, tuple(_restrict(t, sup, tuple(order)) for t in tables), True

    best = None
    for combo in itertools.product(*(itertools.permutations(g) for g in groups)):
        perm = tuple(x for grp in combo for x in grp)
        cand = tuple(_restrict(t, sup, perm) for t in tables)
        if best is None or cand < best:
            best = cand
    return s, best, False


def _signature(tables: Sequence[int], n: int) -> tuple[int, tuple[int, ...], str]:
    """(support size, canonical tables, printable key). Equal key <=> same function."""
    s, canon, approx = _canonical(tables, n)
    body = "-".join(f"{t:x}" for t in canon)
    if len(body) > 24:                          # keep the CSV cell readable
        body = hashlib.blake2b(body.encode(), digest_size=6).hexdigest()
    return s, canon, f"{'~' if approx else ''}{s}:{body}"


def _known_table(n_bits: int, fn) -> int:
    t = 0
    for p in range(1 << n_bits):
        if fn([(p >> i) & 1 for i in range(n_bits)]):
            t |= 1 << p
    return t


_KNOWN_BY_SIG = {_signature([_known_table(k, f)], k)[2]: name
                 for name, k, f in _KNOWN_FUNCS}


def module_expr(mod: Module, gates: Sequence[Gate]) -> str:
    """The body as readable algebra, one s-expression per output: `(NAND (OR i0 i1) i2)`.

    Inputs are the module's FORMAL parameters `i0, i1, ...`, not wires in the outer
    genotype -- a module does not know where its arguments come from. Only the nodes
    feeding an output appear; a body node nothing reads is left out, and the paper
    warns those are common ("the majority of the modules also contained some inactive
    nodes").
    """
    def node(lbl: int) -> str:
        if lbl < mod.n_in:
            return f"i{lbl}"
        j = lbl - mod.n_in
        return (f"({gates[mod.func[j]].name.upper()} "
                f"{node(mod.conn[2 * j])} {node(mod.conn[2 * j + 1])})")
    return " ; ".join(node(o) for o in mod.out)


def module_info(mod: Module, gates: Sequence[Gate]) -> tuple[str, str, str]:
    """(label, signature, expression) for a module.

    `label` names the function when it is one worth naming ("XOR"), per output, joined
    by `;`; empty when unrecognised. `signature` is the canonical key -- two modules
    with the same signature compute the same function up to dummy inputs and input
    relabelling, whatever their bodies look like. `expression` is the structure.

    Memoised on body CONTENT rather than module id, because `module_point_mutate`
    rewrites a body without renumbering: the id is stable across a change of meaning,
    the content is not.
    """
    ops = tuple(g.op for g in gates)
    key = (mod.n_in, tuple(mod.func), tuple(mod.conn), tuple(mod.out), ops)
    hit = _SIG_CACHE.get(key)
    if hit is not None:
        return hit

    tables = module_table(mod, ops)
    _, _, sig = _signature(tables, mod.n_in)
    # per-output labels: each output is a function in its own right, and it is far
    # more informative to read "XOR ; AND" than one name for the pair
    labels = [_KNOWN_BY_SIG.get(_signature([t], mod.n_in)[2], "") for t in tables]
    info = (";".join(labels) if any(labels) else "",
            sig, module_expr(mod, gates))
    if len(_SIG_CACHE) > 8192:                  # unbounded module churn, bounded memory
        _SIG_CACHE.clear()
    _SIG_CACHE[key] = info
    return info


def primitive_module(func: int) -> Module:
    """A primitive dressed as a one-node module, so `module_info` names it too.

    Lets the run log describe a plain AND gate and an acquired module in the same
    columns -- which is the point of the CGP-vs-ECGP comparison.
    """
    return Module(mid=-1, n_in=2, func=[func], conn=[0, 1], out=[2])


def fitness(ind: Individual, gates: Sequence[Gate], in_masks: Sequence[int],
            target: int, mask: int, n_in: int, kind: str = "raw") -> tuple[float, int]:
    """(selection score, hits) -- same contract and same scores as `cgp.fitness`."""
    out = evaluate(ind, gates, in_masks, mask, n_in)[0]
    h = cgp.hits(out, target, mask)
    if kind == "balanced":
        return cgp.balanced_score(out, target, mask), h
    return float(h), h


# ---------------------------------------------------------------------------
# invariants (used by the tests, and cheap enough to call from a debug run)
# ---------------------------------------------------------------------------

def validate(ind: Individual, n_in: int, n_prim: int, ms: int) -> None:
    """Raise AssertionError on any structural violation of sections 1, 4 and 6."""
    n = len(ind.func)
    assert len(ind.ntype) == n and len(ind.conn) == n and len(ind.cout) == n
    for j in range(n):
        t = ind.ntype[j]
        assert t in (0, 1, 2), f"node {j}: bad node type {t}"
        if t == 0:
            assert ind.func[j] < n_prim, f"node {j}: type 0 with a module id"
        else:
            assert ind.func[j] in ind.modules, f"node {j}: unknown module {ind.func[j]}"
        assert len(ind.conn[j]) == arity_of(ind, j), f"node {j}: wrong arity"
        assert len(ind.cout[j]) == len(ind.conn[j]), f"node {j}: conn/cout mismatch"
        for k, lbl in enumerate(ind.conn[j]):
            assert 0 <= lbl < n_in + j, f"node {j} gene {k}: forward reference {lbl}"
            assert 0 <= ind.cout[j][k] < n_outputs_of(ind, lbl, n_in), \
                f"node {j} gene {k}: output index out of range"
    for o, lbl in enumerate(ind.ogene):
        assert 0 <= lbl < n_in + n, f"output {o}: label {lbl} out of range"
        assert 0 <= ind.ocout[o] < n_outputs_of(ind, lbl, n_in), \
            f"output {o}: output index out of range"
    for mid, mod in ind.modules.items():
        assert mod.mid == mid
        assert 2 <= mod.n_nodes <= ms, f"module {mid}: {mod.n_nodes} nodes"
        assert 1 <= mod.n_out <= mod.n_nodes, f"module {mid}: {mod.n_out} outputs"
        assert 2 <= mod.n_in <= 2 * mod.n_nodes, f"module {mid}: {mod.n_in} inputs"
        assert len(mod.conn) == 2 * mod.n_nodes
        for j in range(mod.n_nodes):
            assert mod.func[j] < n_prim, f"module {mid}: nesting (node {j})"
            for t in range(2):
                lbl = mod.conn[2 * j + t]
                assert 0 <= lbl < mod.n_in + j, \
                    f"module {mid} node {j}: forward reference {lbl}"
        for o, lbl in enumerate(mod.out):
            assert mod.n_in <= lbl < mod.n_in + mod.n_nodes, \
                f"module {mid} output {o}: connects to a module input"
