"""necgp -- ECGP extended so a module can contain modules (experiment 6's own idea).

Forked from `experiments/experiment_4/ecgp.py` (2026-08-19). Everything Walker &
Miller's paper states (`experiment_4/PAPER_SPEC.md`) still holds here EXCEPT
section 6's nesting ban, which this file deliberately relaxes. That relaxation is
NOT in any paper -- see `../README.md` ("Where this goes next") and `../RESULTS.md`
for the reasoning; every place that reasoning bears on a specific line is tagged
`EXTENDED` below.

THE ONE-SENTENCE VERSION. `compress` may now absorb a window containing type I/II
nodes (so the new module nests whatever they call), gated by a probability that
DECAYS with the resulting nesting depth (`Params.nest_decay ** (depth - 1)`) --
depth-1 modules form exactly as in plain ECGP, and every extra level of nesting
costs one more decay factor, the parsimony pressure a real mutation-rate penalty
would apply. `expand` is unchanged in spirit (dissolve a type I node back into its
stored body) but now that a body can itself contain type I/II nodes, one `expand`
undoes exactly one level of nesting, not all of them -- see `expand`'s docstring
for why that reading was chosen over a full recursive unroll.

WHAT HAD TO ACTUALLY CHANGE, STRUCTURALLY. Plain ECGP's `Module` stores its body as
a FLAT, fixed-arity-2 `conn` list with no `cout`/`ocout` at all, because a body
was guaranteed to hold primitives only (arity always 2, always one output). Once a
body node can be a call to another module, that assumption breaks in two places at
once: the callee's arity may be anything up to `2*ms`, and it may have more than
one output that a later body node needs to pick between. So `Module` here carries
the same shape `Individual` already does -- `ntype` (0/1/2, same three-way scheme),
`conn`/`cout` as list-of-lists (variable arity), and `out`/`ocout` (paired, like
`ogene`/`ocout`) -- plus a stored `depth`. Every function that used to assume "body
= primitives, flat, single-output" is rewritten below; every function that never
made that assumption (`point_mutate`, `_fit_arity`, `_clamp_refs_to`,
`random_individual`, `n_gene_slots`, `_slot_table`, `active_nodes`) is carried over
UNCHANGED, because top-level nodes were already variable-arity/multi-output.

`[v1 scope]` The four module-INTERFACE operators (`add_input`/`remove_input`/
`add_output`/`remove_output`) repair every node that calls the module they resize,
but that repair walk (`_nodes_using`) only looks at the top-level genotype. Doing
the same repair for a call site sitting inside ANOTHER module's body is
representable (identical mechanics, one level deeper) but not implemented here;
instead these four operators simply refuse to act on a module that is nested-into
by some other module (`_is_nested_into`), exactly like they already refuse past
their own size bounds. `compress`/`expand`/`module_point_mutate` are unaffected --
none of them repair external callers, so nesting itself (creating and undoing it)
works at any depth.

The introspection/canonicalisation section plain ECGP has at the bottom of its file
(module_table, module_expr, the signature machinery) is DROPPED here -- not needed
for a smoke run and it would need its own recursive generalisation; add it back if
a real run needs "what is M73 actually computing" for a nested module.
"""

from __future__ import annotations

import bisect
import itertools
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
    """A module: header + body, EXTENDED for nesting (see module docstring).

    `ntype[b]` marks body node `b` as 0 (primitive), 1 (nested-module OWNER -- the
    node `compress` created when it folded a deeper module in), or 2 (nested-module
    REUSE) -- the same three-way scheme `Individual.ntype` uses at the top level,
    carried through unchanged when a node gets absorbed into a new, deeper module.
    That is what lets `expand` correctly restore owner/reuse status after un-nesting
    one level.

    `conn`/`cout` are list-of-lists (a nested body node's arity is its callee's
    `n_in`, not always 2). `out`/`ocout` name a module's own outputs the same way
    `ogene`/`ocout` do at the top level, for the same reason: an output may need to
    read one specific output of a multi-output nested call.

    `depth` is 1 for an all-primitive body (plain ECGP's only case), else
    1 + max(depth of every nested module folded into this one). Stored, not
    recomputed, since `compress`'s decay roll reads it on every attempt.
    """
    mid: int
    n_in: int
    func: list[int]           # (n_nodes,) primitive id OR module id
    ntype: list[int]          # (n_nodes,) 0 primitive / 1 nested owner / 2 nested reuse
    conn: list[list[int]]     # per body node: source label per input gene
    cout: list[list[int]]     # per body node: which output of that source
    out: list[int]            # (n_out,) body-node labels
    ocout: list[int]          # (n_out,) which output of that body node
    depth: int = 1

    def copy(self) -> "Module":
        return Module(self.mid, self.n_in, self.func[:], self.ntype[:],
                      list(map(list, self.conn)), list(map(list, self.cout)),
                      self.out[:], self.ocout[:], self.depth)

    @property
    def n_nodes(self) -> int:
        return len(self.func)

    @property
    def n_out(self) -> int:
        return len(self.out)


@dataclass
class Individual:
    """One ECGP genotype plus the module list it owns. Unchanged from plain ECGP."""
    func: list[int]           # (n_nodes,)  primitive id (type 0) or module id (types I/II)
    ntype: list[int]          # (n_nodes,)  0 = primitive, 1 = type I (compress), 2 = type II (reuse)
    conn: list[list[int]]     # per node: source label per input gene
    cout: list[list[int]]     # per node: which output of that source
    ogene: list[int]          # (n_outputs,) source label per program output
    ocout: list[int]          # (n_outputs,) which output of it
    modules: dict[int, Module]
    next_id: int               # next unused module id

    def copy(self) -> "Individual":
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
    """Operator probabilities. Defaults are Table II `[verbatim]` except `nest_decay`,
    which has no paper to be verbatim to -- see module docstring and `../RESULTS.md`.
    """
    compress: float = 0.1
    expand: float = 0.2
    module_point: float = 0.04
    add_input: float = 0.01
    remove_input: float = 0.02
    add_output: float = 0.01
    remove_output: float = 0.02
    max_module_size: int = 5          # `ms`, PAPER_SPEC section 9 `[our choice]`
    mutation_rate: float = 0.03       # Table II
    nest_decay: float = 0.5           # EXTENDED, `[our choice]` -- see module docstring


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


def _mod_n_outs(ind: Individual, mod: Module, lbl: int) -> int:
    """How many outputs the thing at body-relative label `lbl` has.

    EXTENDED. A module input (`lbl < mod.n_in`) is always a resolved scalar (the
    call site already picked a definite value before passing it in), so it is
    always 1 regardless of nesting; only a body-node label's count can depend on
    whether that body node is a primitive or a nested call.
    """
    if lbl < mod.n_in:
        return 1
    b = lbl - mod.n_in
    if mod.ntype[b] == 0:
        return 1
    return ind.modules[mod.func[b]].n_out


def n_gene_slots(ind: Individual) -> int:
    """Mutable gene slots: one function gene + its input genes per node, plus outputs."""
    return len(ind.func) + sum(map(len, ind.conn)) + len(ind.ogene)


def _slot_table(ind: Individual) -> tuple[list[int], int, int]:
    """(per-node slot start, node-slot count, total slots) -- the mutable-gene layout."""
    acc = list(itertools.accumulate((1 + n for n in map(len, ind.conn)), initial=0))
    n_node_slots = acc.pop()
    return acc, n_node_slots, n_node_slots + len(ind.ogene)


def _module_slot_table(mod: Module) -> tuple[list[int], int, int]:
    """`_slot_table`'s idea, over a module's own (now variable-arity) body. EXTENDED."""
    acc = list(itertools.accumulate((1 + len(c) for c in mod.conn), initial=0))
    n_node_slots = acc.pop()
    return acc, n_node_slots, n_node_slots + len(mod.out)


def n_mutations(ind: Individual, rate: float) -> int:
    """Gene slots mutated per application; at least 1."""
    return max(1, int(round(rate * n_gene_slots(ind))))


def random_individual(rnd: random.Random, n_nodes: int, n_in: int, n_outputs: int,
                      n_prim: int) -> Individual:
    """A random all-primitive individual with an EMPTY module list. Unchanged."""
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
# genotype point mutation (section 6) -- UNCHANGED, top-level nodes were already
# variable-arity/multi-output, so nothing here assumed the thing this file relaxes.
# ---------------------------------------------------------------------------

def point_mutate(ind: Individual, rnd: random.Random, n_mut: int, n_in: int,
                 n_prim: int, table: tuple[list[int], int, int] | None = None) -> None:
    """CGP's point mutation, extended to modules. Mutates `ind` in place.

    * a function gene may be set to any primitive **or any module in the list**;
    * doing so from type 0 makes the node **type II** (a reuse) -- the only route
      by which a module replicates at the TOP level;
    * the function gene of a type I node is immune; only `expand` removes one;
    * on a type change the node keeps as many of its original inputs as it needs
      and randomly generates any extra;
    * both integers of a reference are mutated together.
    """
    n = len(ind.func)
    starts, n_node_slots, total = table if table is not None else _slot_table(ind)

    choices = list(range(n_prim)) + list(ind.modules)      # primitives + module list
    rand = rnd.random
    ntype, func, modules = ind.ntype, ind.func, ind.modules

    def n_outs(lbl: int) -> int:
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
    """Repair references into node `j` after its output count shrank. Unchanged."""
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
    """Resize node `j`'s input genes to its (possibly new) arity. Unchanged."""
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
# compress / expand (section 6) -- THE core of the extension
# ---------------------------------------------------------------------------

def compress(ind: Individual, rnd: random.Random, ms: int, n_in: int,
             nest_decay: float = 1.0) -> bool:
    """Encapsulate a random run of nodes into a new module. Returns success.

    EXTENDED. Plain ECGP: "pick two random points, respecting the module size
    limits; encapsulate all type 0 nodes between them, abort if any type I/II node
    lies between the points" -- that abort is what forbids nesting. Here the abort
    is replaced by a DEPTH-GATED ACCEPT: a window MAY contain type I/II nodes (so
    the new module nests whatever they call), but the deeper the resulting module
    would be, the less likely the roll succeeds -- `nest_decay ** (depth - 1)`.
    Depth-1 modules (plain ECGP's only case) form at the unmodified base rate;
    every extra level of nesting costs one more `nest_decay` factor.
    `nest_decay=1.0` recovers plain ECGP's unconditional accept.

    `[our choice]` window drawing unchanged from plain ECGP: uniform legal length
    (2..ms) at a uniform position, no rejection loop.
    """
    n = len(ind.func)
    hi_len = min(ms, n)
    if hi_len < 2:
        return False
    ln = 2 + int(rnd.random() * (hi_len - 1))
    i = int(rnd.random() * (n - ln + 1))
    k = i + ln - 1

    nested = [j for j in range(i, k + 1) if ind.ntype[j] != 0]
    if nested:
        depth = 1 + max(ind.modules[ind.func[j]].depth for j in nested)
        if rnd.random() >= nest_decay ** (depth - 1):
            return False                 # decay roll failed -- no retry, same as the old abort
    else:
        depth = 1

    lo_lbl, hi_lbl = n_in + i, n_in + k

    # --- body: each absorbed node keeps its own arity (2 for a primitive, the
    # callee's n_in for a nested call) instead of the fixed 2 plain ECGP assumed.
    body_func = [ind.func[j] for j in range(i, k + 1)]
    body_ntype = [ind.ntype[j] for j in range(i, k + 1)]
    ext_lbl: list[int] = []
    ext_cout: list[int] = []
    slots: list[tuple[bool, int, int]] = []       # (is_internal, index, cout-if-internal)
    for j in range(i, k + 1):
        for t in range(arity_of(ind, j)):
            lbl, c = ind.conn[j][t], ind.cout[j][t]
            if lo_lbl <= lbl <= hi_lbl:
                slots.append((True, lbl - lo_lbl, c))
            else:
                slots.append((False, len(ext_lbl), 0))
                ext_lbl.append(lbl)
                ext_cout.append(c)
    m_in = len(ext_lbl)
    if m_in > 2 * ln:
        # EXTENDED, `[our choice]`. Plain ECGP never needed this check: every
        # absorbed node was arity-2, so m_in <= 2*ln (the module's own n_in<=2n
        # bound) held automatically. A nested call absorbed here can carry up to
        # 2*ms inputs BY ITSELF, which can blow that bound on its own -- abort,
        # no retry, same as every other compress failure mode.
        return False
    flat_conn = [(m_in + idx) if internal else idx for internal, idx, _ in slots]
    flat_cout = [c for _, _, c in slots]

    body_conn: list[list[int]] = []
    body_cout: list[list[int]] = []
    pos = 0
    for j in range(i, k + 1):
        ar = arity_of(ind, j)
        body_conn.append(flat_conn[pos:pos + ar])
        body_cout.append(flat_cout[pos:pos + ar])
        pos += ar

    # --- the module's outputs: keyed by (label, cout) now, not label alone -- a
    # nested call's several outputs can each be read independently by later nodes.
    used: set[tuple[int, int]] = set()
    for j in range(k + 1, n):
        for lbl, c in zip(ind.conn[j], ind.cout[j]):
            if lo_lbl <= lbl <= hi_lbl:
                used.add((lbl - lo_lbl, c))
    for lbl, c in zip(ind.ogene, ind.ocout):
        if lo_lbl <= lbl <= hi_lbl:
            used.add((lbl - lo_lbl, c))
    refs = sorted(used) if used else [(ln - 1, 0)]
    if len(refs) > ln:
        # EXTENDED, `[our choice]` -- same reasoning as the m_in check above, for
        # the output bound (n_out <= n_nodes): a nested call's several outputs can
        # each be read by a different later node, which alone can ask for more
        # distinct outputs than there are body positions (ln) to hold them.
        return False
    out_index = {r: o for o, r in enumerate(refs)}

    mid = ind.next_id
    ind.next_id += 1
    ind.modules[mid] = Module(mid=mid, n_in=m_in, func=body_func, ntype=body_ntype,
                              conn=body_conn, cout=body_cout,
                              out=[m_in + r for r, _ in refs],
                              ocout=[c for _, c in refs], depth=depth)

    # --- splice: the run becomes ONE type I node, so everything after it shifts left.
    shift = ln - 1

    def remap(lbl: int, c: int) -> tuple[int, int]:
        if lbl < lo_lbl:
            return lbl, c
        if lbl <= hi_lbl:
            return lo_lbl, out_index[(lbl - lo_lbl, c)]
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

    Type I only -- type II immune, bounding growth.

    EXTENDED. A module's body can now itself contain type I/II nodes (see
    `compress`), and this preserves that -- `ntype` is spliced through unchanged
    rather than forced to 0, so a nested owner exposed by this expand is itself
    immediately expand-eligible next time. That makes one `expand` undo exactly ONE
    level of nesting, not a full recursive unroll to primitives: it is the local,
    literal inverse of the window `compress` folded (whatever that window
    contained), which keeps expand an O(module size) operation and keeps its
    "twice compress's rate, it's the destruction operator" balance meaningful --
    a full recursive unroll would make genotype-size swings from a single mutation
    much more volatile than the rate was tuned for. Reaching primitives from a
    depth-d module takes d successive expands, the mirror image of the d
    successive (decay-gated) compresses it took to build.
    """
    type1 = [j for j, t in enumerate(ind.ntype) if t == 1]
    if not type1:
        return False
    p = type1[int(rnd.random() * len(type1))]
    mod = ind.modules[ind.func[p]]
    m, base = mod.n_nodes, n_in + p
    shift = m - 1

    def from_module(lbl: int, c: int) -> tuple[int, int]:
        """A (label, cout) inside the module body -> the outer (label, cout)."""
        if lbl < mod.n_in:
            return ind.conn[p][lbl], ind.cout[p][lbl]      # the module's own input
        return base + (lbl - mod.n_in), c                  # an inlined body node

    body_conn: list[list[int]] = []
    body_cout: list[list[int]] = []
    for j in range(m):
        pairs = [from_module(mod.conn[j][t], mod.cout[j][t]) for t in range(len(mod.conn[j]))]
        body_conn.append([q[0] for q in pairs])
        body_cout.append([q[1] for q in pairs])

    def remap(lbl: int, c: int) -> tuple[int, int]:
        if lbl < base:
            return lbl, c
        if lbl == base:                                    # was a reference to the module
            return base + (mod.out[c] - mod.n_in), mod.ocout[c]
        return lbl + shift, c

    func = ind.func[:p] + mod.func[:] + ind.func[p + 1:]
    ntype = ind.ntype[:p] + mod.ntype[:] + ind.ntype[p + 1:]
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
    """Indices of the TOP-LEVEL nodes (type I or II) that represent module `mid`.

    Does not look inside other modules' bodies -- see `_is_nested_into`.
    """
    return [j for j in range(len(ind.func))
            if ind.ntype[j] != 0 and ind.func[j] == mid]


def _is_nested_into(ind: Individual, mid: int) -> bool:
    """True if some OTHER module's body calls module `mid`. EXTENDED, `[v1 scope]`.

    `add_input`/`remove_input`/`add_output`/`remove_output` repair every node that
    calls the module they resize, but that repair (`_nodes_using`) only looks at
    the top-level genotype. Repairing a call site sitting inside another module's
    body is representable (identical mechanics, one level deeper) but not
    implemented for v1; instead these four operators refuse to act on a module
    that is nested-into, exactly like they already refuse past their size bounds.
    `compress`/`expand`/`module_point_mutate` are unaffected -- none of them
    repair external callers.
    """
    for other_id, mod in ind.modules.items():
        if other_id == mid:
            continue
        for b in range(mod.n_nodes):
            if mod.ntype[b] != 0 and mod.func[b] == mid:
                return True
    return False


def _replace_module(ind: Individual, old: int, mod: Module, renumber: bool) -> int:
    """Install `mod` in place of module `old`, optionally under a fresh id. Unchanged."""
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

    Two restrictions, both from plain ECGP: no NEW type II nodes, and a module
    output may never point directly at a module input.

    EXTENDED, `[our choice, v1 scope]`: a body position that is ALREADY a nested
    call (owner or reuse, from `compress`) has its function gene left untouched
    here too -- the same immunity `point_mutate` gives a top-level type I node,
    plus (deliberately narrower than the top level) the same immunity for type II,
    so that `compress`/`expand` stay the ONLY channel that creates or destroys
    nesting. This operator can still rewire a nested call's ARGUMENTS and can still
    redraw any primitive body node -- just never a nested call's own function gene.
    """
    mod = ind.modules[mid].copy()
    n = mod.n_nodes
    starts, n_node_slots, total = _module_slot_table(mod)
    rand = rnd.random
    for s in cgp._draw_slots(rnd, total, max(1, int(round(rate * total)))):
        if s >= n_node_slots:                          # output gene: body nodes only
            o = s - n_node_slots
            lbl = mod.n_in + int(rand() * n)
            mod.out[o] = lbl
            mod.ocout[o] = int(rand() * _mod_n_outs(ind, mod, lbl))
            continue
        j = bisect.bisect_right(starts, s) - 1
        off = s - starts[j]
        if off > 0:                                    # an input gene
            k = off - 1
            lbl = int(rand() * (mod.n_in + j))
            mod.conn[j][k] = lbl
            mod.cout[j][k] = int(rand() * _mod_n_outs(ind, mod, lbl))
            continue
        if mod.ntype[j] != 0:                          # nested call: immune, spend the slot
            continue
        mod.func[j] = int(rand() * n_prim)              # primitives only
    _replace_module(ind, mid, mod, renumber=False)


def add_input(ind: Individual, mid: int, rnd: random.Random, n_in: int) -> int | None:
    """Give the module one more input (section 7, operator 2). Bound: max `2n`.

    `[v1 scope]` also refuses if the module is nested-into -- see `_is_nested_into`.
    """
    mod = ind.modules[mid]
    if mod.n_in >= 2 * mod.n_nodes or _is_nested_into(ind, mid):
        return None
    new = mod.copy()
    new.n_in += 1
    for j in range(new.n_nodes):
        for t in range(len(new.conn[j])):
            if new.conn[j][t] >= mod.n_in:
                new.conn[j][t] += 1
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

    `[v1 scope]` also refuses if the module is nested-into.
    """
    mod = ind.modules[mid]
    if mod.n_in <= 2 or _is_nested_into(ind, mid):
        return False
    idx = int(rnd.random() * mod.n_in)
    new = mod.copy()
    new.n_in -= 1
    survivors = [x for x in range(mod.n_in) if x != idx]
    for j in range(len(new.conn)):
        for t in range(len(new.conn[j])):
            lbl = new.conn[j][t]
            if lbl == idx:
                lbl = survivors[int(rnd.random() * len(survivors))]
            new.conn[j][t] = (lbl - 1) if lbl > idx else lbl
    new.out = [o - 1 for o in new.out]
    users = _nodes_using(ind, mid)
    _replace_module(ind, mid, new, renumber=False)
    for j in users:
        del ind.conn[j][idx]
        del ind.cout[j][idx]
    return True


def add_output(ind: Individual, mid: int, rnd: random.Random) -> int | None:
    """Give the module one more output (section 7, operator 3). Bound: max `n`.

    `[v1 scope]` also refuses if the module is nested-into.
    """
    mod = ind.modules[mid]
    if mod.n_out >= mod.n_nodes or _is_nested_into(ind, mid):
        return None
    new = mod.copy()
    lbl = new.n_in + int(rnd.random() * new.n_nodes)
    new.out.append(lbl)
    new.ocout.append(int(rnd.random() * _mod_n_outs(ind, mod, lbl)))
    return _replace_module(ind, mid, new, renumber=True)


def remove_output(ind: Individual, mid: int, rnd: random.Random, n_in: int) -> bool:
    """Drop one module output (section 7, operator 5). Bound: min 1.

    `[v1 scope]` also refuses if the module is nested-into.
    """
    mod = ind.modules[mid]
    if mod.n_out <= 1 or _is_nested_into(ind, mid):
        return False
    idx = int(rnd.random() * mod.n_out)
    new = mod.copy()
    del new.out[idx]
    del new.ocout[idx]
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
    """Display name for a module: `M1`, `M2`, ... in order of creation. Unchanged."""
    return f"M{mid - n_prim + 1}"


def active_nodes(ind: Individual, n_in: int) -> list[int]:
    """Node indices reachable backwards from the program outputs. Unchanged --
    this only walks TOP-level nodes, same as plain ECGP; nesting doesn't change
    what "active at the top level" means."""
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
    """Delete every module the individual does not use, TRANSITIVELY (section 5).

    EXTENDED. A module can now be alive only because another module's body calls
    it, not just because a top-level node does -- the liveness walk follows nested
    calls, not just the top level, or a still-in-use nested module would be pruned
    out from under its parent the moment nothing at the TOP level references it
    directly.
    """
    live: set[int] = set()
    stack = [ind.func[j] for j in range(len(ind.func)) if ind.ntype[j] != 0]
    while stack:
        mid = stack.pop()
        if mid in live:
            continue
        live.add(mid)
        mod = ind.modules.get(mid)
        if mod is None:
            continue
        for b in range(mod.n_nodes):
            if mod.ntype[b] != 0:
                stack.append(mod.func[b])
    for mid in [m for m in ind.modules if m not in live]:
        del ind.modules[mid]


# ---------------------------------------------------------------------------
# the full offspring operator
# ---------------------------------------------------------------------------

def mutate(ind: Individual, rnd: random.Random, n_in: int, n_prim: int,
           p: Params) -> Individual:
    """One offspring: compress, expand, the five module operators, point mutation.

    Same order/semantics as plain ECGP; `compress` now also takes `p.nest_decay`.
    """
    out = ind.copy()
    rand = rnd.random
    if rand() < p.compress:
        compress(out, rnd, p.max_module_size, n_in, p.nest_decay)
    if rand() < p.expand:
        expand(out, rnd, n_in)
    for mid in list(out.modules):
        if rand() < p.module_point:
            module_point_mutate(out, mid, rnd, n_prim, p.mutation_rate)
        if rand() < p.add_input:
            mid = add_input(out, mid, rnd, n_in) or mid
        if rand() < p.remove_input:
            remove_input(out, mid, rnd)
        if rand() < p.add_output:
            mid = add_output(out, mid, rnd) or mid
        if rand() < p.remove_output:
            remove_output(out, mid, rnd, n_in)
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


def _flatten_module(ind: Individual, mod: Module, n_in: int, args: list[int],
                    tag: str, func: list[int], conn: list[int], origin: list[str],
                    seen: dict[int, int], n_prim: int) -> list[int]:
    """Inline `mod`'s body (recursively, for a nested call) into the flat arrays.

    EXTENDED -- plain ECGP's flatten was a single pass because a body could only
    ever hold primitives; this recurses whenever a body node is itself a nested
    call. `args` are already-resolved outer labels, one per module input (all
    single-output, since a module input is a formal parameter). Returns the
    resolved outer label for each of `mod`'s own outputs.
    """
    body_lbl: list[list[int]] = []          # body node -> one new label per its output

    def resolve(row_c: list[int], row_o: list[int], t: int) -> int:
        lbl = row_c[t]
        if lbl < mod.n_in:
            return args[lbl]
        return body_lbl[lbl - mod.n_in][row_o[t]]

    for b in range(mod.n_nodes):
        row_c, row_o = mod.conn[b], mod.cout[b]
        if mod.ntype[b] == 0:
            src = [resolve(row_c, row_o, t) for t in range(2)]
            new_lbl = n_in + len(func)
            func.append(mod.func[b])
            conn.extend(src)
            origin.append(tag)
            body_lbl.append([new_lbl])
        else:
            sub = ind.modules[mod.func[b]]
            sub_args = [resolve(row_c, row_o, t) for t in range(len(row_c))]
            seen[mod.func[b]] = seen.get(mod.func[b], 0) + 1
            sub_tag = f"{tag}>{module_name(mod.func[b], n_prim)}#{seen[mod.func[b]]}"
            body_lbl.append(_flatten_module(ind, sub, n_in, sub_args, sub_tag,
                                            func, conn, origin, seen, n_prim))
    return [body_lbl[lbl - mod.n_in][c] for lbl, c in zip(mod.out, mod.ocout)]


def flatten_with_origin(ind: Individual, n_in: int,
                        n_prim: int) -> tuple[cgp.Genotype, list[str]]:
    """`flatten`, plus which module instance (and nesting chain) each node came from.

    EXTENDED for nesting: recurses via `_flatten_module` instead of the one-pass
    loop plain ECGP uses. The origin tag chains through nesting: `"M12#2>M7#1"`
    means the first call to M7 inside the second call to M12.
    """
    func: list[int] = []
    conn: list[int] = []
    origin: list[str] = []
    seen_calls: dict[int, int] = {}
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
        outs = _flatten_module(ind, mod, n_in, args, tag, func, conn, origin,
                               seen_calls, n_prim)
        for o, new_lbl in enumerate(outs):
            remap[(lbl, o)] = new_lbl

    n = len(func)
    return cgp.Genotype(func=func, ntype=[0] * n, conn=conn, cout=[0] * (2 * n),
                        ogene=[remap[(l, c)] for l, c in zip(ind.ogene, ind.ocout)],
                        ocout=[0] * len(ind.ogene), arity=2), origin


def _apply(op: int, x: int, y: int, mask: int) -> int:
    """One gate over truth-table masks. Opcodes as in `gates.py`. Unchanged."""
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


def _run_module(mod: Module, modules: dict[int, Module], args: list[int],
                ops: Sequence[int], mask: int) -> list[int]:
    """Evaluate a module body on `args`, returning one mask per module output.

    EXTENDED -- recurses into `modules` whenever a body node is a nested call.
    `v[k]` is a LIST of masks, one per output, the same trick `evaluate` uses at
    the top level -- needed now that a body node can have several.
    """
    v: list[list[int]] = [[a] for a in args] + [[0] for _ in range(mod.n_nodes)]
    for j in range(mod.n_nodes):
        cj, oj = mod.conn[j], mod.cout[j]
        if mod.ntype[j] == 0:
            v[mod.n_in + j] = [_apply(ops[mod.func[j]], v[cj[0]][oj[0]],
                                      v[cj[1]][oj[1]], mask)]
        else:
            sub = modules[mod.func[j]]
            sub_args = [v[cj[t]][oj[t]] for t in range(sub.n_in)]
            v[mod.n_in + j] = _run_module(sub, modules, sub_args, ops, mask)
    return [v[lbl][c] for lbl, c in zip(mod.out, mod.ocout)]


def evaluate(ind: Individual, gates: Sequence[Gate], in_masks: Sequence[int],
             mask: int, n_in: int) -> list[int]:
    """Output truth-table masks, one per program output. Bit-parallel over patterns.

    Direct evaluation rather than `cgp.evaluate(flatten(...))` -- the hot path.
    The two are asserted equal in `test_necgp.py`. Only line changed from plain
    ECGP: `_run_module` now also takes `ind.modules`, to recurse.
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
    active.sort()

    ops = [g.op for g in gates]
    vals: list[list[int]] = [[m] for m in in_masks] + [[0]] * n
    for j in active:
        cj, oj = conn[j], cout[j]
        if ntype[j] == 0:
            vals[n_in + j] = [_apply(ops[func[j]], vals[cj[0]][oj[0]],
                                     vals[cj[1]][oj[1]], mask)]
        else:
            mod = ind.modules[func[j]]
            args = [vals[cj[t]][oj[t]] for t in range(mod.n_in)]
            vals[n_in + j] = _run_module(mod, ind.modules, args, ops, mask)
    return [vals[l][c] for l, c in zip(ind.ogene, ind.ocout)]


def fitness(ind: Individual, gates: Sequence[Gate], in_masks: Sequence[int],
            target: int, mask: int, n_in: int, kind: str = "raw") -> tuple[float, int]:
    """(selection score, hits) -- same contract as `cgp.fitness`. Unchanged."""
    out = evaluate(ind, gates, in_masks, mask, n_in)[0]
    h = cgp.hits(out, target, mask)
    if kind == "balanced":
        return cgp.balanced_score(out, target, mask), h
    return float(h), h


# ---------------------------------------------------------------------------
# invariants (used by the tests, and cheap enough to call from a debug run)
# ---------------------------------------------------------------------------

def validate(ind: Individual, n_in: int, n_prim: int, ms: int) -> None:
    """Raise AssertionError on any structural violation.

    EXTENDED for nesting: a module body node may now be type 1/2 (not just 0), so
    the body checks mirror the top-level ones instead of assuming primitives only,
    and a module's stored `depth` is checked against a fresh recursive count.
    """
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

    def check_module(mid: int, mod: Module, ancestors: frozenset[int]) -> int:
        assert mod.mid == mid
        assert 2 <= mod.n_nodes <= ms, f"module {mid}: {mod.n_nodes} nodes"
        assert 1 <= mod.n_out <= mod.n_nodes, f"module {mid}: {mod.n_out} outputs"
        assert 2 <= mod.n_in <= 2 * mod.n_nodes, f"module {mid}: {mod.n_in} inputs"
        assert len(mod.conn) == mod.n_nodes and len(mod.cout) == mod.n_nodes
        assert len(mod.ntype) == mod.n_nodes
        assert mid not in ancestors, f"module {mid}: cycle in nesting"
        ancestors = ancestors | {mid}
        depth = 1
        for j in range(mod.n_nodes):
            bt = mod.ntype[j]
            assert bt in (0, 1, 2), f"module {mid} node {j}: bad node type {bt}"
            if bt == 0:
                assert mod.func[j] < n_prim, \
                    f"module {mid} node {j}: type 0 with a module id"
                need = 2
            else:
                sub_id = mod.func[j]
                assert sub_id in ind.modules, \
                    f"module {mid} node {j}: unknown module {sub_id}"
                depth = max(depth, 1 + check_module(sub_id, ind.modules[sub_id], ancestors))
                need = ind.modules[sub_id].n_in
            assert len(mod.conn[j]) == need, f"module {mid} node {j}: wrong arity"
            assert len(mod.cout[j]) == need
            for t, lbl in enumerate(mod.conn[j]):
                assert 0 <= lbl < mod.n_in + j, \
                    f"module {mid} node {j}: forward reference {lbl}"
                lim = _mod_n_outs(ind, mod, lbl)
                assert 0 <= mod.cout[j][t] < lim, \
                    f"module {mid} node {j}: output index out of range"
        for o, lbl in enumerate(mod.out):
            assert mod.n_in <= lbl < mod.n_in + mod.n_nodes, \
                f"module {mid} output {o}: connects to a module input"
            assert 0 <= mod.ocout[o] < _mod_n_outs(ind, mod, lbl)
        assert mod.depth == depth, \
            f"module {mid}: stored depth {mod.depth} != computed {depth}"
        return depth

    for mid, mod in ind.modules.items():
        check_module(mid, mod, frozenset())
