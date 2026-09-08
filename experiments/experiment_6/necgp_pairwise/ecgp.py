"""necgp_pairwise -- EXPERIMENTAL, third attempt at a NECGP framework (2026-08-21).

NOT `necgp/`. NOT validated. Forked from `../necgp/ecgp.py` to test one specific
idea: `necgp/`'s decomposition of a solved circuit (`RESULTS.md`, 2026-08-21 entry)
found 37.5% of module CALLS in a solved circuit were to modules with zero internal
gate interaction ("fake" modules -- packaging around a gate, or several gates that
never chain, with no real computation reused). Root cause: `compress` grabs a
window of genome-ADJACENT nodes and accepts unconditionally, and genome position
carries no information about which nodes actually feed each other.

THE CHANGE. `compress` here always proposes exactly node i and node i+1 (never a
wider window), and REJECTS (no-op, no retry) unless node i+1 actually reads node
i's output -- i.e., unless they communicate. This guarantees, by construction,
that no module this operator creates can ever be fake: `is_fake_module` should be
vacuously false on every module in a population evolved under this `compress`,
not something checked after the fact. A module can still grow past 2 primitives:
an existing module can stand in for `i` or `k` in a later pairwise compress (that
is nesting, gated by `nest_decay` exactly as in `necgp/`), so bigger modules build
up incrementally through successive communicating merges rather than one compress
grabbing an arbitrary run in one shot -- the "logical build-up" the change is for.

WHAT ELSE CHANGED, DELIBERATELY, TO KEEP THIS CONTROLLABLE. `necgp/`'s other four
module-interface operators (`add_input`/`remove_input`/`add_output`/`remove_output`)
and `module_point_mutate` are REMOVED here, not just left at zero probability --
they resize or rewire a module's insides after formation, which is exactly the
kind of extra axis of freedom that makes it hard to attribute a result to the one
change under test. Under this scheme a module's interface is fixed forever at
whatever `compress` gave it; the only way to change one is `expand` (dissolve it)
then re-`compress`. So the operator set is: point mutation, compress (this new
communicating-pair-only version), expand, nesting (via `nest_decay`, unchanged
concept from `necgp/`) -- four knobs, matching what was agreed before building this.

`max_module_size` is REINTERPRETED. Under `necgp/`'s window-based compress, `ms`
bounded window LENGTH (body slot count), which happened to equal primitive count
only because bodies were flat there too... no -- even there a body slot could be a
nested module, so `ms` never actually bounded total flattened size, only slot
count. Under this file's scheme every module body is ALWAYS exactly 2 slots
(one pair), so a slot-count bound is vacuous. `ms` here instead bounds the
resulting module's own RECURSIVELY-FLATTENED primitive count
(`module_active_primitive_count` of each operand, summed) -- `[our choice]`,
chosen to keep `ms`'s role ("how big can one module get") meaningful rather than
silently doing nothing. Default 5, same number `necgp/` used, different meaning.

See ../README.md for the full status of this variant and ../RESULTS.md for what
running it found. Do not cite results from here as if they were `necgp/`'s.
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
# representation -- unchanged from necgp/ecgp.py
# ---------------------------------------------------------------------------

@dataclass
class Module:
    """A module: header + body. See necgp/ecgp.py's `Module` docstring -- unchanged
    here; nesting still works the same way, only how a module gets FORMED changed."""
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
    """One ECGP genotype plus the module list it owns. Unchanged."""
    func: list[int]
    ntype: list[int]
    conn: list[list[int]]
    cout: list[list[int]]
    ogene: list[int]
    ocout: list[int]
    modules: dict[int, Module]
    next_id: int

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
    """Operator probabilities. TRIMMED vs necgp/ecgp.py's `Params` -- see module
    docstring ("what else changed"): no `module_point`/`add_input`/`remove_input`/
    `add_output`/`remove_output`. `max_module_size` reinterpreted, see module
    docstring."""
    compress: float = 0.1
    expand: float = 0.2
    max_module_size: int = 5          # now a total-flattened-primitives cap, `[our choice]`
    mutation_rate: float = 0.03
    nest_decay: float = 0.5


def n_outputs_of(ind: Individual, label: int, n_in: int) -> int:
    if label < n_in:
        return 1
    j = label - n_in
    if ind.ntype[j] == 0:
        return 1
    return ind.modules[ind.func[j]].n_out


def arity_of(ind: Individual, j: int) -> int:
    if ind.ntype[j] == 0:
        return 2
    return ind.modules[ind.func[j]].n_in


def _mod_n_outs(ind: Individual, mod: Module, lbl: int) -> int:
    if lbl < mod.n_in:
        return 1
    b = lbl - mod.n_in
    if mod.ntype[b] == 0:
        return 1
    return ind.modules[mod.func[b]].n_out


def n_gene_slots(ind: Individual) -> int:
    return len(ind.func) + sum(map(len, ind.conn)) + len(ind.ogene)


def _slot_table(ind: Individual) -> tuple[list[int], int, int]:
    acc = list(itertools.accumulate((1 + n for n in map(len, ind.conn)), initial=0))
    n_node_slots = acc.pop()
    return acc, n_node_slots, n_node_slots + len(ind.ogene)


def n_mutations(ind: Individual, rate: float) -> int:
    return max(1, int(round(rate * n_gene_slots(ind))))


def random_individual(rnd: random.Random, n_nodes: int, n_in: int, n_outputs: int,
                      n_prim: int) -> Individual:
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
# genotype point mutation -- unchanged from necgp/ecgp.py
# ---------------------------------------------------------------------------

def point_mutate(ind: Individual, rnd: random.Random, n_mut: int, n_in: int,
                 n_prim: int, table: tuple[list[int], int, int] | None = None) -> None:
    n = len(ind.func)
    starts, n_node_slots, total = table if table is not None else _slot_table(ind)

    choices = list(range(n_prim)) + list(ind.modules)
    rand = rnd.random
    ntype, func, modules = ind.ntype, ind.func, ind.modules

    def n_outs(lbl: int) -> int:
        if lbl < n_in:
            return 1
        j = lbl - n_in
        return 1 if ntype[j] == 0 else modules[func[j]].n_out

    for s in cgp._draw_slots(rnd, total, min(n_mut, total)):
        if s >= n_node_slots:
            o = s - n_node_slots
            lbl = int(rand() * (n_in + n))
            ind.ogene[o] = lbl
            ind.ocout[o] = int(rand() * n_outs(lbl))
            continue

        j = bisect.bisect_right(starts, s) - 1
        off = s - starts[j]

        if off > 0:
            k = off - 1
            if k >= len(ind.conn[j]):
                continue
            lbl = int(rand() * (n_in + j))
            ind.conn[j][k] = lbl
            ind.cout[j][k] = int(rand() * n_outs(lbl))
            continue

        if ntype[j] == 1:
            continue

        was = n_outs(n_in + j)
        f = choices[int(rand() * len(choices))]
        func[j] = f
        ntype[j] = 0 if f < n_prim else 2
        _fit_arity(ind, j, n_in, rnd)
        if n_outs(n_in + j) < was:
            _clamp_refs_to(ind, j, n_in)


def _clamp_refs_to(ind: Individual, j: int, n_in: int) -> None:
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
# compress / expand -- compress is THE new mechanism; expand is unchanged
# ---------------------------------------------------------------------------

def compress(ind: Individual, rnd: random.Random, ms: int, n_in: int,
             nest_decay: float = 1.0) -> bool:
    """EXPERIMENTAL. Encapsulate exactly nodes i and i+1 -- genome-adjacent --
    into a new module, REJECTING (no-op, no retry) unless the resulting module's
    own EXPOSED outputs genuinely depend on an internal edge. See module
    docstring for the full rationale; this is the mechanism under test.

    TWO checks, not one -- the first alone is not sufficient (found empirically,
    see the inline comment lower down): (1) a cheap upfront filter, `node i+1
    reads node i's output somewhere in its own arguments` -- necessary, and
    rejects the common case (i, k unrelated) before building anything; (2) the
    authoritative check, `module_has_interaction` on the fully-constructed
    candidate module -- required because a pair can pass (1) and still expose
    only the *first* node as its output (nothing outside the pair ever used the
    second node's raw value), leaving the second node's dependency on the first
    dead weight the module never surfaces -- fake by the same definition
    `is_fake_module` uses everywhere else, despite (1) having passed.

    `nest_decay` gates depth exactly as in necgp/ecgp.py's `compress`: if either
    node is already a module (type I/II), the roll must beat
    `nest_decay ** (depth - 1)` or the pair is rejected (no retry).

    `ms` bounds the two operands' combined recursively-flattened primitive count
    (`module_active_primitive_count`), not slot count -- see module docstring.
    """
    n = len(ind.func)
    if n < 2:
        return False
    i = int(rnd.random() * (n - 1))
    k = i + 1
    lo_lbl, hi_lbl = n_in + i, n_in + k

    if lo_lbl not in ind.conn[k]:
        return False                     # the two nodes don't communicate

    nested = [j for j in (i, k) if ind.ntype[j] != 0]
    if nested:
        depth = 1 + max(ind.modules[ind.func[j]].depth for j in nested)
        if rnd.random() >= nest_decay ** (depth - 1):
            return False
    else:
        depth = 1

    def prim_count(j: int) -> int:
        if ind.ntype[j] == 0:
            return 1
        return module_active_primitive_count(ind.modules[ind.func[j]], ind.modules)

    if prim_count(i) + prim_count(k) > ms:
        return False

    ln = 2
    body_func = [ind.func[j] for j in (i, k)]
    body_ntype = [ind.ntype[j] for j in (i, k)]
    ext_lbl: list[int] = []
    ext_cout: list[int] = []
    slots: list[tuple[bool, int, int]] = []
    for j in (i, k):
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
        return False
    flat_conn = [(m_in + idx) if internal else idx for internal, idx, _ in slots]
    flat_cout = [c for _, _, c in slots]

    body_conn: list[list[int]] = []
    body_cout: list[list[int]] = []
    pos = 0
    for j in (i, k):
        ar = arity_of(ind, j)
        body_conn.append(flat_conn[pos:pos + ar])
        body_cout.append(flat_cout[pos:pos + ar])
        pos += ar

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
        return False
    out_index = {r: o for o, r in enumerate(refs)}

    candidate = Module(mid=-1, n_in=m_in, func=body_func, ntype=body_ntype,
                       conn=body_conn, cout=body_cout,
                       out=[m_in + r for r, _ in refs],
                       ocout=[c for _, c in refs], depth=depth)
    if not module_has_interaction(candidate, ind.modules):
        # The upfront check above (`lo_lbl in ind.conn[k]`) is necessary but NOT
        # sufficient: it only confirms k reads i's output somewhere in k's own
        # arguments, not that the resulting module's own EXPOSED outputs (`refs`,
        # computed from who referenced i/k from outside the pair) still depend on
        # that edge. If nothing outside ever used k's raw output directly, `refs`
        # can end up exposing only i -- and i alone has no internal dependency,
        # so the "communicating" pair produces a module that is fake by the same
        # definition (`module_has_interaction`) everything else in this file uses.
        # Caught empirically: seed 0's run produced exactly this (module 8, body
        # NAND(in0,in1) -> NAND(prev,in2), but only the first NAND's output was
        # ever referenced externally) before this check was added. This is the
        # authoritative check; the upfront one is now only a cheap fast-reject
        # for the common case where i and k don't reference each other at all.
        return False

    mid = ind.next_id
    ind.next_id += 1
    candidate.mid = mid
    ind.modules[mid] = candidate

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
    """Replace a random type I node with its module's nodes. Unchanged from
    necgp/ecgp.py -- see that file's docstring for why one `expand` undoes exactly
    one level of nesting."""
    type1 = [j for j, t in enumerate(ind.ntype) if t == 1]
    if not type1:
        return False
    p = type1[int(rnd.random() * len(type1))]
    mod = ind.modules[ind.func[p]]
    m, base = mod.n_nodes, n_in + p
    shift = m - 1

    def from_module(lbl: int, c: int) -> tuple[int, int]:
        if lbl < mod.n_in:
            return ind.conn[p][lbl], ind.cout[p][lbl]
        return base + (lbl - mod.n_in), c

    body_conn: list[list[int]] = []
    body_cout: list[list[int]] = []
    for j in range(m):
        pairs = [from_module(mod.conn[j][t], mod.cout[j][t]) for t in range(len(mod.conn[j]))]
        body_conn.append([q[0] for q in pairs])
        body_cout.append([q[1] for q in pairs])

    def remap(lbl: int, c: int) -> tuple[int, int]:
        if lbl < base:
            return lbl, c
        if lbl == base:
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


def module_name(mid: int, n_prim: int) -> str:
    return f"M{mid - n_prim + 1}"


def active_nodes(ind: Individual, n_in: int) -> list[int]:
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
    """Delete every module the individual does not use, TRANSITIVELY. Unchanged."""
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
# the full offspring operator -- TRIMMED: no module_point/add_input/remove_input/
# add_output/remove_output, see module docstring.
# ---------------------------------------------------------------------------

def mutate(ind: Individual, rnd: random.Random, n_in: int, n_prim: int,
           p: Params) -> Individual:
    out = ind.copy()
    rand = rnd.random
    if rand() < p.compress:
        compress(out, rnd, p.max_module_size, n_in, p.nest_decay)
    if rand() < p.expand:
        expand(out, rnd, n_in)
    table = _slot_table(out)
    n_mut = max(1, int(round(p.mutation_rate * table[2])))
    point_mutate(out, rnd, n_mut, n_in, n_prim, table)
    return out


# ---------------------------------------------------------------------------
# decoding -- unchanged from necgp/ecgp.py
# ---------------------------------------------------------------------------

def flatten(ind: Individual, n_in: int) -> cgp.Genotype:
    return flatten_with_origin(ind, n_in, 0)[0]


def _flatten_module(mod: Module, modules: dict[int, Module], n_in: int,
                    args: list[int], tag: str, func: list[int], conn: list[int],
                    origin: list[str], seen: dict[int, int], n_prim: int) -> list[int]:
    body_lbl: list[list[int]] = []

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
            sub = modules[mod.func[b]]
            sub_args = [resolve(row_c, row_o, t) for t in range(len(row_c))]
            seen[mod.func[b]] = seen.get(mod.func[b], 0) + 1
            sub_tag = f"{tag}>{module_name(mod.func[b], n_prim)}#{seen[mod.func[b]]}"
            body_lbl.append(_flatten_module(sub, modules, n_in, sub_args, sub_tag,
                                            func, conn, origin, seen, n_prim))
    return [body_lbl[lbl - mod.n_in][c] for lbl, c in zip(mod.out, mod.ocout)]


def _flatten_and_active(mod: Module, modules: dict[int, Module]
                        ) -> tuple[list[int], set[int]]:
    func: list[int] = []
    conn: list[int] = []
    origin: list[str] = []
    n = mod.n_in
    outs = _flatten_module(mod, modules, n, list(range(n)), "", func, conn,
                           origin, {}, 0)
    seen: set[int] = set()
    stack = [o - n for o in outs if o >= n]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        for lbl in (conn[2 * b], conn[2 * b + 1]):
            if lbl >= n:
                stack.append(lbl - n)
    return conn, seen


def module_active_primitive_count(mod: Module, modules: dict[int, Module]) -> int:
    _, active = _flatten_and_active(mod, modules)
    return len(active)


def module_has_interaction(mod: Module, modules: dict[int, Module]) -> bool:
    n = mod.n_in
    conn, active = _flatten_and_active(mod, modules)
    for b in active:
        for lbl in (conn[2 * b], conn[2 * b + 1]):
            if lbl >= n and (lbl - n) in active:
                return True
    return False


def is_trivial_module(mod: Module, modules: dict[int, Module]) -> bool:
    return module_active_primitive_count(mod, modules) <= 1


def is_fake_module(mod: Module, modules: dict[int, Module]) -> bool:
    """Should be vacuously False for every module formed by this file's `compress`
    -- kept for cross-checking against `necgp/`'s numbers, not expected to fire."""
    return not module_has_interaction(mod, modules)


def flatten_with_origin(ind: Individual, n_in: int,
                        n_prim: int) -> tuple[cgp.Genotype, list[str]]:
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
        outs = _flatten_module(mod, ind.modules, n_in, args, tag, func, conn,
                               origin, seen_calls, n_prim)
        for o, new_lbl in enumerate(outs):
            remap[(lbl, o)] = new_lbl

    n = len(func)
    return cgp.Genotype(func=func, ntype=[0] * n, conn=conn, cout=[0] * (2 * n),
                        ogene=[remap[(l, c)] for l, c in zip(ind.ogene, ind.ocout)],
                        ocout=[0] * len(ind.ogene), arity=2), origin


def _apply(op: int, x: int, y: int, mask: int) -> int:
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
    out = evaluate(ind, gates, in_masks, mask, n_in)[0]
    h = cgp.hits(out, target, mask)
    if kind == "balanced":
        return cgp.balanced_score(out, target, mask), h
    return float(h), h


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

def validate(ind: Individual, n_in: int, n_prim: int) -> None:
    """Structural checks, unchanged from necgp/ecgp.py EXCEPT the module body-size
    assertion: every module here always has exactly 2 body slots (a pair), a hard
    floor/ceiling of 2 rather than a `ms`-bounded range -- `ms` no longer bounds
    slot count (see module docstring), so it is not a parameter here."""
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
        assert mod.n_nodes == 2, f"module {mid}: {mod.n_nodes} body slots, expected 2"
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
        assert module_has_interaction(mod, ind.modules), \
            f"module {mid}: FAKE module survived -- compress's guarantee is broken"
