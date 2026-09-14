"""What gates is the circuit made of, right now? -- the per-snapshot census.

Pure Python (stdlib only), so it runs inside the PyPy search. Reads an individual,
never touches the search RNG, so turning snapshots on or off cannot change a run.

Two ways of counting "how much of the circuit is gate X", both logged, because they
answer different questions:

- **top-level share** (`share_top`): of the boxes the unflattened drawing shows (every
  active genome node, primitive or module call), what fraction is X? This is the
  "ratio of gates used" as you would read it off a stage sheet.
- **flattened share** (`share_flat`): of the primitive gates the circuit actually
  evaluates, what fraction sits inside a top-level call to module X? A module called
  twice that holds 4 NANDs each accounts for 8. This is how much of the computation
  lives inside the invented gate.

A module's calls are also counted **transitively** (`calls_all`): top-level calls plus
calls made from inside other active modules' bodies, the same walk
`run.transitive_module_counts` does.

FUNCTION NAMES. Every module gets a truth table, a canonical signature (dummy inputs
dropped, inputs relabelled to the lexicographic minimum -- equal signature iff same
function up to input relabelling) and a human label when it is one worth naming
(`XOR`, `A|~B`, ...). Ported from `experiment_4/ecgp.py` (`_support`, `_influence`,
`_canonical`, `_signature`, `_KNOWN_FUNCS`), extended to nested bodies. In this
variant a module's body never changes after `compress` (no `module_point_mutate`), so
a module id names one function for its whole life and the cache can key on it.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from typing import Sequence

import cgp
import ecgp

MAX_CANON_PERMS = 720

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
    ("NAND3", 3, lambda b: 1 - (b[0] & b[1] & b[2])),
    ("NOR3",  3, lambda b: 1 - (b[0] | b[1] | b[2])),
    ("XOR3",  3, lambda b: b[0] ^ b[1] ^ b[2]),
    ("MAJ3",  3, lambda b: int(b[0] + b[1] + b[2] >= 2)),
    ("MUX",   3, lambda b: b[1] if b[0] else b[2]),
]


# ---------------------------------------------------------------------------
# truth tables and canonical signatures (ported from experiment_4/ecgp.py)
# ---------------------------------------------------------------------------

def module_table(mod: ecgp.Module, modules: dict[int, ecgp.Module],
                 ops: Sequence[int]) -> list[int]:
    """One truth-table mask per module output; bit p = output on input bits of p."""
    n = mod.n_in
    npat = 1 << n
    args = [sum(1 << p for p in range(npat) if (p >> i) & 1) for i in range(n)]
    return ecgp._run_module(mod, modules, args, ops, (1 << npat) - 1)


def _support(tables: Sequence[int], n: int) -> list[int]:
    dep = []
    for i in range(n):
        bit = 1 << i
        if any(((t >> p) & 1) != ((t >> (p ^ bit)) & 1)
               for t in tables for p in range(1 << n)):
            dep.append(i)
    return dep


def _restrict(t: int, sup: Sequence[int], perm: Sequence[int]) -> int:
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
    bit = 1 << i
    return tuple(sum(1 for p in range(1 << n)
                     if ((t >> p) & 1) != ((t >> (p ^ bit)) & 1))
                 for t in tables)


def _canonical(tables: Sequence[int], n: int) -> tuple[int, tuple[int, ...], bool]:
    sup = _support(tables, n)
    s = len(sup)
    prof = [_influence(tables, n, i) for i in sup]
    order = sorted(range(s), key=lambda k: (prof[k], k))

    groups, k = [], 0
    while k < s:
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


def signature(tables: Sequence[int], n: int) -> str:
    """Printable canonical key. Equal key <=> same function up to input relabelling."""
    s, canon, approx = _canonical(tables, n)
    body = "-".join(f"{t:x}" for t in canon)
    if len(body) > 24:
        body = hashlib.blake2b(body.encode(), digest_size=6).hexdigest()
    return f"{'~' if approx else ''}{s}:{body}"


def _known_table(n_bits: int, fn) -> int:
    t = 0
    for p in range(1 << n_bits):
        if fn([(p >> i) & 1 for i in range(n_bits)]):
            t |= 1 << p
    return t


_KNOWN_BY_SIG = {signature([_known_table(k, f)], k): name for name, k, f in _KNOWN_FUNCS}


def module_expr(mod: ecgp.Module, gates, n_prim: int) -> str:
    """The body as an s-expression over formal inputs i0, i1, ...; a nested call
    reads `(M3.0 i0 i1)` = output 0 of module M3."""
    def node(lbl: int, c: int) -> str:
        if lbl < mod.n_in:
            return f"i{lbl}"
        j = lbl - mod.n_in
        args = " ".join(node(l, o) for l, o in zip(mod.conn[j], mod.cout[j]))
        if mod.ntype[j] == 0:
            return f"({gates[mod.func[j]].name.upper()} {args})"
        return f"({ecgp.module_name(mod.func[j], n_prim)}.{c} {args})"
    return " ; ".join(node(l, c) for l, c in zip(mod.out, mod.ocout))


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------

CENSUS_FIELDS = ["gen", "gate", "kind", "label", "signature", "fake", "depth",
                 "n_in", "n_out", "size", "calls_top", "calls_all", "flat_prims",
                 "share_top", "share_flat", "born_gen", "expr"]

LOG_FIELDS = ["gen", "reason", "hits", "acc", "n_modules", "n_modules_active",
              "active_top", "prim_top", "module_top", "module_share_top",
              "flat_active", "flat_in_modules", "module_share_flat", "max_depth_active",
              "distinct_functions_active", "elapsed_s"]


def _module_active_body(mod: ecgp.Module) -> set[int]:
    active: set[int] = set()
    stack = [lbl - mod.n_in for lbl in mod.out if lbl >= mod.n_in]
    while stack:
        b = stack.pop()
        if b in active:
            continue
        active.add(b)
        for lbl in mod.conn[b]:
            if lbl >= mod.n_in:
                stack.append(lbl - mod.n_in)
    return active


class Census:
    """One per seed: owns the module-info cache and the birth generations."""

    def __init__(self, gates, n_in: int, n_prim: int, n_patterns: int):
        self.gates, self.n_in, self.n_prim, self.n_patterns = gates, n_in, n_prim, n_patterns
        self.ops = tuple(g.op for g in gates)
        self.born: dict[int, int] = {}
        self._info: dict[int, tuple[str, str, str, int, bool]] = {}

    def note_births(self, ind: ecgp.Individual, gen: int) -> None:
        """Record the generation each module id first entered the lineage's parent."""
        for mid in ind.modules:
            if mid not in self.born:
                self.born[mid] = gen

    def info(self, mid: int, modules: dict[int, ecgp.Module]) -> tuple[str, str, str, int, bool]:
        """(label, signature, expr, flattened size, fake) -- cached by id (bodies are immutable)."""
        hit = self._info.get(mid)
        if hit is None:
            mod = modules[mid]
            tables = module_table(mod, modules, self.ops)
            # per output: its name when it has one, else its own canonical signature,
            # so a 2-output module never reads as if it were only its named half
            out_sigs = [signature([t], mod.n_in) for t in tables]
            labels = [_KNOWN_BY_SIG.get(s, "") for s in out_sigs]
            hit = (" ; ".join(l or s for l, s in zip(labels, out_sigs)) if any(labels) else "",
                   signature(tables, mod.n_in),
                   module_expr(mod, self.gates, self.n_prim),
                   ecgp.module_active_primitive_count(mod, modules),
                   ecgp.is_fake_module(mod, modules))
            self._info[mid] = hit
        return hit

    def take(self, ind: ecgp.Individual, gen: int, hits: int, reason: str,
             elapsed: float) -> tuple[dict, list[dict]]:
        """(one log.csv row, the gates.csv rows) for `ind` at generation `gen`."""
        n_in, n_prim = self.n_in, self.n_prim
        active = ecgp.active_nodes(ind, n_in)

        calls_top: Counter = Counter()
        prim_top: Counter = Counter()
        for j in active:
            if ind.ntype[j] == 0:
                prim_top[ind.func[j]] += 1
            else:
                calls_top[ind.func[j]] += 1

        calls_all: Counter = Counter()

        def walk(func, ntype, idx, modules) -> None:
            for b in idx:
                if ntype[b] != 0:
                    calls_all[func[b]] += 1
                    m = modules[func[b]]
                    walk(m.func, m.ntype, _module_active_body(m), modules)
        walk(ind.func, ind.ntype, active, ind.modules)

        flat, origin = ecgp.flatten_with_origin(ind, n_in, n_prim)
        flat_active = cgp.active_nodes(flat, n_in, self.gates)
        # NB the flattened active set can be SMALLER than the genotype's: a module
        # input its body never reads keeps its source node genotype-active but
        # flattened-inactive. So primitive flat counts come from the flat walk too.
        flat_by_top: Counter = Counter()
        flat_prim: Counter = Counter()
        for b in flat_active:
            tag = origin[b]
            if tag:
                flat_by_top[tag.split(">", 1)[0].split("#", 1)[0]] += 1
            else:
                flat_prim[flat.func[b]] += 1
        n_flat = len(flat_active)
        n_top = len(active)

        rows: list[dict] = []
        for f in sorted(set(prim_top) | set(flat_prim)):
            name = self.gates[f].name.upper()
            c, fp = prim_top.get(f, 0), flat_prim.get(f, 0)
            rows.append(dict(gen=gen, gate=name, kind="primitive", label=name,
                             signature="", fake=0, depth=0, n_in=2, n_out=1, size=1,
                             calls_top=c, calls_all=c, flat_prims=fp,
                             share_top=round(c / n_top, 6) if n_top else 0.0,
                             share_flat=round(fp / n_flat, 6) if n_flat else 0.0,
                             born_gen="", expr=""))
        depths, functions = [], set()
        for mid in sorted(ind.modules):
            mod = ind.modules[mid]
            name = ecgp.module_name(mid, n_prim)
            label, sig, expr, size, fake = self.info(mid, ind.modules)
            ct, fp = calls_top.get(mid, 0), flat_by_top.get(name, 0)
            if calls_all.get(mid, 0):
                depths.append(mod.depth)
                functions.add(sig)
            rows.append(dict(gen=gen, gate=name, kind="module", label=label,
                             signature=sig, fake=int(fake), depth=mod.depth,
                             n_in=mod.n_in, n_out=mod.n_out, size=size,
                             calls_top=ct, calls_all=calls_all.get(mid, 0), flat_prims=fp,
                             share_top=round(ct / n_top, 6) if n_top else 0.0,
                             share_flat=round(fp / n_flat, 6) if n_flat else 0.0,
                             born_gen=self.born.get(mid, ""), expr=expr))

        module_top = sum(calls_top.values())
        in_mods = sum(flat_by_top.values())
        log = dict(gen=gen, reason=reason, hits=hits,
                   acc=round(hits / self.n_patterns, 6),
                   n_modules=len(ind.modules),
                   n_modules_active=sum(1 for m in ind.modules if calls_all.get(m, 0)),
                   active_top=n_top, prim_top=sum(prim_top.values()), module_top=module_top,
                   module_share_top=round(module_top / n_top, 6) if n_top else 0.0,
                   flat_active=n_flat, flat_in_modules=in_mods,
                   module_share_flat=round(in_mods / n_flat, 6) if n_flat else 0.0,
                   max_depth_active=max(depths, default=0),
                   distinct_functions_active=len(functions),
                   elapsed_s=round(elapsed, 2))
        return log, rows


# ---------------------------------------------------------------------------
# genotype (de)serialisation -- snapshots.jsonl
# ---------------------------------------------------------------------------

def to_json(ind: ecgp.Individual) -> dict:
    return dict(func=ind.func, ntype=ind.ntype, conn=ind.conn, cout=ind.cout,
                ogene=ind.ogene, ocout=ind.ocout, next_id=ind.next_id,
                modules={str(mid): dict(n_in=m.n_in, func=m.func, ntype=m.ntype,
                                        conn=m.conn, cout=m.cout, out=m.out,
                                        ocout=m.ocout, depth=m.depth)
                         for mid, m in ind.modules.items()})


def from_json(d: dict) -> ecgp.Individual:
    modules = {int(k): ecgp.Module(mid=int(k), **v) for k, v in d["modules"].items()}
    return ecgp.Individual(func=d["func"], ntype=d["ntype"], conn=d["conn"],
                           cout=d["cout"], ogene=d["ogene"], ocout=d["ocout"],
                           modules=modules, next_id=d["next_id"])
