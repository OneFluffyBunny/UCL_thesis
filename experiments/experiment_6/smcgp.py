"""SMCGP genotype, development (self-modification) and evaluation.

Spec: `PAPER_SPEC.md` (Harding, Miller & Banzhaf, "Self Modifying Cartesian Genetic
Programming: Parity," CEC 2009). Every design choice below is either [verbatim]
(taken directly from the paper's text) or explicitly marked [inferred]/[our choice]
where the paper is silent or under-specified -- see PAPER_SPEC.md section "Gaps we
had to fill" for the full list and reasoning. Do not treat an unmarked choice as
paper fact; check PAPER_SPEC.md.

REPRESENTATION [verbatim, section III-B/III-D]. A genotype is a fixed-length list
of nodes. Every node has exactly 7 genes -- one function gene, two connection
genes (arity is fixed at 2 for every node, including INPUT and self-modifying
nodes, which simply ignore genes they don't need -- CGP's usual "surplus gene"
neutrality), three real-valued parameter genes (P0, P1, P2), and one binary
output-flag gene.

ADDRESSING [verbatim, section III-B]. Connection genes are RELATIVE and
backward-only: a gene value `c` means "c nodes back from me." A gene must be > 0.
If `c` points before the start of the graph, the connection resolves to a constant
0 (not an error) -- this, not absolute indices, is what lets a duplicated
sub-graph keep working wherever development moves it.

FUNCTION SET [verbatim, section III-B/V]. One unified id space covers the INPUT
terminal, the computational functions (AND/OR/NAND/NOR, or the full 16-function
BF0..BF15 set -- `gates.py`), and the 13 self-modification operators. A node's
function gene picks from ALL of these; evolution decides whether a node computes
or rewrites the graph, so genotype length does not change under mutation --
growth is a property of the PHENOTYPE the self-modifying nodes build during
development, discarded and rebuilt fresh from the genotype at every evaluation
(section III-D: "the genotype [is] invariant during evaluation ... all
modifications [are] performed on the phenotype").

DEVELOPMENT [verbatim, section III-D]. Per test case of `n_in` inputs, the
phenotype starts as a copy of the genotype and is iterated `n_in - 2` times
(so the smallest, 2-input, case runs zero iterations -- phenotype == genotype).
Each iteration is: (1) a backward structural walk from the output node(s) --
identical in spirit to plain CGP's active-node walk, generalised to relative
addressing -- collecting the active self-modifying nodes in left-to-right
(ascending position) order onto a shared "To Do" list, truncated to
`todo_cap` entries (paper default 2); (2) applying that list's operators, in
order, to produce the next phenotype. This module's `develop()` does exactly
that; only the FINAL phenotype (after all iterations) is ever numerically
evaluated -- development itself needs no input values, only graph structure,
because CGP active-node-ness never depends on wire values.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Sequence

from gates import Gate

# --- self-modifying operator names, PAPER_SPEC.md section "Operator table". Order
# here is arbitrary (it only fixes which function-gene ids they occupy).
SM_OPS = ("DU4", "SHIFTCONNECTION", "MULTCONNECTION", "MOV", "DUP", "DU3",
          "DEL", "ADD", "CHF", "CHC", "CHP", "OVR", "COPYTOSTOP")


class FuncSpec(NamedTuple):
    name: str
    kind: str                                   # "input" | "comp" | "sm"
    fn: Callable[[int, int, int], int] | None    # comp only: (a, b, mask) -> mask


def build_function_table(gate_set: dict[str, Gate]) -> tuple[FuncSpec, ...]:
    """The unified function-gene id space: [INP, <gates>, <13 SM ops>]."""
    table = [FuncSpec("INP", "input", None)]
    table += [FuncSpec(name, "comp", g.fn) for name, g in gate_set.items()]
    table += [FuncSpec(op, "sm", None) for op in SM_OPS]
    return tuple(table)


# ---------------------------------------------------------------------------
# genotype
# ---------------------------------------------------------------------------

@dataclass(eq=False)          # identity equality: `apply_todo` locates nodes by
                               # `is`/`list.index`, not by gene-value equality, so
                               # two coincidentally-identical nodes are never confused
class Node:
    func: int
    c0: int
    c1: int
    p0: float
    p1: float
    p2: float
    out: int

    def copy(self) -> "Node":
        return Node(self.func, self.c0, self.c1, self.p0, self.p1, self.p2, self.out)


@dataclass
class Genotype:
    nodes: list[Node]

    def copy(self) -> "Genotype":
        return Genotype([n.copy() for n in self.nodes])

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def random_node(rnd: random.Random, n_funcs: int, addr_max: int,
                param_range: float) -> Node:
    """[our choice] `out` starts at 0 for every node, not a fair coin toss.

    With N nodes and a Bernoulli(0.5) output flag, the LEFTMOST flagged node
    (section III-C's selection rule) lands within the first couple of positions
    almost every time -- and a node that early has nowhere legal to point
    backward to, so every one of its connections resolves to the constant-0
    fallback (section III-B). The result is a circuit trapped as a fixed
    constant, which scores exactly 2/4 against any balanced 2-input target and
    is a plateau mutation can never climb out of (verified empirically: a
    300k-evaluation run never left it). Starting with no flags set means the
    initial population instead uses the paper's OTHER stated fallback --
    "if no nodes are flagged, the last n nodes... revert[ing] the system to
    the previous [i.e. plain CGP] approach" -- which does have full reach, and
    flags are introduced gradually as mutation (rate `mutation_rate`) sets them.
    """
    return Node(
        func=rnd.randrange(n_funcs),
        c0=1 + rnd.randrange(addr_max), c1=1 + rnd.randrange(addr_max),
        p0=rnd.uniform(-param_range, param_range),
        p1=rnd.uniform(-param_range, param_range),
        p2=rnd.uniform(-param_range, param_range),
        out=0)


def random_genotype(rnd: random.Random, n_nodes: int, n_funcs: int,
                    addr_max: int, param_range: float) -> Genotype:
    """[our choice] Uniform-random init over every gene's domain -- the paper
    states the representation but not an init distribution; this is the same
    convention `cgp.random_genotype` uses."""
    return Genotype([random_node(rnd, n_funcs, addr_max, param_range)
                     for _ in range(n_nodes)])


# ---------------------------------------------------------------------------
# mutation [verbatim, section IV]
# ---------------------------------------------------------------------------

def mutate(g: Genotype, rnd: random.Random, n_funcs: int, addr_max: int,
          param_range: float, mutation_rate: float = 0.1,
          param_randomize_prob: float = 0.1, sigma: float = 20.0) -> Genotype:
    """Independent per-gene mutation, PAPER_SPEC.md section IV, paper-verbatim
    rates (0.1 / 0.1 / sigma=20). Function and connection genes are resampled
    uniformly ("unbiased; a gene can be mutated to any other valid value").
    Real-valued genes: 10% chance of a fresh uniform draw, else additive
    N(0, sigma) noise. The output-flag gene is binary, so "any other valid
    value" is a flip.
    """
    out = g.copy()
    for node in out.nodes:
        if rnd.random() < mutation_rate:
            node.func = rnd.randrange(n_funcs)
        if rnd.random() < mutation_rate:
            node.c0 = 1 + rnd.randrange(addr_max)
        if rnd.random() < mutation_rate:
            node.c1 = 1 + rnd.randrange(addr_max)
        for attr in ("p0", "p1", "p2"):
            if rnd.random() < mutation_rate:
                if rnd.random() < param_randomize_prob:
                    setattr(node, attr, rnd.uniform(-param_range, param_range))
                else:
                    setattr(node, attr, getattr(node, attr) + rnd.gauss(0.0, sigma))
        if rnd.random() < mutation_rate:
            node.out = 1 - node.out
    return out


# ---------------------------------------------------------------------------
# output selection + the structural (active-node) walk
# ---------------------------------------------------------------------------

def select_outputs(nodes: list[Node], n_out: int) -> tuple[list[int], bool]:
    """Which node positions feed the program outputs. [verbatim, section III-C]:
    flagged nodes (out==1) are used, leftmost first, if there are enough; with
    none flagged, fall back to the last `n_out` nodes; fewer nodes than `n_out`
    in the graph at all is CORRUPT (bad fitness, not evaluated).

    [inferred] The paper covers "enough flagged" and "none flagged" explicitly
    but not "some flagged, fewer than n_out" -- we fall back to the last-`n_out`
    rule in that case too, rather than inventing an unaddressed hybrid.
    """
    n = len(nodes)
    if n < n_out:
        return [], True
    flagged = [i for i, nd in enumerate(nodes) if nd.out]
    if len(flagged) >= n_out:
        return flagged[:n_out], False
    return list(range(n - n_out, n)), False


def active_walk(nodes: list[Node], ftable: Sequence[FuncSpec],
                n_out: int) -> tuple[list[int], list[int], bool]:
    """(active node positions ascending, output positions, corrupt).

    Purely structural -- CGP active-ness never depends on wire values, only on
    which nodes the outputs reach -- so this needs no input values and is reused
    both by `develop` (to find the active self-modifying nodes) and internally
    by `evaluate` (to know what to compute).
    """
    outs, corrupt = select_outputs(nodes, n_out)
    if corrupt:
        return [], outs, True
    n = len(nodes)
    seen = bytearray(n)
    stack = list(outs)
    while stack:
        pos = stack.pop()
        if seen[pos]:
            continue
        seen[pos] = 1
        node = nodes[pos]
        if ftable[node.func].kind == "input":
            continue
        for c in (node.c0, node.c1):
            src = pos - c
            if 0 <= src < pos:
                stack.append(src)
    active = [i for i in range(n) if seen[i]]
    return active, outs, False


# ---------------------------------------------------------------------------
# self-modification operators [verbatim formulas, section V; edge-case handling
# per the shared rules stated there: Pi+x truncated to int, addresses clamped to
# [0, n], start/end sorted ascending, redundant ops are no-ops]
# ---------------------------------------------------------------------------

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _range_at_x(node: Node, x: int, n: int) -> tuple[int, int]:
    """(start, end) = sorted(P0+x, P0+x+P1), clamped to [0, n]."""
    a = int(node.p0) + x
    b = int(node.p0) + x + int(node.p1)
    a, b = sorted((a, b))
    return _clamp(a, 0, n), _clamp(b, 0, n)


def _apply_du4(nodes: list[Node], node: Node, x: int) -> None:
    n = len(nodes)
    start = _clamp(int(node.p0) + x, 0, n)
    count = max(0, int(node.p1))
    end = min(n, start + count)
    insert_at = _clamp(int(node.p0) + x + int(node.p1) + 1, 0, n)   # "after" -> +1
    scale = node.p2
    copied = [Node(k.func, max(1, int(round(k.c0 * scale))), max(1, int(round(k.c1 * scale))),
                   k.p0, k.p1, k.p2, k.out) for k in nodes[start:end]]
    nodes[insert_at:insert_at] = copied


def _apply_shiftconnection(nodes: list[Node], node: Node, x: int) -> None:
    n = len(nodes)
    start = _clamp(int(node.p0) + x, 0, n)
    count = max(0, int(node.p1))
    delta = int(node.p2)
    for k in nodes[start:start + count]:
        k.c0 = max(1, k.c0 + delta)
        k.c1 = max(1, k.c1 + delta)


def _apply_multconnection(nodes: list[Node], node: Node, x: int) -> None:
    n = len(nodes)
    start = _clamp(int(node.p0) + x, 0, n)
    count = max(0, int(node.p1))
    factor = node.p2
    for k in nodes[start:start + count]:
        k.c0 = max(1, int(round(k.c0 * factor)))
        k.c1 = max(1, int(round(k.c1 * factor)))


def _apply_mov(nodes: list[Node], node: Node, x: int) -> None:
    n = len(nodes)
    start, end = _range_at_x(node, x, n)
    if start >= end:
        return
    segment = nodes[start:end]
    del nodes[start:end]
    insert_at = _clamp(int(node.p0) + x + int(node.p2) + 1, 0, len(nodes))   # "after" -> +1
    nodes[insert_at:insert_at] = segment


def _apply_dup(nodes: list[Node], node: Node, x: int) -> None:
    n = len(nodes)
    start, end = _range_at_x(node, x, n)
    if start >= end:
        return
    copied = [k.copy() for k in nodes[start:end]]
    insert_at = _clamp(int(node.p0) + x + int(node.p2) + 1, 0, len(nodes))   # "after" -> +1
    nodes[insert_at:insert_at] = copied


def _apply_du3(nodes: list[Node], node: Node, x: int) -> None:
    """DuplicatePreservingConnections: copies keep pointing at the ORIGINAL
    absolute targets, so each copy's relative address is recomputed for its new
    position. [inferred] If the new position would need a forward or
    self address (new_pos <= original target), the address is set to guarantee
    an out-of-range (-> resolves to constant 0) result instead, since a
    non-positive relative address is outside this representation's domain."""
    n = len(nodes)
    start, end = _range_at_x(node, x, n)
    if start >= end:
        return
    orig_targets = []
    for k in nodes[start:end]:
        pos_of_k = nodes.index(k)
        orig_targets.append((pos_of_k - k.c0, pos_of_k - k.c1))
    copied = [k.copy() for k in nodes[start:end]]
    insert_at = _clamp(int(node.p0) + x + int(node.p2) + 1, 0, len(nodes))   # "after" -> +1

    def addr_for(new_pos: int, target: int) -> int:
        d = new_pos - target
        return d if d > 0 else new_pos + 1

    for offset, (k, (t0, t1)) in enumerate(zip(copied, orig_targets)):
        new_pos = insert_at + offset
        k.c0 = addr_for(new_pos, t0)
        k.c1 = addr_for(new_pos, t1)
    nodes[insert_at:insert_at] = copied


def _apply_del(nodes: list[Node], node: Node, x: int) -> None:
    n = len(nodes)
    start, end = _range_at_x(node, x, n)
    del nodes[start:end]


def _apply_add(nodes: list[Node], node: Node, x: int, n_funcs: int,
               addr_max: int, param_range: float, rnd: random.Random) -> None:
    n = len(nodes)
    insert_at = _clamp(int(node.p0) + x + 1, 0, n)
    count = max(0, int(node.p1))
    new_nodes = [random_node(rnd, n_funcs, addr_max, param_range) for _ in range(count)]
    nodes[insert_at:insert_at] = new_nodes


def _apply_chf(nodes: list[Node], node: Node, n_funcs: int) -> None:
    """[verbatim] Targets node P0 directly -- NOT P0+x, unlike the range ops above."""
    n = len(nodes)
    idx = _clamp(int(node.p0), 0, n - 1) if n else None
    if idx is not None:
        nodes[idx].func = _clamp(int(node.p1), 0, n_funcs - 1)


def _apply_chc(nodes: list[Node], node: Node) -> None:
    """[verbatim target: node P0, not P0+x]. [our choice] `P1 mod 2`, not the
    paper's `mod 3` -- this representation fixes arity at 2 (two connection
    genes), so mod 3 (from the paper's more general, variable-arity SMCGP) would
    sometimes pick a non-existent third connection slot."""
    n = len(nodes)
    if not n:
        return
    idx = _clamp(int(node.p0), 0, n - 1)
    which = int(node.p1) % 2
    target = nodes[idx]
    val = max(1, int(node.p2))
    if which == 0:
        target.c0 = val
    else:
        target.c1 = val


def _apply_chp(nodes: list[Node], node: Node) -> None:
    """[verbatim] node P0, (P1 mod 3)th of the three real parameter genes."""
    n = len(nodes)
    if not n:
        return
    idx = _clamp(int(node.p0), 0, n - 1)
    which = int(node.p1) % 3
    target = nodes[idx]
    attr = ("p0", "p1", "p2")[which]
    setattr(target, attr, node.p2)


def _apply_ovr(nodes: list[Node], node: Node, x: int) -> None:
    """[inferred] Overwrite does not grow the graph: the copy is clipped to fit
    within the existing node count (the paper does not say whether OVR may
    extend the phenotype; DUP/MOV/ADD already cover deliberate growth)."""
    n = len(nodes)
    start, end = _range_at_x(node, x, n)
    if start >= end:
        return
    segment = [k.copy() for k in nodes[start:end]]
    dest = _clamp(int(node.p0) + x + int(node.p2), 0, n)
    fit = min(len(segment), n - dest)
    for i in range(fit):
        nodes[dest + i] = segment[i]


def _apply_copytostop(nodes: list[Node], node: Node, x: int,
                      ftable: Sequence[FuncSpec]) -> None:
    """[inferred] No standalone "STOP" function is defined in Table I (the only
    stop marker in this implementation's function set is another COPYTOSTOP
    node), so the scan-forward-for-a-stop-marker rule is applied with just that
    one terminator plus end-of-graph."""
    n = len(nodes)
    if x >= n:
        return
    stop = n
    for j in range(x + 1, n):
        if ftable[nodes[j].func].name == "COPYTOSTOP":
            stop = j
            break
    segment = [k.copy() for k in nodes[x:min(stop, n)]]
    insert_at = min(stop, len(nodes))
    nodes[insert_at:insert_at] = segment


_OP_TABLE = {
    "DU4": lambda nodes, node, x, ctx: _apply_du4(nodes, node, x),
    "SHIFTCONNECTION": lambda nodes, node, x, ctx: _apply_shiftconnection(nodes, node, x),
    "MULTCONNECTION": lambda nodes, node, x, ctx: _apply_multconnection(nodes, node, x),
    "MOV": lambda nodes, node, x, ctx: _apply_mov(nodes, node, x),
    "DUP": lambda nodes, node, x, ctx: _apply_dup(nodes, node, x),
    "DU3": lambda nodes, node, x, ctx: _apply_du3(nodes, node, x),
    "DEL": lambda nodes, node, x, ctx: _apply_del(nodes, node, x),
    "ADD": lambda nodes, node, x, ctx: _apply_add(
        nodes, node, x, ctx["n_funcs"], ctx["addr_max"], ctx["param_range"], ctx["rnd"]),
    "CHF": lambda nodes, node, x, ctx: _apply_chf(nodes, node, ctx["n_funcs"]),
    "CHC": lambda nodes, node, x, ctx: _apply_chc(nodes, node),
    "CHP": lambda nodes, node, x, ctx: _apply_chp(nodes, node),
    "OVR": lambda nodes, node, x, ctx: _apply_ovr(nodes, node, x),
    "COPYTOSTOP": lambda nodes, node, x, ctx: _apply_copytostop(nodes, node, x, ctx["ftable"]),
}


def apply_todo(nodes: list[Node], todo_positions: list[int],
               ftable: Sequence[FuncSpec], n_funcs: int, addr_max: int,
               param_range: float, rnd: random.Random) -> list[Node]:
    """Apply the queued self-modifying nodes, in order, FIFO. [inferred: `x`,
    "the integer position of the node ... that contained the self modifying
    function", is re-resolved at the moment each op executes -- not frozen at
    collection time -- since earlier ops in the same batch can shift positions.
    Nodes are identified by object identity (`Node` has `eq=False`) so a node
    deleted by an earlier op in the batch is simply skipped as a no-op, matching
    "redundant operations are ignored."]
    """
    nodes = list(nodes)
    todo_nodes = [nodes[i] for i in todo_positions]
    ctx = dict(n_funcs=n_funcs, addr_max=addr_max, param_range=param_range,
              rnd=rnd, ftable=ftable)
    for node in todo_nodes:
        if node not in nodes:
            continue
        x = nodes.index(node)
        op = ftable[node.func].name
        _OP_TABLE[op](nodes, node, x, ctx)
    return nodes


def develop(genotype: Genotype, ftable: Sequence[FuncSpec], n_out: int,
           n_iterations: int, todo_cap: int, n_funcs: int, addr_max: int,
           param_range: float, rnd: random.Random) -> list[Node]:
    """The phenotype after `n_iterations` development steps from `genotype`.

    `n_iterations` is always `n_in - 2` for the test case being scored
    (PAPER_SPEC.md section VI-A), so 0 for the 2-input case: phenotype ==
    genotype exactly, no development happens for the smallest test case.
    """
    nodes = [n.copy() for n in genotype.nodes]
    for _ in range(n_iterations):
        active, _outs, corrupt = active_walk(nodes, ftable, n_out)
        if corrupt:
            break
        todo = [i for i in active if ftable[nodes[i].func].kind == "sm"][:todo_cap]
        if not todo:
            continue
        nodes = apply_todo(nodes, todo, ftable, n_funcs, addr_max, param_range, rnd)
    return nodes


# ---------------------------------------------------------------------------
# evaluation -- bit-parallel over truth-table masks, final phenotype only
# ---------------------------------------------------------------------------

def evaluate(nodes: list[Node], ftable: Sequence[FuncSpec], n_out: int,
            in_masks: Sequence[int], mask: int) -> list[int] | None:
    """Output truth-table masks for the (already-developed) phenotype, or
    `None` if it is corrupt (fewer nodes than outputs, PAPER_SPEC.md III-C)."""
    active, outs, corrupt = active_walk(nodes, ftable, n_out)
    if corrupt:
        return None
    n_in = len(in_masks)

    # INP consumption order [verbatim rule, ambiguous ordering -- inferred as
    # ascending active-node position, i.e. "leftmost first", consistent with
    # every other left-to-right rule this paper states]: each successive INP
    # call (in that order) gets the next input, wrapping around.
    vals: dict[int, int] = {}
    inp_i = 0
    for j in active:
        node = nodes[j]
        f = ftable[node.func]
        if f.kind == "input":
            vals[j] = in_masks[inp_i % n_in]
            inp_i += 1
        elif f.kind == "comp":
            a = _resolve(nodes, vals, j, node.c0)
            b = _resolve(nodes, vals, j, node.c1)
            vals[j] = f.fn(a, b, mask)
        else:                     # sm node left un-resolved in the final phenotype:
            vals[j] = _resolve(nodes, vals, j, node.c0)   # pass-through of its
            # first input [inferred, generalising the paper's one worked example:
            # "The DUP function ... is defined to return the first input"]
    return [vals.get(j, 0) for j in outs]


def _resolve(nodes: list[Node], vals: dict[int, int], pos: int, c: int) -> int:
    src = pos - c
    if 0 <= src < pos:
        return vals[src]
    return 0        # out-of-range address -> constant 0 [verbatim, section III-B]


def hits(out_mask: int, target: int, mask: int) -> int:
    return int((~(out_mask ^ target) & mask).bit_count())
