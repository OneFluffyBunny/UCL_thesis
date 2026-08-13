"""CGP genotype, mutation and evaluation -- pure Python/numpy, no JAX.

Spec: `PAPER_SPEC.md` (Walker & Miller, IEEE TEVC 12(4), 2008). Sections cited
below refer to it.

REPRESENTATION (spec section 1). One row; a node may connect to *any* previous
node -- there is no levels-back parameter in this formulation. Program inputs are
labelled `0 .. n_in-1`; node `j` is labelled `n_in + j` and may reference any
label strictly below its own.

Genes are stored as **pairs**, which is the ECGP encoding:

    func[j], ntype[j]        function gene:  (function-or-module id, node type)
    conn[j,k], cout[j,k]     input gene k:   (source label, which output of it)
    ogene[o], ocout[o]       output gene o:  (source label, which output of it)

For pure CGP every `ntype` is 0 and every `*cout` is 0, exactly as the paper
states ("every node encoded in the ECGP genotype is of node type 0, and the second
integer of each pair encoding the node inputs is always 0"). Keeping the pairs now
means ECGP slots in later without rewriting the genotype.

EVALUATION -- truth-table masks. Each wire's behaviour over *all* 2**n_in input
patterns is a single Python int: bit `r` is the wire's value on pattern `r`. A
gate is then one bitwise op over every pattern simultaneously, and fitness is a
population count. This is exact, not approximate -- `evaluate_slow` below is the
obvious per-pattern implementation and `test_cgp.py` asserts the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gates import Gate


@dataclass
class Genotype:
    """One individual. Arrays are int32; see the module docstring for the layout."""
    func: np.ndarray    # (n_nodes,)            function id, indexes the gate set
    ntype: np.ndarray   # (n_nodes,)            node type; always 0 in CGP
    conn: np.ndarray    # (n_nodes, arity)      source label per input gene
    cout: np.ndarray    # (n_nodes, arity)      which output of that source; always 0 in CGP
    ogene: np.ndarray   # (n_outputs,)          source label per program output
    ocout: np.ndarray   # (n_outputs,)          which output of it; always 0 in CGP

    def copy(self) -> "Genotype":
        return Genotype(self.func.copy(), self.ntype.copy(), self.conn.copy(),
                        self.cout.copy(), self.ogene.copy(), self.ocout.copy())

    @property
    def n_nodes(self) -> int:
        return int(self.func.shape[0])

    @property
    def arity(self) -> int:
        return int(self.conn.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.ogene.shape[0])


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

def random_genotype(rng: np.random.Generator, n_nodes: int, n_in: int,
                    n_outputs: int, n_funcs: int, arity: int) -> Genotype:
    """A uniformly random valid genotype.

    Output genes are initialised to the last `n_outputs` nodes, per the paper:
    "These integers are initially chosen so that the program outputs are given by
    the outputs of the last O nodes in the genotype."
    """
    func = rng.integers(0, n_funcs, size=n_nodes, dtype=np.int32)
    ntype = np.zeros(n_nodes, dtype=np.int32)
    conn = np.zeros((n_nodes, arity), dtype=np.int32)
    for j in range(n_nodes):
        # node j is labelled n_in + j and may reference any label below that
        conn[j] = rng.integers(0, n_in + j, size=arity, dtype=np.int32)
    cout = np.zeros((n_nodes, arity), dtype=np.int32)
    ogene = np.arange(n_in + n_nodes - n_outputs, n_in + n_nodes, dtype=np.int32)
    ocout = np.zeros(n_outputs, dtype=np.int32)
    return Genotype(func, ntype, conn, cout, ogene, ocout)


# ---------------------------------------------------------------------------
# mutation (spec section 6, "genotype point mutation")
# ---------------------------------------------------------------------------

def mutate(g: Genotype, rng: np.random.Generator, n_mut: int, n_in: int,
           n_funcs: int) -> Genotype:
    """Return a mutated copy with exactly `n_mut` gene slots resampled.

    Table II gives "genotype point mutation probability 1" -- the operator is
    applied every time, mutating a fixed *count* of slots (3% = 9), not each slot
    with probability 0.03.

    A resampled gene is drawn uniformly from its legal range and may land on its
    current value; the paper does not require a mutation to change anything, and
    forcing a change would alter the neutral-drift behaviour the (1+4) ES depends
    on.
    """
    out = g.copy()
    n, a, n_out = g.n_nodes, g.arity, g.n_outputs

    # Flat slot indexing: [0, n)          -> function gene of node j
    #                     [n, n + n*a)    -> input gene (j, k)
    #                     [n + n*a, ...)  -> output gene o
    total = n_gene_slots(n, a, n_out)
    slots = rng.choice(total, size=min(n_mut, total), replace=False)

    for s in slots.tolist():
        if s < n:
            out.func[s] = rng.integers(0, n_funcs)
        elif s < n + n * a:
            t = s - n
            j, k = divmod(t, a)
            # any previous node or program input
            out.conn[j, k] = rng.integers(0, n_in + j)
            # the paired output-index gene is mutated at the same time -- "Both of
            # these are mutated at the same time, to ensure that every connection
            # in the graph is still valid". Single-output nodes in CGP => always 0.
            out.cout[j, k] = 0
        else:
            o = s - (n + n * a)
            out.ogene[o] = rng.integers(0, n_in + n)
            out.ocout[o] = 0
    return out


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def active_nodes(g: Genotype, n_in: int, gates: Sequence[Gate]) -> list[int]:
    """Node indices (0-based, not labels) reachable from the program outputs.

    The genotype-phenotype map leaves the rest inactive -- CGP's neutrality. Only
    the arity actually used by a node's gate is followed, so surplus input genes
    do not drag nodes into the phenotype.
    """
    seen: set[int] = set()
    stack = [int(lbl) for lbl in g.ogene.tolist()]
    while stack:
        lbl = stack.pop()
        if lbl < n_in:
            continue                      # a program input, not a node
        j = lbl - n_in
        if j in seen:
            continue
        seen.add(j)
        for k in range(gates[int(g.func[j])].arity):
            stack.append(int(g.conn[j, k]))
    return sorted(seen)


def evaluate(g: Genotype, gates: Sequence[Gate], in_masks: Sequence[int],
             mask: int, n_in: int) -> list[int]:
    """Output truth-table masks, one per program output. Bit-parallel over patterns.

    Every node is computed, not just the active ones: at 100 nodes the bookkeeping
    to skip inactive ones costs more than the bitwise ops it saves.
    """
    vals = list(in_masks) + [0] * g.n_nodes
    for j in range(g.n_nodes):
        gate = gates[int(g.func[j])]
        args = [vals[int(g.conn[j, k])] for k in range(gate.arity)]
        vals[n_in + j] = gate.fn(args, mask)
    return [vals[int(lbl)] for lbl in g.ogene.tolist()]


def evaluate_slow(g: Genotype, gates: Sequence[Gate], n_in: int,
                  n_patterns: int) -> list[list[bool]]:
    """Reference implementation: one pattern at a time, plain bools.

    Exists purely so `test_cgp.py` can assert the fast path agrees with it. Never
    used in the search loop.
    """
    outs: list[list[bool]] = [[] for _ in range(g.n_outputs)]
    for r in range(n_patterns):
        # pattern r's bit for input i. Must match tasks.input_masks' packing: that
        # takes column i of itertools.product([0,1], repeat=n_in), whose first
        # element varies slowest, so input i is bit (n_in-1-i) of r.
        vals = [bool((r >> (n_in - 1 - i)) & 1) for i in range(n_in)]
        vals = vals + [False] * g.n_nodes
        for j in range(g.n_nodes):
            gate = gates[int(g.func[j])]
            args = [vals[int(g.conn[j, k])] for k in range(gate.arity)]
            vals[n_in + j] = bool(gate.slow(args))
        for o, lbl in enumerate(g.ogene.tolist()):
            outs[o].append(bool(vals[int(lbl)]))
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
