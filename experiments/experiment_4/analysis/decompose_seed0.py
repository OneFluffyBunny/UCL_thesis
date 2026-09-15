"""Reproduce seed 0's final ECGP individual (nandfg run) and draw its decomposition
via `decompose.draw_decomposition` (the reusable, non-scratch renderer).

The checkpoint for a finished seed is deleted on completion (see train.py), so the
`Individual` object is gone; it is reconstructed by replaying the exact same
deterministic (1+4) ES with `random.Random(0)`, which consumes the RNG in the same
order run_seed does (see train.run_seed) and so lands on a bit-identical genotype.
Verified against runs/ecgp_retina_ka2005_fg-and_n100_m0.03_g300000_nandfg's own
seed0_result.json before drawing anything.

Run from experiment_4/: python analysis/decompose_seed0.py
"""

from __future__ import annotations

import json
import sys
import pathlib
import random

EXP4 = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP4))

import decompose
import ecgp
import gates as gates_mod
import tasks as tasks_mod
import cgp

RUN_DIR = EXP4 / pathlib.Path("runs/ecgp_retina_ka2005_fg-and_n100_m0.03_g300000_nandfg")
RESULT = RUN_DIR / "ecgp_retina_ka2005_fg-and_n100_m0.03_seed0_result.json"
OUT_PNG = RUN_DIR / "seed0_decomposition.png"

SEED = 0
N_NODES = 100
N_IN = tasks_mod.n_inputs("retina_ka2005")
GATE_SET = gates_mod.build_set("nand")
N_FUNCS = len(GATE_SET)
TARGET = tasks_mod.target_mask("retina_ka2005", "and")
MASK = tasks_mod.full_mask("retina_ka2005")
IN_MASKS = tasks_mod.input_masks("retina_ka2005")
N_PATTERNS = tasks_mod.n_patterns("retina_ka2005")
SPLIT = N_IN // 2

EPARAMS = ecgp.Params(compress=0.1, expand=0.2, module_point=0.04,
                      add_input=0.01, remove_input=0.02,
                      add_output=0.01, remove_output=0.02,
                      max_module_size=5, mutation_rate=0.03)


def replay_final_individual() -> ecgp.Individual:
    """Exactly train.run_seed's (1+4) loop, minus logging/IO. Same RNG order."""
    rnd = random.Random(SEED)

    def new():
        return ecgp.random_individual(rnd, N_NODES, N_IN, 1, N_FUNCS)

    def offspring(par):
        return ecgp.mutate(par, rnd, N_IN, N_FUNCS, EPARAMS)

    def score(g):
        return ecgp.fitness(g, GATE_SET, IN_MASKS, TARGET, MASK, N_IN, "raw")

    pop = [new() for _ in range(5)]
    scored = [score(g) for g in pop]
    i = max(range(len(scored)), key=lambda k: scored[k][0])
    parent, (p_score, p_hits) = pop[i], scored[i]
    solved_gen = -1

    for gen in range(300_000):
        kids = [offspring(parent) for _ in range(4)]
        o_scored = [score(g) for g in kids]
        o_scores = [s for s, _ in o_scored]
        top = max(o_scores)
        promoted = False
        if top > p_score:
            i = o_scores.index(top)
            parent, (p_score, p_hits) = kids[i], o_scored[i]
            promoted = True
        elif top == p_score:
            ties = [i for i, s in enumerate(o_scores) if s == top]
            i = rnd.choice(ties)
            parent, (p_score, p_hits) = kids[i], o_scored[i]
            promoted = True
        if promoted:
            ecgp.prune_modules(parent)
        if solved_gen < 0 and p_hits == N_PATTERNS:
            solved_gen = gen
        if p_hits == N_PATTERNS:
            break

    print(f"replayed: gen {gen}  hits {p_hits}/{N_PATTERNS}  solved_gen {solved_gen}")
    return parent


def main() -> None:
    parent = replay_final_individual()

    expected = json.loads(RESULT.read_text(encoding="utf-8"))
    view = ecgp.flatten(parent, N_IN)
    pheno = cgp.phenotype(view, N_IN, GATE_SET, SPLIT)
    assert pheno.n_active == expected["active_nodes"], \
        (pheno.n_active, expected["active_nodes"])
    assert len(parent.modules) == expected["n_modules"]
    assert sum(1 for t in parent.ntype if t) == expected["module_nodes"]
    print("reproduction matches seed0_result.json -- ", expected["gates"])

    counts = decompose.used_module_counts(parent, N_IN)
    n_real = sum(1 for m in counts if not ecgp.is_fake_module(parent.modules[m]))
    print(f"{len(counts)} distinct module types used ({n_real} real, "
          f"{len(counts) - n_real} fake):")
    for m in sorted(counts, key=lambda m: -counts[m]):
        fake = ecgp.is_fake_module(parent.modules[m])
        kind = "fake -- greyed on the left, no panel" if fake else "real"
        print(f"  {ecgp.module_name(m, N_FUNCS):>6s}  x{counts[m]:<3d} {kind}")

    out = decompose.draw_decomposition(
        parent, N_IN, GATE_SET, N_FUNCS, OUT_PNG, split=SPLIT,
        circuit_title="final circuit (unflattened)  |  seed 0  |  100.00% (256/256)",
        caption_prefix="ECGP  |  NAND-only FG  |  seed 0 final circuit, decomposed  |  ")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
