"""Reproduce seed 0's solved nested-ECGP circuit and count its fake modules.

    python decompose_seed0.py

Replays `stage_run.run_to_solution` with the settings of `stage_run.py --seed 0`
(nest_decay 0.5, NAND only, `retina_ka2005`/xor, 100 nodes); the search is
deterministic, so this lands on the same genotype. Prints every module type reachable
in the active circuit (recursively through module bodies) with its call count and
whether it is fake (no two of its primitives are chained), and draws the circuit with
one panel per real module via `decompose.draw_decomposition`. Backs
`../RESULTS.md`, "necgp".
"""
from __future__ import annotations

import pathlib
import types

import decompose
import ecgp
import gates as gates_mod
import tasks as tasks_mod
from stage_run import run_to_solution

OUT_PNG = pathlib.Path(__file__).parent / "runs" / "decomposition_seed0.png"

SEED = 0
NEST_DECAY = 0.5
TASK = "retina_ka2005"
OPERATION = "xor"

ARGS = types.SimpleNamespace(
    nodes=100, popsize=5, max_generations=300_000, mutation_rate=0.03,
    compress_prob=0.1, expand_prob=0.2, module_point_prob=0.04,
    add_input_prob=0.01, remove_input_prob=0.02, add_output_prob=0.01,
    remove_output_prob=0.02, max_module_size=5, fitness="raw",
    log_interval=10_000,
)


def main() -> None:
    OUT_PNG.parent.mkdir(exist_ok=True)
    gate_set = gates_mod.build_set("nand")
    n_prim = len(gate_set)
    n_in = tasks_mod.n_inputs(TASK)
    in_masks = tasks_mod.input_masks(TASK)
    mask = tasks_mod.full_mask(TASK)
    target = tasks_mod.target_mask(TASK, OPERATION)
    n_patterns = tasks_mod.n_patterns(TASK)
    split = n_in // 2

    res = run_to_solution(SEED, NEST_DECAY, ARGS, gate_set, n_prim, n_in,
                          in_masks, mask, target, n_patterns, collect_stages=False)
    parent = res["parent"]
    assert res["p_hits"] == n_patterns, (res["p_hits"], n_patterns)
    print(f"reproduced: gen {res['gen']}  hits {res['p_hits']}/{n_patterns}  "
          f"solved_gen {res['solved_gen']}  modules alive {len(parent.modules)}")

    counts = decompose.used_module_counts(parent, n_in)
    n_real = sum(1 for m in counts
                if not ecgp.is_fake_module(parent.modules[m], parent.modules))
    print(f"{len(counts)} distinct module types reachable "
          f"({n_real} real, {len(counts) - n_real} fake):")
    for m in sorted(counts, key=lambda m: -counts[m]):
        fake = ecgp.is_fake_module(parent.modules[m], parent.modules)
        kind = "fake -- greyed, no panel" if fake else "real"
        mod = parent.modules[m]
        print(f"  {ecgp.module_name(m, n_prim):>6s}  x{counts[m]:<3d}  "
              f"depth={mod.depth}  {kind}")

    out = decompose.draw_decomposition(
        parent, n_in, gate_set, n_prim, OUT_PNG, split=split,
        circuit_title=f"final circuit (unflattened)  |  seed {SEED}  |  100.00% (256/256)",
        caption_prefix="necgp (nested ECGP)  |  NAND-only FG xor  |  seed 0 final "
                       "circuit, decomposed  |  ")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
