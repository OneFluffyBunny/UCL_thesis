"""Tests for the fake-module colouring in necgp's `visualize._box_style`.
Run: conda run -n lndp python test_visualize.py

Same rule as `experiment_4/test_visualize.py`, generalised to nesting: a module
whose recursively-flattened body is a single primitive gate is drawn grey
(`visualize.NEUTRAL`), whatever nesting depth it took to build, and does not
consume a MODULE_COLOURS slot.
"""

from __future__ import annotations

import os
import tempfile

import ecgp
import gates as gates_mod
import visualize

N_IN = 4
GATE_SET = gates_mod.build_set("and,nand,or,nor")
N_PRIM = len(GATE_SET)


def _trivial_and_nested_real_modules() -> tuple[ecgp.Module, ecgp.Module]:
    """(A, C): A is a bare single-gate module; C nests A and adds one real gate.

    Same fixture `test_necgp.py::test_is_trivial_module_hand_built_nested_case`
    builds, so this file's expectations are known-good independent of it.
    """
    A = ecgp.Module(mid=N_PRIM, n_in=2, func=[0, 0], ntype=[0, 0],
                    conn=[[0, 1], [0, 0]], cout=[[0, 0], [0, 0]],
                    out=[2], ocout=[0], depth=1)
    C = ecgp.Module(mid=N_PRIM + 1, n_in=2, func=[A.mid, 0], ntype=[2, 0],
                    conn=[[0, 1], [2, 1]], cout=[[0, 0], [0, 0]],
                    out=[3], ocout=[0], depth=2)
    modules = {A.mid: A, C.mid: C}
    assert ecgp.is_trivial_module(A, modules)
    assert not ecgp.is_trivial_module(C, modules)
    return A, C


def _individual_calling(A: ecgp.Module, C: ecgp.Module) -> ecgp.Individual:
    """Two top-level nodes: one calls the trivial module A directly, the other
    calls the real (nested) module C. Both program outputs live."""
    modules = {A.mid: A, C.mid: C}
    ind = ecgp.Individual(
        func=[A.mid, C.mid], ntype=[1, 1],
        conn=[[0, 1], [0, 1]], cout=[[0, 0], [0, 0]],
        ogene=[N_IN, N_IN + 1], ocout=[0, 0],
        modules=modules, next_id=max(modules) + 1)
    ecgp.validate(ind, N_IN, N_PRIM, 5)
    return ind


def test_box_style_greys_trivial_and_colours_real_nested_modules() -> None:
    A, C = _trivial_and_nested_real_modules()
    ind = _individual_calling(A, C)
    mod_colour: dict[str, str] = {}

    name0, col0 = visualize._box_style(ind, 0, GATE_SET, N_PRIM, mod_colour)
    name1, col1 = visualize._box_style(ind, 1, GATE_SET, N_PRIM, mod_colour)

    assert col0 == visualize.NEUTRAL, "trivial module was not drawn grey"
    assert col1 != visualize.NEUTRAL, "real nested module was drawn grey"
    assert col1 in visualize.MODULE_COLOURS, "real module's colour is not from the cycle"
    assert mod_colour.get(name0) is None, \
        "a trivial module must not consume a MODULE_COLOURS slot"
    assert mod_colour[name1] == col1, "real module's colour was not recorded"
    print(f"ok  _box_style: trivial module ({name0}) grey, real nested module "
          f"({name1}) coloured {col1}")


def test_draw_modular_renders_a_mixed_individual_without_error() -> None:
    """End-to-end smoke test: a circuit mixing a trivial and a real nested module
    must still draw, `|x` nesting-factor suffix included."""
    A, C = _trivial_and_nested_real_modules()
    ind = _individual_calling(A, C)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "smoke.png")
        out = visualize.draw_modular(ind, N_IN, GATE_SET, N_PRIM, path)
        assert os.path.exists(out) and os.path.getsize(out) > 0, \
            "draw_modular did not produce a non-empty PNG"
    print("ok  draw_modular renders a mixed trivial/real-nested-module circuit")


def test_box_style_greys_independent_parallel_gates_too() -> None:
    """A module with 2+ active primitives (possibly via nesting) that never
    interact -- each reads straight off the module's own inputs, none reads
    another -- must ALSO be drawn grey. This is the case `is_trivial_module`
    does not catch but `is_fake_module` does (see `test_necgp.py`'s
    `test_is_fake_module_hand_built_nested_case`): a plain gate plus an
    independent nested call to a trivial module.
    """
    A = ecgp.Module(mid=N_PRIM, n_in=2, func=[0, 0], ntype=[0, 0],
                    conn=[[0, 1], [0, 0]], cout=[[0, 0], [0, 0]],
                    out=[2], ocout=[0], depth=1)
    D = ecgp.Module(mid=N_PRIM + 2, n_in=2, func=[0, A.mid], ntype=[0, 2],
                    conn=[[0, 1], [0, 1]], cout=[[0, 0], [0, 0]],
                    out=[2, 3], ocout=[0, 0], depth=2)
    _, C = _trivial_and_nested_real_modules()
    modules = {A.mid: A, D.mid: D, C.mid: C}
    assert not ecgp.is_trivial_module(D, modules), "fixture must NOT be trivial (2 active)"
    assert ecgp.is_fake_module(D, modules), "fixture must be fake (no interaction)"

    ind = ecgp.Individual(
        func=[D.mid, C.mid], ntype=[1, 1],
        conn=[[0, 1], [0, 1]], cout=[[0, 0], [0, 0]],
        ogene=[N_IN, N_IN + 1], ocout=[0, 0],
        modules=modules, next_id=max(modules) + 1)
    ecgp.validate(ind, N_IN, N_PRIM, 5)
    mod_colour: dict[str, str] = {}

    name0, col0 = visualize._box_style(ind, 0, GATE_SET, N_PRIM, mod_colour)
    name1, col1 = visualize._box_style(ind, 1, GATE_SET, N_PRIM, mod_colour)

    assert col0 == visualize.NEUTRAL, \
        "independent-parallel-gates (nested) module was not drawn grey"
    assert col1 != visualize.NEUTRAL, "chained (real) nested module was drawn grey"
    assert mod_colour.get(name0) is None, \
        "a fake (non-interacting) module must not consume a MODULE_COLOURS slot"
    print(f"ok  _box_style: independent-parallel-gates nested module ({name0}) grey "
          f"despite not being `is_trivial_module`")


if __name__ == "__main__":
    test_box_style_greys_trivial_and_colours_real_nested_modules()
    test_draw_modular_renders_a_mixed_individual_without_error()
    test_box_style_greys_independent_parallel_gates_too()
    print("\nall tests passed")
