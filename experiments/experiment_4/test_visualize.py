"""Tests for the fake-module colouring in `visualize._box_style`/`_render_modular`.
Run: conda run -n lndp python test_visualize.py

`_box_style` is the one place the "grey out a trivial module" rule lives (see
`visualize.py`'s FAKE MODULES note); everything else -- `_render_modular`,
`draw_modular`, `stage_grid` -- just calls it. So the rule is tested directly here,
plus one end-to-end smoke test that a circuit mixing both kinds of module still
renders to a real PNG.
"""

from __future__ import annotations

import os
import tempfile

import ecgp
import gates as gates_mod
import visualize

N_IN = 4
GATE_SET = gates_mod.build_set("nand")
N_PRIM = len(GATE_SET)
NA = 0                                # the only primitive in a NAND-only set


def _pair_of_modules() -> tuple[ecgp.Module, ecgp.Module]:
    """(trivial, real): one active NAND vs. two chained, active NANDs.

    Same two bodies `test_ecgp.py::test_is_trivial_module_hand_built_cases` hand-
    builds, so the fixture is known-good independent of this file.
    """
    trivial = ecgp.Module(mid=N_PRIM, n_in=2, func=[NA, NA],
                          conn=[0, 1, 0, 0], out=[2])             # 1 active node
    real = ecgp.Module(mid=N_PRIM + 1, n_in=2, func=[NA, NA],
                       conn=[0, 1, 0, 2], out=[2, 3])              # 2 chained, active
    assert ecgp.is_trivial_module(trivial)
    assert not ecgp.is_trivial_module(real)
    return trivial, real


def _individual_calling(trivial: ecgp.Module, real: ecgp.Module) -> ecgp.Individual:
    """Two top-level nodes, one calling each module; both program outputs live."""
    modules = {trivial.mid: trivial, real.mid: real}
    ind = ecgp.Individual(
        func=[trivial.mid, real.mid], ntype=[1, 1],
        conn=[[0, 1], [0, 1]], cout=[[0, 0], [0, 0]],
        ogene=[N_IN, N_IN + 1], ocout=[0, 0],
        modules=modules, next_id=max(modules) + 1)
    ecgp.validate(ind, N_IN, N_PRIM, 5)
    return ind


def test_box_style_greys_trivial_and_colours_real_modules() -> None:
    trivial, real = _pair_of_modules()
    ind = _individual_calling(trivial, real)
    mod_colour: dict[str, str] = {}

    name0, col0 = visualize._box_style(ind, 0, GATE_SET, N_PRIM, mod_colour)
    name1, col1 = visualize._box_style(ind, 1, GATE_SET, N_PRIM, mod_colour)

    assert col0 == visualize.NEUTRAL, "trivial module was not drawn grey"
    assert col1 != visualize.NEUTRAL, "real module was drawn grey"
    assert col1 in visualize.MODULE_COLOURS, "real module's colour is not from the cycle"
    assert mod_colour.get(name0) is None, \
        "a trivial module must not consume a MODULE_COLOURS slot"
    assert mod_colour[name1] == col1, "real module's colour was not recorded"
    print(f"ok  _box_style: trivial module ({name0}) grey, real module ({name1}) "
          f"coloured {col1}")


def test_draw_modular_renders_a_mixed_individual_without_error() -> None:
    """End-to-end smoke test: a circuit with both kinds of module must still draw."""
    trivial, real = _pair_of_modules()
    ind = _individual_calling(trivial, real)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "smoke.png")
        out = visualize.draw_modular(ind, N_IN, GATE_SET, N_PRIM, path)
        assert os.path.exists(out) and os.path.getsize(out) > 0, \
            "draw_modular did not produce a non-empty PNG"
    print("ok  draw_modular renders a mixed trivial/real-module circuit")


def test_box_style_greys_independent_parallel_gates_too() -> None:
    """A module with 2+ active gates that never interact (each reads straight off
    the module's own inputs, none reads another) must ALSO be drawn grey -- this is
    exactly the case `is_trivial_module` does not catch but `is_fake_module` does
    (see `test_ecgp.py::test_is_fake_module_hand_built_cases`). Caught in production
    on seed 0's M1294: two independent NANDs, `is_trivial_module` said real,
    `is_fake_module` correctly says fake.
    """
    parallel = ecgp.Module(mid=N_PRIM + 2, n_in=4, func=[NA, NA],
                           conn=[0, 3, 1, 2], out=[4, 5])   # M1294's own shape
    assert not ecgp.is_trivial_module(parallel), "fixture must NOT be trivial"
    assert ecgp.is_fake_module(parallel), "fixture must be fake (no interaction)"

    real = _pair_of_modules()[1]
    modules = {parallel.mid: parallel, real.mid: real}
    ind = ecgp.Individual(
        func=[parallel.mid, real.mid], ntype=[1, 1],
        conn=[[0, 1, 2, 3], [0, 1]], cout=[[0, 0, 0, 0], [0, 0]],
        ogene=[N_IN, N_IN + 1], ocout=[0, 0],
        modules=modules, next_id=max(modules) + 1)
    ecgp.validate(ind, N_IN, N_PRIM, 5)
    mod_colour: dict[str, str] = {}

    name0, col0 = visualize._box_style(ind, 0, GATE_SET, N_PRIM, mod_colour)
    name1, col1 = visualize._box_style(ind, 1, GATE_SET, N_PRIM, mod_colour)

    assert col0 == visualize.NEUTRAL, \
        "independent-parallel-gates module was not drawn grey"
    assert col1 != visualize.NEUTRAL, "chained (real) module was drawn grey"
    assert mod_colour.get(name0) is None, \
        "a fake (non-interacting) module must not consume a MODULE_COLOURS slot"
    print(f"ok  _box_style: independent-parallel-gates module ({name0}) grey "
          f"despite not being `is_trivial_module`")


if __name__ == "__main__":
    test_box_style_greys_trivial_and_colours_real_modules()
    test_draw_modular_renders_a_mixed_individual_without_error()
    test_box_style_greys_independent_parallel_gates_too()
    print("\nall tests passed")
