"""Draw the circuits a PyPy run saved. CPython only -- it needs matplotlib.

    conda run -n lndp python render.py runs/<run-dir>
    conda run -n lndp python render.py runs/<run-dir> --seeds 0,3 --colour-cones

The search runs under PyPy, where matplotlib does not exist, so `--viz` is refused
there (`train.py::_require_viz`). Every run still writes `--save-best` circuits as
JSON, and this replays them into the same diagrams `--viz` would have produced. The
split is deliberate: drawing is a few seconds at the END of a run, so paying for it in
a second process costs nothing, whereas keeping matplotlib importable would have cost
the whole PyPy speed-up.

WHAT IT CANNOT RECOVER. Per-generation frames and the stage grid need snapshots taken
DURING the run, which a headless run does not keep. This draws the best circuit of
each seed. If the animation is what you are after, run that seed under CPython with
`--viz` -- it is one seed, and the run is reproducible from its seed by construction
(`test_equivalence.py`).
"""

from __future__ import annotations

import json
import pathlib
import sys

import cgp
import gates as gates_mod
import tasks as tasks_mod
import visualize as viz_mod


def _load(path: pathlib.Path) -> cgp.Genotype:
    d = json.loads(path.read_text(encoding="utf-8"))
    return cgp.Genotype(func=d["func"], ntype=d["ntype"], conn=d["conn"],
                        cout=d["cout"], ogene=d["ogene"], ocout=d["ocout"],
                        arity=d["arity"])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    run_dir = pathlib.Path(argv[0])
    colour_cones = "--colour-cones" in argv
    wanted = None
    if "--seeds" in argv:
        wanted = {int(s) for s in argv[argv.index("--seeds") + 1].split(",")}

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json in {run_dir} -- is that a run directory?")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    task = cfg["task"]
    gate_set = gates_mod.build_set(cfg["gates"])
    n_in = tasks_mod.n_inputs(task)
    n_patterns = tasks_mod.n_patterns(task)
    n_out = tasks_mod.n_outputs(task)
    total_hits = cgp.max_hits(n_out, n_patterns)
    groups = tasks_mod.input_groups(task)
    in_masks = tasks_mod.input_masks(task)
    mask = tasks_mod.full_mask(task)
    targets = tasks_mod.target_masks(task, cfg["operation"])
    viz_split = min(groups[1]) if len(groups) == 2 else None

    saved = sorted(run_dir.glob("*_best.json"))
    if not saved:
        raise SystemExit(f"no *_best.json in {run_dir} -- the run needed --save-best")

    frames = run_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for path in saved:
        seed = int(path.name.split("_seed")[1].split("_")[0])
        if wanted is not None and seed not in wanted:
            continue
        g = _load(path)
        pheno = cgp.phenotype(g, n_in, gate_set, groups)
        outs = cgp.evaluate(g, gate_set, in_masks, mask, n_in)
        hits = cgp.hits(outs, targets, mask)
        # The behavioural readout goes in the caption, because it is the one number a
        # picture of the wiring cannot show: a node can be drawn connected and have no
        # influence at all on the output it feeds.
        beh = cgp.behavioural_classes(outs, in_masks, mask, n_in, groups)
        pure = sum(1 for c in beh if c not in ("mixed", "const"))
        title = viz_mod.frame_title(
            cfg["generations"], cfg["operation"], hits, total_hits, pheno, seed,
            alg="ECGP" if cfg["ecgp"] else "CGP",
            mods=f"{pure}/{n_out} outputs behaviourally pure")
        out_png = frames / f"seed{seed}_best.png"
        viz_mod.draw(g, pheno, gate_set, n_in, out_png, title=title,
                     split=viz_split, colour_cones=colour_cones)
        print(f"  seed {seed}: {hits}/{total_hits} hits, {pheno.n_active} active, "
              f"{pure}/{n_out} outputs behaviourally pure -> {out_png}")
        drawn += 1
    print(f"\n  {drawn} circuit(s) drawn -> {frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
