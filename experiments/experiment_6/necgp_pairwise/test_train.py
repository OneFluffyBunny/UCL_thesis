"""train.py must be run.py's search plus read-only bookkeeping -- nothing else.

Run: conda run -n lndp python test_train.py   (also passes under PyPy)
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile

import census
import ecgp
import gates as gates_mod
import run
import tasks as tasks_mod
import train

SEEDS = (0, 4)
GENS = 3000


def _args(out: str, **kw):
    a = train.build_parser().parse_args(["--max-generations", str(GENS), "--out", out,
                                         "--tag", "t", "--snapshot-interval", "250",
                                         "--print-interval", str(10 ** 9)])
    for k, v in kw.items():
        setattr(a, k, v)
    a.log_interval = 10 ** 9            # run.py's own flag, silenced
    return a


def _run_py(seed: int, args) -> dict:
    gs = gates_mod.build_set(args.gates)
    with contextlib.redirect_stdout(io.StringIO()):
        return run.run_to_solution(seed, args, gs, len(gs), tasks_mod.n_inputs(args.task),
                                   tasks_mod.input_masks(args.task),
                                   tasks_mod.full_mask(args.task),
                                   tasks_mod.target_mask(args.task, args.operation),
                                   tasks_mod.n_patterns(args.task))


def test_same_search_as_run_py_with_and_without_snapshots() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            args = _args(tmp)
            ref = _run_py(seed, args)
            with contextlib.redirect_stdout(io.StringIO()):
                off = train.run_seed(seed, args, snapshots=False)
                on = train.run_seed(seed, args, snapshots=True)
            want = census.to_json(ref["parent"])
            for label, got in (("snapshots off", off), ("snapshots on", on)):
                assert census.to_json(got["parent"]) == want, f"seed {seed}: {label} diverged"
                assert got["hits"] == ref["p_hits"] and got["gens"] == ref["gen"], seed
            print(f"ok  seed {seed}: train.py == run.py after {ref['gen']} gens "
                  f"(hits {ref['p_hits']}), snapshots on and off")


def test_snapshots_replay_and_census_is_consistent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args = _args(tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            train.run_seed(0, args)
        d = pathlib.Path(tmp) / "t" / "seed0"
        gs = gates_mod.build_set(args.gates)
        n_in = tasks_mod.n_inputs(args.task)
        masks, mask = tasks_mod.input_masks(args.task), tasks_mod.full_mask(args.task)
        target = tasks_mod.target_mask(args.task, args.operation)

        snaps = [json.loads(l) for l in (d / "snapshots.jsonl").read_text().splitlines()]
        logs = (d / "log.csv").read_text().splitlines()
        assert len(snaps) == len(logs) - 1, "one log row per snapshot"
        gens = [s["gen"] for s in snaps]
        assert gens == sorted(gens) and gens[0] == 0 and gens[-1] == GENS

        for s in snaps:
            ind = census.from_json(s["ind"])
            ecgp.validate(ind, n_in, len(gs))
            assert ecgp.fitness(ind, gs, masks, target, mask, n_in)[1] == s["hits"], \
                f"gen {s['gen']}: replayed genotype does not score its logged hits"

        import csv
        with open(d / "gates.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        by_gen: dict[str, list[dict]] = {}
        for r in rows:
            by_gen.setdefault(r["gen"], []).append(r)
        for g, rs in by_gen.items():
            top = sum(float(r["share_top"]) for r in rs)
            flat = sum(float(r["share_flat"]) for r in rs)
            assert abs(top - 1) < 1e-4 and abs(flat - 1) < 1e-4, (g, top, flat)
        n_mod_rows = sum(1 for r in rows if r["kind"] == "module")
        print(f"ok  {len(snaps)} snapshots replay to their logged hits; top-level and "
              f"flattened shares sum to 1 at all {len(by_gen)} ({n_mod_rows} module rows)")


def test_module_label_and_signature() -> None:
    gs = gates_mod.build_set("nand")
    # NAND(i0, i1) -> NAND(prev, prev) == AND
    m_and = ecgp.Module(mid=1, n_in=2, func=[0, 0], ntype=[0, 0],
                        conn=[[0, 1], [2, 2]], cout=[[0, 0], [0, 0]], out=[3], ocout=[0])
    # nest it: AND(i0, i1) -> NAND(prev, i2) == NAND3
    m_nand3 = ecgp.Module(mid=2, n_in=3, func=[1, 0], ntype=[2, 0],
                          conn=[[0, 1], [3, 2]], cout=[[0, 0], [0, 0]], out=[4], ocout=[0],
                          depth=2)
    mods = {1: m_and, 2: m_nand3}
    cen = census.Census(gs, 3, 1, 8)
    assert cen.info(1, mods)[0] == "AND", cen.info(1, mods)
    assert cen.info(2, mods)[0] == "NAND3", cen.info(2, mods)
    assert cen.info(2, mods)[3] == 3 and not cen.info(2, mods)[4]
    print(f"ok  labels: {cen.info(1, mods)[2]} = AND, {cen.info(2, mods)[2]} = NAND3")


PYPY = pathlib.Path(__file__).resolve().parents[2] / "experiment_5" / ".venv-pypy" / "Scripts" / "pypy.exe"
CPYTHON = pathlib.Path.home() / "miniconda3" / "envs" / "lndp" / "python.exe"
_PROBE = ("import contextlib,hashlib,io,json,sys,census,train;"
          "a=train.build_parser().parse_args(['--max-generations','3000']);"
          "f=io.StringIO();"
          "r=contextlib.redirect_stdout(f).__enter__();"
          "res=train.run_seed(int(sys.argv[1]),a,snapshots=False);"
          "sys.__stdout__.write(hashlib.sha1(json.dumps(census.to_json(res['parent']),"
          "sort_keys=True).encode()).hexdigest()+' '+str(res['hits']))")


def test_pypy_and_cpython_walk_the_same_search() -> None:
    import subprocess
    if not (PYPY.exists() and CPYTHON.exists()):
        print(f"  skip test_pypy_and_cpython_walk_the_same_search (need {PYPY} and {CPYTHON})")
        return
    here = pathlib.Path(__file__).resolve().parent
    for seed in SEEDS:
        outs = [subprocess.run([str(exe), "-c", _PROBE, str(seed)], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
                for exe in (CPYTHON, PYPY)]
        assert outs[0] == outs[1], f"seed {seed}: CPython {outs[0]} != PyPy {outs[1]}"
        print(f"ok  seed {seed}: PyPy == CPython after {GENS} gens ({outs[0]})")


if __name__ == "__main__":
    test_module_label_and_signature()
    test_same_search_as_run_py_with_and_without_snapshots()
    test_snapshots_replay_and_census_is_consistent()
    test_pypy_and_cpython_walk_the_same_search()
    print("\nall tests passed")
