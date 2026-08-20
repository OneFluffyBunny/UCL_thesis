"""A/B the `x ^ mask` complement rewrite against `~x & mask`, at equal effort.

Builds a copy of experiment 5 with the rewrite REVERTED, then drives both copies
through `bench_driver.py` at the same generation counts and repeat count. Comparing
across two separate `--quick` sweeps would have mixed best-of-1 noise into a ~30%
effect; this runs them back to back at best-of-3.
"""
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = pathlib.Path(sys.argv[1]) / "exp5_old_complement"
PYPY = HERE / ".venv-pypy" / ("Scripts/python.exe" if sys.platform == "win32"
                              else "bin/python")

REVERTS = {
    "cgp.py": [
        ("vals[n_in + j] = (x & vals[conn[b + 1]]) ^ mask",
         "vals[n_in + j] = ~(x & vals[conn[b + 1]]) & mask"),
        ("vals[n_in + j] = (x | vals[conn[b + 1]]) ^ mask",
         "vals[n_in + j] = ~(x | vals[conn[b + 1]]) & mask"),
        ("vals[n_in + j] = x ^ vals[conn[b + 1]] ^ mask",
         "vals[n_in + j] = ~(x ^ vals[conn[b + 1]]) & mask"),
        ("        t += n_pat - (o ^ tg).bit_count()",
         "        t += (~(o ^ tg) & mask).bit_count()"),
    ],
    "ecgp.py": [
        ("        return (x & y) ^ mask", "        return ~(x & y) & mask"),
        ("        return (x | y) ^ mask", "        return ~(x | y) & mask"),
        ("        return x ^ y ^ mask", "        return ~(x ^ y) & mask"),
    ],
}

shutil.rmtree(SCRATCH, ignore_errors=True)
SCRATCH.mkdir(parents=True)
for name in ("cgp.py", "ecgp.py", "gates.py", "tasks.py"):
    shutil.copy(HERE / name, SCRATCH / name)
for name, subs in REVERTS.items():
    p = SCRATCH / name
    s = p.read_text(encoding="utf-8")
    for old, new in subs:
        assert old in s, (name, old[:40])
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")

CASES = [(14, 4000), (15, 2500), (16, 1500), (17, 900), (18, 500)]
REPS = 3


def run(python, exp_dir, task, gens):
    r = subprocess.run([str(python), str(HERE / "bench_driver.py"), str(exp_dir),
                        "cgp", task, "200", str(gens), str(REPS)],
                       capture_output=True, text=True, cwd=str(HERE))
    r.check_returncode()
    return json.loads(r.stdout.strip().splitlines()[-1])


print("| inputs | CPython `~x & mask` | CPython `x ^ mask` | gain | "
      "PyPy `~x & mask` | PyPy `x ^ mask` | gain |")
print("|---|---|---|---|---|---|---|")
for n_in, gens in CASES:
    task = f"parity{n_in}"
    co = run(sys.executable, SCRATCH, task, gens)
    cn = run(sys.executable, HERE, task, gens)
    po = run(PYPY, SCRATCH, task, gens)
    pn = run(PYPY, HERE, task, gens)
    assert co["hits"] == cn["hits"] == po["hits"] == pn["hits"], "different searches!"
    print(f"| {n_in} | {co['ms_per_gen']:.3f} | {cn['ms_per_gen']:.3f} | "
          f"{co['ms_per_gen'] / cn['ms_per_gen']:.2f}x | "
          f"{po['ms_per_gen']:.3f} | {pn['ms_per_gen']:.3f} | "
          f"{po['ms_per_gen'] / pn['ms_per_gen']:.2f}x |")
shutil.rmtree(SCRATCH, ignore_errors=True)
