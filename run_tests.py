"""Run every test suite in the repository and report pass/fail per file.

    python run_tests.py            # correctness tests
    python run_tests.py --perf     # also the timing tests (machine-dependent thresholds)

Each suite is a plain script run from its own directory, as its docstring says.
Use the `lndp` environment (see README.md). The NDP code has its own environment
and is not covered here.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent

SUITES = [
    "experiments/experiment_1/smoke_test.py",
    "experiments/experiment_1/test_ga.py",
    "experiments/experiment_4/test_cgp.py",
    "experiments/experiment_4/test_ecgp.py",
    "experiments/experiment_4/test_visualize.py",
    "experiments/experiment_5/test_tasks.py",
    "experiments/experiment_5/test_cgp.py",
    "experiments/experiment_5/test_ecgp.py",
    "experiments/experiment_5/test_visualize.py",
    "experiments/experiment_5/test_equivalence.py",
    "experiments/experiment_6/test_smcgp.py",
    "experiments/experiment_6/necgp/test_necgp.py",
    "experiments/experiment_6/necgp/test_visualize.py",
    "experiments/experiment_6/necgp_pairwise/test_tasks.py",
    "experiments/experiment_6/necgp_pairwise/test_train.py",
    "experiments/experiment_6/necgp_pairwise/test_visualize.py",
    "kashtan_alon/test_tasks.py",
]
# Throughput and parallel-efficiency thresholds tuned on one laptop; they can fail
# on a loaded or different machine without anything being wrong.
PERF_SUITES = [
    "experiments/experiment_4/test_perf.py",
    "experiments/experiment_5/test_perf.py",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf", action="store_true", help="also run the timing tests")
    args = ap.parse_args()

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    failed = []
    for rel in SUITES + (PERF_SUITES if args.perf else []):
        path = ROOT / rel
        t0 = time.time()
        r = subprocess.run([sys.executable, path.name], cwd=path.parent, env=env,
                           capture_output=True, text=True, encoding="utf-8")
        ok = r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {time.time() - t0:6.1f}s  {rel}", flush=True)
        if not ok:
            failed.append(rel)
            print("\n".join("      " + line for line in
                            (r.stdout + r.stderr).strip().splitlines()[-15:]))
    print(f"\n{len(failed)} failed" if failed else "\nall suites passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
