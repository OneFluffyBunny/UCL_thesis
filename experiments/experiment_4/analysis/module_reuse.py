"""Do ECGP modules get reused, at the final logged genotype of each seed?

    python analysis/module_reuse.py        # from experiment_4/

Reads each seed's `*_gates.csv`. Per module at the last log point: `copies` is the
number of genotype nodes calling it (active or not), `count` the number of those
calls in the active circuit. `copies > 1` is cheap (a function-gene mutation can
point any node at an existing module); `count > 1` means the module is really used
more than once. Backs RESULTS.md, "ECGP under MVG".
"""
from __future__ import annotations

import csv
import glob
import pathlib
import statistics

EXP4 = pathlib.Path(__file__).resolve().parents[1]
RUNS = [
    ("ECGP FG  50n", "runs/_fg50cmp/ecgp_retina_ka2005_fg-and_n50_m0.03_g300000_cmp"),
    ("ECGP MVG 50n", "runs/ecgp_retina_ka2005_mvg-and-or_n50_m0.03_g800000_mvg50"),
    ("ECGP MVG 400n", "runs/ecgp_retina_ka2005_mvg-and-or_n400_m0.03_g800000_mvg400"),
]


def final_module_rows(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    last = max(int(r["gen"]) for r in rows)
    return [r for r in rows if r["kind"] != "prim" and int(r["gen"]) == last]


def analyze(label: str, run_dir: pathlib.Path) -> None:
    results = sorted(glob.glob(str(run_dir / "*_result.json")))
    if not results:
        print(f"{label}: no runs at {run_dir}")
        return
    prefix = pathlib.Path(results[0]).name.split("_seed")[0]
    n_mod, reused, active2, active1 = [], [], [], []
    for r in results:
        seed = int(r.split("_seed")[1].split("_")[0])
        mods = final_module_rows(run_dir / f"{prefix}_seed{seed}_gates.csv")
        if not mods:
            continue
        copies = [int(m["copies"]) for m in mods]
        counts = [int(m["count"]) for m in mods]
        n_mod.append(len(mods))
        reused.append(sum(c > 1 for c in copies) / len(mods))
        active2.append(sum(c > 1 for c in counts) / len(mods))
        active1.append(sum(c >= 1 for c in counts) / len(mods))
    med = statistics.median
    print(f"{label}: modules/seed {med(n_mod):.0f} | copies>1 {med(reused):.3f} | "
          f"count>1 {med(active2):.3f} | count>=1 {med(active1):.3f}  (medians over seeds)")


if __name__ == "__main__":
    for label, rel in RUNS:
        analyze(label, EXP4 / rel)
