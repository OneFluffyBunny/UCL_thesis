"""Which 2-input, 1-output modules does NAND-only ECGP build, and are they kept?

    python analysis/nand_compositions.py        # from experiment_4/

Reads the per-log gate census (`*_gates.csv`) of the NAND-only ECGP run. Modules
labelled NAND, NOT, WIRE or a constant re-wrap what a single node already computes;
AND, OR, A|~B and A&~B need more than one NAND. Prints, per seed, how often each
label was active during the run and which were still active at the last log point.
Backs RESULTS.md, "NAND-only function set".
"""
from __future__ import annotations

import collections
import csv
import glob
import pathlib

RUN = (pathlib.Path(__file__).resolve().parents[1] / "runs"
       / "ecgp_retina_ka2005_fg-and_n100_m0.03_g300000_nandfg")
TRIVIAL = {"NAND", "NOT", "WIRE", "0", "1"}


def main() -> None:
    files = sorted(glob.glob(str(RUN / "*_gates.csv")),
                   key=lambda f: int(f.split("_seed")[1].split("_")[0]))
    if not files:
        raise SystemExit(f"no gate logs under {RUN}")
    uses = collections.Counter()
    kept = 0
    for f in files:
        seed = int(f.split("_seed")[1].split("_")[0])
        with open(f, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if r["kind"] != "prim" and r["ins"] == "2" and r["outs"] == "1"]
        active = collections.Counter(r["label"] for r in rows if int(r["count"]) > 0)
        last = max((int(r["gen"]) for r in rows), default=-1)
        final = sorted({r["label"] for r in rows
                        if int(r["gen"]) == last and int(r["count"]) > 0})
        genuine_final = [x for x in final if x not in TRIVIAL]
        kept += bool(genuine_final)
        uses.update({lab: 1 for lab in active if lab not in TRIVIAL})
        print(f"seed {seed:2d}: active during run {dict(active)}; active at end {final}")
    print(f"\nseeds that ever used a genuine composition: {dict(uses)} (of {len(files)})")
    print(f"seeds with a genuine composition still active at the end: {kept}/{len(files)}")


if __name__ == "__main__":
    main()
