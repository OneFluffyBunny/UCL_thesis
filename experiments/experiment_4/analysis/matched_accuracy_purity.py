"""(1+4) ES study: are FG and MVG circuits equally pure at the same accuracy?

    python analysis/matched_accuracy_purity.py [--root ../runs/fgmvg50]   # from analysis/

Takes every champion on the sparse archive grid (every 100 generations, so the
per-generation windows do not dominate), scores it on AND, keeps those with
0.81 <= acc(AND) <= 0.85 (the MVG plateau), and compares circuit purity and size
between the arms. Backs RESULTS.md, "FG vs MVG".
"""
from __future__ import annotations

import argparse
import pathlib
import statistics

import fgmvg_common as fc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[1]
                                          / "runs" / "fgmvg50"))
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--lo", type=float, default=0.81)
    ap.add_argument("--hi", type=float, default=0.85)
    args = ap.parse_args()
    for arm in fc.load_study(pathlib.Path(args.root)):
        pur, size = [], []
        for s in arm.seeds:
            for r in arm.archive(s):
                if r["gen"] % args.every:
                    continue
                g = arm.genotype(r)
                if args.lo <= arm.acc(g, "and") <= args.hi:
                    p, _, ph = arm.purity(g)
                    if p == p:
                        pur.append(p)
                        size.append(ph.n_active)
        print(f"{arm.name}: n = {len(pur)}, median purity {statistics.median(pur):.2f}, "
              f"median active gates {statistics.median(size):.0f}")


if __name__ == "__main__":
    main()
