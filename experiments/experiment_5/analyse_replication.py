"""Phase 3.5 replication -- EXACTLY the two tests preregistered in SPECIALISATION.md.

⚠️ AI-AUTHORED sub-study. Read SPECIALISATION.md's banner.

    conda run -n lndp python analyse_replication.py

Deliberately a separate file from `analyse_specialisation.py`, and deliberately
hard-coded rather than looped over cells: the value of a replication is that its
analysis was fixed in advance, and a script that can only run two tests cannot quietly
grow a third. Seeds 1000-1199, disjoint from phase 3's 0-49.
"""

from __future__ import annotations

import json
import pathlib
import statistics

import numpy as np
from scipy.stats import mannwhitneyu

OUT = pathlib.Path(__file__).resolve().parent / "runs" / "_specrep"
ALPHA = 0.025            # Bonferroni over the two preregistered contrasts
TESTS = (("R1", "cgp4"), ("R2", "cgpnand"))


def arm(tag: str) -> list[float]:
    """SPEC for the solved seeds of one arm, from its run directory's `--tag`."""
    vals = []
    for d in OUT.glob(f"*_{tag}"):
        for f in sorted(d.glob("*_result.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["solved_gen"] >= 0 and r.get("spec") not in ("", None):
                vals.append(float(r["spec"]))
    return vals


def rank_biserial(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    u = (x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()
    return 2 * u / (len(x) * len(y)) - 1


def main() -> int:
    print("PHASE 3.5 REPLICATION -- seeds 1000-1199, alpha = 0.025 (Bonferroni over 2)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    verdicts = []
    for name, enc in TESTS:
        a = arm(f"rep-staged-partial-{enc}")
        b = arm(f"rep-cold-partial-{enc}")
        if not a or not b:
            print(f"{name} ({enc}): MISSING DATA (staged n={len(a)}, cold n={len(b)})")
            continue
        u, p = mannwhitneyu(a, b, alternative="greater")
        r = rank_biserial(a, b)
        boot = [rank_biserial(rng.choice(a, len(a)), rng.choice(b, len(b)))
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ok = p < ALPHA
        verdicts.append((name, enc, ok, p, r))
        print(f"\n{name}  staged > cold, overlap=partial, encoding={enc}")
        print(f"  staged  n={len(a):<4} median SPEC {statistics.median(a):.4f}")
        print(f"  cold    n={len(b):<4} median SPEC {statistics.median(b):.4f}")
        print(f"  median difference {statistics.median(a) - statistics.median(b):+.4f}")
        print(f"  U = {u:.1f}   one-sided p = {p:.5g}   rank-biserial r = {r:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]")
        print(f"  --> {'REJECT null' if ok else 'FAIL TO REJECT null'} at alpha={ALPHA}")

    print("\n" + "=" * 74)
    for name, enc, ok, p, r in verdicts:
        print(f"{name} ({enc:<8}): {'SUPPORTED' if ok else 'NOT SUPPORTED':<14} "
              f"p={p:.5g}  r={r:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
