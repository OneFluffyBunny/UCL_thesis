# kashtan_alon — RESULTS (lab notebook)

The run record for the Kashtan–Alon retina MVG→modularity reproduction. Reference
docs (`README.md`, `PAPER_SPEC.md`) describe *what* the setup is; this file records
*what happened* when we ran it. Newest runs first.

**The claim under test (KA 2005):** under Modularly-Varying Goals (G_AND ↔ G_OR
every 20 gens) Newman **Q** rises and *stays high* (~0.4+); under a Fixed Goal it
stays low (~0.15–0.2). Same network, same search, only the goal schedule differs.

---

## Run 1 — full paper preset, 5 seeds (2026-07-31) → **near-null: weak separation**

- **Command:** `conda run -n lndp python run_paper.py --n-seeds 5 --viz`
- **Where:** laptop (CPU, pure-numpy, 0 VRAM) · ~30–34 ms/gen · full run ≈ 2.3 h
- **Config (locked to paper):** net 8-8-4-2-1, tanh λ=20, weights {−2,−1,1,2},
  biases {−2..2}, pop 1000, 25000 gens, mutation-only GA (add/remove 20%, weight
  ±1 @ 2/n, bias ±1 @ 1/24), tournament k=3 + 1 elite, **raw** fraction-correct
  fitness. Two conditions: **MVG** (and↔or/20) vs **FG(L AND R)**.
- **Logs:** `runs/retina_{mvg,fg}_raw_seed{0..4}_log.csv` (per-gen Q/density/fitness),
  `runs/*_best.png` (final brains, module-coloured), `runs/*_result.json` (per-seed
  summary). NB: `runs/paper_run.log` (console) was buffered by `conda run` and only
  flushed at exit — use `--no-capture-output` next time for a live console stream.

### Result — the paper's separation did NOT reproduce

**Best-net Q** (saved best-fitness individual, from `result.json`):

| Condition | mean Q | Q per seed | mean best fit |
|---|---|---|---|
| MVG (and↔or) | **0.309** | 0.339, 0.296, 0.298, 0.314, 0.299 | 1.000 |
| FG (L AND R) | **0.279** | 0.308, 0.296, 0.271, 0.219, 0.302 | 1.000 |

**Per-generation Q** (fairer — best net *each* logged gen, over the whole run):

| Condition | mean (all gens) | mean (2nd half) | max ever | last gen / seed |
|---|---|---|---|---|
| MVG | 0.335 | 0.339 | 0.549 | 25000 (all seeds) |
| FG  | 0.248 | 0.254 | 0.376 | 480, 220, 8090, 7490, 490 |

**Verdict vs paper:** ❌ **Not reproduced at the paper's magnitude.** Paper claims
MVG ~0.4+ vs FG ~0.15–0.2. We get MVG ≈ 0.31–0.34 vs FG ≈ 0.25–0.28 — *directionally*
correct (MVG more modular) but only a ~0.05–0.09 gap, and both well inside each
other's range. Every seed solved the task (fit 1.000).

### Confounds / caveats (read before trusting this)

- **FG early-stops, MVG doesn't** — the biggest one. FG hits fit 1.0 → `--target 1.0`
  early-stop fires (3/5 seeds before **gen 500**; others ~7500–8000), while MVG runs
  the full 25000 gens (its goal keeps moving, so it never early-stops). So FG is
  scored on a barely-evolved first solution and MVG on a long-evolved one — **not
  matched on evolutionary time.** *Fix for the next run:* `--target 1.1` (or a
  no-early-stop flag) so FG also evolves the full 25000 gens. Even handicapped-short,
  though, FG's separation from MVG is already weak.
- **Object patterns ≠ KA's.** We use `L=(p0∧p1)∨(p2∧p3)`, `R=(p4∧p5)∨(p6∧p7)`; KA's
  exact retina bit-patterns are unpublished. The MVG effect is known to be sensitive
  to task details.
- **Consistent with the follow-up literature.** Clune, Mouret & Lipson 2013 found MVG
  *alone* is an unreliable modularity driver without a connection-cost term — a weak
  separation here is not surprising. Adding a connection-cost variant is the obvious
  next experiment (a *variation*, not the paper).
- **Best-net Q ≈ per-gen Q** here (0.31 vs 0.34 MVG; 0.28 vs 0.25 FG), so the
  "save the best-*fitness* net, not the most-modular one" choice is not what's
  suppressing the signal — the signal is genuinely weak.

### Next steps (proposed, not yet run)
1. Re-run with FG early-stop disabled (`--target 1.1`) — the fair apples-to-apples.
2. If still weak: try a connection-cost variant (Clune's driver) and/or `xor` goal.
3. Consider the exact KA goal pair / object set if we can recover it.

---

## Conventions for this file
- One `## Run N` block per run, newest first; keep the exact command and the config
  that mattered so it's reproducible.
- Record the **Q separation** (the headline), not just final fitness.
- `runs/` is gitignored (regenerable) — conclusions live here.
