# Modularity metrics across 5 FG / 5 MVG retina seeds (2026-09-11)

> ⚠️ **Not a budget-matched comparison (noted 2026-09-13).** The FG runs below ran
> 200 generations, the MVG runs 1500. The matched set (FG at 1500) is listed in
> `RESULTS.md` under "Budget-matched FG vs MVG seed set" and has not been scored yet.

Read-only analysis of the 10 completed runs logged in `RESULTS.md`'s "5-seed FG
vs MVG size/accuracy table" entry. No new training here -- everything below is
`grow_network()` on each run's saved `config.yml` + `solution_best.npy`, same
load path as `analysis.py`'s `load_run()`, fed through `qmetrics`.

Script: `experiments_paper/retina/modularity_analysis.py`

```
python experiments_paper/retina/modularity_analysis.py
```

which for each run does:

```python
G = from_matrix(W)   # see API-mismatch note below -- no `order` kwarg needed
Q, comms = newman_q(G)
score, extras = left_right_q(G, {i: i // 4 for i in range(8)}, exclude=[8], assign="optimal")
Qm, qm_extras = normalized_qm(G)   # defaults: n_rand=200, q_max='plant'
```

**API mismatch, fixed**: the original plan called for
`from_matrix(W, order="ioh")`. `from_matrix()` does not accept an `order`
kwarg -- that parameter only exists on `roles_for`/`role_mask` (cosmetic
node-role labels for plots), not on anything `newman_q`/`left_right_q` read.
It doesn't matter here regardless: NDP's node layout already puts the 8 inputs
at indices 0-7 and the output at index 8 (confirmed against
`train_backend.py`'s `retina_fitness`, which reads `network_state[obs_dim]` as
the output, `obs_dim=8`) -- exactly what `left_right_q`'s
`pinned={i: i // 4 for i in range(8)}, exclude=[8]` already assumes. So
`from_matrix(W)` with no extra argument is correct as-is.

## Results table

| Run ID | Seed | Cond | Nodes | Edges | Bal. Acc | Q | Q_m | Q_m z | Q_m p | sat? | r | r z | r p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1786033855 | 7950840 | FG | 40 | 502 | 0.8438 | 0.1272 | 0.1947 | 3.2880 | 0.0050 | No | 0.1916 | 1.1712 | 0.1194 |
| 1786102425 | 1842456 | FG | 40 | 502 | 0.8438 | 0.1272 | 0.1947 | 3.2880 | 0.0050 | No | 0.1916 | 1.1712 | 0.1194 |
| 1788808796 | 1748523 | FG | 24 | 386 | 0.8432 | 0.0285 | -0.2558 | -3.5750 | 1.0000 | No | 0.0567 | -1.0156 | 0.8557 |
| 1788808995 | 1589167 | FG | 12 | 76 | 0.8540 | 0.0000 | nan | nan | 1.0000 | **Yes** | -0.0739 | nan | 1.0000 |
| 1788809505 | 7287042 | FG | 48 | 978 | 0.8438 | 0.0741 | -0.0322 | -0.9224 | 0.8109 | No | 0.1130 | -0.7978 | 0.7960 |
| 1786053806 | 7318332 | MVG | 80 | 1586 | 0.8438 | 0.1673 | 0.2808 | 13.0511 | 0.0050 | No | 0.2263 | 3.4095 | 0.0050 |
| 1788817038 | 6550047 | MVG | 40 | 502 | 0.8438 | 0.1272 | 0.1947 | 3.2880 | 0.0050 | No | 0.1916 | 1.1712 | 0.1194 |
| 1788821061 | 7766560 | MVG | 16 | 166 | 0.8540 | 0.0060 | -0.7012 | -4.2528 | 1.0000 | No | 0.0190 | -0.5787 | 0.7910 |
| 1788866451 | 6779840 | MVG | 40 | 502 | 0.8438 | 0.1272 | 0.1947 | 3.2880 | 0.0050 | No | 0.1916 | 1.1712 | 0.1194 |
| 1788871442 | 9985617 | MVG | 40 | 502 | 0.8438 | 0.1272 | 0.1947 | 3.2880 | 0.0050 | No | 0.1916 | 1.1712 | 0.1194 |

`Q` = Newman modularity (greedy community detection, unweighted, raw/un-normalized).
`Q_m` = Kashtan-Alon's density-normalized modularity (`normalized_qm`, defaults:
`n_rand=200` null rewirings, `q_max` estimated via the `'plant'` planted-partition
method); its own `z`/`p` compare against the null distribution directly, same as
`Q`'s. `sat?` = the `saturated` flag `normalized_qm` returns when its maximiser
found nothing better than the graph itself (`gain <= 1e-9`) -- per its own
docstring this is only trustworthy on small graphs, and "MEANINGLESS when the
maximiser merely failed" otherwise; here it's the 12-node run, so treat its `Q_m`
(reported as `nan` -- the null and max coincided, see below) as not usable, not
as evidence either way. `r` = Newman's discrete assortativity at the left/right
planted partition (inputs pinned into 2 groups of 4, output excluded); its own
`z`/`p` are against 200 null rewirings at that fixed partition.

## ⚠️ Critical caveat: 5 of these 10 runs are the identical graph

Before reading the aggregates below: **verified directly** (boolean
`|W| > 0` edge-mask equality, not just matching node/edge counts) that runs
1786033855, 1786102425, 1788817038, 1788866451, and 1788871442 -- **2 FG and 3
MVG** -- all grew the exact same 40-node/502-edge wiring pattern. Their
continuous weights differ by ~1e-4 (different CMA-ES trajectories converging
to the same basin), but the edge set itself is bit-for-bit identical across
all 5, which is why they report bit-identical Q/r/z/p (topology-only metrics
can't tell them apart -- they aren't 5 independent structural samples, they're
1 structural sample observed 5 times, 2 of which happen to be labeled FG and 3
MVG). This matches the "sharp global attractor" behaviour already documented
in the "Mechanistic explanation" section of `RESULTS.md` -- it just wasn't
previously known to be common enough to cross the FG/MVG label and dominate
both 5-seed samples this heavily. Concretely: only 4 distinct FG topologies
and 3 distinct MVG topologies exist across these "5 seeds each" -- one of
which (the 40n/502e one) is shared between conditions. The per-condition
statistics below are computed on the full n=5 samples as specified, but should
be read with this in mind, not as 5 independent draws per condition.

## Per-condition aggregates

| | FG (n=5, Q_m n=4*) | MVG (n=5) |
|---|---|---|
| Q: mean / median / stdev | 0.0714 / 0.0741 / 0.0574 | 0.1110 / 0.1272 / 0.0612 |
| Q: min / max | 0.0000 / 0.1272 | 0.0060 / 0.1673 |
| Q_m: mean / median / stdev | 0.0254 / 0.0812 / 0.2158 | 0.0327 / 0.1947 / 0.4120 |
| Q_m: min / max | -0.2558 / 0.1947 | -0.7012 / 0.2808 |
| r: mean / median / stdev | 0.0958 / 0.1130 / 0.1106 | 0.1640 / 0.1916 / 0.0824 |
| r: min / max | -0.0739 / 0.1916 | 0.0190 / 0.2263 |

\* run 1788808995's `Q_m` is `nan` (saturated, small-graph degenerate case --
see table note), dropped from the FG `Q_m` aggregate only; still n=5 for `Q`
and `r`.

Mann-Whitney U (scipy 1.15.3, two-sided, ties present):
- Q (n=5 vs 5): U=7.0, p=0.2652
- Q_m (n=4 vs 5): U=7.0, p=0.5023
- r (n=5 vs 5): U=7.0, p=0.2652

(Identical U for Q and r is a direct consequence of the topology duplication
above -- with most of the mass at one shared value, both metrics see the same
rank ordering on this sample, not evidence they're redundant in general.)

## Does the single-seed pilot's contrast hold up?

Pilot (1 seed each): Q 0.13 (FG) vs 0.17 (MVG); r +0.19/p=0.20 (FG) vs
+0.23/p=0.005 (MVG). Note the pilot runs themselves are exactly
1786033855 (FG) and 1786053806 (MVG), both present in this 5-seed set.
(`Q_m` was not part of the original pilot comparison -- no prior number to
check it against.)

For raw **Q** and **r**: MVG's mean is still higher than FG's in both (Q:
0.111 vs 0.071; r: 0.164 vs 0.096) -- the *direction* of the pilot's contrast
holds up -- but Mann-Whitney finds no significant difference at this sample
size (p=0.265 for both), and per the topology-duplication caveat above, the
effective independent sample is smaller than n=5 per condition.

For **Q_m** (the density-normalized metric, arguably the more appropriate one
per its own documentation -- raw Q is known to conflate modularity with
density/size): the picture is noticeably weaker. FG mean 0.025 vs MVG mean
0.033 -- nearly the same, and now driven by two large opposite outliers
within MVG itself: the 80-node pilot run (Q_m=+0.281, the single highest
value in the whole dataset) and the 16-node run 1788821061 (Q_m=-0.701, the
single lowest, and the only strongly *negative* value in the dataset --
i.e. less modular than its own degree-preserving null, not merely "not
detected as modular"). These two roughly cancel in the MVG mean. Mann-Whitney
on Q_m: p=0.502.

So: direction of the pilot's raw-Q/r contrast is preserved but not
significant, and normalizing for density (Q_m) weakens it further, close to
parity between conditions -- and the FG/MVG contrast that does show up in Q_m
is dominated by single-run outliers, not a consistent per-condition shift.
Ambiguous, not resolved -- no stronger claim than that is supported by this
data.
