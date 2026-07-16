# Pre-migration git history

Before this work was folded into the `UCL_thesis` repo (2026-07-16, fresh-start
migration — see the first commit of this repo), `experiments/` was its own
standalone git repo with the two commits below. That `.git` was dropped so
`experiments/` could become a plain subfolder; the full commit messages are
preserved here so no context is lost.

---

## commit 9ccedc8 — Add oracle + reachability probes, margin fitness, viz fixes
*Andrei Raducea-Marin — 2026-07-01*

Diagnose why `left`/`retina` plateau below 1.0 despite converging:

- `oracle.py`: hand-wire a perfect `left` brain, OLS-fit g to it, run the
  resulting "oracle DNA" through real inference. Shows a perfect genome
  EXISTS even at g_width=16 → failure is reachability, not representability.
- `reachability.py`: basin-of-attraction probe. CMA-ES seeded at the oracle
  (and perturbed outward) never recovers 1.0 — the optimum is a needle
  narrower than the training mutation scale.
- `train.py`/`config.py`: add `--fitness {accuracy,margin}`. margin is a smooth
  balanced hinged signed-margin surrogate for CMA-ES selection; accuracy is
  still logged/early-stopped/reported. Defaults to accuracy (unchanged).
- `visualize.py`: distinct node/edge colour scheme (hidden types no longer
  clash with input/output; excitatory/inhibitory edges reserved colours).
- `README.md`: future ideas / levers section.

---

## commit 792c5f9 — Experiment 1: cell-type DNA → static brain + CMA-ES training
*Andrei Raducea-Marin — 2026-06-26*

Minimal non-spatial model for studying emergence of modularity. DNA stores K
hidden cell-type identities (+1 input, +1 output), per-type abundance genes, and
a shared connection rule g; size is O(K), independent of neuron count. g maps
`[type|position|role]` feature pairs to a directed, [-1,1]-bounded weight; the
static brain is grown once and run as a synchronous recurrent pass (NDP-style,
no within-life plasticity).

- `model.py`: Genome + BrainConfig; per-signature weight build (U² not N²).
- `tasks.py`: copy / and2 / left / retina (Kashtan-Alon) staircase.
- `train.py`: CMA-ES (evosax), deterministic balanced-accuracy fitness, MVG
  goal-switching, LNDP-style logs, multi-seed, target early-stop, viz-interval.
- `visualize.py` / `visualize_ckpt.py`: brain rendering + auto-open + ckpt reload.
- `config.py`: grouped CLI for all parameters.
