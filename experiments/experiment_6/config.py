"""CLI -> configuration for experiment 6 (SMCGP on the even-parity curriculum).

Defaults marked [paper] are PAPER_SPEC.md's (Harding/Miller/Banzhaf CEC 2009) and
should not be changed without recording it. Defaults marked [our choice] are gaps
the paper leaves unspecified for this representation -- see PAPER_SPEC.md.

`--max-inputs 20 --max-evals 10000000` reproduces the paper's own experiment; that
is hours of pure-Python work, so the CLI default is a much smaller smoke-test scale
and prints an explicit note when run below the paper's settings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gates as gates_mod

PAPER_MAX_INPUTS = 20
PAPER_MAX_EVALS = 10_000_000


@dataclass
class RunConfig:
    nodes: int = 100                    # [our choice] genotype length (fixed for the run)
    gates: str = gates_mod.DEFAULT_GATES  # [paper] restricted set = AND,NAND,OR,NOR
    # [our choice] connection-gene domain [1, addr_max]. 0 means "= --nodes",
    # resolved in `parse` -- the paper gives no explicit upper bound, but leaving it
    # much larger than the genotype (this module's very first draft used a flat 200
    # against 30 nodes) means most addresses land out-of-range by construction, so
    # nearly EVERY random genotype's active output degenerates to a constant and the
    # (1+4) ES has no gradient to climb from -- verified empirically: a 300k-eval run
    # never left the "any constant" plateau until this was tied to genotype size.
    addr_max: int = 0
    param_range: float = 10.0           # [our choice] init / randomize domain for P0..P2
    mutation_rate: float = 0.1          # [paper] per-gene mutation probability
    param_randomize_prob: float = 0.1   # [paper] P(randomize) vs P(add noise) for real genes
    sigma: float = 20.0                 # [paper] additive-noise stddev for real genes
    todo_cap: int = 2                   # [paper] "To Do" list length per development iteration
    bootstrap: int = 50                 # [paper] initial random population size
    popsize: int = 5                    # [paper] (1+4) ES: parent + 4 offspring
    max_inputs: int = 8                 # smoke-test scale; paper = 20 (see PAPER_MAX_INPUTS)
    max_evals: int = 200_000            # smoke-test scale; paper = 10,000,000
    seed: int = 0
    n_seeds: int = 1
    log_interval: int = 50
    checkpoint_interval: int = 500
    resume: bool = True
    out_dir: str = "runs"
    tag: str = ""


def parse(argv=None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=RunConfig.nodes)
    p.add_argument("--gates", type=str, default=RunConfig.gates)
    p.add_argument("--addr-max", type=int, default=RunConfig.addr_max)
    p.add_argument("--param-range", type=float, default=RunConfig.param_range)
    p.add_argument("--mutation-rate", type=float, default=RunConfig.mutation_rate)
    p.add_argument("--param-randomize-prob", type=float,
                   default=RunConfig.param_randomize_prob)
    p.add_argument("--sigma", type=float, default=RunConfig.sigma)
    p.add_argument("--todo-cap", type=int, default=RunConfig.todo_cap)
    p.add_argument("--bootstrap", type=int, default=RunConfig.bootstrap)
    p.add_argument("--max-inputs", type=int, default=RunConfig.max_inputs)
    p.add_argument("--max-evals", type=int, default=RunConfig.max_evals)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    p.add_argument("--n-seeds", type=int, default=RunConfig.n_seeds)
    p.add_argument("--log-interval", type=int, default=RunConfig.log_interval)
    p.add_argument("--checkpoint-interval", type=int,
                   default=RunConfig.checkpoint_interval)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--out-dir", type=str, default=RunConfig.out_dir)
    p.add_argument("--tag", type=str, default=RunConfig.tag)
    ns = p.parse_args(argv)
    cfg = RunConfig(**vars(ns))
    if cfg.addr_max <= 0:
        cfg.addr_max = cfg.nodes
    gates_mod.build_set(cfg.gates)      # validate early, fail before any run dir is made
    if cfg.max_inputs > PAPER_MAX_INPUTS:
        raise ValueError(f"--max-inputs {cfg.max_inputs} exceeds the paper's own "
                         f"ceiling of {PAPER_MAX_INPUTS} (exhaustive truth tables "
                         f"become impractical beyond it -- see CLAUDE.md)")
    return cfg
