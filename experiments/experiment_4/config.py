"""CLI -> configuration for experiment 4 (CGP).

Flag groups mirror experiments 2/3 (`representation` here plays the role their
`architecture` group plays). Defaults marked [Table II] are the paper's and should
not be changed without recording it -- see PAPER_SPEC.md.

The ECGP flags (compress/expand, module point mutation, add/remove input/output,
max module size) are deliberately absent: this is the CGP half. They land in an
`ecgp` group later without disturbing anything here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gates as gates_mod
import tasks as tasks_mod


@dataclass
class RunConfig:
    # representation
    nodes: int
    mutation_rate: float
    gates: str
    # task
    task: str
    operation: str
    mvg: bool
    mvg_ops: tuple[str, ...]
    switch_interval: int
    # evolution
    popsize: int
    generations: int
    stop_on_solution: bool
    n_seeds: int
    seed: int
    fitness: str
    # run
    out_dir: str
    log_interval: int
    save_best: bool
    tag: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="experiment_4/train.py",
        description="CGP on the Kashtan-Alon retina (experiment 4). See README.md "
                    "for the hypothesis and PAPER_SPEC.md for every parameter's source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("representation")
    g.add_argument("--nodes", type=int, default=100,
                   help="initial genotype size in nodes [Table II]")
    g.add_argument("--mutation-rate", type=float, default=0.03,
                   help="fraction of gene slots mutated per application; "
                        "the operator itself is always applied [Table II]")
    g.add_argument("--gates", type=str, default=gates_mod.DEFAULT_GATES,
                   help="comma-separated function set. Known: "
                        + ",".join(gates_mod.ALL_GATES)
                        + ". NOTE: adding 'xor' hands the retina its combiner for "
                          "free and is not comparable to default runs.")
    # levels-back is intentionally not a flag: the paper's formulation lets a node
    # connect to ANY previous node, so it is not a free parameter. Node arity is
    # likewise derived (max arity over --gates), not set.

    g = p.add_argument_group("task")
    g.add_argument("--task", type=str, default="retina_ka2005",
                   choices=sorted(tasks_mod.TASKS),
                   help="retina_ka2005 is KA 2005's verified object rule, loaded "
                        "from kashtan_alon/tasks.py")
    g.add_argument("--operation", type=str, default="and",
                   choices=list(tasks_mod.OPERATIONS),
                   help="top-level combiner OP(L,R); ignored by non-retina tasks")
    g.add_argument("--mvg", action="store_true",
                   help="modularly varying goals: cycle --mvg-ops every "
                        "--switch-interval generations")
    g.add_argument("--mvg-ops", type=str, default="and,or",
                   help="goal cycle under --mvg")
    g.add_argument("--switch-interval", type=int, default=20,
                   help="generations per goal epoch under --mvg (E; matches kashtan_alon/)")

    g = p.add_argument_group("evolution")
    g.add_argument("--popsize", type=int, default=5,
                   help="(1+lambda) ES with lambda = popsize-1, i.e. (1+4) [Table II]")
    g.add_argument("--generations", type=int, default=1000,
                   help="generation budget per seed")
    g.add_argument("--stop-on-solution", dest="stop_on_solution",
                   action="store_true", default=True,
                   help="halt a seed once it is perfect (the paper's protocol)")
    g.add_argument("--no-stop-on-solution", dest="stop_on_solution",
                   action="store_false",
                   help="always run the full budget; forced under --mvg")
    g.add_argument("--n-seeds", type=int, default=50,
                   help="independent runs [Table II]")
    g.add_argument("--seed", type=int, default=0, help="base RNG seed")
    g.add_argument("--fitness", type=str, default="raw", choices=("raw", "balanced"),
                   help="selection score. raw = correct patterns (default, and the "
                        "one to report). balanced manufactures a gradient into the "
                        "one-module solution -- see cgp.balanced_score.")

    g = p.add_argument_group("run")
    g.add_argument("--out-dir", type=str, default="runs",
                   help="output root (gitignored)")
    g.add_argument("--log-interval", type=int, default=100,
                   help="generations between log rows when the goal is fixed; under "
                        "--mvg rows are emitted at the end of each goal epoch instead")
    g.add_argument("--save-best", action="store_true",
                   help="write the best genotype of each seed to the run directory")
    g.add_argument("--tag", type=str, default="",
                   help="suffix appended to the run directory name")
    return p


def parse(argv=None) -> RunConfig:
    args = build_parser().parse_args(argv)

    gate_set = gates_mod.build_set(args.gates)          # validates early
    mvg_ops = tuple(s.strip() for s in args.mvg_ops.split(",") if s.strip())

    if args.mvg:
        if not tasks_mod.uses_operation(args.task):
            raise SystemExit(
                f"--mvg is meaningless for --task {args.task}: its target does not "
                f"depend on --operation. Use a retina task or drop --mvg.")
        if len(mvg_ops) < 2:
            raise SystemExit("--mvg needs at least two goals in --mvg-ops")
        bad = [o for o in mvg_ops if o not in tasks_mod.OPERATIONS]
        if bad:
            raise SystemExit(f"unknown --mvg-ops: {bad} (known: {list(tasks_mod.OPERATIONS)})")
        if args.switch_interval < 1:
            raise SystemExit("--switch-interval must be >= 1")
    if args.popsize < 2:
        raise SystemExit("--popsize must be >= 2 (one parent plus at least one offspring)")
    if not 0.0 < args.mutation_rate <= 1.0:
        raise SystemExit("--mutation-rate must be in (0, 1]")
    if args.nodes < 1:
        raise SystemExit("--nodes must be >= 1")

    # Under MVG the target keeps moving, so "solved" is not terminal -- stopping on
    # it would end a run at whichever goal happened to be easy.
    stop = args.stop_on_solution and not args.mvg

    del gate_set
    return RunConfig(
        nodes=args.nodes, mutation_rate=args.mutation_rate, gates=args.gates,
        task=args.task, operation=args.operation, mvg=args.mvg, mvg_ops=mvg_ops,
        switch_interval=args.switch_interval,
        popsize=args.popsize, generations=args.generations, stop_on_solution=stop,
        n_seeds=args.n_seeds, seed=args.seed, fitness=args.fitness,
        out_dir=args.out_dir, log_interval=args.log_interval,
        save_best=args.save_best, tag=args.tag,
    )
