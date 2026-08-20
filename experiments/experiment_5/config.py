"""CLI -> configuration for experiment 5 (big-brain CGP/ECGP).

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

# `[exp5]` Above this many program inputs, `parse` prints a cost warning. 16 inputs is
# 65 536 patterns = 8 KB per wire, which still runs at a sane speed; every input past
# it doubles that. Not a limit -- just the point where the user should know.
MAX_INPUTS_QUIET = 16


@dataclass
class RunConfig:
    # representation
    nodes: int
    mutation_rate: float
    wiring_weight: float
    gates: str
    # ECGP (all inert unless --ecgp)
    ecgp: bool
    compress_prob: float
    expand_prob: float
    module_point_prob: float
    add_input_prob: float
    remove_input_prob: float
    add_output_prob: float
    remove_output_prob: float
    max_module_size: int
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
    post_solve_gens: int
    parsimony_tiebreak: bool
    n_seeds: int
    seed: int
    fitness: str
    # run
    out_dir: str
    log_interval: int
    save_best: bool
    tag: str
    # checkpointing
    checkpoint_interval: int
    resume: bool
    # visualisation
    viz: bool
    viz_seeds: int
    grid: bool
    grid_seed: int
    colour_cones: bool
    # parallelism
    workers: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="experiment_5/train.py",
        description="CGP/ECGP on many-input, many-output tasks (experiment 5). See README.md "
                    "for the hypothesis and PAPER_SPEC.md for every parameter's source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("representation")
    g.add_argument("--nodes", type=int, default=100,
                   help="initial genotype size in nodes [Table II]")
    g.add_argument("--mutation-rate", type=float, default=0.03,
                   help="fraction of gene slots mutated per application; "
                        "the operator itself is always applied [Table II]")
    g.add_argument("--wiring-weight", type=float, default=1.0,
                   help="relative probability of mutating a WIRING gene (a node "
                        "input, or the output gene) versus a function gene. "
                        "1.0 = every slot equally likely [Table II]; below 1.0 "
                        "biases mutation toward swapping gates in place rather "
                        "than rewiring [our choice, experimental]")
    g.add_argument("--gates", type=str, default=gates_mod.DEFAULT_GATES,
                   help="comma-separated function set. Known: "
                        + ",".join(gates_mod.ALL_GATES)
                        + ". NOTE: adding 'xor' hands the retina its combiner for "
                          "free and is not comparable to default runs.")
    # levels-back is intentionally not a flag: the paper's formulation lets a node
    # connect to ANY previous node, so it is not a free parameter. Node arity is
    # likewise derived (max arity over --gates), not set.

    # ECGP. Every probability below is Table II `[verbatim]` and marked `*` there as
    # ECGP-only, so leaving them alone reproduces the paper. They do nothing without
    # --ecgp: the CGP arm never constructs a module list.
    g = p.add_argument_group("ecgp")
    g.add_argument("--ecgp", dest="ecgp", action="store_true", default=False,
                   help="evolve modules: compress/expand, a global module list and "
                        "the five module operators (PAPER_SPEC sections 4-7). "
                        "Default off = plain CGP, the baseline in RESULTS.md")
    g.add_argument("--no-ecgp", dest="ecgp", action="store_false")
    g.add_argument("--compress-prob", type=float, default=0.1,
                   help="per offspring: encapsulate a run of type 0 nodes into a new "
                        "module [Table II]")
    g.add_argument("--expand-prob", type=float, default=0.2,
                   help="per offspring: dissolve a type I node back into its nodes. "
                        "Twice compress -- it is the destruction operator [Table II]")
    g.add_argument("--module-point-prob", type=float, default=0.04,
                   help="per module per offspring: mutate the module's body [Table II]")
    g.add_argument("--add-input-prob", type=float, default=0.01,
                   help="per module per offspring [Table II]")
    g.add_argument("--remove-input-prob", type=float, default=0.02,
                   help="per module per offspring [Table II]")
    g.add_argument("--add-output-prob", type=float, default=0.01,
                   help="per module per offspring [Table II]")
    g.add_argument("--remove-output-prob", type=float, default=0.02,
                   help="per module per offspring [Table II]")
    g.add_argument("--max-module-size", type=int, default=5,
                   help="ms: nodes per module, 2..ms. PAPER_SPEC section 9 "
                        "[our choice] -- the paper's own sweep found no correlation "
                        "between ms and performance")

    g = p.add_argument_group("task")
    # `[exp5]` NOT `choices=`: a task name carries its SIZE (`retina_x3`, `add6`,
    # `mult4`), so the legal set is infinite and validation has to be a parse. The
    # name is checked in `parse` below, which raises with the family list.
    g.add_argument("--task", type=str, default="retina_ka2005",
                   help="task name, family + size. Families: "
                        + ", ".join(tasks_mod.TASK_FAMILIES)
                        + ". Examples: " + ", ".join(tasks_mod.EXAMPLE_TASKS)
                        + ". retina_ka2005 (== retina_x1) is KA 2005's verified "
                          "object rule; retina_xN is N independent copies of it, "
                          "one program output each")
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
    g.add_argument("--post-solve-gens", type=int, default=0,
                   help="keep evolving this many MORE generations after a seed first "
                        "reaches a perfect score, instead of halting immediately "
                        "[default: 0 = halt immediately, the paper's protocol]. "
                        "The (1+4) ES is elitist, so once solved the score can only "
                        "stay at the max or neutrally tie-drift -- this is for "
                        "watching whether unconstrained drift shrinks a bloated "
                        "solution or the bloat is endemic. No effect if "
                        "--no-stop-on-solution is set (nothing to extend from).")
    g.add_argument("--parsimony-tiebreak", dest="parsimony_tiebreak",
                   action="store_true", default=False,
                   help="lexicographic parsimony pressure [our choice, NOT the "
                        "paper's protocol -- default off, so a plain run is "
                        "bit-identical to before this flag existed]. Among "
                        "offspring TIED with the parent's score (selection rule 4b, "
                        "the neutral-drift step), prefer the one with fewest active "
                        "gates in its FLATTENED circuit (a module counts its "
                        "constituent primitives, not itself as one node), breaking "
                        "any remaining tie at random as before. Never trades fitness "
                        "for size -- only biases which point on an already-tied "
                        "neutral plateau the drift step lands on.")
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

    g = p.add_argument_group("checkpointing")
    g.add_argument("--checkpoint-interval", type=int, default=1000,
                   help="generations between full-state checkpoints (0 = off); "
                        "enables --resume. Matches kashtan_alon/train.py.")
    g.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="resume from a checkpoint and skip already-finished seeds "
                        "[default: on]")
    g.add_argument("--no-resume", dest="resume", action="store_false",
                   help="ignore any existing checkpoint/result and start fresh "
                        "(overwrites logs)")

    g = p.add_argument_group("parallelism")
    g.add_argument("--workers", type=int, default=0,
                   help="processes running seeds concurrently. 0 = auto (see "
                        "train.plan_workers: adapts to seed count, core count and "
                        "estimated run length). 1 = force the serial path. Seeds are "
                        "the ONLY parallel axis -- generations are serial by "
                        "construction and an offspring is too small to hand to a "
                        "process (see RESULTS.md).")

    g = p.add_argument_group("visualisation")
    g.add_argument("--viz", dest="viz", action="store_true", default=True,
                   help="save a circuit diagram alongside every log row [default: on]")
    g.add_argument("--no-viz", dest="viz", action="store_false")
    g.add_argument("--grid", dest="grid", action="store_true", default=True,
                   help="at the end of a run, tile 6 evenly spaced snapshots of ONE "
                        "seed's history into runs/<name>/seed<k>_stages.png and open "
                        "it -- the circuit through generational time [default: on]")
    g.add_argument("--no-grid", dest="grid", action="store_false",
                   help="skip the stage grid (and the window it opens) -- use for "
                        "headless or batched runs")
    g.add_argument("--grid-seed", type=int, default=0,
                   help="which seed the stage grid follows")
    g.add_argument("--colour-cones", dest="colour_cones", action="store_true",
                   default=False,
                   help="colour each gate by whether its input cone is the LEFT half of "
                        "the retina, the RIGHT half, or both. That is the Kashtan-Alon "
                        "modularity question; with it off (the default) colour carries "
                        "MODULE identity instead, which is the ECGP question.")
    g.add_argument("--viz-seeds", type=int, default=0,
                   help="how many seeds get a frame per log row. The FINAL circuit of "
                        "every seed is always drawn regardless, and the stage grid "
                        "covers the usual need for 9 renders instead of thousands, so "
                        "this defaults to OFF: at a 300k-generation budget it is "
                        "seeds x rows matplotlib calls and dominates the run.")
    return p


def parse(argv=None) -> RunConfig:
    args = build_parser().parse_args(argv)

    gate_set = gates_mod.build_set(args.gates)          # validates early
    mvg_ops = tuple(s.strip() for s in args.mvg_ops.split(",") if s.strip())

    try:                                                # `[exp5]` validates the name
        n_in = tasks_mod.n_inputs(args.task)
    except ValueError as e:
        raise SystemExit(str(e))
    # The 2**n_in wall (see tasks.py): one wire is an n_in-bit-wide truth table, so
    # cost per gate doubles with every extra input. Refusing outright would be wrong
    # -- it is a slow task, not an invalid one -- so this warns and runs.
    if n_in > MAX_INPUTS_QUIET:
        print(f"  WARNING: --task {args.task} has {n_in} inputs -> {1 << n_in} patterns, "
              f"{(1 << n_in) // 8192} KB per wire. Evaluation is exhaustive, so this "
              f"is ~{2 ** (n_in - MAX_INPUTS_QUIET)}x the cost of a {MAX_INPUTS_QUIET}"
              f"-input task. Expect it to be slow.")

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
    if args.post_solve_gens < 0:
        raise SystemExit("--post-solve-gens must be >= 0")
    # 0 would make wiring genes unreachable and the topology frozen for the whole
    # run; the rejection sampler would also never terminate once every function
    # slot is already picked.
    if not 0.0 < args.wiring_weight <= 1.0:
        raise SystemExit("--wiring-weight must be in (0, 1]")
    if args.ecgp:
        if args.max_module_size < 2:
            raise SystemExit("--max-module-size must be >= 2 (PAPER_SPEC section 4)")
        # PAPER_SPEC section 1: "node arity 2" [verbatim], and a module body is
        # encoded with two input genes per node. A function set whose max arity is
        # not 2 would silently mean something different inside modules than outside.
        if gates_mod.max_arity(gate_set) != 2:
            raise SystemExit("--ecgp requires a function set of arity 2 "
                             f"(--gates {args.gates} has arity "
                             f"{gates_mod.max_arity(gate_set)})")
        for name in ("compress_prob", "expand_prob", "module_point_prob",
                     "add_input_prob", "remove_input_prob", "add_output_prob",
                     "remove_output_prob"):
            if not 0.0 <= getattr(args, name) <= 1.0:
                raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")

    # Under MVG the target keeps moving, so "solved" is not terminal -- stopping on
    # it would end a run at whichever goal happened to be easy.
    stop = args.stop_on_solution and not args.mvg

    del gate_set
    return RunConfig(
        nodes=args.nodes, mutation_rate=args.mutation_rate,
        wiring_weight=args.wiring_weight, gates=args.gates,
        ecgp=args.ecgp, compress_prob=args.compress_prob,
        expand_prob=args.expand_prob, module_point_prob=args.module_point_prob,
        add_input_prob=args.add_input_prob, remove_input_prob=args.remove_input_prob,
        add_output_prob=args.add_output_prob,
        remove_output_prob=args.remove_output_prob,
        max_module_size=args.max_module_size,
        task=args.task, operation=args.operation, mvg=args.mvg, mvg_ops=mvg_ops,
        switch_interval=args.switch_interval,
        popsize=args.popsize, generations=args.generations, stop_on_solution=stop,
        post_solve_gens=args.post_solve_gens,
        parsimony_tiebreak=args.parsimony_tiebreak,
        n_seeds=args.n_seeds, seed=args.seed, fitness=args.fitness,
        out_dir=args.out_dir, log_interval=args.log_interval,
        save_best=args.save_best, tag=args.tag,
        checkpoint_interval=max(0, args.checkpoint_interval), resume=args.resume,
        viz=args.viz, viz_seeds=max(0, args.viz_seeds), grid=args.grid,
        grid_seed=args.grid_seed, colour_cones=args.colour_cones,
        workers=max(0, args.workers),
    )
