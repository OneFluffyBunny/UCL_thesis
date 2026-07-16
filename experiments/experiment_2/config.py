"""Command-line configuration for experiment 2 (direct encoding).

Deliberately mirrors ``experiment_1/config.py`` so the two experiments share the
same flags, defaults and structure and are directly comparable. The ONLY
difference is the architecture group: direct encoding has NO encoding knobs
(no K / type_dim / pos_dim / g_width / g_depth) — the genome IS the weight
vector, so the only architectural choices are the topology + inference dynamics.

Groups:
  * architecture -- maps directly onto model.BrainConfig.
  * evolution    -- CMA-ES / search settings.
  * task         -- Kashtan-Alon logic task + modularly-varying-goal settings.
  * analysis     -- density measurement / logging / visualisation.

Run `python config.py --help` to see every flag, or `python config.py ...` to
print the resolved configuration.
"""

from __future__ import annotations

import argparse
import dataclasses

import jax
import jax.numpy as jnp

from model import BrainConfig


# sigma (activation) options selectable from the CLI (same set as experiment 1)
ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "linear": lambda x: x,
}


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Everything that is not part of the brain architecture itself.

    A trimmed mirror of experiment_1's RunConfig: it drops the fields exp 1
    carries but never reads (elitism / eval_reps / test_reps / n_examples --
    dead there because the tasks are deterministic full enumerations), so this
    only lists knobs that actually affect a run.
    """
    # evolution / search
    strategy: str
    fitness: str            # 'accuracy' (raw 0/1) or 'margin' (smooth hinged surrogate)
    popsize: int
    generations: int
    sigma_init: float
    seed: int
    n_seeds: int            # how many seeds to run in one invocation
    target: float           # stop a seed early once best accuracy reaches this
    # task
    task: str
    operation: str
    input_encoding: str     # bipolar {-1,+1} or binary {0,1}
    mvg: bool               # modularly-varying goal: switch target every interval
    switch_interval: int    # generations between goal switches (if mvg)
    mvg_ops: tuple          # ops to cycle through under mvg, e.g. ("and", "or") or ("xor", "and")
    # analysis / logging
    prune_threshold: float  # |w| below this is treated as "no edge" for analysis
    log_interval: int
    viz_interval: int       # >0: render+open the best brain every N gens during training
    open_image: bool        # auto-open the saved brain image at the end
    balanced: bool          # balanced accuracy (chance=50%) vs raw accuracy
    out_dir: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 2: evolve a DIRECTLY-encoded network (K-A-style control).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- architecture (-> BrainConfig) ----------------------------------------
    # NOTE: no --n-types / --type-dim / --pos-dim / --g-width / --g-depth here --
    # direct encoding has no encoding hyper-parameters, just the topology.
    arch = p.add_argument_group("architecture")
    arch.add_argument("--n-in", type=int, default=8, help="number of input neurons")
    arch.add_argument("--n-hidden", type=int, default=20, help="number of hidden neurons (fixed total)")
    arch.add_argument("--n-out", type=int, default=1, help="number of output neurons")
    arch.add_argument("--rnn-iters", type=int, default=8, help="synchronous recurrent passes at inference")
    arch.add_argument("--no-bias", action="store_true", help="disable the per-neuron bias")
    arch.add_argument("--activation", choices=sorted(ACTIVATIONS), default="tanh", help="sigma activation")

    # --- evolution / search ---------------------------------------------------
    evo = p.add_argument_group("evolution")
    evo.add_argument("--strategy", default="CMA_ES", help="evosax strategy (CMA-ES for now)")
    evo.add_argument("--fitness", choices=["accuracy", "margin"], default="accuracy",
                     help="training signal for selection: raw balanced accuracy (default) "
                          "or a smooth hinged signed-margin surrogate on the tanh output "
                          "(accuracy is still what gets logged / early-stopped / reported)")
    evo.add_argument("--popsize", type=int, default=64,
                     help="population size (individuals per generation)")
    evo.add_argument("--generations", type=int, default=1000, help="number of generations")
    evo.add_argument("--sigma-init", type=float, default=0.1, help="initial ES mutation scale")
    evo.add_argument("--seed", type=int, default=0, help="PRNG seed (first seed)")
    evo.add_argument("--n-seeds", type=int, default=1, help="run this many consecutive seeds")
    evo.add_argument("--target", type=float, default=1.0, help="stop a seed early once best accuracy reaches this")

    # --- task -----------------------------------------------------------------
    task = p.add_argument_group("task")
    task.add_argument("--task", default="retina",
                      choices=["copy", "and2", "left", "retina"],
                      help="task: copy (1 bit) | and2 (2 bits) | left (4 bits) | retina (full Kashtan-Alon)")
    task.add_argument("--operation", choices=["and", "or", "xor"], default="xor", help="combining operation OP(L,R)")
    task.add_argument("--input-encoding", choices=["bipolar", "binary"], default="bipolar",
                      help="bipolar: bits -> {-1,+1}; binary: bits -> {0,1}")
    task.add_argument("--mvg", action="store_true", help="modularly-varying goal: switch target periodically")
    task.add_argument("--switch-interval", type=int, default=20, help="generations between goal switches (if --mvg)")
    task.add_argument("--mvg-ops", type=str, default="and,or",
                      help="comma-separated operations to cycle through under --mvg "
                           "(each still shares the fixed left/right features; classic "
                           "Kashtan-Alon uses and,or -- pass e.g. xor,and to include xor)")

    # --- analysis / logging ---------------------------------------------------
    ana = p.add_argument_group("analysis")
    ana.add_argument("--prune-threshold", type=float, default=0.05, help="|w| below this = no edge (analysis only)")
    ana.add_argument("--log-interval", type=int, default=10, help="generations between log lines")
    ana.add_argument("--viz-interval", type=int, default=0, help=">0: render+open best brain every N gens during training")
    ana.add_argument("--no-open", action="store_true", help="do not auto-open the brain image at the end")
    ana.add_argument("--no-balanced", action="store_true", help="use raw accuracy instead of balanced (chance=50%%)")
    ana.add_argument("--out-dir", default="./runs", help="directory for logs/checkpoints")

    return p


def build_brain_config(args: argparse.Namespace) -> BrainConfig:
    return BrainConfig(
        n_in=args.n_in,
        n_hidden=args.n_hidden,
        n_out=args.n_out,
        rnn_iters=args.rnn_iters,
        use_bias=not args.no_bias,
        activation=ACTIVATIONS[args.activation],
    )


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        strategy=args.strategy,
        fitness=args.fitness,
        popsize=args.popsize,
        generations=args.generations,
        sigma_init=args.sigma_init,
        seed=args.seed,
        n_seeds=args.n_seeds,
        target=args.target,
        task=args.task,
        operation=args.operation,
        input_encoding=args.input_encoding,
        mvg=args.mvg,
        switch_interval=args.switch_interval,
        mvg_ops=tuple(op.strip() for op in args.mvg_ops.split(",")),
        prune_threshold=args.prune_threshold,
        log_interval=args.log_interval,
        viz_interval=args.viz_interval,
        open_image=not args.no_open,
        balanced=not args.no_balanced,
        out_dir=args.out_dir,
    )


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    return build_brain_config(args), build_run_config(args), args


if __name__ == "__main__":
    brain_cfg, run_cfg, _ = parse_args()
    print("BrainConfig:")
    for f in dataclasses.fields(brain_cfg):
        print(f"  {f.name:14s} = {getattr(brain_cfg, f.name)}")
    print(f"  {'n_total':14s} = {brain_cfg.n_total}")
    print("\nRunConfig:")
    for f in dataclasses.fields(run_cfg):
        print(f"  {f.name:16s} = {getattr(run_cfg, f.name)}")
