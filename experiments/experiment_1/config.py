"""Command-line configuration for experiment 1.

Groups:
  * architecture -- maps directly onto model.BrainConfig (built now).
  * evolution    -- CMA-ES / search settings (used by the training loop, TBD).
  * task         -- Kashtan-Alon logic task + modularly-varying-goal settings.
  * analysis     -- modularity measurement / logging.

Run `python config.py --help` to see every flag, or `python config.py ...` to
print the resolved configuration.
"""

from __future__ import annotations

import argparse
import dataclasses

import jax
import jax.numpy as jnp

import tasks
from model import BrainConfig


# sigma (activation) options selectable from the CLI
ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "linear": lambda x: x,
}


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Everything that is not part of the brain architecture itself."""
    # evolution / search
    strategy: str
    fitness: str            # 'accuracy' (raw 0/1) or 'margin' (smooth hinged surrogate)
    popsize: int
    generations: int
    sigma_init: float
    seed: int
    n_seeds: int            # how many seeds to run in one invocation
    target: float           # stop a seed early once best accuracy reaches this
    early_stop: bool        # honour `target` at all (already forced off under mvg)
    # NOTE: elitism / eval_reps / test_reps / n_examples used to live here but
    # train.py never read any of them -- evaluation is ALWAYS the full 2**n_in
    # enumeration (deterministic, so reps are pointless) and elitism is whatever
    # evosax's CMA-ES does internally. They were removed rather than left as
    # flags that silently do nothing. experiment_2/3 never had them.
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
        description="Experiment 1: evolve a DNA that grows a static modular brain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- architecture (-> BrainConfig) ----------------------------------------
    arch = p.add_argument_group("architecture")
    arch.add_argument("--n-in", type=int, default=8, help="number of input neurons")
    arch.add_argument("--n-hidden", type=int, default=20, help="number of hidden neurons (fixed total)")
    arch.add_argument("--n-out", type=int, default=1, help="number of output neurons")
    arch.add_argument("--n-types", "-K", type=int, default=4, help="distinct hidden cell-types (K)")
    arch.add_argument("--type-dim", type=int, default=4, help="dim of an evolved type identity vector")
    arch.add_argument("--pos-dim", type=int, default=4, help="dim of the fixed positional code (I/O only)")
    arch.add_argument("--g-width", type=int, default=16, help="hidden width of the connection rule g")
    arch.add_argument("--g-depth", type=int, default=1, help="hidden layers in g (1 or 2 recommended)")
    arch.add_argument("--rnn-iters", type=int, default=8, help="synchronous recurrent passes at inference")
    arch.add_argument("--no-bias", action="store_true", help="disable the per-type neuron bias")
    arch.add_argument("--activation", choices=sorted(ACTIVATIONS), default="tanh", help="sigma activation")
    arch.add_argument("--w-threshold", type=float, default=0.0,
                      help="SYNAPTIC GATE: |g(feat_i,feat_j)| below this is forced to exactly 0 "
                           "and the brain is EVALUATED on the gated matrix, so sparsity is part "
                           "of the phenotype rather than a drawing convention (0 = off, the "
                           "historical behaviour). Not to be confused with --prune-threshold, "
                           "which only affects counting/plotting. Weights are tanh-bounded to "
                           "[-1,1] but the evolved scale varies ~10x between seeds, so a value "
                           "that sparsifies one run can silence another entirely -- watch the "
                           "logged density, and see RESULTS.md for measured |w| distributions. "
                           "MEASURED TO FAIL as a density control: prefer --synaptic-budget")
    arch.add_argument("--synaptic-budget", type=float, default=0.0,
                      help="SYNAPTIC BUDGET: give every neuron this fixed total incoming "
                           "|weight| to share out among its synapses (0 = off). Density stops "
                           "being free -- an extra connection dilutes the ones already there -- "
                           "so specialising becomes the cheap way to get a strong signal. Also "
                           "drops the tanh bound on g (redundant once renormalised, and a "
                           "saturation attractor) and bounds each neuron's pre-activation by "
                           "this value, so ~1-2 keeps tanh in its useful range. On its own it "
                           "makes weights small but not zero -- pair it with --shrink for "
                           "structural sparsity. Mutually exclusive with --w-threshold")
    arch.add_argument("--shrink", type=float, default=0.0,
                      help="with --synaptic-budget: zero any synapse weaker than this FRACTION "
                           "of its target neuron's mean incoming |g|, before the budget is "
                           "shared out. In [0, 1). Relative on purpose -- an absolute cutoff is "
                           "what --w-threshold already tried, and evolution escaped it by "
                           "inflating g; nothing can inflate its way above its own mean. Higher "
                           "= sparser, but watch for premature collapse onto a single input")

    # --- evolution / search ---------------------------------------------------
    evo = p.add_argument_group("evolution")
    evo.add_argument("--strategy", default="CMA_ES", help="evosax strategy (CMA-ES for now)")
    evo.add_argument("--fitness", choices=["accuracy", "margin"], default="accuracy",
                     help="training signal for selection: raw balanced accuracy (default, as before) "
                          "or a smooth hinged signed-margin surrogate on the tanh output "
                          "(accuracy is still what gets logged / early-stopped / reported)")
    evo.add_argument("--popsize", type=int, default=64,
                     help="population size (individuals per generation)")
    evo.add_argument("--generations", type=int, default=1000, help="number of generations")
    evo.add_argument("--sigma-init", type=float, default=0.1, help="initial ES mutation scale")
    evo.add_argument("--seed", type=int, default=0, help="PRNG seed (first seed)")
    evo.add_argument("--n-seeds", type=int, default=1, help="run this many consecutive seeds")
    evo.add_argument("--target", type=float, default=1.0, help="stop a seed early once best accuracy reaches this")
    evo.add_argument("--no-early-stop", action="store_true",
                     help="never stop early, even if --target is reached (it is already ignored "
                          "under --mvg). USE THIS FOR ANY FG-vs-MVG COMPARISON: otherwise the FG "
                          "arm exits as soon as it solves the task while the MVG arm runs the "
                          "full budget, so the two arms differ in generations AND in how much "
                          "post-solution drift they were exposed to -- the very structural "
                          "difference the comparison is measuring")

    # --- task -----------------------------------------------------------------
    task = p.add_argument_group("task")
    task.add_argument("--task", default="retina",
                      choices=tasks.TASKS,   # single source of truth: ../shared_tasks.py
                      help="task: copy (1 bit) | and2 (2 bits) | left (4 bits) | "
                           "retina (8 bits, stand-in object rule) | "
                           "retina_ka2005 (8 bits, the ORIGINAL Kashtan-Alon 2005 object "
                           "rule -- equal (L,R) cells, so raw accuracy caps every "
                           "one-module solution at 0.75; prefer this for modularity claims)")
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
        n_types=args.n_types,
        type_dim=args.type_dim,
        pos_dim=args.pos_dim,
        g_width=args.g_width,
        g_depth=args.g_depth,
        rnn_iters=args.rnn_iters,
        use_bias=not args.no_bias,
        activation=ACTIVATIONS[args.activation],
        w_threshold=args.w_threshold,
        synaptic_budget=args.synaptic_budget,
        shrink=args.shrink,
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
        early_stop=not args.no_early_stop,
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
    p = build_parser()
    args = p.parse_args(argv)
    if args.synaptic_budget > 0.0 and args.w_threshold > 0.0:
        p.error("--synaptic-budget and --w-threshold are alternative sparsity "
                "mechanisms; set only one. The budget subsumes the gate: --shrink "
                "zeroes weak synapses relative to each neuron's own scale, which is "
                "the part the absolute gate got wrong")
    if not 0.0 <= args.shrink < 1.0:
        p.error("--shrink must be in [0, 1): it is a FRACTION of each target "
                "neuron's mean incoming |g|, not an absolute weight")
    if args.shrink > 0.0 and args.synaptic_budget <= 0.0:
        p.error("--shrink has no meaning without --synaptic-budget > 0 "
                "(there is no budget to share out)")
    if args.synaptic_budget < 0.0:
        p.error("--synaptic-budget must be >= 0 (0 = off)")
    return build_brain_config(args), build_run_config(args), args


if __name__ == "__main__":
    brain_cfg, run_cfg, _ = parse_args()
    print("BrainConfig:")
    for f in dataclasses.fields(brain_cfg):
        print(f"  {f.name:14s} = {getattr(brain_cfg, f.name)}")
    print(f"  {'n_total':14s} = {brain_cfg.n_total}")
    print(f"  {'feat_dim':14s} = {brain_cfg.feat_dim}")
    print("\nRunConfig:")
    for f in dataclasses.fields(run_cfg):
        print(f"  {f.name:16s} = {getattr(run_cfg, f.name)}")
