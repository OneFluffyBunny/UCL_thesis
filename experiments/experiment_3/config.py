"""Command-line configuration for experiment 3 (direct encoding + gradient descent).

Experiment 3 is the OPTIMISER axis: it trains the exact same direct-encoding
network as experiment 2, but with gradient descent (Optax) instead of CMA-ES.
So this config mirrors ``experiment_2/config.py`` field-for-field, with ONE group
swapped: the "evolution" group (popsize / sigma / strategy) becomes an
"optimisation" group (optimizer / lr / steps / loss / grad-clip). The
architecture, task and analysis groups are identical so runs are comparable.

Run `python config.py --help` for every flag, or `python config.py ...` to print
the resolved configuration.
"""

from __future__ import annotations

import argparse
import dataclasses

import jax
import jax.numpy as jnp

from model import BrainConfig


# sigma (activation) options selectable from the CLI (same set as experiments 1/2)
ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "linear": lambda x: x,
}


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Everything that is not part of the brain architecture itself.

    Mirrors experiment_2's RunConfig, with the search fields replaced by
    gradient-descent ones. ``loss`` selects the differentiable objective:
    'margin' (the exact surrogate CMA-ES maximises in exp 2 -> the fair
    optimiser-only comparison) or 'bce' (a proper logistic loss -> the
    gradient-oracle 'best a GD method can do' bound). Accuracy stays the reported
    metric in both cases (it is non-differentiable, so it is never the loss).
    """
    # optimisation (gradient descent)
    optimizer: str          # 'adam' or 'sgd'
    lr: float               # learning rate
    steps: int              # gradient steps (one full-batch step each; cf. a CMA-ES generation)
    loss: str               # 'margin' (fair surrogate) or 'bce' (logistic / gradient oracle)
    grad_clip: float        # global-norm gradient clip (<=0 disables)
    seed: int
    n_seeds: int
    target: float           # stop a seed early once best accuracy reaches this
    # task
    task: str
    operation: str
    input_encoding: str     # bipolar {-1,+1} or binary {0,1}
    mvg: bool               # modularly-varying goal: switch target every interval
    switch_interval: int    # steps between goal switches (if mvg)
    mvg_ops: tuple          # ops to cycle through under mvg, e.g. ("and", "or") or ("xor", "and")
    # analysis / logging
    prune_threshold: float  # |w| below this is treated as "no edge" for analysis
    log_interval: int
    viz_interval: int       # >0: render+open the best brain every N steps during training
    open_image: bool        # auto-open the saved brain image at the end
    balanced: bool          # balanced accuracy (chance=50%) vs raw accuracy
    out_dir: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 3: train a DIRECTLY-encoded network with gradient descent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- architecture (-> BrainConfig) --- identical to experiment 2 ----------
    arch = p.add_argument_group("architecture")
    arch.add_argument("--n-in", type=int, default=8, help="number of input neurons")
    arch.add_argument("--n-hidden", type=int, default=20, help="number of hidden neurons (fixed total)")
    arch.add_argument("--n-out", type=int, default=1, help="number of output neurons")
    arch.add_argument("--rnn-iters", type=int, default=8, help="synchronous recurrent passes at inference")
    arch.add_argument("--no-bias", action="store_true", help="disable the per-neuron bias")
    arch.add_argument("--activation", choices=sorted(ACTIVATIONS), default="tanh", help="sigma activation")

    # --- optimisation (gradient descent) --- replaces exp 2's 'evolution' -----
    opt = p.add_argument_group("optimisation")
    opt.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optax optimiser")
    opt.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    opt.add_argument("--steps", type=int, default=2000,
                     help="number of gradient steps (each is one full-batch update)")
    opt.add_argument("--loss", choices=["margin", "bce"], default="margin",
                     help="differentiable objective: 'margin' (the exact surrogate exp 2's "
                          "CMA-ES maximises -> fair optimiser-only comparison) or 'bce' "
                          "(logistic loss -> gradient-oracle upper bound). Accuracy is still "
                          "what gets logged / early-stopped / reported either way")
    opt.add_argument("--grad-clip", type=float, default=1.0,
                     help="clip gradients to this global norm (<=0 disables) -- cheap insurance "
                          "against blow-ups through the recurrent passes")
    opt.add_argument("--seed", type=int, default=0, help="PRNG seed for init (first seed)")
    opt.add_argument("--n-seeds", type=int, default=1, help="run this many consecutive seeds")
    opt.add_argument("--target", type=float, default=1.0, help="stop a seed early once best accuracy reaches this")

    # --- task --- identical to experiment 2 -----------------------------------
    task = p.add_argument_group("task")
    task.add_argument("--task", default="retina",
                      choices=["copy", "and2", "left", "retina"],
                      help="task: copy (1 bit) | and2 (2 bits) | left (4 bits) | retina (full Kashtan-Alon)")
    task.add_argument("--operation", choices=["and", "or", "xor"], default="xor", help="combining operation OP(L,R)")
    task.add_argument("--input-encoding", choices=["bipolar", "binary"], default="bipolar",
                      help="bipolar: bits -> {-1,+1}; binary: bits -> {0,1}")
    task.add_argument("--mvg", action="store_true", help="modularly-varying goal: switch target periodically")
    task.add_argument("--switch-interval", type=int, default=20, help="gradient steps between goal switches (if --mvg)")
    task.add_argument("--mvg-ops", type=str, default="and,or",
                      help="comma-separated operations to cycle through under --mvg "
                           "(each still shares the fixed left/right features; classic "
                           "Kashtan-Alon uses and,or -- pass e.g. xor,and to include xor)")

    # --- analysis / logging --- identical to experiment 2 ---------------------
    ana = p.add_argument_group("analysis")
    ana.add_argument("--prune-threshold", type=float, default=0.05, help="|w| below this = no edge (analysis only)")
    ana.add_argument("--log-interval", type=int, default=10, help="gradient steps between log lines")
    ana.add_argument("--viz-interval", type=int, default=0, help=">0: render+open best brain every N steps during training")
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
        optimizer=args.optimizer,
        lr=args.lr,
        steps=args.steps,
        loss=args.loss,
        grad_clip=args.grad_clip,
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
