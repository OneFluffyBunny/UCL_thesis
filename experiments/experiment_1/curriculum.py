"""Curriculum vs cold-start experiment (experiment 1).

Question: does a brain reach a hard task *faster* — and does it end up
*structurally different* — when it is grown on an incrementally harder sequence
of tasks, versus being thrown at the hardest task from scratch?

Biological framing: an organism does not learn a hard task from nothing. It gets
good at an easy task, then the environment changes / competition arises, and the
same fixed sensory apparatus is re-purposed for harder problems. Here the brain
keeps a FIXED input count (n_in, the "eyes") across the whole schedule; only the
*task* — i.e. which inputs matter and how they combine — gets harder. Easy tasks
simply ignore some inputs (`left` reads pixels 0-3; `retina` reads all 8).

Two arms, matched on total generation budget:

  * curriculum : one continuous CMA-ES run stepped through a schedule of stages,
                 e.g. left(200 gens) -> retina(400 gens). The population (CMA
                 mean/cov) is carried across stage boundaries -- the head start.
  * cold       : a fresh CMA-ES run spending ALL of those generations on the
                 hardest (final) task alone.

We report, on the hard task: (a) generations-to-threshold, both counted from the
start of the hard stage AND cumulatively, and (b) a structural fingerprint of the
two final brains (density, exc/inh balance, active cell-types, and a provisional
left/right input-segregation proxy) to see if they are "fundamentally different".

NOTE: the segregation proxy is a stand-in for a real modularity metric (graph Q /
block structure), which is still an open thread. It is retina-specific (splits
inputs into a left and a right half) but directly relevant to the retina's
modular decomposition.

Example:
  python curriculum.py --curriculum "left:200,retina/and:400" -K 6 --n-hidden 30 \
      --n-seeds 3 --fitness margin
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import dataclasses
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import evosax as ex
import numpy as np

import tasks
from model import Genome, BrainConfig
from train import _make_eval, _encode
from config import ACTIVATIONS
from visualize import brain_stats, visualize_brain


# ---------------------------------------------------------------------------
# schedule spec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Stage:
    task: str
    op: str
    gens: int

    def label(self) -> str:
        return f"{self.task}{'/' + self.op if self.task == 'retina' else ''}({self.gens})"


def parse_schedule(spec: str) -> list[Stage]:
    """'left:200,retina/and:400' -> [Stage(left,and,200), Stage(retina,and,400)].

    Each comma-separated entry is `task[/op]:gens`. `op` defaults to 'and' and
    only affects retina.
    """
    stages = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, gens = chunk.partition(":")
        if not gens:
            raise ValueError(f"stage {chunk!r} missing ':gens' (e.g. left:200)")
        task, _, op = name.partition("/")
        task, op = task.strip(), (op.strip() or "and")
        if task not in tasks.TASKS:
            raise ValueError(f"unknown task {task!r} (known: {tasks.TASKS})")
        stages.append(Stage(task=task, op=op, gens=int(gens)))
    if not stages:
        raise ValueError("empty curriculum")
    return stages


# ---------------------------------------------------------------------------
# core: run one CMA-ES trajectory through a schedule of stages
# ---------------------------------------------------------------------------

def evolve(brain_cfg, schedule, *, seed, popsize, sigma_init, sigma_restart,
           fitness, balanced, strategy_name, thresholds, log_interval,
           prune_threshold, arm_label):
    """Run a single continuous CMA-ES trajectory over `schedule`.

    Returns (best_final_genome, static, reshaper, per_stage_history). The returned
    genome is the best-accuracy member found during the FINAL stage (the brain the
    organism ends life with).
    """
    key = jr.PRNGKey(seed)
    X = tasks.all_binary_inputs(brain_cfg.n_in)
    X_enc = _encode(X, "bipolar")

    key, init_key = jr.split(key)
    template = Genome.init(init_key, brain_cfg)
    params, static = eqx.partition(template, eqx.is_inexact_array)
    reshaper = ex.ParameterReshaper(params)
    batched_eval = _make_eval(static, reshaper, brain_cfg, X_enc, balanced, fitness)

    strategy = ex.Strategies[strategy_name](
        popsize=popsize, num_dims=reshaper.total_params, sigma_init=sigma_init)
    es_params = strategy.default_params
    key, es_key = jr.split(key)
    state = strategy.initialize(es_key, es_params)
    shaper = ex.FitnessShaper(maximize=True)

    history = []           # per-stage summary (speed table, best-per-stage)
    rows = []              # per-generation trace (for convergence curves / CSV)
    best_flat_final = None
    cum_gens = 0
    for si, stage in enumerate(schedule):
        y = tasks.targets(stage.task, stage.op, X)

        # at a stage boundary the mutation scale has usually collapsed onto the
        # easy-task optimum; re-inflate it so the harder task is explorable again
        if si > 0 and sigma_restart > 0:
            try:
                state = state.replace(sigma=sigma_restart)
            except Exception:
                pass

        stage_best, stage_best_flat = -1.0, None
        reached = {t: None for t in thresholds}   # gens into THIS stage
        reached_cum = {t: None for t in thresholds}
        t0 = time.time()
        for g in range(stage.gens):
            key, ask_key = jr.split(key)
            x, state = strategy.ask(ask_key, state, es_params)
            sel, acc = batched_eval(x, y)
            state = strategy.tell(x, shaper.apply(x, sel), state, es_params)

            best_idx = int(acc.argmax())
            gen_best = float(acc[best_idx])
            if gen_best > stage_best:
                stage_best, stage_best_flat = gen_best, jnp.asarray(x[best_idx])
            for t in thresholds:
                if reached[t] is None and gen_best >= t:
                    reached[t], reached_cum[t] = g, cum_gens

            try:
                cur_sigma = float(state.sigma)
            except AttributeError:
                cur_sigma = float("nan")
            sel_max = float(sel.max()) if fitness != "accuracy" else float("nan")
            genome = eqx.combine(reshaper.reshape_single(x[best_idx]), static)
            density = brain_stats(genome, brain_cfg, prune_threshold)["density"]

            # per-generation trace row (every gen -> full convergence curve)
            rows.append(dict(
                arm=arm_label, seed=seed, global_gen=cum_gens, stage_idx=si,
                task=stage.task, op=stage.op, stage_gen=g,
                best_acc=gen_best, stage_best_acc=stage_best,
                pop_mean=float(acc.mean()), sel_max=sel_max,
                sigma=cur_sigma, density=density))

            if g % log_interval == 0 or g == stage.gens - 1:
                sig = "" if cur_sigma != cur_sigma else f" | σ: {cur_sigma:.3f}"
                sel_str = f" | Sel: {sel_max:.3f}" if fitness != "accuracy" else ""
                sps = (time.time() - t0) / (g + 1)
                print(f"  [{arm_label}] {stage.label():>14s} gen {g:4d} | "
                      f"Best: {gen_best:.3f}{sel_str} | Pop mean: {float(acc.mean()):.3f}"
                      f" | Density: {density:.1f}%{sig} | {sps:.2f}s/gen")
            cum_gens += 1

        history.append(dict(idx=si, stage=stage, best=stage_best,
                            reached=reached, reached_cum=reached_cum,
                            cum_gens_end=cum_gens))
        best_flat_final = stage_best_flat

    best_genome = eqx.combine(reshaper.reshape_single(best_flat_final), static)
    return best_genome, static, reshaper, history, rows


# ---------------------------------------------------------------------------
# structural comparison of two final brains
# ---------------------------------------------------------------------------

def input_segregation(genome, cfg, threshold):
    """Provisional left/right modularity proxy for the retina.

    For each hidden neuron, compare total |weight| coming from the LEFT input half
    (pixels 0..n_in/2-1) vs the RIGHT half. s = (L-R)/(L+R) in [-1,+1]:
      s ~ +1 -> neuron driven only by left pixels (a left-module neuron)
      s ~ -1 -> right-module neuron
      s ~  0 -> mixed (non-segregated)
    A modular retina brain has hidden neurons piled at s ~ +/-1 (bimodal); a
    tangled one has them near 0. Reported: mean |s| over wired hidden neurons and
    the fraction that are strongly lateralized (|s| > 0.5).
    """
    w = np.asarray(genome.build_weights(cfg)[0])
    n_in = cfg.n_in
    half = n_in // 2
    W_in_hid = np.abs(w[:n_in, n_in:n_in + cfg.n_hidden])   # (n_in, n_hidden)
    left = W_in_hid[:half].sum(0)
    right = W_in_hid[half:].sum(0)
    tot = left + right
    active = tot > threshold
    s = np.where(tot > 1e-9, (left - right) / (tot + 1e-9), 0.0)
    if active.any():
        mean_abs_s = float(np.abs(s[active]).mean())
        frac_lat = float((np.abs(s[active]) > 0.5).mean())
    else:
        mean_abs_s, frac_lat = 0.0, 0.0
    return dict(mean_abs_s=mean_abs_s, frac_lateralized=frac_lat,
                n_active_hidden=int(active.sum()), per_hidden_s=s)


def fingerprint(genome, cfg, threshold):
    st = brain_stats(genome, cfg, threshold)
    seg = input_segregation(genome, cfg, threshold)
    active_types = int(sum(1 for c in st["type_counts"] if c > 0))
    n_edges = st["n_edges"]
    sign = ("mixed" if st["n_exc"] and st["n_inh"]
            else "all-exc" if st["n_exc"] else "all-inh" if st["n_inh"] else "none")
    return dict(density=st["density"], n_exc=st["n_exc"], n_inh=st["n_inh"],
                sign=sign, active_types=active_types, type_counts=st["type_counts"],
                mean_abs_s=seg["mean_abs_s"], frac_lateralized=seg["frac_lateralized"],
                n_active_hidden=seg["n_active_hidden"], n_edges=n_edges)


# ---------------------------------------------------------------------------
# persistence (everything needed for a deep offline comparison)
# ---------------------------------------------------------------------------

import csv
import json


def save_history_csv(path, rows):
    """Per-generation trace -> CSV (one file per arm; plot convergence from this)."""
    if not rows:
        return
    fields = ["arm", "seed", "global_gen", "stage_idx", "task", "op", "stage_gen",
              "best_acc", "stage_best_acc", "pop_mean", "sel_max", "sigma", "density"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def save_weights(path, genome, cfg):
    """Grown weight matrix + hidden type ids -> .npz (offline structural analysis)."""
    w, hid_ids = genome.build_weights(cfg)
    np.savez(path, w=np.asarray(w), hid_ids=np.asarray(hid_ids),
             n_in=cfg.n_in, n_hidden=cfg.n_hidden, n_out=cfg.n_out, n_types=cfg.n_types)


def _reached_json(hist, thresholds):
    hard = hist[-1]
    return {f"{t:.3f}": dict(into_stage=hard["reached"][t],
                             cumulative=hard["reached_cum"][t])
            for t in thresholds}


def save_summary_json(path, seed, schedule, cur_hist, cold_hist, cur_fp, cold_fp,
                      thresholds, brain_cfg, fitness, sigma_restart):
    """Machine-readable summary of one seed: speed + final structure, both arms."""
    def stage_dump(h):
        return [dict(task=s["stage"].task, op=s["stage"].op, gens=s["stage"].gens,
                     best_acc=s["best"], cum_gens_end=s["cum_gens_end"]) for s in h]
    out = dict(
        seed=seed, fitness=fitness, sigma_restart=sigma_restart,
        brain=dict(n_in=brain_cfg.n_in, n_hidden=brain_cfg.n_hidden,
                   n_types=brain_cfg.n_types, g_width=brain_cfg.g_width),
        curriculum=dict(schedule=[st.label() for st in schedule],
                        stages=stage_dump(cur_hist),
                        best_acc=cur_hist[-1]["best"],
                        reached=_reached_json(cur_hist, thresholds),
                        structure=cur_fp),
        cold=dict(stages=stage_dump(cold_hist),
                  best_acc=cold_hist[-1]["best"],
                  reached=_reached_json(cold_hist, thresholds),
                  structure=cold_fp),
    )
    # per_hidden_s arrays are numpy -> make json-safe
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(path, "w") as f:
        json.dump(_clean(out), f, indent=2)


def save_convergence_plot(path, cur_rows, cold_rows, schedule, hard):
    """Best-accuracy-vs-generation for both arms on one axis.

    Curriculum's easy-stage portion is drawn lightly (it's scored on a different,
    easier task); its hard-stage portion and the whole cold curve are solid, so
    convergence ON THE HARD TASK is directly comparable. Stage boundaries dashed.
    """
    import matplotlib.pyplot as plt
    cur = np.array([(r["global_gen"], r["best_acc"], r["stage_idx"]) for r in cur_rows])
    cold = np.array([(r["global_gen"], r["best_acc"]) for r in cold_rows])
    hard_idx = len(schedule) - 1

    fig, ax = plt.subplots(figsize=(10, 5))
    # curriculum: easy stages faint, hard stage solid
    easy = cur[cur[:, 2] < hard_idx]
    hard_seg = cur[cur[:, 2] == hard_idx]
    if len(easy):
        ax.plot(easy[:, 0], easy[:, 1], color="#9B59B6", alpha=0.35, lw=1.3,
                label="curriculum (easy stages)")
    ax.plot(hard_seg[:, 0], hard_seg[:, 1], color="#9B59B6", lw=2.0,
            label="curriculum (hard stage)")
    ax.plot(cold[:, 0], cold[:, 1], color="#34495E", lw=2.0, label="cold (hard task)")

    # stage boundaries
    b = 0
    for st in schedule[:-1]:
        b += st.gens
        ax.axvline(b, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("generation (cumulative)")
    ax.set_ylabel("best balanced accuracy")
    ax.set_title(f"curriculum vs cold — hard task: {hard.task}"
                 f"{'/' + hard.op if hard.task == 'retina' else ''}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Convergence plot saved -> {path}")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(v):
    return "  — " if v is None else f"{v:4d}"


def report_speed(cur_hist, cold_hist, thresholds):
    """Print gens-to-threshold on the hard task for both arms."""
    cur_hard = cur_hist[-1]          # curriculum's final (hard) stage
    cold_hard = cold_hist[-1]        # cold arm has a single stage = the hard task
    hard = cur_hard["stage"]
    print(f"\n=== speed on hard task [{hard.task}"
          f"{'/' + hard.op if hard.task == 'retina' else ''}] ===")
    print("  (gens to first reach each accuracy threshold)")
    print(f"  {'thresh':>7s} | {'curric (into stage)':>19s} | "
          f"{'curric (cumulative)':>19s} | {'cold (from 0)':>13s}")
    for t in thresholds:
        print(f"  {t:7.3f} | {_fmt(cur_hard['reached'][t]):>19s} | "
              f"{_fmt(cur_hard['reached_cum'][t]):>19s} | "
              f"{_fmt(cold_hard['reached'][t]):>13s}")
    print(f"  best acc: curriculum {cur_hard['best']:.3f}  |  cold {cold_hard['best']:.3f}")


def report_structure(cur_fp, cold_fp):
    print("\n=== final brain structure (do they look different?) ===")
    rows = [
        ("best-stage density %", f"{cur_fp['density']:.1f}", f"{cold_fp['density']:.1f}"),
        ("edges", f"{cur_fp['n_edges']}", f"{cold_fp['n_edges']}"),
        ("sign", cur_fp["sign"], cold_fp["sign"]),
        ("exc / inh", f"{cur_fp['n_exc']}/{cur_fp['n_inh']}", f"{cold_fp['n_exc']}/{cold_fp['n_inh']}"),
        ("active cell-types", f"{cur_fp['active_types']}", f"{cold_fp['active_types']}"),
        ("type counts", f"{cur_fp['type_counts']}", f"{cold_fp['type_counts']}"),
        ("L/R mean|s| (modular?)", f"{cur_fp['mean_abs_s']:.2f}", f"{cold_fp['mean_abs_s']:.2f}"),
        ("frac lateralized", f"{cur_fp['frac_lateralized']:.2f}", f"{cold_fp['frac_lateralized']:.2f}"),
    ]
    w0 = max(len(r[0]) for r in rows)
    print(f"  {'metric':<{w0}} | {'curriculum':>12s} | {'cold':>12s}")
    for name, a, b in rows:
        print(f"  {name:<{w0}} | {a:>12s} | {b:>12s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Curriculum vs cold-start on the experiment-1 framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--curriculum", default="left:200,retina/and:400",
                   help="stages as 'task[/op]:gens,...'; last stage = the hard task. "
                        "The cold arm spends the SAME total gens on that hard task alone.")

    # architecture (defaults tuned for retina: K>=6 makes retina representable)
    p.add_argument("--n-in", type=int, default=8)
    p.add_argument("--n-hidden", type=int, default=30)
    p.add_argument("--n-out", type=int, default=1)
    p.add_argument("--n-types", "-K", type=int, default=6)
    p.add_argument("--type-dim", type=int, default=4)
    p.add_argument("--pos-dim", type=int, default=4)
    p.add_argument("--g-width", type=int, default=16)
    p.add_argument("--g-depth", type=int, default=1)
    p.add_argument("--rnn-iters", type=int, default=8)
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--activation", choices=sorted(ACTIVATIONS), default="tanh")

    # search
    p.add_argument("--strategy", default="CMA_ES")
    p.add_argument("--fitness", choices=["accuracy", "margin"], default="accuracy")
    p.add_argument("--popsize", type=int, default=64)
    p.add_argument("--sigma-init", type=float, default=0.1)
    p.add_argument("--sigma-restart", type=float, default=0.1,
                   help="re-inflate CMA sigma to this at each new curriculum stage "
                        "(0 = leave it; the collapsed easy-task sigma carries over)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)

    # analysis / logging
    p.add_argument("--thresholds", default="0.90,0.95,0.99,1.0",
                   help="accuracy thresholds for the speed table")
    p.add_argument("--prune-threshold", type=float, default=0.05)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--out-dir", default="./runs/curriculum")
    return p


def main():
    args = build_parser().parse_args()
    schedule = parse_schedule(args.curriculum)
    total_gens = sum(s.gens for s in schedule)
    hard = schedule[-1]
    cold_schedule = [Stage(task=hard.task, op=hard.op, gens=total_gens)]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    brain_cfg = BrainConfig(
        n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out, n_types=args.n_types,
        type_dim=args.type_dim, pos_dim=args.pos_dim, g_width=args.g_width,
        g_depth=args.g_depth, rnn_iters=args.rnn_iters, use_bias=not args.no_bias,
        activation=ACTIVATIONS[args.activation])

    need = max(tasks.min_inputs(s.task) for s in schedule)
    assert brain_cfg.n_in >= need, f"n_in={brain_cfg.n_in} < {need} required by schedule"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"curriculum : {' -> '.join(s.label() for s in schedule)}  (total {total_gens} gens)")
    print(f"cold       : {cold_schedule[0].label()}  (same {total_gens} gens on hard task)")
    print(f"brain      : n_in={brain_cfg.n_in} n_hidden={brain_cfg.n_hidden} "
          f"K={brain_cfg.n_types} g_width={brain_cfg.g_width} | fitness={args.fitness}\n")

    def run_arm(sched, seed, label):
        return evolve(brain_cfg, sched, seed=seed, popsize=args.popsize,
                      sigma_init=args.sigma_init, sigma_restart=args.sigma_restart,
                      fitness=args.fitness, balanced=not args.no_balanced,
                      strategy_name=args.strategy, thresholds=thresholds,
                      log_interval=args.log_interval,
                      prune_threshold=args.prune_threshold, arm_label=label)

    agg = []
    for i in range(args.n_seeds):
        seed = args.seed + i
        print(f"\n########## seed {seed} ##########")
        print("--- curriculum arm ---")
        cur_g, _, _, cur_h, cur_rows = run_arm(schedule, seed, "cur")
        print("--- cold arm ---")
        cold_g, _, _, cold_h, cold_rows = run_arm(cold_schedule, seed, "cold")

        cur_fp = fingerprint(cur_g, brain_cfg, args.prune_threshold)
        cold_fp = fingerprint(cold_g, brain_cfg, args.prune_threshold)
        report_speed(cur_h, cold_h, thresholds)
        report_structure(cur_fp, cold_fp)

        # ---- persist everything for a deep offline comparison ----
        d = args.out_dir
        pre = os.path.join(d, f"seed{seed}")
        save_history_csv(f"{pre}_curriculum_history.csv", cur_rows)
        save_history_csv(f"{pre}_cold_history.csv", cold_rows)
        eqx.tree_serialise_leaves(f"{pre}_curriculum_dna.eqx", cur_g)
        eqx.tree_serialise_leaves(f"{pre}_cold_dna.eqx", cold_g)
        save_weights(f"{pre}_curriculum_weights.npz", cur_g, brain_cfg)
        save_weights(f"{pre}_cold_weights.npz", cold_g, brain_cfg)
        save_summary_json(f"{pre}_summary.json", seed, schedule, cur_h, cold_h,
                          cur_fp, cold_fp, thresholds, brain_cfg, args.fitness,
                          args.sigma_restart)
        save_convergence_plot(f"{pre}_convergence.png", cur_rows, cold_rows,
                              schedule, hard)

        open_it = (not args.no_open) and args.n_seeds == 1
        for tag, g, fp in (("curriculum", cur_g, cur_fp), ("cold", cold_g, cold_fp)):
            png = f"{pre}_{tag}_brain.png"
            visualize_brain(g, brain_cfg, args.prune_threshold, png,
                            title=f"{tag} seed{seed} - {hard.task} "
                                  f"acc {fp['density']:.0f}%den | L/R|s| {fp['mean_abs_s']:.2f}",
                            open_after=open_it)
        print(f"[seed {seed}] artifacts -> {d}\\seed{seed}_*")
        agg.append((seed, cur_h[-1]["best"], cold_h[-1]["best"], cur_fp, cold_fp))

    # aggregate across seeds -> one CSV for the whole run
    agg_path = os.path.join(args.out_dir, "aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seed", "cur_best_acc", "cold_best_acc",
                     "cur_density", "cold_density", "cur_sign", "cold_sign",
                     "cur_mean_abs_s", "cold_mean_abs_s",
                     "cur_frac_lat", "cold_frac_lat"])
        for seed, cb, kb, cfp, kfp in agg:
            wr.writerow([seed, f"{cb:.4f}", f"{kb:.4f}",
                         f"{cfp['density']:.1f}", f"{kfp['density']:.1f}",
                         cfp["sign"], kfp["sign"],
                         f"{cfp['mean_abs_s']:.3f}", f"{kfp['mean_abs_s']:.3f}",
                         f"{cfp['frac_lateralized']:.3f}", f"{kfp['frac_lateralized']:.3f}"])
    print(f"\naggregate -> {agg_path}")

    if args.n_seeds > 1:
        print("\n=== multi-seed summary (best acc on hard task) ===")
        print(f"  {'seed':>4s} | {'curriculum':>10s} | {'cold':>10s} | "
              f"{'cur L/R|s|':>10s} | {'cold L/R|s|':>11s}")
        for seed, cb, kb, cfp, kfp in agg:
            print(f"  {seed:4d} | {cb:10.3f} | {kb:10.3f} | "
                  f"{cfp['mean_abs_s']:10.2f} | {kfp['mean_abs_s']:11.2f}")
        cur_mean = np.mean([a[1] for a in agg])
        cold_mean = np.mean([a[2] for a in agg])
        print(f"  {'mean':>4s} | {cur_mean:10.3f} | {cold_mean:10.3f}")


if __name__ == "__main__":
    main()
