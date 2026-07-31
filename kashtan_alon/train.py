"""Kashtan-Alon retina experiment: does MVG favour a modular brain?

Evolves a layered feedforward net (model.py) with a discrete mutation GA (ga.py)
on the retina task, under either:
  * --mvg           : Modularly-Varying Goals -- alternate AND <-> OR every
                      --switch-interval generations (shared L/R sub-goals).
  * (default, FG)   : Fixed Goal -- one operation the whole run (the control).

Each generation logs the best network's Newman Q (modularity), density and
fitness, and appends them to runs/<name>_log.csv so the MVG-vs-FG Q trajectories
can be compared. Kashtan-Alon's claim: Q climbs and stays high under MVG but not
under a fixed goal.

Run `python train.py --help` for all flags. This is a reference reproduction; run
long experiments in a per-experiment chat / on the GPU box, not the hub.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import tasks
import model as M
import ga
from modularity import newman_q, density, n_edges


def _encode(X, encoding):
    return X * 2.0 - 1.0 if encoding == "bipolar" else X


def build_parser():
    p = argparse.ArgumentParser(
        description="Kashtan-Alon retina: MVG vs fixed-goal, with Newman-Q modularity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # architecture
    p.add_argument("--layers", type=str, default="8,8,4,2,1", help="feedforward layer sizes (comma-sep)")
    p.add_argument("--lam", type=float, default=20.0, help="tanh steepness (near-sign at 20)")
    p.add_argument("--input-encoding", choices=["bipolar", "binary"], default="bipolar")
    # search / GA
    p.add_argument("--pop", type=int, default=1000, help="population size (Kashtan-Alon used ~1000)")
    p.add_argument("--generations", type=int, default=25000, help="generations (paper: 25000)")
    p.add_argument("--init-density", type=float, default=0.5, help="prob each connection is present at init")
    p.add_argument("--p-add", type=float, default=0.2, help="prob a network adds one connection")
    p.add_argument("--p-remove", type=float, default=0.2, help="prob a network removes one connection")
    p.add_argument("--bias-prob", type=float, default=1.0 / 24.0, help="per-node bias mutation prob")
    p.add_argument("--tournament-k", type=int, default=3, help="tournament size for selection")
    p.add_argument("--n-elite", type=int, default=1, help="elites carried forward unmutated")
    p.add_argument("--fitness", choices=["raw", "balanced"], default="raw",
                   help="raw = Kashtan-Alon fraction-correct (paper default); "
                        "balanced = thesis balanced accuracy (chance 0.5, shortcut-safe)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--target", type=float, default=1.0,
                   help="fitness threshold for --early-stop (only used when early stopping is on)")
    p.add_argument("--early-stop", dest="early_stop", action="store_true", default=False,
                   help="stop a fixed-goal run once best fitness >= --target [default: OFF -> run all gens]")
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                   help="run the full --generations regardless of fitness (the default)")
    # task
    p.add_argument("--task", default="retina", help="task from shared_tasks (retina is the KA one)")
    p.add_argument("--operation", choices=["and", "or", "xor"], default="and",
                   help="fixed-goal operation (ignored under --mvg)")
    p.add_argument("--mvg", action="store_true", help="modularly-varying goal: alternate the ops below")
    p.add_argument("--mvg-ops", type=str, default="and,or", help="ops to cycle under --mvg (comma-sep)")
    p.add_argument("--switch-interval", type=int, default=20, help="generations between goal switches (KA: 20)")
    # analysis / logging
    p.add_argument("--weighted-q", action="store_true", help="use |weight| edge weights in Newman Q")
    p.add_argument("--log-interval", type=int, default=10, help="generations between log lines / CSV rows")
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--viz", dest="viz", action="store_true", default=True,
                   help="render the best network, K-A-style, coloured by module [default: on]")
    p.add_argument("--no-viz", dest="viz", action="store_false",
                   help="disable the end-of-run visualisation")
    p.add_argument("--open", dest="open_img", action="store_true", default=True,
                   help="open the final brain image when the run ends [default: on]")
    p.add_argument("--no-open", dest="open_img", action="store_false",
                   help="save the image but don't auto-open it")
    # checkpointing / resume (for long runs)
    p.add_argument("--checkpoint-interval", type=int, default=1000,
                   help="generations between full-state checkpoints (0 = off); enables --resume")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="resume from a checkpoint and skip already-finished seeds [default: on]")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="ignore any existing checkpoint/result and start fresh (overwrites logs)")
    return p


def run_name_for(args, seed):
    return f"{args.task}_{'mvg' if args.mvg else 'fg'}_{args.fitness}_seed{seed}"


def _save_best_npz(path, best_indiv, cfg):
    wm, bm = best_indiv
    save = {f"w{l}": wm[l] for l in range(cfg.n_blocks)}
    save.update({f"b{l}": bm[l] for l in range(cfg.n_blocks)})
    np.savez(path, **save)


def save_checkpoint(path, gen, rng, weights, biases, best_fit, best_indiv, best_op):
    """Atomically pickle full GA state so a run can resume exactly where it stopped."""
    obj = {"gen": gen, "rng_state": rng.bit_generator.state,
           "weights": weights, "biases": biases,
           "best_fit": best_fit, "best_weights": best_indiv[0],
           "best_biases": best_indiv[1], "best_op": best_op}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)   # atomic: a crash mid-write can't corrupt the checkpoint


def train_seed(cfg, X, X_bits, args, seed, open_after=False):
    rng = np.random.default_rng(seed)
    mvg_ops = tuple(o.strip() for o in args.mvg_ops.split(","))

    def goal_op(gen):
        if not args.mvg:
            return args.operation
        return mvg_ops[(gen // args.switch_interval) % len(mvg_ops)]

    mode = f"MVG({'/'.join(mvg_ops)})" if args.mvg else f"FG({args.operation})"
    run_name = run_name_for(args, seed)
    csv_path = os.path.join(args.out_dir, f"{run_name}_log.csv")
    npz_path = os.path.join(args.out_dir, f"{run_name}_best.npz")
    ckpt_path = os.path.join(args.out_dir, f"{run_name}_ckpt.pkl")
    result_path = os.path.join(args.out_dir, f"{run_name}_result.json")

    # --- resume from a mid-run checkpoint, or start fresh ---
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            c = pickle.load(f)
        rng.bit_generator.state = c["rng_state"]
        weights, biases = c["weights"], c["biases"]
        start_gen, best_fit, best_op = c["gen"], c["best_fit"], c["best_op"]
        best_indiv = (c["best_weights"], c["best_biases"])
        csv_f = open(csv_path, "a", newline="")
        writer = csv.writer(csv_f)
        print(f"[seed {seed}] RESUME {mode} from gen {start_gen}/{args.generations} "
              f"(best {best_fit:.3f})")
    else:
        weights, biases = M.init_population(rng, cfg, args.pop, args.init_density)
        start_gen, best_fit, best_indiv, best_op = 0, -1.0, None, None
        csv_f = open(csv_path, "w", newline="")
        writer = csv.writer(csv_f)
        writer.writerow(["gen", "op", "best_fit", "mean_fit", "Q", "density", "edges"])
        print(f"[seed {seed}] mode={mode} fitness={args.fitness} task={args.task} "
              f"layers={list(cfg.layers)} pop={args.pop} max_edges={cfg.max_edges} "
              f"switch={args.switch_interval if args.mvg else '-'} gens={args.generations} "
              f"ckpt_every={args.checkpoint_interval}")

    cur_op, y = None, None
    stopped_gen = args.generations - 1
    # The network we report/save/visualise is the LAST generation's champion (the
    # evolved topology), not `best_indiv` (the first net to hit peak fitness, which
    # under a fixed goal freezes very early and is un-evolved). `best_*` is kept only
    # to report peak performance ("did it ever solve the task?").
    final_indiv, final_fit, final_op = best_indiv, best_fit, best_op
    t0 = time.time()
    for gen in range(start_gen, args.generations):
        op = goal_op(gen)
        if op != cur_op:
            cur_op, y = op, np.asarray(tasks.targets(args.task, op, X_bits))

        fit = M.fitness(weights, biases, X, y, cfg, args.fitness)
        gi = int(fit.argmax())
        gen_best = float(fit[gi])
        if gen_best > best_fit:
            best_fit, best_op = gen_best, cur_op
            best_indiv = M.individual(weights, biases, gi)
        # overwrite every gen -> after the loop this holds the last generation's champion
        final_indiv, final_fit, final_op = M.individual(weights, biases, gi), gen_best, cur_op

        if gen % args.log_interval == 0 or gen == args.generations - 1:
            wm, _ = M.individual(weights, biases, gi)
            q, _ = newman_q(wm, cfg, weighted=args.weighted_q)
            dens = density(wm, cfg)
            writer.writerow([gen, cur_op, f"{gen_best:.4f}", f"{float(fit.mean()):.4f}",
                             f"{q:.4f}", f"{dens:.2f}", n_edges(wm)])
            csv_f.flush()   # intermediary results readable mid-run
            sps = (time.time() - t0) / (gen - start_gen + 1)
            op_str = f" | op: {cur_op}" if args.task == "retina" else ""
            print(f"  Gen {gen:5d} | Best: {gen_best:.3f} | Mean: {float(fit.mean()):.3f}"
                  f" | Q: {q:.3f} | Density: {dens:.1f}%{op_str} | {sps:.3f}s/gen")

        if args.early_stop and (not args.mvg) and best_fit >= args.target:
            print(f"  early stop: best {best_fit:.3f} >= target {args.target:.3f} at gen {gen}")
            stopped_gen = gen
            break

        # --- reproduce: elitism + tournament + mutation ---
        order = np.argsort(fit)[::-1]
        elite_w, elite_b = M.gather(weights, biases, order[:args.n_elite])
        parents = ga.tournament_select(rng, fit, args.pop, args.tournament_k)
        weights, biases = M.gather(weights, biases, parents)
        ga.mutate(rng, weights, biases, cfg, args.p_add, args.p_remove, args.bias_prob)
        for l in range(cfg.n_blocks):
            weights[l][:args.n_elite] = elite_w[l]
            biases[l][:args.n_elite] = elite_b[l]

        # --- periodic checkpoint (full state) + best-so-far snapshot ---
        if args.checkpoint_interval and (gen + 1) % args.checkpoint_interval == 0 \
                and best_indiv is not None:
            save_checkpoint(ckpt_path, gen + 1, rng, weights, biases, best_fit, best_indiv, best_op)
            _save_best_npz(npz_path, best_indiv, cfg)
            print(f"  [checkpoint @ gen {gen + 1}] -> {os.path.basename(ckpt_path)} | "
                  f"best {best_fit:.3f} -> {os.path.basename(npz_path)}")

    csv_f.close()
    # Modularity is reported for the FINAL generation's champion (evolved topology).
    wm, _ = final_indiv
    q, comm = newman_q(wm, cfg, weighted=args.weighted_q)
    print(f"[seed {seed}] final-gen fit {final_fit:.3f} (op={final_op}) | Q {q:.3f} | "
          f"density {density(wm, cfg):.1f}% | edges {n_edges(wm)} | modules {len(comm)} "
          f"| peak fit {best_fit:.3f} (op={best_op})")

    _save_best_npz(npz_path, final_indiv, cfg)
    print(f"[seed {seed}] final network -> {npz_path}")

    # completion marker: lets a rerun (and run_paper.py) skip finished seeds.
    # `best_fit`/`best_op` = peak performance; `q` is now the FINAL-gen net's modularity.
    with open(result_path, "w") as f:
        json.dump({"run_name": run_name, "mode": mode, "seed": seed,
                   "q": q, "final_fit": final_fit, "final_op": final_op,
                   "best_fit": best_fit, "best_op": best_op,
                   "density": density(wm, cfg), "edges": n_edges(wm),
                   "modules": len(comm), "gen_reached": stopped_gen}, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)   # run finished -> checkpoint no longer needed
    print(f"[seed {seed}] result -> {result_path}")

    if args.viz:
        import visualize
        png = os.path.join(args.out_dir, f"{run_name}_best.png")
        visualize.visualize_net(final_indiv, cfg, png,
                                title=f"{mode} seed{seed} - final fit {final_fit:.3f} Q {q:.3f}",
                                weighted_q=args.weighted_q, open_after=open_after)
    return best_fit, q


def main():
    args = build_parser().parse_args()
    layers = tuple(int(x) for x in args.layers.split(","))
    cfg = M.NetConfig(layers=layers, lam=args.lam)
    assert layers[0] >= tasks.min_inputs(args.task), \
        f"task {args.task!r} needs >= {tasks.min_inputs(args.task)} input pixels"
    os.makedirs(args.out_dir, exist_ok=True)

    X_bits = np.asarray(tasks.all_binary_inputs(layers[0]))     # {0,1} for targets()
    X = _encode(X_bits, args.input_encoding)                    # encoded for the net

    results = []
    for i in range(args.n_seeds):
        seed = args.seed + i
        result_path = os.path.join(args.out_dir, f"{run_name_for(args, seed)}_result.json")
        if args.resume and os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            print(f"[seed {seed}] already complete (best {r['best_fit']:.3f} | Q {r['q']:.3f}) -> skip")
            results.append((seed, r["best_fit"], r["q"]))
            continue
        open_after = args.viz and args.open_img and (i == args.n_seeds - 1)
        results.append((seed, *train_seed(cfg, X, X_bits, args, seed, open_after)))

    if args.n_seeds > 1:
        print("\n=== summary ===")
        for seed, bf, q in results:
            print(f"  seed {seed}: best fit {bf:.3f} | Q {q:.3f}")
        print(f"mean Q {np.mean([r[2] for r in results]):.3f}  |  "
              f"mean best fit {np.mean([r[1] for r in results]):.3f}")


if __name__ == "__main__":
    main()
