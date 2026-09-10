"""Kashtan-Alon retina experiment: does MVG favour a modular brain?

Evolves a layered feedforward net (model.py) with the Kashtan-Alon GA (ga.py:
elite strategy + crossover + mutation) on the retina task, under either:
  * --mvg           : Modularly-Varying Goals -- alternate AND <-> OR every
                      --switch-interval generations (shared L/R sub-goals).
  * (default, FG)   : Fixed Goal -- one operation the whole run (the control).

Each generation logs the best network's Newman Q (modularity), left/right circuit
purity, density and fitness, and appends them to runs/<name>_log.csv so the
MVG-vs-FG trajectories
can be compared. Kashtan-Alon's claim: Q climbs and stays high under MVG but not
under a fixed goal.

The champion BRAIN at each of those logged generations is archived too, to
runs/<name>_brains.npz (see BrainArchive) -- so a metric invented later can be run
over the whole trajectory without re-evolving it. runs/<name>_best.npz remains the
single final/best network.

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
from modularity import newman_q, normalized_qm, density, n_edges

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrics import circuit_purity, from_blocks


def _encode(X, encoding):
    return X * 2.0 - 1.0 if encoding == "bipolar" else X


def purity_of(wm, cfg):
    """Circuit purity (qmetrics METRIC 4) of one champion's weight matrices.

    Left half of the retina pinned to 0, right half to 1; every downstream neuron
    takes the mean of its parents, and purity = 2*|x-0.5| averaged over the hidden
    neurons (inputs and the output neuron excluded). 1.0 = every neuron reads one
    side only; 0.0 = every neuron is a perfect 50/50 mix.
    """
    G = from_blocks([np.asarray(w, dtype=float) for w in wm],
                    offsets=cfg.offsets, directed=True)
    if G.number_of_edges() == 0:
        return float("nan")
    n_in = cfg.layers[0]
    pinned = {i: (0 if i < n_in // 2 else 1) for i in range(n_in)}
    p, _ = circuit_purity(G, pinned, exclude=[cfg.offsets[-1]])
    return p


def build_parser():
    p = argparse.ArgumentParser(
        description="Kashtan-Alon retina: MVG vs fixed-goal, with Newman-Q modularity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # architecture (KA 2005 retina: 8 pixels -> 8 -> 4 -> 2 -> 1, hard-threshold units)
    p.add_argument("--layers", type=str, default="8,8,4,2,1", help="feedforward layer sizes (comma-sep)")
    p.add_argument("--input-encoding", choices=["bipolar", "binary"], default="binary",
                   help="retina pixel encoding; KA uses binary {0,1}")
    # search / GA (KA 2005 neural-net retina: elite strategy + crossover + mutation)
    p.add_argument("--pop", type=int, default=600, help="population size S (KA neural-net: 600)")
    p.add_argument("--generations", type=int, default=25000, help="generations")
    p.add_argument("--init-density", type=float, default=0.5,
                   help="fraction of each neuron's fan-in cap filled at init (0..1)")
    p.add_argument("--n-elite", type=int, default=150, help="elite L kept unchanged each gen (KA: 150/600)")
    p.add_argument("--pc", type=float, default=0.5, help="crossover probability per offspring (KA: 0.5)")
    p.add_argument("--pm", type=float, default=0.5, help="mutation probability per genome (KA: 0.5)")
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
    p.add_argument("--qm-nrand", type=int, default=1000,
                   help="randomizations for the final normalized Q_m (KA used 1000)")
    p.add_argument("--log-interval", type=int, default=10, help="generations between log lines / CSV rows")
    p.add_argument("--dense-log", default="",
                   help="extra generation windows logged EVERY generation, "
                        "'lo:hi,lo:hi'. --log-interval 10 vs a 20-generation goal "
                        "switch gives only 2 points per phase, which cannot resolve "
                        "the post-switch recovery curve; this can. Coarse rows are "
                        "still emitted, so the CSV stays comparable to a normal run.")
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


def _save_best_npz(path, indiv, cfg):
    wm, bm = indiv
    save = {f"w{l}": wm[l] for l in range(cfg.n_blocks)}
    save.update({f"b{l}": bm[l] for l in range(cfg.n_blocks)})
    np.savez(path, **save)


class BrainArchive:
    """Every logged generation's champion brain, kept so metrics can be recomputed.

    The CSV records what we thought to measure at the time; this records the
    NETWORKS, so a new metric (or a fixed one) can be run over the whole
    trajectory later without re-evolving anything. One 8-8-4-2-1 champion is 106
    int8 weights + 15 biases, so a 25k-generation run logged every 10 gens costs
    ~300 KB -- cheap enough to be unconditional.

    Stored as `<run>_brains.npz`: `gen` (T,), `w{l}` (T, n_l, n_l+1), `b{l}` (T, n_l+1).
    Reload with `BrainArchive.load(path)`, or plain
    `np.load(path)` -> `d["w0"][t]` is the champion at generation `d["gen"][t]`.
    """

    def __init__(self, path, cfg, resume=False):
        self.path, self.cfg = path, cfg
        self.gens, self.w, self.b = [], [[] for _ in range(cfg.n_blocks)], \
            [[] for _ in range(cfg.n_blocks)]
        if resume and os.path.exists(path):
            d = np.load(path)
            keep = d["gen"].tolist()
            self.gens = keep
            for l in range(cfg.n_blocks):
                self.w[l] = list(d[f"w{l}"])
                self.b[l] = list(d[f"b{l}"])

    def trim_from(self, gen):
        """Drop entries at or after `gen` -- they are about to be re-recorded."""
        keep = sum(1 for g in self.gens if g < gen)
        self.gens = self.gens[:keep]
        for l in range(self.cfg.n_blocks):
            self.w[l], self.b[l] = self.w[l][:keep], self.b[l][:keep]

    def append(self, gen, wm, bm):
        self.gens.append(gen)
        for l in range(self.cfg.n_blocks):
            self.w[l].append(np.asarray(wm[l]).copy())
            self.b[l].append(np.asarray(bm[l]).copy())

    def save(self):
        if not self.gens:
            return
        out = {"gen": np.asarray(self.gens, dtype=np.int32)}
        for l in range(self.cfg.n_blocks):
            out[f"w{l}"] = np.stack(self.w[l])
            out[f"b{l}"] = np.stack(self.b[l])
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, self.path)     # atomic, like the checkpoint

    @staticmethod
    def load(path):
        """-> (gens, [w-blocks per step], [b-blocks per step]) for analysis code."""
        d = np.load(path)
        n_blocks = sum(1 for k in d.files if k.startswith("w"))
        gens = d["gen"]
        ws = [[d[f"w{l}"][t] for l in range(n_blocks)] for t in range(len(gens))]
        bs = [[d[f"b{l}"][t] for l in range(n_blocks)] for t in range(len(gens))]
        return gens, ws, bs


def acc_by_op(indiv, cfg, X, X_bits, args, ops):
    """Accuracy of ONE genome against EVERY goal the run could face.

    The only fair way to report an MVG champion. An MVG champion is always a
    single-phase snapshot (it is whatever the goal was in its own generation),
    so a lone scalar invites exactly the mistake of comparing it against a
    fixed-goal champion measured on a different goal."""
    wm, bm = indiv
    w = [np.asarray(m)[None, ...] for m in wm]
    b = [np.asarray(v)[None, ...] for v in bm]
    out = {}
    for op in ops:
        y = np.asarray(tasks.targets(args.task, op, X_bits))
        out[op] = float(M.fitness(w, b, X, y, cfg, args.fitness)[0])
    return out


def save_checkpoint(path, gen, rng, weights, biases, peak_fit, peak_indiv, peak_op):
    """Atomically pickle full GA state so a run can resume exactly where it stopped."""
    obj = {"gen": gen, "rng_state": rng.bit_generator.state,
           "weights": weights, "biases": biases,
           "peak_fit": peak_fit, "best_weights": peak_indiv[0],
           "best_biases": peak_indiv[1], "peak_op": peak_op}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)   # atomic: a crash mid-write can't corrupt the checkpoint


def train_seed(cfg, X, X_bits, args, seed, open_after=False):
    rng = np.random.default_rng(seed)
    mvg_ops = tuple(o.strip() for o in args.mvg_ops.split(","))
    # Extra per-generation logging windows. Logging draws no randomness, so a run
    # with --dense-log is the SAME run as one without: same seed -> same stream.
    dense_windows = [tuple(int(v) for v in w.split(":"))
                     for w in getattr(args, "dense_log", "").split(",") if w.strip()]

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
    brains_path = os.path.join(args.out_dir, f"{run_name}_brains.npz")

    # --- resume from a mid-run checkpoint, or start fresh ---
    resuming = args.resume and os.path.exists(ckpt_path)
    brains = BrainArchive(brains_path, cfg, resume=resuming)
    if resuming:
        with open(ckpt_path, "rb") as f:
            c = pickle.load(f)
        rng.bit_generator.state = c["rng_state"]
        weights, biases = c["weights"], c["biases"]
        start_gen, peak_fit, peak_op = c["gen"], c["peak_fit"], c["peak_op"]
        peak_indiv = (c["best_weights"], c["best_biases"])
        csv_f = open(csv_path, "a", newline="")
        writer = csv.writer(csv_f)
        brains.trim_from(start_gen)   # a post-checkpoint tail would be re-recorded
        print(f"[seed {seed}] RESUME {mode} from gen {start_gen}/{args.generations} "
              f"(peak {peak_fit:.3f}, {len(brains.gens)} brains kept)")
    else:
        weights, biases = M.init_population(rng, cfg, args.pop, args.init_density)
        start_gen, peak_fit, peak_indiv, peak_op = 0, -1.0, None, None
        csv_f = open(csv_path, "w", newline="")
        writer = csv.writer(csv_f)
        writer.writerow(["gen", "op", "best_fit", "mean_fit", "Q", "purity",
                         "density", "edges"])
        print(f"[seed {seed}] mode={mode} fitness={args.fitness} task={args.task} "
              f"layers={list(cfg.layers)} pop={args.pop} max_edges={cfg.max_edges} "
              f"switch={args.switch_interval if args.mvg else '-'} gens={args.generations} "
              f"ckpt_every={args.checkpoint_interval}")

    cur_op, y = None, None
    stopped_gen = args.generations - 1
    # The network we report/save/visualise is the LAST generation's champion (the
    # evolved topology, always scored against the goal ACTIVE IN ITS OWN GENERATION).
    # `peak_*` is a cross-generation maximum kept only to answer "did it ever solve
    # the task?" -- under MVG it maximises over TWO DIFFERENT GOALS, so it is not an
    # accuracy and must never be compared against a fixed-goal number.
    final_indiv, final_fit, final_op = None, -1.0, None
    t0 = time.time()
    for gen in range(start_gen, args.generations):
        op = goal_op(gen)
        if op != cur_op:
            cur_op, y = op, np.asarray(tasks.targets(args.task, op, X_bits))

        fit = M.fitness(weights, biases, X, y, cfg, args.fitness)
        gi = int(fit.argmax())
        gen_best = float(fit[gi])
        if gen_best > peak_fit:
            peak_fit, peak_op = gen_best, cur_op
            peak_indiv = M.individual(weights, biases, gi)
        # overwrite every gen -> after the loop this holds the last generation's champion
        final_indiv, final_fit, final_op = M.individual(weights, biases, gi), gen_best, cur_op

        coarse = gen % args.log_interval == 0 or gen == args.generations - 1
        if coarse or any(lo <= gen <= hi for lo, hi in dense_windows):
            wm, bm = M.individual(weights, biases, gi)
            if coarse:
                brains.append(gen, wm, bm)  # the brain itself, not just its metrics
            q, _ = newman_q(wm, cfg, weighted=args.weighted_q)
            pur = purity_of(wm, cfg)
            dens = density(wm, cfg)
            writer.writerow([gen, cur_op, f"{gen_best:.4f}", f"{float(fit.mean()):.4f}",
                             f"{q:.4f}", f"{pur:.4f}", f"{dens:.2f}", n_edges(wm)])
            csv_f.flush()   # intermediary results readable mid-run
            sps = (time.time() - t0) / (gen - start_gen + 1)
            op_str = f" | op: {cur_op}" if args.task == "retina" else ""
            print(f"  Gen {gen:5d} | Best: {gen_best:.3f} | Mean: {float(fit.mean()):.3f}"
                  f" | Q: {q:.3f} | Purity: {pur:.3f} | Density: {dens:.1f}%{op_str}"
                  f" | {sps:.3f}s/gen")

        if args.early_stop and (not args.mvg) and peak_fit >= args.target:
            print(f"  early stop: best {peak_fit:.3f} >= target {args.target:.3f} at gen {gen}")
            stopped_gen = gen
            break

        # --- reproduce: KA elite strategy + crossover (Pc) + mutation (Pm) ---
        weights, biases = ga.reproduce(rng, weights, biases, fit, cfg,
                                       n_elite=args.n_elite, pc=args.pc, pm=args.pm)

        # --- periodic checkpoint (full state) + best-so-far snapshot ---
        if args.checkpoint_interval and (gen + 1) % args.checkpoint_interval == 0 \
                and peak_indiv is not None:
            save_checkpoint(ckpt_path, gen + 1, rng, weights, biases, peak_fit, peak_indiv, peak_op)
            # save the CURRENT champion, so the npz means the same thing mid-run as
            # it does at completion (the final generation's task-matched champion)
            _save_best_npz(npz_path, final_indiv, cfg)
            brains.save()   # crash-safe: the trajectory survives an interrupted run
            print(f"  [checkpoint @ gen {gen + 1}] -> {os.path.basename(ckpt_path)} | "
                  f"peak {peak_fit:.3f} -> {os.path.basename(npz_path)} | "
                  f"{len(brains.gens)} brains -> {os.path.basename(brains_path)}")

    csv_f.close()
    brains.save()
    if final_indiv is None:
        # resumed at/after the generation cap, so the loop never ran: score the
        # restored population once under the final goal to get its champion.
        # (Never fall back to `peak_indiv` -- under MVG that is a cross-goal max.)
        final_op = goal_op(args.generations - 1)
        y = np.asarray(tasks.targets(args.task, final_op, X_bits))
        fit = M.fitness(weights, biases, X, y, cfg, args.fitness)
        gi = int(fit.argmax())
        final_indiv, final_fit = M.individual(weights, biases, gi), float(fit[gi])
    # Modularity is reported for the FINAL generation's champion (evolved topology).
    # Headline metric is KA's NORMALIZED Q_m (raw Newman Q is density-confounded and
    # only kept for the live per-gen trace); q is reported alongside for continuity.
    wm, _ = final_indiv
    q, comm = newman_q(wm, cfg, weighted=args.weighted_q)
    q_m, qm_parts = normalized_qm(wm, cfg, n_rand=args.qm_nrand, seed=seed)
    # The saved champion scored against EVERY goal, not just the one live in its own
    # generation. Under MVG the final generation is always mid-epoch on one goal, so
    # `final_fit` alone is a single-goal number and is NOT comparable to an FG run's.
    run_ops = mvg_ops if args.mvg else (args.operation,)
    accs = acc_by_op(final_indiv, cfg, X, X_bits, args, run_ops)
    acc_str = "  ".join(f"{o}={v:.3f}" for o, v in accs.items())
    print(f"[seed {seed}] final-gen fit {final_fit:.3f} (op={final_op}) | "
          f"champion by goal: {acc_str} | "
          f"Q_m {q_m:.3f} (q_real {qm_parts['q_real']:.3f}, q_rand {qm_parts['q_rand']:.3f}, "
          f"q_max {qm_parts['q_max']:.3f}) | raw Q {q:.3f} | "
          f"density {density(wm, cfg):.1f}% | edges {n_edges(wm)} | modules {len(comm)} "
          f"| peak-any-op {peak_fit:.3f} (op={peak_op})")

    _save_best_npz(npz_path, final_indiv, cfg)
    print(f"[seed {seed}] final network -> {npz_path}")

    # completion marker: lets a rerun (and run_paper.py) skip finished seeds.
    # `acc_by_op` is the headline accuracy -- the saved champion on every goal, the
    # only figure comparable across FG and MVG. `final_fit`/`final_op` = that same
    # champion on the goal live in its own generation. `peak_fit_any_op` maximises
    # across goals under MVG and is NOT an accuracy -- never compare it to an FG run.
    # `q`/`q_m` describe the FINAL-gen champion's topology.
    with open(result_path, "w") as f:
        json.dump({"run_name": run_name, "mode": mode, "seed": seed,
                   "q_m": q_m, "q_real": qm_parts["q_real"], "q_rand": qm_parts["q_rand"],
                   "q_max": qm_parts["q_max"], "q": q,
                   "acc_by_op": accs,
                   "final_fit": final_fit, "final_op": final_op,
                   "peak_fit_any_op": peak_fit, "peak_op": peak_op,
                   "density": density(wm, cfg), "edges": n_edges(wm),
                   "modules": len(comm), "gen_reached": stopped_gen}, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)   # run finished -> checkpoint no longer needed
    print(f"[seed {seed}] result -> {result_path}")

    if args.viz:
        import visualize
        png = os.path.join(args.out_dir, f"{run_name}_best.png")
        visualize.visualize_net(final_indiv, cfg, png,
                                title=f"{mode} seed{seed} - final fit {final_fit:.3f}",
                                weighted_q=args.weighted_q, open_after=open_after, q_m=q_m)
    return final_fit, q_m


def main():
    args = build_parser().parse_args()
    layers = tuple(int(x) for x in args.layers.split(","))
    cfg = M.NetConfig(layers=layers)
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
            qm = r.get("q_m", r.get("q"))
            fit = r.get("final_fit", r.get("best_fit"))
            print(f"[seed {seed}] already complete (final-gen fit {fit:.3f} | Q_m {qm:.3f}) -> skip")
            results.append((seed, fit, qm))
            continue
        open_after = args.viz and args.open_img and (i == args.n_seeds - 1)
        results.append((seed, *train_seed(cfg, X, X_bits, args, seed, open_after)))

    if args.n_seeds > 1:
        print("\n=== summary ===")
        for seed, bf, qm in results:
            print(f"  seed {seed}: best fit {bf:.3f} | Q_m {qm:.3f}")
        print(f"mean Q_m {np.mean([r[2] for r in results]):.3f}  |  "
              f"mean best fit {np.mean([r[1] for r in results]):.3f}")


if __name__ == "__main__":
    main()
