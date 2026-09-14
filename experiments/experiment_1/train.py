"""CMA-ES training loop for experiment 1.

Evolves the DNA (Genome) so that the static brain it grows solves a boolean task.
Fitness = (balanced) fraction of correct outputs over the full input enumeration
-> deterministic, noise-free evaluation (no winner's-curse possible).

Modularly-varying goal (--mvg): the target operation switches every
--switch-interval generations, alternating AND <-> OR (the two goals share the
left/right module structure -- the Kashtan-Alon setup).

Run `python train.py --help` for all flags.
"""

from __future__ import annotations

import sys
# line-buffer stdout/stderr so logs flush immediately under redirection/background
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import csv
import dataclasses
import json
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import evosax as ex
import numpy as np

import tasks
from model import Genome
from config import parse_args
from ga import KAGeneticAlgorithm, gene_blocks
from visualize import brain_stats, visualize_brain


def _encode(X, encoding):
    return X * 2.0 - 1.0 if encoding == "bipolar" else X


# hinge cap for the margin surrogate: once the raw output is this far onto the
# correct side of 0, a case stops paying, so all selection pressure flows to the
# still-wrong cases (why hinge losses exist). Output is tanh-bounded to [-1, 1].
_MARGIN_CAP = 0.5


def _balanced_mean(vals, yf):
    """Mean of per-class means of `vals` (balanced; chance-corrected)."""
    pos = jnp.sum(vals * yf) / (jnp.sum(yf) + 1e-8)
    neg = jnp.sum(vals * (1.0 - yf)) / (jnp.sum(1.0 - yf) + 1e-8)
    return 0.5 * (pos + neg)


def _make_eval(genome_template_static, reshaper, brain_cfg, X_enc, balanced, fitness):
    """Build a jitted, vmapped eval: flat DNA + targets -> (selection_fitness, accuracy).

    `accuracy` is always the raw 0/1 (balanced) metric -- what we log, early-stop
    and report. `selection_fitness` is what CMA-ES maximises: identical to accuracy
    when fitness='accuracy', or a smooth hinged signed-margin surrogate on the raw
    tanh output when fitness='margin' (the decision is still sign(output)).
    """
    static = genome_template_static

    def eval_genome(flat_params, y):
        genome = eqx.combine(reshaper.reshape_single(flat_params), static)
        yf = y.astype(jnp.float32)

        out = jax.vmap(lambda o: genome.forward(o, brain_cfg))(X_enc)[:, 0]  # raw tanh
        correct = ((out > 0).astype(jnp.int32) == y).astype(jnp.float32)
        acc = _balanced_mean(correct, yf) if balanced else jnp.mean(correct)

        if fitness == "margin":
            s = 2.0 * yf - 1.0                              # target sign in {-1,+1}
            m = jnp.minimum(out * s, _MARGIN_CAP)          # hinge on signed margin
            score = (m + 1.0) / (1.0 + _MARGIN_CAP)        # rescale to [0, 1]
            sel = _balanced_mean(score, yf) if balanced else jnp.mean(score)
        else:
            sel = acc
        return sel, acc

    return jax.jit(jax.vmap(eval_genome, in_axes=(0, None)))


def _git_commit() -> str:
    """Short HEAD hash, so a result can be traced back to the code that made it."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_name_for(brain_cfg, run_cfg, seed) -> str:
    """Directory//artifact name for one seed of one arm.

    Everything that changes the phenotype has to appear here, or two runs write
    the same files: the ARM (an FG and an MVG run of the same task would collide)
    and the synaptic GATE (a t=0.2 run would silently clobber a t=0 one, and the
    only record of the difference is inside the config sidecar it just overwrote).
    """
    if tasks.uses_operation(run_cfg.task):
        arm = "mvg-" + "-".join(run_cfg.mvg_ops) if run_cfg.mvg else f"fg-{run_cfg.operation}"
        name = f"{run_cfg.task}_{arm}"
    else:
        name = run_cfg.task
    if brain_cfg.w_threshold > 0:
        name += f"_w{brain_cfg.w_threshold:g}"
    if brain_cfg.synaptic_budget > 0:
        name += f"_b{brain_cfg.synaptic_budget:g}"
        if brain_cfg.shrink > 0:
            name += f"s{brain_cfg.shrink:g}"
    if run_cfg.strategy == "KA_GA":      # a GA run must never overwrite a CMA-ES one
        name += f"_ga{run_cfg.popsize}e{run_cfg.ga_elite}m{run_cfg.ga_mut_sigma:g}"
    return f"{name}_seed{seed}"


@dataclasses.dataclass
class SeedResult:
    """Everything one seed produced. Returned instead of a tuple because the
    caller needs the artifacts, the metrics AND the run metadata."""
    seed: int
    run_name: str
    run_dir: str
    best: float             # best accuracy at ANY generation
    best_genome: object
    final: float            # best accuracy in the LAST generation run
    final_genome: object
    centroid_acc: float     # accuracy of the CMA-ES distribution mean
    centroid_genome: object
    gens_run: int
    wall_s: float
    # --- goal-matched champion (added 2026-09-12; see train_seed's docstring) ---
    matched: float = float("nan")   # its accuracy ON `matched_op`
    matched_genome: object = None
    matched_op: str = ""            # the goal it was last selected under
    matched_gen: int = -1           # the generation it came from
    # tag -> {op: accuracy}: every saved champion scored against EVERY goal the
    # run could face, so no number is ever read against the wrong target.
    acc_by_op: dict = dataclasses.field(default_factory=dict)
    archive_path: str = ""          # champions.npz, or "" if archiving was off
    archive_n: int = 0


def train_seed(brain_cfg, run_cfg, seed) -> SeedResult:
    """Run one seed; return a SeedResult.

    TWO champions are returned on purpose, and which one you should measure
    depends on the arm:

    * ``best_*``  -- highest accuracy seen at ANY generation. Fine under a fixed
      goal. Under ``--mvg`` it is the peak across a CHANGING target, i.e.
      whichever member happened to top out during whichever AND/OR phase, so it
      is not comparable to a fixed-goal champion.
    * ``final_*`` -- best member of the LAST generation. This is the one to use
      for any FG-vs-MVG structural/modularity comparison: it is the endpoint of
      the same number of generations of the same pressure in both arms.
    * ``centroid_*`` -- the CMA-ES distribution MEAN, not a sampled member. The
      champions above are single draws and can be lucky; the mean is where the
      search actually settled, so it is the more stable thing to measure
      structure on when the two disagree.
    * ``matched_*`` -- THE ONE TO REPORT UNDER ``--mvg`` (added 2026-09-12). The
      last generation's champion from an epoch of the REFERENCE goal
      (``run_cfg.operation``, i.e. the goal the fixed-goal arm also ran). This is
      the fix for the reporting bug ``kashtan_alon/`` already hit: under
      ``--mvg`` the schedule is deterministic, so with an even number of epochs
      the run ALWAYS ends mid-OR and ``final_*`` is an OR-selected network being
      compared against an FG arm's AND-selected one -- different tasks, silently.
      ``matched_*`` is goal-matched by construction, so an FG and an MVG number
      are the same measurement. Under a fixed goal it is identical to ``final_*``.

    Every saved champion is additionally scored against EVERY goal in play and
    returned as ``acc_by_op``, so a number can never be read against the wrong
    target even by accident.

    With ``--archive-interval > 0`` the champion of every Nth generation is kept
    (flat DNA + its accuracy on every goal) and written to
    ``<run_dir>/champions.npz``, which is what lets modularity be scored
    generation-by-generation after the run rather than only at the endpoint.

    Also writes ``<run_dir>/log.csv`` as it goes (flushed every log interval, so
    it is readable mid-run) -- the FG-vs-MVG claim is about trajectories, and a
    trajectory that only exists in terminal scrollback cannot be plotted.
    """
    key = jr.PRNGKey(seed)

    X = tasks.all_binary_inputs(brain_cfg.n_in)
    X_enc = _encode(X, run_cfg.input_encoding)

    key, init_key = jr.split(key)
    template = Genome.init(init_key, brain_cfg)
    params, static = eqx.partition(template, eqx.is_inexact_array)
    reshaper = ex.ParameterReshaper(params)
    batched_eval = _make_eval(static, reshaper, brain_cfg, X_enc, run_cfg.balanced,
                              run_cfg.fitness)

    if run_cfg.strategy == "KA_GA":
        # Kashtan-Alon-style GA (ga.py): same ask/tell/initialize calls, so the
        # rest of this loop -- goal schedule, archive, champions -- is unchanged.
        strategy = KAGeneticAlgorithm(
            popsize=run_cfg.popsize, num_dims=reshaper.total_params,
            sigma_init=run_cfg.sigma_init,
            block_ids=gene_blocks(params, reshaper, brain_cfg),
            n_elite=run_cfg.ga_elite, pc=run_cfg.ga_pc, pm=run_cfg.ga_pm,
            mut_sigma=run_cfg.ga_mut_sigma)
    else:
        strategy = ex.Strategies[run_cfg.strategy](
            popsize=run_cfg.popsize, num_dims=reshaper.total_params, sigma_init=run_cfg.sigma_init,
        )
    es_params = strategy.default_params
    key, es_key = jr.split(key)
    state = strategy.initialize(es_key, es_params)
    shaper = ex.FitnessShaper(maximize=True)

    def goal_op(gen):
        if not run_cfg.mvg:
            return run_cfg.operation
        ops = run_cfg.mvg_ops
        return ops[(gen // run_cfg.switch_interval) % len(ops)]

    # Every goal this run could be scored against. The reference goal
    # (--operation) is always included even under --mvg: it is the goal the
    # fixed-goal arm runs, so it is the only one an FG-vs-MVG comparison can be
    # made on. For a task whose target ignores --operation this collapses to one
    # entry and everything below is a no-op.
    if tasks.uses_operation(run_cfg.task):
        ops_in_play = tuple(dict.fromkeys(
            (run_cfg.mvg_ops if run_cfg.mvg else ()) + (run_cfg.operation,)))
    else:
        ops_in_play = (run_cfg.operation,)
    y_by_op = {op: tasks.targets(run_cfg.task, op, X) for op in ops_in_play}

    def acc_on(flat, op):
        """Accuracy of one flat DNA vector on one goal (batch of 1)."""
        return float(batched_eval(jnp.asarray(flat)[None, :], y_by_op[op])[1][0])

    # One directory per run, so artifacts of different arms/gates can never mix.
    run_name = run_name_for(brain_cfg, run_cfg, seed)
    run_dir = os.path.join(run_cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[seed {seed}] task={run_cfg.task} op0={goal_op(0)} mvg={run_cfg.mvg} "
          f"balanced={run_cfg.balanced} fitness={run_cfg.fitness} n_in={brain_cfg.n_in} "
          f"n_hidden={brain_cfg.n_hidden} K={brain_cfg.n_types} dims={reshaper.total_params} "
          f"pop={run_cfg.popsize} w_threshold={brain_cfg.w_threshold}")
    print(f"[seed {seed}] -> {run_dir}")

    # Extra per-generation logging windows; see RunConfig.dense_log.
    dense_windows = [tuple(int(v) for v in w.split(":"))
                     for w in run_cfg.dense_log.split(",") if w.strip()]

    csv_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["gen", "op", "best_acc", "mean_acc", "sel_best",
                     "edges", "max_edges", "density", "sigma", "secs_per_gen"])

    best_flat, best = None, -1.0        # best-EVER (spans goal switches under --mvg)
    final_flat, final = None, -1.0      # best of the last generation actually run
    # GOAL-MATCHED champion: overwritten every generation the reference goal is
    # active, so after the loop it holds the last champion selected UNDER THAT
    # GOAL -- never an OR-selected net masquerading as an AND result.
    matched_flat, matched, matched_gen = None, -1.0, -1
    # champion archive (only populated when --archive-interval > 0)
    arc_flat, arc_gen, arc_op, arc_acc = [], [], [], {op: [] for op in ops_in_play}
    cur_op, y = None, None
    gens_run = 0
    t_start = time.time()
    interval_start = t_start
    for gen in range(run_cfg.generations):
        gens_run = gen + 1
        op = goal_op(gen)
        if op != cur_op:
            cur_op, y = op, tasks.targets(run_cfg.task, op, X)

        key, ask_key = jr.split(key)
        x, state = strategy.ask(ask_key, state, es_params)
        sel, acc = batched_eval(x, y)                     # CMA-ES selects on `sel`...
        state = strategy.tell(x, shaper.apply(x, sel), state, es_params)

        # ...but we track/save/report the best *accuracy* member (the real metric).
        best_idx = int(acc.argmax())
        gen_best = float(acc[best_idx])
        gen_best_flat = x[best_idx]
        if gen_best > best:
            best, best_flat = gen_best, jnp.asarray(gen_best_flat)
        # ...and always keep the CURRENT generation's champion, so whatever
        # generation the loop exits on we still have the endpoint genome.
        final, final_flat = gen_best, jnp.asarray(gen_best_flat)
        # ...and separately keep the last champion selected under the REFERENCE
        # goal, which is the only endpoint an FG arm and an MVG arm share.
        if cur_op == run_cfg.operation:
            matched, matched_flat, matched_gen = gen_best, jnp.asarray(gen_best_flat), gen

        # Champion archive: the flat DNA plus its accuracy on every goal in play.
        # Scoring the champion against the goals it was NOT selected on is the
        # only way to see, later, whether an MVG run holds both goals at once or
        # just trades one for the other every epoch.
        if run_cfg.archive_interval > 0 and gen % run_cfg.archive_interval == 0:
            arc_flat.append(np.asarray(gen_best_flat, dtype=np.float32))
            arc_gen.append(gen)
            arc_op.append(cur_op)
            for op in ops_in_play:
                arc_acc[op].append(gen_best if op == cur_op else acc_on(gen_best_flat, op))

        # Under --mvg, log at the END of each goal epoch instead of on
        # --log-interval: every row is then the same thing -- a snapshot of a goal
        # the population has had the full interval to adapt to -- and rows line up
        # with the switches. (On --log-interval they do not: 25 against a
        # switch-interval of 20 samples each epoch at a drifting offset.)
        # --log-interval still applies whenever the goal is fixed.
        epoch_len = max(1, run_cfg.switch_interval)
        due = ((gen + 1) % epoch_len == 0 if run_cfg.mvg
               else gen % run_cfg.log_interval == 0)
        # --dense-log windows log EVERY generation inside them. Under --mvg the
        # rule above gives one row per goal epoch, which is the right cadence for
        # a 10000-generation trend and useless for reading the shape of a single
        # switch: 10 points in a 200-generation window. The champion archive is
        # already per-generation, so this exists only for the POPULATION mean,
        # which no archive of champions can reconstruct after the fact.
        due = due or any(lo <= gen <= hi for lo, hi in dense_windows)
        # gen 0 and the last generation are always logged, so every run has a
        # baseline row before selection has done anything, and an endpoint row.
        if due or gen == 0 or gen == run_cfg.generations - 1:
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            st = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)
            density, n_edges = st["density"], st["n_edges"]
            sigma = float(getattr(state, "sigma", float("nan")))
            sigma_str = "" if sigma != sigma else f" | σ: {sigma:.4f}"
            # Inside a dense window rows are one generation apart, so dividing
            # by the nominal interval would under-report s/gen by up to 20x.
            in_dense = any(lo <= gen <= hi for lo, hi in dense_windows)
            gens_in = 1 if (gen == 0 or in_dense) else (
                epoch_len if run_cfg.mvg else run_cfg.log_interval)
            secs_per_gen = (time.time() - interval_start) / gens_in
            interval_start = time.time()
            mean_acc = float(acc.mean())
            sel_best = float(sel.max())
            op_str = f" | op: {cur_op}" if tasks.uses_operation(run_cfg.task) else ""
            # show the selection surrogate too when it isn't just accuracy
            sel_str = f" | Sel: {sel_best:.3f}" if run_cfg.fitness != "accuracy" else ""
            print(f"  Gen {gen:4d} | Best: {gen_best:.3f}{sel_str} | Pop mean: {mean_acc:.3f}"
                  f" | Edges: {n_edges}/{st['max_edges']} | Density: {density:.1f}%"
                  f"{sigma_str}{op_str} | {secs_per_gen:.2f}s/gen")
            writer.writerow([gen, cur_op, f"{gen_best:.4f}", f"{mean_acc:.4f}",
                             f"{sel_best:.4f}", n_edges, st["max_edges"],
                             f"{density:.2f}", f"{sigma:.6f}", f"{secs_per_gen:.3f}"])
            csv_f.flush()   # intermediary results readable mid-run

        # live visualisation of the current best brain during training
        if (run_cfg.viz_interval > 0 and gen > 0 and gen % run_cfg.viz_interval == 0):
            genome = eqx.combine(reshaper.reshape_single(gen_best_flat), static)
            vp = os.path.join(run_dir, f"gen{gen}.png")
            visualize_brain(genome, brain_cfg, run_cfg.prune_threshold, vp,
                            title=f"{run_cfg.task} seed{seed} gen{gen} - acc {gen_best:.3f}",
                            open_after=run_cfg.open_image)

        # Early stop on target (only meaningful with a fixed goal, and OFF under
        # --mvg). Pass --no-early-stop to disable it for a fixed goal too: it
        # otherwise makes an FG arm exit the moment it solves the task while the
        # MVG arm always runs the full budget -- unequal generations AND unequal
        # opportunity for post-solution drift, which is exactly the structural
        # difference an FG-vs-MVG modularity comparison is trying to measure.
        if run_cfg.early_stop and (not run_cfg.mvg) and best >= run_cfg.target:
            print(f"  early stop: best {best:.3f} >= target {run_cfg.target:.3f} at gen {gen}")
            break

    csv_f.close()

    # --- champion archive -> one npz per seed ------------------------------
    # Written once at the end rather than streamed: a 10k-generation archive is
    # ~18 MB in RAM, and a single atomic write cannot leave a half-file behind.
    archive_path, archive_n = "", 0
    if arc_flat:
        archive_path = os.path.join(run_dir, "champions.npz")
        payload = {"flat": np.stack(arc_flat),
                   "gen": np.asarray(arc_gen, dtype=np.int32),
                   "op": np.asarray(arc_op),
                   "ops_in_play": np.asarray(ops_in_play),
                   "reference_op": np.asarray(run_cfg.operation)}
        for op in ops_in_play:
            payload[f"acc_{op}"] = np.asarray(arc_acc[op], dtype=np.float32)
        tmp = archive_path + ".tmp.npz"
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, archive_path)       # atomic: a crash cannot corrupt it
        archive_n = len(arc_flat)
        print(f"[seed {seed}] archived {archive_n} champions -> {archive_path}")

    best_genome = eqx.combine(reshaper.reshape_single(best_flat), static)
    final_genome = eqx.combine(reshaper.reshape_single(final_flat), static)
    # Under a fixed goal every generation runs the reference goal, so the matched
    # champion IS the final one; the branch only bites under --mvg.
    if matched_flat is None:
        matched_flat, matched, matched_gen = final_flat, final, gens_run - 1
    matched_genome = eqx.combine(reshaper.reshape_single(matched_flat), static)

    # CMA-ES distribution mean: where the search settled, as opposed to the
    # single lucky draw the champions are. Scored on the goal that was active at
    # the end (the only one it can be compared against under --mvg).
    mean_flat = getattr(state, "mean", None)
    if mean_flat is None:
        centroid_genome, centroid_acc = None, float("nan")
    else:
        mean_flat = jnp.asarray(mean_flat)
        centroid_genome = eqx.combine(reshaper.reshape_single(mean_flat), static)
        centroid_acc = float(batched_eval(mean_flat[None, :], y)[1][0])

    # Score EVERY saved champion against EVERY goal in play. Without this a
    # reader has to remember which goal a given number belongs to, and that is
    # exactly the mistake this run exists to stop making.
    acc_by_op = {"best": {op: acc_on(best_flat, op) for op in ops_in_play},
                 "final": {op: acc_on(final_flat, op) for op in ops_in_play},
                 "matched": {op: acc_on(matched_flat, op) for op in ops_in_play}}
    if mean_flat is not None:
        acc_by_op["centroid"] = {op: acc_on(mean_flat, op) for op in ops_in_play}

    return SeedResult(seed=seed, run_name=run_name, run_dir=run_dir,
                      best=best, best_genome=best_genome,
                      final=final, final_genome=final_genome,
                      centroid_acc=centroid_acc, centroid_genome=centroid_genome,
                      gens_run=gens_run, wall_s=time.time() - t_start,
                      matched=matched, matched_genome=matched_genome,
                      matched_op=run_cfg.operation, matched_gen=matched_gen,
                      acc_by_op=acc_by_op,
                      archive_path=archive_path, archive_n=archive_n)


def main():
    brain_cfg, run_cfg, _ = parse_args()
    need = tasks.min_inputs(run_cfg.task)
    assert brain_cfg.n_in >= need, f"task {run_cfg.task!r} needs n_in >= {need}"
    if run_cfg.mvg and not tasks.uses_operation(run_cfg.task):
        raise SystemExit(f"--mvg is meaningless for task {run_cfg.task!r}: its target does not "
                         f"depend on --operation, so the goal would never actually change. "
                         f"Use one of {sorted(t for t in tasks.TASKS if tasks.uses_operation(t))}.")
    os.makedirs(run_cfg.out_dir, exist_ok=True)

    # Which champion is THE result depends on the arm.
    #   fixed goal -> "best": every generation ran the same goal, so the peak is
    #                 a fair reading of what the arm achieved.
    #   --mvg      -> "matched": NOT "final". The switch schedule is
    #                 deterministic, so a run with an even number of epochs ends
    #                 mid-OR and "final" would be an OR-selected network reported
    #                 as an AND result -- the reporting bug kashtan_alon/ hit and
    #                 fixed on 2026-09-10. "matched" is the last champion
    #                 selected under --operation, which is the goal the FG arm
    #                 ran too, so the two arms are the same measurement.
    headline = "matched" if run_cfg.mvg else "best"

    commit = _git_commit()
    results = []
    for i in range(run_cfg.n_seeds):
        seed = run_cfg.seed + i

        # --resume: a seed that already finished is left alone, so re-issuing the
        # same command after a crash or reboot picks up where it stopped rather
        # than throwing away hours of completed work.
        if run_cfg.resume:
            done = os.path.join(run_cfg.out_dir, run_name_for(brain_cfg, run_cfg, seed),
                                "result.json")
            if os.path.exists(done):
                try:
                    with open(done) as fh:
                        prev = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    prev = {}
                if prev.get("complete"):
                    hl = prev.get("stats", {}).get(prev.get("headline", ""), {})
                    print(f"[seed {seed}] already complete "
                          f"({prev.get('headline')} {hl.get('accuracy', float('nan')):.3f}) -> skip")
                    continue

        res = train_seed(brain_cfg, run_cfg, seed)

        # Sidecar config. The gate (--w-threshold) is part of the PHENOTYPE but
        # not of the genome, so a .eqx reloaded without it grows a different
        # brain than the one that was evaluated. Record what it takes to regrow
        # this brain exactly.
        with open(os.path.join(res.run_dir, "config.json"), "w") as fh:
            json.dump({"brain": {f.name: getattr(brain_cfg, f.name)
                                 for f in dataclasses.fields(brain_cfg)
                                 if f.name != "activation"},
                       "activation": getattr(brain_cfg.activation, "__name__", "custom"),
                       "seed": seed,
                       "git_commit": commit,
                       "run": {f.name: getattr(run_cfg, f.name)
                               for f in dataclasses.fields(run_cfg)}},
                      fh, indent=2, default=str)

        pngs, stats = {}, {}
        tagged = [("best", res.best, res.best_genome), ("final", res.final, res.final_genome),
                  ("matched", res.matched, res.matched_genome)]
        if res.centroid_genome is not None:
            tagged.append(("centroid", res.centroid_acc, res.centroid_genome))
        for tag, acc, genome in tagged:
            dna_path = os.path.join(res.run_dir, f"{tag}_dna.eqx")
            eqx.tree_serialise_leaves(dna_path, genome)
            st = brain_stats(genome, brain_cfg, run_cfg.prune_threshold)
            stats[tag] = {k: st[k] for k in
                          ("n_edges", "max_edges", "density", "n_exc", "n_inh", "type_counts")}
            stats[tag]["accuracy"] = acc
            stats[tag]["acc_by_op"] = res.acc_by_op.get(tag, {})
            print(f"[seed {seed}] {tag:8s} accuracy {acc:.3f} | edges {st['n_edges']}/{st['max_edges']}"
                  f" | density {st['density']:.1f}% | exc(+) {st['n_exc']} inh(-) {st['n_inh']}"
                  f" | hidden type counts {st['type_counts']}")

            pngs[tag] = os.path.join(res.run_dir, f"{tag}_brain.png")
            # auto-open only the headline arm, and only for a single seed
            visualize_brain(genome, brain_cfg, run_cfg.prune_threshold, pngs[tag],
                            title=f"{tag.capitalize()} DNA - {res.run_name} - accuracy {acc:.3f}",
                            open_after=(run_cfg.open_image and run_cfg.n_seeds == 1
                                        and tag == headline))

        # Results on disk, not just in scrollback. `headline` is recorded because
        # which champion is the comparable one depends on the arm.
        with open(os.path.join(res.run_dir, "result.json"), "w") as fh:
            json.dump({"run_name": res.run_name, "seed": seed, "git_commit": commit,
                       "headline": headline, "gens_run": res.gens_run,
                       "wall_s": round(res.wall_s, 1), "stats": stats,
                       # goal-matching provenance: which goal the headline number
                       # belongs to, and which generation it was selected in.
                       "reference_op": run_cfg.operation,
                       "matched_gen": res.matched_gen,
                       "acc_by_op": res.acc_by_op,
                       "archive": {"path": os.path.basename(res.archive_path),
                                   "n": res.archive_n} if res.archive_path else None,
                       "complete": True},
                      fh, indent=2, default=str)
        print(f"[seed {seed}] artifacts -> {res.run_dir}")
        results.append((res, pngs[headline]))

    # `results` holds only the seeds actually trained here, so a fully-resumed
    # batch prints nothing rather than crashing on an empty max().
    if results and run_cfg.n_seeds > 1:
        print("\n=== summary ===")
        for res, _ in results:
            print(f"  seed {res.seed}: best {res.best:.3f} | final {res.final:.3f} "
                  f"| matched({res.matched_op}) {res.matched:.3f} "
                  f"| centroid {res.centroid_acc:.3f}")
        # rank on the headline metric for this arm (see `headline` above)
        key = (lambda r: getattr(r[0], headline))
        top, best_png = max(results, key=key)
        print(f"best seed by {headline}: {top.seed} (best {top.best:.3f} | final {top.final:.3f}"
              f" | matched {top.matched:.3f})")
        print(f"mean {headline} {np.mean([getattr(r[0], headline) for r in results]):.3f}")
        if run_cfg.open_image:
            from visualize import _auto_open
            _auto_open(best_png)


if __name__ == "__main__":
    main()
