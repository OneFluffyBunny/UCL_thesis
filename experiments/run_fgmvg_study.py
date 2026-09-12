"""The FG-vs-MVG x constraint study, on both encodings (2026-09-12).

Reproduces, on experiments 1 and 2, the 4-group design `kashtan_alon/` ran:

    unconstrained  x  {fixed goal, modularly-varying goal}   5 seeds each
    constrained    x  {fixed goal, modularly-varying goal}   5 seeds each

where "constrained" is the synaptic budget -- this framework's analogue of
Kashtan-Alon's per-neuron fan-in cap. Both are a cap on how much incoming
connection a neuron may spend, and in both the ablation (removing it) is what
drives density to the complete graph and makes the modularity question
unanswerable rather than answering it low.

    python run_fgmvg_study.py --experiment 1 --lanes 10
    python run_fgmvg_study.py --experiment 2 --lanes 6
    python run_fgmvg_study.py --status            # what is done, what is not

EVERY RUN IS RESUMABLE. Each seed is a separate `train.py` process writing its
own directory, and a seed whose `result.json` says `"complete": true` is skipped
(`--resume` is passed to train.py). Re-issuing the same command after a crash,
a kill, or a reboot picks up exactly where it stopped. Nothing is held in
memory that is not also on disk.

Why a script and not a shell loop: `conda run` CANNOT be launched concurrently
on this machine (every instance races on one activation temp file and all but
one dies, silently, in under a second -- measured 2026-09-12). So the conda
environment's interpreter is invoked directly, which also means PYTHONIOENCODING
has to be set by hand or a non-ASCII character in a log line kills the run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# The conda env's interpreter, called directly -- see the module docstring.
PYTHON = os.environ.get(
    "LNDP_PYTHON", r"C:\Users\raduc\miniconda3\envs\lndp\python.exe")

# --------------------------------------------------------------------------
# The settled configuration. Preflight evidence for every choice is in
# experiments/OVERNIGHT_2026-09-12.md section 5; change it there too.
# --------------------------------------------------------------------------

SEEDS = [0, 1, 2, 3, 4]

# Task: Kashtan-Alon's ORIGINAL object rule, scored RAW (--no-balanced).
# With KA's equal (L,R) cells raw fraction-correct IS the mean over the four
# cells, so every predictor that reads only one half is provably capped at
# 0.750. Balanced accuracy would instead hand that same one-module cheat 0.833.
TASK = ["--task", "retina_ka2005", "--operation", "and", "--no-balanced"]

# AND <-> OR every 20 generations: the classic KA pairing (3/4 truth-table
# agreement -- a mild perturbation, unlike and/xor which is adversarial), and
# E=20 was measured right for this framework (median recovery 10 gens).
MVG = ["--mvg", "--mvg-ops", "and,or", "--switch-interval", "20"]

COMMON = [
    "--fitness", "margin",       # accuracy is piecewise-constant -> no gradient for CMA-ES
    "--no-early-stop",           # REQUIRED for FG-vs-MVG: otherwise FG exits on solve
    "--no-open",
    "--archive-interval", "1",   # every generation: a goal epoch is only 20 gens long
    "--resume",
]

EXPERIMENTS = {
    "1": dict(
        dir="experiment_1",
        # K=8 over K=6 on measured accuracy (0.920 vs 0.896 endpoint, 3 seeds,
        # 2000 gens) AND on argument: at K=8 a non-modular solution is
        # expressible, so modularity cannot be dismissed as forced by the
        # encoding. K costs ~12 of ~443 genome params -- `g` dominates.
        arch=["--n-hidden", "24", "-K", "8"],
        generations="10000",
        budget=["--synaptic-budget", "6", "--shrink", "0.9"],
    ),
    "2": dict(
        dir="experiment_2",
        arch=["--n-hidden", "24"],
        # Fewer generations than experiment 1 on purpose: CMA-ES is superlinear
        # in dimension and exp 2 searches 793 free weights against exp 1's 443,
        # so a generation costs ~5x. exp 2 also converges far sooner (historically
        # 290-456 generations to solve retina/xor), so 5000 is ~11x its own solve
        # time. Both ARMS WITHIN exp 2 get the identical budget, which is what the
        # FG-vs-MVG claim needs; cross-encoding generation counts are not matched
        # and any exp1-vs-exp2 statement must say so.
        generations="5000",
        budget=["--synaptic-budget", "4", "--shrink", "0.9"],
    ),
}

ARMS = ["nobudget_fg", "nobudget_mvg", "budget_fg", "budget_mvg"]


def arm_args(exp: dict, arm: str) -> list:
    a = list(exp["arch"]) + TASK + COMMON + ["--generations", exp["generations"]]
    if arm.endswith("_mvg"):
        a += MVG
    if arm.startswith("budget_"):
        a += exp["budget"]
    return a


def out_dir(exp_key: str, arm: str) -> str:
    return os.path.join(HERE, EXPERIMENTS[exp_key]["dir"], "runs", "fgmvg", arm)


def run_name(exp_key: str, arm: str, seed: int) -> str:
    """Mirror train.py's run_name_for without importing it (the two train.py
    modules have colliding names, so importing both in one process is unsafe)."""
    exp = EXPERIMENTS[exp_key]
    name = "retina_ka2005_" + ("mvg-and-or" if arm.endswith("_mvg") else "fg-and")
    if arm.startswith("budget_"):
        b = exp["budget"]
        name += f"_b{float(b[1]):g}s{float(b[3]):g}"
    return f"{name}_seed{seed}"


def result_path(exp_key: str, arm: str, seed: int) -> str:
    return os.path.join(out_dir(exp_key, arm), run_name(exp_key, arm, seed), "result.json")


def is_done(exp_key: str, arm: str, seed: int) -> bool:
    p = result_path(exp_key, arm, seed)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as fh:
            return bool(json.load(fh).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def jobs_for(exp_key: str) -> list:
    out = []
    for arm in ARMS:
        for seed in SEEDS:
            out.append((exp_key, arm, seed))
    return out


def launch(exp_key: str, arm: str, seed: int):
    exp = EXPERIMENTS[exp_key]
    cwd = os.path.join(HERE, exp["dir"])
    od = out_dir(exp_key, arm)
    os.makedirs(od, exist_ok=True)
    cmd = ([PYTHON, "train.py"] + arm_args(exp, arm)
           + ["--seed", str(seed), "--n-seeds", "1", "--out-dir", od])
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    log = open(os.path.join(od, f"seed{seed}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, cmd


def status(exp_keys):
    total = done = 0
    for k in exp_keys:
        print(f"\n=== experiment {k} ({EXPERIMENTS[k]['generations']} gens) ===")
        for arm in ARMS:
            marks = []
            for seed in SEEDS:
                d = is_done(k, arm, seed)
                total += 1
                done += d
                marks.append("#" if d else ".")
            print(f"  {arm:14s} [{''.join(marks)}]  {sum(m == '#' for m in marks)}/{len(SEEDS)}")
    print(f"\n{done}/{total} runs complete")
    return done, total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["1", "2", "both"], default="both")
    p.add_argument("--lanes", type=int, default=8,
                   help="max concurrent training processes (each uses ~2 threads)")
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma-separated subset of arms to run")
    p.add_argument("--status", action="store_true", help="print progress and exit")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = p.parse_args()

    keys = ["1", "2"] if args.experiment == "both" else [args.experiment]
    if args.status:
        status(keys)
        return

    want = [a.strip() for a in args.arms.split(",")]
    pending = [j for k in keys for j in jobs_for(k)
               if j[1] in want and not is_done(*j)]
    skipped = sum(1 for k in keys for j in jobs_for(k) if j[1] in want and is_done(*j))
    print(f"{len(pending)} runs to do, {skipped} already complete, {args.lanes} lanes")

    if args.dry_run:
        for j in pending:
            exp = EXPERIMENTS[j[0]]
            cmd = ([os.path.basename(PYTHON), "train.py"] + arm_args(exp, j[1])
                   + ["--seed", str(j[2]), "--n-seeds", "1", "--out-dir", out_dir(j[0], j[1])])
            print(f"  (exp {j[0]}) " + " ".join(cmd))
        return

    running, t0 = [], time.time()
    while pending or running:
        while pending and len(running) < args.lanes:
            job = pending.pop(0)
            proc, log, cmd = launch(*job)
            running.append((job, proc, log, time.time()))
            print(f"[{time.time()-t0:7.0f}s] START exp{job[0]} {job[1]} seed{job[2]}")
        time.sleep(5)
        for entry in list(running):
            job, proc, log, started = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            log.close()
            ok = is_done(*job)
            print(f"[{time.time()-t0:7.0f}s] {'DONE ' if ok else 'FAIL '}"
                  f"exp{job[0]} {job[1]} seed{job[2]} "
                  f"rc={proc.returncode} after {time.time()-started:.0f}s"
                  + ("" if ok else f"  -> see {out_dir(job[0], job[1])}/seed{job[2]}.log"))

    print(f"\nall launched work finished in {time.time()-t0:.0f}s")
    status(keys)


if __name__ == "__main__":
    main()
