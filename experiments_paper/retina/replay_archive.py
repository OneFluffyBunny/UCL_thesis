"""Replay saved retina runs from their stored seed and archive every generation's
champion -- the NDP counterpart of kashtan_alon/analysis/dense_replay.py.

NDP only saves the final genome, and its logger.csv `pop_best_eval` is CMA-ES's
best-EVER fitness (es.best.f, optimizers.py:146), which under MVG mixes AND and
OR epochs. Neither can give a per-generation champion. A run is deterministic
given its seed (x0, the CMA-ES stream, the shared seed graph and growth are all
seeded from it, nb_growth_evals=1), so this re-runs train.py with that seed and
hooks cma's tell() to record, per generation:

  champions   (G, P) genome of the generation's best individual
  champ_fit   (G,)   its fitness on the goal active that generation
  pop_mean    (G,)   population mean fitness on that goal
  best_ever   (G,)   es.best.f, to check against logger.csv
  op          (G,)   active goal

Verification, stored in the archive and printed: best_ever and pop_mean must
equal logger.csv exactly for every generation, and the saved solution_best.npy
must equal the replay's own (es.best.x for FG, the last champion for MVG). A run
that fails is archived with verified=False, never silently.

The hook stops the replay right after the last generation's tell(), before
train.py writes config.yml, so no extra run appears in saved_models/ for
top_up_seeds.py to count; the empty directory train.py creates is removed.
Does not modify train.py / train_backend.py / optimizers.py.

Output: saved_models/<run>/replay_champions.npz. Resumable -- runs that already
have one are skipped.

Usage: conda run --no-capture-output -n ndp python -u experiments_paper/retina/replay_archive.py [run_id ...]
       (no ids: the 10 runs of the budget-matched FG/MVG set)
"""
from __future__ import annotations

import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NDP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NDP_ROOT))
SAVED_MODELS = NDP_ROOT / "saved_models"
CONF = NDP_ROOT / "experiments_paper" / "retina" / "run_experiment.yaml"
MATCHED = ["1789303257", "1789303833", "1789304372", "1789304978", "1789305630",
           "1786053806", "1788817038", "1788821061", "1788866451", "1788871442"]


class _Done(BaseException):
    """Raised from the tell() hook after the last generation; BaseException so the
    training loop's own handlers don't swallow it."""


def replay(run_id: str) -> bool:
    import cma

    with open(SAVED_MODELS / run_id / "config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    n_gens = int(cfg["generations"])
    mvg = bool(cfg.get("mvg"))
    ops = list(cfg.get("mvg_ops", ["and", "or"]))
    interval = int(cfg.get("mvg_switch_interval", 20))

    with open(CONF) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    conf["seed"] = int(cfg["seed"])
    tmp = Path(tempfile.mkdtemp(prefix=f"replay_{run_id}_"))
    tmp_conf = tmp / "conf.yaml"
    with open(tmp_conf, "w") as f:
        yaml.dump(conf, f)

    argv = ["train.py", "--conf", str(tmp_conf), "--generations", str(n_gens),
            "--popsize", str(cfg["popsize"]), "--balanced-fitness", "--no-early-stopping", "--snapshot"]
    argv += (["--mvg", "--mvg-ops", ",".join(ops), "--mvg-switch-interval", str(interval)] if mvg
             else ["--operation", cfg.get("operation", "and")])

    rec = {"champions": [], "champ_fit": [], "pop_mean": [], "best_ever": [], "best_x": None}
    orig_tell = cma.CMAEvolutionStrategy.tell

    def tell(self, X, f, *a, **k):
        out = orig_tell(self, X, f, *a, **k)
        f = np.asarray(f, dtype=float)             # minimised: -fitness when maximising
        i = int(np.argmin(f))
        rec["champions"].append(np.asarray(X[i], dtype=float).copy())
        rec["champ_fit"].append(-f[i])
        rec["pop_mean"].append(float(np.mean(f)))
        rec["best_ever"].append(float(self.best.f))
        if len(rec["champions"]) >= n_gens:
            rec["best_x"] = np.asarray(self.best.x, dtype=float).copy()
            raise _Done
        return out

    before = set(os.listdir(SAVED_MODELS))
    cwd, old_argv = os.getcwd(), sys.argv
    cma.CMAEvolutionStrategy.tell = tell
    try:
        os.chdir(NDP_ROOT)
        sys.argv = argv
        runpy.run_path(str(NDP_ROOT / "train.py"), run_name="__main__")
    except _Done:
        pass
    finally:
        cma.CMAEvolutionStrategy.tell = orig_tell
        sys.argv = old_argv
        os.chdir(cwd)
        for d in set(os.listdir(SAVED_MODELS)) - before:
            if not (SAVED_MODELS / d / "config.yml").exists():
                shutil.rmtree(SAVED_MODELS / d, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)

    if len(rec["champions"]) != n_gens:
        print(f"[{run_id}] replay stopped after {len(rec['champions'])}/{n_gens} generations", flush=True)
        return False

    log = pd.read_csv(SAVED_MODELS / run_id / "logger.csv", index_col=0)
    best_ok = bool(np.array_equal(np.array(rec["best_ever"]), log["pop_best_eval"].values))
    mean_ok = bool(np.allclose(np.array(rec["pop_mean"]), log["mean_eval"].values, rtol=0, atol=1e-12))
    saved = np.load(SAVED_MODELS / run_id / "solution_best.npy")
    final = rec["champions"][-1] if mvg else rec["best_x"]
    dna_ok = bool(np.array_equal(saved, final))
    verified = best_ok and mean_ok and dna_ok

    gens = np.arange(n_gens)
    op = np.array([ops[(g // interval) % len(ops)] for g in gens] if mvg
                  else [cfg.get("operation", "and")] * n_gens)
    np.savez(SAVED_MODELS / run_id / "replay_champions.npz",
             champions=np.stack(rec["champions"]), champ_fit=np.array(rec["champ_fit"]),
             pop_mean=-np.array(rec["pop_mean"]), best_ever=-np.array(rec["best_ever"]),
             op=op, seed=int(cfg["seed"]), mvg=mvg, verified=verified,
             best_ok=best_ok, mean_ok=mean_ok, dna_ok=dna_ok)
    print(f"[{run_id}] archived {n_gens} generations | best-ever matches log: {best_ok} | "
          f"pop mean matches log: {mean_ok} | final genome matches saved: {dna_ok} | "
          f"VERIFIED={verified}", flush=True)
    return verified


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        sys.exit(0 if replay(sys.argv[2]) else 1)
    if sys.platform == "win32":                    # stop the laptop sleeping mid-replay
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    # One subprocess per run: the replay is stopped mid-training, which leaves
    # train.py's process pool alive -- exiting the subprocess is what reaps it.
    import subprocess
    for rid in sys.argv[1:] or MATCHED:
        if (SAVED_MODELS / rid / "replay_champions.npz").exists():
            print(f"[{rid}] already archived, skipping", flush=True)
            continue
        print(f"\n[{rid}] replaying ...", flush=True)
        subprocess.run([sys.executable, "-u", __file__, "--one", rid], cwd=NDP_ROOT)


if __name__ == "__main__":
    main()
