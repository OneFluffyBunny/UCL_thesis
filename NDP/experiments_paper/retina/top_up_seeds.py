"""Top up FG and MVG retina runs to N completed seeds each (default 5), resumably.

Scans saved_models/*/config.yml for runs matching each condition's signature,
then launches however many more are needed via train.py, ONE AT A TIME. Does
not touch train.py / train_backend.py / ka_task.py / qmetrics -- only calls
train.py's existing CLI.

Idempotent: rerun any time (after a shutdown, a crash, whatever) and it
rescans saved_models/ from scratch, so it only launches what's still missing.
No separate state file needed -- a run only counts once BOTH config.yml and
solution_best.npy exist, which train.py only writes on successful completion.

FG and MVG are budget-matched at 1500 generations (2026-09-13). The earlier
200-generation FG runs are left in saved_models/ but no longer counted: with
fitness flat from ~gen 60, MVG's extra 1300 generations of drift would
otherwise be confounded with the goal switching itself.

Keeps the machine from idle-sleeping while it runs (SetThreadExecutionState,
released automatically when this process exits). Closing the lid still sleeps.

Usage: python experiments_paper/retina/top_up_seeds.py [--target 5]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

NDP_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS = NDP_ROOT / "saved_models"
CONF = "experiments_paper/retina/run_experiment.yaml"

SIZE_REG_KEYS = ("size_reg_alpha", "size_reg_alpha_edges", "size_regularisation")

FG_CMD = [
    sys.executable, "train.py", "--conf", CONF,
    "--generations", "1500", "--popsize", "128",
    "--balanced-fitness", "--no-early-stopping", "--operation", "and", "--snapshot",
]
MVG_CMD = [
    sys.executable, "train.py", "--conf", CONF,
    "--generations", "1500", "--popsize", "128",
    "--balanced-fitness", "--no-early-stopping", "--mvg",
    "--mvg-ops", "and,or", "--mvg-switch-interval", "20", "--snapshot",
]


def classify_run(cfg: dict) -> str | None:
    env = str(cfg.get("environment", ""))
    if "retina" not in env.lower():
        return None
    if cfg.get("balanced_fitness") is not True:
        return None
    if cfg.get("early_stopping", True) is not False:
        return None
    if cfg.get("popsize") != 128:
        return None
    if any(k in cfg for k in SIZE_REG_KEYS):
        return None
    if cfg.get("prunning_phase", False):
        return None
    if cfg.get("operation", "and") != "and":
        return None
    gens = cfg.get("generations")
    mvg = cfg.get("mvg", False)
    if gens == 1500 and not mvg:
        return "FG"
    if (gens == 1500 and mvg is True and cfg.get("mvg_switch_interval") == 20
            and list(cfg.get("mvg_ops", [])) == ["and", "or"]):
        return "MVG"
    return None


def scan_existing() -> dict[str, list[str]]:
    found = {"FG": [], "MVG": []}
    if not SAVED_MODELS.exists():
        return found
    for d in sorted(SAVED_MODELS.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "config.yml"
        dna_path = d / "solution_best.npy"
        if not cfg_path.exists() or not dna_path.exists():
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.load(f, Loader=yaml.Loader)
        except Exception as e:
            print(f"  [skip] {d.name}: could not parse config.yml ({e})")
            continue
        kind = classify_run(cfg)
        if kind:
            found[kind].append((d.name, cfg.get("seed")))
    return found


def launch(kind: str, cmd: list[str]) -> str | None:
    print(f"\n[{kind}] launching: {' '.join(cmd)}", flush=True)
    run_id = None
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd, cwd=NDP_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("Model ID:"):
            run_id = line.split("Model ID:", 1)[1].strip()
    ret = proc.wait()
    if ret != 0:
        print(f"[{kind}] FAILED (exit {ret}), run_id={run_id} -- will not count "
              f"unless it wrote config.yml+solution_best.npy; rescan will decide", flush=True)
        return None
    print(f"[{kind}] completed: run {run_id}", flush=True)
    if run_id is not None:
        png_path = SAVED_MODELS / run_id / "graph_best.png"
        if png_path.exists():
            os.startfile(str(png_path.resolve()))
    return run_id


def keep_awake():
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5)
    args = ap.parse_args()

    while True:
        found = scan_existing()
        n_fg, n_mvg = len(found["FG"]), len(found["MVG"])
        print(f"\n=== Scan: FG={n_fg}/{args.target} {found['FG']} | "
              f"MVG={n_mvg}/{args.target} {found['MVG']} ===", flush=True)
        need_fg = max(0, args.target - n_fg)
        need_mvg = max(0, args.target - n_mvg)
        if need_fg == 0 and need_mvg == 0:
            print("\nTarget reached for both conditions. Nothing to do.", flush=True)
            break
        if need_fg > 0:
            launch("FG", FG_CMD)
        else:
            launch("MVG", MVG_CMD)


if __name__ == "__main__":
    main()
