"""Shared loading and scoring for the experiment-4 FG-vs-MVG figures.

Reads a study directory written by `train.py --archive-interval ... --dense-archive
...` holding exactly one FG run and one MVG run (e.g. `runs/fgmvg50`), and turns
archived champions into the three quantities every figure plots:

  acc      hits / 256 on a NAMED goal -- not on whichever goal was active, which
           under MVG alternates between two different tasks
  purity   `qmetrics.circuit_purity` on the active circuit, program output excluded
  gates    active gate count, the size confound purity has to be read against

Used by `fig_fgmvg_circuits.py`, `fig_fgmvg_progress.py`, `fig_fgmvg_windows.py`.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
EXP4 = _HERE.parents[1]
REPO = _HERE.parents[3]
sys.path.insert(0, str(EXP4))                  # cgp, gates, tasks, visualize
sys.path.insert(0, str(REPO))                  # qmetrics

import networkx as nx

import cgp
import gates as gates_mod
import tasks as tasks_mod
from qmetrics import circuit_purity

csv.field_size_limit(10 ** 7)


class Arm:
    """One run directory (FG or MVG) and everything needed to score its circuits."""

    def __init__(self, run_dir: pathlib.Path):
        self.dir = run_dir
        self.cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.mvg = bool(self.cfg["mvg"])
        self.name = "MVG" if self.mvg else "FG"
        # which search made the run: train.py (1+4) ES, or train_pop.py (KA's GA)
        self.alg = (f"GA pop {self.cfg['pop']}, elite {self.cfg['n_elite']}"
                    if self.cfg.get("search") == "ka_ga" else f"(1+{self.cfg['popsize'] - 1}) ES")
        self.E = int(self.cfg["switch_interval"])
        self.task = self.cfg["task"]
        self.gates = gates_mod.build_set(self.cfg["gates"])
        self.n_in = tasks_mod.n_inputs(self.task)
        self.split = self.n_in // 2
        self.n_patterns = tasks_mod.n_patterns(self.task)
        self.mask = tasks_mod.full_mask(self.task)
        self.in_masks = tasks_mod.input_masks(self.task)
        self.targets = {op: tasks_mod.target_mask(self.task, op) for op in ("and", "or")}
        self.run = next(run_dir.glob("*_seed0_log.csv")).name.rsplit("_seed0_", 1)[0]
        self.seeds = sorted(int(p.name.split("_seed")[1].split("_")[0])
                            for p in run_dir.glob("*_result.json"))

    def path(self, seed: int, kind: str) -> pathlib.Path:
        return self.dir / f"{self.run}_seed{seed}_{kind}"

    def archive(self, seed: int) -> list[dict]:
        with self.path(seed, "archive.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["gen"] = int(r["gen"])
            r["hits"] = int(r["hits"])
            r["pop_mean_hits"] = float(r["pop_mean_hits"])
        return rows

    def recovery(self, seed: int) -> list[dict]:
        p = self.path(seed, "recovery.csv")
        if not p.exists():
            return []
        with p.open(newline="", encoding="utf-8") as f:
            return [{k: (v if k == "goal" else int(v)) for k, v in r.items()}
                    for r in csv.DictReader(f)]

    def goal_at(self, gen: int) -> str:
        if not self.mvg:
            return self.cfg["operation"]
        ops = self.cfg["mvg_ops"]
        return ops[(gen // self.E) % len(ops)]

    # ---- scoring --------------------------------------------------------------

    @staticmethod
    def genotype(row: dict) -> cgp.Genotype:
        func = [int(v) for v in row["func"].split()]
        conn = [int(v) for v in row["conn"].split()]
        ogene = [int(v) for v in row["ogene"].split()]
        return cgp.Genotype(func, [0] * len(func), conn, [0] * len(conn),
                            ogene, [0] * len(ogene), len(conn) // len(func))

    def acc(self, g: cgp.Genotype, goal: str) -> float:
        _, h = cgp.fitness(g, self.gates, self.in_masks, self.targets[goal],
                           self.mask, self.n_in)
        return h / self.n_patterns

    def digraph(self, g: cgp.Genotype):
        ph = cgp.phenotype(g, self.n_in, self.gates, self.split)
        act = set(ph.active)
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_in))
        G.add_nodes_from(self.n_in + j for j in act)
        for j in ph.active:
            for k in range(self.gates[g.func[j]].arity):
                s = g.conn[j * g.arity + k]
                if s < self.n_in or (s - self.n_in) in act:
                    G.add_edge(s, self.n_in + j)
        return G, ph

    def purity(self, g: cgp.Genotype):
        """(purity, detail dict, phenotype). Purity is nan for a circuit with no
        gate left once the output is excluded."""
        G, ph = self.digraph(g)
        pinned = {i: (0 if i < self.split else 1) for i in range(self.n_in)}
        p, d = circuit_purity(G, pinned, exclude=list(g.ogene))
        return p, d, ph

    def measure(self, row: dict) -> dict:
        g = self.genotype(row)
        p, _, ph = self.purity(g)
        return dict(gen=row["gen"], goal=row["goal"], hits=row["hits"],
                    pop_mean=row["pop_mean_hits"] / self.n_patterns,
                    acc_active=row["hits"] / self.n_patterns,
                    acc_and=self.acc(g, "and"), purity=p, gates=ph.n_active)


def load_study(root: pathlib.Path) -> tuple[Arm, Arm]:
    """(FG arm, MVG arm) from a study root holding one run directory of each.

    Or from `root/study.json` = {"fg": "<run dir>", "mvg": "<run dir>"} (paths
    relative to root), which pairs run directories living elsewhere without copying
    their archives. Figures are then written to `root/figures`.
    """
    spec = root / "study.json"
    if spec.exists():
        s = json.loads(spec.read_text(encoding="utf-8"))
        dirs = [(root / s["fg"]).resolve(), (root / s["mvg"]).resolve()]
    else:
        dirs = [p.parent for p in sorted(root.glob("*/config.json"))]
    arms = [Arm(d) for d in dirs]
    fg = [a for a in arms if not a.mvg]
    mvg = [a for a in arms if a.mvg]
    if len(fg) != 1 or len(mvg) != 1:
        raise SystemExit(f"{root}: need exactly one FG and one MVG run directory, "
                         f"found {len(fg)} FG / {len(mvg)} MVG")
    return fg[0], mvg[0]


def last_and_epoch_row(arm: Arm, rows: list[dict]) -> dict:
    """The champion drawn and scored for a seed's END state, on the AND goal.

    FG: the final archived champion. MVG: the champion at the end of the LAST AND
    epoch -- the run always ends in an OR epoch, whose champion is an OR specialist
    and would compare a different task against FG.
    """
    if not arm.mvg:
        return rows[-1]
    ends = [r for r in rows if r["goal"] == "and" and (r["gen"] + 1) % arm.E == 0]
    return ends[-1]
