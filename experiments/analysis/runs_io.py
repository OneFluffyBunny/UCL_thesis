"""Load a finished run directory and regrow its brains — experiments 1 and 2.

One loader for both encodings, so every table and figure downstream is written
once. A run directory is whatever `train.py` wrote:

    config.json     brain + run config (everything needed to regrow exactly)
    result.json     the numbers, incl. acc_by_op and "complete": true
    log.csv         the per-log-interval trace
    champions.npz   per-generation champion DNA + accuracy on every goal
    {best,final,matched,centroid}_dna.eqx

Two things here are load-bearing and easy to get wrong:

* THE MODEL MODULES ARE LOADED BY FILE PATH, not by `import model`. Experiments 1
  and 2 each have a `model.py` and a `tasks.py`, so importing both in one process
  by name gives you whichever came first on sys.path -- silently, and with the
  wrong architecture. Both model files happen to be standalone (they import only
  jax/equinox), so importlib can load each under its own name.

* THE TEMPLATE GENOME MUST BE RE-INITIALISED WITH THE RUN'S OWN SEED. A flat
  CMA-ES vector only carries the *inexact array* leaves; everything else (`g`'s
  shape for experiment 1, the static edge_rows/edge_cols for experiment 2) lives
  in the `static` half of the partition and is reproduced by re-running
  `init(jr.split(jr.PRNGKey(seed))[1], cfg)` exactly as train.py did.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import evosax as ex

EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ACTIVATIONS = {"tanh": jnp.tanh, "relu": jax.nn.relu,
               "sigmoid": jax.nn.sigmoid, "linear": lambda x: x}


@functools.lru_cache(maxsize=4)
def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: @dataclasses.dataclass resolves annotations through
    # sys.modules[cls.__module__], so a module that is not yet registered makes
    # every dataclass in it fail with a bare AttributeError on NoneType.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Run:
    """One seed's run directory, with everything needed to score it."""

    def __init__(self, run_dir: str):
        self.dir = os.path.abspath(run_dir)
        with open(os.path.join(self.dir, "config.json")) as fh:
            self.cj = json.load(fh)
        with open(os.path.join(self.dir, "result.json")) as fh:
            self.rj = json.load(fh)
        self.seed = self.cj["seed"]
        self.run = self.cj["run"]
        brain = dict(self.cj["brain"])
        act = ACTIVATIONS[self.cj.get("activation", "tanh")]

        # experiment 1's genome has cell types; the direct encoding has none.
        self.encoding = "compressed" if "n_types" in brain else "direct"
        if self.encoding == "compressed":
            M = _load_module("exp1_model", os.path.join(EXPERIMENTS, "experiment_1", "model.py"))
            self.cfg = M.BrainConfig(activation=act, **brain)
            template = M.Genome.init(jr.split(jr.PRNGKey(self.seed))[1], self.cfg)
            self._build = lambda g: np.asarray(g.build_weights(self.cfg)[0])
        else:
            M = _load_module("direct_model", os.path.join(EXPERIMENTS, "shared_direct_model.py"))
            self.cfg = M.BrainConfig(activation=act, **brain)
            template = M.DirectGenome.init(jr.split(jr.PRNGKey(self.seed))[1], self.cfg)
            self._build = lambda g: np.asarray(g.build_weights(self.cfg))
        self._template = template
        params, self._static = eqx.partition(template, eqx.is_inexact_array)
        self._reshaper = ex.ParameterReshaper(params, verbose=False)

    # -- identity -----------------------------------------------------------

    @property
    def is_mvg(self) -> bool:
        return bool(self.run["mvg"])

    @property
    def arm(self) -> str:
        """nobudget_fg / nobudget_mvg / budget_fg / budget_mvg."""
        con = "budget" if float(self.cj["brain"].get("synaptic_budget", 0)) > 0 else "nobudget"
        return f"{con}_{'mvg' if self.is_mvg else 'fg'}"

    @property
    def reference_op(self) -> str:
        return self.rj.get("reference_op", self.run["operation"])

    @property
    def shape(self):
        return self.cfg.n_in, self.cfg.n_hidden, self.cfg.n_out

    # -- brains -------------------------------------------------------------

    def weights_from_flat(self, flat) -> np.ndarray:
        genome = eqx.combine(self._reshaper.reshape_single(jnp.asarray(flat)), self._static)
        return self._build(genome)

    def weights(self, tag: str = "matched") -> np.ndarray:
        """Weight matrix of a saved champion ('best'/'final'/'matched'/'centroid').

        Default 'matched' on purpose: it is the goal-matched champion, the only
        one an FG arm and an MVG arm can be compared on. Under a fixed goal it is
        identical to 'final'.
        """
        path = os.path.join(self.dir, f"{tag}_dna.eqx")
        genome = eqx.tree_deserialise_leaves(path, self._template)
        return self._build(genome)

    def accuracy(self, tag: str = "matched", op: str | None = None) -> float:
        """Accuracy of a saved champion on a NAMED goal (default: the reference)."""
        op = op or self.reference_op
        by_op = self.rj.get("acc_by_op", {}).get(tag)
        if by_op and op in by_op:
            return float(by_op[op])
        return float(self.rj["stats"][tag]["accuracy"])

    # -- trajectory ---------------------------------------------------------

    def archive(self):
        """The per-generation champion archive, or None if it was not written.

        -> dict with `gen` (G,), `op` (G,), `flat` (G, D), `acc` {op: (G,)},
           `ops_in_play`, `reference_op`.
        """
        path = os.path.join(self.dir, "champions.npz")
        if not os.path.exists(path):
            return None
        z = np.load(path, allow_pickle=False)
        ops = [str(o) for o in z["ops_in_play"]]
        return {"gen": z["gen"], "op": np.array([str(o) for o in z["op"]]),
                "flat": z["flat"], "ops_in_play": ops,
                "reference_op": str(z["reference_op"]),
                "acc": {o: z[f"acc_{o}"] for o in ops}}

    def log(self):
        """log.csv as a dict of columns (floats where possible)."""
        import csv
        path = os.path.join(self.dir, "log.csv")
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return {}
        out = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            try:
                out[k] = np.array([float(v) for v in vals])
            except ValueError:
                out[k] = np.array(vals)
        return out

    def __repr__(self):
        return (f"<Run {os.path.basename(self.dir)} {self.encoding} {self.arm} "
                f"seed{self.seed} gens={self.rj.get('gens_run')}>")


def find_runs(root: str, complete_only: bool = True):
    """Every run directory under `root`, sorted by arm then seed."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "result.json" not in filenames or "config.json" not in filenames:
            continue
        if complete_only:
            try:
                with open(os.path.join(dirpath, "result.json")) as fh:
                    if not json.load(fh).get("complete"):
                        continue
            except (OSError, json.JSONDecodeError):
                continue
        out.append(Run(dirpath))
    return sorted(out, key=lambda r: (r.arm, r.seed))
