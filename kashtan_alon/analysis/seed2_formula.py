"""Write out MVG seed 2's drawn brain as explicit threshold formulas, and verify.

The brain is the one `analysis/paper_grid.py` draws: the LAST champion archived
during an AND epoch (generation 24,970) of retina_mvg_raw_seed2, the panel that
reports accuracy 1.00 on AND while looking only moderately modular (purity 0.86).

Two things are printed:
  1. every live neuron as `name = [ w.x + ... + bias > 0 ]`, dead neurons dropped;
  2. an INDEPENDENT check -- the formulas are re-evaluated over all 256 patterns by
     a plain-Python interpreter written here, NOT by model.forward, and compared
     against tasks.targets. If the 1.00 were an artefact of the training-time
     forward pass, these two would disagree.

It also reports which hidden neurons compute KA's left/right object features
exactly, which is what "modular" should mean for this task.

Usage: conda run -n lndp python kashtan_alon/scratch_seed2_formula.py
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # kashtan_alon/
sys.path.insert(0, str(_HERE.parents[2]))      # repo root: qmetrics

import tasks
import train as T
from model import NetConfig

D = str(_HERE.parents[1] / "runs_purity")
RUN = "retina_mvg_raw_seed2"
GOAL = "and"


def last_on_goal(name, goal):
    with open(os.path.join(D, f"{name}_log.csv"), newline="") as f:
        op_at = {int(r["gen"]): r["op"] for r in csv.DictReader(f)}
    gens, ws, bs = T.BrainArchive.load(os.path.join(D, f"{name}_brains.npz"))
    for i in range(len(gens) - 1, -1, -1):
        if op_at.get(int(gens[i])) == goal:
            return ws[i], bs[i], int(gens[i])
    raise SystemExit("no such champion")


def node_names(cfg):
    """Layer 0 = x0..x7 (retina), then h1_*, h2_*, h3_*, out."""
    names = [[f"x{i}" for i in range(cfg.layers[0])]]
    for l in range(1, len(cfg.layers)):
        if l == len(cfg.layers) - 1:
            names.append(["out"])
        else:
            names.append([f"h{l}_{j}" for j in range(cfg.layers[l])])
    return names


def live_mask(wm, cfg):
    """Neurons with a path to the output AND at least one live input.

    A neuron with no outgoing path cannot affect the answer; a neuron with no
    incoming edge is a constant (its bias alone)."""
    n_l = len(cfg.layers)
    reaches = [np.zeros(n, bool) for n in cfg.layers]
    reaches[-1][:] = True
    for l in range(n_l - 2, -1, -1):
        W = np.asarray(wm[l])
        for i in range(cfg.layers[l]):
            reaches[l][i] = bool(np.any((W[i] != 0) & reaches[l + 1]))
    return reaches


def formulas(wm, bm, cfg):
    """-> (lines, spec) where spec[l][j] = (list of (src_index, weight), bias)."""
    names = node_names(cfg)
    reaches = live_mask(wm, cfg)
    lines, spec = [], []
    for l in range(len(wm)):
        W, b = np.asarray(wm[l]), np.asarray(bm[l])
        layer = []
        for j in range(cfg.layers[l + 1]):
            terms = [(i, int(W[i, j])) for i in range(cfg.layers[l])
                     if W[i, j] != 0]
            layer.append((terms, int(b[j])))
            if not reaches[l + 1][j]:
                continue
            if not terms:
                lines.append(f"  {names[l + 1][j]:6s} = [ {int(b[j])} > 0 ]"
                             f"   (constant {int(int(b[j]) > 0)}, no live input)")
                continue
            body = " ".join(
                (("+ " if w > 0 else "- ") + (names[l][i] if abs(w) == 1
                                              else f"{abs(w)}*{names[l][i]}"))
                for i, w in terms).lstrip("+ ").strip()
            if b[j]:
                body += f" {'+' if b[j] > 0 else '-'} {abs(int(b[j]))}"
            lines.append(f"  {names[l + 1][j]:6s} = [ {body} > 0 ]")
        spec.append(layer)
    return lines, spec


def evaluate(spec, x):
    """Independent interpreter: plain Python ints, no numpy, no model.forward."""
    a = list(x)
    for layer in spec:
        a = [1 if (sum(w * a[i] for i, w in terms) + bias) > 0 else 0
             for terms, bias in layer]
    return a[0]


def main():
    cfg = NetConfig()
    wm, bm, gen = last_on_goal(RUN, GOAL)
    X = np.asarray(tasks.all_binary_inputs(cfg.layers[0]))
    y_and = np.asarray(tasks.targets("retina", "and", X))
    y_or = np.asarray(tasks.targets("retina", "or", X))
    left = np.asarray(tasks.targets("left", "and", X))
    right = np.asarray(tasks._right_feature(X)).astype(int)

    print(f"{RUN}, last {GOAL.upper()}-epoch champion, generation {gen}")
    print(f"layers {list(cfg.layers)}, weights in {{-1,+1}}, "
          f"neuron fires iff (weighted sum + bias) > 0\n")

    lines, spec = formulas(wm, bm, cfg)
    print("THE NETWORK (dead neurons omitted):")
    print("\n".join(lines))

    pred = np.array([evaluate(spec, row) for row in X])
    print(f"\nINDEPENDENT CHECK over all {len(X)} patterns, "
          f"re-evaluating the formulas above in plain Python:")
    for nm, y in (("AND", y_and), ("OR", y_or)):
        n = int((pred == y).sum())
        print(f"  vs retina {nm:3s}: {n}/{len(X)} correct  = {n / len(X):.4f}"
              + ("   <-- perfect" if n == len(X) else ""))

    # --- the fully substituted expression, one nested threshold per neuron ---
    names = node_names(cfg)
    expand = {f"x{i}": f"x{i}" for i in range(cfg.layers[0])}
    for l, layer in enumerate(spec):
        for j, (terms, bias) in enumerate(layer):
            if not terms:
                expand[names[l + 1][j]] = str(int(bias > 0))
                continue
            body = " ".join(("+ " if w > 0 else "- ") + expand[names[l][i]]
                            for i, w in terms).lstrip("+ ").strip()
            if bias:
                body += f" {'+' if bias > 0 else '-'} {abs(int(bias))}"
            expand[names[l + 1][j]] = f"[{body}>0]"
    print("\nFULLY SUBSTITUTED (every [.] is a hard threshold on the sum inside):")
    print(f"  out = {expand['out']}")

    print("\nWhat each live hidden neuron computes, vs KA's own object features:")
    names = node_names(cfg)
    a = X.T.tolist()
    acts = [list(map(list, X.T))]
    cur = [list(col) for col in X.T]
    for l, layer in enumerate(spec):
        nxt = []
        for terms, bias in layer:
            nxt.append([1 if (sum(w * cur[i][p] for i, w in terms) + bias) > 0 else 0
                        for p in range(len(X))])
        acts.append(nxt)
        cur = nxt
    for l in range(1, len(acts)):
        for j, v in enumerate(acts[l]):
            v = np.array(v)
            tags = []
            for nm, ref in (("LEFT object", left), ("RIGHT object", right),
                            ("AND", y_and), ("OR", y_or)):
                if np.array_equal(v, ref):
                    tags.append(f"== {nm}")
                elif np.array_equal(v, 1 - ref):
                    tags.append(f"== NOT {nm}")
            if v.std() == 0:
                tags.append("constant")
            deps = sorted({i for i, _ in spec[l - 1][j][0]})
            print(f"  {names[l][j]:6s} fires on {int(v.sum()):3d}/256 patterns"
                  f"  inputs from layer {l - 1}: {deps}"
                  + ("   " + ", ".join(tags) if tags else ""))

    # --- which retina pixels each neuron transitively reads ---
    print("\nWhich retina pixels each live neuron transitively depends on"
          "  (L = 0-3, R = 4-7):")
    reach = {("x", i): {i} for i in range(cfg.layers[0])}
    src = [[{i} for i in range(cfg.layers[0])]]
    for l, layer in enumerate(spec):
        nxt = []
        for terms, _ in layer:
            s = set()
            for i, _w in terms:
                s |= src[l][i]
            nxt.append(s)
        src.append(nxt)
    for l in range(1, len(src)):
        for j, s in enumerate(src[l]):
            if not s:
                continue
            side = ("LEFT only" if s <= {0, 1, 2, 3} else
                    "RIGHT only" if s <= {4, 5, 6, 7} else "MIXED")
            print(f"  {names[l][j]:6s} <- pixels {sorted(s)}   {side}")

    # --- is there a real left module / right module behind the bottleneck? ---
    print("\nIS THERE A MODULE? -- does each side's layer-2 pair alone determine "
          "KA's object feature for that side?")
    l2 = acts[2]
    for tag, idx, ref, nm in (("LEFT", [j for j, s in enumerate(src[2])
                                        if s and s <= {0, 1, 2, 3}], left, "left object"),
                              ("RIGHT", [j for j, s in enumerate(src[2])
                                         if s and s <= {4, 5, 6, 7}], right, "right object")):
        codes = {}
        ok = True
        for p in range(len(X)):
            key = tuple(l2[j][p] for j in idx)
            if key in codes and codes[key] != int(ref[p]):
                ok = False
            codes[key] = int(ref[p])
        print(f"  {tag:5s} branch = layer-2 neurons {[names[2][j] for j in idx]}"
              f" ({len(idx)} bits): {'DETERMINES' if ok else 'does NOT determine'}"
              f" the {nm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
