import json, random, sys, pathlib
from collections import Counter
import os; HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); os.chdir(HERE)
import census, ecgp
N_IN, N_PRIM = 8, 1
# 1) the operator itself: single point mutations on a real evolved genotype
ind0 = census.from_json(json.loads(pathlib.Path("runs/base/seed1/snapshots.jsonl").read_text().splitlines()[-1])["ind"])
print("genotype: nodes by type", Counter(ind0.ntype), "(0=NAND, 1=compressed module, 2=reused module)")
rnd = random.Random(0); c = Counter()
for t in range(200000):
    ind = ind0.copy()
    ecgp.point_mutate(ind, rnd, 1, N_IN, N_PRIM)
    ecgp.validate(ind, N_IN, N_PRIM)
    for j in range(len(ind.func)):
        a, b = (ind0.ntype[j], ind0.func[j]), (ind.ntype[j], ind.func[j])
        if a != b:
            c[f"node function: {'NAND' if a[0]==0 else 'module'}(type {a[0]}) -> {'NAND' if b[0]==0 else 'module M%d' % 0 if False else ('NAND' if b[0]==0 else 'module')}(type {b[0]})"] += 1
        for s, (l0, l1) in enumerate(zip(ind0.conn[j], ind.conn[j])):
            if l0 != l1:
                kind = lambda l: "input" if l < N_IN else "gate"
                c[f"wire: {kind(l0)} -> {kind(l1)}"] += 1
                if l0 < N_IN and l1 < N_IN: c["  e.g. input %d -> input %d" % (l0, l1)] += 0
                if l0 == 2 and l1 == 5: c["wire: exactly input 2 -> input 5"] += 1
    if ind.modules != ind0.modules and any(
        vars(ind.modules[m]) != vars(ind0.modules[m]) for m in ind0.modules if m in ind.modules):
        c["module BODY changed"] += 1
for k, v in sorted(c.items()):
    if v: print(f"  {v:7d}  {k}")
# 2) does evolution actually use it? type-2 (reused) module calls in the final ACTIVE circuits
for seed in range(5):
    fin = census.from_json(json.loads(pathlib.Path(f"runs/base/seed{seed}/snapshots.jsonl").read_text().splitlines()[-1])["ind"])
    act = ecgp.active_nodes(fin, N_IN)
    t = Counter(fin.ntype[j] for j in act)
    print(f"seed {seed} final active circuit: {t[0]} NAND, {t[1]} compressed-module calls, {t[2]} NAND-replaced-by-module calls")
