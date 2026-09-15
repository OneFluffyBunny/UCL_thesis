"""For every used gate->gate wire i->k in saved snapshots: can i and k be made
genome-adjacent without changing the circuit?"""
import json, sys, pathlib
from collections import Counter
import os; HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); os.chdir(HERE)
import census, ecgp
N_IN = 8
c = Counter()
for seed in range(5):
    lines = pathlib.Path(f"runs/base/seed{seed}/snapshots.jsonl").read_text().splitlines()
    for line in lines[::max(1, len(lines)//40)]:
        ind = census.from_json(json.loads(line)["ind"])
        n = len(ind.func)
        act = set(ecgp.active_nodes(ind, N_IN))
        reads = [set(l - N_IN for l in ind.conn[j][:ecgp.arity_of(ind, j)] if l >= N_IN)
                 for j in range(n)]
        size = lambda j: 1 if ind.ntype[j] == 0 else \
            ecgp.module_active_primitive_count(ind.modules[ind.func[j]], ind.modules)
        for k in sorted(act):
            for i in reads[k]:
                if i not in act:
                    continue
                c["wires"] += 1
                if i == k - 1:
                    c["already adjacent"] += 1
                    continue
                c["non-adjacent"] += 1
                if size(i) + size(k) <= 5:
                    c["non-adjacent, fits cap"] += 1
                for scope, alive in (("strict", lambda j: True), ("active-only", lambda j: j in act)):
                    between = [j for j in range(i + 1, k) if alive(j)]
                    mv_i = not any(i in reads[j] for j in between)        # slide i down to k-1
                    mv_k = not any(j in reads[k] for j in between)        # slide k up to i+1
                    # full reorder possible iff no other path i ~> j ~> k
                    desc = {i}
                    for j in range(i + 1, k):
                        if alive(j) and reads[j] & desc:
                            desc.add(j)
                    path = any(j in desc for j in reads[k] if j != i)
                    c[f"{scope}: move i"] += mv_i
                    c[f"{scope}: move k"] += mv_k
                    c[f"{scope}: move i or k"] += (mv_i or mv_k)
                    c[f"{scope}: some reorder (no other path i~>k)"] += (not path)
                    c[f"{scope}: some reorder AND fits cap"] += (not path and size(i) + size(k) <= 5)
na = c["non-adjacent"]
print(f"used gate->gate wires: {c['wires']}, already adjacent {c['already adjacent']}, non-adjacent {na}")
for key, v in c.items():
    if ":" in key or "fits" in key:
        print(f"  {key:48s} {100*v/na:5.1f}% of non-adjacent")
