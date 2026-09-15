import json, sys, pathlib
from collections import Counter
import os; HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); os.chdir(HERE)
import census, ecgp, gates as G
N_IN = 8
cen = census.Census(G.build_set("nand"), N_IN, 1, 256)
ops = cen.ops if hasattr(cen, "ops") else cen._ops
NAND01 = (~(0b1010 & 0b1100)) & 0xF        # NAND of module inputs 0 and 1 (2-input table)
iface, usage = Counter(), Counter()
for seed in range(5):
    lines = pathlib.Path(f"runs/base/seed{seed}/snapshots.jsonl").read_text().splitlines()
    for line in lines[::max(1, len(lines)//40)]:
        ind = census.from_json(json.loads(line)["ind"])
        for m in ind.modules.values():
            iface[(m.n_in, m.n_out)] += 1
        act = set(ecgp.active_nodes(ind, N_IN))
        for j in act:
            if ind.ntype[j] != 2: continue
            m = ind.modules[ind.func[j]]
            read = set()
            for x in act:
                for l, c in zip(ind.conn[x][:ecgp.arity_of(ind, x)], ind.cout[x]):
                    if l == N_IN + j: read.add(c)
            for l, c in zip(ind.ogene, ind.ocout):
                if l == N_IN + j: read.add(c)
            tabs = census.module_table(m, ind.modules, ops)
            n = m.n_in
            def is_nand01(t):   # output == NAND(in0, in1), ignoring other inputs
                for p in range(1 << n):
                    if (t >> p & 1) != (0 if (p & 1 and p >> 1 & 1) else 1): return False
                return True
            usage["active module calls"] += 1
            if read and all(is_nand01(tabs[c]) for c in read):
                usage["...only outputs equal to a plain NAND of its first 2 wires are read"] += 1
print("module interface (inputs, outputs) over all module-list entries:", dict(sorted(iface.items())))
for k, v in usage.items(): print(f"  {v}  {k}")
