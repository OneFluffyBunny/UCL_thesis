import json, random, sys, pathlib
import os; HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); os.chdir(HERE)
import census, ecgp
N_IN = 8
tot = dict(snaps=0, edges=0, adj_edges=0, tries=0, ok=0, ok_active=0, act=0, nodes=0)
for seed in range(5):
    lines = pathlib.Path(f"runs/base/seed{seed}/snapshots.jsonl").read_text().splitlines()
    for line in lines[::max(1, len(lines)//40)]:
        ind = census.from_json(json.loads(line)["ind"])
        act = set(ecgp.active_nodes(ind, N_IN))
        tot["snaps"] += 1; tot["act"] += len(act); tot["nodes"] += len(ind.func)
        for k in act:
            for l in ind.conn[k][:ecgp.arity_of(ind, k)]:
                if l >= N_IN and (l - N_IN) in act:
                    tot["edges"] += 1
                    tot["adj_edges"] += (l - N_IN == k - 1)
        rnd = random.Random(seed)
        for t in range(500):
            c = ind.copy()
            before = [(c.func[j], c.ntype[j]) for j in range(len(c.func))]
            n0 = c.next_id
            tot["tries"] += 1
            # find which pair compress would pick: replay its first draw
            state = rnd.getstate(); i = int(rnd.random() * (len(c.func) - 1)); rnd.setstate(state)
            if ecgp.compress(c, rnd, 5, N_IN, 0.5):
                tot["ok"] += 1
                tot["ok_active"] += (i in act and i + 1 in act)
print(tot)
print("active nodes per circuit %.1f of %.1f" % (tot["act"]/tot["snaps"], tot["nodes"]/tot["snaps"]))
print("active gate->gate wires that are genome-adjacent: %.1f%%" % (100*tot["adj_edges"]/tot["edges"]))
print("compress attempts that succeed: %.2f%%" % (100*tot["ok"]/tot["tries"]))
print("  ...and both gates active: %.2f%% of attempts (%.0f%% of successes)" % (100*tot["ok_active"]/tot["tries"], 100*tot["ok_active"]/max(1,tot["ok"])))
