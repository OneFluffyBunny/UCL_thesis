# minimum NAND-gate count for every Boolean function of <=4 inputs, by exhaustive search
import itertools, sys, time
N = 4; MASK = 0xFFFF; KMAX = int(sys.argv[1])
X = [0,0,0,0]
for p in range(16):
    for i in range(N):
        if p >> i & 1: X[i] |= 1 << p
best = {}          # truth table (16 bit) -> min gates
for x in X: best.setdefault(x, 0)
def rec(sigs, k):
    n = len(sigs)
    for i in range(n):
        si = sigs[i]
        for j in range(i, n):
            f = ~(si & sigs[j]) & MASK
            if f in sigs: continue           # duplicate signal never helps a minimum
            kk = k + 1
            if best.get(f, 99) > kk: best[f] = kk
            if kk < KMAX:
                sigs.append(f); rec(sigs, kk); sigs.pop()
t=time.time(); rec(list(X), 0); print("search s", round(time.time()-t,1), file=sys.stderr)
def support(f):
    s=[]
    for i in range(N):
        a=b=0
        for p in range(16):
            if (f>>p&1) != (f>>(p^(1<<i))&1): s.append(i); break
    return [i for i in range(N) if i in s]
def canon(f):
    s = support(f); n=len(s); out=None
    for perm in itertools.permutations(s):
        t=0
        for q in range(1<<n):
            p=0
            for k,v in enumerate(perm):
                if q>>k&1: p|=1<<v
            if f>>p&1: t|=1<<q
        out = t if out is None else min(out,t)
    return n, out
def tt(expr, n):
    t=0
    for q in range(1<<n):
        v=[q>>k&1 for k in range(n)]
        if expr(*v): t|=1<<q
    return t
NAMES = {}
def name(nm, n, e): NAMES[canon(tt(lambda *v: e(*v), 4) if n==4 else sum((1<<p) for p in range(16) if e(*[p>>k&1 for k in range(4)])) )] = nm
def add(nm, e): name(nm, 4, e)
add("NOT a", lambda a,b,c,d: not a)
add("NAND", lambda a,b,c,d: not(a and b)); add("AND", lambda a,b,c,d: a and b)
add("OR", lambda a,b,c,d: a or b); add("NOR", lambda a,b,c,d: not(a or b))
add("a AND NOT b", lambda a,b,c,d: a and not b); add("a OR NOT b (implication)", lambda a,b,c,d: a or not b)
add("XOR", lambda a,b,c,d: a^b); add("XNOR", lambda a,b,c,d: not(a^b))
add("AND3", lambda a,b,c,d: a and b and c); add("NAND3", lambda a,b,c,d: not(a and b and c))
add("OR3", lambda a,b,c,d: a or b or c); add("NOR3", lambda a,b,c,d: not(a or b or c))
add("MUX (c ? a : b)", lambda a,b,c,d: a if c else b)
add("MAJ3", lambda a,b,c,d: a+b+c>=2); add("XOR3 (parity)", lambda a,b,c,d: a^b^c)
add("AND4", lambda a,b,c,d: a and b and c and d); add("OR4", lambda a,b,c,d: a or b or c or d)
add("KA object: (c XOR d) ? (a AND b) : (a OR b)", lambda a,b,c,d: (a and b) if (c^d) else (a or b))
add("constant 1", lambda a,b,c,d: 1)
from collections import defaultdict
bysize = defaultdict(set)
for f,k in best.items():
    n,c = canon(f); bysize[k].add((n,c))
seen=set()
for k in sorted(bysize):
    fs = sorted(bysize[k]); cnt = defaultdict(int)
    for n,c in fs: cnt[n]+=1
    named = [NAMES[x] for x in fs if x in NAMES]
    print(f"{k} NANDs: {len(fs)} new functions (by #inputs {dict(sorted(cnt.items()))}); named: {named}")
for nm_key, nm in NAMES.items():
    if not any(nm_key in bysize[k] for k in bysize): print("needs >", KMAX, ":", nm)
