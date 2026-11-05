"""Dev-only selection over E12 (test half is not read here)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
P = {d: load_json(d, "perquery_e12.json") for d in DATASETS}; SP = {}
for d in DATASETS:
    q = sorted(P[d]["base|mm|0.9|0.0"]["ndcg"]); r = np.random.default_rng(SEED).permutation(len(q)); SP[d] = [q[i] for i in r[:len(q) // 2]]
sm = lambda d, k, f="ndcg": float(np.mean([P[d][k][f][q] for q in SP[d]]))
keys = [k for k in P[DATASETS[0]] if k != "__enc__"]; cfg = sorted({k.split("|")[0] for k in keys}); res = {}
for c in cfg:
    ks = [k for k in keys if k.startswith(c + "|")]; v, r, pk = [], [], {}
    for d in DATASETS:
        b = max(ks, key=lambda k: np.mean([sm(o, k) for o in DATASETS if o != d])); pk[d] = b; v.append(sm(d, b)); r.append(sm(d, b, "recall100"))
    res[c] = (float(np.mean(v)), float(np.mean(r)), sorted(set(x.split("|", 1)[1] for x in pk.values())))
for c, x in sorted(res.items(), key=lambda x: -x[1][0]): print(f"{c:12s} {x[0]:.4f} {x[1]:.3f}", x[2], round(float(np.mean([P[d]['__enc__'][c] for d in DATASETS])), 1))
