"""Macro comparison. The fusion weight and K are chosen on the other collections only, then scored on the held-out one."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; KS = [20, 50, 100, 200]
P = {d: {**load_json(d, "perquery.json"), **load_json(d, "perquery_improve.json")} for d in DATASETS}
m = lambda d, s, f="ndcg": mean(P[d][s][f])
macro = lambda s, f="ndcg": float(np.mean([m(d, s, f) for d in DATASETS]))
for s in ["bm25", "dense_exact", "rrf_bm25_dense", "jit_k50", "jit_k100"] + [f"rrf_k{K}" for K in KS] + [f"multi_k{K}" for K in KS]:
    print(f"{s:18s} nDCG {macro(s):.4f}  R@100 {macro(s,'recall100'):.4f}")
for K in KS:
    print("cc K", K, {a: round(macro(f"cc{a}_k{K}"), 4) for a in ALPHAS})
# leave-one-domain-out choice of alpha for a fixed K
out = {}
for K in KS:
    pick, vals = {}, {}
    for d in DATASETS:
        rest = [x for x in DATASETS if x != d]
        a = max(ALPHAS, key=lambda a: np.mean([m(x, f"cc{a}_k{K}") for x in rest]))
        pick[d] = a; vals[d] = P[d][f"cc{a}_k{K}"]["ndcg"]
    out[K] = (pick, vals)
    print("LODO cc K", K, round(float(np.mean([mean(v) for v in vals.values()])), 4), pick)
print("\nPaired against RRF(BM25+dense), per collection; Holm across collections")
for lbl, get in [("jit_k100", lambda d: P[d]["jit_k100"]["ndcg"]), ("fuse_rrf_k100", lambda d: P[d]["rrf_k100"]["ndcg"]), ("lodo_cc_k100", lambda d: out[100][1][d]), ("lodo_cc_k200", lambda d: out[200][1][d]), ("multi_k100", lambda d: P[d]["multi_k100"]["ndcg"])]:
    t = {d: paired_test(get(d), P[d]["rrf_bm25_dense"]["ndcg"]) for d in DATASETS}
    h = holm([t[d]["p"] for d in DATASETS])
    print(lbl, "macro", round(float(np.mean([mean(get(d)) for d in DATASETS])), 4), {d[:6]: (round(t[d]["diff"], 3), "*" if hh < .05 and (t[d]["lo"] > 0 or t[d]["hi"] < 0) else "") for d, hh in zip(DATASETS, h)})
print("\nEncoded per query, multi:", {d[:6]: P[d]["multi_encoded"]["multi_k100"] for d in DATASETS})
