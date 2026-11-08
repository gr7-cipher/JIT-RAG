"""E7: improvements to JIT-RAG that need no vector index. Settings are chosen leave-one-domain-out in e07_report.py.
fuse_rrf  : RRF of the BM25 order and the bi-encoder order over the same K candidates
fuse_cc_a : convex combination of min-max normalised BM25 and cosine scores, weight a on cosine
multi     : candidates from BM25 and RM3 (more encodings), fused by RRF of BM25, RM3 and cosine orders
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KS = [20, 50, 100, 200]
def ranks(idx):
    return {int(x): r for r, x in enumerate(idx)}
def mm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
def rrf(lists, k=60):
    sc = {}
    for l in lists:
        for r, x in enumerate(l):
            sc[int(x)] = sc.get(int(x), 0.0) + 1.0 / (k + r + 1)
    o = sorted(sc, key=lambda x: -sc[x])
    return np.array(o, dtype=np.int64), np.array([sc[x] for x in o])

for name in (sys.argv[1:] or DATASETS):
    doc_ids, docs, qids, queries, qrels = load_dataset(name)
    d = os.path.join(RES, name)
    C = np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True); S = np.load(os.path.join(d, "bm25_scores.npy"), allow_pickle=True)
    E = np.load(os.path.join(d, "emb.npy"), mmap_mode="r"); Q = np.load(os.path.join(d, "qemb.npy"))
    bm = BM25(docs)
    runs, enc = {}, {}
    def add(key, q, idx, sc):
        runs.setdefault(key, {})[q] = run_from_ranked(doc_ids, q, idx, sc)
    for i, q in enumerate(qids):
        c0 = np.asarray(C[i], dtype=np.int64); s0 = np.asarray(S[i], dtype=np.float32)
        r3, _ = bm.topk(bm.rm3_weights(queries[i]), max(KS) + 1)
        for K in KS:
            c, s = c0[:K + 1], s0[:K + 1]
            if len(c) == 0:
                for key in [f"rrf_k{K}", f"multi_k{K}"] + [f"cc{a}_k{K}" for a in ALPHAS]:
                    add(key, q, c, s)
                enc.setdefault(f"multi_k{K}", []).append(0)
                continue
            cs = np.asarray(E[c]) @ Q[i]
            o = c[np.argsort(-cs)]
            add(f"rrf_k{K}", q, *rrf([c, o]))
            nb, nc = mm(s), mm(cs)
            for a in ALPHAS:
                f = a * nc + (1 - a) * nb; p = np.argsort(-f)
                add(f"cc{a}_k{K}", q, c[p], f[p])
            r = r3[:K + 1]
            u = np.array(sorted(set(c.tolist()) | set(r.tolist())), dtype=np.int64)
            cu = np.asarray(E[u]) @ Q[i]
            add(f"multi_k{K}", q, *rrf([c, r, u[np.argsort(-cu)]]))
            enc.setdefault(f"multi_k{K}", []).append(len(u))
    per = {k: evaluate(v, qrels) for k, v in runs.items()}
    per["multi_encoded"] = {k: float(np.mean(v)) for k, v in enc.items()}
    save_json(per, name, "perquery_improve.json")
    log(name, {k: round(mean(per[k]["ndcg"]), 4) for k in ["rrf_k50", "rrf_k100", "cc0.5_k100", "multi_k100"]})
