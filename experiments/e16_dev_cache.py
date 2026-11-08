"""E16 step 1: cache per-query candidates and signals for DEV-half queries only, so later design work cannot see held-out queries."""
import os, sys, pickle
sys.argv = ["x", "noop"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e14_afh import *
OUT = res("e16"); os.makedirs(OUT, exist_ok=True)
for name in DATASETS:
    doc_ids, docs, qids, queries, qrels, E = load(name); Q = np.load(res(f"{name}/bge_qemb.npy")); bm = BM25(docs)
    idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); C = np.load(res(f"{name}/bm25_cands.npy"), allow_pickle=True)
    dv = set(dev_ids(qids)); rows = []
    for i, q in enumerate(qids):
        if q not in dv: continue
        v = Q[i]; c0 = np.asarray(C[i], dtype=np.int64); w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(bm.n, np.float32); sim = E @ v
        u, cs, l, _ = af_run(bm, idf, w0, S0, c0, v, lambda ids, vv: E[ids] @ vv)
        k = min(201, len(sim)); top = np.argpartition(-sim, k - 1)[:k]; top = top[np.argsort(-sim[top])]
        al = np.array(sorted(set(u.tolist()) | set(top.tolist()) | set(c0[:201].tolist())), dtype=np.int64)
        rd = {int(x): j for j, x in enumerate(top)}; rb = {int(x): j for j, x in enumerate(c0[:201])}; ua = set(u.tolist())
        rows.append({"q": q, "text": queries[i], "ids": [doc_ids[int(x)] for x in al], "cos": sim[al], "s0": S0[al], "l": l[al], "in_af": np.array([int(x) in ua for x in al]),
                     "rd": np.array([rd.get(int(x), 10**6) for x in al]), "rb": np.array([rb.get(int(x), 10**6) for x in al]), "nterms": len(w0), "idfs": [float(idf[t]) for t in w0]})
    pickle.dump({"rows": rows, "qrels": {r["q"]: qrels[r["q"]] for r in rows}}, open(os.path.join(OUT, f"dev_{name}.pkl"), "wb")); log(name, len(rows))
