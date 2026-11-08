"""E9b: the same cross-encoder stage applied to the indexed hybrid (BM25 + dense convex fusion), so the rival gets the same upgrade."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device(), max_length=256)
PICK = load_json("e08_report_minilm.json")["systems"]["hybcc_k100"]["alpha"]
def mm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
BS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for name in DATASETS:
    doc_ids, docs, qids, queries, qrels = load_dataset(name)
    d = os.path.join(RES, name)
    C = np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True); S = np.load(os.path.join(d, "bm25_scores.npy"), allow_pickle=True)
    E = np.load(os.path.join(d, "emb.npy")); Q = np.load(os.path.join(d, "qemb.npy")); a = PICK[name]
    n = min(150, len(qids)); runs = {}
    for i in range(n):
        q = qids[i]; c = np.asarray(C[i][:101], dtype=np.int64); s = np.asarray(S[i][:101], dtype=np.float64)
        sim = E @ Q[i]; dn = np.argpartition(-sim, 100)[:101]
        u = np.array(sorted(set(c.tolist()) | set(dn.tolist())), dtype=np.int64); bs = np.zeros(len(u)); pos = {int(x): j for j, x in enumerate(u)}
        if len(c):
            for x, v in zip(mm(s).tolist(), c.tolist()):
                bs[pos[int(v)]] = x
        f = a * mm(sim[u]) + (1 - a) * bs; o = np.argsort(-f); u, f = u[o], f[o]
        top = u[:51]; sc = ce.predict([(queries[i], docs[int(x)]) for x in top], batch_size=64, show_progress_bar=False)
        for N in [10, 20, 50]:
            m = min(N + 1, len(top))
            for b in BS:
                g = b * mm(sc[:m]) + (1 - b) * mm(f[:m]); p = np.argsort(-g)
                runs.setdefault(f"n{N}_b{b}", {})[q] = run_from_ranked(doc_ids, q, np.concatenate([u[:m][p], u[m:]]), np.concatenate([g[p] + 2.0, f[m:]]))
    sub = {q: qrels[q] for q in qids[:n]}
    per = {k: evaluate(v, sub) for k, v in runs.items()}
    save_json(per, name, "perquery_e09b.json")
    log(name, {k: round(mean(per[k]["ndcg"]), 4) for k in ["n20_b0.0", "n20_b0.4", "n50_b0.3"]})
