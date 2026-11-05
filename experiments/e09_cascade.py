"""E9: cross-encoder over the top n of the fused list, on the first 150 queries. Encoder scores fused with weight b, chosen LODO in the report."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device(), max_length=256)
PICK = {"scifact": .8, "nfcorpus": .8, "arguana": .7, "fiqa": .7, "scidocs": .7, "trec-covid": .8, "quora": .7}  # alpha chosen on the other collections in e07_report
def mm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
BS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for name in DATASETS:
    doc_ids, docs, qids, queries, qrels = load_dataset(name)
    d = os.path.join(RES, name)
    C = np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True); S = np.load(os.path.join(d, "bm25_scores.npy"), allow_pickle=True)
    E = np.load(os.path.join(d, "emb.npy"), mmap_mode="r"); Q = np.load(os.path.join(d, "qemb.npy")); a = PICK[name]
    n = min(150, len(qids)); runs = {}; ts = []
    for i in range(n):
        q = qids[i]; c = np.asarray(C[i][:101], dtype=np.int64); s = np.asarray(S[i][:101], dtype=np.float64)
        if len(c) == 0:
            continue
        f = a * mm(np.asarray(E[c]) @ Q[i]) + (1 - a) * mm(s); o = np.argsort(-f); c, f = c[o], f[o]
        top = c[:51]; t = time.perf_counter(); sc = ce.predict([(queries[i], docs[int(x)]) for x in top], batch_size=64, show_progress_bar=False); ts.append((time.perf_counter() - t) / len(top) * 1e3)
        for N in [10, 20, 50]:
            m = min(N + 1, len(top))
            for b in BS:
                g = b * mm(sc[:m]) + (1 - b) * mm(f[:m]); p = np.argsort(-g)
                rest = c[m:]  # documents beyond the top n keep their fused order below
                idx = np.concatenate([c[:m][p], rest]); val = np.concatenate([g[p] + 2.0, f[m:]])
                runs.setdefault(f"n{N}_b{b}", {})[q] = run_from_ranked(doc_ids, q, idx, val)
    sub = {q: qrels[q] for q in qids[:n]}
    per = {k: evaluate(v, sub) for k, v in runs.items()}; per["ms_per_pair"] = float(np.median(ts))
    save_json(per, name, "perquery_e09.json")
    log(name, {k: round(mean(per[k]["ndcg"]), 4) for k in ["n20_b0.0", "n20_b0.5", "n50_b0.5", "n50_b1.0"]})
