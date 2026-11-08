"""E1: quality of every architecture on one dataset, plus the shared artefacts other experiments reuse.

Quality for JIT-RAG at each K is computed from corpus embeddings restricted to the BM25 top-K. That
gives the same ranking as encoding the K documents at query time; latency is measured separately
in e02 by actually encoding them. check_equivalence() confirms this on a sample.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import hnswlib

name = sys.argv[1]
d = res_dir(name)
doc_ids, docs, qids, queries, qrels = load_dataset(name)
log(name, "docs", len(docs), "queries", len(qids))
out = {"n_docs": len(docs), "n_q": len(qids), "avg_doc_words": float(np.mean([len(x.split()) for x in docs[:50000]])),
       "avg_q_words": float(np.mean([len(q.split()) for q in queries])), "avg_rel": float(np.mean([sum(r > 0 for r in v.values()) for v in qrels.values()]))}

# ---------- first stage
t = time.time(); bm = BM25(docs); out["bm25_build_s"] = time.time() - t; out["bm25_bytes"] = bm.nbytes()
KMAX = max(K_GRID)
cands, cand_scores, ts = [], [], []
for q in queries:
    t = time.perf_counter(); idx, sc = bm.search(q, KMAX + 1); ts.append(time.perf_counter() - t)
    cands.append(idx); cand_scores.append(sc)
out["bm25_ms_p50"], out["bm25_ms_p95"] = float(np.percentile(ts, 50) * 1e3), float(np.percentile(ts, 95) * 1e3)
rm3_cands = []
for q in queries:
    idx, _ = bm.topk(bm.rm3_weights(q), KMAX + 1)
    rm3_cands.append(idx)
out["zero_match_queries"] = int(sum(len(c) == 0 for c in cands))
np.save(os.path.join(d, "bm25_cands.npy"), np.array(cands, dtype=object), allow_pickle=True)
np.save(os.path.join(d, "bm25_scores.npy"), np.array(cand_scores, dtype=object), allow_pickle=True)
log("first stage done", out["bm25_ms_p50"])

# ---------- embeddings
epath = os.path.join(d, "emb.npy")
m = encoder()
if os.path.exists(epath):
    E = np.load(epath, mmap_mode="r")
else:
    t = time.time(); E = encode(m, docs, progress=False); out["encode_corpus_s"] = time.time() - t
    np.save(epath, E)
t = time.time(); Q = encode(m, queries); out["encode_q_ms"] = (time.time() - t) / len(queries) * 1e3
np.save(os.path.join(d, "qemb.npy"), Q)
log("embeddings done", out.get("encode_corpus_s"))

# ---------- dense: exact and HNSW
E = np.asarray(E)
flat = []
for i in range(len(Q)):
    s = E @ Q[i]
    k = min(1001, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top])]
    flat.append((top, s[top]))
t = time.perf_counter()
_ = [np.argpartition(-(E @ Q[i]), 100)[:100] for i in range(min(30, len(Q)))]
out["flat_ms"] = (time.perf_counter() - t) / min(30, len(Q)) * 1e3
ip = os.path.join(d, "hnsw.bin")
hn = hnswlib.Index(space="ip", dim=E.shape[1])
t = time.time()
hn.init_index(max_elements=len(E), ef_construction=200, M=16)
hn.set_num_threads(os.cpu_count())
hn.add_items(E, np.arange(len(E)))
out["hnsw_build_s"] = time.time() - t
hn.save_index(ip); out["hnsw_bytes"] = os.path.getsize(ip); os.remove(ip)
hn.set_num_threads(1)
sweep = {}
hnsw_final = None
for ef in [50, 100, 200, 400, 1000]:
    hn.set_ef(ef)
    ts, res = [], []
    kq = min(100 if ef < 1000 else 1000, len(E))
    for i in range(len(Q)):
        t = time.perf_counter(); lab, dist = hn.knn_query(Q[i], k=kq); ts.append(time.perf_counter() - t)
        res.append((lab[0], 1 - dist[0]))
    ov = np.mean([len(set(res[i][0][:10]) & set(flat[i][0][:10])) / 10 for i in range(len(Q))])
    sweep[ef] = {"ms_p50": float(np.percentile(ts, 50) * 1e3), "ms_p95": float(np.percentile(ts, 95) * 1e3), "recall10_vs_exact": float(ov)}
    if ef == 1000:
        hnsw_final = res
out["hnsw_sweep"] = sweep
del hn
log("dense done", sweep[100])

# ---------- runs
per = {}
def add(sysname, ranked):
    run = {q: run_from_ranked(doc_ids, q, *r) for q, r in zip(qids, ranked)}
    per[sysname] = evaluate(run, qrels)
add("bm25", list(zip(cands, cand_scores)))
add("dense_exact", flat)
add("dense_hnsw_ef1000", hnsw_final)
add("bm25_rm3", [(c, -np.arange(len(c), dtype=np.float32)) for c in rm3_cands])
# RRF hybrid with k=60: needs both indexes, included as the stateful quality reference
rrf = []
for i in range(len(qids)):
    sc = {}
    for lst in (cands[i][:1000], flat[i][0][:1000]):
        for r, x in enumerate(lst):
            sc[int(x)] = sc.get(int(x), 0.0) + 1.0 / (60 + r + 1)
    o = sorted(sc, key=lambda x: -sc[x])[:1000]
    rrf.append((np.array(o), np.array([sc[x] for x in o])))
add("rrf_bm25_dense", rrf)

def jit(cand_lists, K):
    ranked = []
    for i, c in enumerate(cand_lists):
        c = c[:K + 1]  # +1 leaves K candidates after the query's own document is dropped
        if len(c) == 0:
            ranked.append((c, np.empty(0))); continue
        s = E[c] @ Q[i]
        o = np.argsort(-s)
        ranked.append((c[o], s[o]))
    return ranked
for K in K_GRID:
    add(f"jit_k{K}", jit(cands, K))
    add(f"jit_rm3_k{K}", jit(rm3_cands, K))

# ---------- first-stage recall at each K (the ceiling for a re-ranker)
ceil = {}
for lbl, lists in [("bm25", cands), ("rm3", rm3_cands)]:
    ceil[lbl] = {}
    for K in K_GRID:
        v = {}
        for q, c in zip(qids, lists):
            rel = {x for x, r in qrels[q].items() if r > 0}
            top = [doc_ids[int(i)] for i in c[:K + 1] if doc_ids[int(i)] != q][:K]
            v[q] = len(rel & set(top)) / len(rel)
        ceil[lbl][K] = v
out["ceiling"] = {l: {K: mean(v) for K, v in x.items()} for l, x in ceil.items()}
save_json(ceil, name, "ceiling_perq.json")

# ---------- equivalence check: true on-the-fly encoding vs. cached embeddings
idxs = list(range(min(10, len(qids))))
agree = []
for i in idxs:
    c = cands[i][:100]
    if len(c) == 0:
        continue
    live = encode(m, [docs[int(x)] for x in c]) @ Q[i]
    agree.append(float(np.corrcoef(live, E[c] @ Q[i])[0, 1]) if len(c) > 1 else 1.0)
    agree[-1] = min(agree[-1], float(np.mean(np.argsort(-live)[:10] == np.argsort(-(E[c] @ Q[i]))[:10])))
out["equivalence_top10_match"] = float(np.mean(agree))

save_json(per, name, "perquery.json")
save_json(out, name, "meta.json")
save_json({"qids": qids}, name, "qids.json")
log(name, "DONE", {k: round(mean(per[k]["ndcg"]), 4) for k in ["bm25", "dense_exact", "jit_k100", "jit_k1000", "rrf_bm25_dense"]})
