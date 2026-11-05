"""E8: the fair comparison. For one encoder, per query on every collection:
jit_kK            cosine re-rank of BM25 top K
jitcc_kK_a        convex fusion over BM25 top K (no vector index)  <- proposed
dense, rrf        exact dense and RRF of BM25 and dense top 1000 (needs a vector index)
hybcc_kK_a        convex fusion over the union of BM25 top K and dense top K (needs a vector index)
Usage: e08_compare.py encode|eval <minilm|bge> """
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
what, enc = sys.argv[1], sys.argv[2]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; KS = [50, 100, 200]
BGE = "BAAI/bge-small-en-v1.5"; PFX = "Represent this sentence for searching relevant passages: "
def mm(x):
    if len(x) == 0: return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
def rrf(lists, k=60):
    sc = {}
    for l in lists:
        for r, x in enumerate(l):
            sc[int(x)] = sc.get(int(x), 0.0) + 1.0 / (k + r + 1)
    o = sorted(sc, key=lambda x: -sc[x])[:1000]
    return np.array(o, dtype=np.int64), np.array([sc[x] for x in o])
if what == "encode":
    m = encoder(BGE, 256); out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        p = os.path.join(RES, name, "bge_emb.npy")
        t = time.time()
        if not os.path.exists(p):
            X = np.zeros((len(docs), 384), dtype=np.float32)
            order = np.argsort([len(x) for x in docs])
            for s in range(0, len(docs), 2048):
                ix = order[s:s + 2048]; X[ix] = encode(m, [docs[i] for i in ix], bs=16)
            np.save(p, X)
        np.save(os.path.join(RES, name, "bge_qemb.npy"), encode(m, [PFX + q for q in queries], bs=16))
        out[name] = time.time() - t; log(name, out[name]); save_json(out, "bge_encode.json")
if what == "eval":
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        d = os.path.join(RES, name)
        C = np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True); S = np.load(os.path.join(d, "bm25_scores.npy"), allow_pickle=True)
        E = np.load(os.path.join(d, "emb.npy" if enc == "minilm" else "bge_emb.npy")); Q = np.load(os.path.join(d, "qemb.npy" if enc == "minilm" else "bge_qemb.npy"))
        runs = {}
        def add(k, q, i, s):
            runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, i, s)
        for i, q in enumerate(qids):
            c0 = np.asarray(C[i], dtype=np.int64); s0 = np.asarray(S[i], dtype=np.float64)
            sim = E @ Q[i]; k = min(1001, len(sim)); top = np.argpartition(-sim, k - 1)[:k]; top = top[np.argsort(-sim[top])]
            add("dense", q, top, sim[top]); add("rrf", q, *rrf([c0[:1000], top[:1000]]))
            for K in KS:
                c, s = c0[:K + 1], s0[:K + 1]; cs = sim[c]; o = np.argsort(-cs)
                add(f"jit_k{K}", q, c[o], cs[o])
                nb, nc = mm(s), mm(cs)
                for a in ALPHAS:
                    f = a * nc + (1 - a) * nb; p = np.argsort(-f); add(f"jitcc_k{K}_{a}", q, c[p], f[p])
                dn = top[:K + 1]; u = np.array(sorted(set(c.tolist()) | set(dn.tolist())), dtype=np.int64)
                bs = np.zeros(len(u)); pos = {int(x): j for j, x in enumerate(u)}
                for x, v in zip(nb.tolist(), c.tolist()):
                    bs[pos[int(v)]] = x
                cu = mm(sim[u])
                for a in ALPHAS:
                    f = a * cu + (1 - a) * bs; p = np.argsort(-f); add(f"hybcc_k{K}_{a}", q, u[p], f[p])
        per = {k: evaluate(v, qrels) for k, v in runs.items()}
        save_json(per, name, f"perquery_e08_{enc}.json")
        log(name, enc, {k: round(mean(per[k]["ndcg"]), 4) for k in ["dense", "rrf", "jit_k100", "jitcc_k100_0.7", "hybcc_k100_0.7"]})
