"""E10: final comparison with bge-small for every system.
cascade  same cross-encoder stage on JIT-CC, hybrid CC, RRF and dense (first 150 queries per collection)
lag      quality while part of the corpus waits for the vector index; JIT-CC encodes its waiting candidates
latency  end-to-end time of JIT-CC at several backlog sizes, and of RRF over HNSW, same machine and encoder
Usage: e10_final.py cascade|lag|latency"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
what = sys.argv[1]
BGE = "BAAI/bge-small-en-v1.5"; PFX = "Represent this sentence for searching relevant passages: "
K = 100; A = load_json("e08_report_bge.json")["systems"]
AJ, AH = A["jitcc_k100"]["alpha"], A["hybcc_k100"]["alpha"]
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
def load(name):
    d = os.path.join(RES, name)
    return (np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True), np.load(os.path.join(d, "bm25_scores.npy"), allow_pickle=True),
            np.load(os.path.join(d, "bge_emb.npy"), mmap_mode="r"), np.load(os.path.join(d, "bge_qemb.npy")))
def jitcc(c, s, sim, a):
    f = a * mm(sim[c]) + (1 - a) * mm(s); o = np.argsort(-f)
    return c[o], f[o]
def hybcc(c, s, sim, top, a, hidden=None):
    u = np.array(sorted(set(c.tolist()) | set(top[:K + 1].tolist())), dtype=np.int64)
    bs = np.zeros(len(u)); pos = {int(x): j for j, x in enumerate(u)}
    if len(c):
        for x, v in zip(mm(s).tolist(), c.tolist()):
            bs[pos[int(v)]] = x
    ds = sim[u].astype(np.float64)
    if hidden is None:
        ds = mm(ds)
    else:  # a waiting document has no dense score yet, so it is ranked on BM25 alone
        vis = ~hidden[u]; out = np.zeros(len(u)); out[vis] = mm(ds[vis]); ds = out
    f = a * ds + (1 - a) * bs; o = np.argsort(-f)
    return u[o], f[o]
def dense_top(sim, hidden=None):
    s = sim
    if hidden is not None:
        s = sim.copy(); s[hidden] = -9
    k = min(1001, len(s)); top = np.argpartition(-s, k - 1)[:k]; top = top[np.argsort(-s[top])]
    return top[s[top] > -9]

if what == "cascade":
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device(), max_length=256)
    BS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]; NS = [10, 20, 50]
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        C, S, E, Q = load(name); E = np.asarray(E); n = min(150, len(qids)); runs = {}; ms = []; np_ = 0
        for i in range(n):
            q = qids[i]; c0 = np.asarray(C[i], dtype=np.int64); s0 = np.asarray(S[i], dtype=np.float64)
            c, s = c0[:K + 1], s0[:K + 1]; sim = E @ Q[i]; top = dense_top(sim)
            base = {"jitcc": jitcc(c, s, sim, AJ[name]), "hybcc": hybcc(c, s, sim, top, AH[name]), "rrf": rrf([c0[:1000], top[:1000]]), "dense": (top, sim[top])}
            need = sorted({int(x) for u, _ in base.values() for x in u[:51]})
            t = time.perf_counter()
            sc = ce.predict([(queries[i], docs[x]) for x in need], batch_size=64, show_progress_bar=False) if need else []
            if need:
                ms.append((time.perf_counter() - t) / len(need) * 1e3)
            cs = dict(zip(need, np.asarray(sc, dtype=np.float64).tolist()))
            for sysn, (u, f) in base.items():
                for N in NS:
                    m = min(N + 1, len(u))
                    if m == 0:
                        for b in BS: runs.setdefault(f"{sysn}_n{N}_b{b}", {})[q] = {}
                        continue
                    x = np.array([cs[int(v)] for v in u[:m]])
                    for b in BS:
                        g = b * mm(x) + (1 - b) * mm(np.asarray(f[:m], dtype=np.float64)); p = np.argsort(-g, kind="stable")
                        runs.setdefault(f"{sysn}_n{N}_b{b}", {})[q] = run_from_ranked(doc_ids, q, np.concatenate([u[:m][p], u[m:]]), np.concatenate([g[p] + 2.0, np.asarray(f[m:], dtype=np.float64)]))
        sub = {q: qrels[q] for q in qids[:n]}
        per = {k: evaluate(v, sub) for k, v in runs.items()}; per["ms_per_pair"] = float(np.median(ms))
        save_json(per, name, "perquery_e10.json")
        log(name, per["ms_per_pair"], {k: round(mean(per[f"{k}_n20_b0.0"]["ndcg"]), 4) for k in ["jitcc", "hybcc", "rrf", "dense"]}, {k: round(mean(per[f"{k}_n20_b0.4"]["ndcg"]), 4) for k in ["jitcc", "hybcc", "rrf", "dense"]})

if what == "lag":
    FS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]; out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        C, S, E, Q = load(name); E = np.asarray(E); n = len(E); pos = {d: i for i, d in enumerate(doc_ids)}
        masks = {f: np.random.default_rng(SEED).random(n) < f for f in FS}
        rng = np.random.default_rng(SEED); runs = {}; enc = {f: [] for f in FS}; enc["recent"] = []
        def add(k, q, u, f): runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, u, f)
        for i, q in enumerate(qids):
            c0 = np.asarray(C[i], dtype=np.int64); s0 = np.asarray(S[i], dtype=np.float64)
            c, s = c0[:K + 1], s0[:K + 1]; sim = E @ Q[i]
            add("jitcc", q, *jitcc(c, s, sim, AJ[name]))
            rel = sorted(d for d, r in qrels[q].items() if r > 0 and d in pos)
            hid = np.zeros(n, bool)
            if rel:
                hid[[pos[d] for d in rng.choice(rel, size=max(1, len(rel) // 2), replace=False).tolist()]] = True
            for tag, h in [*masks.items(), ("recent", hid)]:
                top = dense_top(sim, h); enc[tag].append(int(h[c].sum()))
                add(f"dense_{tag}", q, top, sim[top]); add(f"rrf_{tag}", q, *rrf([c0[:1000], top[:1000]])); add(f"hybcc_{tag}", q, *hybcc(c, s, sim, top, AH[name], h))
        per = {k: evaluate(v, qrels)["ndcg"] for k, v in runs.items()}
        save_json(per, name, "perquery_e10_lag.json")
        out[name] = {"encoded_per_q": {str(k): float(np.mean(v)) for k, v in enc.items()}, "jitcc": mean(per["jitcc"]),
                     **{str(t): {s: mean(per[f"{s}_{t}"]) for s in ["dense", "rrf", "hybcc"]} for t in [*FS, "recent"]}}
        log(name, out[name]["jitcc"], out[name]["0.1"], out[name]["recent"], out[name]["encoded_per_q"]["0.1"]); save_json(out, "e10_lag.json")

if what == "lagimp":
    # Strongest fair rival under lag: a waiting document is scored on the signal it has, at full weight.
    FS = [0.01, 0.05, 0.1, 0.25, 0.5]; out = {}
    def hyb_imp(c, s, sim, top, a, h):
        u = np.array(sorted(set(c.tolist()) | set(top[:K + 1].tolist())), dtype=np.int64)
        bs = np.zeros(len(u)); pos = {int(x): j for j, x in enumerate(u)}
        if len(c):
            for x, v in zip(mm(s).tolist(), c.tolist()):
                bs[pos[int(v)]] = x
        vis = ~h[u]; ds = np.zeros(len(u)); ds[vis] = mm(sim[u][vis].astype(np.float64))
        f = np.where(vis, a * ds + (1 - a) * bs, bs); o = np.argsort(-f)
        return u[o], f[o]
    def rrf_imp(c0, top, h, k=60):
        sc = {}
        for r, x in enumerate(top.tolist()): sc[x] = sc.get(x, 0.0) + 1.0 / (k + r + 1)
        for r, x in enumerate(c0.tolist()): sc[x] = sc.get(x, 0.0) + (2.0 if h[x] else 1.0) / (k + r + 1)
        o = sorted(sc, key=lambda x: -sc[x])[:1000]
        return np.array(o, dtype=np.int64), np.array([sc[x] for x in o])
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        C, S, E, Q = load(name); E = np.asarray(E); n = len(E); pos = {d: i for i, d in enumerate(doc_ids)}
        masks = {f: np.random.default_rng(SEED).random(n) < f for f in FS}
        rng = np.random.default_rng(SEED); runs = {}
        def add(k, q, u, f): runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, u, f)
        for i, q in enumerate(qids):
            c0 = np.asarray(C[i], dtype=np.int64)[:1000]; s0 = np.asarray(S[i], dtype=np.float64)
            c, s = c0[:K + 1], s0[:K + 1]; sim = E @ Q[i]
            rel = sorted(d for d, r in qrels[q].items() if r > 0 and d in pos)
            hid = np.zeros(n, bool)
            if rel:
                hid[[pos[d] for d in rng.choice(rel, size=max(1, len(rel) // 2), replace=False).tolist()]] = True
            for tag, h in [*masks.items(), ("recent", hid)]:
                top = dense_top(sim, h)
                add(f"rrfimp_{tag}", q, *rrf_imp(c0, top[:1000], h)); add(f"hybimp_{tag}", q, *hyb_imp(c, s, sim, top, AH[name], h))
        per = {k: evaluate(v, qrels)["ndcg"] for k, v in runs.items()}
        save_json(per, name, "perquery_e10_lagimp.json")
        log(name, {k: round(mean(v), 4) for k, v in per.items() if k.endswith("0.1") or k.endswith("recent")})

if what == "lathyb":
    # Latency of the indexed hybrid over the same HNSW settings as RRF, on the same queries.
    import hnswlib
    m = encoder(BGE, 256); out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        _, _, E, _ = load(name); E = np.asarray(E); bm = BM25(docs); a = AH[name]
        h = hnswlib.Index(space="ip", dim=384); h.init_index(max_elements=len(E), ef_construction=200, M=16); h.set_num_threads(os.cpu_count()); h.add_items(E, np.arange(len(E))); h.set_num_threads(1); h.set_ef(100)
        qs = queries[:100]; ts = []; tb = []
        for q in qs[:5]: encode(m, [PFX + q], bs=1)
        for q in qs:
            t0 = time.perf_counter(); c, s = bm.search(q, K); tb.append(time.perf_counter() - t0); c = np.asarray(c, dtype=np.int64); v = encode(m, [PFX + q], bs=1)[0]
            lab, _ = h.knn_query(v, k=K)
            u = np.array(sorted(set(c.tolist()) | set(lab[0].tolist())), dtype=np.int64); bs = np.zeros(len(u)); pos = {int(x): j for j, x in enumerate(u)}
            if len(c):
                for x, y in zip(mm(np.asarray(s, dtype=np.float64)).tolist(), c.tolist()): bs[pos[int(y)]] = x
            f = a * mm(E[u] @ v) + (1 - a) * bs; np.argsort(-f)
            ts.append(time.perf_counter() - t0)
        out[name] = {"p50": float(np.median(ts) * 1e3), "p95": float(np.percentile(ts, 95) * 1e3), "bm25_p50": float(np.median(tb) * 1e3)}
        log(name, out[name]); save_json(out, "e10_latency_hyb.json"); del h, E

if what == "latency":
    import hnswlib
    from sentence_transformers import CrossEncoder
    m = encoder(BGE, 256); ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device(), max_length=256); out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        _, _, E, _ = load(name); bm = BM25(docs); N = min(100, len(queries)); r = {}
        E = np.asarray(E); a = AJ[name]
        h = hnswlib.Index(space="ip", dim=384); h.init_index(max_elements=len(E), ef_construction=200, M=16); h.set_num_threads(os.cpu_count()); t = time.time(); h.add_items(E, np.arange(len(E))); r["hnsw_build_s"] = time.time() - t; h.set_num_threads(1); h.set_ef(100)
        qs = queries[:N]
        for q in qs[:5]: encode(m, [PFX + q], bs=1)  # warm up
        def jit(f, rerank=0):
            mask = np.random.default_rng(SEED).random(len(E)) < f; ts = []; ne = []
            for q in qs:
                t0 = time.perf_counter(); c, s = bm.search(q, K); c = np.asarray(c, dtype=np.int64); v = encode(m, [PFX + q], bs=1)[0]
                X = np.array(E[c], copy=True) if len(c) else np.zeros((0, 384), dtype=np.float32); w = mask[c] if len(c) else np.zeros(0, bool)
                if w.any(): X[w] = encode(m, [docs[int(i)] for i in c[w]])
                if len(c):
                    g = a * mm(X @ v) + (1 - a) * mm(np.asarray(s, dtype=np.float64)); o = np.argsort(-g)
                    if rerank:
                        ce.predict([(q, docs[int(i)]) for i in c[o][:rerank]], batch_size=64, show_progress_bar=False)
                ts.append(time.perf_counter() - t0); ne.append(int(w.sum()))
            return {"p50": float(np.median(ts) * 1e3), "p95": float(np.percentile(ts, 95) * 1e3), "encoded": float(np.mean(ne))}
        for f in [1.0, 0.5, 0.1, 0.01, 0.0]:
            r[f"jitcc_f{f}"] = jit(f)
        r["jitcc_f0.1_ce20"] = jit(0.1, 20); r["jitcc_f0.0_ce20"] = jit(0.0, 20)
        ts = []
        for q in qs:
            t0 = time.perf_counter(); c, s = bm.search(q, 1000); v = encode(m, [PFX + q], bs=1); lab, _ = h.knn_query(v, k=1000 if len(E) > 1000 else len(E) - 1); rrf([np.asarray(c, dtype=np.int64), lab[0]]); ts.append(time.perf_counter() - t0)
        r["rrf_hnsw"] = {"p50": float(np.median(ts) * 1e3), "p95": float(np.percentile(ts, 95) * 1e3)}
        out[name] = r; log(name, {k: (round(v["p50"], 1) if isinstance(v, dict) else round(v, 1)) for k, v in r.items()}); save_json(out, "e10_latency.json")
        del h, E
