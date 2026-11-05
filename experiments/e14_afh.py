"""E14: JIT-AF with an optional ANN assist. Candidates = adaptive BM25 loop (waiting ones encoded at query time) + dense top 50 over ENCODED documents.
grid    f=0 settings, used to freeze one setting on dev queries
lag     frozen setting under backlog, against signal-aware rivals at K=50 and K=100
unseen  frozen setting on untouched BEIR query sets
latency live, with HNSW"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys_argv = sys.argv; what = sys.argv[1]; sys.argv = ["x", "noop"]
from e13_af import *
sys.argv = sys_argv; what = sys_argv[1]
A = 50; PK = res("e14_pick.json"); F13 = json.load(open(PICK))
def cands(bm, idf, w0, S0, c0, E, v, h, sim):
    get = lambda ids, vv: E[ids] @ vv
    u, cs, l, e = af_run(bm, idf, w0, S0, c0, v, get, h)
    s = sim if h is None else np.where(h, -9.0, sim); k = min(A + 1, len(s)); top = np.argpartition(-s, k - 1)[:k]; top = top[s[top] > -9]
    extra = np.array(sorted(set(top.tolist()) - set(u.tolist())), dtype=np.int64)
    return np.concatenate([u, extra]), np.concatenate([cs, sim[extra]]), l, e
def rival(c0, S0, sim, h, K, nr, a):
    s = np.where(h, -9.0, sim) if h is not None else sim; k = min(K + 1, len(s)); top = np.argpartition(-s, k - 1)[:k]; top = top[s[top] > -9]; us = np.array(sorted(set(c0[:K + 1].tolist()) | set(top.tolist())), dtype=np.int64)
    vis = np.ones(len(us), bool) if h is None else ~h[us]; nd = np.zeros(len(us)); nd[vis] = norm(sim[us][vis], nr); nb = norm(S0[us], nr)
    return us, np.where(vis, a * nd + (1 - a) * nb, nb)
if what == "grid":
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, E = load(name); Q = np.load(res(f"{name}/bge_qemb.npy")); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); C = np.load(res(f"{name}/bm25_cands.npy"), allow_pickle=True); runs = {}
        for i, q in enumerate(qids):
            v = Q[i]; c0 = np.asarray(C[i], dtype=np.int64); w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(bm.n, np.float32); sim = E @ v; u, cs, l, _ = cands(bm, idf, w0, S0, c0, E, v, None, sim)
            for nr in ["max", "mm"]:
                for a in [0.7, 0.8, 0.9, 0.95]:
                    for g in [0.0, 0.15, 0.3]:
                        f = af_score(u, cs, S0, l, nr, a, g); o = np.argsort(-f, kind="stable"); runs.setdefault(f"{nr}|{a}|{g}", {})[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=300)
        per = {k: (lambda e: {"ndcg": e["ndcg"], "recall100": e["recall100"]})(evaluate(v, qrels)) for k, v in runs.items()}; save_json(per, name, "perquery_e14.json"); log(name, round(mean(per["max|0.9|0.0"]["ndcg"]), 4), round(mean(per["max|0.9|0.0"]["recall100"]), 3))
    P = {d: load_json(d, "perquery_e14.json") for d in DATASETS}; dv = {d: dev_ids(P[d]["max|0.9|0.0"]["ndcg"]) for d in DATASETS}
    b = max(P[DATASETS[0]], key=lambda k: np.mean([np.mean([P[d][k]["ndcg"][q] for q in dv[d]]) for d in DATASETS])); json.dump(b, open(PK, "w")); print("picked", b)
else:
    B14 = json.load(open(PK)) if os.path.exists(PK) else None
if what == "lag":
    nr, a, g = B14.split("|"); a, g = float(a), float(g); hp = {c: (F13[c].split("|")[1], float(F13[c].split("|")[2])) for c in ["h50", "h100"]}; afn, afa, afg = F13["af"].split("|")[1:]; afa, afg = float(afa), float(afg)
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, E = load(name); Q = np.load(res(f"{name}/bge_qemb.npy")); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); C = np.load(res(f"{name}/bm25_cands.npy"), allow_pickle=True); runs = {}; enc = {}; n = len(E); pos = {x: j for j, x in enumerate(doc_ids)}; rng = np.random.default_rng(SEED)
        masks = {f: np.random.default_rng(SEED).random(n) < f for f in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]}
        def add(k, q, u, f):
            o = np.argsort(-f, kind="stable"); runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=300)
        for i, q in enumerate(qids):
            v = Q[i]; c0 = np.asarray(C[i], dtype=np.int64); w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(n, np.float32); sim = E @ v
            rel = sorted(d for d, r in qrels[q].items() if r > 0 and d in pos); hid = np.zeros(n, bool)
            if rel: hid[[pos[d] for d in rng.choice(rel, size=max(1, len(rel) // 2), replace=False).tolist()]] = True
            for t, h in [*masks.items(), ("recent", hid)]:
                u, cs, l, e = cands(bm, idf, w0, S0, c0, E, v, h, sim); enc.setdefault(str(t), []).append(e); add(f"afh_{t}", q, u, af_score(u, cs, S0, l, nr, a, g))
                for c, K in [("h50", 50), ("h100", 100)]: add(f"{c}_{t}", q, *rival(c0, S0, sim, h, K, *hp[c]))
        per = {k: (lambda e: {"ndcg": e["ndcg"], "recall100": e["recall100"]})(evaluate(v, qrels)) for k, v in runs.items()}; per["__enc__"] = {k: float(np.mean(v)) for k, v in enc.items()}; save_json(per, name, "perquery_e14_lag.json")
        log(name, {k: round(mean(per[k]["ndcg"]), 3) for k in ["afh_0.0", "h50_0.0", "h100_0.0", "afh_0.1", "h50_0.1", "h100_0.1", "afh_recent", "h100_recent"]})
if what == "unseen":
    nr, a, g = B14.split("|"); a, g = float(a), float(g); m = encoder(BGE, 256)
    for name, sp in {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev", "quora": "dev"}.items():
        doc_ids, docs, qids, queries, qrels, E = load(name, sp); Q = encode(m, [PFX + q for q in queries], bs=16); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); runs = {}
        for i, q in enumerate(qids):
            v = Q[i]; w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(bm.n, np.float32); c0 = bm.topk(w0, 200)[0] if w0 else np.empty(0, dtype=np.int64); c0 = np.asarray(c0, dtype=np.int64); sim = E @ v
            u, cs, l, _ = cands(bm, idf, w0, S0, c0, E, v, None, sim); f = af_score(u, cs, S0, l, nr, a, g); o = np.argsort(-f, kind="stable"); runs[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=300)
        e = evaluate(runs, qrels); P = load_json(name, "perquery_e13_unseen.json"); P["afh"] = {"ndcg": e["ndcg"], "recall100": e["recall100"]}; save_json(P, name, "perquery_e13_unseen.json"); log(name, sp, {k: (round(mean(P[k]["ndcg"]), 4), round(mean(P[k]["recall100"]), 3)) for k in P})
if what == "latency":
    import hnswlib
    m_ = encoder(BGE, 256); nr, a, g = B14.split("|"); a, g = float(a), float(g); out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, E = load(name); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); qs = queries[:100]; r = {}
        for q in qs[:5]: encode(m_, [PFX + q], bs=1)
        for f in [0.0, 0.1]:
            hd = np.random.default_rng(SEED).random(len(E)) < f; vis = np.where(~hd)[0]
            H = hnswlib.Index(space="ip", dim=384); H.init_index(max_elements=len(vis), ef_construction=200, M=16); H.set_num_threads(os.cpu_count()); H.add_items(E[vis], vis); H.set_num_threads(1); H.set_ef(100); ts = []
            for q in qs:
                t0 = time.perf_counter(); w0 = weights(bm, q); S0 = bm.score(w0) if w0 else np.zeros(bm.n, np.float32); c0 = np.asarray(bm.topk(w0, 40)[0] if w0 else np.empty(0, dtype=np.int64), dtype=np.int64); v = encode(m_, [PFX + q], bs=1)[0]
                def get(ids, vv):
                    w = hd[ids]; X = np.array(E[ids], copy=True)
                    if w.any(): X[w] = encode(m_, [docs[int(i)] for i in ids[w]])
                    return X @ vv
                u, cs, l, _ = af_run(bm, idf, w0, S0, c0, v, get, hd); lab, _ = H.knn_query(v, k=A); ex = np.array(sorted(set(lab[0].tolist()) - set(u.tolist())), dtype=np.int64)
                u = np.concatenate([u, ex]); cs = np.concatenate([cs, E[ex] @ v]); sc = af_score(u, cs, S0, l, nr, a, g); np.argsort(-sc)[:10]; ts.append(time.perf_counter() - t0)
            r[f"f{f}"] = {"p50": float(np.median(ts) * 1e3), "p95": float(np.percentile(ts, 95) * 1e3)}; del H
        out[name] = r; log(name, r); save_json(out, "e14_latency.json")
