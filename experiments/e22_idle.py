"""E22: live end-to-end time of lazy fusion with nothing else running, 100 queries per collection, one at a time, HNSW over visible documents."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a = sys.argv[1:]; sys.argv = ["x", "noop"]
from e14_afh import *
import hnswlib
m_ = encoder(BGE, 256); out = {}
for name in (_a or DATASETS):
    doc_ids, docs, qids, queries, qrels, E = load(name); bm = BM25(docs); qs = queries[:100]; r = {}
    for q in qs[:5]: encode(m_, [PFX + q], bs=1)
    for f in [0.0, 0.1, 0.5]:
        hd = np.random.default_rng(SEED).random(len(E)) < f; vis = np.where(~hd)[0]; H = hnswlib.Index(space="ip", dim=384); H.init_index(max_elements=len(vis), ef_construction=200, M=16); H.set_num_threads(os.cpu_count()); H.add_items(E[vis], vis); H.set_num_threads(1); H.set_ef(100)
        for K in [50, 100]:
            ts, en = [], []
            for q in qs:
                t0 = time.perf_counter(); w0 = weights(bm, q); S0 = bm.score(w0) if w0 else np.zeros(bm.n, np.float32); c0 = np.asarray(bm.topk(w0, K + 1)[0] if w0 else np.empty(0, dtype=np.int64), dtype=np.int64); v = encode(m_, [PFX + q], bs=1)[0]; lab, _ = H.knn_query(v, k=K + 1)
                w = c0[hd[c0]]; X = {}
                if len(w): X = dict(zip(w.tolist(), encode(m_, [docs[int(i)] for i in w])))
                us = np.array(sorted(set(c0.tolist()) | set(lab[0].tolist())), dtype=np.int64); V = np.array(E[us], copy=True)
                for j, u in enumerate(us.tolist()):
                    if u in X: V[j] = X[u]
                f_ = 0.9 * norm(V @ v, "max") + 0.1 * norm(S0[us], "max"); np.argsort(-f_)[:10]; ts.append(time.perf_counter() - t0); en.append(len(w))
            r[f"lazy{K}_f{f}"] = {"p50": float(np.median(ts) * 1e3), "p95": float(np.percentile(ts, 95) * 1e3), "enc": float(np.mean(en))}
        del H
    out[name] = r; log(name, {k: (round(v["p50"]), round(v["p95"]), round(v["enc"], 1)) for k, v in r.items()}); save_json(out, "e22_idle.json")
