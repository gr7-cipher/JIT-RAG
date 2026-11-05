"""E5: ingest visibility under load, quality with not-yet-visible documents, deletion behaviour, ACL filtering."""
import os, sys, time, threading, queue, shutil, tempfile, secrets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import hnswlib, tantivy

what = sys.argv[1]

def t_index(path, heap=200_000_000):
    b = tantivy.SchemaBuilder()
    b.add_integer_field("id", stored=False, indexed=True, fast=True)
    b.add_text_field("body", stored=False, tokenizer_name="en_stem")
    s = b.build()
    return tantivy.Index(s, path=path), s

if what == "stream":
    doc_ids, docs, qids, queries, qrels = load_dataset("fiqa")
    E = np.load(os.path.join(RES, "fiqa", "emb.npy"), mmap_mode="r")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(docs))
    pre, fresh = perm[:50000], perm[50000:]
    m = encoder()
    out = {"lucene_like": {}, "hnsw": {}}
    DUR = 20
    def arrivals(rate):
        n = min(len(fresh), int(rate * DUR))
        gaps = rng.exponential(1.0 / rate, n)
        return np.cumsum(gaps)
    # ---- inverted index (the JIT-RAG side): NRT segments, visible after commit + reader reload
    tmp = tempfile.mkdtemp(dir=RES)
    ix, _ = t_index(tmp)
    w = ix.writer(200_000_000, 1)
    t = time.time()
    for i in pre:
        w.add_document(tantivy.Document(id=int(i), body=docs[int(i)]))
    w.commit(); ix.reload(); out["tantivy_build_s"] = time.time() - t
    for rate in [100, 500, 1000, 5000]:
        for R in [0.1, 1.0]:
            arr = arrivals(rate); lags = []; start = time.perf_counter(); j = 0; n = len(arr); pending = []; last = start
            while j < n or pending:
                now = time.perf_counter() - start
                while j < n and arr[j] <= now:
                    d = int(fresh[j]); w.add_document(tantivy.Document(id=d, body=docs[d])); pending.append(arr[j]); j += 1
                    now = time.perf_counter() - start
                if time.perf_counter() - last >= R or (j >= n and pending):
                    w.commit(); ix.reload(); vis = time.perf_counter() - start
                    lags += [vis - a for a in pending]; pending = []; last = time.perf_counter()
                else:
                    time.sleep(0.001)
            total = time.perf_counter() - start
            out["lucene_like"][f"{rate}|{R}"] = {"p50_s": float(np.percentile(lags, 50)), "p95_s": float(np.percentile(lags, 95)), "max_s": float(np.max(lags)), "docs": n, "achieved_rate": n / total}
            log("inv", rate, R, out["lucene_like"][f"{rate}|{R}"])
            save_json(out, "stream.json")
    del w, ix; shutil.rmtree(tmp, ignore_errors=True)
    # ---- streaming dense: encode + HNSW insert, continuously or in micro-batches
    for rate in [25, 50, 100, 200, 500]:
        for W, label in [(0.0, "continuous"), (1.0, "1s"), (10.0, "10s")]:
            hn = hnswlib.Index(space="ip", dim=384); hn.init_index(max_elements=len(docs), ef_construction=200, M=16)
            hn.set_num_threads(os.cpu_count()); hn.add_items(np.asarray(E[pre]), pre)
            arr = arrivals(rate); n = len(arr); start = time.perf_counter(); j = 0; pending = []; lags = []; last = start; enc_t = ins_t = 0
            deadline = DUR * 4  # abandon when the pipeline clearly cannot keep up
            while (j < n or pending) and time.perf_counter() - start < deadline:
                now = time.perf_counter() - start
                while j < n and arr[j] <= now:
                    pending.append((arr[j], int(fresh[j]))); j += 1
                due = (W == 0 and pending) or (W > 0 and time.perf_counter() - last >= W) or (j >= n and pending)
                if due and pending:
                    batch = pending[:64] if W == 0 else pending[:2000]
                    pending = pending[len(batch):]
                    t0 = time.perf_counter(); V = encode(m, [docs[d] for _, d in batch], bs=64); t1 = time.perf_counter()
                    hn.add_items(V, [d for _, d in batch]); t2 = time.perf_counter()
                    enc_t += t1 - t0; ins_t += t2 - t1
                    vis = time.perf_counter() - start
                    lags += [vis - a for a, _ in batch]; last = time.perf_counter()
                else:
                    time.sleep(0.001)
            total = time.perf_counter() - start
            out["hnsw"][f"{rate}|{label}"] = {"p50_s": float(np.percentile(lags, 50)), "p95_s": float(np.percentile(lags, 95)), "max_s": float(np.max(lags)), "docs_visible": len(lags), "docs_offered": n,
                                              "backlog_end": n - len(lags), "achieved_rate": len(lags) / total, "encode_s": enc_t, "insert_s": ins_t}
            log("hnsw", rate, label, out["hnsw"][f"{rate}|{label}"])
            save_json(out, "stream.json")
            del hn
    log("DONE stream")

if what == "visibility":
    # Quality for queries whose relevant documents arrived recently, with documents hidden vs. visible.
    out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        cands = np.load(os.path.join(RES, name, "bm25_cands.npy"), allow_pickle=True)
        E = np.load(os.path.join(RES, name, "emb.npy")); Q = np.load(os.path.join(RES, name, "qemb.npy"))
        rng = np.random.default_rng(SEED)
        res = {"hidden": {}, "visible": {}}
        # Each query's relevant documents form the "recent" set: hide half of them, at least one.
        pos = {d: i for i, d in enumerate(doc_ids)}
        hide = {}
        for q in qids:
            rel = sorted(d for d, r in qrels[q].items() if r > 0 and d in pos)
            n = max(1, len(rel) // 2)
            hide[q] = set(rng.choice(rel, size=n, replace=False).tolist()) if rel else set()
        for mode in ["dense", "jit200"]:
            rv, rh = {}, {}
            for i, q in enumerate(qids):
                if mode == "dense":
                    s = E @ Q[i]; k = min(1100, len(s)); top = np.argpartition(-s, k - 1)[:k]; top = top[np.argsort(-s[top])]; sc = s[top]
                else:
                    c = np.asarray(cands[i], dtype=np.int64)
                    hid = {pos[d] for d in hide[q]}
                    c_vis = c[:201]; cs = E[c_vis] @ Q[i] if len(c_vis) else np.empty(0); o = np.argsort(-cs); top, sc = c_vis[o], cs[o]
                    c2 = np.array([x for x in c if int(x) not in hid][:201], dtype=np.int64)
                    cs2 = E[c2] @ Q[i] if len(c2) else np.empty(0); o2 = np.argsort(-cs2)
                    rh[q] = run_from_ranked(doc_ids, q, c2[o2], cs2[o2])
                rv[q] = run_from_ranked(doc_ids, q, top, sc)
                if mode == "dense":
                    hid = {pos[d] for d in hide[q]}
                    keep = np.array([int(x) not in hid for x in top])
                    rh[q] = run_from_ranked(doc_ids, q, top[keep], sc[keep])
            res["visible"][mode] = mean(evaluate(rv, qrels)["ndcg"]); res["hidden"][mode] = mean(evaluate(rh, qrels)["ndcg"])
        out[name] = res
        log(name, res); save_json(out, "visibility.json")

if what == "erasure":
    out = {"hnsw": {}, "rebuild": {}}
    doc_ids, docs, qids, queries, qrels = load_dataset("fiqa")
    E = np.load(os.path.join(RES, "fiqa", "emb.npy")); Q = np.load(os.path.join(RES, "fiqa", "qemb.npy"))
    n = len(E); rng = np.random.default_rng(SEED); perm = rng.permutation(n)
    def build(X, ids):
        h = hnswlib.Index(space="ip", dim=384); h.init_index(max_elements=len(X), ef_construction=200, M=16)
        h.set_num_threads(os.cpu_count()); h.add_items(X, ids); h.set_num_threads(1); h.set_ef(100); return h
    h = build(E, np.arange(n))
    def q_eval(h, alive):
        ts, ov = [], []
        al = np.where(alive)[0]
        for v in Q:
            t = time.perf_counter(); lab, _ = h.knn_query(v, k=10); ts.append(time.perf_counter() - t)
            s = E[al] @ v; true = al[np.argpartition(-s, 9)[:10]]
            ov.append(len(set(lab[0]) & set(true)) / 10)
        return float(np.percentile(ts, 50) * 1e3), float(np.percentile(ts, 95) * 1e3), float(np.mean(ov))
    alive = np.ones(n, bool); done = 0
    out["hnsw"]["0"] = dict(zip(["ms_p50", "ms_p95", "recall10"], q_eval(h, alive)))
    for pct in [1, 5, 10, 20, 50, 80]:
        tgt = int(n * pct / 100); ids = perm[done:tgt]
        t = time.perf_counter()
        for x in ids:
            h.mark_deleted(int(x))
        del_us = (time.perf_counter() - t) / max(len(ids), 1) * 1e6
        alive[ids] = False; done = tgt
        r = dict(zip(["ms_p50", "ms_p95", "recall10"], q_eval(h, alive))); r["delete_us_per_doc"] = del_us
        # rebuild for comparison
        al = np.where(alive)[0]
        t = time.time(); h2 = build(E[al], al); r["rebuild_s"] = time.time() - t
        r["rebuilt"] = dict(zip(["ms_p50", "ms_p95", "recall10"], q_eval(h2, alive)))
        del h2
        out["hnsw"][str(pct)] = r
        log(pct, r); save_json(out, "erasure.json")
    # Residue: is a logically deleted vector still in a saved index?
    v = E[perm[0]].tobytes()
    p = os.path.join(RES, "tmp_hnsw.bin"); h.save_index(p); raw = open(p, "rb").read(); out["hnsw_residue_after_mark_deleted"] = v in raw; os.remove(p)
    # Inverted index: canary test. Tokens of a deleted document stay on disk until a merge rewrites the segment.
    tmp = tempfile.mkdtemp(dir=RES)
    ix, schema = t_index(tmp); w = ix.writer(100_000_000, 1)
    canary = "zqx" + secrets.token_hex(12).replace("0", "g").replace("1", "h")
    for i in perm[:5000]:
        w.add_document(tantivy.Document(id=int(i), body=docs[int(i)]))
    w.add_document(tantivy.Document(id=10 ** 7, body="personal record for " + canary))
    w.commit(); ix.reload()
    def on_disk(c):
        tail = c[6:].encode()
        found = 0
        for f in os.listdir(tmp):
            fp = os.path.join(tmp, f)
            if os.path.isfile(fp):
                b = open(fp, "rb").read()
                if tail in b or tail[::-1] in b:
                    found += 1
        return found
    def hits(c):
        return ix.searcher().search(ix.parse_query(c, ["body"]), 5).count
    r = {"before_delete_hits": hits(canary), "before_delete_files": on_disk(canary)}
    t = time.perf_counter(); w.delete_documents("id", 10 ** 7); w.commit(); ix.reload(); r["delete_commit_ms"] = (time.perf_counter() - t) * 1e3
    r["after_delete_hits"] = hits(canary); r["after_delete_files"] = on_disk(canary)
    t = time.time(); tmp2 = tempfile.mkdtemp(dir=RES); ix2, _ = t_index(tmp2); w2 = ix2.writer(100_000_000, 1)
    for i in perm[:5000]:
        w2.add_document(tantivy.Document(id=int(i), body=docs[int(i)]))
    w2.commit(); r["rewrite_5k_s"] = time.time() - t
    out["inverted_canary"] = r
    log(r)
    del w, ix, w2, ix2; shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(tmp2, ignore_errors=True)
    # Rebuild scaling
    for nn in [5000, 20000, 57638]:
        idx = perm[:min(nn, n)]
        t = time.time(); hh = build(E[idx], idx); out["rebuild"][nn] = {"hnsw_only_s": time.time() - t}
        del hh
    save_json(out, "erasure.json"); log("DONE erasure")

if what == "filter":
    # Tenant/ACL filter: a document is visible with probability s. JIT-RAG filters inside BM25 scoring.
    out = {}
    for name in ["fiqa", "scidocs"]:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        E = np.load(os.path.join(RES, name, "emb.npy")); Q = np.load(os.path.join(RES, name, "qemb.npy"))
        bm = BM25(docs); n = len(E)
        h = hnswlib.Index(space="ip", dim=384); h.init_index(max_elements=n, ef_construction=200, M=16); h.set_num_threads(os.cpu_count()); h.add_items(E, np.arange(n)); h.set_num_threads(1)
        out[name] = {}
        for sel in [1.0, 0.5, 0.1, 0.01, 0.001]:
            rng = np.random.default_rng(SEED)
            mask = rng.random(n) < sel
            al = np.where(mask)[0]; mf = mask.astype(np.float32); ms = set(al.tolist())
            pos = {d: i for i, d in enumerate(doc_ids)}
            qr = {q: {d: r for d, r in qrels[q].items() if d in pos and mask[pos[d]]} for q in qids}
            use = [i for i, q in enumerate(qids) if any(r > 0 for r in qr[q].values())]
            r_e, r_j, r_h = {}, {}, {}
            tj, th, ov = [], [], []
            flt = lambda x: mask[x]
            for ef in [100]:
                h.set_ef(ef)
                for i in use[:200]:
                    q = qids[i]
                    s = E[al] @ Q[i]; k = min(100, len(al)); top = np.argpartition(-s, k - 1)[:k]; top = top[np.argsort(-s[top])]
                    r_e[q] = run_from_ranked(doc_ids, q, al[top], s[top])
                    t = time.perf_counter(); c, _ = bm.search(queries[i], 201, mask=mf); tj.append(time.perf_counter() - t)
                    sc = E[c] @ Q[i]; o = np.argsort(-sc); r_j[q] = run_from_ranked(doc_ids, q, c[o], sc[o])
                    t = time.perf_counter()
                    try:
                        lab, dist = h.knn_query(Q[i], k=min(100, len(al)), filter=flt)
                        ids, ss = lab[0], 1 - dist[0]
                    except RuntimeError:
                        ids, ss = np.empty(0, dtype=np.int64), np.empty(0)
                    th.append(time.perf_counter() - t)
                    r_h[q] = run_from_ranked(doc_ids, q, ids, ss)
                    ov.append(len(set(ids[:10].tolist()) & set(al[top[:10]].tolist())) / min(10, len(al)))
            sub = {qids[i]: qr[qids[i]] for i in use[:200]}
            out[name][sel] = {"n_q": len(sub), "ndcg_exact": mean(evaluate(r_e, sub)["ndcg"]), "ndcg_jit200": mean(evaluate(r_j, sub)["ndcg"]), "ndcg_hnsw": mean(evaluate(r_h, sub)["ndcg"]),
                              "hnsw_ms_p50": float(np.median(th) * 1e3), "hnsw_ms_p95": float(np.percentile(th, 95) * 1e3), "bm25_ms_p50": float(np.median(tj) * 1e3), "hnsw_recall10_vs_exact": float(np.mean(ov))}
            log(name, sel, out[name][sel]); save_json(out, "filter.json")
