"""E6: experiments added after reviewing the first results.
canary    - what a delete changes on disk in an inverted index (hash every segment file)
cacheonce - embedding-cache hit rate when no query ever repeats
backlog   - streaming + JIT combined: ANN over encoded documents, query-time encoding for the backlog
latcal    - per-dataset encoding cost with one protocol
latrepeat - repeatability of the latency measurements
controller- QLF controller: per-query K under a latency budget, leave-one-domain-out
"""
import os, sys, time, hashlib, tempfile, shutil, secrets
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

what = sys.argv[1]

def cands_of(name):
    return np.load(os.path.join(RES, name, "bm25_cands.npy"), allow_pickle=True)

if what == "canary":
    import tantivy
    doc_ids, docs, *_ = load_dataset("fiqa")
    def build(path, ids, extra=None):
        b = tantivy.SchemaBuilder()
        b.add_integer_field("id", stored=False, indexed=True, fast=True)
        b.add_text_field("body", stored=True, tokenizer_name="en_stem")
        ix = tantivy.Index(b.build(), path=path); w = ix.writer(100_000_000, 1)
        for i in ids:
            w.add_document(tantivy.Document(id=int(i), body=docs[int(i)]))
        if extra:
            w.add_document(tantivy.Document(id=10 ** 7, body=extra))
        w.commit(); w.wait_merging_threads(); ix = tantivy.Index.open(path); return ix
    def snap(p):
        return {f: hashlib.sha256(open(os.path.join(p, f), "rb").read()).hexdigest() for f in os.listdir(p)
                if os.path.isfile(os.path.join(p, f)) and not f.endswith(".lock") and f != "meta.json"}
    tmp = tempfile.mkdtemp(dir=RES)
    canary = "zqx" + secrets.token_hex(12).replace("0", "g").replace("1", "h")
    ix = build(tmp, range(5000), "personal record for " + canary)
    hits = lambda i: i.searcher().search(i.parse_query(canary, ["body"]), 5).count
    before = snap(tmp); r = {"hits_before": hits(ix), "files_before": len(before)}
    raw = lambda p: sum(canary[6:].encode() in open(os.path.join(p, f), "rb").read() for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)))
    r["files_with_plain_text_before"] = raw(tmp)
    w = ix.writer(100_000_000, 1)
    t = time.perf_counter(); w.delete_documents("id", 10 ** 7); w.commit(); ix.reload(); r["delete_commit_ms"] = (time.perf_counter() - t) * 1e3
    w.wait_merging_threads(); del w
    after = snap(tmp)
    r["hits_after"] = hits(ix)
    r["old_files_unchanged"] = sum(after.get(f) == h for f, h in before.items())
    r["new_files"] = sorted(os.path.splitext(f)[1] for f in after if f not in before)
    r["files_with_plain_text_after"] = raw(tmp)
    t = time.time(); tmp2 = tempfile.mkdtemp(dir=RES); ix2 = build(tmp2, range(5000)); r["rewrite_5k_s"] = time.time() - t
    r["files_with_plain_text_after_rewrite"] = raw(tmp2)
    log(r); save_json(r, "canary.json")
    del ix, ix2; shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(tmp2, ignore_errors=True)

if what == "cacheonce":
    out = {}
    for name in DATASETS:
        c = cands_of(name); N = load_json(name, "meta.json")["n_docs"]; rng = np.random.default_rng(SEED)
        out[name] = {}
        for K in [50, 200, 1000]:
            for frac in [0.01, 0.05, 0.1, 0.25, 1.0]:
                cap = max(10, int(N * frac)); lru = OrderedDict(); hit = tot = 0
                for q in rng.permutation(len(c)):
                    for x in c[q][:K]:
                        x = int(x); tot += 1
                        if x in lru:
                            lru.move_to_end(x); hit += 1
                        else:
                            lru[x] = 1
                            if len(lru) > cap:
                                lru.popitem(last=False)
                out[name][f"{K}|{frac}"] = hit / max(tot, 1)
        log(name, out[name]["200|0.05"], out[name]["200|1.0"]); save_json(out, "cacheonce.json")

if what == "backlog":
    out = {}
    DEPTH, KJ = 100, 200
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        cands = cands_of(name)
        E = np.load(os.path.join(RES, name, "emb.npy")); Q = np.load(os.path.join(RES, name, "qemb.npy"))
        n = len(E); pos = {d: i for i, d in enumerate(doc_ids)}
        out[name] = {"uniform": {}, "recent": {}}
        S = E @ Q.T if n * len(Q) < 3e8 else None
        def scores(i):
            return S[:, i] if S is not None else E @ Q[i]
        per = {}
        def one(i, hidden_mask=None, hidden_set=None):
            """Returns ranked lists for ANN-only and for ANN + on-the-fly encoding of the backlog candidates."""
            s = scores(i).copy()
            c = np.asarray(cands[i][:KJ + 1], dtype=np.int64)
            if hidden_mask is not None:
                back = c[hidden_mask[c]] if len(c) else c
                s_vis = s.copy(); s_vis[hidden_mask] = -9
            else:
                back = np.array([x for x in c if int(x) in hidden_set], dtype=np.int64)
                s_vis = s.copy(); s_vis[list(hidden_set)] = -9
            top = np.argpartition(-s_vis, DEPTH)[:DEPTH]; top = top[np.argsort(-s_vis[top])]
            u = np.concatenate([top, back]); su = s[u]; o = np.argsort(-su)
            return (top, s[top]), (u[o], su[o]), len(back)
        # uniform backlog: documents wait for the encoder regardless of topic
        for f in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]:
            mask = np.random.default_rng(SEED).random(n) < f
            ra, rh, nb = {}, {}, []
            for i, q in enumerate(qids):
                a, h, b = one(i, hidden_mask=mask)
                ra[q] = run_from_ranked(doc_ids, q, *a); rh[q] = run_from_ranked(doc_ids, q, *h); nb.append(b)
            ea, eh = evaluate(ra, qrels), evaluate(rh, qrels)
            per[f"ann_{f}"] = ea["ndcg"]; per[f"hyb_{f}"] = eh["ndcg"]
            out[name]["uniform"][f] = {"ann": mean(ea["ndcg"]), "hybrid": mean(eh["ndcg"]), "encoded_per_q": float(np.mean(nb)), "p": paired_test(eh["ndcg"], ea["ndcg"])}
        # recent: half of each query's relevant documents are the newest ones, still in the backlog
        rng = np.random.default_rng(SEED); ra, rh, nb = {}, {}, []
        for i, q in enumerate(qids):
            rel = sorted(d for d, r in qrels[q].items() if r > 0 and d in pos)
            hid = {pos[d] for d in rng.choice(rel, size=max(1, len(rel) // 2), replace=False).tolist()} if rel else set()
            a, h, b = one(i, hidden_set=hid or {0})
            ra[q] = run_from_ranked(doc_ids, q, *a); rh[q] = run_from_ranked(doc_ids, q, *h); nb.append(b)
        ea, eh = evaluate(ra, qrels), evaluate(rh, qrels)
        out[name]["recent"] = {"ann": mean(ea["ndcg"]), "hybrid": mean(eh["ndcg"]), "encoded_per_q": float(np.mean(nb)), "p": paired_test(eh["ndcg"], ea["ndcg"])}
        log(name, out[name]["uniform"][0.1], out[name]["recent"]); save_json(out, "backlog.json")

def bench(m, bm, docs, queries, K, n_q):
    rows = []
    for q in queries[:n_q]:
        t0 = time.perf_counter(); idx, _ = bm.search(q, K); t1 = time.perf_counter(); v = encode(m, [q], bs=1); t2 = time.perf_counter()
        X = encode(m, [docs[int(i)] for i in idx]); t3 = time.perf_counter(); np.argsort(-(X @ v[0]))[:10]; t4 = time.perf_counter()
        rows.append((t1 - t0, t2 - t1, t3 - t2, len(idx), t4 - t0))
    a = np.array(rows)
    return {"first_ms": float(np.median(a[:, 0]) * 1e3), "qenc_ms": float(np.median(a[:, 1]) * 1e3), "ms_per_doc": float(a[:, 2].sum() / max(a[:, 3].sum(), 1) * 1e3), "p50": float(np.median(a[:, 4]) * 1e3), "p95": float(np.percentile(a[:, 4], 95) * 1e3)}

if what == "latcal":
    m = encoder(); out = {}
    encode(m, ["warm up"] * 64)
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        bm = BM25(docs)
        out[name] = {K: bench(m, bm, docs, queries, K, 15) for K in [50, 200]}
        out[name]["avg_words"] = float(np.mean([len(d.split()) for d in docs[:50000]]))
        log(name, out[name][200]["ms_per_doc"], out[name][200]["p50"]); save_json(out, "latcal.json")
        del bm

if what == "latrepeat":
    m = encoder(); out = {}
    for name in ["fiqa", "quora"]:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        bm = BM25(docs); encode(m, docs[:64]); out[name] = {}
        for K in [50, 200, 1000]:
            out[name][K] = [bench(m, bm, docs, queries, K, 15 if K < 1000 else 8) for _ in range(3)]
            log(name, K, [round(x["p50"]) for x in out[name][K]]); save_json(out, "latrepeat.json")
        del bm

if what == "controller":
    from sklearn.ensemble import HistGradientBoostingRegressor as GB
    from scipy.stats import wilcoxon
    X, Y = np.load(os.path.join(RES, "adaptive_X.npy")), np.load(os.path.join(RES, "adaptive_Y.npy"))
    names = np.load(os.path.join(RES, "adaptive_names.npy"))
    GRID = np.array([10, 20, 50, 100, 200, 500, 1000]); cal = load_json("latcal.json")
    t0 = {d: cal[d]["200"]["first_ms"] + cal[d]["200"]["qenc_ms"] for d in DATASETS}
    c = {d: cal[d]["200"]["ms_per_doc"] for d in DATASETS}
    cols = {"lexical": list(range(13)), "lexical+probe": list(range(18))}
    out = {"grid": GRID.tolist(), "t0": t0, "c": c}
    for lbl, col in cols.items():
        H = np.zeros_like(Y)
        for d in DATASETS:
            tr, te = names != d, names == d
            for j in range(len(GRID)):
                r = GB(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0, random_state=SEED)
                r.fit(X[tr][:, col], Y[tr, j]); H[te, j] = r.predict(X[te][:, col])
        np.save(os.path.join(RES, f"controller_pred_{lbl.replace('+','_')}.npy"), H)
        floor = 20 if "probe" in lbl else 0
        out[lbl] = {}
        for B in [100, 250, 500, 1000, 2000, 5000]:
            rows = {}
            for d in DATASETS:
                te = names == d
                L = t0[d] + c[d] * np.maximum(GRID, floor)
                ok = L <= B
                if not ok.any():
                    ok = np.zeros(len(GRID), bool); ok[0] = True
                # Baseline: largest K inside the budget, the only choice open to a static deployment
                jf = int(np.where(ok)[0].max())
                # Second baseline: the fixed K with the best average on other domains, within budget
                jo = int(np.argmax(Y[names != d].mean(0)[ok]))
                jo = int(np.where(ok)[0][jo])
                # Controller: most useful K with a small latency preference to break near-ties
                pen = 0.02 * L / max(B, 1)
                pick = np.argmax(np.where(ok, H[te] - pen, -9), axis=1)
                got = Y[te, pick]; fix = Y[te, jf]; fo = Y[te, jo]
                x = got - fix
                p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
                x2 = got - fo
                p2 = float(wilcoxon(x2).pvalue) if np.any(x2 != 0) else 1.0
                rows[d] = {"ndcg": float(got.mean()), "meanK": float(GRID[pick].mean()), "lat": float(L[pick].mean()), "fix_k": int(GRID[jf]), "fix_ndcg": float(fix.mean()), "p": p,
                           "best_fixed_k": int(GRID[jo]), "best_fixed_ndcg": float(fo.mean()), "p_best_fixed": p2, "oracle": float(Y[te][:, ok].max(1).mean()), "feasible": bool((L <= B).any())}
            out[lbl][B] = rows
        log(lbl, out[lbl][1000]["fiqa"]); save_json(out, "controller.json")

if what == "ksweep":
    # Clean repeat of the K sweep on an otherwise idle machine, one protocol for all collections.
    m = encoder(); out = {}
    for name in ["quora", "trec-covid", "fiqa"]:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        bm = BM25(docs); encode(m, docs[:64]); out[name] = {}
        for K in K_GRID:
            out[name][K] = bench(m, bm, docs, queries, K, 15)
            log(name, K, round(out[name][K]["p50"]), round(out[name][K]["ms_per_doc"], 2)); save_json(out, "ksweep.json")
        del bm
