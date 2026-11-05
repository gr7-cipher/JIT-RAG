"""E3: cross-encoder comparison, learned-sparse first stage (SPLADE), model/truncation sweeps, BM25 grid,
and recall as a function of the measured lexical gap."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch

dev = device()
what = sys.argv[1]

def load(name):
    doc_ids, docs, qids, queries, qrels = load_dataset(name)
    d = os.path.join(RES, name)
    cands = np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True)
    return doc_ids, docs, qids, queries, qrels, cands, np.load(os.path.join(d, "emb.npy"), mmap_mode="r"), np.load(os.path.join(d, "qemb.npy"))

if what == "ce":
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=dev, max_length=256)
    CE_K = 200
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, cands, _, _ = load(name)
        n = min(len(qids), 150)
        ts, runs = [], {}
        for K in [50, 100, 200]:
            runs[K] = {}
        for i in range(n):
            c = cands[i][:CE_K + 1]
            if len(c) == 0:
                continue
            t = time.perf_counter()
            s = ce.predict([(queries[i], docs[int(x)]) for x in c], batch_size=64, show_progress_bar=False)
            ts.append((time.perf_counter() - t) / max(len(c), 1) * 1e3)
            for K in runs:
                cc, ss = c[:K + 1], s[:K + 1]
                o = np.argsort(-ss)
                runs[K][qids[i]] = run_from_ranked(doc_ids, qids[i], cc[o], ss[o])
        sub = {q: qrels[q] for q in qids[:n]}
        per = {f"ce_k{K}": evaluate(r, sub) for K, r in runs.items()}
        per["ms_per_pair"] = float(np.median(ts))
        save_json(per, name, "perquery_ce.json")
        log(name, {K: round(mean(per[f"ce_k{K}"]["ndcg"]), 4) for K in runs}, per["ms_per_pair"])

if what == "splade":
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import scipy.sparse as sp
    mn = "naver/splade-cocondenser-ensembledistil"
    tok = AutoTokenizer.from_pretrained(mn); mdl = AutoModelForMaskedLM.from_pretrained(mn).to(dev).eval()
    def enc(texts, bs, maxlen=256, top=256):
        rows, cols, vals = [], [], []
        order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), bs):
            ix = order[s:s + bs]
            b = tok([texts[i] for i in ix], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(dev)
            with torch.no_grad():
                lg = mdl(**b).logits
                v = (torch.log1p(torch.relu(lg)) * b["attention_mask"].unsqueeze(-1)).amax(1)
                w, ti = v.topk(top, dim=1)
            w, ti = w.cpu().numpy(), ti.cpu().numpy()
            for j, i in enumerate(ix):
                keep = w[j] > 0
                rows += [i] * int(keep.sum()); cols += ti[j][keep].tolist(); vals += w[j][keep].tolist()
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), tok.vocab_size), dtype=np.float32)
    out = {}
    for name in ["scifact", "nfcorpus", "arguana", "scidocs"]:
        doc_ids, docs, qids, queries, qrels, cands, E, Q = load(name)
        t = time.time(); D = enc(docs, 16); enc_s = time.time() - t
        Qs = enc(queries, 16)
        Dt = D.T.tocsr()
        ceil, per = {}, {}
        top_all = []
        tq = []
        for i in range(len(qids)):
            t = time.perf_counter()
            s = np.asarray(Dt[Qs[i].indices].T.dot(Qs[i].data)).ravel()
            k = min(2001, len(s))
            top = np.argpartition(-s, k - 1)[:k]; top = top[np.argsort(-s[top])]; top = top[s[top] > 0]
            tq.append(time.perf_counter() - t)
            top_all.append(top)
        per["splade"] = evaluate({q: run_from_ranked(doc_ids, q, c, -np.arange(len(c), dtype=float)) for q, c in zip(qids, top_all)}, qrels)
        for K in K_GRID:
            r = {}
            for i, q in enumerate(qids):
                c = top_all[i][:K + 1]
                s = np.asarray(E[c]) @ Q[i]; o = np.argsort(-s)
                r[q] = run_from_ranked(doc_ids, q, c[o], s[o])
            per[f"jit_splade_k{K}"] = evaluate(r, qrels)
            v = {}
            for q, c in zip(qids, top_all):
                rel = {x for x, rr in qrels[q].items() if rr > 0}
                v[q] = len(rel & {doc_ids[int(x)] for x in c[:K + 1] if doc_ids[int(x)] != q}) / len(rel)
            ceil[K] = mean(v)
        out[name] = {"encode_s": enc_s, "ms_per_doc": enc_s / len(docs) * 1e3, "query_ms": float(np.median(tq) * 1e3), "nnz_per_doc": D.nnz / len(docs), "ceiling": ceil, "bytes": int(D.data.nbytes + D.indices.nbytes + D.indptr.nbytes)}
        save_json(per, name, "perquery_splade.json"); save_json(out, "splade.json")
        log(name, out[name]["ms_per_doc"], round(mean(per["splade"]["ndcg"]), 4), ceil[200])

if what == "models":
    out = {}
    PFX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}
    for name in ["scifact", "nfcorpus", "arguana"]:
        doc_ids, docs, qids, queries, qrels, cands, _, _ = load(name)
        out[name] = {}
        union = sorted({int(x) for c in cands for x in c[:1001]})
        pos = {u: i for i, u in enumerate(union)}
        texts = [docs[u] for u in union]
        for mdl, ms in [(BASE_MODEL, 32), (BASE_MODEL, 64), (BASE_MODEL, 128), (BASE_MODEL, 256),
                        ("BAAI/bge-small-en-v1.5", 256), ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 256),
                        ("sentence-transformers/all-mpnet-base-v2", 256)]:
            m = encoder(mdl, ms)
            t = time.time(); X = encode(m, texts, bs=16 if "mpnet" in mdl else 32); s = (time.time() - t) / len(texts) * 1e3
            Q = encode(m, [PFX.get(mdl, "") + q for q in queries])
            key = f"{mdl.split('/')[-1]}@{ms}"
            out[name][key] = {"ms_per_doc": s}
            for K in [50, 100, 200, 500, 1000]:
                run = {}
                for i, q in enumerate(qids):
                    c = cands[i][:K + 1]
                    sc = X[np.array([pos[int(x)] for x in c], dtype=np.int64)] @ Q[i]; o = np.argsort(-sc)
                    run[q] = run_from_ranked(doc_ids, q, c[o], sc[o])
                out[name][key][K] = mean(evaluate(run, qrels)["ndcg"])
            log(name, key, out[name][key])
            del m, X
            if dev == "mps": torch.mps.empty_cache()
            save_json(out, "models.json")

if what == "bm25grid":
    out = {}
    for name in ["scifact", "nfcorpus", "scidocs", "arguana"]:
        doc_ids, docs, qids, queries, qrels, _, E, Q = load(name)
        E = np.asarray(E)
        out[name] = {}
        for k1 in [0.6, 0.9, 1.2, 1.5, 2.0]:
            for b in [0.2, 0.4, 0.75, 0.9]:
                bm = BM25(docs, k1=k1, b=b)
                r0, r1, ceil = {}, {}, {}
                for i, q in enumerate(qids):
                    c, s = bm.search(queries[i], 201)
                    r0[q] = run_from_ranked(doc_ids, q, c, s)
                    rel = {x for x, rr in qrels[q].items() if rr > 0}
                    ceil[q] = len(rel & set(list(r0[q])[:200])) / len(rel)
                    sc = E[c] @ Q[i]; o = np.argsort(-sc)
                    r1[q] = run_from_ranked(doc_ids, q, c[o], sc[o])
                out[name][f"{k1}/{b}"] = {"bm25_ndcg": mean(evaluate(r0, qrels)["ndcg"]), "jit200_ndcg": mean(evaluate(r1, qrels)["ndcg"]), "recall200": mean(ceil)}
        log(name, "grid done"); save_json(out, "bm25grid.json")

if what == "lexgap":
    # Measured gap: lambda(q,d) = 1 - IDF-weighted share of q's terms that occur in relevant document d.
    out = {}
    edges = [0, 0.25, 0.5, 0.75, 1.0001]
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, cands, E, Q = load(name)
        bm = BM25(docs); pos = {d: i for i, d in enumerate(doc_ids)}
        n = bm.n
        idf = {t: np.log(1 + n / (1 + bm.indptr[t + 1] - bm.indptr[t])) for t in range(len(bm.indptr) - 1)}
        ids = bm.tok.ids
        buckets = [{"n": 0, "hit": {K: 0 for K in K_GRID}, "dense100": 0} for _ in edges[:-1]]
        Ea = np.asarray(E)
        for i, q in enumerate(qids):
            qt = set(bm.q_terms(queries[i]))
            rank = {doc_ids[int(c)]: r for r, c in enumerate(cands[i])}
            s = Ea @ Q[i]
            top100 = set(np.argpartition(-s, min(100, len(s) - 1))[:100].tolist())
            for d, r in qrels[q].items():
                if r <= 0 or d not in pos:
                    continue
                dt = set(np.asarray(ids[pos[d]]).tolist())
                tot = sum(idf[t] for t in qt)
                lam = 1.0 if tot == 0 else min(1.0, max(0.0, 1 - sum(idf[t] for t in qt & dt) / tot))
                j = next(x for x in range(4) if edges[x] <= lam < edges[x + 1])
                b = buckets[j]; b["n"] += 1
                b["dense100"] += pos[d] in top100
                for K in K_GRID:
                    b["hit"][K] += rank.get(d, 10 ** 9) < K
        out[name] = buckets
        log(name, [b["n"] for b in buckets]); save_json(out, "lexgap.json")
