"""E4: document popularity, embedding-cache replay, a live cache run, and adaptive K (leave-one-domain-out)."""
import os, sys, time, hashlib
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

what = sys.argv[1]

def load_c(name):
    d = os.path.join(RES, name)
    return np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True)

def zipf_stream(n_items, s, n, seed=SEED):
    rng = np.random.default_rng(seed)
    if s == 0:
        return rng.integers(0, n_items, n)
    p = 1.0 / np.arange(1, n_items + 1) ** s
    return rng.choice(n_items, size=n, p=p / p.sum())

def replay(cands, stream, K, cap):
    lru, hit, tot = OrderedDict(), 0, 0
    for q in stream:
        for x in cands[q][:K]:
            x = int(x); tot += 1
            if x in lru:
                lru.move_to_end(x); hit += 1
            else:
                lru[x] = 1
                if len(lru) > cap:
                    lru.popitem(last=False)
    return hit / max(tot, 1)

if what == "cache":
    out = {}
    for name in DATASETS:
        cands = load_c(name)
        meta = load_json(name, "meta.json")
        N = meta["n_docs"]
        # 1. Skew in candidate lists when every query is seen once (no query repetition at all)
        K = 200
        cnt = {}
        for c in cands:
            for x in c[:K]:
                cnt[int(x)] = cnt.get(int(x), 0) + 1
        v = np.sort(np.array(list(cnt.values())))[::-1]
        tot = v.sum()
        top1 = v[:max(1, int(0.01 * len(v)))].sum() / tot
        r = np.arange(1, len(v) + 1)
        lim = max(10, int(len(v) * .5))
        slope = float(np.polyfit(np.log(r[:lim]), np.log(v[:lim]), 1)[0])
        g = 1 - 2 * np.sum((np.cumsum(v[::-1]) / tot)) / len(v) + 1 / len(v)
        out[name] = {"n_docs": N, "distinct": len(v), "occurrences": int(tot), "top1pct_share": float(top1), "gini": float(g), "zipf_slope": slope, "max_df": int(v[0]), "replay": {}}
        # 2. LRU replay over query streams; request count scales with cost and is capped for speed
        for s in [0.0, 0.8, 1.0, 1.2]:
            stream = zipf_stream(len(cands), s, 3000)
            for K in [50, 200, 1000]:
                for frac in [0.001, 0.01, 0.05, 0.1, 0.25]:
                    cap = max(10, int(N * frac))
                    h = replay(cands, stream if K < 1000 else stream[:1000], K, cap)
                    out[name]["replay"][f"{s}|{K}|{frac}"] = h
        log(name, out[name]["top1pct_share"], out[name]["replay"]["1.0|200|0.05"])
        save_json(out, "cache.json")

if what == "cachelive":
    # Wall-clock run. Keys are content hashes: an edited document misses, a deleted document can be purged.
    m = encoder()
    doc_ids, docs, qids, queries, qrels = load_dataset("fiqa")
    cands = load_c("fiqa")
    Q = np.load(os.path.join(RES, "fiqa", "qemb.npy"))
    out = {}
    for frac in [0.0, 0.05, 0.1]:
        cap = int(len(docs) * frac)
        cache, ts, enc_docs = OrderedDict(), [], 0
        stream = zipf_stream(len(qids), 1.0, 300)
        for q in stream:
            t = time.perf_counter()
            c = cands[q][:200]
            hs = [hashlib.blake2b(docs[int(x)].encode(), digest_size=16).digest() for x in c]
            X = np.zeros((len(c), 384), dtype=np.float32); miss = []
            for j, h in enumerate(hs):
                if h in cache:
                    X[j] = cache[h]; cache.move_to_end(h)
                else:
                    miss.append(j)
            if miss:
                V = encode(m, [docs[int(c[j])] for j in miss])
                enc_docs += len(miss)
                for j, v in zip(miss, V):
                    X[j] = v
                    if cap:
                        cache[hs[j]] = v
                        if len(cache) > cap:
                            cache.popitem(last=False)
            np.argsort(-(X @ Q[q]))[:10]
            ts.append(time.perf_counter() - t)
        a = np.array(ts[100:]) * 1e3
        out[frac] = {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()), "enc_docs": enc_docs, "cache_entries": len(cache)}
        # Erasure of one key
        if cache:
            k = next(iter(cache)); t = time.perf_counter(); del cache[k]; out[frac]["purge_us"] = (time.perf_counter() - t) * 1e6
        log(frac, out[frac]); save_json(out, "cachelive.json")

if what == "adaptive":
    import bm25s
    from sklearn.ensemble import HistGradientBoostingRegressor
    GRID = [10, 20, 50, 100, 200, 500, 1000]
    feats, ndcgs, names, qq = [], [], [], []
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        bm = BM25(docs)
        cands = load_c(name)
        sc = np.load(os.path.join(RES, name, "bm25_scores.npy"), allow_pickle=True)
        per = load_json(name, "perquery.json")
        E = np.load(os.path.join(RES, name, "emb.npy"), mmap_mode="r"); Q = np.load(os.path.join(RES, name, "qemb.npy"))
        for i, q in enumerate(qids):
            ts = bm.q_terms(queries[i])
            dfs = np.array([bm.indptr[t + 1] - bm.indptr[t] for t in ts] or [0], dtype=float)
            idf = np.log(1 + bm.n / (1 + dfs))
            s = np.asarray(sc[i], dtype=float)
            s = s if len(s) else np.zeros(1)
            top = s[:100]
            f = [len(queries[i].split()), len(ts), float(idf.mean()), float(idf.max()), float(idf.min()), float(idf.std()),
                 float(s[0]), float(s[0] / (s[min(9, len(s) - 1)] + 1e-6)), float(top.std()), float(s[0] - top.mean()), np.log1p(np.count_nonzero(s)),
                 np.log10(bm.n), float(np.log1p(dfs.sum()) - np.log(bm.n))]
            # probe features: cost is encoding 20 candidates, which any K>=20 already pays
            c = np.asarray(cands[i][:20], dtype=np.int64)
            if len(c) >= 2:
                cs = np.asarray(E[c]) @ Q[i]
                from scipy.stats import spearmanr
                rho = spearmanr(np.arange(len(c)), -cs).statistic
                f += [float(cs.max()), float(cs.mean()), float(cs.std()), 0.0 if np.isnan(rho) else float(rho), float(np.argmax(cs))]
            else:
                f += [0.0, 0.0, 0.0, 0.0, 0.0]
            feats.append(f); names.append(name); qq.append(q)
            ndcgs.append([per[f"jit_k{K}"]["ndcg"][q] for K in GRID])
    X, Y = np.array(feats, dtype=float), np.array(ndcgs)
    names = np.array(names)
    eps = 0.0
    target = np.array([np.log10(GRID[np.argmax(y >= y.max() - eps)]) for y in Y])  # smallest K reaching the query's best nDCG
    idx_of = {K: j for j, K in enumerate(GRID)}
    def apply(pred):
        """Round predicted log K to the grid and look up that query's measured nDCG."""
        grid = np.log10(GRID)
        j = np.abs(pred[:, None] - grid[None, :]).argmin(1)
        return np.array(GRID)[j], Y[np.arange(len(Y)), j]
    out = {"grid": GRID, "fixed": {}, "oracle": {}}
    for ds in DATASETS + ["ALL"]:
        msk = names == ds if ds != "ALL" else np.ones(len(names), bool)
        w = np.zeros(msk.sum()) + 1
        out["fixed"][ds] = {K: float(Y[msk, idx_of[K]].mean()) for K in GRID}
        b = np.array(GRID)[Y[msk].argmax(1)]
        out["oracle"][ds] = {"meanK": float(b.mean()), "ndcg": float(Y[msk].max(1).mean())}
    for lbl, cols in [("lexical", list(range(13))), ("lexical+probe", list(range(18)))]:
        pred = np.zeros(len(Y))
        for ds in DATASETS:  # leave one domain out
            tr, te = names != ds, names == ds
            r = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=SEED)
            r.fit(X[tr][:, cols], target[tr])
            pred[te] = r.predict(X[te][:, cols])
        out[lbl] = {}
        for delta in [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]:
            Ks, v = apply(pred + delta)
            rows = {}
            for ds in DATASETS + ["ALL"]:
                msk = names == ds if ds != "ALL" else np.ones(len(names), bool)
                # "ALL" pools queries here; macro averages across datasets are computed in make_tables.py
                rows[ds] = {"meanK": float(Ks[msk].mean()), "ndcg": float(v[msk].mean()), "p95K": float(np.percentile(Ks[msk], 95))}
            out[lbl][delta] = rows
            np.save(os.path.join(RES, f"adaptive_{lbl.replace('+','_')}_{delta}.npy"), np.c_[Ks, v])
        imp = None
        log(lbl, out[lbl][0.0]["ALL"])
    np.save(os.path.join(RES, "adaptive_X.npy"), X); np.save(os.path.join(RES, "adaptive_Y.npy"), Y); np.save(os.path.join(RES, "adaptive_names.npy"), names)
    log("fixed", out["fixed"]["ALL"], "oracle", out["oracle"]["ALL"])
    save_json(out, "adaptive.json")
