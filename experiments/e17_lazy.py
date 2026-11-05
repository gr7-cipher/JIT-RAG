"""E17: ablation of JIT-AF+ANN and the lazy Hybrid CC baseline.
Lazy Hybrid CC is frozen Hybrid CC (max normalization, alpha 0.9, no feedback) that encodes its waiting BM25 candidates at
query time. Scoring reads stored vectors, so a run measures quality and counts the encodings a live system would pay.
Usage: e17_lazy.py [set ...]. Sets: the seven collections, or external ones (see EXT). Writes results/e17/<set>.json."""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a = sys.argv[1:]; sys.argv = ["x", "noop"]
from e14_afh import *
sys.argv = ["x"] + _a
NR = "max"; AF_A, AF_G = 0.8, 0.3; H_A = 0.9; SEEDS = [13, 14, 15, 16, 17]; FS = [0.01, 0.05, 0.1, 0.25, 0.5]; MS = [5, 10, 20, 50]
EXT = ["webis-touche2020", "cqa-webmasters", "cqa-android", "cqa-wordpress", "fiqa_fresh", "nfcorpus_fresh"]
OUT = os.path.join(RES, "e17"); os.makedirs(OUT, exist_ok=True)
def fresh(name, n_q=1000):
    doc_ids, docs, qids, queries, qrels = load_dataset(name, max_q=0, split="train")
    old = set(load_dataset(name, split="train")[2]); keep = [i for i, q in enumerate(qids) if q not in old]
    random.Random(SEED + 1).shuffle(keep); keep = sorted(keep[:n_q])
    return doc_ids, docs, [qids[i] for i in keep], [queries[i] for i in keep], {qids[i]: qrels[qids[i]] for i in keep}
class Metrics:
    """trec_eval conventions: ties by score then docno, both descending; gain = grade; relevance level 1."""
    def __init__(s, doc_ids, qids, qrels):
        s.ids = doc_ids; s.qrels = qrels; s.pos = {d: i for i, d in enumerate(doc_ids)}; o = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i], reverse=True)
        s.dr = np.empty(len(doc_ids), np.int64); s.dr[np.array(o)] = np.arange(len(o)); s.R = {}
        for q in qids:
            g = np.zeros(len(doc_ids) if False else 0); rel = {s.pos[d]: r for d, r in qrels[q].items() if d in s.pos}
            ideal = np.array(sorted((r for r in qrels[q].values() if r > 0), reverse=True)[:10], float)
            s.R[q] = (rel, float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()), sum(1 for r in qrels[q].values() if r > 0), s.pos.get(q, -1))
    def __call__(s, q, us, f):
        us = np.asarray(us, np.int64); f = np.asarray(f, np.float64); rel, idcg, nrel, me = s.R[q]
        if len(us) and me >= 0:
            k = us != me; us, f = us[k], f[k]
        o = np.lexsort((s.dr[us], -f))[:100]; top = us[o].tolist(); dcg = 0.0; r10 = 0; r100 = 0; j = 0
        for i, d in enumerate(top):
            g = rel.get(d)
            if g is not None and i < 10: j += 1
            if g is not None and g > 0:
                r100 += 1
                if i < 10: dcg += g / np.log2(i + 2); r10 += 1
        return (dcg / idcg if idcg else 0.0, r10 / nrel if nrel else 0.0, r100 / nrel if nrel else 0.0, j / min(10, len(top)) if top else 0.0, len(us))
class Store:
    def __init__(s): s.d = {}
    def add(s, k, m, enc=None):
        v = s.d.setdefault(k, ([], [], [], [], [], []))
        for a, b in zip(v, m): a.append(b)
        v[5].append(0 if enc is None else enc)
    def dump(s, qids): return {k: {"ndcg": np.round(v[0], 6).tolist(), "r10": np.round(v[1], 6).tolist(), "r100": np.round(v[2], 6).tolist(), "j10": np.round(v[3], 4).tolist(), "pool": float(np.mean(v[4])), "enc": float(np.mean(v[5])), "n": len(v[0])} for k, v in s.d.items()}
def ann_extra(order, sim, h, u, A=51):
    """Top A visible documents by cosine that are not in u."""
    top = order[:A] if h is None else order[~h[order]][:A]; return np.array(sorted(set(top.tolist()) - set(u.tolist())), dtype=np.int64)
def visible(order, h, k): return order[:k] if h is None else order[~h[order]][:k]
def hyb(c0, S0, sim, top, K, h, m=None, lazy=False, a=H_A, g=0.0, l=None):
    """Union of the BM25 and ANN top K. Waiting documents enter by BM25. lazy: encode them, best BM25 first, at most m. Not lazy: lexical score only."""
    bm = c0[:K + 1]; us = np.array(sorted(set(bm.tolist()) | set(top[:K + 1].tolist())), dtype=np.int64); known = np.ones(len(us), bool); enc = 0
    if h is not None:
        known = ~h[us]
        if lazy:
            w = bm[h[bm]]; e = w if m is None else w[:m]; enc = len(e); known |= np.isin(us, e)
    nb = norm(S0[us], NR) if g == 0 else (1 - g) * norm(S0[us], NR) + g * norm(l[us], NR); nd = np.zeros(len(us)); nd[known] = norm(sim[us][known], NR)
    return us, np.where(known, a * nd + (1 - a) * nb, nb), enc
def rrf(order, c0, h):
    d = order[:1000] if h is None else order[~h[order]][:1000]; scr = {}
    for rk, x in enumerate(d.tolist()): scr[x] = scr.get(x, 0.0) + 1.0 / (61 + rk)
    for rk, x in enumerate(c0[:1000].tolist()): scr[x] = scr.get(x, 0.0) + (2.0 if h is not None and h[x] else 1.0) / (61 + rk)
    ks = np.array(list(scr), dtype=np.int64); return ks, np.array([scr[x] for x in ks.tolist()])
def stream(bm, idf, W, Ss, Cs, Q, E, h0, kind, K=None):
    """Replay queries once with write-back. Returns encodings with write-back and without."""
    have = ~h0; tot = 0; plain = 0; get = lambda ids, v: E[ids] @ v
    for i in np.random.default_rng(SEED).permutation(len(Q)):
        c0 = Cs[i]
        if kind == "lazy": ids = c0[:K + 1]
        else: ids = af_run(bm, idf, W[i], Ss(i), c0, Q[i], get)[0]
        w = ids[~have[ids]]; tot += len(w); plain += int(h0[ids].sum()); have[w] = True
    return tot, plain
def run(tag):
    t0 = time.time(); base = tag.replace("_fresh", ""); isfresh = tag.endswith("_fresh")
    doc_ids, docs, qids, queries, qrels = fresh(base) if isfresh else load_dataset(base, max_q=500)
    E = np.load(res(f"{base}/bge_emb.npy"), mmap_mode="r"); E = np.asarray(E); n = len(E); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr)))
    seven = tag in DATASETS
    if seven: Q = np.load(res(f"{base}/bge_qemb.npy")); C = np.load(res(f"{base}/bm25_cands.npy"), allow_pickle=True)
    else: Q = encode(encoder(BGE, 256), [PFX + q for q in queries], bs=16); C = None
    M = Metrics(doc_ids, qids, qrels); S = Store(); pos = M.pos; rng = np.random.default_rng(SEED); get = lambda ids, v: E[ids] @ v
    masks = {(f, s): np.random.default_rng(s).random(n) < f for f in FS for s in SEEDS}; budf = [0.1, 0.5, 1.0]; ones = np.ones(n, bool); W = []; Cs = []
    for i, q in enumerate(qids):
        v = Q[i]; w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(n, np.float32); W.append(w0); c0 = np.asarray(C[i], dtype=np.int64) if seven else np.asarray(bm.topk(w0, 2000)[0] if w0 else np.empty(0, dtype=np.int64), dtype=np.int64); Cs.append(c0[:1000].copy())
        sim = E @ v; k = min(5000, n - 1) if n > 5001 else n; order = np.argpartition(-sim, k - 1)[:k] if k < n else np.arange(n); order = order[np.argsort(-sim[order], kind="stable")]
        u0, cs0, l0, _ = af_run(bm, idf, w0, S0, c0, v, get); top0 = order[:101]
        # ablation at a complete index
        ex = ann_extra(order, sim, None, u0); ua = np.concatenate([u0, ex]); ca = np.concatenate([cs0, sim[ex]])
        S.add("full", M(q, ua, af_score(ua, ca, S0, l0, NR, AF_A, AF_G))); S.add("noGamma", M(q, ua, af_score(ua, ca, S0, l0, NR, AF_A, 0.0))); S.add("full_a9", M(q, ua, af_score(ua, ca, S0, l0, NR, 0.9, AF_G))); S.add("noGamma_a9", M(q, ua, af_score(ua, ca, S0, l0, NR, 0.9, 0.0)))
        S.add("noANN", M(q, u0, af_score(u0, cs0, S0, l0, NR, AF_A, AF_G))); S.add("jitaf", M(q, u0, af_score(u0, cs0, S0, l0, "max", 0.95, AF_G)))
        b = c0[:100]; exb = ann_extra(order, sim, None, b); ub = np.concatenate([b, exb]); cb = sim[ub]
        S.add("noExp", M(q, ub, af_score(ub, cb, S0, S0, NR, AF_A, 0.0))); S.add("noExp_a9", M(q, ub, af_score(ub, cb, S0, S0, NR, 0.9, 0.0)))
        if len(b):
            sc = 0.2 * mm(S0[b].astype(np.float64)) + 0.8 * mm(sim[b]); o = np.argsort(-sc)[:AF["f"]]; p = sc[o] + 1e-9; p = p / p.sum(); lb = bm.score(prf(bm, idf, w0, b[o], p, AF))
        else: lb = S0
        S.add("noExp_gamma", M(q, ub, af_score(ub, cb, S0, lb, NR, AF_A, AF_G)))
        for K, nm in [(50, "h50"), (100, "h100")]:
            us, f, _ = hyb(c0, S0, sim, top0, K, None); S.add(nm, M(q, us, f))
        us, f, _ = hyb(c0, S0, sim, top0, 100, None, a=AF_A); S.add("h100_a8", M(q, us, f)); us, f, _ = hyb(c0, S0, sim, top0, 100, None, a=AF_A, g=AF_G, l=l0); S.add("h100_gamma", M(q, us, f))
        c = c0[:101]; S.add("jit100", M(q, c, sim[c])); S.add("jitcc100", M(q, c, 0.95 * norm(sim[c], NR) + 0.05 * norm(S0[c], NR)) if len(c) else M(q, c, c)); S.add("dense", M(q, order[:1000], sim[order[:1000]])); S.add("rrf", M(q, *rrf(order, c0, None))); S.add("bm25", M(q, c0[:1000], S0[c0[:1000]]))
        # backlog, unlimited budget
        for (f_, s), h in masks.items():
            ex = ann_extra(order, sim, h, u0); ua = np.concatenate([u0, ex]); ca = np.concatenate([cs0, sim[ex]]); fk = f"|f{f_}|s{s}"
            S.add("afh" + fk, M(q, ua, af_score(ua, ca, S0, l0, NR, AF_A, AF_G)), int(h[u0].sum())); top = visible(order, h, 101)
            for K in [50, 100]:
                us, f, e = hyb(c0, S0, sim, top, K, h, lazy=True); S.add(f"lazy{K}" + fk, M(q, us, f), e); us, f, _ = hyb(c0, S0, sim, top, K, h); S.add(f"sa{K}" + fk, M(q, us, f))
            if s == SEED: S.add("rrfsa" + fk, M(q, *rrf(order, c0, h)))
        # relevant documents waiting
        rel = sorted(d for d, r in qrels[q].items() if r > 0 and d in pos); hid = np.zeros(n, bool)
        if rel: hid[[pos[d] for d in rng.choice(rel, size=max(1, len(rel) // 2), replace=False).tolist()]] = True
        ex = ann_extra(order, sim, hid, u0); ua = np.concatenate([u0, ex]); ca = np.concatenate([cs0, sim[ex]]); S.add("afh|recent", M(q, ua, af_score(ua, ca, S0, l0, NR, AF_A, AF_G)), int(hid[u0].sum())); top = visible(order, hid, 101)
        for K in [50, 100]:
            us, f, e = hyb(c0, S0, sim, top, K, hid, lazy=True); S.add(f"lazy{K}|recent", M(q, us, f), e); us, f, _ = hyb(c0, S0, sim, top, K, hid); S.add(f"sa{K}|recent", M(q, us, f))
        S.add("rrfsa|recent", M(q, *rrf(order, c0, hid)))
        # encoding budget
        for f_ in budf:
            h = ones if f_ == 1.0 else masks[(f_, SEED)]; top = visible(order, h, 101) if f_ < 1 else order[:0]
            for m_ in MS:
                fk = f"|f{f_}|m{m_}"; u, cs, l, e = af_run(bm, idf, w0, S0, c0, v, get, h, m_); ex = ann_extra(order, sim, h, u) if f_ < 1 else np.empty(0, dtype=np.int64); ua = np.concatenate([u, ex]); ca = np.concatenate([cs, sim[ex]])
                S.add("afh" + fk, M(q, ua, af_score(ua, ca, S0, l, NR, AF_A, AF_G)), e)
                for K in [50, 100]:
                    us, f, e2 = hyb(c0, S0, sim, top, K, h, m=m_, lazy=True); S.add(f"lazy{K}" + fk, M(q, us, f), e2)
    out = {"qids": qids, "N": n, "sys": S.dump(qids)}; wb = {}
    if seven or isfresh is False:
        Ss = lambda i: bm.score(W[i]) if W[i] else np.zeros(n, np.float32)
        for f_ in [0.1, 0.5, 1.0]:
            h0 = ones if f_ == 1.0 else masks[(f_, SEED)]
            for kind, K, nm in [("lazy", 50, "lazy50"), ("lazy", 100, "lazy100"), ("af", None, "afh")]:
                wb[f"{nm}|f{f_}"] = stream(bm, idf, W, Ss, Cs, Q, E, h0, kind, K)
    out["writeback"] = wb; out["secs"] = time.time() - t0; json.dump(out, open(os.path.join(OUT, tag + ".json"), "w"))
    log(tag, "secs", round(out["secs"]), {k: round(float(np.mean(out["sys"][k]["ndcg"])), 4) for k in ["full", "h100", "h50", "jitaf"]}, flush_check(tag, out, qids))
def flush_check(tag, out, qids):
    """Control against the stored per-query results of the frozen systems."""
    try:
        if tag in DATASETS:
            P = load_json(tag, "perquery_e14_lag.json"); r = {}
            for mine, old in [("full", "afh_0.0"), ("h100", "h100_0.0"), ("h50", "h50_0.0"), ("sa100|f0.1|s13", "h100_0.1"), ("afh|f0.1|s13", "afh_0.1")]:
                a = np.array(out["sys"][mine]["ndcg"]); b = np.array([P[old]["ndcg"][q] for q in qids]); r[mine] = round(float(np.mean(np.abs(a - b) < 1e-5)), 4)
            return r
        P = json.load(open(os.path.join(RES, "e16", f"test_{tag}.json"))); a = np.array(out["sys"]["full"]["ndcg"]); b = np.array([P["af_f0.0"]["ndcg"][q] for q in qids]); return round(float(np.mean(np.abs(a - b) < 1e-5)), 4)
    except Exception as e: return repr(e)
if __name__ == "__main__":
    for t in (sys.argv[1:] or DATASETS + EXT): run(t)
