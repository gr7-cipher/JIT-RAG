"""E11: stronger JIT-CC. Adaptive candidate expansion (feedback from the fused ranking through the inverted index only),
and alternative score normalizations, at a fixed encoding budget. Rivals get the same normalizations and expansion.
Evaluation uses stored vectors; the encoding count of each configuration is what a live system would pay."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]; NORMS = ["mm", "max", "z", "rank"]
# name: (first-round K, [later rounds], feedback docs)
CFG = {"b100": (100, [], 0), "b200": (200, [], 0), "e70+30_f3": (70, [30], 3), "e70+30_f10": (70, [30], 10), "e50+50_f3": (50, [50], 3), "e50+50_f10": (50, [50], 10),
       "e40+30+30_f3": (40, [30, 30], 3), "e40+30+30_f10": (40, [30, 30], 10), "e100+100_f3": (100, [100], 3), "e100+100_f10": (100, [100], 10)}
HYB = {"h50": (50, [], 0), "h100": (100, [], 0), "h50+50_f3": (50, [50], 3)}
def mm(x):
    if len(x) == 0: return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
def norm(x, how):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0: return x
    if how == "mm": return mm(x)
    if how == "max": return x / x.max() if x.max() > 0 else np.zeros_like(x)
    if how == "z":
        sd = x.std(); return mm(np.tanh((x - x.mean()) / sd)) if sd > 0 else np.zeros_like(x)
    if how == "rank":
        r = np.empty(len(x)); r[np.argsort(-x, kind="stable")] = np.arange(len(x)); return mm(1.0 / (60 + r))
    raise ValueError(how)
def prf(bm, w0, docs, p, fb_terms=20, lam=0.5):
    rel = {}; ids = bm.tok.ids
    for d, pp in zip(docs, p):
        t = np.asarray(ids[int(d)])
        if len(t) == 0: continue
        u, c = np.unique(t, return_counts=True)
        for a, n in zip(u.tolist(), c.tolist()):
            rel[a] = rel.get(a, 0.0) + float(pp) * n / len(t)
    keep = sorted(rel.items(), key=lambda x: -x[1])[:fb_terms]; tot = sum(v for _, v in keep) or 1.0; qt = sum(w0.values()) or 1.0
    out = {t: lam * v / qt for t, v in w0.items()}
    for t, v in keep: out[t] = out.get(t, 0.0) + (1 - lam) * v / tot
    return out
def expand(bm, w0, S0, u, Ev, q_vec, rounds, f):
    """Each round: rank what is encoded so far, feed its top f back through BM25, encode the best unseen documents."""
    for add in rounds:
        if len(u) == 0: break
        sc = 0.8 * mm(Ev[u] @ q_vec) + 0.2 * mm(S0[u]); o = np.argsort(-sc)[:f]; p = sc[o] + 1e-9; p = p / p.sum()
        s2 = bm.score(prf(bm, w0, u[o], p)); s2[u] = 0; nz = int(np.count_nonzero(s2)); n = min(add, nz)
        if n == 0: break
        new = np.argpartition(-s2, n - 1)[:n]; u = np.concatenate([u, new])
    return u
for name in DATASETS:
    doc_ids, docs, qids, queries, qrels = load_dataset(name)
    d = os.path.join(RES, name); E = np.load(os.path.join(d, "bge_emb.npy")); Q = np.load(os.path.join(d, "bge_qemb.npy")); bm = BM25(docs)
    C = np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True); runs = {}; enc = {}
    for i, q in enumerate(qids):
        v = Q[i]; c0 = np.asarray(C[i], dtype=np.int64); w0 = {}
        for t in bm.q_terms(queries[i]): w0[t] = w0.get(t, 0.0) + 1.0
        S0 = bm.score(w0) if w0 else np.zeros(bm.n, dtype=np.float32); sim = None
        for nm, (k1, rd, f) in {**CFG, **HYB}.items():
            u = expand(bm, w0, S0, c0[:k1 + 1].copy(), E, v, rd, f)
            enc.setdefault(nm, []).append(len(u))
            if nm in HYB:
                if sim is None: sim = E @ v
                k = min(k1 + 1, len(sim)); top = np.argpartition(-sim, k - 1)[:k]; u = np.array(sorted(set(u.tolist()) | set(top.tolist())), dtype=np.int64)
            cs = E[u] @ v if len(u) else np.empty(0); bs = S0[u] if len(u) else np.empty(0)
            for nr in NORMS:
                nc, nb = norm(cs, nr), norm(bs, nr)
                for a in ALPHAS:
                    g = a * nc + (1 - a) * nb; o = np.argsort(-g, kind="stable")
                    runs.setdefault(f"{nm}|{nr}|{a}", {})[q] = run_from_ranked(doc_ids, q, u[o], g[o], depth=200)
    per = {k: (lambda e: {"ndcg": e["ndcg"], "recall100": e["recall100"]})(evaluate(v, qrels)) for k, v in runs.items()}
    per["__enc__"] = {k: float(np.mean(v)) for k, v in enc.items()}
    save_json(per, name, "perquery_e11.json")
    log(name, {k: round(mean(per[k]["ndcg"]), 4) for k in ["b100|mm|0.9", "b100|max|0.9", "b100|z|0.9", "b100|rank|0.9", "e50+50_f3|mm|0.9", "e70+30_f3|mm|0.9", "h100|mm|0.9"]}, {k: round(mean(per[k]["recall100"]), 3) for k in ["b100|mm|0.9", "e50+50_f3|mm|0.9", "e100+100_f3|mm|0.9", "h100|mm|0.9"]}, per["__enc__"]["e50+50_f3"])
