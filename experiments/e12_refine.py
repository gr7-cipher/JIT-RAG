"""E12: refinement of the adaptive loop at a budget of 100 encodings. Coordinate scan around the E11 selection.
Config: rounds, feedback docs, terms, lambda, idf weighting of feedback terms, feedback per document or pooled.
Also a third signal: BM25 under the final expanded query (gamma)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ALPHAS = [0.7, 0.8, 0.9, 0.95]; NORMS = ["mm", "max"]; GAMMAS = [0.0, 0.15, 0.3]
D = dict(rounds=(40, 30, 30), f=3, terms=20, lam=0.5, idf=False, per=False)
CFG = {"base": {}, "r25x4": dict(rounds=(25, 25, 25, 25)), "r20x5": dict(rounds=(20, 20, 20, 20, 20)), "r10x10": dict(rounds=(10,) * 10), "r60+20+20": dict(rounds=(60, 20, 20)), "r30+35+35": dict(rounds=(30, 35, 35)),
       "f2": dict(f=2), "f5": dict(f=5), "t10": dict(terms=10), "t40": dict(terms=40), "l.3": dict(lam=.3), "l.7": dict(lam=.7), "idf": dict(idf=True), "per": dict(per=True), "idf_t40": dict(idf=True, terms=40), "idf_l.3": dict(idf=True, lam=.3), "idf_r25x4": dict(idf=True, rounds=(25,) * 4)}
def mm(x):
    if len(x) == 0: return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
def norm(x, how):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0: return x
    if how == "mm": return mm(x)
    return x / x.max() if x.max() > 0 else np.zeros_like(x)
def prf(bm, idf, w0, docs, p, c):
    rel = {}; ids = bm.tok.ids
    for d, pp in zip(docs, p):
        t = np.asarray(ids[int(d)])
        if len(t) == 0: continue
        u, n = np.unique(t, return_counts=True)
        for a, b in zip(u.tolist(), n.tolist()):
            rel[a] = rel.get(a, 0.0) + float(pp) * b / len(t) * (idf[a] if c["idf"] else 1.0)
    keep = sorted(rel.items(), key=lambda x: -x[1])[:c["terms"]]; tot = sum(v for _, v in keep) or 1.0; qt = sum(w0.values()) or 1.0
    out = {t: c["lam"] * v / qt for t, v in w0.items()}
    for t, v in keep: out[t] = out.get(t, 0.0) + (1 - c["lam"]) * v / tot
    return out
def adaptive(bm, idf, w0, S0, c0, Ev, v, c):
    """Returns encoded candidates and the BM25 scores under the last expanded query."""
    r = c["rounds"]; u = c0[:r[0] + (1 if len(c0) > r[0] else 0)].copy(); last = S0
    for add in r[1:]:
        if len(u) == 0: break
        sc = 0.8 * mm(Ev[u] @ v) + 0.2 * mm(S0[u]); o = np.argsort(-sc)[:c["f"]]; p = sc[o] + 1e-9; p = p / p.sum()
        if c["per"]:
            new = []; seen = set(u.tolist())
            for dd in u[o]:
                s2 = bm.score(prf(bm, idf, w0, [dd], [1.0], c)); s2[list(seen)] = 0; n = min(add // c["f"] + 1, int(np.count_nonzero(s2)))
                if n: 
                    x = np.argpartition(-s2, n - 1)[:n]; new += x.tolist(); seen |= set(x.tolist())
            new = np.array(new[:add], dtype=np.int64); last = S0
        else:
            w = prf(bm, idf, w0, u[o], p, c); last = bm.score(w); s2 = last.copy(); s2[u] = 0; n = min(add, int(np.count_nonzero(s2)))
            new = np.argpartition(-s2, n - 1)[:n] if n else np.empty(0, dtype=np.int64)
        if len(new) == 0: break
        u = np.concatenate([u, new])
    return u, last
if __name__ == "__main__":
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels = load_dataset(name)
        d = os.path.join(RES, name); E = np.load(os.path.join(d, "bge_emb.npy")); Q = np.load(os.path.join(d, "bge_qemb.npy")); bm = BM25(docs)
        idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr)))
        C = np.load(os.path.join(d, "bm25_cands.npy"), allow_pickle=True); runs = {}; enc = {}
        for i, q in enumerate(qids):
            v = Q[i]; c0 = np.asarray(C[i], dtype=np.int64); w0 = {}
            for t in bm.q_terms(queries[i]): w0[t] = w0.get(t, 0.0) + 1.0
            S0 = bm.score(w0) if w0 else np.zeros(bm.n, dtype=np.float32)
            for nm, ov in CFG.items():
                c = {**D, **ov}; u, last = adaptive(bm, idf, w0, S0, c0, E, v, c); enc.setdefault(nm, []).append(len(u))
                cs = E[u] @ v if len(u) else np.empty(0); bs = S0[u] if len(u) else np.empty(0); ls = last[u] if len(u) else np.empty(0)
                for nr in NORMS:
                    nc, nb, nl = norm(cs, nr), norm(bs, nr), norm(ls, nr)
                    for a in ALPHAS:
                        for g in GAMMAS:
                            f = a * nc + (1 - a) * ((1 - g) * nb + g * nl); o = np.argsort(-f, kind="stable")
                            runs.setdefault(f"{nm}|{nr}|{a}|{g}", {})[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=200)
        per = {k: (lambda e: {"ndcg": e["ndcg"], "recall100": e["recall100"]})(evaluate(v, qrels)) for k, v in runs.items()}
        per["__enc__"] = {k: float(np.mean(v)) for k, v in enc.items()}
        save_json(per, name, "perquery_e12.json"); log(name, {k: round(mean(per[f"{k}|max|0.9|0.0"]["ndcg"]), 4) for k in CFG})
