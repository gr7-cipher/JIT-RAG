"""E15: last check on query sets used by nothing earlier (FiQA and NFCorpus train), all settings frozen first."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["x", "noop"]
from e14_afh import *
nr, a, g = B14.split("|"); a, g = float(a), float(g); _, nf, af_, gf = F13["af"].split("|"); af_, gf = float(af_), float(gf); hp = {c: (F13[c].split("|")[1], float(F13[c].split("|")[2])) for c in ["h50", "h100"]}
m = encoder(BGE, 256); out = {}
for name in ["nfcorpus", "fiqa"]:
    doc_ids, docs, qids, queries, qrels, E = load(name, "train"); Q = encode(m, [PFX + q for q in queries], bs=16); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); runs = {}; n = len(E)
    def add(k, q, u, f):
        o = np.argsort(-f, kind="stable"); runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=1000 if k in ("rrf", "dense") else 300)
    for i, q in enumerate(qids):
        v = Q[i]; w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(n, np.float32); c0 = np.asarray(bm.topk(w0, 1000)[0] if w0 else np.empty(0, dtype=np.int64), dtype=np.int64); sim = E @ v
        u, cs, l, _ = af_run(bm, idf, w0, S0, c0, v, lambda ids, vv: E[ids] @ vv); add("af", q, u, af_score(u, cs, S0, l, nf, af_, gf))
        u, cs, l, _ = cands(bm, idf, w0, S0, c0, E, v, None, sim); add("afh", q, u, af_score(u, cs, S0, l, nr, a, g))
        for c, K in [("h50", 50), ("h100", 100)]: add(c, q, *rival(c0, S0, sim, None, K, *hp[c]))
        k = min(1001, n); top = np.argpartition(-sim, k - 1)[:k]; top = top[np.argsort(-sim[top])]; add("dense", q, top, sim[top]); scr = {}
        for L_ in (top[:1000], c0[:1000]):
            for rk, x in enumerate(L_.tolist()): scr[x] = scr.get(x, 0.0) + 1.0 / (61 + rk)
        ks = np.array(list(scr), dtype=np.int64); add("rrf", q, ks, np.array([scr[x] for x in ks.tolist()]))
    per = {k: (lambda e: {"ndcg": e["ndcg"], "recall100": e["recall100"]})(evaluate(v, qrels)) for k, v in runs.items()}; save_json(per, name, "perquery_e15.json")
    t = {o: paired_test(per["afh"]["ndcg"], per[o]["ndcg"]) for o in ["h100", "rrf", "dense", "af"]}; tr = {o: paired_test(per["afh"]["recall100"], per[o]["recall100"]) for o in ["h100", "rrf", "dense", "af"]}
    out[name] = {"n": len(qids), "sys": {k: (mean(per[k]["ndcg"]), mean(per[k]["recall100"])) for k in per}, "afh_vs": t, "afh_vs_recall": tr}; save_json(out, "e15.json")
    print(name, len(qids), {k: tuple(round(x, 4) for x in v) for k, v in out[name]["sys"].items()})
    print("  nDCG", {o: "%+.4f [%+.4f,%+.4f] p=%.3g" % (x["diff"], x["lo"], x["hi"], x["p"]) for o, x in t.items()}); print("  R100", {o: "%+.4f [%+.4f,%+.4f] p=%.3g" % (x["diff"], x["lo"], x["hi"], x["p"]) for o, x in tr.items()})
