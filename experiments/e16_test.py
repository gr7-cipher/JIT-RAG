"""E16 step 3: frozen agreement-weighted rule on data never used for design.
New collections (Touche-2020, three CQADupStack forums) and fresh FiQA / NFCorpus training queries (excluding every query sampled earlier).
Systems:
  af      frozen JIT-AF+ANN (max, alpha .8, gamma .3)
  afr     JIT-AF+ANN with the agreement rule (the candidate for adoption)
  af9     JIT-AF+ANN with fixed alpha .9, gamma .3 (the simple alternative)
  h50,h100  frozen hybrids (dev-chosen max, alpha .9)
  h100r   hybrid with its own dev-fitted agreement rule
  h100p   hybrid with the same lexical-feedback signal and the same kind of rule (the upgrade made available to rivals)
  rrf, dense
Each system is also run at backlog f = .1 and .5 (except rrf, dense) with signal-aware handling."""
import os, sys, json, random
sys.argv = ["x", "noop"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e14_afh import *
from scipy.stats import rankdata
P = json.load(open(res("e16_pick.json"))); NR, A0, G0 = "max", 0.8, 0.3
def rho(c, s):
    if len(c) < 3 or np.ptp(s) == 0 or np.ptp(c) == 0: return 0.0
    return float(np.corrcoef(rankdata(c), rankdata(s))[0, 1])
def alpha(r, cfg): return min(1.0, max(0.7, cfg["a0"] + cfg["b"] * (r - cfg["mu"]) / cfg["sd"]))
def fuse(u, cs, s0, l, cfg=None, a=None, g=0.0):
    """Max-normalized fusion. cs may hold nan for documents with no dense score; they keep their lexical part."""
    kn = ~np.isnan(cs); lex = (1 - g) * norm(s0, NR) + g * norm(l, NR); f = lex.copy()
    if kn.any():
        aa = alpha(rho(cs[kn], s0[kn]), cfg) if cfg else a
        f[kn] = aa * norm(cs[kn], NR) + (1 - aa) * lex[kn]
    return f
def fresh(name, n_q=1000):
    """Training queries never sampled by any earlier experiment."""
    doc_ids, docs, qids, queries, qrels = load_dataset(name, max_q=0, split="train")
    old = set(load_dataset(name, split="train")[2]); keep = [i for i, q in enumerate(qids) if q not in old]
    random.Random(SEED + 1).shuffle(keep); keep = sorted(keep[:n_q])
    return doc_ids, docs, [qids[i] for i in keep], [queries[i] for i in keep], {qids[i]: qrels[qids[i]] for i in keep}
SETS = [("webis-touche2020", None), ("cqa-webmasters", None), ("cqa-android", None), ("cqa-wordpress", None), ("fiqa", "fresh"), ("nfcorpus", "fresh")]
if os.environ.get("JIT_E16_SETS"): SETS = [(x, None) for x in os.environ["JIT_E16_SETS"].split(",")]  # the A100 run: nq,msmarco
m = encoder(BGE, 256); out = {}
for name, how in SETS:
    tag = name + ("_fresh" if how else "")
    doc_ids, docs, qids, queries, qrels = fresh(name) if how else load_dataset(name, max_q=500)
    E = np.load(res(f"{name}/bge_emb.npy"), mmap_mode="r"); E = np.asarray(E); Q = encode(m, [PFX + q for q in queries], bs=16); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); n = len(E); runs = {}
    masks = {f: np.random.default_rng(SEED).random(n) < f for f in [0.1, 0.5]}; masks[0.0] = None
    def add(k, q, u, f, d=300):
        o = np.argsort(-f, kind="stable"); runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=d)
    for i, q in enumerate(qids):
        v = Q[i]; w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(n, np.float32); c0 = np.asarray(bm.topk(w0, 1000)[0] if w0 else np.empty(0, dtype=np.int64), dtype=np.int64); sim = E @ v
        for f, h in masks.items():
            u, cs, l, _ = cands(bm, idf, w0, S0, c0, E, v, h, sim); t = f"_f{f}"
            add("af" + t, q, u, fuse(u, cs, S0[u], l[u], a=A0, g=G0)); add("afr" + t, q, u, fuse(u, cs, S0[u], l[u], P["af"], g=G0)); add("af9" + t, q, u, fuse(u, cs, S0[u], l[u], a=0.9, g=G0))
            for K, nm in [(50, "h50"), (100, "h100")]:
                s = sim if h is None else np.where(h, -9.0, sim); k = min(K + 1, n); top = np.argpartition(-s, k - 1)[:k]; top = top[s[top] > -9]
                us = np.array(sorted(set(c0[:K + 1].tolist()) | set(top.tolist())), dtype=np.int64); cu = sim[us].astype(np.float64)
                if h is not None: cu[h[us]] = np.nan
                add(nm + t, q, us, fuse(us, cu, S0[us], S0[us], a=0.9, g=0.0))
                if K == 100:
                    add("h100r" + t, q, us, fuse(us, cu, S0[us], S0[us], P["h100"], g=0.0)); add("h100p" + t, q, us, fuse(us, cu, S0[us], l[us], P["h100p"], g=0.3))
        k = min(1001, n); top = np.argpartition(-sim, k - 1)[:k]; top = top[np.argsort(-sim[top])]; add("dense", q, top, sim[top], 1000); scr = {}
        for L in (top[:1000], c0[:1000]):
            for rk, x in enumerate(L.tolist()): scr[x] = scr.get(x, 0.0) + 1.0 / (61 + rk)
        ks = np.array(list(scr), dtype=np.int64); add("rrf", q, ks, np.array([scr[x] for x in ks.tolist()]), 1000)
    per = {k: (lambda e: {"ndcg": e["ndcg"], "recall100": e["recall100"]})(evaluate(r, qrels)) for k, r in runs.items()}
    save_json(per, "e16", f"test_{tag}.json"); out[tag] = {"n": len(qids), "N": n, **{k: (round(mean(per[k]["ndcg"]), 4), round(mean(per[k]["recall100"]), 4)) for k in per}}
    log(tag, out[tag]); save_json(out, "e16_test.json")
