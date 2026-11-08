"""E13: JIT-AF, the final algorithm: adaptive candidates + fusion + encoding budget + write-back.
pick      freeze one global setting for JIT-AF and each rival from dev-half queries only
budget    quality against backlog share f and per-query encoding budget m; signal-aware rivals on the same backlogs
unseen    frozen settings on BEIR query sets never used before (dev/train splits)
warm      encodings per query as write-back fills the store, from an empty store
overload  discrete simulation of the fluid model f_inf = (u - mu) / (u + Q K)
latency   live end-to-end time of JIT-AF
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from e12_refine import mm, norm, prf, D, CFG
what = sys.argv[1]
BGE = "BAAI/bge-small-en-v1.5"; PFX = "Represent this sentence for searching relevant passages: "
AF = {**D, **CFG["idf_l.3"]}; INF = 10 ** 9; PICK = os.path.join(RES, "e13_pick.json")
def dev_ids(q):
    q = sorted(q); r = np.random.default_rng(SEED).permutation(len(q)); return [q[i] for i in r[:len(q) // 2]]
def weights(bm, text):
    w = {}
    for t in bm.q_terms(text): w[t] = w.get(t, 0.0) + 1.0
    return w
def af_run(bm, idf, w0, S0, c0, v, get, hidden=None, m=INF, c=AF):
    """The adaptive loop. get(ids) returns vectors for documents being encoded now or read from the store.
    A waiting document is encoded only while budget remains, best first. Returns ids, cos (nan if unknown), last BM25 scores, encoded count."""
    r = c["rounds"]; tot = sum(r); u = c0[:r[0]].astype(np.int64); cos = np.full(len(u), np.nan); used = 0; last = S0; allow = 0.0
    def fill(start, frac):
        nonlocal used, allow
        ids = u[start:]
        if len(ids) == 0: return
        wait = hidden[ids] if hidden is not None else np.zeros(len(ids), bool); allow += frac * m if m < INF else INF
        room = max(0, int(allow + 1e-9) - used) if m < INF else INF; ok = ~wait; w = np.where(wait)[0]
        ok[w[:room]] = True; used += min(len(w), room)
        if ok.any(): cos[start:][ok] = get(ids[ok], v)
    fill(0, r[0] / tot)
    for add in r[1:]:
        if len(u) == 0: break
        kn = ~np.isnan(cos); sc = 0.2 * mm(S0[u].astype(np.float64)); 
        if kn.any(): sc[kn] += 0.8 * mm(cos[kn])
        o = np.argsort(-sc)[:c["f"]]; p = sc[o] + 1e-9; p = p / p.sum()
        last = bm.score(prf(bm, idf, w0, u[o], p, c)); s2 = last.copy(); s2[u] = 0; n = min(add, int(np.count_nonzero(s2)))
        if n == 0: break
        new = np.argpartition(-s2, n - 1)[:n]; new = new[np.argsort(-s2[new], kind="stable")]; st = len(u)
        u = np.concatenate([u, new]); cos = np.concatenate([cos, np.full(n, np.nan)]); fill(st, add / tot)
    return u, cos, last, used
def af_score(u, cos, S0, last, nr, a, g, imp="bm"):
    kn = ~np.isnan(cos); nb, nl = norm(S0[u], nr), norm(last[u], nr); lex = (1 - g) * nb + g * nl; f = lex.copy()
    if kn.any():
        nc = norm(cos[kn], nr); f[kn] = a * nc + (1 - a) * lex[kn]
        if imp == "mean" and (~kn).any(): f[~kn] = a * nc.mean() + (1 - a) * lex[~kn]
    return f
def res(n): return os.path.join(RES, n)
def load(name, split="test"):
    doc_ids, docs, qids, queries, qrels = load_dataset(name, split=split)
    return doc_ids, docs, qids, queries, qrels, np.load(res(f"{name}/bge_emb.npy"))
store = lambda ids, v, E=None: None
if what == "pick":
    P12 = {d: load_json(d, "perquery_e12.json") for d in DATASETS}; P11 = {d: load_json(d, "perquery_e11.json") for d in DATASETS}
    dv = {d: dev_ids(P12[d]["base|mm|0.9|0.0"]["ndcg"]) for d in DATASETS}
    def best(P, pre):
        ks = [k for k in P[DATASETS[0]] if k.startswith(pre + "|")]
        return max(ks, key=lambda k: np.mean([np.mean([P[d][k]["ndcg"][q] for q in dv[d]]) for d in DATASETS]))
    o = {"af": best(P12, "idf_l.3")}
    for c in ["b100", "h50", "h100"]: o[c] = best(P11, c)
    print(o); json.dump(o, open(PICK, "w"))
else:
    F = json.load(open(PICK)) if os.path.exists(PICK) else None
if what == "budget":
    _, nr, a, g = F["af"].split("|"); a = float(a); g = float(g); _, hr, ha = F["h100"].split("|"); ha = float(ha)
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, E = load(name); Q = np.load(res(f"{name}/bge_qemb.npy")); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr)))
        C = np.load(res(f"{name}/bm25_cands.npy"), allow_pickle=True); get = lambda ids, v: E[ids] @ v; runs = {}; enc = {}; n = len(E)
        def add(k, q, u, f):
            o = np.argsort(-f, kind="stable"); runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=200)
        masks = {f: np.random.default_rng(SEED).random(n) < f for f in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]}
        for i, q in enumerate(qids):
            v = Q[i]; c0 = np.asarray(C[i], dtype=np.int64); w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(n, np.float32)
            u0, cs0, l0, _ = af_run(bm, idf, w0, S0, c0, v, get); add("af", q, u0, af_score(u0, cs0, S0, l0, nr, a, g)); sim = E @ v
            for f, h in masks.items():
                # unlimited budget: must reproduce the fully known ranking (Proposition 1)
                u, cs, l, e = af_run(bm, idf, w0, S0, c0, v, get, h, INF); assert len(u) == len(u0) and (u == u0).all() and not np.isnan(cs).any(), (name, q, f)
                enc.setdefault(f"inf_f{f}", []).append(e)
                for m in [2, 5, 10, 20, 50]:
                    u, cs, l, e = af_run(bm, idf, w0, S0, c0, v, get, h, m); enc.setdefault(f"m{m}_f{f}", []).append(e)
                    for imp in ["bm", "mean"]: add(f"m{m}_f{f}_{imp}", q, u, af_score(u, cs, S0, l, nr, a, g, imp))
                    if f == 1.0:  # same budget spent on a shallow list with no feedback
                        c = c0[:m]; 
                        if len(c): add(f"shallow_m{m}", q, c, a * norm(E[c] @ v, nr) + (1 - a) * norm(S0[c], nr))
                if f < 1.0:  # signal-aware rivals over the same backlog, with their dev-chosen normalization
                    s = sim.copy(); s[h] = -9; k = min(101, n); top = np.argpartition(-s, k - 1)[:k]; top = top[s[top] > -9]
                    us = np.array(sorted(set(c0[:101].tolist()) | set(top.tolist())), dtype=np.int64); vis = ~h[us]; nd = np.zeros(len(us)); nd[vis] = norm(sim[us][vis], hr); nb = norm(S0[us], hr)
                    add(f"hyb_f{f}", q, us, np.where(vis, ha * nd + (1 - ha) * nb, nb))
                    dn = np.argpartition(-s, min(1000, n - 1))[:min(1000, n - 1)]; dn = dn[np.argsort(-s[dn])]; dn = dn[s[dn] > -9]; scr = {}
                    for rk, x in enumerate(dn.tolist()): scr[x] = scr.get(x, 0.0) + 1.0 / (61 + rk)
                    for rk, x in enumerate(c0[:1000].tolist()): scr[x] = scr.get(x, 0.0) + (2.0 if h[x] else 1.0) / (61 + rk)
                    ks = np.array(list(scr), dtype=np.int64); add(f"rrf_f{f}", q, ks, np.array([scr[x] for x in ks.tolist()]))
        per = {k: evaluate(v, qrels)["ndcg"] for k, v in runs.items()}; per["__enc__"] = {k: float(np.mean(v)) for k, v in enc.items()}
        save_json(per, name, "perquery_e13_budget.json")
        log(name, {k: round(mean(per[k]), 3) for k in ["af", "m10_f0.1_bm", "m10_f0.1_mean", "hyb_f0.1", "rrf_f0.1", "m20_f1.0_bm", "m20_f1.0_mean", "shallow_m20"]}, per["__enc__"]["inf_f0.1"])
if what == "unseen":
    SETS = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev", "quora": "dev"}; m = encoder(BGE, 256); out = {}
    _, nr, a, g = F["af"].split("|"); a = float(a); g = float(g); pr = {c: (F[c].split("|")[1], float(F[c].split("|")[2])) for c in ["b100", "h50", "h100"]}
    for name, sp in SETS.items():
        doc_ids, docs, qids, queries, qrels, E = load(name, sp); Q = encode(m, [PFX + q for q in queries], bs=16); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); get = lambda ids, v: E[ids] @ v; runs = {}; n = len(E)
        def add(k, q, u, f):
            o = np.argsort(-f, kind="stable"); runs.setdefault(k, {})[q] = run_from_ranked(doc_ids, q, u[o], f[o], depth=1000 if k in ("rrf", "dense") else 300)
        for i, q in enumerate(qids):
            v = Q[i]; w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(n, np.float32); c0, _ = bm.topk(w0, 1000) if w0 else (np.empty(0, dtype=np.int64), None); c0 = np.asarray(c0, dtype=np.int64); sim = E @ v
            u, cs, l, _ = af_run(bm, idf, w0, S0, c0, v, get); add("af", q, u, af_score(u, cs, S0, l, nr, a, g))
            c = c0[:101]
            if len(c): add("b100", q, c, pr["b100"][1] * norm(sim[c], pr["b100"][0]) + (1 - pr["b100"][1]) * norm(S0[c], pr["b100"][0]))
            k = min(1001, n); top = np.argpartition(-sim, k - 1)[:k]; top = top[np.argsort(-sim[top])]; add("dense", q, top, sim[top]); scr = {}
            for L in (top[:1000], c0[:1000]):
                for rk, x in enumerate(L.tolist()): scr[x] = scr.get(x, 0.0) + 1.0 / (61 + rk)
            ks = np.array(list(scr), dtype=np.int64); add("rrf", q, ks, np.array([scr[x] for x in ks.tolist()]))
            for h, K in [("h50", 50), ("h100", 100)]:
                us = np.array(sorted(set(c0[:K + 1].tolist()) | set(top[:K + 1].tolist())), dtype=np.int64); hn, ha = pr[h]; add(h, q, us, ha * norm(sim[us], hn) + (1 - ha) * norm(S0[us], hn))
        per = {k: (lambda e: {"ndcg": e["ndcg"], "recall100": e["recall100"]})(evaluate(v, qrels)) for k, v in runs.items()}; save_json(per, name, "perquery_e13_unseen.json")
        out[name] = {"split": sp, "n": len(qids), **{k: (mean(per[k]["ndcg"]), mean(per[k]["recall100"])) for k in per}}; log(name, sp, len(qids), {k: tuple(round(x, 3) for x in out[name][k]) for k in per}); save_json(out, "e13_unseen.json")
if what == "warm":
    out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, E = load(name); Q = np.load(res(f"{name}/bge_qemb.npy")); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); C = np.load(res(f"{name}/bm25_cands.npy"), allow_pickle=True)
        have = np.zeros(len(E), bool); series = []; get = lambda ids, v: E[ids] @ v
        for i in np.random.default_rng(SEED).permutation(len(qids)):
            w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(bm.n, np.float32); c0 = np.asarray(C[i], dtype=np.int64); u, cs, l, e = af_run(bm, idf, w0, S0, c0, Q[i], get, ~have); have[u] = True; series.append(e)
        out[name] = {"series": series, "distinct": int(have.sum()), "N": len(E)}; log(name, "first10", np.mean(series[:10]), "last10", np.mean(series[-10:]), "distinct", int(have.sum()), "sum", sum(series), "share", have.mean()); save_json(out, "e13_warm.json")
if what == "overload":
    # Replay arrivals on FiQA: half the corpus preloaded, the rest arriving at u/s; encoder drains FIFO at mu/s; each query encodes its waiting candidates and writes them back.
    name = "fiqa"; doc_ids, docs, qids, queries, qrels, E = load(name); C = np.load(res(f"{name}/bm25_cands.npy"), allow_pickle=True); n = len(E); out = {}
    for u_, mu, Qr, m in [(200, 100, 10, INF), (200, 100, 5, INF), (200, 100, 2, INF), (400, 100, 10, INF), (200, 150, 10, INF), (200, 100, 10, 5), (200, 100, 10, 15), (100, 150, 10, INF)]:
        rng = np.random.default_rng(SEED); perm = rng.permutation(n); N0 = n // 2; arrived = np.zeros(n, bool); arrived[perm[:N0]] = True; enc = arrived.copy(); nxt = N0; head = N0; dt = 0.1; hist = []; budget = 0.0; carry = 0.0; qenc = []; qc = 0.0; K = 100; t = 0.0
        while nxt < n:
            for x in perm[nxt:int(min(n, nxt + u_ * dt))]: arrived[x] = True
            nxt = int(min(n, nxt + u_ * dt)); carry += mu * dt; k = int(carry); carry -= k
            while k > 0 and head < nxt:
                if not enc[perm[head]]: enc[perm[head]] = True; k -= 1
                head += 1
            qc += Qr * dt
            while qc >= 1:
                qc -= 1; c = np.asarray(C[int(rng.integers(len(qids)))], dtype=np.int64); c = c[arrived[c]][:K]; w = c[~enc[c]]; w = w[:m] if m < INF else w; enc[w] = True; qenc.append(len(w))
            t += dt; hist.append((t, float((arrived & ~enc).sum() / arrived.sum())))
        h = np.array(hist); tail = h[len(h) // 2:, 1]; net = u_ - mu; f_inf = net / (u_ + Qr * K); bind = m < INF and m < K * f_inf; pred = 0.0 if net <= 0 else ((net - Qr * m) * t / (N0 + u_ * t) if bind else f_inf)  # corrected: the budget binds iff m < K f_inf
        key = f"u{u_}_mu{mu}_Q{Qr}_m{m if m < INF else 'inf'}"; out[key] = {"pred": min(1.0, pred), "budget_binds": bool(bind), "steady_mean": float(tail.mean()), "last": float(h[-1, 1]), "enc_q": float(np.mean(qenc[len(qenc) // 2:])) if qenc else 0.0, "T": float(t)}; log(key, out[key]); save_json(out, "e13_overload.json")
if what == "latency":
    m_ = encoder(BGE, 256); _, nr, a, g = F["af"].split("|"); a = float(a); g = float(g); out = {}
    for name in DATASETS:
        doc_ids, docs, qids, queries, qrels, E = load(name); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); qs = queries[:100]; r = {}
        for q in qs[:5]: encode(m_, [PFX + q], bs=1)
        for f, m in [(0.0, INF), (0.01, INF), (0.1, INF), (1.0, INF), (0.1, 5), (0.1, 10), (1.0, 10), (1.0, 20), (1.0, 50)]:
            h = np.random.default_rng(SEED).random(len(E)) < f; ts, en = [], []
            for q in qs:
                t0 = time.perf_counter(); w0 = weights(bm, q); S0 = bm.score(w0) if w0 else np.zeros(bm.n, np.float32); c0, _ = bm.topk(w0, 40) if w0 else (np.empty(0, dtype=np.int64), None); v = encode(m_, [PFX + q], bs=1)[0]
                def get(ids, v):
                    w = h[ids]; X = np.array(E[ids], copy=True)
                    if w.any(): X[w] = encode(m_, [docs[int(i)] for i in ids[w]])
                    return X @ v
                u, cs, l, e = af_run(bm, idf, w0, S0, np.asarray(c0, dtype=np.int64), v, get, h, m); sc = af_score(u, cs, S0, l, nr, a, g); np.argsort(-sc)[:10]; ts.append(time.perf_counter() - t0); en.append(e)
            r[f"f{f}_m{m if m < INF else 'inf'}"] = {"p50": float(np.median(ts) * 1e3), "p95": float(np.percentile(ts, 95) * 1e3), "enc": float(np.mean(en))}
        out[name] = r; log(name, {k: (round(v["p50"]), round(v["p95"]), round(v["enc"], 1)) for k, v in r.items()}); save_json(out, "e13_latency.json")
