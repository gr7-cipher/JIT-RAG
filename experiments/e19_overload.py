"""E19: overload simulation on FiQA, repeating the earlier replay with queries drawn without replacement from the 6,648 FiQA queries
(test and train), to separate candidate overlap from query repetition. Adds the corrected budget test and a shared-capacity variant,
in which query-time encodings and the background encoder draw on one capacity C and queries have priority."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a = sys.argv[1:]; sys.argv = ["x", "noop"]
from e14_afh import *
INF = 10 ** 9; name = "fiqa"
doc_ids, docs, qt, qqt, _ = load_dataset(name, max_q=0, split="test"); _, _, qt2, qq2, _ = load_dataset(name, max_q=0, split="train"); bm = BM25(docs); n = bm.n
# deduplicate by query id across splits
seen = {}; 
for a, b in zip(qt + qt2, qqt + qq2): seen.setdefault(a, b)
Q_ = list(seen.values()); log("queries", len(Q_)); C = [np.asarray(bm.topk(weights(bm, t), 2000)[0], dtype=np.int64) for t in Q_]
C500 = np.load(res(f"{name}/bm25_cands.npy"), allow_pickle=True)
def sim(u_, mu, Qr, m, mode, shared=None, K=100):
    rng = np.random.default_rng(SEED); perm = rng.permutation(n); N0 = n // 2; arrived = np.zeros(n, bool); arrived[perm[:N0]] = True; enc = arrived.copy(); nxt = N0; head = N0; dt = 0.1; carry = 0.0; qc = 0.0; t = 0.0; qenc = []; h = []; qorder = rng.permutation(len(C)); qi = 0; distinct = set(); short = 0.0
    while nxt < n:
        for x in perm[nxt:int(min(n, nxt + u_ * dt))]: arrived[x] = True
        nxt = int(min(n, nxt + u_ * dt)); qc += Qr * dt; qe = 0
        while qc >= 1:
            qc -= 1
            if mode == "repl": j = int(rng.integers(len(C500))); c = np.asarray(C500[j], dtype=np.int64)
            else: j = int(qorder[qi % len(qorder)]); qi += 1; c = C[j]
            distinct.add(j); c = c[arrived[c]][:K]; w = c[~enc[c]]; w = w[:m] if m < INF else w; enc[w] = True; qenc.append(len(w)); qe += len(w)
        rate = mu if shared is None else max(0.0, shared - qe / dt); short += 0 if shared is None else max(0.0, qe / dt - shared) * dt
        carry += rate * dt; k = int(carry); carry -= k
        while k > 0 and head < nxt:
            if not enc[perm[head]]: enc[perm[head]] = True; k -= 1
            head += 1
        t += dt; h.append(float((arrived & ~enc).sum() / arrived.sum()))
    tail = h[len(h) // 2:]; return {"T": t, "f_end": h[-1], "f_tail": float(np.mean(tail)), "enc_q": float(np.mean(qenc[len(qenc) // 2:])) if qenc else 0.0, "distinct": len(distinct), "draws": len(qenc), "Keff": float(np.mean(qenc[len(qenc) // 2:]) / np.mean(tail)) if np.mean(tail) > 0 else None, "overdemand_docs": short, "series": [round(x, 4) for x in h[::100]]}
out = {}
for u_, mu, Qr, m in [(200, 100, 10, INF), (200, 100, 5, INF), (200, 100, 2, INF), (400, 100, 10, INF), (200, 150, 10, INF), (200, 100, 10, 5), (200, 100, 10, 8), (200, 100, 10, 9), (200, 100, 10, 15)]:
    N0 = n // 2; fi = (u_ - mu) / (u_ + Qr * 100); binds = m < 100 * fi
    for mode in ["repl", "norepl"]:
        k = f"u{u_}_mu{mu}_Q{Qr}_m{m if m < INF else 'inf'}|{mode}"; r = sim(u_, mu, Qr, m, mode); N = N0 + u_ * r["T"]; cf = lambda K: (u_ - mu) / (u_ + Qr * K) * (1 - (N0 / N) ** ((u_ + Qr * K) / u_)); r.update(u=u_, mu=mu, Q=Qr, m=None if m == INF else m, mode=mode, f_inf=fi, model_K100=cf(100), model_Keff=cf(r["Keff"]) if r["Keff"] else None, binds_corrected=bool(binds), binds_v2=bool(m < INF and (u_ - mu) > Qr * m), limit_binding=(u_ - mu - Qr * m) / u_ if m < INF else None); out[k] = r; log(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items() if a != "series"})
for u_, C_, Qr, m in [(100, 60, 2, INF), (100, 60, 5, INF), (100, 60, 10, INF), (100, 60, 10, 10), (200, 60, 10, INF), (40, 60, 5, INF)]:
    k = f"shared_u{u_}_C{C_}_Q{Qr}_m{m if m < INF else 'inf'}"; r = sim(u_, 0, Qr, m, "norepl", shared=C_); r.update(u=u_, C=C_, Q=Qr, m=None if m == INF else m, limit=max(0.0, (u_ - C_) / u_)); out[k] = r; log(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items() if a != "series"})
save_json(out, "e19_overload.json")
