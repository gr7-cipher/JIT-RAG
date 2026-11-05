"""E18: live load test on one shared accelerator. A background thread encodes arriving documents while query workers serve an
open-loop Poisson query stream and, depending on the system, encode waiting candidates at query time and write them back.
Every model call holds one lock, so the encoder capacity is shared. FiQA, half the corpus preloaded, bge-small.
Systems: afh (JIT-AF+ANN), lazyK (lazy Hybrid CC), lazyKmM (with budget M), sa100 (signal-aware Hybrid CC, no document encoding at query time).
Usage: e18_live.py grid [repeat]   or   e18_live.py one sys u Q policy backlog0 repeat"""
import os, sys, json, time, threading, collections, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a = sys.argv[1:]; sys.argv = ["x", "noop"]
from e14_afh import *
import hnswlib
NAME = "fiqa"; DUR = float(os.environ.get("E18_DUR", 120)); WARM = 10.0; WORKERS = 4; OUTF = res("e18_live.json")
doc_ids, docs, qt, qqt, qr = load_dataset(NAME, max_q=0, split="test"); _, _, qt2, qq2, qr2 = load_dataset(NAME, max_q=0, split="train")
POOL = list(zip(qt + qt2, qqt + qq2)); QR = {**qr, **qr2}; E = np.asarray(np.load(res(f"{NAME}/bge_emb.npy"))); n = len(E); BM = BM25(docs); idf = np.log(1 + BM.n / (1 + np.diff(BM.indptr))); MODEL = encoder(BGE, 256); POSD = {d: i for i, d in enumerate(doc_ids)}; AF_A, AF_G = 0.8, 0.3; H_A = 0.9
class View:
    """The inverted index restricted to arrived documents. IDF stays that of the full corpus."""
    def __init__(s, arrived): s.a = arrived; s.tok = BM.tok; s.n = BM.n
    def score(s, w): return BM.score(w, s.a)
def calibrate():
    t = time.perf_counter(); m = 512; encode(MODEL, docs[:m], bs=8); c8 = (time.perf_counter() - t) / m; t = time.perf_counter(); encode(MODEL, docs[m:2 * m], bs=32); c32 = (time.perf_counter() - t) / m
    ts = []
    for q, tx in POOL[:30]: t = time.perf_counter(); encode(MODEL, [PFX + tx], bs=1); ts.append(time.perf_counter() - t)
    return {"c8": c8, "c32": c32, "tq": float(np.median(ts))}
def run(sys_, u, Q, policy, backlog0, repeat):
    seed = SEED + repeat; rng = np.random.default_rng(seed); perm = rng.permutation(n); N0 = n // 2; arrived = np.zeros(n, np.float32); arrived[perm[:N0]] = 1.0; have = np.zeros(n, bool); store = np.zeros_like(E); nwait0 = int(backlog0 * N0)
    pre = perm[nwait0:N0]; have[pre] = True; store[pre] = E[pre]; H = hnswlib.Index(space="ip", dim=384); H.init_index(max_elements=n + 1000, ef_construction=200, M=16); H.set_num_threads(4); H.add_items(E[pre], pre); H.set_num_threads(1); H.set_ef(100)
    waiting = collections.deque(perm[:nwait0].tolist()); gpu = threading.Lock(); annl = threading.Lock(); stl = threading.Lock(); qwait = [0]; enc_count = np.zeros(n, np.int32); stop = threading.Event(); bg_batch = 8 if policy == "priority" else 32; counts = {"bg": 0, "q": 0}; view = View(arrived); cand = K_of(sys_); budget = m_of(sys_)
    def enc_docs(ids, kind):
        """Encode documents not yet stored and write them back."""
        ids = np.asarray(ids, np.int64)
        if kind == "q": qwait[0] += 1
        with gpu:
            if kind == "q": qwait[0] -= 1
            ids = ids[~have[ids]]
            if len(ids) == 0: return 0
            X = encode(MODEL, [docs[int(i)] for i in ids], bs=len(ids) if kind == "q" else bg_batch)
        with stl:
            new = ~have[ids]; enc_count[ids] += 1; store[ids] = X; have[ids] = True
        with annl:
            if new.any(): H.add_items(X[new], ids[new])
        counts[kind] += len(ids); return len(ids)
    def ingest():
        t0 = time.perf_counter(); nxt = N0; acc = 0.0; last = t0; r = np.random.default_rng(seed + 1)
        while not stop.is_set() and nxt < n:
            time.sleep(0.01); now = time.perf_counter(); k = int(r.poisson(u * (now - last))); last = now
            for x in perm[nxt:min(n, nxt + k)].tolist(): arrived[x] = 1.0; waiting.append(x)
            nxt = min(n, nxt + k)
    def background():
        while not stop.is_set():
            if policy == "priority" and qwait[0] > 0: time.sleep(0.001); continue
            b = []
            while waiting and len(b) < bg_batch:
                x = waiting.popleft()
                if not have[x]: b.append(x)
            if not b: time.sleep(0.002); continue
            enc_docs(b, "bg")
    def q_vec(tx):
        qwait[0] += 1
        with gpu: qwait[0] -= 1; return encode(MODEL, [PFX + tx], bs=1)[0]
    def ann(v, k):
        with annl: lab, _ = H.knn_query(v, k=k)
        return lab[0].astype(np.int64)
    def serve(tx):
        v = q_vec(tx); w0 = weights(BM, tx); S0 = BM.score(w0, arrived) if w0 else np.zeros(n, np.float32); e = 0
        if sys_ == "afh":
            c0, _ = BM.topk(w0, 40, arrived) if w0 else (np.empty(0, dtype=np.int64), None)
            def get(ids, vv):
                enc_docs(ids[~have[ids]], "q") if (~have[ids]).any() else None
                return store[ids] @ vv
            hid = ~have; u_, cs, l, e = af_run(view, idf, w0, S0, np.asarray(c0, dtype=np.int64), v, get, hid); ex = np.array(sorted(set(ann(v, 50).tolist()) - set(u_.tolist())), dtype=np.int64); us = np.concatenate([u_, ex]); ca = np.concatenate([cs, store[ex] @ v]); f = af_score(us, ca, S0, l, "max", AF_A, AF_G)
        else:
            c0 = np.asarray(BM.topk(w0, cand + 1, arrived)[0] if w0 else np.empty(0, dtype=np.int64), dtype=np.int64)
            if sys_.startswith("lazy"):
                w = c0[~have[c0]]; w = w if budget is None else w[:budget]
                if len(w): e = enc_docs(w, "q")
            us = np.array(sorted(set(c0.tolist()) | set(ann(v, cand + 1).tolist())), dtype=np.int64); kn = have[us]; nb = norm(S0[us], "max"); nd = np.zeros(len(us)); nd[kn] = norm(store[us][kn] @ v, "max"); f = np.where(kn, H_A * nd + (1 - H_A) * nb, nb)
        o = np.argsort(-f, kind="stable")[:12]; return us[o], e
    # open-loop schedule
    r = random.Random(seed); pool = POOL[:]; r.shuffle(pool); gaps = np.random.default_rng(seed + 2).exponential(1.0 / Q, size=int(Q * DUR * 1.5) + 10); at = np.cumsum(gaps); at = at[at < DUR]; pending = collections.deque((float(a), pool[i]) for i, a in enumerate(at)); done = []; dl = threading.Lock(); samples = []
    th = [threading.Thread(target=ingest, daemon=True), threading.Thread(target=background, daemon=True)]
    t0 = time.perf_counter()
    def worker():
        while True:
            with dl:
                if not pending: return
                a, (qid, tx) = pending[0]
                if time.perf_counter() - t0 < a: wait = True
                else: pending.popleft(); wait = False
            if wait: time.sleep(0.0005); continue
            top, e = serve(tx); fin = time.perf_counter() - t0
            with dl: done.append((a, fin - a, e, qid, top.tolist()))
    for x in th: x.start()
    ws = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for x in ws: x.start()
    def sampler():
        while not stop.is_set():
            time.sleep(1.0); ar = arrived.astype(bool); samples.append((time.perf_counter() - t0, float((ar & ~have).sum() / ar.sum()), int((ar & ~have).sum()), counts["bg"], counts["q"]))
    sm = threading.Thread(target=sampler, daemon=True); sm.start()
    time.sleep(DUR); end = time.perf_counter() - t0; ar = arrived.astype(bool); fend = float((ar & ~have).sum() / ar.sum()); bg_at, q_at = counts["bg"], counts["q"]; backlog_q = len(pending)
    grace = time.perf_counter() + 20.0
    while pending and time.perf_counter() < grace: time.sleep(0.05)
    with dl: left = list(pending); pending.clear()
    for x in ws: x.join(timeout=20)
    stop.set(); now = time.perf_counter() - t0
    for x in th: x.join(timeout=5)
    lat = np.array([d[1] for d in done if d[0] >= WARM]); cens = np.array([now - a for a, _ in left if a >= WARM]); allv = np.concatenate([lat, cens]) if len(cens) else lat
    nd, rr = [], []
    for a, _, _, qid, top in done:
        rel = QR.get(qid, {}); tp = [doc_ids[i] for i in top if doc_ids[i] != qid][:10]; ide = sorted((x for x in rel.values() if x > 0), reverse=True)[:10]; idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ide))
        nd.append(sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(tp) if rel.get(d, 0) > 0) / idcg if idcg else 0.0)
    pc = lambda p: float(np.percentile(allv, p) * 1e3) if len(allv) else None; ten = [s for s in samples if s[0] > DUR * 2 / 3]
    out = {"sys": sys_, "u": u, "Q": Q, "policy": policy, "backlog0": backlog0, "repeat": repeat, "dur": DUR, "n_done": len(lat), "n_cens": int(len(cens)), "backlog_at_end": backlog_q, "p50": pc(50), "p95": pc(95), "p99": pc(99), "mean": float(allv.mean() * 1e3) if len(allv) else None, "f_end": fend, "f_tail": float(np.mean([s[1] for s in ten])) if ten else None, "bg_rate": bg_at / DUR, "q_enc_rate": q_at / DUR, "enc_per_q": float(np.mean([d[2] for d in done])) if done else 0.0, "ndcg": float(np.mean(nd)) if nd else None, "dup": int((enc_count > 1).sum()), "enc_max": int(enc_count.max()), "series": [(round(s[0], 1), round(s[1], 4)) for s in samples[::5]], "lat_ms": np.round(allv * 1e3, 1).tolist()}
    return out
def K_of(s):
    if s == "afh": return 100
    if s == "sa100": return 100
    return int(s[4:].split("m")[0])
def m_of(s):
    return int(s.split("m")[1]) if s.startswith("lazy") and "m" in s[4:] else None
def save(r):
    o = json.load(open(OUTF)) if os.path.exists(OUTF) else {"calib": [], "runs": []}
    if "calib" in r: o["calib"].append(r)
    else: o["runs"].append(r)
    json.dump(o, open(OUTF, "w"))
def grid(rep):
    S = []
    for u, b0 in [(20, 0.0), (100, 0.1)]:
        for Q in [1, 2, 5, 10]:
            for s in ["sa100", "lazy50", "afh"]: S.append((s, u, Q, "priority", b0))
    for Q in [1, 2, 5, 10]: S.append(("lazy50m10", 100, Q, "priority", 0.1))
    for s in ["sa100", "lazy50", "afh"]: S.append((s, 100, 5, "fifo", 0.1)); S.append((s, 200, 5, "priority", 0.1))
    return S
if __name__ == "__main__":
    save({"calib": calibrate(), "t": time.time()}); log("calib", json.load(open(OUTF))["calib"][-1])
    if _a[0] == "one":
        s, u, Q, p, b0, rp = _a[1], float(_a[2]), float(_a[3]), _a[4], float(_a[5]), int(_a[6]); r = run(s, u, Q, p, b0, rp); save(r); log({k: v for k, v in r.items() if k not in ("lat_ms", "series")})
    else:
        rep = int(_a[1]) if len(_a) > 1 else 0; S = grid(rep); random.Random(rep).shuffle(S)
        for s, u, Q, p, b0 in S:
            DUR = max(60.0, 200.0 / Q); r = run(s, u, Q, p, b0, rep); save(r); log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("lat_ms", "series")})
