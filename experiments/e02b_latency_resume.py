"""E2: real query-time latency. Nothing is cached: every candidate is tokenised and encoded per query."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch

N_Q = 30
def bench(bm, docs, queries, model, dev, K, max_seq=256, bs=64, quant=False, n_q=N_Q):
    m = encoder(model, max_seq, dev)
    if quant:
        torch.backends.quantized.engine = "qnnpack"
        m[0].auto_model = torch.quantization.quantize_dynamic(m[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8)
    encode(m, queries[:2]); encode(m, docs[:bs], bs=bs)  # warm-up
    rows = []
    for q in queries[:n_q]:
        t0 = time.perf_counter(); idx, _ = bm.search(q, K)
        t1 = time.perf_counter(); v = encode(m, [q], bs=1)
        t2 = time.perf_counter(); X = encode(m, [docs[int(i)] for i in idx], bs=bs)
        t3 = time.perf_counter(); o = np.argsort(-(X @ v[0]))[:10]
        t4 = time.perf_counter()
        rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))
    a = np.array(rows) * 1e3
    del m
    if dev == "mps":
        torch.mps.empty_cache()
    return {"first_ms": float(np.median(a[:, 0])), "qenc_ms": float(np.median(a[:, 1])), "denc_ms": float(np.median(a[:, 2])),
            "score_ms": float(np.median(a[:, 3])), "p50": float(np.percentile(a[:, 4], 50)), "p95": float(np.percentile(a[:, 4], 95)),
            "p99": float(np.percentile(a[:, 4], 99)), "mean": float(a[:, 4].mean()), "enc_ms_per_doc": float(np.median(a[:, 2])) / max(K, 1)}

# Resumes the first run, which was lost at the mpnet step: on this 8 GB machine, batch 64 with a 768-dim
# model drove the system into swap (266 s per query), so those timings are discarded and mpnet uses batch 16.
out = load_json("latency.json")
dev = device()
out["models"].pop("sentence-transformers/all-mpnet-base-v2", None)
doc_ids, docs, qids, queries, qrels = load_dataset("fiqa")
bm = BM25(docs)
out["models"]["sentence-transformers/all-mpnet-base-v2"] = {K: bench(bm, docs, queries, "sentence-transformers/all-mpnet-base-v2", dev, K, bs=16, n_q=8) for K in [50, 200]}
log("mpnet", out["models"]["sentence-transformers/all-mpnet-base-v2"][200]["p50"]); save_json(out, "latency.json")
out["max_seq"] = {s: bench(bm, docs, queries, BASE_MODEL, dev, 200, max_seq=s, n_q=20) for s in [32, 64, 128, 256]}
log("max_seq"); save_json(out, "latency.json")
out["batch"] = {b: bench(bm, docs, queries, BASE_MODEL, dev, 200, bs=b, n_q=10) for b in [8, 32, 64, 128]}
log("batch"); save_json(out, "latency.json")
out["cpu"] = {K: bench(bm, docs, queries, BASE_MODEL, "cpu", K, n_q=10) for K in [50, 200]}
log("cpu"); save_json(out, "latency.json")
try:
    out["cpu_q8"] = {K: bench(bm, docs, queries, BASE_MODEL, "cpu", K, quant=True, n_q=10) for K in [50, 200]}
except Exception as e:
    out["cpu_q8_error"] = str(e)
save_json(out, "latency.json"); del bm
doc_ids, docs, qids, queries, qrels = load_dataset("trec-covid")
bm = BM25(docs)
out["k_sweep"]["trec-covid"] = {}
for K in [10, 20, 50, 100, 200, 500]:
    out["k_sweep"]["trec-covid"][K] = bench(bm, docs, queries, BASE_MODEL, dev, K, n_q=15)
    log("trec", K, out["k_sweep"]["trec-covid"][K]["p50"]); save_json(out, "latency.json")
del bm
doc_ids, docs, qids, queries, qrels = load_dataset("quora")
out["scale"] = {}
for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
    n = int(len(docs) * frac)
    bm = BM25(docs[:n])
    ts = []
    for q in queries:
        t = time.perf_counter(); bm.search(q, 1000); ts.append(time.perf_counter() - t)
    out["scale"][n] = {"ms_p50": float(np.percentile(ts, 50) * 1e3), "ms_p95": float(np.percentile(ts, 95) * 1e3), "bytes": bm.nbytes()}
    del bm
save_json(out, "latency.json")
log("DONE")
