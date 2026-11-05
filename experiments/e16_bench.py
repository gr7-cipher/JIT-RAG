"""E16 step 4: speed of query-time document encoding: accelerator float vs CPU float vs CPU dynamic qint8."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
BGE = "BAAI/bge-small-en-v1.5"; PFX = "Represent this sentence for searching relevant passages: "
_, docs, _, _, _ = load_dataset("fiqa"); r = np.random.default_rng(SEED); S = [docs[i] for i in r.integers(0, len(docs), 800)]; out = {}
def bench(m, bs):
    for x in S[:8]: encode(m, [x], bs=bs)
    ts = []
    for i in range(0, 400, bs):
        t0 = time.perf_counter(); encode(m, S[i:i + bs], bs=bs); ts.append((time.perf_counter() - t0) / bs)
    return {"ms_per_doc_p50": float(np.median(ts) * 1e3), "ms_per_doc_mean": float(np.mean(ts) * 1e3)}
ms = {device() + "_float": encoder(BGE, 256), "cpu_float": encoder(BGE, 256, "cpu")}
qm = encoder(BGE, 256, "cpu"); torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines and device() != "cuda" else torch.backends.quantized.engine
qm[0].auto_model = torch.ao.quantization.quantize_dynamic(qm[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8); ms["cpu_qint8"] = qm
for k, mm in ms.items():
    for bs in [1, 4, 16]: out[f"{k}_bs{bs}"] = bench(mm, bs); log(k, bs, out[f"{k}_bs{bs}"])
    save_json(out, "e16_bench.json")
