"""E21: monthly cost of each design from measured per-document costs and sizes. All prices are assumptions; sensitivity is reported.
Billing variants: 'utilised' charges accelerator seconds used; 'always-on' charges a whole accelerator for any design that encodes on the query path."""
import json
from common import save_json, load_json
SEC = 30 * 86400; C = 0.0162; HNSW = 1685; FWD = 1536
def memcost(b, pm): return b / 1e9 * pm
def designs(N, u, Q, rho, pa, pm, c=C, K=100, h=0.5, touched=0.1, f=0.1, always=True):
    pas = pa / 3600.0; enc = lambda r: r * SEC * c * pas; idle = pa * SEC / 3600.0 if always else 0.0; eager_enc = enc(u + rho * N); out = {}
    out["eager"] = eager_enc + memcost(N * HNSW, pm)
    out["lazy_fusion"] = eager_enc + memcost(N * HNSW, pm) + idle  # same encodings with write-back, one vector per document; query path needs an accelerator
    out["jit_af_ann"] = eager_enc + memcost(N * HNSW, pm) + idle
    out["jit_af_wb"] = enc(u * touched + rho * N * touched) + memcost(N * touched * FWD, pm) + idle  # no ANN, write-back, share of documents ever retrieved
    out["jit_nowb"] = enc(Q * K * (1 - h)) + idle  # plain query-time re-ranking with a bounded cache
    return out
base = dict(N=1e6, u=1.0, Q=1.0, rho=0.0, pa=1.0, pm=4.0); T = {"base_alwayson": designs(**base), "base_utilised": designs(**base, always=False)}
sens = {}
for name, ch in [("pa0.5", dict(pa=0.5)), ("pa3", dict(pa=3.0)), ("pm2", dict(pm=2.0)), ("pm8", dict(pm=8.0)), ("N1e5", dict(N=1e5)), ("N1e7", dict(N=1e7)), ("u0.1", dict(u=0.1)), ("u10", dict(u=10.0)), ("Q0.1", dict(Q=0.1)), ("Q10", dict(Q=10.0)), ("c20x", dict(c=C / 20)), ("rho", dict(rho=1 / SEC))]:
    p = {**base, **ch}; sens[name] = {"alwayson": designs(**p), "utilised": designs(**p, always=False)}
T["sens"] = sens
def qstar(N, u, rho, pa, pm, K, h, c, always):
    pas = pa / 3600; eager = (u + rho * N) * c * pas * SEC + N * HNSW / 1e9 * pm
    if always: return max(0.0, (eager - pa * SEC / 3600) / (SEC * K * (1 - h) * c * pas))
    return eager / (SEC * K * (1 - h) * c * pas)
T["qstar_v2_prices"] = {"utilised": qstar(1e6, 0, 0, 1, 4, 100, .5, 0.0082, False), "utilised_bge": qstar(1e6, 0, 0, 1, 4, 100, .5, C, False), "always_on": qstar(1e6, 0, 0, 1, 4, 100, .5, 0.0082, True), "u1_utilised_bge": qstar(1e6, 1, 0, 1, 4, 100, .5, C, False)}
T["mem_vs_enc_rate_minilm"] = (1e6 * HNSW / 1e9 * 4 / SEC) / (0.0082 / 3600); T["always_on_month"] = 720.0; T["mem_1M"] = memcost(1e6 * HNSW, 4.0); T["assumptions"] = {"accelerator_usd_per_hour": 1.0, "memory_usd_per_gb_month": 4.0, "c_s": C}
save_json(T, "e21_cost.json"); print(json.dumps({k: T[k] for k in T if k != "sens"}, indent=0)[:1500])
