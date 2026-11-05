"""Leave-one-domain-out comparison of the proposed fusion against every baseline, for one encoder."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
enc = sys.argv[1]; ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; KS = [50, 100, 200]
P = {d: load_json(d, f"perquery_e08_{enc}.json") for d in DATASETS}; B = {d: load_json(d, "perquery.json") for d in DATASETS}
m = lambda d, s, f="ndcg": mean(P[d][s][f])
def lodo(tag, f="ndcg"):
    picks, vals, rec = {}, {}, {}
    for d in DATASETS:
        a = max(ALPHAS, key=lambda a: np.mean([m(x, f"{tag}_{a}") for x in DATASETS if x != d]))
        picks[d] = a; vals[d] = P[d][f"{tag}_{a}"]["ndcg"]; rec[d] = P[d][f"{tag}_{a}"]["recall100"]
    return picks, vals, rec
R = {"encoder": enc, "systems": {}, "tests": {}}
def row(lbl, v, r):
    R["systems"][lbl] = {"per": {d: mean(v[d]) for d in DATASETS}, "macro": float(np.mean([mean(v[d]) for d in DATASETS])), "recall": float(np.mean([mean(r[d]) for d in DATASETS])), "recall_per": {d: mean(r[d]) for d in DATASETS}}
for s in ["dense", "rrf", "jit_k50", "jit_k100", "jit_k200"]:
    row(s, {d: P[d][s]["ndcg"] for d in DATASETS}, {d: P[d][s]["recall100"] for d in DATASETS})
row("bm25", {d: B[d]["bm25"]["ndcg"] for d in DATASETS}, {d: B[d]["bm25"]["recall100"] for d in DATASETS})
V = {}
for K in KS:
    for t in ["jitcc", "hybcc"]:
        pk, v, r = lodo(f"{t}_k{K}"); V[f"{t}_k{K}"] = v; row(f"{t}_k{K}", v, r); R["systems"][f"{t}_k{K}"]["alpha"] = pk
for lbl, sysv in [("jitcc_k100", V["jitcc_k100"]), ("jitcc_k200", V["jitcc_k200"])]:
    for base, bv in [("rrf", {d: P[d]["rrf"]["ndcg"] for d in DATASETS}), ("hybcc_k100", V["hybcc_k100"]), ("hybcc_k200", V["hybcc_k200"]), ("dense", {d: P[d]["dense"]["ndcg"] for d in DATASETS}), ("bm25", {d: B[d]["bm25"]["ndcg"] for d in DATASETS})]:
        t = {d: paired_test(sysv[d], bv[d]) for d in DATASETS}
        h = holm([t[d]["p"] for d in DATASETS])
        for d, x in zip(DATASETS, h): t[d]["p_holm"] = x
        # macro test: bootstrap across collections' queries, each collection weighted equally
        rng = np.random.default_rng(SEED); mb = []
        arr = {d: np.array([sysv[d][q] - bv[d][q] for q in sysv[d]]) for d in DATASETS}
        for _ in range(4000):
            mb.append(np.mean([rng.choice(arr[d], len(arr[d])).mean() for d in DATASETS]))
        t["macro"] = {"diff": float(np.mean([arr[d].mean() for d in DATASETS])), "lo": float(np.percentile(mb, 2.5)), "hi": float(np.percentile(mb, 97.5))}
        R["tests"][f"{lbl}|{base}"] = t
save_json(R, f"e08_report_{enc}.json")
for k, v in R["systems"].items():
    print(f"{k:12s} {v['macro']:.4f} R100 {v['recall']:.3f} ", {d[:5]: round(x, 3) for d, x in v["per"].items()})
for k, t in R["tests"].items():
    print(k, "macro %+.4f [%+.4f,%+.4f]" % (t["macro"]["diff"], t["macro"]["lo"], t["macro"]["hi"]), {d[:5]: ("%+.3f" % t[d]["diff"]) + ("*" if t[d]["p_holm"] < .05 and (t[d]["lo"] > 0 or t[d]["hi"] < 0) else "") for d in DATASETS})
