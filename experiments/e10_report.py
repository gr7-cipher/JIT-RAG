"""Summarises E10: cascade with leave-one-domain-out (n, beta), and quality under indexing lag."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
SYS = ["jitcc", "hybcc", "rrf", "dense"]; NS = [10, 20, 50]; BS = [round(.1 * i, 1) for i in range(11)]
P = {d: load_json(d, "perquery_e10.json") for d in DATASETS}
XT = [d for d in DATASETS if d != "trec-covid"]
m = lambda d, k: mean(P[d][k]["ndcg"])
def macro_test(a, b, ds=DATASETS):
    rng = np.random.default_rng(SEED); arr = {d: np.array([a[d][q] - b[d][q] for q in a[d]]) for d in ds}
    mb = [np.mean([rng.choice(arr[d], len(arr[d])).mean() for d in ds]) for _ in range(4000)]
    return {"diff": float(np.mean([arr[d].mean() for d in ds])), "lo": float(np.percentile(mb, 2.5)), "hi": float(np.percentile(mb, 97.5))}
def per_test(a, b):
    t = {d: paired_test(a[d], b[d]) for d in DATASETS}; h = holm([t[d]["p"] for d in DATASETS])
    for d, x in zip(DATASETS, h): t[d]["p_holm"] = x; t[d]["sig"] = bool(x < .05 and (t[d]["lo"] > 0 or t[d]["hi"] < 0))
    return t
R = {"cascade": {}, "tests": {}, "lag": {}, "lag_tests": {}}
V, B = {}, {}
for s in SYS:
    B[s] = {d: P[d][f"{s}_n20_b0.0"]["ndcg"] for d in DATASETS}; V[s] = {}; pk = {}
    for d in DATASETS:
        n, b = max(((n, b) for n in NS for b in BS), key=lambda x: np.mean([m(o, f"{s}_n{x[0]}_b{x[1]}") for o in DATASETS if o != d]))
        pk[d] = [n, b]; V[s][d] = P[d][f"{s}_n{n}_b{b}"]["ndcg"]
    R["cascade"][s] = {"pick": pk, "base": {d: mean(B[s][d]) for d in DATASETS}, "ce": {d: mean(V[s][d]) for d in DATASETS}, "base_macro": float(np.mean([mean(B[s][d]) for d in DATASETS])), "ce_macro": float(np.mean([mean(V[s][d]) for d in DATASETS])), "ce_macro_xt": float(np.mean([mean(V[s][d]) for d in XT]))}
    print(s, "base %.4f  +CE %.4f  (no TREC-COVID %.4f)" % (R["cascade"][s]["base_macro"], R["cascade"][s]["ce_macro"], R["cascade"][s]["ce_macro_xt"]), pk, {d[:5]: round(mean(V[s][d]), 3) for d in DATASETS})
for a, b, ka, kb in [("jitcc+ce", "hybcc+ce", V["jitcc"], V["hybcc"]), ("jitcc+ce", "rrf+ce", V["jitcc"], V["rrf"]), ("jitcc+ce", "dense+ce", V["jitcc"], V["dense"]), ("jitcc+ce", "jitcc", V["jitcc"], B["jitcc"]), ("hybcc+ce", "rrf+ce", V["hybcc"], V["rrf"])]:
    t = per_test(ka, kb); t["macro"] = macro_test(ka, kb); t["macro_xt"] = macro_test(ka, kb, XT); R["tests"][f"{a}|{b}"] = t
    print(a, "vs", b, "macro %+.4f [%+.4f,%+.4f]; no TREC %+.4f [%+.4f,%+.4f]" % (t["macro"]["diff"], t["macro"]["lo"], t["macro"]["hi"], t["macro_xt"]["diff"], t["macro_xt"]["lo"], t["macro_xt"]["hi"]), {d[:5]: "%+.3f%s" % (t[d]["diff"], "*" if t[d]["sig"] else "") for d in DATASETS})
# lag
L = {d: {**load_json(d, "perquery_e10_lag.json"), **load_json(d, "perquery_e10_lagimp.json")} for d in DATASETS}
enc = load_json("e10_lag.json")
TAGS = ["0.0", "0.01", "0.05", "0.1", "0.25", "0.5", "recent"]
for t in TAGS:
    row = {"jitcc": float(np.mean([mean(L[d]["jitcc"]) for d in DATASETS])), "encoded": float(np.mean([enc[d]["encoded_per_q"][t] for d in DATASETS]))}
    for s in ["dense", "rrf", "hybcc", "rrfimp", "hybimp"]:
        k = f"{s}_{t}"
        if k in L[DATASETS[0]]: row[s] = float(np.mean([mean(L[d][k]) for d in DATASETS]))
    R["lag"][t] = row; print(t, {k: round(v, 4) for k, v in row.items()})
J = {d: L[d]["jitcc"] for d in DATASETS}
for t in ["0.01", "0.05", "0.1", "0.25", "0.5", "recent"]:
    for s in ["rrfimp", "hybimp"]:
        b = {d: L[d][f"{s}_{t}"] for d in DATASETS}; x = per_test(J, b); x["macro"] = macro_test(J, b); R["lag_tests"][f"jitcc|{s}_{t}"] = x
        print("jitcc vs", s, t, "macro %+.4f [%+.4f,%+.4f]" % (x["macro"]["diff"], x["macro"]["lo"], x["macro"]["hi"]), {d[:5]: "%+.3f%s" % (x[d]["diff"], "*" if x[d]["sig"] else "") for d in DATASETS})
save_json(R, "e10_report.json")
