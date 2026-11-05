"""Nested selection for E11. Query sets are split once, by seed, into dev and test halves.
Weights and normalization: chosen on dev queries of the OTHER six collections.
Configuration: chosen on dev queries only. Test queries are used only for the final numbers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]; NORMS = ["mm", "max", "z", "rank"]
P = {d: load_json(d, "perquery_e11.json") for d in DATASETS}; B = {d: load_json(d, "perquery_e08_bge.json") for d in DATASETS}
SPLIT = {}
for d in DATASETS:
    q = sorted(P[d]["b100|mm|0.9"]["ndcg"]); r = np.random.default_rng(SEED).permutation(len(q)); SPLIT[d] = {"dev": [q[i] for i in r[:len(q) // 2]], "test": [q[i] for i in r[len(q) // 2:]]}
sm = lambda d, k, part, f="ndcg": float(np.mean([P[d][k][f][q] for q in SPLIT[d][part]]))
def lodo_pick(c, d):
    return max(((n, a) for n in NORMS for a in ALPHAS), key=lambda x: np.mean([sm(o, f"{c}|{x[0]}|{x[1]}", "dev") for o in DATASETS if o != d]))
def assemble(c, part):
    v, r, pk = {}, {}, {}
    for d in DATASETS:
        n, a = lodo_pick(c, d); pk[d] = (n, a); k = f"{c}|{n}|{a}"
        v[d] = {q: P[d][k]["ndcg"][q] for q in SPLIT[d][part]}; r[d] = {q: P[d][k]["recall100"][q] for q in SPLIT[d][part]}
    return v, r, pk
mac = lambda v: float(np.mean([np.mean(list(v[d].values())) for d in v]))
CF = [k for k in P[DATASETS[0]]["__enc__"]]
print("config dev nDCG, dev R100, encodings (macro)")
S = {}
for c in CF:
    v, r, pk = assemble(c, "dev"); S[c] = (mac(v), mac(r), float(np.mean([P[d]["__enc__"][c] for d in DATASETS])), sorted(set(pk.values())))
    print(f"{c:16s} {S[c][0]:.4f} {S[c][1]:.3f} {S[c][2]:6.1f}", S[c][3])
b = [c for c in CF if not c.startswith("h") and S[c][2] <= 101]; best = max(b, key=lambda c: S[c][0]); best2 = max([c for c in CF if not c.startswith("h")], key=lambda c: S[c][0])
hb = max([c for c in CF if c.startswith("h")], key=lambda c: S[c][0])
print("selected at budget 100:", best, "| any budget:", best2, "| rival:", hb)
def mt(a, b, ds):
    rng = np.random.default_rng(SEED); arr = {d: np.array([a[d][q] - b[d][q] for q in a[d]]) for d in ds}
    mb = [np.mean([rng.choice(arr[d], len(arr[d])).mean() for d in ds]) for _ in range(4000)]
    return float(np.mean([arr[d].mean() for d in ds])), float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))
T = {c: assemble(c, "test") for c in {best, best2, hb, "b100", "h100", "h50"}}
def base(s, f="ndcg"): return {d: {q: B[d][s][f][q] for q in SPLIT[d]["test"]} for d in DATASETS}
T["rrf"] = (base("rrf"), base("rrf", "recall100"), None); T["dense"] = (base("dense"), base("dense", "recall100"), None)
print("\nHELD-OUT TEST HALF")
for k, (v, r, pk) in T.items():
    print(f"{k:16s} nDCG {mac(v):.4f} R100 {mac(r):.3f}", {d[:5]: round(float(np.mean(list(v[d].values()))), 3) for d in DATASETS})
XT = [d for d in DATASETS if d != "trec-covid"]; out = {"split": {d: {k: len(x) for k, x in SPLIT[d].items()} for d in DATASETS}, "selected": best, "selected_any": best2, "rival": hb, "dev": {c: S[c][:3] for c in S}, "test": {}, "tests": {}}
for k, (v, r, pk) in T.items():
    out["test"][k] = {"ndcg": mac(v), "recall": mac(r), "per": {d: float(np.mean(list(v[d].values()))) for d in DATASETS}, "recall_per": {d: float(np.mean(list(r[d].values()))) for d in DATASETS}, "pick": pk}
for a in [best, best2]:
    for o in ["b100", "rrf", "dense", "h100", "h50", hb]:
        if a == o: continue
        A, Bb = T[a][0], T[o][0]; t = {d: paired_test(list(A[d].values()) if False else A[d], Bb[d]) for d in DATASETS}; h = holm([t[d]["p"] for d in DATASETS])
        for d, x in zip(DATASETS, h): t[d]["sig"] = bool(x < .05 and (t[d]["lo"] > 0 or t[d]["hi"] < 0))
        m, mx = mt(A, Bb, DATASETS), mt(A, Bb, XT); rr = mt(T[a][1], T[o][1], DATASETS)
        out["tests"][f"{a}|{o}"] = {"per": t, "macro": m, "macro_xt": mx, "recall": rr}
        print(f"{a} vs {o}: nDCG %+.4f [%+.4f,%+.4f]; noTREC %+.4f [%+.4f,%+.4f]; R100 %+.4f [%+.4f,%+.4f]" % (*m, *mx, *rr), {d[:5]: "%+.3f%s" % (t[d]["diff"], "*" if t[d]["sig"] else "") for d in DATASETS})
save_json(out, "e11_report.json")
