"""E20: statistics over E17. Equivalence tests against the margin in e17_margin.json, paired t and sign-flip permutation tests per collection,
Holm across the whole family of per-collection tests, collection-level evidence, spread over backlog masks and over dev/test splits."""
import os, sys, json
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, binomtest
from common import load_json, save_json, holm, RES, DATASETS, SEED
EXT = ["webis-touche2020", "cqa-webmasters", "cqa-android", "cqa-wordpress", "fiqa_fresh", "nfcorpus_fresh"]
L = {s: json.load(open(os.path.join(RES, "e17", s + ".json"))) for s in DATASETS + EXT}; D = load_json("e17_margin.json")["delta_macro_ndcg10"]
def half(q, seed=SEED):
    q = sorted(q); r = np.random.default_rng(seed).permutation(len(q)); return [q[i] for i in r[len(q) // 2:]], [q[i] for i in r[:len(q) // 2]]
IDX = {}
for s in DATASETS:
    qs = L[s]["qids"]; pos = {q: i for i, q in enumerate(qs)}; IDX[s] = np.array([pos[q] for q in half(qs)[0]])
for s in EXT: IDX[s] = np.arange(len(L[s]["qids"]))
def arr(s, k, met): return np.asarray(L[s]["sys"][k][met])[IDX[s]]
def perm_p(x, n=10000, seed=SEED):
    r = np.random.default_rng(seed); o = abs(x.mean()); s = r.choice([-1.0, 1.0], size=(n, len(x))); return float((np.sum(np.abs((s * x).mean(1)) >= o - 1e-15) + 1) / (n + 1))
def cmp(a, b, sets, met="ndcg", nb=2000):
    r = np.random.default_rng(SEED); xs = [arr(s, a, met) - arr(s, b, met) for s in sets]; ds = np.array([x.mean() for x in xs]); bs = np.mean([[x[r.integers(0, len(x), len(x))].mean() for _ in range(nb)] for x in xs], 0)
    per = {}
    for s, x in zip(sets, xs): per[s] = {"diff": float(x.mean()), "n": len(x), "t_p": 1.0 if np.allclose(x, 0) else float(ttest_rel(x, np.zeros_like(x)).pvalue), "perm_p": perm_p(x), "zero_share": float(np.mean(np.isclose(x, 0)))}
    w, l = int((ds > 1e-9).sum()), int((ds < -1e-9).sum()); nz = ds[np.abs(ds) > 1e-9]
    return {"diff": float(ds.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))], "tost_05": bool(np.percentile(bs, 5) > -D and np.percentile(bs, 95) < D), "tost_002": bool(np.percentile(bs, 5) > -.002 and np.percentile(bs, 95) < .002), "wins": w, "losses": l, "sign_p": float(binomtest(w, w + l).pvalue) if w + l else 1.0, "wilcoxon_coll_p": float(wilcoxon(nz).pvalue) if len(nz) else 1.0, "loco": [float(min((ds.sum() - x) / (len(ds) - 1) for x in ds)), float(max((ds.sum() - x) / (len(ds) - 1) for x in ds))], "per": per}
def mac(k, sets, met="ndcg"): return float(np.mean([arr(s, k, met).mean() for s in sets]))
R = {"delta": D, "macro": {}, "cmp": {}}
for k in ["full", "noGamma", "full_a9", "noGamma_a9", "noANN", "jitaf", "noExp", "noExp_a9", "noExp_gamma", "h50", "h100", "h100_a8", "h100_gamma", "jit100", "jitcc100", "dense", "rrf", "bm25"]:
    for nm, ss in [("seven", DATASETS), ("ext", EXT)]:
        R["macro"][f"{k}|{nm}"] = {m: mac(k, ss, m) for m in ["ndcg", "r10", "r100", "j10"]} ; R["macro"][f"{k}|{nm}"]["pool"] = float(np.mean([L[s]["sys"][k]["pool"] for s in ss]))
        R["macro"][f"{k}|{nm}"]["per"] = {s: {m: float(arr(s, k, m).mean()) for m in ["ndcg", "r10", "r100", "j10"]} for s in ss}
BK = [f"{f}" for f in [0.01, 0.05, 0.1, 0.25, 0.5]]; lag = {}
for nm, ss in [("seven", DATASETS), ("ext", EXT)]:
    for f in BK:
        for sysk in ["afh", "lazy50", "lazy100", "sa50", "sa100"]:
            vals = [mac(f"{sysk}|f{f}|s{s}", ss) for s in [13, 14, 15, 16, 17]]; r10 = [mac(f"{sysk}|f{f}|s{s}", ss, "r10") for s in [13, 14, 15, 16, 17]]; r100 = [mac(f"{sysk}|f{f}|s{s}", ss, "r100") for s in [13, 14, 15, 16, 17]]
            lag[f"{sysk}|{f}|{nm}"] = {"seed13": vals[0], "mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "r10": float(np.mean(r10)), "r100": float(np.mean(r100)), "enc": float(np.mean([L[s]["sys"][f"{sysk}|f{f}|s13"]["enc"] for s in ss]))}
        lag[f"rrfsa|{f}|{nm}"] = {"seed13": mac(f"rrfsa|f{f}|s13", ss), "enc": 0.0}
    for sysk in ["afh", "lazy50", "lazy100", "sa50", "sa100", "rrfsa"]:
        lag[f"{sysk}|recent|{nm}"] = {"seed13": mac(f"{sysk}|recent", ss), "r10": mac(f"{sysk}|recent", ss, "r10"), "r100": mac(f"{sysk}|recent", ss, "r100"), "enc": float(np.mean([L[s]["sys"][f"{sysk}|recent"]["enc"] for s in ss]))}
R["lag"] = lag
BUD = {}
for nm, ss in [("seven", DATASETS), ("ext", EXT)]:
    for f in ["0.1", "0.5", "1.0"]:
        for m in [5, 10, 20, 50]:
            for sysk in ["afh", "lazy50", "lazy100"]:
                k = f"{sysk}|f{f}|m{m}"; BUD[f"{k}|{nm}"] = {"ndcg": mac(k, ss), "r100": mac(k, ss, "r100"), "enc": float(np.mean([L[s]["sys"][k]["enc"] for s in ss]))}
R["budget"] = BUD
C = {}
PR = [("full", "h100", "ndcg"), ("full", "h100", "r10"), ("full", "h100", "r100"), ("full", "h50", "r100"), ("full", "rrf", "ndcg"), ("full", "dense", "ndcg"), ("full", "noExp", "ndcg"), ("full", "noExp", "r100"), ("full", "noExp", "r10"), ("full", "noGamma", "ndcg"), ("full", "noANN", "ndcg"), ("full", "noANN", "r100"), ("noGamma_a9", "h100", "ndcg"), ("noExp_a9", "h100", "ndcg"), ("jitaf", "h100", "ndcg"), ("jitaf", "h100", "r100"), ("h100_gamma", "full", "ndcg"), ("h100_gamma", "h100", "ndcg"), ("full_a9", "full", "ndcg"), ("full_a9", "h100", "ndcg"), ("h50", "h100", "ndcg")]
for nm, ss in [("seven", DATASETS), ("ext", EXT)]:
    for a, b, m in PR: C[f"{a}-{b}|{m}|{nm}"] = cmp(a, b, ss, m)
    for f in BK:
        for a, b in [("afh", "lazy50"), ("afh", "lazy100"), ("afh", "sa100"), ("lazy50", "sa100"), ("lazy100", "sa100"), ("lazy100", "lazy50")]:
            for m in ["ndcg", "r10", "r100"]: C[f"{a}-{b}|{m}|f{f}|{nm}"] = cmp(f"{a}|f{f}|s13", f"{b}|f{f}|s13", ss, m)
        C[f"afh-rrfsa|ndcg|f{f}|{nm}"] = cmp(f"afh|f{f}|s13", f"rrfsa|f{f}|s13", ss); C[f"lazy100-rrfsa|ndcg|f{f}|{nm}"] = cmp(f"lazy100|f{f}|s13", f"rrfsa|f{f}|s13", ss)
    for a, b in [("afh", "lazy50"), ("afh", "lazy100"), ("afh", "sa100"), ("afh", "rrfsa"), ("lazy100", "rrfsa"), ("lazy100", "sa100")]: C[f"{a}-{b}|ndcg|recent|{nm}"] = cmp(f"{a}|recent", f"{b}|recent", ss)
# Holm over the whole family of per-collection nDCG tests
fam = [(k, s) for k in C if "|ndcg|" in k for s in C[k]["per"]]; h = holm([C[k]["per"][s]["t_p"] for k, s in fam])
for (k, s), x in zip(fam, h): C[k]["per"][s]["holm"] = x
R["family_size"] = len(fam); R["cmp"] = C
for k in C: pass
# repeated dev/test splits: reselect each system on dev, evaluate on test, from stored grids
P14 = {d: load_json(d, "perquery_e14.json") for d in DATASETS}; P11 = {d: load_json(d, "perquery_e11.json") for d in DATASETS}; SP = []
for sd in range(13, 33):
    dv = {}; te = {}
    for d in DATASETS: te[d], dv[d] = half(P14[d]["max|0.9|0.0"]["ndcg"], sd)
    ms = lambda P, k, Q, m="ndcg": float(np.mean([np.mean([P[d][k][m][q] for q in Q[d]]) for d in DATASETS]))
    a = max(P14["fiqa"], key=lambda k: ms(P14, k, dv)); hk = [k for k in P11["fiqa"] if k.startswith("h100|")]; b = max(hk, key=lambda k: ms(P11, k, dv))
    SP.append({"seed": sd, "af": a, "h": b, "af_ndcg": ms(P14, a, te), "h_ndcg": ms(P11, b, te), "diff": ms(P14, a, te) - ms(P11, b, te), "r100_diff": ms(P14, a, te, "recall100") - ms(P11, b, te, "recall100")})
R["splits"] = {"runs": SP, "diff_mean": float(np.mean([x["diff"] for x in SP])), "diff_min": float(np.min([x["diff"] for x in SP])), "diff_max": float(np.max([x["diff"] for x in SP])), "r100_mean": float(np.mean([x["r100_diff"] for x in SP])), "r100_min": float(np.min([x["r100_diff"] for x in SP])), "r100_max": float(np.max([x["r100_diff"] for x in SP])), "af_choices": sorted({x["af"] for x in SP}), "h_choices": sorted({x["h"] for x in SP})}
R["writeback"] = {s: L[s]["writeback"] for s in DATASETS + EXT if L[s].get("writeback")}; R["N"] = {s: L[s]["N"] for s in DATASETS + EXT}; R["nq"] = {s: len(IDX[s]) for s in DATASETS + EXT}
save_json(R, "e20_stats.json"); print("family", len(fam))
for k in ["full-h100|ndcg|seven", "full-h100|r10|seven", "full-h100|r100|seven", "full-h100|ndcg|ext", "full-h100|r100|ext", "full-noExp|ndcg|seven", "full-noExp|r100|seven", "afh-lazy50|ndcg|f0.1|seven", "afh-lazy100|ndcg|f0.1|seven", "afh-lazy50|ndcg|f0.5|seven", "afh-lazy100|ndcg|f0.5|seven", "lazy100-sa100|ndcg|f0.1|seven", "afh-lazy50|ndcg|f0.1|ext", "afh-lazy100|ndcg|f0.5|ext"]:
    x = C[k]; print(k, "%+.4f [%+.4f,%+.4f] 90[%+.4f,%+.4f] tost %s +%d/-%d" % (x["diff"], *x["ci95"], *x["ci90"], x["tost_05"], x["wins"], x["losses"]))
print(R["splits"]["diff_mean"], R["splits"]["diff_min"], R["splits"]["diff_max"], R["splits"]["af_choices"], R["splits"]["h_choices"])
