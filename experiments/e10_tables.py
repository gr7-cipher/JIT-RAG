"""Tables, figure and utility sweep for the improved pipeline. Run after e08_report.py and e10_report.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("JIT_OUT", os.path.join(ROOT, "..", "resubmission"))
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "mute": "#6f6e69", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["mute"], "axes.linewidth": .6, "xtick.color": C["mute"], "ytick.color": C["mute"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .5, "legend.frameon": False, "axes.axisbelow": True})
NICE = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "fiqa": "FiQA", "scidocs": "SciDocs", "trec-covid": "TREC-COVID", "quora": "Quora", "nq": "NQ", "msmarco": "MS MARCO"}
Bg = load_json("e08_report_bge.json"); Mi = load_json("e08_report_minilm.json"); R = load_json("e10_report.json"); La = load_json("e10_latency.json"); Lh = load_json("e10_latency_hyb.json")
def wr(n, s):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True); open(os.path.join(OUT, "tables", n + ".tex"), "w").write(s + "\n")
def row(lbl, vals, f="%.3f"):
    b = max(vals); return lbl + " & " + " & ".join(("\\textbf{%s}" % (f % v)) if v == b else f % v for v in vals) + " \\\\"
# table: quality with bge
S = ["bm25", "dense", "rrf", "jit_k100", "hybcc_k50", "hybcc_k100", "jitcc_k100"]
r = []
for d in DATASETS:
    r.append(row(NICE[d], [Bg["systems"][s]["per"][d] for s in S]))
r += ["\\midrule", row("Macro nDCG@10", [Bg["systems"][s]["macro"] for s in S]), row("Macro Recall@100", [Bg["systems"][s]["recall"] for s in S]),
      row("MiniLM macro nDCG@10", [Mi["systems"][s]["macro"] for s in S])]
wr("cc_quality", "\n".join(r))
# table: paired tests
T = load_json("e08_report_bge.json")["tests"]
def cell(t): return "%+.3f%s" % (t["diff"], "$^{*}$" if t["p_holm"] < .05 and (t["lo"] > 0 or t["hi"] < 0) else "")
pairs = ["jitcc_k100|bm25", "jitcc_k100|dense", "jitcc_k100|rrf", "jitcc_k100|hybcc_k100"]
r = [NICE[d] + " & " + " & ".join(cell(T[p][d]) for p in pairs) + " \\\\" for d in DATASETS]
r += ["\\midrule", "Macro [95\\% CI] & " + " & ".join("%+.4f [%+.4f, %+.4f]" % (T[p]["macro"]["diff"], T[p]["macro"]["lo"], T[p]["macro"]["hi"]) for p in pairs) + " \\\\"]
wr("cc_sig", "\n".join(r))
# table: lag
order = ["0.0", "0.01", "0.05", "0.1", "0.25", "0.5", "recent"]; lbl = {"0.0": "0\\%", "0.01": "1\\%", "0.05": "5\\%", "0.1": "10\\%", "0.25": "25\\%", "0.5": "50\\%", "recent": "Recent relevant"}
r = []
for t in order:
    x = R["lag"][t]; v = [x["jitcc"], x["rrf"], x.get("rrfimp", x["rrf"]), x["hybcc"], x.get("hybimp", x["hybcc"]), x["dense"]]
    r.append(lbl[t] + " & " + row("%.1f" % x["encoded"], v).split(" & ", 1)[0] + " & " + row("", v).split(" & ", 1)[1])
wr("cc_lag", "\n".join(r))
# table: latency
M = ["jitcc_f0.0", "jitcc_f0.01", "jitcc_f0.1", "jitcc_f0.5", "jitcc_f1.0", "jitcc_f0.0_ce20"]
r = []
for d in DATASETS:
    r.append(NICE[d] + " & " + " & ".join("{:,.0f} / {:,.0f}".format(La[d][m]["p50"], La[d][m]["p95"]) for m in M) + " & {:,.0f} / {:,.0f}".format(La[d]["rrf_hnsw"]["p50"], La[d]["rrf_hnsw"]["p95"]) + " & {:,.0f} / {:,.0f}".format(Lh[d]["p50"], Lh[d]["p95"]) + " \\\\")
wr("cc_latency", "\n".join(r))
# cascade table
Cs = R["cascade"]; K = ["jitcc", "hybcc", "rrf", "dense"]; r = []
for s, n in zip(K, ["JIT-CC $K{=}100$", "Hybrid CC $K{=}100$", "RRF", "Dense"]):
    r.append(n + " & %.3f & %.3f & %+.3f & %.3f \\\\" % (Cs[s]["base_macro"], Cs[s]["ce_macro"], Cs[s]["ce_macro"] - Cs[s]["base_macro"], Cs[s]["ce_macro_xt"]))
wr("cc_cascade", "\n".join(r))
# figure
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
fig, a = plt.subplots(figsize=(3.6, 2.4)); F = ["0.0", "0.01", "0.05", "0.1", "0.25", "0.5"]; xs = [float(x) * 100 for x in F]; Lg = R["lag"]
g = lambda k: [Lg[t].get(k, Lg[t][k.replace("imp", "cc") if k == "hybimp" else "rrf"]) for t in F]
a.plot(xs, [Lg[t]["jitcc"] for t in F], "-o", ms=3, lw=1.5, color=C["blue"], label="JIT-CC")
a.plot(xs, g("hybimp"), "-s", ms=3, lw=1, color=C["aqua"], label="Hybrid CC, signal-aware")
a.plot(xs, g("rrfimp"), "-^", ms=3, lw=1, color=C["orange"], label="RRF, signal-aware")
a.plot(xs, [Lg[t]["rrf"] for t in F], ":", lw=1, color=C["orange"], label="RRF, unmodified")
a.plot(xs, [Lg[t]["dense"] for t in F], "--", lw=1, color=C["mute"], label="Dense only")
a.set_xlabel("Corpus waiting for the encoder (%)"); a.set_ylabel("Macro nDCG@10"); a.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "cc_lag.pdf")); fig.savefig(os.path.join(OUT, "figures", "cc_lag.png"), dpi=200)
# utility sweep: inverted-index lag applies to all, so freshness cancels except for encoder-only systems
cal = load_json("latcal.json") if os.path.exists(os.path.join(RES, "latcal.json")) else None
grid = [(a_ / 20, b_ / 20, 1 - a_ / 20 - b_ / 20) for a_ in range(21) for b_ in range(21 - a_)]
LMAX = 2000.0; out = {}
P = {d: load_json(d, "perquery.json") for d in DATASETS}
bmq = float(np.mean([mean(P[d]["bm25"]["ndcg"]) for d in DATASETS]))
bml = float(np.mean([Lh[d]["bm25_p50"] for d in DATASETS]))
for tag, fj in [("0.0", "jitcc_f0.0"), ("0.1", "jitcc_f0.1"), ("0.5", "jitcc_f0.5")]:
    Lg_ = Lg[tag]; sysq = {"BM25": bmq, "JIT-CC": Lg_["jitcc"], "RRF": Lg_.get("rrfimp", Lg_["rrf"]), "Hybrid CC": Lg_.get("hybimp", Lg_["hybcc"])}
    syl = {"BM25": bml, "JIT-CC": float(np.mean([La[d][fj]["p50"] for d in DATASETS])), "RRF": float(np.mean([La[d]["rrf_hnsw"]["p50"] for d in DATASETS])), "Hybrid CC": float(np.mean([Lh[d]["p50"] for d in DATASETS]))}
    w = {}
    for wq, wl, wf in grid:
        b = max(sysq, key=lambda s: wq * sysq[s] + wl * (1 - min(1, syl[s] / LMAX)) + wf); w[b] = w.get(b, 0) + 1 / len(grid)
    out[tag] = {"q": sysq, "l": syl, "share": w}; print(tag, {k: round(v, 3) for k, v in sysq.items()}, {k: round(v, 1) for k, v in syl.items()}, {k: round(v, 3) for k, v in w.items()})
save_json(out, "e10_utility.json")
# plain JIT vs JIT-CC, same encoder
Pb = {d: load_json(d, "perquery_e08_bge.json") for d in DATASETS}
J = {d: Pb[d]["jitcc_k100_%s" % Bg["systems"]["jitcc_k100"]["alpha"][d]]["ndcg"] for d in DATASETS}; pj = {d: Pb[d]["jit_k100"]["ndcg"] for d in DATASETS}
t = {d: paired_test(J[d], pj[d]) for d in DATASETS}; h = holm([t[d]["p"] for d in DATASETS])
print("jitcc-jit", {d[:5]: "%+.3f%s" % (t[d]["diff"], "*" if x < .05 and (t[d]["lo"] > 0 or t[d]["hi"] < 0) else "") for d, x in zip(DATASETS, h)})
print("mean L", {m: round(float(np.mean([La[d][m]["p50"] for d in DATASETS if d != "quora"]))) for m in M}, "quora", {m: round(La["quora"][m]["p50"]) for m in M})
print("e08 macros bge", {s: round(Bg["systems"][s]["macro"], 4) for s in Bg["systems"]})
