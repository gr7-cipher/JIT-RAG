"""LaTeX tables and figures for the v3 manuscript, written to resubmission/tables_v3 and resubmission/figures_v3 (or $JIT_OUT)."""
import os, re, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_json, RES, DATASETS, ROOT
OUT = os.environ.get("JIT_OUT", os.path.join(ROOT, "..", "resubmission")); T = os.path.join(OUT, "tables_v3"); F = os.path.join(OUT, "figures_v3"); os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "mute": "#6f6e69", "grid": "#e4e3df", "red": "#c23b3b"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["mute"], "axes.linewidth": .6, "xtick.color": C["mute"], "ytick.color": C["mute"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .5, "legend.frameon": False, "axes.axisbelow": True, "pdf.fonttype": 42})
NICE = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "fiqa": "FiQA", "scidocs": "SciDocs", "trec-covid": "TREC-COVID", "quora": "Quora", "webis-touche2020": "Touch\\'e-2020", "cqa-webmasters": "CQA Webmasters", "cqa-android": "CQA Android", "cqa-wordpress": "CQA WordPress", "fiqa_fresh": "FiQA (fresh)", "nfcorpus_fresh": "NFCorpus (fresh)"}
EXT = ["webis-touche2020", "cqa-webmasters", "cqa-android", "cqa-wordpress", "fiqa_fresh", "nfcorpus_fresh"]
R = load_json("e20_stats.json"); M = R["macro"]; L = R["lag"]; K = R["cmp"]; B = R["budget"]
def wr(n, s): open(os.path.join(T, n + ".tex"), "w").write(s.rstrip() + "\n")
def bold(vals, f="%.3f"):
    b = max(v for v in vals if v is not None); return [("--" if v is None else (("\\textbf{%s}" % (f % v)) if abs(v - b) < 1e-9 else f % v)) for v in vals]
# collections
rows = open(os.path.join(OUT, "tables", "collections.tex")).read().strip().split("\n"); r = []
for line in rows:
    c = [x.strip().rstrip("\\").strip() for x in line.split("&")]; name = c[0]; key = [k for k, v in NICE.items() if v == name][0]
    r.append(f"{name} & {c[1]} & {c[2]} & {R['nq'][key]} & {c[3]} & {c[4]} & {c[5]} \\\\")
r.append("\\midrule")
for s in EXT[:4]: r.append(f"{NICE[s]} & {R['N'][s]:,} & {R['nq'][s]} & {R['nq'][s]} & -- & -- & -- \\\\")
wr("collections", "\n".join(r))
# complete index, per collection
S = ["full", "full_a9", "h100", "jitaf", "rrf", "dense", "bm25"]; r = []
for s in DATASETS: r.append(NICE[s] + " & " + " & ".join(bold([M[k + "|seven"]["per"][s]["ndcg"] for k in S])) + " \\\\")
r += ["\\midrule", "Macro nDCG@10 & " + " & ".join(bold([M[k + "|seven"]["ndcg"] for k in S], "%.4f")) + " \\\\", "Macro Recall@10 & " + " & ".join(bold([M[k + "|seven"]["r10"] for k in S])) + " \\\\", "Macro Recall@100 & " + " & ".join(bold([M[k + "|seven"]["r100"] for k in S])) + " \\\\", "Macro Judged@10 & " + " & ".join(bold([M[k + "|seven"]["j10"] for k in S])) + " \\\\", "\\midrule"]
for s in EXT: r.append(NICE[s] + " & " + " & ".join(bold([M[k + "|ext"]["per"][s]["ndcg"] for k in S])) + " \\\\")
r += ["\\midrule", "Macro nDCG@10 & " + " & ".join(bold([M[k + "|ext"]["ndcg"] for k in S], "%.4f")) + " \\\\", "Macro Recall@10 & " + " & ".join(bold([M[k + "|ext"]["r10"] for k in S])) + " \\\\", "Macro Recall@100 & " + " & ".join(bold([M[k + "|ext"]["r100"] for k in S])) + " \\\\", "Macro Judged@10 & " + " & ".join(bold([M[k + "|ext"]["j10"] for k in S])) + " \\\\"]
wr("complete", "\n".join(r))
# ablation
A = [("JIT-AF+ANN, all parts ($\\alpha{=}0.8$, $\\gamma{=}0.3$)", "full"), ("\\quad without the $\\gamma$ signal", "noGamma"), ("\\quad without expansion rounds, $\\gamma$ kept", "noExp_gamma"), ("\\quad without expansion and $\\gamma$", "noExp"), ("\\quad without ANN", "noANN"), ("\\quad with $\\alpha{=}0.9$ and no $\\gamma$", "noGamma_a9"), ("Indexed fusion, $K{=}100$ ($\\alpha{=}0.9$)", "h100"), ("\\quad with $\\alpha{=}0.8$", "h100_a8"), ("\\quad with $\\alpha{=}0.8$ and the $\\gamma$ signal", "h100_gamma"), ("Indexed fusion, $K{=}50$", "h50"), ("JIT-AF ($\\alpha{=}0.95$, no ANN)", "jitaf"), ("Sparse-first fusion, $K{=}100$", "jitcc100"), ("Sparse-first re-ranking, $K{=}100$", "jit100")]; r = []
for lab, k in A:
    a, b = M[k + "|seven"], M[k + "|ext"]; r.append(f"{lab} & {a['ndcg']:.4f} & {a['r10']:.3f} & {a['r100']:.3f} & {a['pool']:.0f} & {b['ndcg']:.4f} & {b['r10']:.3f} & {b['r100']:.3f} \\\\")
wr("ablation", "\n".join(r))
# backlog
def lagrows(nm):
    r = []; g0 = {"lazy50": "h50", "lazy100": "h100", "afh": "full", "sa100": "h100", "rrfsa": "rrf"}
    v = [M[g0[x] + "|" + nm]["ndcg"] for x in ["lazy50", "lazy100", "afh", "sa100", "rrfsa"]]; b = bold(v, "%.4f"); r.append(f"0\\% & 0 & {b[0]} & 0 & {b[1]} & {b[2]} & {b[3]} & {b[4]} \\\\")
    for f, lab in [("0.01", "1\\%"), ("0.05", "5\\%"), ("0.1", "10\\%"), ("0.25", "25\\%"), ("0.5", "50\\%"), ("recent", "Half of relevant")]:
        v = [L[f"{x}|{f}|{nm}"]["seed13"] for x in ["lazy50", "lazy100", "afh", "sa100", "rrfsa"]]; b = bold(v, "%.4f"); e50 = L[f"lazy50|{f}|{nm}"]["enc"]; e100 = L[f"lazy100|{f}|{nm}"]["enc"]; r.append(f"{lab} & {e50:.1f} & {b[0]} & {e100:.1f} & {b[1]} & {b[2]} & {b[3]} & {b[4]} \\\\")
    return r
wr("backlog", "\n".join(lagrows("seven") + ["\\midrule"] + lagrows("ext")))
# tests
def row(lab, key):
    x = K[key]; return f"{lab} & {x['diff']:+.4f} & [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] & [{x['ci90'][0]:+.4f}, {x['ci90'][1]:+.4f}] & {'yes' if x['tost_05'] else 'no'} & {'yes' if x['tost_002'] else 'no'} & {x['wins']}/{x['losses']} \\\\"
Q = [("JIT-AF+ANN $-$ indexed fusion, complete", "full-h100|ndcg|"), ("JIT-AF+ANN $-$ lazy fusion 50, 10\\%", "afh-lazy50|ndcg|f0.1|"), ("JIT-AF+ANN $-$ lazy fusion 100, 10\\%", "afh-lazy100|ndcg|f0.1|"), ("JIT-AF+ANN $-$ lazy fusion 50, 50\\%", "afh-lazy50|ndcg|f0.5|"), ("JIT-AF+ANN $-$ lazy fusion 100, 50\\%", "afh-lazy100|ndcg|f0.5|"), ("Lazy fusion 100 $-$ lazy fusion 50, 10\\%", "lazy100-lazy50|ndcg|f0.1|"), ("Lazy fusion 50 $-$ signal-aware, 10\\%", "lazy50-sa100|ndcg|f0.1|"), ("Lazy fusion 50 $-$ signal-aware, 50\\%", "lazy50-sa100|ndcg|f0.5|")]; r = []
r.append("\\multicolumn{7}{l}{\\textit{Seven collections, held-out queries}} \\\\")
for lab, k in Q: r.append(row(lab, (k + "seven") if k.endswith("|") and k.count("|") >= 2 else k))
r.append("\\midrule"); r.append("\\multicolumn{7}{l}{\\textit{Six external query sets}} \\\\")
for lab, k in Q: r.append(row(lab, k + "ext"))
wr("tests", "\n".join(r))
# budget
r = []
for f, ms in [("0.1", [5, 10, 20]), ("0.5", [5, 10, 20, 50]), ("1.0", [5, 10, 20, 50])]:
    for m in ms:
        v = [B[f"{s}|f{f}|m{m}|seven"]["ndcg"] for s in ["afh", "lazy50", "lazy100"]]; w = [B[f"{s}|f{f}|m{m}|ext"]["ndcg"] for s in ["afh", "lazy50", "lazy100"]]; b = bold(v, "%.4f"); c = bold(w, "%.4f"); e = [B[f"{s}|f{f}|m{m}|seven"]["enc"] for s in ["afh", "lazy50", "lazy100"]]
        r.append(f"{float(f) * 100:.0f}\\% & {m} & {b[0]} & {b[1]} & {b[2]} & {c[0]} & {c[1]} & {c[2]} & {e[0]:.1f} & {e[1]:.1f} \\\\")
    r.append("\\midrule")
wr("budget", "\n".join(r[:-1]))
# write-back
r = []; W = R["writeback"]
for s in DATASETS:
    x = W[s]; n = R["N"][s]; r.append(f"{NICE[s]} & {n:,} & {x['afh|f1.0'][0]:,} & {x['afh|f1.0'][1]:,} & {100 * x['afh|f1.0'][0] / n:.1f}\\% & {x['lazy50|f1.0'][0]:,} & {x['lazy50|f1.0'][1]:,} & {100 * x['lazy50|f1.0'][0] / n:.1f}\\% \\\\")
wr("writeback", "\n".join(r))
# overload
O = load_json("e19_overload.json"); r = []
for k in O:
    if not k.endswith("|repl"): continue
    a = O[k]; b = O[k.replace("|repl", "|norepl")]; m = "--" if a["m"] is None else a["m"]; r.append(f"{a['u']} & {a['mu']} & {a['Q']} & {m} & {'yes' if a['binds_corrected'] else 'no'} & {a['f_inf']:.3f} & {a['model_K100']:.3f} & {a['f_end']:.3f} & {b['f_end']:.3f} & {a['Keff']:.0f} & {b['Keff']:.0f} \\\\")
wr("overload", "\n".join(r))
# idle latency
if os.path.exists(os.path.join(RES, "e22_idle.json")):
    I = load_json("e22_idle.json"); Lh = load_json("e10_latency_hyb.json"); A14 = load_json("e14_latency.json"); r = []
    fm = lambda x: "{:,.0f} / {:,.0f}".format(x["p50"], x["p95"])
    for d in DATASETS: r.append(NICE[d] + " & " + " & ".join([fm(Lh[d]), fm(A14[d]["f0.0"]), fm(I[d]["lazy50_f0.1"]), fm(I[d]["lazy100_f0.1"]), fm(A14[d]["f0.1"]), fm(I[d]["lazy50_f0.5"]), fm(I[d]["lazy100_f0.5"])]) + " \\\\")
    wr("idle", "\n".join(r))
# cost
Cc = load_json("e21_cost.json"); r = []; lab = [("eager", "Ingest-time encoding"), ("lazy_fusion", "Lazy fusion with write-back"), ("jit_af_ann", "JIT-AF+ANN with write-back"), ("jit_af_wb", "JIT-AF, write-back, 10\\% retrieved"), ("jit_nowb", "Sparse-first re-ranking, no write-back")]
for k, l in lab: r.append(f"{l} & {Cc['base_utilised'][k]:,.0f} & {Cc['base_alwayson'][k]:,.0f} & {Cc['sens']['pa3']['alwayson'][k]:,.0f} & {Cc['sens']['N1e7']['alwayson'][k]:,.0f} & {Cc['sens']['c20x']['utilised'][k]:,.0f} \\\\")
wr("cost", "\n".join(r))
# figures: backlog
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.3)); FS = [0, 1, 5, 10, 25, 50]; fk = ["0.01", "0.05", "0.1", "0.25", "0.5"]
for a, nm, tt in [(ax[0], "seven", "Seven collections, held-out queries"), (ax[1], "ext", "Six external query sets")]:
    g0 = {"lazy50": "h50", "lazy100": "h100", "afh": "full", "sa100": "h100", "rrfsa": "rrf"}
    for s, lb, col, mk, ls in [("afh", "JIT-AF+ANN", C["blue"], "o", "-"), ("lazy50", "Lazy fusion, $K$=50", C["aqua"], "D", "-"), ("lazy100", "Lazy fusion, $K$=100", C["ink"], "x", "--"), ("sa100", "Indexed fusion, signal-aware", C["orange"], "s", "-"), ("rrfsa", "RRF, signal-aware", C["red"], "^", "-")]:
        a.plot(FS, [M[g0[s] + "|" + nm]["ndcg"]] + [L[f"{s}|{f}|{nm}"]["seed13"] for f in fk], ls, marker=mk, ms=3, lw=1.1, color=col, label=lb)
    a.set_title(tt, fontsize=8); a.set_xlabel("Corpus waiting for the encoder (%)"); a.set_ylabel("Macro nDCG@10")
ax[0].legend(fontsize=6, loc="lower left"); fig.tight_layout(); fig.savefig(os.path.join(F, "backlog.pdf"))
# figure: visibility lag, legend clear of the axis label
St = load_json("stream.json"); fig, a = plt.subplots(figsize=(4.6, 2.4)); rs = [100, 500, 1000, 5000]; a.plot(rs, [St["lucene_like"][f"{x}|1.0"]["p50_s"] for x in rs], "-o", ms=3, color=C["ink"], label="Inverted index, 1 s refresh"); a.plot(rs, [St["lucene_like"][f"{x}|0.1"]["p50_s"] for x in rs], "--o", ms=3, color=C["mute"], label="Inverted index, 0.1 s refresh")
for w, c, lb in [("continuous", C["blue"], "continuous"), ("1s", C["aqua"], "1 s batches"), ("10s", C["orange"], "10 s batches")]:
    rr = [25, 50, 100, 200, 500]; a.plot(rr, [St["hnsw"][f"{x}|{w}"]["p50_s"] for x in rr], "-s", ms=3, color=c, label=f"Encode + HNSW, {lb}")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Offered ingest rate (documents/s)"); a.set_ylabel("Median visibility lag (s)"); a.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5)); fig.tight_layout(); fig.savefig(os.path.join(F, "visibility_lag.pdf"))
# figure: live
if os.path.exists(os.path.join(RES, "e18_live.json")):
    Lv = load_json("e18_live.json")["runs"]; Lv = [x for x in Lv if x["repeat"] < 90]
    if Lv:
        fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.4), sharey=True)
        for a, u, tt in [(ax[0], 20, "20 documents/s, encoder keeps up"), (ax[1], 100, "100 documents/s, encoder saturated")]:
            for s, lb, col, mk in [("sa100", "Indexed fusion, signal-aware", C["orange"], "s"), ("lazy50m10", "Lazy fusion, $K$=50, $m$=10", C["ink"], "x"), ("lazy50", "Lazy fusion, $K$=50", C["aqua"], "D"), ("afh", "JIT-AF+ANN", C["blue"], "o")]:
                pts = sorted((x["Q"], x["p95"], x["repeat"]) for x in Lv if x["sys"] == s and x["u"] == u and x["policy"] == "priority")
                if not pts: continue
                qs = sorted({p[0] for p in pts}); a.plot(qs, [np.mean([p[1] for p in pts if p[0] == q]) for q in qs], "-", marker=mk, ms=3, lw=1.1, color=col, label=lb); a.scatter([p[0] for p in pts], [p[1] for p in pts], s=5, color=col, alpha=.5)
            a.set_xscale("log"); a.set_yscale("log"); a.set_title(tt, fontsize=8); a.set_xlabel("Offered queries per second"); a.axhline(1000, color=C["mute"], lw=.5, ls=":")
        ax[0].set_ylabel("p95 latency (ms)"); ax[1].legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.18)); fig.tight_layout(); fig.savefig(os.path.join(F, "live.pdf"))
        def L1(s, u, Q, p="priority"):
            return sorted([x for x in Lv if x["sys"] == s and x["u"] == u and x["Q"] == Q and x["policy"] == p], key=lambda x: x["repeat"])
        def cell(x):
            return "{:,.0f}/{:,.0f}/{:,.0f}".format(x["p50"], x["p95"], x["p99"])
        def lrow(lab, s, u, Q, p="priority"):
            xs = L1(s, u, Q, p)
            if not xs: return None
            c = [cell(x) for x in xs] + ["--"] * (2 - len(xs)); return f"{lab} & {u} & {Q} & {c[0]} & {c[1]} & {np.mean([x['f_end'] for x in xs]):.3f} & {np.mean([x['enc_per_q'] for x in xs]):.1f} & {np.mean([x['bg_rate'] for x in xs]):.0f} & {np.mean([x['q_enc_rate'] for x in xs]):.0f} & {sum(x['n_done'] for x in xs)}/{sum(x['n_done'] + x['n_cens'] for x in xs)} \\\\"
        r = []
        for u in [20, 100]:
            for lab, s in [("Signal-aware IF", "sa100"), ("Lazy fusion, $K$=50", "lazy50"), ("Lazy, $K$=50, $m$=10", "lazy50m10"), ("JIT-AF+ANN", "afh")]:
                for Q in ([1, 10] if u == 20 else [1, 2, 5, 10]):
                    y = lrow(lab, s, u, Q)
                    if y: r.append(y)
                r.append("\\addlinespace[1pt]")
            r.append("\\midrule")
        for lab, s in [("Signal-aware IF", "sa100"), ("Lazy fusion, $K$=50", "lazy50"), ("JIT-AF+ANN", "afh")]:
            y = lrow(lab + ", FIFO", s, 100, 5, "fifo"); 
            if y: r.append(y)
            y = lrow(lab, s, 200, 5)
            if y: r.append(y)
        wr("live", "\n".join(r))
print("tables written")
