"""Turns results/*.json into LaTeX table fragments, figures and a flat numbers.json used for the manuscript."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, "..", "resubmission") if "JIT_RES" not in os.environ else os.path.join(RES, "paper"); TB = os.path.join(OUT, "tables"); FG = os.path.join(OUT, "figures")
os.makedirs(TB, exist_ok=True); os.makedirs(FG, exist_ok=True)
NICE = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "fiqa": "FiQA", "scidocs": "SciDocs", "trec-covid": "TREC-COVID", "quora": "Quora", "nq": "NQ", "msmarco": "MS MARCO"}
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "ink": "#0b0b0b", "mute": "#6f6e69", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["mute"], "axes.linewidth": .6, "xtick.color": C["mute"], "ytick.color": C["mute"],
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .5, "lines.linewidth": 1.5, "lines.markersize": 3.5, "legend.frameon": False, "legend.fontsize": 7, "axes.axisbelow": True, "pdf.fonttype": 42})
J = lambda *p: load_json(*p)
PER = {d: J(d, "perquery.json") for d in DATASETS}; META = {d: J(d, "meta.json") for d in DATASETS}
M = lambda d, s, m="ndcg": mean(PER[d][s][m])
N = {}  # flat numbers
def wr(name, s):
    open(os.path.join(TB, name + ".tex"), "w").write(s)
def save(fig, name):
    fig.savefig(os.path.join(FG, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FG, name + ".png"), dpi=170, bbox_inches="tight"); plt.close(fig)
def bold(vals, fmt="{:.3f}"):
    b = max(vals); return [("\\textbf{" + fmt.format(v) + "}" if fmt.format(v) == fmt.format(b) else fmt.format(v)) for v in vals]

# ---- T: collections (NFCorpus encode time comes from logs/e01_nfcorpus.log, 43.6 s; its rerun reused the vectors)
r = []
for d in DATASETS:
    m = META[d]
    r.append(f"{NICE[d]} & {m['n_docs']:,} & {m['n_q']} & {m['avg_doc_words']:.0f} & {m['avg_q_words']:.1f} & {m['bm25_bytes']/m['n_docs']:.0f} & {m['hnsw_bytes']/m['n_docs']:.0f} & {round(m.get('encode_corpus_s', 43.6))} & {m['hnsw_build_s']:.0f} \\\\")
wr("collections", "\n".join(r))

# ---- T: quality
S = [("bm25", "BM25"), ("bm25_rm3", "BM25+RM3"), ("dense_exact", "Dense"), ("rrf_bm25_dense", "RRF"), ("jit_k100", "JIT $K{=}100$"), ("jit_k200", "JIT $K{=}200$"), ("jit_k1000", "JIT $K{=}1000$")]
r = []
for d in DATASETS:
    n = bold([M(d, s) for s, _ in S]); a = bold([M(d, s, "recall100") for s, _ in S])
    r.append(f"{NICE[d]} & " + " & ".join(n) + " \\\\")
    r.append("\\quad Recall@100 & " + " & ".join(a) + " \\\\[1pt]")
macro = lambda s, m="ndcg": float(np.mean([M(d, s, m) for d in DATASETS]))
r.append("\\midrule Macro nDCG@10 & " + " & ".join(bold([macro(s) for s, _ in S])) + " \\\\")
r.append("Macro Recall@100 & " + " & ".join(bold([macro(s, "recall100") for s, _ in S])) + " \\\\")
wr("quality", "\n".join(r))
N["quality"] = {d: {s: {"ndcg": M(d, s), "recall100": M(d, s, "recall100")} for s in PER[d] if s in dict(S) or s.startswith("jit_k")} for d in DATASETS}
N["macro"] = {s: macro(s) for s in [x for x, _ in S] + [f"jit_k{k}" for k in K_GRID]}
N["macro_recall"] = {s: macro(s, "recall100") for s in [x for x, _ in S] + [f"jit_k{k}" for k in K_GRID]}

# ---- T: significance (Holm within each comparison family across datasets)
cmp = [("jit_k100", "dense_exact"), ("jit_k100", "bm25"), ("jit_k1000", "jit_k100"), ("rrf_bm25_dense", "jit_k100")]
sig = {}
for a, b in cmp:
    ts = {d: paired_test(PER[d][a]["ndcg"], PER[d][b]["ndcg"]) for d in DATASETS}
    ad = holm([ts[d]["p"] for d in DATASETS])
    for d, h in zip(DATASETS, ad):
        ts[d]["p_holm"] = h
    sig[f"{a}|{b}"] = ts
N["sig"] = sig
r = []
for d in DATASETS:
    cells = []
    for a, b in cmp:
        t = sig[f"{a}|{b}"][d]
        star = "$^{*}$" if t["p_holm"] < 0.05 and (t["lo"] > 0 or t["hi"] < 0) else ""
        cells.append(f"{t['diff']:+.3f}{star} [{t['lo']:+.3f}, {t['hi']:+.3f}]")
    r.append(f"{NICE[d]} & " + " & ".join(cells) + " \\\\")
wr("significance", "\n".join(r))

# ---- F: quality vs K
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.3)); ax = ax.ravel()
for i, d in enumerate(DATASETS):
    a = ax[i]; y = [M(d, f"jit_k{k}") for k in K_GRID]
    a.plot(K_GRID, y, "-o", color=C["blue"], label="JIT re-ranking")
    a.axhline(M(d, "dense_exact"), color=C["orange"], ls="--", lw=1.2, label="Dense (exact)")
    a.axhline(M(d, "bm25"), color=C["mute"], ls=":", lw=1.2, label="BM25")
    a.axhline(M(d, "rrf_bm25_dense"), color=C["aqua"], ls="-.", lw=1.2, label="RRF hybrid")
    a.set_xscale("log"); a.set_title(NICE[d], fontsize=8, loc="left")
    if i >= 3: a.set_xlabel("K")
    if i % 4 == 0: a.set_ylabel("nDCG@10")
h, l = ax[0].get_legend_handles_labels(); ax[7].axis("off"); ax[7].legend(h, l, loc="center")
fig.tight_layout(); save(fig, "quality_vs_k")

# ---- F/T: latency
L = J("latency.json"); cal = J("latcal.json"); rep = J("latrepeat.json")
KS = J("ksweep.json"); fig, a = plt.subplots(figsize=(3.4, 2.3)); fits = {}
for d, c in [("quora", C["blue"]), ("fiqa", C["orange"]), ("trec-covid", C["aqua"])]:
    ks = sorted(int(k) for k in KS[d]); y = [KS[d][str(k)]["p50"] for k in ks]; e = [KS[d][str(k)]["p95"] for k in ks]
    a.plot(ks, y, "-o", color=c, label=f"{NICE[d]} (p50)"); a.plot(ks, e, ":", color=c, lw=1)
    x = np.array(ks, float); A = np.vstack([np.ones_like(x), x]).T; c0, c1 = np.linalg.lstsq(A, np.array(y), rcond=None)[0]
    fits[d] = {"c0": float(c0), "c1": float(c1), "r2": float(1 - ((A @ [c0, c1] - y) ** 2).sum() / ((np.array(y) - np.mean(y)) ** 2).sum())}
a.axhline(1000, color=C["mute"], lw=.8, ls="--"); a.text(10, 1080, "1 s", color=C["mute"], fontsize=7)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Candidates encoded per query, K"); a.set_ylabel("End-to-end latency (ms)"); a.legend()
fig.tight_layout(); save(fig, "latency_vs_k")
N["ksweep"] = KS
wr("ksweep", "\n".join(f"{NICE[d]} & " + " & ".join(f"{KS[d][str(k)]['p50']:,.0f} / {KS[d][str(k)]['p95']:,.0f}" for k in [10, 50, 100, 200, 500, 1000, 2000]) + " \\\\" for d in ["quora", "fiqa", "trec-covid"]))
N["fits"] = fits; N["latcal"] = cal; N["k_sweep"] = {d: {k: {x: v[x] for x in ["p50", "p95", "first_ms", "qenc_ms", "enc_ms_per_doc"]} for k, v in L["k_sweep"][d].items()} for d in L["k_sweep"]}
N["latrepeat"] = {d: {k: {"p50s": [x["p50"] for x in v], "cv": float(np.std([x["p50"] for x in v]) / np.mean([x["p50"] for x in v]))} for k, v in rep[d].items()} for d in rep}
r = []
for d in DATASETS:
    c = cal[d]; r.append(f"{NICE[d]} & {c['avg_words']:.0f} & {c['200']['first_ms']:.1f} & {c['200']['qenc_ms']:.1f} & {c['200']['ms_per_doc']:.2f} & {c['200']['p50']:.0f} & {c['200']['p95']:.0f} \\\\")
wr("latcal", "\n".join(r))
N["scale"] = L["scale"]

# ---- T: optimisation levers (FiQA, K=200 unless noted) with quality from the model sweep
Mo = J("models.json")
mq = lambda key, K="100": float(np.mean([Mo[d][key][K] for d in Mo]))
B = "all-MiniLM-L6-v2@"
r = []
for s in [32, 64, 128, 256]:
    v = L["max_seq"][str(s)]; r.append(f"MiniLM, truncate to {s} tokens & {v['enc_ms_per_doc']:.1f} & {v['p50']:.0f} & {mq(B+str(s)):.3f} \\\\")
r.append("\\midrule")
for m, lab, key in [("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "multi-qa-MiniLM", "multi-qa-MiniLM-L6-cos-v1@256"), ("BAAI/bge-small-en-v1.5", "bge-small", "bge-small-en-v1.5@256"), ("sentence-transformers/all-mpnet-base-v2", "mpnet (batch 16)", "all-mpnet-base-v2@256")]:
    v = L["models"][m]["200"]; r.append(f"{lab} & {v['enc_ms_per_doc']:.1f} & {v['p50']:.0f} & {mq(key):.3f} \\\\")
r.append("\\midrule")
for b in [8, 32, 64, 128]:
    v = L["batch"][str(b)]; r.append(f"MiniLM, batch size {b} & {v['enc_ms_per_doc']:.1f} & {v['p50']:.0f} & -- \\\\")
r.append("\\midrule")
v = L["cpu"]["200"]; r.append(f"MiniLM on CPU, 32-bit float & {v['enc_ms_per_doc']:.1f} & {v['p50']:.0f} & -- \\\\")
v = L["cpu_q8"]["200"]; r.append(f"MiniLM on CPU, dynamic int8 & {v['enc_ms_per_doc']:.1f} & {v['p50']:.0f} & -- \\\\")
wr("levers", "\n".join(r))
N["models"] = Mo; N["levers"] = {k: {a: {x: y[x] for x in ["enc_ms_per_doc", "p50", "p95"]} for a, y in L[k].items()} for k in ["max_seq", "batch", "cpu", "cpu_q8"]}
N["model_lat"] = {m: {k: {x: L["models"][m][k][x] for x in ["enc_ms_per_doc", "p50", "p95"]} for k in L["models"][m]} for m in L["models"]}

# ---- T: cross-encoder, same query subset
r = []; N["ce"] = {}
for d in DATASETS:
    ce = J(d, "perquery_ce.json"); qs = list(ce["ce_k100"]["ndcg"])
    sub = lambda s: {q: PER[d][s]["ndcg"][q] for q in qs}
    t = paired_test(ce["ce_k100"]["ndcg"], sub("jit_k100")); t2 = paired_test(ce["ce_k100"]["ndcg"], sub("bm25"))
    N["ce"][d] = {"n": len(qs), "bm25": mean(sub("bm25")), "dense": mean(sub("dense_exact")), "jit100": mean(sub("jit_k100")), "ce50": mean(ce["ce_k50"]["ndcg"]), "ce100": mean(ce["ce_k100"]["ndcg"]), "ce200": mean(ce["ce_k200"]["ndcg"]), "ms_per_pair": ce["ms_per_pair"], "jit_ms_per_doc": cal[d]["200"]["ms_per_doc"], "test": t, "test_bm25": t2}
ps = holm([N["ce"][d]["test"]["p"] for d in DATASETS])
for d, h in zip(DATASETS, ps):
    x = N["ce"][d]; x["p_holm"] = h; st = "$^{*}$" if h < .05 and (x["test"]["lo"] > 0 or x["test"]["hi"] < 0) else ""
    b = bold([x["bm25"], x["dense"], x["jit100"], x["ce100"]])
    r.append(f"{NICE[d]} & {x['n']} & " + " & ".join(b) + f" & {x['test']['diff']:+.3f}{st} & {x['jit_ms_per_doc']:.1f} & {x['ms_per_pair']:.1f} \\\\")
wr("ce", "\n".join(r))

# ---- T: SPLADE
sp = J("splade.json"); r = []; N["splade"] = {}
for d in sp:
    ps_ = J(d, "perquery_splade.json"); m = META[d]
    x = {"splade": mean(ps_["splade"]["ndcg"]), "jit_splade200": mean(ps_["jit_splade_k200"]["ndcg"]), "jit_splade100": mean(ps_["jit_splade_k100"]["ndcg"]), "jit200": M(d, "jit_k200"), "jit100": M(d, "jit_k100"), "bm25": M(d, "bm25"), "rm3": M(d, "bm25_rm3"),
         "rec200": sp[d]["ceiling"]["200"], "rec200_bm25": m["ceiling"]["bm25"]["200"], "rec200_rm3": m["ceiling"]["rm3"]["200"], "ms_doc": sp[d]["ms_per_doc"], "minilm_ms_doc": cal[d]["200"]["ms_per_doc"], "bytes_doc": sp[d]["bytes"] / m["n_docs"], "bm25_bytes_doc": m["bm25_bytes"] / m["n_docs"], "query_ms": sp[d]["query_ms"]}
    x["test"] = paired_test(ps_["jit_splade_k200"]["ndcg"], PER[d]["jit_k200"]["ndcg"]); N["splade"][d] = x
    b = bold([x["bm25"], x["splade"]]); j = bold([x["jit200"], x["jit_splade200"]]); c = bold([x["rec200_bm25"], x["rec200_rm3"], x["rec200"]])
    r.append(f"{NICE[d]} & " + " & ".join(c) + " & " + " & ".join(b) + " & " + " & ".join(j) + f" & {x['ms_doc']:.0f} & {x['bytes_doc']/x['bm25_bytes_doc']:.1f}$\\times$ \\\\")
wr("splade", "\n".join(r))

# ---- F: lexical gap
G = J("lexgap.json"); edges = ["0 to .25", ".25 to .5", ".5 to .75", ".75 to 1"]
tot = [{"n": 0, **{str(k): 0 for k in K_GRID}, "dense": 0} for _ in edges]
for d in DATASETS:
    for j, b in enumerate(G[d]):
        tot[j]["n"] += b["n"]; tot[j]["dense"] += b["dense100"]
        for k in K_GRID: tot[j][str(k)] += b["hit"][str(k)]
mac = {}  # macro by dataset, using buckets with at least 30 judged pairs
for j in range(4):
    mac[j] = {}
    for k in [10, 100, 200, 1000, 2000]:
        v = [G[d][j]["hit"][str(k)] / G[d][j]["n"] for d in DATASETS if G[d][j]["n"] >= 30]; mac[j][k] = float(np.mean(v))
    mac[j]["dense"] = float(np.mean([G[d][j]["dense100"] / G[d][j]["n"] for d in DATASETS if G[d][j]["n"] >= 30])); mac[j]["n_ds"] = sum(G[d][j]["n"] >= 30 for d in DATASETS); mac[j]["pairs"] = sum(G[d][j]["n"] for d in DATASETS)
N["lexgap"] = mac; N["lexgap_raw"] = G
fig, a = plt.subplots(figsize=(3.4, 2.3))
for k, c in [(10, C["yellow"]), (100, C["blue"]), (200, C["aqua"]), (1000, C["orange"])]:
    a.plot(range(4), [mac[j][k] for j in range(4)], "-o", color=c, label=f"BM25 top-{k}")
a.plot(range(4), [mac[j]["dense"] for j in range(4)], "--s", color=C["ink"], label="Dense top-100")
a.set_xticks(range(4)); a.set_xticklabels(edges); a.set_xlabel(r"Measured lexical gap $\lambda(q,d)$"); a.set_ylabel("Relevant documents retrieved"); a.set_ylim(0, 1.02); a.legend(fontsize=6, loc="lower left")
fig.tight_layout(); save(fig, "lexgap")

# ---- F: streaming lag
St = J("stream.json"); N["stream"] = St
fig, a = plt.subplots(figsize=(3.4, 2.3))
rs = [100, 500, 1000, 5000]; a.plot(rs, [St["lucene_like"][f"{x}|1.0"]["p50_s"] for x in rs], "-o", color=C["ink"], label="Inverted index, 1 s refresh")
a.plot(rs, [St["lucene_like"][f"{x}|0.1"]["p50_s"] for x in rs], "--o", color=C["mute"], label="Inverted index, 0.1 s refresh")
for w, c in [("continuous", C["blue"]), ("1s", C["aqua"]), ("10s", C["orange"])]:
    rr = [25, 50, 100, 200, 500]; a.plot(rr, [St["hnsw"][f"{x}|{w}"]["p50_s"] for x in rr], "-s", color=c, label=f"Encode + HNSW, {w if w!='continuous' else 'continuous'}")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Offered ingest rate (documents/s)"); a.set_ylabel("Median visibility lag (s)"); a.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
fig.tight_layout(); save(fig, "visibility_lag")

# ---- T: deletion, filter
Er = J("erasure.json"); N["erasure"] = Er; N["canary"] = J("canary.json"); r = []
for p in ["1", "5", "10", "20", "50", "80"]:
    x = Er["hnsw"][p]; r.append(f"{p}\\% & {x['delete_us_per_doc']:.2f} & {x['ms_p50']:.2f} & {x['ms_p95']:.2f} & {x['recall10']:.3f} & {x['rebuild_s']:.1f} & {x['rebuilt']['ms_p50']:.2f} & {x['rebuilt']['recall10']:.3f} \\\\")
wr("deletion", "\n".join(r))
Fl = J("filter.json"); N["filter"] = Fl; r = []
for d in Fl:
    for s in ["1.0", "0.5", "0.1", "0.01"]:
        x = Fl[d][s]; b = bold([x["ndcg_exact"], x["ndcg_jit200"]])
        r.append(f"{NICE[d]} & {s} & {x['n_q']} & {b[0]} & {b[1]} & {x['hnsw_ms_p50']:.1f} & {x['bm25_ms_p50']:.1f} & {x['hnsw_recall10_vs_exact']:.3f} \\\\")
wr("filter", "\n".join(r))

# ---- F: cache
Ca = J("cache.json"); Co = J("cacheonce.json"); N["cache"] = {d: {k: v for k, v in Ca[d].items() if k != "replay"} for d in Ca}; N["cache_replay"] = {d: Ca[d]["replay"] for d in Ca}; N["cacheonce"] = Co; N["cachelive"] = J("cachelive.json")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
fr = [0.01, 0.05, 0.1, 0.25]
for a, d in zip(ax, ["fiqa", "scidocs", "quora"]):
    a.plot(fr, [Co[d][f"200|{x}"] for x in fr], "-o", color=C["ink"], label="No repeated queries")
    for s, c in [("0.0", C["aqua"]), ("0.8", C["yellow"]), ("1.0", C["blue"]), ("1.2", C["orange"])]:
        a.plot(fr, [Ca[d]["replay"][f"{s}|200|{x}"] for x in fr], "-o", color=c, label=f"Query stream Zipf s={s}" if s != "0.0" else "Query stream uniform")
    a.set_xscale("log"); a.set_title(NICE[d], fontsize=8, loc="left"); a.set_xlabel("Cache size (share of corpus)")
ax[0].set_ylabel("Embedding hit rate, K=200"); ax[2].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1))
fig.tight_layout(); save(fig, "cache")

# ---- F: backlog
Bl = J("backlog.json"); N["backlog"] = Bl
fr = ["0.0", "0.01", "0.05", "0.1", "0.25", "0.5"]; xs = [float(x) for x in fr]
fig, a = plt.subplots(figsize=(3.4, 2.3))
ann = [np.mean([Bl[d]["uniform"][x]["ann"] for d in DATASETS]) for x in fr]; hy = [np.mean([Bl[d]["uniform"][x]["hybrid"] for d in DATASETS]) for x in fr]
a.plot(xs, ann, "-o", color=C["orange"], label="ANN only (backlog invisible)"); a.plot(xs, hy, "-o", color=C["blue"], label="ANN + query-time encoding of backlog")
a.set_xlabel("Share of corpus waiting for the encoder"); a.set_ylabel("Macro nDCG@10"); a.legend()
fig.tight_layout(); save(fig, "backlog")
r = []
for d in DATASETS:
    u, v = Bl[d]["uniform"]["0.1"], Bl[d]["recent"]
    r.append(f"{NICE[d]} & {u['ann']:.3f} & {u['hybrid']:.3f} & {u['encoded_per_q']:.0f} & {v['ann']:.3f} & {v['hybrid']:.3f} & {v['encoded_per_q']:.1f} \\\\")
wr("backlog", "\n".join(r))
N["visibility"] = J("visibility.json")

# ---- adaptive + controller
Ad = J("adaptive.json"); Y = np.load(os.path.join(RES, "adaptive_Y.npy")); names = np.load(os.path.join(RES, "adaptive_names.npy")); GRID = Ad["grid"]
mac_fixed = {k: float(np.mean([Ad["fixed"][d][str(k)] for d in DATASETS])) for k in GRID}; mac_or = float(np.mean([Ad["oracle"][d]["ndcg"] for d in DATASETS])); mac_ork = float(np.mean([Ad["oracle"][d]["meanK"] for d in DATASETS]))
N["adaptive"] = {"fixed": mac_fixed, "oracle": mac_or, "oracle_k": mac_ork, "lexical": {}, "probe": {}}
fig, a = plt.subplots(figsize=(3.4, 2.3))
a.plot(GRID, [mac_fixed[k] for k in GRID], "-o", color=C["ink"], label="Fixed K")
for lbl, c, nm in [("lexical", C["blue"], "lexical"), ("lexical+probe", C["orange"], "probe")]:
    pts = []
    for dl in ["-0.75", "-0.5", "-0.25", "0", "0.25", "0.5", "0.75"]:
        key = dl if dl in Ad[lbl] else dl + ".0"
        pts.append((float(np.mean([Ad[lbl][key][d]["meanK"] for d in DATASETS])), float(np.mean([Ad[lbl][key][d]["ndcg"] for d in DATASETS]))))
    N["adaptive"][nm] = pts
    a.plot(*zip(*pts), "--s", color=c, label=f"Predicted K ({'lexical' if nm=='lexical' else 'lexical + probe'})")
a.plot([mac_ork], [mac_or], "*", color=C["aqua"], ms=9, label="Oracle per query")
a.set_xscale("log"); a.set_xlabel("Mean K per query"); a.set_ylabel("Macro nDCG@10"); a.set_ylim(0.495, 0.6); a.legend(fontsize=6, loc="upper right")
fig.tight_layout(); save(fig, "adaptive")
Ct = J("controller.json"); N["controller"] = Ct
r = []
for B_ in ["100", "250", "500", "1000", "2000", "5000"]:
    x = Ct["lexical"][B_]; pr = Ct["lexical+probe"][B_]
    fix = np.mean([x[d]["fix_ndcg"] for d in DATASETS]); bf = np.mean([x[d]["best_fixed_ndcg"] for d in DATASETS]); ctl = np.mean([x[d]["ndcg"] for d in DATASETS]); ctp = np.mean([pr[d]["ndcg"] for d in DATASETS]); o = np.mean([x[d]["oracle"] for d in DATASETS])
    k = np.mean([x[d]["meanK"] for d in DATASETS]); lt = np.mean([x[d]["lat"] for d in DATASETS]); kf = np.mean([x[d]["fix_k"] for d in DATASETS]); kb = np.mean([x[d]["best_fixed_k"] for d in DATASETS])
    sigs = sum(x[d]["p_best_fixed"] < .05 and x[d]["ndcg"] > x[d]["best_fixed_ndcg"] for d in DATASETS); neg = sum(x[d]["p_best_fixed"] < .05 and x[d]["ndcg"] < x[d]["best_fixed_ndcg"] for d in DATASETS)
    r.append(f"{int(B_):,} & {fix:.3f} ({kf:.0f}) & {bf:.3f} ({kb:.0f}) & {ctl:.3f} ({k:.0f}) & {ctp:.3f} & {o:.3f} & {sigs}/{neg} \\\\")
wr("controller", "\n".join(r))

# ---- QLF
HN = {d: META[d]["hnsw_sweep"]["100"]["ms_p50"] for d in DATASETS}
def lat(d, s):
    c = cal[d]["200"]; t = c["qenc_ms"]; f = c["first_ms"]; e = c["ms_per_doc"]
    if s == "bm25": return f
    if s == "dense": return t + HN[d]
    if s == "rrf": return t + HN[d] + f
    if s.startswith("jit"): return f + t + e * int(s[3:])
    if s.startswith("ce"): return f + N["ce"][d]["ms_per_pair"] * int(s[2:])
SYS = {"bm25": ("BM25", "inv"), "dense": ("Stream", "ann"), "rrf": ("RRF", "both"), "jit50": ("JIT-50", "inv"), "jit100": ("JIT-100", "inv"), "jit200": ("JIT-200", "inv"), "ce100": ("CE-100", "inv"), "static": ("Static-24h", "static")}
def qual(d, s):
    ce = J(d, "perquery_ce.json"); qs = list(ce["ce_k100"]["ndcg"])
    key = {"bm25": "bm25", "dense": "dense_exact", "static": "dense_exact", "rrf": "rrf_bm25_dense", "jit50": "jit_k50", "jit100": "jit_k100", "jit200": "jit_k200"}.get(s)
    return mean(ce["ce_k100"]["ndcg"]) if s == "ce100" else float(np.mean([PER[d][key]["ndcg"][q] for q in qs]))
TAU, LMAX = 60.0, 2000.0
def lag(kind, rate):
    inv = St["lucene_like"]["500|1.0"]["p50_s"]; ann = St["hnsw"][f"{rate}|continuous"]["p50_s"]
    return {"inv": inv, "ann": ann, "both": inv, "static": 12 * 3600}[kind]  # RRF sees a new document through BM25 straight away
WS = {"Balanced": (.5, .25, .25), "Quality first": (.8, .1, .1), "Interactive": (.4, .5, .1), "Freshness first": (.3, .1, .6)}
qlf = {}; r = []
for rate in [50, 500]:
    for w, (wq, wl, wf) in WS.items():
        U = {}
        for s, (nm, kind) in SYS.items():
            sl = "dense" if s == "static" else s
            U[s] = float(np.mean([wq * qual(d, s) + wl * (1 - min(1, lat(d, sl) / LMAX)) + wf * np.exp(-lag(kind, rate) / TAU) for d in DATASETS]))
        qlf[f"{rate}|{w}"] = U
        wins = {}
        for d in DATASETS:
            u = {s: wq * qual(d, s) + wl * (1 - min(1, lat(d, "dense" if s == "static" else s) / LMAX)) + wf * np.exp(-lag(SYS[s][1], rate) / TAU) for s in SYS}
            b = max(u, key=u.get); wins[b] = wins.get(b, 0) + 1
        qlf[f"{rate}|{w}|wins"] = wins
        cells = bold([U[s] for s in SYS])
        r.append(f"{rate} & {w} ({wq},{wl},{wf}) & " + " & ".join(cells) + " \\\\")
    if rate == 50: r.append("\\midrule")
wr("qlf", "\n".join(r))
N["qlf"] = qlf; N["qlf_lat"] = {d: {s: lat(d, s) for s in SYS if s != "static"} for d in DATASETS}; N["hn"] = HN
# Sensitivity: share of weight simplex where each system wins (macro), both load conditions
for rate in [50, 500]:
    grid = [(a / 20, b / 20, 1 - a / 20 - b / 20) for a in range(21) for b in range(21 - a)]
    Qm = {s: np.mean([qual(d, s) for d in DATASETS]) for s in SYS}; Lm = {s: np.mean([1 - min(1, lat(d, "dense" if s == "static" else s) / LMAX) for d in DATASETS]) for s in SYS}; Fm = {s: np.exp(-lag(SYS[s][1], rate) / TAU) for s in SYS}
    w = {}
    for g in grid:
        b = max(SYS, key=lambda s: g[0] * Qm[s] + g[1] * Lm[s] + g[2] * Fm[s]); w[b] = w.get(b, 0) + 1 / len(grid)
    N[f"qlf_simplex_{rate}"] = w; N[f"qlf_comp_{rate}"] = {s: (Qm[s], Lm[s], Fm[s]) for s in SYS}
# Model/K selection per collection by the same utility (deployment-level "model" dial)
sel = {}
for d in Mo:
    best = {}
    for wname, (wq, wl, wf) in WS.items():
        u = {}
        for key, v in Mo[d].items():
            for k in ["50", "100", "200", "500", "1000"]:
                u[f"{key}|{k}"] = wq * v[k] + wl * (1 - min(1, (cal[d]["200"]["first_ms"] + cal[d]["200"]["qenc_ms"] + v["ms_per_doc"] * int(k)) / LMAX))
        b = max(u, key=u.get); best[wname] = (b, u[b])
    sel[d] = best
N["model_select"] = sel

# ---- Break-even
fiq = META["fiqa"]; ce_ms = cal["fiqa"]["200"]["ms_per_doc"]; PG = 1.0; PM = 4.0; bytes_doc = fiq["hnsw_bytes"] / fiq["n_docs"]
ce_usd = ce_ms / 3.6e6 * PG
month = 30 * 86400
def qstar(u, K, h, Ndocs, pm=PM):
    m = Ndocs * bytes_doc / 1e9 * pm / month
    return (u * ce_usd + m) / (K * (1 - h) * ce_usd)
N["breakeven"] = {"ce_ms": ce_ms, "ce_usd": ce_usd, "bytes_doc": bytes_doc, "mem_doc_equiv_per_s_1M": 1e6 * bytes_doc / 1e9 * PM / month / ce_usd, "ratio": {f"{K}|{h}": 1 / (K * (1 - h)) for K in [50, 100, 200, 1000] for h in [0, .5, .9]},
                  "reembed_fiqa_s": fiq.get("encode_corpus_s"), "touched": {d: Ca[d]["distinct"] / Ca[d]["n_docs"] for d in Ca}}
fig, a = plt.subplots(figsize=(3.4, 2.5)); U_ = np.logspace(-2, 3, 40)
for (K, h, c) in [(50, .5, C["aqua"]), (100, .5, C["blue"]), (100, 0, C["orange"]), (1000, .5, C["yellow"])]:
    a.plot(U_, [qstar(u, K, h, 1e6) for u in U_], color=c, label=f"K={K}, hit rate h={h}")
a.plot(U_, U_, ":", color=C["mute"], lw=1, label="1 query per write"); a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("Writes per second, u"); a.set_ylabel("Break-even queries per second, Q*"); a.legend(fontsize=6)
a.text(0.97, 0.06, "lazy cheaper below a line", transform=a.transAxes, ha="right", fontsize=6.5, color=C["mute"])
fig.tight_layout(); save(fig, "breakeven")

# ---- hyperparameters
Gd = J("bm25grid.json"); N["grid"] = {d: {"base": Gd[d]["1.2/0.75"], "jit": [min(x["jit200_ndcg"] for x in Gd[d].values()), max(x["jit200_ndcg"] for x in Gd[d].values())], "bm25": [min(x["bm25_ndcg"] for x in Gd[d].values()), max(x["bm25_ndcg"] for x in Gd[d].values())], "rec": [min(x["recall200"] for x in Gd[d].values()), max(x["recall200"] for x in Gd[d].values())]} for d in Gd}
N["meta"] = {d: {k: v for k, v in META[d].items()} for d in DATASETS}
json.dump(N, open(os.path.join(RES, "numbers.json"), "w"), indent=1, default=float)
print("ok")
