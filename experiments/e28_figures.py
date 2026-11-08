"""E28: additional result figures for the v3 manuscript, drawn from stored results only.

budget_curves.pdf   nDCG@10 against encoding budget m at three backlog shares (e20_stats.json)
effects.pdf         per-collection paired differences in nDCG@10 at a 10% backlog, with the equivalence margin (e17/*.json, e20_stats.json)
model_fit.pdf       measured backlog share against the shared- and separate-capacity predictions (e25_shared.json)
latency_cdf.pdf     distribution of live query latency at u=100 (e18_live.json)
coverage.pdf        corpus share touched against retrievals per document, with the uniform baseline (e26_coverage.json)
trajectory.pdf      backlog share over time from the closed forms of Section 4.1, showing when the budget binds

Output goes to resubmission/figures_v3, or to $JIT_OUT/figures_v3.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_json, RES, DATASETS, ROOT, SEED

OUT = os.environ.get("JIT_OUT", os.path.join(ROOT, "..", "resubmission")); F = os.path.join(OUT, "figures_v3"); os.makedirs(F, exist_ok=True)
# same palette and rcParams as e21_tables.py; validated for colour-vision deficiency, aqua always carries a marker and label
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "mute": "#6f6e69", "grid": "#e4e3df", "red": "#c23b3b"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C["mute"], "axes.linewidth": .6, "xtick.color": C["mute"], "ytick.color": C["mute"],
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .5, "legend.frameon": False, "lines.linewidth": 1.4})
NICE = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "fiqa": "FiQA", "scidocs": "SciDocs", "trec-covid": "TREC-COVID", "quora": "Quora"}
SYS = {"lazy50": ("Lazy fusion, K=50", C["blue"], "o"), "lazy100": ("Lazy fusion, K=100", C["aqua"], "s"),
       "afh": ("JIT-AF+ANN", C["orange"], "^"), "sa100": ("Indexed fusion, signal-aware", C["mute"], "D")}
R = load_json("e20_stats.json"); out_numbers = {}
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(F, name)); plt.close(fig)

# 1. budget curves
B = R["budget"]; ms = [5, 10, 20, 50]; complete = R["lag"]["lazy50|0.01|seven"]["seed13"]
full_idx = R["macro"]["h100|seven"]["ndcg"] if "h100|seven" in R["macro"] else None
fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
for a, f, title in zip(ax, ["f0.1", "f0.5", "f1.0"], ["10% waiting", "50% waiting", "100% waiting"]):
    for s in ["lazy100", "afh", "lazy50"]:
        lab, col, mk = SYS[s]; y = [B[f"{s}|{f}|m{m}|seven"]["ndcg"] for m in ms]
        if s == "lazy50": a.plot(ms, y, "--", marker=mk, ms=5, mfc="white", mew=1, lw=1, color=col, label=lab, zorder=4)
        else: a.plot(ms, y, "-", marker=mk, ms=4, color=col, label=lab)
    if full_idx: a.axhline(full_idx, ls="--", lw=.8, color=C["ink"])
    a.set_xscale("log"); a.set_xticks(ms); a.set_xticklabels([str(m) for m in ms]); a.minorticks_off()
    a.set_title(title, fontsize=8); a.set_xlabel("Encoding budget m per query")
ax[0].set_ylabel("Macro nDCG@10"); h, l = ax[0].get_legend_handles_labels(); o = [l.index(x) for x in ["Lazy fusion, K=50", "Lazy fusion, K=100", "JIT-AF+ANN"]]; ax[0].legend([h[i] for i in o], [l[i] for i in o], fontsize=6, loc="lower right")
if full_idx: ax[2].annotate("complete index", xy=(5, full_idx), xytext=(0, 3), textcoords="offset points", fontsize=6, color=C["ink"])
save(fig, "budget_curves.pdf")

# 2. per-collection effects at a 10% backlog, seed 13 masks, bootstrap 90% intervals over queries
rng = np.random.default_rng(0); MARGIN = load_json("e17_margin.json")["delta_macro_ndcg10"]
pairs = [("lazy50", "sa100", "Lazy fusion K=50 minus\nsignal-aware indexed fusion"), ("afh", "lazy50", "JIT-AF+ANN minus\nlazy fusion K=50")]
eff = {}
for a_, b_, _ in pairs:
    for d in DATASETS:
        J = json.load(open(os.path.join(RES, "e17", f"{d}.json"))); S = J["sys"]
        qs = sorted(J["qids"]); perm = np.random.default_rng(SEED).permutation(len(qs)); pos = {q: i for i, q in enumerate(J["qids"])}
        idx = np.array([pos[qs[i]] for i in perm[len(qs) // 2:]])  # held-out half, as in e20_stats.half()
        x = (np.array(S[f"{a_}|f0.1|s13"]["ndcg"]) - np.array(S[f"{b_}|f0.1|s13"]["ndcg"]))[idx]
        assert abs(x.mean() - R["cmp"][f"{a_}-{b_}|ndcg|f0.1|seven"]["per"][d]["diff"]) < 1e-9, (a_, b_, d)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        eff[(a_, b_, d)] = (float(x.mean()), float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
order = list(DATASETS)[::-1]
for a, (a_, b_, title) in zip(ax, pairs):
    a.axvspan(-MARGIN, MARGIN, color=C["grid"], alpha=.8, lw=0); a.axvline(0, color=C["ink"], lw=.6)
    for i, d in enumerate(order):
        m, lo, hi = eff[(a_, b_, d)]; a.plot([lo, hi], [i, i], color=C["blue"], lw=1.2); a.plot(m, i, "o", ms=4, color=C["blue"])
    c = R["cmp"][f"{a_}-{b_}|ndcg|f0.1|seven"]; lo, hi = c["ci90"]
    a.plot([lo, hi], [-1.2, -1.2], color=C["ink"], lw=1.6); a.plot(c["diff"], -1.2, "D", ms=4.5, color=C["ink"])
    a.set_title(title, fontsize=7.5); a.set_xlabel("Difference in nDCG@10")
    out_numbers[f"{a_}-{b_}"] = {"macro": c["diff"], "ci90": c["ci90"], "per": {d: eff[(a_, b_, d)] for d in DATASETS}}
ax[0].set_yticks(list(range(len(order))) + [-1.2]); ax[0].set_yticklabels([NICE[d] for d in order] + ["Macro (90% CI)"])
ax[0].set_ylim(-2, len(order) - .4)
ax[1].annotate(f"margin $\\pm${MARGIN}", xy=(MARGIN, len(order) - .6), xytext=(3, 0), textcoords="offset points", fontsize=6, color=C["mute"], va="center")
save(fig, "effects.pdf")

# 3. model fit on the live runs
M = load_json("e25_shared.json")
fig, a = plt.subplots(figsize=(3.3, 3.0))
sh = np.array([[r["f_pred"], r["f_meas"]] for r in M["rows"]]); sp = np.array([[r["f_sep"], r["f_meas"]] for r in M["sep"]])
lim = [0, max(sh.max(), sp.max()) * 1.08]
a.plot(lim, lim, color=C["ink"], lw=.7, ls="--")
a.plot(sp[:, 0], sp[:, 1], "s", ms=4, mfc="none", mec=C["orange"], mew=1, label=f"Separate capacity (MAE {M['sep_mae']:.3f})")
a.plot(sh[:, 0], sh[:, 1], "o", ms=4, color=C["blue"], label=f"Shared capacity (MAE {M['mae_all']:.3f})")
a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
a.set_xlabel("Predicted backlog share"); a.set_ylabel("Measured backlog share"); a.legend(fontsize=6, loc="upper left")
save(fig, "model_fit.pdf")

# 4. latency distributions, u=100, priority scheduling, both repeats pooled
L = load_json("e18_live.json")["runs"]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
for a, Q in zip(ax, [5, 10]):
    for s in ["sa100", "lazy50", "afh"]:
        lab, col, mk = SYS[s]
        x = np.sort(np.concatenate([np.array(r["lat_ms"]) for r in L if r["sys"] == s and r["u"] == 100 and r["Q"] == Q and r["policy"] == "priority"]))
        y = np.arange(1, len(x) + 1) / len(x); a.plot(x, y, color=col, label=lab)
        a.plot(np.percentile(x, 95), .95, marker=mk, ms=4, color=col)
    a.axhline(.95, color=C["mute"], lw=.5, ls=":"); a.set_xscale("log"); a.set_title(f"{Q} queries/s, 100 writes/s", fontsize=8)
    a.set_xlabel("Query latency (ms)")
ax[0].set_ylabel("Share of queries"); h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=3); fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(os.path.join(F, "latency_cdf.pdf")); plt.close(fig)

# 5. corpus coverage against the uniform baseline
V = load_json("e26_coverage.json")
fig, a = plt.subplots(figsize=(3.6, 2.7)); xs = np.logspace(-2, 1.1, 200)
a.plot(xs, 1 - np.exp(-xs), color=C["ink"], lw=.8, ls="--", label="Uniform retrieval, $1-e^{-nK/N}$")
for tag, (lab, col, mk) in [("jitaf", ("JIT-AF", C["orange"], "^")), ("lazy50", ("Lazy fusion, K=50", C["blue"], "o"))]:
    a.plot([V[d][tag]["retrievals_per_doc"] for d in DATASETS], [V[d][tag]["share"] for d in DATASETS], mk, ms=4.5, color=col, label=lab, ls="none")
OFF = {"nfcorpus": (6, 2), "scifact": (-34, 4), "arguana": (5, -6), "fiqa": (5, -6), "scidocs": (5, -6), "trec-covid": (-14, 7), "quora": (5, -6)}
for d in DATASETS:
    v = V[d]["jitaf"]; a.annotate(NICE[d], (v["retrievals_per_doc"], v["share"]), xytext=OFF[d], textcoords="offset points", fontsize=5.5, color=C["mute"])
a.set_xlim(8e-3, 40); a.set_ylim(-.03, 1.1); a.set_xscale("log"); a.set_xlabel("Candidate retrievals per document, $nK/N$"); a.set_ylabel("Share of corpus touched"); a.legend(fontsize=6, loc="upper left")
save(fig, "coverage.pdf")

# 6. trajectory of the backlog share under the separate-capacity model (Equations 9-11)
def traj(t, u, mu, Q, K, m, N0):
    finf = (u - mu) / (u + Q * K); a_ = (u + Q * K) / u; N = N0 + u * t
    if m >= K * finf: return finf * (1 - (N0 / N) ** a_), None
    Nb = N0 * (1 - m / (K * finf)) ** (-1 / a_); tb = (Nb - N0) / u; fb = (u - mu - Q * m) / u
    return np.where(t < tb, finf * (1 - (N0 / N) ** a_), fb + (m / K - fb) * Nb / N), tb
u, mu, Q, K, N0 = 200, 100, 10, 100, 1e5; t = np.logspace(-1, 4, 600)
fig, a = plt.subplots(figsize=(4.4, 2.6)); finf = (u - mu) / (u + Q * K)
for m, col in [(2, C["red"]), (5, C["orange"]), (9, C["blue"])]:
    f, tb = traj(t, u, mu, Q, K, m, N0); a.plot(t, f, color=col, lw=1.6, label=f"m = {m}")
    if tb is not None: a.plot(tb, m / K, "o", ms=4.5, color=col, zorder=5)
f, _ = traj(t, u, mu, Q, K, 1e9, N0); a.plot(t, f, color=C["ink"], lw=.9, ls="--", label="no budget", zorder=6)
a.axhline(finf, color=C["mute"], lw=.5, ls=":"); a.annotate("$f_\\infty$", xy=(.1, finf), xytext=(2, 3), textcoords="offset points", fontsize=7, color=C["mute"])
a.set_xscale("log"); a.set_xlabel("Time (s)"); a.set_ylabel("Backlog share $f(t)$")
a.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, .5)); a.annotate("dots: budget starts\nto bind at $t_b$", xy=(.12, .3), fontsize=6, color=C["mute"])
save(fig, "trajectory.pdf")

json.dump(out_numbers, open(os.path.join(RES, "e28_effects.json"), "w"), indent=1)
print("written to", F)
