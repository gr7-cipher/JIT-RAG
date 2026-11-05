"""RQ9 tables and figure from E11 to E15. Uses test-half queries and untouched splits only."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("JIT_OUT", os.path.join(ROOT, "..", "resubmission")); os.makedirs(os.path.join(OUT, "tables"), exist_ok=True); os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "mute": "#6f6e69", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["mute"], "axes.linewidth": .6, "xtick.color": C["mute"], "ytick.color": C["mute"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .5, "legend.frameon": False, "axes.axisbelow": True})
NICE = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "fiqa": "FiQA", "scidocs": "SciDocs", "trec-covid": "TREC-COVID", "quora": "Quora", "nq": "NQ", "msmarco": "MS MARCO"}
def wr(n, s): open(os.path.join(OUT, "tables", n + ".tex"), "w").write(s + "\n")
def row(lbl, vals, f="%.3f"):
    b = max(v for v in vals if v is not None); return lbl + " & " + " & ".join("--" if v is None else (("\\textbf{%s}" % (f % v)) if v == b else f % v) for v in vals) + " \\\\"
R = load_json("e14_report.json"); L = {d: load_json(d, "perquery_e14_lag.json") for d in DATASETS}; B = {d: load_json(d, "perquery_e13_budget.json") for d in DATASETS}
def split(q):
    q = sorted(q); r = np.random.default_rng(SEED).permutation(len(q)); return [q[i] for i in r[len(q) // 2:]]
T = {d: split(L[d]["afh_0.0"]["ndcg"]) for d in DATASETS}
def mt(A, Bb):
    rng = np.random.default_rng(SEED); arr = {d: np.array([A[d][q] - Bb[d][q] for q in T[d]]) for d in DATASETS}; mb = [np.mean([rng.choice(arr[d], len(arr[d])).mean() for d in DATASETS]) for _ in range(4000)]
    return float(np.mean([arr[d].mean() for d in DATASETS])), float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))
# held-out, complete index
S = ["JIT-AF+ANN", "JIT-AF", "Hybrid100", "RRF", "Dense"]; r = [row(NICE[d], [R["test"][s]["per"][d] for s in S]) for d in DATASETS]
r += ["\\midrule", row("Macro nDCG@10", [R["test"][s]["ndcg"] for s in S], "%.4f"), row("Macro Recall@100", [R["test"][s]["recall"] for s in S])]
wr("af_test", "\n".join(r))
# untouched
U = R["unseen"]; US = ["afh", "af", "h100", "rrf", "dense"]; D4 = ["scifact", "nfcorpus", "fiqa", "quora"]; r = [row(NICE[d], [U[s]["per"][d] for s in US]) for d in D4]
r += ["\\midrule", row("Macro nDCG@10", [U[s]["ndcg"] for s in US], "%.4f"), row("Macro Recall@100", [U[s]["recall"] for s in US]), "\\midrule"]
E = load_json("e15.json")
for d in ["nfcorpus", "fiqa"]:
    r += [row(NICE[d] + " train, nDCG@10", [E[d]["sys"][s][0] for s in US]), row(NICE[d] + " train, Recall@100", [E[d]["sys"][s][1] for s in US])]
wr("af_unseen", "\n".join(r))
# backlog
I = {d: load_json(d, "perquery_e10_lagimp.json") for d in DATASETS}; lab = {"0.0": "0\\%", "0.01": "1\\%", "0.05": "5\\%", "0.1": "10\\%", "0.25": "25\\%", "0.5": "50\\%", "recent": "Recent relevant"}; r = []; O = {}
for t in lab:
    A = {d: L[d][f"afh_{t}"]["ndcg"] for d in DATASETS}; H = {d: L[d][f"h100_{t}"]["ndcg"] for d in DATASETS}
    if t == "0.0": Rr = {d: load_json(d, "perquery_e08_bge.json")["rrf"]["ndcg"] for d in DATASETS}
    elif t == "recent": Rr = {d: I[d]["rrfimp_recent"] for d in DATASETS}
    else: Rr = {d: B[d][f"rrf_f{t}"] for d in DATASETS}
    mh = lambda P: float(np.mean([np.mean([P[d][q] for q in T[d]]) for d in DATASETS])); best = H if mh(H) >= mh(Rr) else Rr; m = mt(A, best); O[t] = {"afh": mh(A), "hyb": mh(H), "rrf": mh(Rr), "diff": m, "enc": R["lag"][t]["enc"]}
    r.append(lab[t] + " & %.1f & " % R["lag"][t]["enc"] + row("", [mh(A), mh(H), mh(Rr)], "%.4f").split(" & ", 1)[1].replace(" \\\\", "") + " & %+.4f [%+.4f, %+.4f] \\\\" % m)
wr("af_lag", "\n".join(r))
# budget, JIT-AF without ANN
mb = lambda k: float(np.mean([np.mean([B[d][k].get(q, 0.0) for q in T[d]]) for d in DATASETS])); La = load_json("e13_latency.json"); P = [d for d in DATASETS if d != "quora"]; lt = lambda k: "{:,.0f} / {:,.0f}".format(np.mean([La[d][k]["p50"] for d in P]), np.mean([La[d][k]["p95"] for d in P]))
r = []
for f, ms in [("0.1", [5, 10, 20]), ("0.5", [5, 10, 20, 50]), ("1.0", [10, 20, 50])]:
    for m in ms:
        k = f"f{f}_m{m}"; r.append("%d\\%% & %d & %.4f & %.4f & %s & %s \\\\" % (float(f) * 100, m, mb(f"m{m}_f{f}_bm"), mb(f"m{m}_f{f}_mean"), ("%.4f" % mb(f"shallow_m{m}")) if f == "1.0" else "--", lt(k) if k in La["fiqa"] else "--"))
    r.append("\\midrule")
wr("af_budget", "\n".join(r[:-1]))
sig = mt({d: B[d]["m5_f0.1_bm"] for d in DATASETS}, {d: L[d]["h100_0.1"]["ndcg"] for d in DATASETS}); print("m5 f.1 vs hybrid", sig)
# overload
Ov = load_json("e13_overload.json"); r = []
for k, x in Ov.items():
    u, mu, Qr, m = k.split("_"); u, mu, Qr = int(u[1:]), int(mu[2:]), int(Qr[1:])
    if m != "minf" or u <= mu: continue
    N0 = 28819; N = N0 + u * x["T"]; cf = lambda K: (u - mu) / (u + Qr * K) * (1 - (N0 / N) ** ((u + Qr * K) / u)); Ke = x["enc_q"] / x["steady_mean"]
    r.append("%d & %d & %d & %.3f & %.3f & %.3f & %.0f & %.3f \\\\" % (u, mu, Qr, (u - mu) / (u + Qr * 100), cf(100), x["last"], Ke, cf(Ke)))
wr("af_overload", "\n".join(r))
# warm
W = load_json("e13_warm.json"); r = []
for d in DATASETS:
    s = np.array(W[d]["series"]); r.append("%s & %d & %.1f & %.1f & {:,} & {:,} & %.1f\\%% \\\\".format(int(s.sum()), 100 * len(s)) % (NICE[d], len(s), s[:25].mean(), s[-25:].mean(), 100 * W[d]["distinct"] / W[d]["N"]))
wr("af_warm", "\n".join(r))
# latency
A = load_json("e14_latency.json"); Lh = load_json("e10_latency_hyb.json"); r = []
for d in DATASETS:
    r.append(NICE[d] + " & " + " & ".join("{:,.0f} / {:,.0f}".format(x["p50"], x["p95"]) for x in [A[d]["f0.0"], Lh[d], A[d]["f0.1"], La[d]["f0.1_m5"], La[d]["f1.0_m20"], La[d]["f1.0_minf"]]) + " \\\\")
wr("af_latency", "\n".join(r))
# figure
fig, a = plt.subplots(figsize=(3.6, 2.4)); F = ["0.0", "0.01", "0.05", "0.1", "0.25", "0.5"]; xs = [float(x) * 100 for x in F]
a.plot(xs, [O[t]["afh"] for t in F], "-o", ms=3, lw=1.5, color=C["blue"], label="JIT-AF+ANN"); a.plot(xs, [O[t]["hyb"] for t in F], "-s", ms=3, lw=1, color=C["aqua"], label="Hybrid CC, signal-aware"); a.plot(xs, [O[t]["rrf"] for t in F], "-^", ms=3, lw=1, color=C["orange"], label="RRF, signal-aware")
a.set_xlabel("Corpus waiting for the encoder (%)"); a.set_ylabel("Macro nDCG@10"); a.legend(loc="lower left", fontsize=6.5); fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "af_lag.pdf")); fig.savefig(os.path.join(OUT, "figures", "af_lag.png"), dpi=200)
save_json({"lag": O, "m5_vs_hyb": sig}, "e14_tables.json"); print(json.dumps(O, indent=0)[:900])
# macro over the two clean splits
for o in ["h100", "rrf", "dense"]:
    rng = np.random.default_rng(SEED); X = {d: load_json(d, "perquery_e15.json") for d in ["nfcorpus", "fiqa"]}; arr = [np.array([X[d]["afh"]["ndcg"][q] - X[d][o]["ndcg"][q] for q in X[d]["afh"]["ndcg"]]) for d in X]; arr2 = [np.array([X[d]["afh"]["recall100"][q] - X[d][o]["recall100"][q] for q in X[d]["afh"]["ndcg"]]) for d in X]
    mbs = [np.mean([rng.choice(y, len(y)).mean() for y in arr]) for _ in range(4000)]; mbr = [np.mean([rng.choice(y, len(y)).mean() for y in arr2]) for _ in range(4000)]
    print("clean macro", o, "%+.4f [%+.4f,%+.4f]" % (np.mean([y.mean() for y in arr]), *np.percentile(mbs, [2.5, 97.5])), "R %+.4f [%+.4f,%+.4f]" % (np.mean([y.mean() for y in arr2]), *np.percentile(mbr, [2.5, 97.5])))
