"""Final held-out numbers. Settings were frozen on dev-half queries. Reports test-half queries and untouched BEIR splits."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
L = {d: load_json(d, "perquery_e14_lag.json") for d in DATASETS}; B13 = {d: load_json(d, "perquery_e13_budget.json") for d in DATASETS}; B8 = {d: load_json(d, "perquery_e08_bge.json") for d in DATASETS}
def split(q):
    q = sorted(q); r = np.random.default_rng(SEED).permutation(len(q)); return [q[i] for i in r[len(q) // 2:]]
T = {d: split(L[d]["afh_0.0"]["ndcg"]) for d in DATASETS}; XT = [d for d in DATASETS if d != "trec-covid"]
sub = lambda P, ds, tq=None: {d: {q: P[d][q] for q in (tq[d] if tq else P[d])} for d in ds}
def stats(a, b, ds):
    rng = np.random.default_rng(SEED); arr = {d: np.array([a[d][q] - b[d][q] for q in a[d]]) for d in ds}; mb = [np.mean([rng.choice(arr[d], len(arr[d])).mean() for d in ds]) for _ in range(4000)]
    return [float(np.mean([arr[d].mean() for d in ds])), float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))]
def per(a, b, ds):
    t = {d: paired_test(a[d], b[d]) for d in ds}; h = holm([t[d]["p"] for d in ds])
    return {d: {"diff": t[d]["diff"], "sig": bool(x < .05 and (t[d]["lo"] > 0 or t[d]["hi"] < 0))} for d, x in zip(ds, h)}
g = lambda k, f="ndcg": {d: {q: L[d][k][f][q] for q in T[d]} for d in DATASETS}
base = lambda k, f="ndcg": {d: {q: B8[d][k][f][q] for q in T[d]} for d in DATASETS}
af = {d: {q: B13[d]["af"][q] for q in T[d]} for d in DATASETS}
mac = lambda v: float(np.mean([np.mean(list(v[d].values())) for d in v])); R = {"test": {}, "tests": {}, "lag": {}, "lag_tests": {}, "unseen": {}, "unseen_tests": {}}
S = {"JIT-AF+ANN": (g("afh_0.0"), g("afh_0.0", "recall100")), "Hybrid50": (g("h50_0.0"), g("h50_0.0", "recall100")), "Hybrid100": (g("h100_0.0"), g("h100_0.0", "recall100")), "RRF": (base("rrf"), base("rrf", "recall100")), "Dense": (base("dense"), base("dense", "recall100")), "JIT-AF": (af, None)}
print("== TEST HALF, f=0")
for k, (v, r) in S.items():
    R["test"][k] = {"ndcg": mac(v), "recall": mac(r) if r else None, "per": {d: float(np.mean(list(v[d].values()))) for d in DATASETS}}; print(f"{k:12s} {mac(v):.4f}", "R100 %.3f" % mac(r) if r else "", {d[:5]: round(R['test'][k]['per'][d], 3) for d in DATASETS})
for a in ["JIT-AF+ANN", "JIT-AF"]:
    for o in S:
        if o == a or (a == "JIT-AF" and o == "JIT-AF+ANN"): continue
        A, Bb = S[a][0], S[o][0]; m = stats(A, Bb, DATASETS); x = stats(A, Bb, XT); p = per(A, Bb, DATASETS); rr = stats(S[a][1], S[o][1], DATASETS) if S[a][1] and S[o][1] else None
        R["tests"][f"{a}|{o}"] = {"macro": m, "xt": x, "per": p, "recall": rr}; print(a, "vs", o, "%+.4f [%+.4f,%+.4f] noTREC %+.4f [%+.4f,%+.4f]" % (*m, *x), "R %+.4f [%+.4f,%+.4f]" % tuple(rr) if rr else "", {d[:5]: "%+.3f%s" % (p[d]["diff"], "*" if p[d]["sig"] else "") for d in DATASETS})
print("== LAG, test half")
rrfk = {f: f"rrf_f{f}" for f in ["0.01", "0.05", "0.1", "0.25", "0.5"]}
for t in ["0.0", "0.01", "0.05", "0.1", "0.25", "0.5", "recent"]:
    A = g(f"afh_{t}"); row = {"afh": mac(A), "h50": mac(g(f"h50_{t}")), "h100": mac(g(f"h100_{t}")), "enc": float(np.mean([L[d]["__enc__"][t] for d in DATASETS]))}
    if t in rrfk: row["rrf"] = mac({d: {q: B13[d][rrfk[t]][q] for q in T[d]} for d in DATASETS})
    best = max(["h50", "h100"], key=lambda k: row[k]); Bb = g(f"{best}_{t}"); m = stats(A, Bb, DATASETS); p = per(A, Bb, DATASETS); R["lag"][t] = row; R["lag_tests"][t] = {"vs": best, "macro": m, "per": p}
    print(t, {k: round(v, 4) for k, v in row.items()}, "vs", best, "%+.4f [%+.4f,%+.4f]" % tuple(m), {d[:5]: "%+.3f%s" % (p[d]["diff"], "*" if p[d]["sig"] else "") for d in DATASETS})
print("== UNTOUCHED SPLITS")
U = ["scifact", "nfcorpus", "fiqa", "quora"]; P = {d: load_json(d, "perquery_e13_unseen.json") for d in U}
for k in P[U[0]]:
    R["unseen"][k] = {"ndcg": float(np.mean([mean(P[d][k]["ndcg"]) for d in U])), "recall": float(np.mean([mean(P[d][k]["recall100"]) for d in U])), "per": {d: mean(P[d][k]["ndcg"]) for d in U}}; print(f"{k:6s} {R['unseen'][k]['ndcg']:.4f} R100 {R['unseen'][k]['recall']:.3f}", {d[:5]: round(x, 3) for d, x in R["unseen"][k]["per"].items()})
for a in ["afh", "af"]:
    for o in ["h50", "h100", "rrf", "dense", "b100"]:
        A = {d: P[d][a]["ndcg"] for d in U}; Bb = {d: P[d][o]["ndcg"] for d in U}; m = stats(A, Bb, U); rr = stats({d: P[d][a]["recall100"] for d in U}, {d: P[d][o]["recall100"] for d in U}, U); p = per(A, Bb, U); R["unseen_tests"][f"{a}|{o}"] = {"macro": m, "recall": rr, "per": p}
        print(a, "vs", o, "%+.4f [%+.4f,%+.4f] R %+.4f [%+.4f,%+.4f]" % (*m, *rr), {d[:5]: "%+.3f%s" % (p[d]["diff"], "*" if p[d]["sig"] else "") for d in U})
print("== BUDGET (test half, signal-aware vs mean imputation, f=1 and f=.1)")
for f in ["0.1", "0.5", "1.0"]:
    for m in [2, 5, 10, 20, 50]:
        print("f", f, "m", m, {i: round(mac({d: {q: B13[d][f"m{m}_f{f}_{i}"][q] for q in T[d]} for d in DATASETS}), 4) for i in ["bm", "mean"]}, "enc %.1f" % float(np.mean([B13[d]["__enc__"][f"m{m}_f{f}"] for d in DATASETS])), ("shallow %.4f" % mac({d: {q: B13[d][f"shallow_m{m}"].get(q, 0.0) for q in T[d]} for d in DATASETS})) if f == "1.0" else "")
print("== OVERLOAD: ODE dB/dt = u - mu - Q*min(m, K*B/N)")
O = load_json("e13_overload.json"); R["overload"] = {}
for key, x in O.items():
    u, mu, Qr, m = key.split("_"); u, mu, Qr = int(u[1:]), int(mu[2:]), int(Qr[1:]); m = 10 ** 9 if m == "minf" else int(m[1:]); N0 = 28819; K = 100; B = 0.0; dt = .01; t = 0.0; N = N0; tt = []
    while t < x["T"]:
        B = max(0.0, B + dt * (u - mu - Qr * min(m, K * B / max(N, 1)))); N += u * dt; t += dt; tt.append(B / N)
    tail = float(np.mean(tt[len(tt) // 2:])); R["overload"][key] = {"ode_tail": tail, "ode_last": tt[-1], **x}; print(key, "sim tail %.3f last %.3f | ODE tail %.3f last %.3f | f*=%.3f" % (x["steady_mean"], x["last"], tail, tt[-1], x["pred"]))
save_json(R, "e14_report.json")
LA = load_json("e14_latency.json"); print({d: {k: (round(v["p50"]), round(v["p95"])) for k, v in LA[d].items()} for d in LA})
