"""E23: quality-latency-freshness trade-offs from the live test. Reports the non-dominated systems per load, and how often each system maximises
U = wq Q + wl (1 - min(1, L/Lmax)) + wf exp(-A/tau) over a weight simplex, for a sweep of Lmax and tau. A is 0.57 s (1 s refresh) for systems that read the inverted index;
dense-only streaming takes its measured lag at the offered rate and its quality at the measured backlog."""
import json, os
import numpy as np
from common import load_json, save_json, RES
Lv = [x for x in load_json("e18_live.json")["runs"] if x["policy"] == "priority" and x["repeat"] < 90]; R = load_json("e20_stats.json"); out = {"fronts": {}, "wins": {}}
SYS = {"sa100": "Signal-aware indexed fusion", "lazy50": "Lazy fusion K=50", "lazy50m10": "Lazy fusion K=50, m=10", "afh": "JIT-AF+ANN"}; settings = sorted({(x["u"], x["Q"]) for x in Lv})
def pick(s, u, Q):
    xs = [x for x in Lv if x["sys"] == s and x["u"] == u and x["Q"] == Q]
    return None if not xs else {"p50": float(np.mean([x["p50"] for x in xs])), "p95": float(np.mean([x["p95"] for x in xs])), "ndcg": float(np.mean([x["ndcg"] for x in xs])), "f": float(np.mean([x["f_end"] for x in xs]))}
for u, Q in settings:
    P = {s: pick(s, u, Q) for s in SYS}; P = {k: v for k, v in P.items() if v}
    nd = [a for a, x in P.items() if not any(b != a and y["ndcg"] >= x["ndcg"] and y["p95"] <= x["p95"] and (y["ndcg"] > x["ndcg"] or y["p95"] < x["p95"]) for b, y in P.items())]
    out["fronts"][f"u{u}_Q{Q}"] = {"front": nd, "points": P}
W = [(a / 20, b / 20, 1 - (a + b) / 20) for a in range(21) for b in range(21 - a)]
for u, Q in settings:
    P = {s: pick(s, u, Q) for s in SYS}; P = {k: v for k, v in P.items() if v}; res = {}
    if not P: continue
    for Lmax in [0.5, 2.0, 10.0]:
        for tau in [10.0, 60.0, 600.0]:
            wins = {k: 0 for k in P}
            for wq, wl, wf in W:
                U = {k: wq * x["ndcg"] + wl * (1 - min(1.0, x["p50"] / 1e3 / Lmax)) + wf * np.exp(-0.57 / tau) for k, x in P.items()}; wins[max(U, key=U.get)] += 1
            res[f"Lmax{Lmax}_tau{tau}"] = {k: v / len(W) for k, v in wins.items()}
    out["wins"][f"u{u}_Q{Q}"] = res
save_json(out, "e23_pareto.json"); print({k: v["front"] for k, v in out["fronts"].items()})
