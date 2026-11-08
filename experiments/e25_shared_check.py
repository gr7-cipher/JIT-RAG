"""E25: compare the shared-capacity model with the live runs. C is the total encoding rate of signal-aware indexed fusion, which encodes no documents at query time,
at the same query rate. The backlog of every other system is then predicted as B0 + (u - C) t, independently of K and m."""
import numpy as np
from common import load_json, save_json
D = load_json("e18_live.json"); R = [x for x in D["runs"] if x["repeat"] < 90 and x["policy"] == "priority" and x["u"] > 20]; N0 = 28819; rows = []
C = {Q: float(np.mean([x["bg_rate"] for x in R if x["sys"] == "sa100" and x["Q"] == Q])) for Q in sorted({x["Q"] for x in R})}
for x in R:
    c = C[x["Q"]]; t = x["dur"]; B = 0.1 * N0 + (x["u"] - c) * t; f = B / (N0 + x["u"] * t); tot = x["bg_rate"] + x["q_enc_rate"]; rows.append({"sys": x["sys"], "u": x["u"], "Q": x["Q"], "repeat": x["repeat"], "C_sa": c, "total_rate": tot, "f_pred": f, "f_meas": x["f_end"]})
err = [abs(r["f_pred"] - r["f_meas"]) for r in rows]; byS = {s: {"mean_total_rate": float(np.mean([r["total_rate"] for r in rows if r["sys"] == s])), "mae": float(np.mean([abs(r["f_pred"] - r["f_meas"]) for r in rows if r["sys"] == s]))} for s in {r["sys"] for r in rows}}
# separate-capacity model for the same settings, with mu taken from the same C
sep = []
for r in rows:
    if r["sys"] == "sa100": continue
    u, Q = r["u"], r["Q"]; K = 50 if r["sys"].startswith("lazy") else 100; t = next(x["dur"] for x in R if x["sys"] == r["sys"] and x["u"] == u and x["Q"] == Q and x["repeat"] == r["repeat"]); mu = r["C_sa"]; b = 0.1 * N0; n = N0; dt = 0.05
    for _ in range(int(t / dt)):
        b = max(0.0, b + dt * (u - mu - Q * K * b / n)); n += u * dt
    sep.append({"sys": r["sys"], "u": u, "Q": Q, "f_sep": b / n, "f_meas": r["f_meas"]})
out = {"C": C, "rows": rows, "mae_all": float(np.mean(err)), "max_err": float(np.max(err)), "by_system": byS, "sep_mae": float(np.mean([abs(s["f_sep"] - s["f_meas"]) for s in sep])), "sep": sep, "calib": D["calib"], "n_settings": len(R)}
save_json(out, "e25_shared.json"); print({k: out[k] for k in ["C", "mae_all", "max_err", "by_system", "sep_mae"]})
