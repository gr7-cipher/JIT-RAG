"""E26: corpus coverage of the write-back replays against a uniform-retrieval baseline.

Reads the write-back replays stored in e20_stats.json (Table tab:writeback). For each collection and design,
T is the number of candidate encodings without write-back (one per candidate retrieval) and D the number
with write-back (distinct documents touched). If each retrieval drew a document uniformly from the N in the
corpus, the expected touched share after T retrievals would be 1 - (1 - 1/N)^T ~ 1 - exp(-T/N).
The ratio of observed to uniform share is below one when queries concentrate on part of the corpus.
"""
import math
from common import load_json, save_json, DATASETS

R = load_json("e20_stats.json"); W = R["writeback"]; out = {}
for s in DATASETS:
    n = R["N"][s]; out[s] = {"N": n}
    for k, tag in [("afh|f1.0", "jitaf"), ("lazy50|f1.0", "lazy50")]:
        d, t = W[s][k]; uni = 1 - (1 - 1 / n) ** t
        out[s][tag] = {"touched": d, "retrievals": t, "retrievals_per_doc": t / n, "share": d / n,
                       "uniform_share": uni, "share_over_uniform": (d / n) / uni, "reuse": t / d}
save_json(out, "e26_coverage.json")
for s, v in out.items():
    a, b = v["jitaf"], v["lazy50"]
    print(f"{s:11s} T/N={a['retrievals_per_doc']:.3f} share={a['share']:.3f} uniform={a['uniform_share']:.3f} "
          f"ratio={a['share_over_uniform']:.3f} reuse={a['reuse']:.2f} | lazy50 ratio={b['share_over_uniform']:.3f}")
