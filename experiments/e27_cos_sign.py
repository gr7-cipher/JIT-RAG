"""E27: sign of cosine similarities inside the fusion candidate sets.

Max normalization divides each signal by its maximum over the candidate set. It keeps the order of the
dense signal whenever that maximum is positive; it would reverse the order only if every candidate had a
negative cosine. For each of the seven collections this script forms the candidate set used by indexed
fusion at K=100 (BM25 top 100 plus exact dense top 100) from stored bge-small vectors and reports the
share of negative cosines, the smallest cosine seen, and the number of queries whose maximum is not positive.
"""
import numpy as np
from common import RES, DATASETS, save_json
import os

K = 100; out = {}
for name in DATASETS:
    E = np.load(os.path.join(RES, name, "bge_emb.npy"), mmap_mode="r")
    Q = np.load(os.path.join(RES, name, "bge_qemb.npy"))
    C = np.load(os.path.join(RES, name, "bm25_cands.npy"), allow_pickle=True)
    n_neg = n_all = n_bad = 0; lo = 1.0; qmax_min = 1.0
    for i in range(len(Q)):
        v = Q[i]
        sims = np.asarray(E @ v) if E.shape[0] <= 60000 else None
        if sims is None:  # large corpora: blockwise to bound memory
            sims = np.concatenate([np.asarray(E[j:j + 100000]) @ v for j in range(0, E.shape[0], 100000)])
        dense = np.argpartition(-sims, K)[:K]
        cand = np.union1d(np.asarray(C[i][:K], dtype=np.int64), dense)
        s = sims[cand]
        n_neg += int((s < 0).sum()); n_all += len(s); lo = min(lo, float(s.min()))
        qmax_min = min(qmax_min, float(s.max())); n_bad += int(s.max() <= 0)
    out[name] = {"queries": len(Q), "candidates": n_all, "negative_share": n_neg / n_all,
                 "min_cos": lo, "min_of_query_max": qmax_min, "queries_max_nonpositive": n_bad}
    print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in out[name].items()}, flush=True)
save_json(out, "e27_cos_sign.json")
