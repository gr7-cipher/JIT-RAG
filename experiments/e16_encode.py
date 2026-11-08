"""Encode the new collections with bge-small-en-v1.5 (same settings as the seven earlier collections)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
m = encoder("BAAI/bge-small-en-v1.5", 256)
for name in sys.argv[1:]:
    p = res_dir(name); f = os.path.join(p, "bge_emb.npy")
    if os.path.exists(f): continue
    doc_ids, docs, qids, queries, qrels = load_dataset(name); log(name, len(docs), len(qids))
    E = np.lib.format.open_memmap(f + ".tmp.npy", mode="w+", dtype=np.float32, shape=(len(docs), 384)); t0 = time.time()
    for i in range(0, len(docs), 8192):
        E[i:i + 8192] = encode(m, docs[i:i + 8192], bs=32)
        if i % 81920 == 0: E.flush(); log(name, i, round(time.time() - t0))
    E.flush(); del E; os.replace(f + ".tmp.npy", f); log(name, "done", round(time.time() - t0))
