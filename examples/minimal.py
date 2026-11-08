"""Lazy fusion on a tiny corpus with a real encoder. Run: PYTHONPATH=src python examples/minimal.py"""
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from jit_rag import Corpus, VectorStore, lazy_fusion

TEXTS = ["The central bank raised interest rates by a quarter point.",
         "Mortgage payments rise when lending rates go up.",
         "Herons nest along the river bank in spring.",
         "A savings account pays interest every month.",
         "Dogs and cats need regular vaccinations."]
toks = lambda t: [w.strip(".,").lower() for w in t.split()]
df = {}
for t in TEXTS:
    for w in set(toks(t)):
        df[w] = df.get(w, 0) + 1
idf = lambda w: math.log(1 + len(TEXTS) / (1 + df.get(w, 0)))
def bm25(weights):
    return np.array([sum(wt * toks(t).count(w) * idf(w) for w, wt in weights.items()) for t in TEXTS])
corpus = Corpus(TEXTS, bm25, lambda i: toks(TEXTS[i]), toks, idf)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
encode = lambda texts: model.encode(texts, normalize_embeddings=True)
store = VectorStore(encode, model_version="bge-small-en-v1.5")
query = "how do interest rates affect loans"
query_vector = encode(["Represent this sentence for searching relevant passages: " + query])[0]
top = lazy_fusion(corpus, store, query_vector, query, k=3, top_n=3)
print("encoded at query time:", store.encodings)
for i in top:
    print(i, TEXTS[i])
lazy_fusion(corpus, store, query_vector, query, k=3)
print("after a repeat query, total encodings:", store.encodings)
