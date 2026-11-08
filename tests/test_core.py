import hashlib
import math
import numpy as np
import pytest
from jit_rag import Corpus, VectorStore, jit_af, lazy_fusion, max_norm

TEXTS = ["bank rates rise again", "central bank cuts interest rates", "river bank erosion study", "interest on savings accounts",
         "cats and dogs at home", "dogs chase cats", "savings and loan rates", "mortgage interest rates fall"]


def toks(t):
    return t.lower().split()


def make_corpus():
    n = len(TEXTS); df = {}
    for t in TEXTS:
        for w in set(toks(t)):
            df[w] = df.get(w, 0) + 1
    idf = lambda w: math.log(1 + n / (1 + df.get(w, 0)))
    def bm25(w):
        s = np.zeros(n)
        for i, t in enumerate(TEXTS):
            for term, wt in w.items():
                s[i] += wt * toks(t).count(term) * idf(term)
        return s
    return Corpus(TEXTS, bm25, lambda i: toks(TEXTS[i]), toks, idf)


def encoder(texts):
    out = []
    for t in texts:
        v = np.zeros(32)
        for w in toks(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out)


def qv(q):
    return encoder([q])[0]


def test_max_norm():
    assert max_norm(np.array([1.0, 2.0])).tolist() == [0.5, 1.0]
    assert max_norm(np.array([-1.0, 0.0])).tolist() == [0.0, 0.0]


def test_at_most_once_lazy():
    c = make_corpus(); s = VectorStore(encoder)
    for _ in range(3):
        lazy_fusion(c, s, qv("interest rates"), "interest rates", k=5)
    first = s.encodings
    lazy_fusion(c, s, qv("interest rates"), "interest rates", k=5)
    assert s.encodings == first <= len(TEXTS)


def test_at_most_once_jit_af():
    c = make_corpus(); s = VectorStore(encoder)
    for q in ["interest rates", "bank rates", "cats dogs", "interest rates"]:
        jit_af(c, s, qv(q), q, rounds=(3, 2, 2))
    assert s.encodings <= len(TEXTS)
    assert s.encodings == len(s.vectors)


def test_edit_misses_and_model_version_misses():
    s = VectorStore(encoder, model_version="a"); s.encode_missing(["x y"]); s.encode_missing(["x y"]); assert s.encodings == 1
    s.encode_missing(["x y z"]); assert s.encodings == 2
    other = VectorStore(encoder, model_version="b"); assert other.key("x y") != s.key("x y")


def test_budget_limits_encodings():
    c = make_corpus(); s = VectorStore(encoder)
    lazy_fusion(c, s, qv("interest rates"), "interest rates", k=5, budget=1); assert s.encodings == 1
    s2 = VectorStore(encoder); jit_af(c, s2, qv("interest rates"), "interest rates", rounds=(3, 2, 2), budget=2); assert s2.encodings <= 2


def test_ranking_does_not_depend_on_store_state_when_budget_suffices():
    """Backlog invariance of JIT-AF without ANN, for a deterministic encoder."""
    c = make_corpus(); q = "interest rates"
    cold = VectorStore(encoder); warm = VectorStore(encoder); warm.encode_missing(TEXTS); part = VectorStore(encoder); part.encode_missing(TEXTS[::2])
    a = jit_af(c, cold, qv(q), q, rounds=(3, 2, 2)); b = jit_af(c, warm, qv(q), q, rounds=(3, 2, 2)); d = jit_af(c, part, qv(q), q, rounds=(3, 2, 2))
    assert a == b == d


def test_unencoded_candidate_is_still_ranked():
    c = make_corpus(); s = VectorStore(encoder)
    out = lazy_fusion(c, s, qv("interest rates"), "interest rates", k=5, budget=0)
    assert out and s.encodings == 0
