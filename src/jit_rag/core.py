"""A small reference implementation of the two query-time designs in the paper.

The experiments in ``experiments/`` use their own vectorised code paths and produced every reported number.
This module exists so the algorithms can be read and tested on their own. It needs numpy and any BM25 scorer
and encoder you pass in.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

Encoder = Callable[[List[str]], np.ndarray]


def max_norm(x: np.ndarray) -> np.ndarray:
    """x / max(x), or zeros when the maximum is not positive."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.max() <= 0:
        return np.zeros_like(x)
    return x / x.max()


def min_max(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


@dataclass
class VectorStore:
    """Write-back store keyed by a hash of model version and text.

    Sequential use encodes each document version at most once. Concurrent callers need a single-flight step
    on top of this class to keep that property.
    """

    encoder: Encoder
    model_version: str = "model"
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    encodings: int = 0

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_version}\0{text}".encode()).hexdigest()

    def has(self, text: str) -> bool:
        return self.key(text) in self.vectors

    def get(self, text: str) -> Optional[np.ndarray]:
        return self.vectors.get(self.key(text))

    def encode_missing(self, texts: Sequence[str]) -> int:
        """Encode the given texts that have no stored vector, as one batch. Returns the number encoded."""
        missing = list(dict.fromkeys(t for t in texts if not self.has(t)))
        if not missing:
            return 0
        for text, vector in zip(missing, self.encoder(missing)):
            self.vectors[self.key(text)] = np.asarray(vector, dtype=np.float32)
        self.encodings += len(missing)
        return len(missing)


@dataclass
class Corpus:
    """Inputs the designs need. ``bm25(query_terms)`` returns a score per document for {term: weight}.
    ``tokens(i)`` returns the terms of document i. ``tokenize(text)`` returns terms for a query."""

    texts: List[str]
    bm25: Callable[[Dict[str, float]], np.ndarray]
    tokens: Callable[[int], List[str]]
    tokenize: Callable[[str], List[str]]
    idf: Callable[[str], float]


def _weights(terms: Sequence[str]) -> Dict[str, float]:
    w: Dict[str, float] = {}
    for t in terms:
        w[t] = w.get(t, 0.0) + 1.0
    return w


def _top(scores: np.ndarray, k: int, exclude: Sequence[int] = ()) -> np.ndarray:
    s = np.array(scores, dtype=np.float64)
    if len(exclude):
        s[list(exclude)] = 0.0
    k = min(k, int(np.count_nonzero(s > 0)))
    return np.argsort(-s, kind="stable")[:k]


def _fuse(cos: np.ndarray, lex: np.ndarray, alpha: float) -> np.ndarray:
    """Equation 1 and 2 of the paper. A candidate with no vector (nan) takes its lexical term at full weight."""
    known = ~np.isnan(cos)
    out = np.array(lex, dtype=np.float64)
    if known.any():
        out[known] = alpha * max_norm(cos[known]) + (1 - alpha) * lex[known]
    return out


def lazy_fusion(
    corpus: Corpus,
    store: VectorStore,
    query_vector: np.ndarray,
    query: str,
    ann: Optional[Callable[[np.ndarray, int], Sequence[int]]] = None,
    k: int = 50,
    alpha: float = 0.9,
    budget: Optional[int] = None,
    top_n: int = 10,
) -> List[int]:
    """Algorithm 1: indexed fusion that encodes waiting BM25 candidates at query time."""
    s0 = corpus.bm25(_weights(corpus.tokenize(query)))
    bm = _top(s0, k)
    waiting = [int(i) for i in bm if not store.has(corpus.texts[int(i)])]
    store.encode_missing([corpus.texts[i] for i in (waiting if budget is None else waiting[:budget])])
    dense = list(ann(query_vector, k)) if ann else []
    pool = np.array(sorted(set(bm.tolist()) | {int(i) for i in dense}), dtype=np.int64)
    cos = np.array([np.nan if store.get(corpus.texts[i]) is None else float(store.get(corpus.texts[i]) @ query_vector) for i in pool])
    f = _fuse(cos, max_norm(s0[pool]), alpha)
    return pool[np.argsort(-f, kind="stable")[:top_n]].tolist()


def _rm3(corpus: Corpus, query_terms: Dict[str, float], docs: Sequence[int], p: np.ndarray, terms: int = 20, lam: float = 0.3) -> Dict[str, float]:
    """RM3 interpolation with IDF-weighted feedback terms, the variant the paper uses."""
    rel: Dict[str, float] = {}
    for d, pd in zip(docs, p):
        toks = corpus.tokens(int(d))
        for t in set(toks):
            rel[t] = rel.get(t, 0.0) + float(pd) * toks.count(t) / max(1, len(toks)) * corpus.idf(t)
    keep = sorted(rel.items(), key=lambda x: -x[1])[:terms]
    total = sum(v for _, v in keep) or 1.0
    qt = sum(query_terms.values()) or 1.0
    out = {t: lam * v / qt for t, v in query_terms.items()}
    for t, v in keep:
        out[t] = out.get(t, 0.0) + (1 - lam) * v / total
    return out


def jit_af(
    corpus: Corpus,
    store: VectorStore,
    query_vector: np.ndarray,
    query: str,
    rounds: Sequence[int] = (40, 30, 30),
    alpha: float = 0.95,
    gamma: float = 0.3,
    budget: Optional[int] = None,
    ann: Optional[Callable[[np.ndarray, int], Sequence[int]]] = None,
    ann_k: int = 50,
    feedback_docs: int = 3,
    top_n: int = 10,
) -> List[int]:
    """Algorithm 2, as implemented in the experiments. Use alpha=0.8 with an ANN index."""
    q = _weights(corpus.tokenize(query))
    s0 = corpus.bm25(q)
    last = s0
    total = sum(rounds)
    pool = _top(s0, rounds[0]).tolist()
    used = 0
    allow = 0.0

    def spend(new: Sequence[int], share: float) -> None:
        nonlocal used, allow
        wait = [i for i in new if not store.has(corpus.texts[i])]
        if budget is not None:
            allow += share * budget
            wait = wait[: max(0, int(allow + 1e-9) - used)]
        used += store.encode_missing([corpus.texts[i] for i in wait])

    def cosines(ids: Sequence[int]) -> np.ndarray:
        return np.array([np.nan if store.get(corpus.texts[i]) is None else float(store.get(corpus.texts[i]) @ query_vector) for i in ids])

    spend(pool, rounds[0] / total)
    for add in rounds[1:]:
        if not pool:
            break
        ids = np.array(pool)
        cos = cosines(pool)
        known = ~np.isnan(cos)
        sc = 0.2 * min_max(s0[ids])
        if known.any():
            sc[known] += 0.8 * min_max(cos[known])
        order = np.argsort(-sc)[:feedback_docs]
        p = sc[order] + 1e-9
        last = corpus.bm25(_rm3(corpus, q, ids[order].tolist(), p / p.sum()))
        new = _top(last, add, exclude=pool).tolist()
        if not new:
            break
        pool += new
        spend(new, add / total)
    if ann:
        pool += [int(i) for i in ann(query_vector, ann_k) if int(i) not in set(pool)]
    ids = np.array(pool, dtype=np.int64)
    lex = (1 - gamma) * max_norm(s0[ids]) + gamma * max_norm(last[ids])
    f = _fuse(cosines(pool), lex, alpha)
    return ids[np.argsort(-f, kind="stable")[:top_n]].tolist()
