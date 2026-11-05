"""Lazy fusion and JIT-AF: query-time encoding of waiting candidates."""
from .core import VectorStore, Corpus, lazy_fusion, jit_af, max_norm

__all__ = ["VectorStore", "Corpus", "lazy_fusion", "jit_af", "max_norm"]
