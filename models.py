"""
JIT-RAG Models Implementation

This module implements the core JIT-RAG architecture and baseline models
as described in the paper "JIT-RAG: A Just-in-Time Embedding Architecture 
for Fresh, Compliant, and Scalable Retrieval-Augmented Generation"
"""

import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from tqdm import tqdm
import time


class JITRAG:
    """
    JIT-RAG: Just-in-Time Retrieval-Augmented Generation
    
    Implements the three-stage pipeline:
    1. Candidate Generation (BM25 sparse retrieval)
    2. JIT Semantic Reranking (on-the-fly dense embeddings)
    3. Result Finalization (cosine similarity ranking)
    
    Algorithm 1 from the paper.
    """
    
    def __init__(self, 
                 corpus: List[Dict[str, Any]], 
                 encoder_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize JIT-RAG system.
        
        Args:
            corpus: List of documents, each with 'id' and 'text' fields
            encoder_model_name: Name of the sentence transformer model
        """
        self.corpus = corpus
        self.encoder_model_name = encoder_model_name
        
        # Initialize bi-encoder model for dense embeddings
        self.encoder = SentenceTransformer(encoder_model_name)
        
        # Build BM25 index on the corpus
        self._build_bm25_index()
        
        # Create document ID to index mapping
        self.doc_id_to_idx = {doc['id']: idx for idx, doc in enumerate(corpus)}
        self.idx_to_doc_id = {idx: doc['id'] for idx, doc in enumerate(corpus)}
    
    def _build_bm25_index(self):
        """Build BM25 inverted index on the corpus."""
        # Tokenize corpus for BM25
        tokenized_corpus = [doc['text'].lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, K: int = 200, N: int = 10) -> List[Dict[str, Any]]:
        """
        Main retrieval method implementing Algorithm 1 from the paper.
        
        Args:
            query: Query string
            K: Number of candidates to retrieve in Stage 1
            N: Number of final results to return
            
        Returns:
            List of top-N documents with id, text, and similarity score
        """
        # Stage 1: Candidate Generation using BM25
        D_cand = self._sparse_retrieve(query, K)
        
        # Stage 2: JIT Semantic Reranking
        # Encode query to dense vector
        v_q = self.encoder.encode(query, convert_to_tensor=False, show_progress_bar=False)
        
        # Encode candidates on-the-fly and compute similarities
        V_cand = []
        for doc_idx in D_cand:
            doc_text = self.corpus[doc_idx]['text']
            v_i = self.encoder.encode(doc_text, convert_to_tensor=False, show_progress_bar=False)
            V_cand.append((doc_idx, v_i))
        
        # Stage 3: Semantic Re-ranking
        # Compute cosine similarities
        S = []
        for doc_idx, v_i in V_cand:
            s_i = self._cosine_similarity(v_q, v_i)
            S.append((doc_idx, s_i))
        
        # Sort by descending score
        D_ranked = sorted(S, key=lambda x: x[1], reverse=True)
        
        # Return top-N results
        results = []
        for doc_idx, score in D_ranked[:N]:
            results.append({
                'id': self.corpus[doc_idx]['id'],
                'text': self.corpus[doc_idx]['text'],
                'score': float(score)
            })
        
        return results
    
    def _sparse_retrieve(self, query: str, K: int) -> List[int]:
        """
        Stage 1: BM25 candidate retrieval.
        
        Args:
            query: Query string
            K: Number of candidates to retrieve
            
        Returns:
            List of document indices (top-K by BM25 score)
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-K indices
        top_k_indices = np.argsort(scores)[::-1][:K]
        return top_k_indices.tolist()
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity score in [-1, 1]
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def delete_document(self, doc_id: str) -> float:
        """
        Delete a document from the corpus and BM25 index.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Time taken for deletion in seconds
        """
        start_time = time.time()
        
        # Remove from corpus
        doc_idx = self.doc_id_to_idx[doc_id]
        del self.corpus[doc_idx]
        
        # Rebuild BM25 index (lightweight operation)
        self._build_bm25_index()
        
        # Update mappings
        self.doc_id_to_idx = {doc['id']: idx for idx, doc in enumerate(self.corpus)}
        self.idx_to_doc_id = {idx: doc['id'] for idx, doc in enumerate(self.corpus)}
        
        end_time = time.time()
        return end_time - start_time


class DenseRAG:
    """
    Dense RAG baseline with pre-computed FAISS HNSW index.
    
    This represents the conventional approach that pre-computes embeddings
    and builds a persistent vector index.
    """
    
    def __init__(self, 
                 corpus: List[Dict[str, Any]], 
                 encoder_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 index_type: str = 'hnsw'):
        """
        Initialize Dense RAG system.
        
        Args:
            corpus: List of documents, each with 'id' and 'text' fields
            encoder_model_name: Name of the sentence transformer model
            index_type: Type of FAISS index ('hnsw' or 'flat')
        """
        self.corpus = corpus
        self.encoder_model_name = encoder_model_name
        self.index_type = index_type
        
        # Initialize encoder
        self.encoder = SentenceTransformer(encoder_model_name)
        
        # Pre-compute embeddings and build index
        self._build_index()
        
        # Tombstone set for deleted documents
        self.tombstones = set()
    
    def _build_index(self):
        """Pre-compute embeddings and build FAISS index."""
        print(f"Pre-computing embeddings for {len(self.corpus)} documents...")
        
        # Encode all documents
        texts = [doc['text'] for doc in self.corpus]
        self.embeddings = self.encoder.encode(texts, 
                                              convert_to_tensor=False, 
                                              show_progress_bar=True,
                                              batch_size=32)
        
        # Build FAISS index
        d = self.embeddings.shape[1]  # Embedding dimension
        
        if self.index_type == 'hnsw':
            # HNSW index for efficient ANN search
            self.index = faiss.IndexHNSWFlat(d, 32)  # 32 is M parameter
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
        else:
            # Flat index for exact search
            self.index = faiss.IndexFlatIP(d)  # Inner product (cosine with normalized vectors)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add to index
        self.index.add(self.embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, N: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-N documents using FAISS ANN search.
        
        Args:
            query: Query string
            N: Number of results to return
            
        Returns:
            List of top-N documents with id, text, and similarity score
        """
        # Encode query
        query_vec = self.encoder.encode(query, convert_to_tensor=False, show_progress_bar=False)
        query_vec = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        # Search with extra candidates to account for tombstones
        k_search = N + len(self.tombstones)
        distances, indices = self.index.search(query_vec, k_search)
        
        # Filter out tombstones and collect results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx not in self.tombstones and len(results) < N:
                results.append({
                    'id': self.corpus[idx]['id'],
                    'text': self.corpus[idx]['text'],
                    'score': float(score)
                })
        
        return results
    
    def delete_document(self, doc_id: str, method: str = 'tombstone') -> float:
        """
        Delete a document using either tombstoning or full re-index.
        
        Args:
            doc_id: Document ID to delete
            method: 'tombstone' or 'reindex'
            
        Returns:
            Time taken for deletion in seconds
        """
        start_time = time.time()
        
        # Find document index
        doc_idx = None
        for idx, doc in enumerate(self.corpus):
            if doc['id'] == doc_id:
                doc_idx = idx
                break
        
        if doc_idx is None:
            return 0.0
        
        if method == 'tombstone':
            # Mark as deleted
            self.tombstones.add(doc_idx)
        else:
            # Full re-index
            del self.corpus[doc_idx]
            self._build_index()
            self.tombstones = set()
        
        end_time = time.time()
        return end_time - start_time
    
    def get_tombstone_overhead(self) -> float:
        """
        Get the percentage of tombstoned documents.
        
        Returns:
            Percentage of tombstoned documents
        """
        if len(self.corpus) == 0:
            return 0.0
        return len(self.tombstones) / len(self.corpus) * 100


class BM25Baseline:
    """
    Pure BM25 sparse retrieval baseline.
    """
    
    def __init__(self, corpus: List[Dict[str, Any]]):
        """
        Initialize BM25 baseline.
        
        Args:
            corpus: List of documents, each with 'id' and 'text' fields
        """
        self.corpus = corpus
        
        # Build BM25 index
        tokenized_corpus = [doc['text'].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, N: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-N documents using BM25.
        
        Args:
            query: Query string
            N: Number of results to return
            
        Returns:
            List of top-N documents with id, text, and BM25 score
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-N indices
        top_n_indices = np.argsort(scores)[::-1][:N]
        
        results = []
        for idx in top_n_indices:
            results.append({
                'id': self.corpus[idx]['id'],
                'text': self.corpus[idx]['text'],
                'score': float(scores[idx])
            })
        
        return results


class BM25CrossEncoder:
    """
    BM25 + Cross-Encoder baseline.
    
    High-quality but high-latency two-stage pipeline using BM25 for candidate
    generation and a cross-encoder for re-ranking.
    """
    
    def __init__(self, 
                 corpus: List[Dict[str, Any]], 
                 cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize BM25 + Cross-Encoder baseline.
        
        Args:
            corpus: List of documents, each with 'id' and 'text' fields
            cross_encoder_model: Name of the cross-encoder model
        """
        self.corpus = corpus
        
        # Build BM25 index
        tokenized_corpus = [doc['text'].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize cross-encoder
        self.cross_encoder = CrossEncoder(cross_encoder_model)
    
    def retrieve(self, query: str, K: int = 1000, N: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-N documents using BM25 + Cross-Encoder.
        
        Args:
            query: Query string
            K: Number of candidates to retrieve with BM25
            N: Number of final results to return
            
        Returns:
            List of top-N documents with id, text, and cross-encoder score
        """
        # Stage 1: BM25 candidate retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:K]
        
        # Stage 2: Cross-encoder re-ranking
        pairs = [[query, self.corpus[idx]['text']] for idx in top_k_indices]
        cross_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        # Sort by cross-encoder scores
        scored_docs = [(idx, score) for idx, score in zip(top_k_indices, cross_scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-N
        results = []
        for idx, score in scored_docs[:N]:
            results.append({
                'id': self.corpus[idx]['id'],
                'text': self.corpus[idx]['text'],
                'score': float(score)
            })
        
        return results
