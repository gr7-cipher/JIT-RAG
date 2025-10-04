"""
Evaluation Framework for JIT-RAG

This module implements the Quality-Latency-Freshness (QLF) utility framework
and experimental analysis functions as described in the paper.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import (
    calculate_ndcg, calculate_average_age, filter_corpus_by_time,
    create_temporal_corpus, evaluate_retrieval, measure_latency
)


def calculate_qlf_utility(Q: float, 
                          L: float, 
                          F: float, 
                          w_q: float = 1.0, 
                          w_l: float = 0.1, 
                          w_f: float = 0.5) -> float:
    """
    Calculate Quality-Latency-Freshness (QLF) utility score.
    
    Implements Equation 7 from the paper:
    U = w_q * Q - w_l * L + w_f * F
    
    Args:
        Q: Quality score (e.g., nDCG@10)
        L: Latency in seconds
        F: Freshness score (0 to 1)
        w_q: Weight for quality (default: 1.0)
        w_l: Weight for latency (default: 0.1, penalizing each 100ms by 0.01)
        w_f: Weight for freshness (default: 0.5)
        
    Returns:
        Utility score
    """
    return w_q * Q - w_l * L + w_f * F


def calculate_freshness(documents: List[Dict[str, Any]], 
                        current_time: datetime, 
                        alpha: float = 0.01) -> float:
    """
    Calculate freshness score using exponential decay.
    
    Implements Equation 8 from the paper:
    F = e^(-alpha * A)
    
    where A is the average age of documents in hours.
    
    Args:
        documents: List of retrieved documents with timestamps
        current_time: Current time
        alpha: Decay constant (default: 0.01 per hour)
        
    Returns:
        Freshness score between 0 and 1
    """
    if len(documents) == 0:
        return 1.0
    
    # Calculate average age in hours
    A = calculate_average_age(documents, current_time)
    
    # Apply exponential decay
    F = np.exp(-alpha * A)
    
    return float(F)


def data_staleness_simulation(models: Dict[str, Any],
                              corpus: List[Dict[str, Any]],
                              queries: List[Dict[str, Any]],
                              reindex_intervals: List[int] = [1, 6, 12, 24, 48, 72],
                              current_time: datetime = None) -> Dict[str, List[float]]:
    """
    Simulate data staleness and measure performance degradation.
    
    Implements the analysis from Section 6.3.2 and Figure 3.
    
    Args:
        models: Dictionary of model instances
        corpus: Temporal corpus with timestamps
        queries: List of queries with relevant document IDs
        reindex_intervals: List of re-indexing intervals in hours
        current_time: Current time (default: max timestamp in corpus)
        
    Returns:
        Dictionary mapping model names to performance lists
    """
    if current_time is None:
        current_time = max(doc['timestamp'] for doc in corpus)
    
    results = {model_name: [] for model_name in models.keys()}
    
    print("\n=== Data Staleness Simulation ===")
    
    for interval_hours in tqdm(reindex_intervals, desc="Testing intervals"):
        # For Dense RAG, filter corpus to simulate stale index
        cutoff_time = current_time - timedelta(hours=interval_hours)
        
        interval_results = {model_name: [] for model_name in models.keys()}
        
        for query in queries:
            query_text = query['text']
            relevant_docs = query['relevant_docs']
            
            for model_name, model in models.items():
                if 'DenseRAG' in model_name or 'Streaming' in model_name:
                    # Simulate stale index by filtering corpus
                    stale_corpus = filter_corpus_by_time(corpus, cutoff_time)
                    
                    # Create temporary model with stale corpus
                    if hasattr(model, '_build_index'):
                        # For Dense RAG, rebuild index with stale corpus
                        original_corpus = model.corpus
                        model.corpus = stale_corpus
                        model._build_index()
                        
                        retrieved = model.retrieve(query_text, N=10)
                        
                        # Restore original corpus
                        model.corpus = original_corpus
                        model._build_index()
                    else:
                        retrieved = model.retrieve(query_text, N=10)
                else:
                    # JIT-RAG and BM25 always use live corpus
                    retrieved = model.retrieve(query_text, N=10)
                
                # Calculate nDCG
                ndcg = calculate_ndcg(retrieved, relevant_docs, k=10)
                interval_results[model_name].append(ndcg)
        
        # Average performance for this interval
        for model_name in models.keys():
            avg_ndcg = np.mean(interval_results[model_name])
            results[model_name].append(avg_ndcg)
    
    return results


def ablation_study(corpus: List[Dict[str, Any]],
                   queries: List[Dict[str, Any]],
                   encoder_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                   K: int = 200) -> Dict[str, Dict[str, float]]:
    """
    Perform ablation study to analyze component contributions.
    
    Implements the analysis from Section 6.4 and Figure 5.
    
    Tests three configurations:
    1. Full JIT-RAG (BM25 + Dense Reranking)
    2. BM25-Only (no dense reranking)
    3. Random + Dense Ranker (random candidates + dense reranking)
    
    Args:
        corpus: Document corpus
        queries: List of queries with relevant document IDs
        encoder_model: Encoder model name
        K: Candidate set size
        
    Returns:
        Dictionary mapping configuration names to metric dictionaries
    """
    from models import JITRAG, BM25Baseline
    from sentence_transformers import SentenceTransformer
    
    print("\n=== Ablation Study ===")
    
    # Initialize models
    print("Initializing models...")
    jit_rag = JITRAG(corpus, encoder_model)
    bm25_only = BM25Baseline(corpus)
    encoder = SentenceTransformer(encoder_model)
    
    results = {
        'Full JIT-RAG': [],
        'BM25-Only': [],
        'Random + Dense Ranker': []
    }
    
    print(f"Running ablation study on {len(queries)} queries...")
    
    for query in tqdm(queries):
        query_text = query['text']
        relevant_docs = query['relevant_docs']
        
        # 1. Full JIT-RAG
        retrieved_full = jit_rag.retrieve(query_text, K=K, N=10)
        ndcg_full = calculate_ndcg(retrieved_full, relevant_docs, k=10)
        results['Full JIT-RAG'].append(ndcg_full)
        
        # 2. BM25-Only
        retrieved_bm25 = bm25_only.retrieve(query_text, N=10)
        ndcg_bm25 = calculate_ndcg(retrieved_bm25, relevant_docs, k=10)
        results['BM25-Only'].append(ndcg_bm25)
        
        # 3. Random + Dense Ranker
        # Randomly select K candidates
        random_indices = random.sample(range(len(corpus)), min(K, len(corpus)))
        
        # Encode query
        v_q = encoder.encode(query_text, convert_to_tensor=False, show_progress_bar=False)
        
        # Encode random candidates and compute similarities
        scored_docs = []
        for idx in random_indices:
            doc_text = corpus[idx]['text']
            v_i = encoder.encode(doc_text, convert_to_tensor=False, show_progress_bar=False)
            similarity = np.dot(v_q, v_i) / (np.linalg.norm(v_q) * np.linalg.norm(v_i))
            scored_docs.append((idx, similarity))
        
        # Sort by similarity
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-10
        retrieved_random = [
            {'id': corpus[idx]['id'], 'text': corpus[idx]['text'], 'score': score}
            for idx, score in scored_docs[:10]
        ]
        
        ndcg_random = calculate_ndcg(retrieved_random, relevant_docs, k=10)
        results['Random + Dense Ranker'].append(ndcg_random)
    
    # Calculate average metrics
    avg_results = {}
    for config_name, scores in results.items():
        avg_results[config_name] = {
            'ndcg@10': np.mean(scores),
            'std': np.std(scores)
        }
    
    return avg_results


def governance_overhead_analysis(corpus: List[Dict[str, Any]],
                                 queries: List[Dict[str, Any]],
                                 deletion_percentages: List[float] = [0, 1, 5, 10, 20, 30],
                                 encoder_model: str = 'sentence-transformers/all-MiniLM-L6-v2') -> Dict[str, Any]:
    """
    Analyze governance overhead for data deletion.
    
    Implements the analysis from Section 6.5 and Figure 6.
    
    Measures:
    1. Cost to process deletions (time)
    2. Ongoing query latency overhead (with tombstoning)
    
    Args:
        corpus: Document corpus
        queries: List of queries
        deletion_percentages: List of deletion percentages to test
        encoder_model: Encoder model name
        
    Returns:
        Dictionary with deletion costs and latency overheads
    """
    from models import JITRAG, DenseRAG
    
    print("\n=== Governance Overhead Analysis ===")
    
    results = {
        'deletion_percentages': deletion_percentages,
        'jitrag_deletion_time': [],
        'denserage_reindex_time': [],
        'denserage_tombstone_time': [],
        'jitrag_query_latency': [],
        'denserage_query_latency': []
    }
    
    for pct in tqdm(deletion_percentages, desc="Testing deletion percentages"):
        # Create fresh model instances
        jit_rag = JITRAG(corpus.copy(), encoder_model)
        dense_rag = DenseRAG(corpus.copy(), encoder_model)
        
        # Calculate number of documents to delete
        n_delete = int(len(corpus) * pct / 100)
        docs_to_delete = random.sample([doc['id'] for doc in corpus], n_delete) if n_delete > 0 else []
        
        # Measure deletion time for JIT-RAG
        jitrag_del_times = []
        for doc_id in docs_to_delete:
            del_time = jit_rag.delete_document(doc_id)
            jitrag_del_times.append(del_time)
        
        avg_jitrag_del_time = np.mean(jitrag_del_times) if jitrag_del_times else 0.0
        results['jitrag_deletion_time'].append(avg_jitrag_del_time)
        
        # Measure deletion time for Dense RAG (re-index)
        dense_rag_reindex = DenseRAG(corpus.copy(), encoder_model)
        start_time = time.time()
        for doc_id in docs_to_delete:
            dense_rag_reindex.delete_document(doc_id, method='reindex')
        reindex_time = (time.time() - start_time) / max(len(docs_to_delete), 1)
        results['denserage_reindex_time'].append(reindex_time)
        
        # Measure deletion time for Dense RAG (tombstone)
        tombstone_times = []
        for doc_id in docs_to_delete:
            tomb_time = dense_rag.delete_document(doc_id, method='tombstone')
            tombstone_times.append(tomb_time)
        
        avg_tombstone_time = np.mean(tombstone_times) if tombstone_times else 0.0
        results['denserage_tombstone_time'].append(avg_tombstone_time)
        
        # Measure query latency with tombstones
        jitrag_latencies = []
        denserage_latencies = []
        
        for query in queries[:20]:  # Sample queries for latency measurement
            query_text = query['text']
            
            # JIT-RAG latency
            _, jitrag_lat = measure_latency(jit_rag.retrieve, query_text, K=200, N=10)
            jitrag_latencies.append(jitrag_lat)
            
            # Dense RAG latency (with tombstones)
            _, denserage_lat = measure_latency(dense_rag.retrieve, query_text, N=10)
            denserage_latencies.append(denserage_lat)
        
        results['jitrag_query_latency'].append(np.mean(jitrag_latencies))
        results['denserage_query_latency'].append(np.mean(denserage_latencies))
    
    return results


def plot_staleness_results(results: Dict[str, List[float]], 
                           intervals: List[int],
                           output_path: str = 'staleness_analysis.png'):
    """
    Plot data staleness simulation results (Figure 3 from paper).
    
    Args:
        results: Dictionary mapping model names to performance lists
        intervals: List of re-indexing intervals in hours
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, scores in results.items():
        plt.plot(intervals, scores, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Re-indexing Interval (hours)', fontsize=12)
    plt.ylabel('nDCG@10', fontsize=12)
    plt.title('Performance vs. Information Lag', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Staleness plot saved to {output_path}")


def plot_ablation_results(results: Dict[str, Dict[str, float]],
                          output_path: str = 'ablation_study.png'):
    """
    Plot ablation study results (Figure 5 from paper).
    
    Args:
        results: Dictionary mapping configuration names to metrics
        output_path: Path to save plot
    """
    configs = list(results.keys())
    ndcg_scores = [results[config]['ndcg@10'] for config in configs]
    std_scores = [results[config]['std'] for config in configs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, ndcg_scores, yerr=std_scores, capsize=5, alpha=0.7)
    
    # Color bars
    bars[0].set_color('green')
    bars[1].set_color('orange')
    bars[2].set_color('red')
    
    plt.ylabel('nDCG@10', fontsize=12)
    plt.title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Ablation plot saved to {output_path}")


def plot_governance_overhead(results: Dict[str, Any],
                             output_path: str = 'governance_overhead.png'):
    """
    Plot governance overhead analysis results (Figure 6 from paper).
    
    Args:
        results: Dictionary with deletion costs and latency overheads
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Deletion cost
    ax1.plot(results['deletion_percentages'], results['jitrag_deletion_time'], 
             marker='o', label='JIT-RAG', linewidth=2)
    ax1.plot(results['deletion_percentages'], results['denserage_tombstone_time'], 
             marker='s', label='Dense RAG (Tombstone)', linewidth=2)
    ax1.plot(results['deletion_percentages'], results['denserage_reindex_time'], 
             marker='^', label='Dense RAG (Re-index)', linewidth=2)
    ax1.set_xlabel('Percentage of Deleted Documents (%)', fontsize=11)
    ax1.set_ylabel('Avg Deletion Time (seconds)', fontsize=11)
    ax1.set_title('Cost to Process Deletions', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Query latency overhead
    ax2.plot(results['deletion_percentages'], results['jitrag_query_latency'], 
             marker='o', label='JIT-RAG', linewidth=2)
    ax2.plot(results['deletion_percentages'], results['denserage_query_latency'], 
             marker='s', label='Dense RAG (Tombstone)', linewidth=2)
    ax2.set_xlabel('Percentage of Deleted Documents (%)', fontsize=11)
    ax2.set_ylabel('Avg Query Latency (seconds)', fontsize=11)
    ax2.set_title('Query Latency Overhead', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Governance overhead plot saved to {output_path}")
