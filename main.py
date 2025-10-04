"""
JIT-RAG Main Demonstration Script

This script demonstrates the usage of JIT-RAG and compares it against baseline models.
It also provides examples of running the evaluation framework.
"""

import argparse
import json
from datetime import datetime, timedelta
import numpy as np

from models import JITRAG, DenseRAG, BM25Baseline, BM25CrossEncoder
from utils import (
    create_sample_corpus, generate_queries, evaluate_retrieval,
    format_results_table, measure_latency, create_temporal_corpus
)
from evaluation import (
    calculate_qlf_utility, calculate_freshness, data_staleness_simulation,
    ablation_study, governance_overhead_analysis, plot_staleness_results,
    plot_ablation_results, plot_governance_overhead
)


def demo_basic_retrieval():
    """Demonstrate basic retrieval with JIT-RAG and baselines."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Retrieval Comparison")
    print("="*70)
    
    # Create sample corpus
    print("\n1. Creating sample corpus...")
    corpus = create_sample_corpus(n_docs=1000, seed=42)
    print(f"   Created corpus with {len(corpus)} documents")
    
    # Generate queries
    print("\n2. Generating queries...")
    queries = generate_queries(corpus, n_queries=50, seed=42)
    print(f"   Generated {len(queries)} queries")
    
    # Initialize models
    print("\n3. Initializing models...")
    print("   - JIT-RAG (with BM25 + on-the-fly dense embeddings)")
    jit_rag = JITRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    print("   - Dense RAG (with pre-computed FAISS index)")
    dense_rag = DenseRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    print("   - BM25 Baseline (sparse retrieval only)")
    bm25_baseline = BM25Baseline(corpus)
    
    # Run evaluation
    print("\n4. Running evaluation on queries...")
    results = {
        'JIT-RAG': {'ndcg@10': [], 'recall@100': [], 'mrr@10': [], 'latency': []},
        'Dense RAG': {'ndcg@10': [], 'recall@100': [], 'mrr@10': [], 'latency': []},
        'BM25': {'ndcg@10': [], 'recall@100': [], 'mrr@10': [], 'latency': []}
    }
    
    for query in queries:
        query_text = query['text']
        relevant_docs = query['relevant_docs']
        
        # JIT-RAG
        retrieved_jit, lat_jit = measure_latency(jit_rag.retrieve, query_text, K=200, N=10)
        metrics_jit = evaluate_retrieval(retrieved_jit, relevant_docs)
        results['JIT-RAG']['ndcg@10'].append(metrics_jit['ndcg@10'])
        results['JIT-RAG']['recall@100'].append(metrics_jit['recall@100'])
        results['JIT-RAG']['mrr@10'].append(metrics_jit['mrr@10'])
        results['JIT-RAG']['latency'].append(lat_jit * 1000)  # Convert to ms
        
        # Dense RAG
        retrieved_dense, lat_dense = measure_latency(dense_rag.retrieve, query_text, N=10)
        metrics_dense = evaluate_retrieval(retrieved_dense, relevant_docs)
        results['Dense RAG']['ndcg@10'].append(metrics_dense['ndcg@10'])
        results['Dense RAG']['recall@100'].append(metrics_dense['recall@100'])
        results['Dense RAG']['mrr@10'].append(metrics_dense['mrr@10'])
        results['Dense RAG']['latency'].append(lat_dense * 1000)
        
        # BM25
        retrieved_bm25, lat_bm25 = measure_latency(bm25_baseline.retrieve, query_text, N=10)
        metrics_bm25 = evaluate_retrieval(retrieved_bm25, relevant_docs)
        results['BM25']['ndcg@10'].append(metrics_bm25['ndcg@10'])
        results['BM25']['recall@100'].append(metrics_bm25['recall@100'])
        results['BM25']['mrr@10'].append(metrics_bm25['mrr@10'])
        results['BM25']['latency'].append(lat_bm25 * 1000)
    
    # Calculate averages
    print("\n5. Results:")
    avg_results = {}
    for model_name, metrics in results.items():
        avg_results[model_name] = {
            'nDCG@10': np.mean(metrics['ndcg@10']),
            'Recall@100': np.mean(metrics['recall@100']),
            'MRR@10': np.mean(metrics['mrr@10']),
            'Latency (ms)': np.mean(metrics['latency'])
        }
    
    print("\n" + format_results_table(avg_results))
    
    return corpus, queries


def demo_qlf_utility(corpus, queries):
    """Demonstrate QLF utility calculation."""
    print("\n" + "="*70)
    print("DEMO 2: Quality-Latency-Freshness (QLF) Utility")
    print("="*70)
    
    # Add timestamps to corpus
    print("\n1. Adding timestamps to corpus...")
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    temporal_corpus = create_temporal_corpus(corpus, start_date, end_date)
    current_time = end_date
    
    # Initialize models
    print("\n2. Initializing models...")
    jit_rag = JITRAG(temporal_corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
    dense_rag = DenseRAG(temporal_corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Run sample query
    print("\n3. Running sample query and calculating QLF utility...")
    query = queries[0]
    query_text = query['text']
    relevant_docs = query['relevant_docs']
    
    # JIT-RAG
    retrieved_jit, latency_jit = measure_latency(jit_rag.retrieve, query_text, K=200, N=10)
    quality_jit = evaluate_retrieval(retrieved_jit, relevant_docs)['ndcg@10']
    freshness_jit = calculate_freshness(retrieved_jit, current_time, alpha=0.01)
    utility_jit = calculate_qlf_utility(quality_jit, latency_jit, freshness_jit)
    
    # Dense RAG
    retrieved_dense, latency_dense = measure_latency(dense_rag.retrieve, query_text, N=10)
    quality_dense = evaluate_retrieval(retrieved_dense, relevant_docs)['ndcg@10']
    freshness_dense = calculate_freshness(retrieved_dense, current_time, alpha=0.01)
    utility_dense = calculate_qlf_utility(quality_dense, latency_dense, freshness_dense)
    
    print("\n4. QLF Utility Scores:")
    print(f"\n   JIT-RAG:")
    print(f"      Quality (Q):   {quality_jit:.4f}")
    print(f"      Latency (L):   {latency_jit:.4f} seconds")
    print(f"      Freshness (F): {freshness_jit:.4f}")
    print(f"      Utility (U):   {utility_jit:.4f}")
    
    print(f"\n   Dense RAG:")
    print(f"      Quality (Q):   {quality_dense:.4f}")
    print(f"      Latency (L):   {latency_dense:.4f} seconds")
    print(f"      Freshness (F): {freshness_dense:.4f}")
    print(f"      Utility (U):   {utility_dense:.4f}")
    
    print(f"\n   Utility Improvement: {((utility_jit - utility_dense) / utility_dense * 100):.2f}%")


def demo_ablation_study(corpus, queries):
    """Demonstrate ablation study."""
    print("\n" + "="*70)
    print("DEMO 3: Ablation Study")
    print("="*70)
    
    print("\nTesting three configurations:")
    print("  1. Full JIT-RAG (BM25 + Dense Reranking)")
    print("  2. BM25-Only (no dense reranking)")
    print("  3. Random + Dense Ranker (random candidates + dense reranking)")
    
    results = ablation_study(corpus, queries[:20], K=200)  # Use subset for demo
    
    print("\nResults:")
    for config_name, metrics in results.items():
        print(f"\n  {config_name}:")
        print(f"    nDCG@10: {metrics['ndcg@10']:.4f} ± {metrics['std']:.4f}")
    
    # Plot results
    plot_ablation_results(results, output_path='ablation_study.png')


def demo_staleness_simulation(corpus, queries):
    """Demonstrate data staleness simulation."""
    print("\n" + "="*70)
    print("DEMO 4: Data Staleness Simulation")
    print("="*70)
    
    # Add timestamps
    print("\n1. Creating temporal corpus...")
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    temporal_corpus = create_temporal_corpus(corpus, start_date, end_date)
    
    # Initialize models
    print("\n2. Initializing models...")
    models = {
        'JIT-RAG': JITRAG(temporal_corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2'),
        'Dense RAG': DenseRAG(temporal_corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
    }
    
    # Run simulation
    print("\n3. Running staleness simulation...")
    intervals = [1, 6, 12, 24, 48, 72]
    results = data_staleness_simulation(models, temporal_corpus, queries[:10], intervals, end_date)
    
    print("\n4. Results:")
    for model_name, scores in results.items():
        print(f"\n   {model_name}:")
        for interval, score in zip(intervals, scores):
            print(f"      {interval}h lag: nDCG@10 = {score:.4f}")
    
    # Plot results
    plot_staleness_results(results, intervals, output_path='staleness_analysis.png')


def demo_governance_overhead(corpus, queries):
    """Demonstrate governance overhead analysis."""
    print("\n" + "="*70)
    print("DEMO 5: Governance Overhead Analysis")
    print("="*70)
    
    print("\nAnalyzing cost of data deletion and query latency overhead...")
    
    deletion_percentages = [0, 5, 10, 20]
    results = governance_overhead_analysis(corpus, queries[:10], deletion_percentages)
    
    print("\nResults:")
    print("\nDeletion Time (seconds):")
    for pct, jit_time, dense_time in zip(
        results['deletion_percentages'],
        results['jitrag_deletion_time'],
        results['denserage_reindex_time']
    ):
        print(f"  {pct}% deleted:")
        print(f"    JIT-RAG:   {jit_time:.6f}s")
        print(f"    Dense RAG: {dense_time:.6f}s (re-index)")
    
    print("\nQuery Latency (seconds):")
    for pct, jit_lat, dense_lat in zip(
        results['deletion_percentages'],
        results['jitrag_query_latency'],
        results['denserage_query_latency']
    ):
        print(f"  {pct}% deleted:")
        print(f"    JIT-RAG:   {jit_lat:.4f}s")
        print(f"    Dense RAG: {dense_lat:.4f}s (with tombstones)")
    
    # Plot results
    plot_governance_overhead(results, output_path='governance_overhead.png')


def main():
    """Main function to run demonstrations."""
    parser = argparse.ArgumentParser(description='JIT-RAG Demonstration')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'basic', 'qlf', 'ablation', 'staleness', 'governance'],
                       help='Which demo to run')
    parser.add_argument('--corpus-size', type=int, default=1000,
                       help='Size of sample corpus')
    parser.add_argument('--n-queries', type=int, default=50,
                       help='Number of queries to generate')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("JIT-RAG: Just-in-Time Embedding Architecture")
    print("Demonstration Script")
    print("="*70)
    
    # Create corpus and queries
    print(f"\nInitializing with corpus size: {args.corpus_size}")
    print(f"Number of queries: {args.n_queries}")
    
    corpus = create_sample_corpus(n_docs=args.corpus_size, seed=42)
    queries = generate_queries(corpus, n_queries=args.n_queries, seed=42)
    
    # Run selected demos
    if args.demo == 'all' or args.demo == 'basic':
        corpus, queries = demo_basic_retrieval()
    
    if args.demo == 'all' or args.demo == 'qlf':
        demo_qlf_utility(corpus, queries)
    
    if args.demo == 'all' or args.demo == 'ablation':
        demo_ablation_study(corpus, queries)
    
    if args.demo == 'all' or args.demo == 'staleness':
        demo_staleness_simulation(corpus, queries)
    
    if args.demo == 'all' or args.demo == 'governance':
        demo_governance_overhead(corpus, queries)
    
    print("\n" + "="*70)
    print("Demonstration Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
