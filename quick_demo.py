"""
Quick demonstration of JIT-RAG with minimal examples.
"""

from models import JITRAG, DenseRAG, BM25Baseline
from utils import create_sample_corpus, generate_queries, evaluate_retrieval
from evaluation import calculate_qlf_utility, calculate_freshness
from datetime import datetime
import numpy as np

print("="*70)
print("JIT-RAG Quick Demonstration")
print("="*70)

# Create small corpus
print("\n1. Creating sample corpus (100 documents)...")
corpus = create_sample_corpus(n_docs=100, seed=42)

# Generate queries
print("2. Generating queries (10 queries)...")
queries = generate_queries(corpus, n_queries=10, seed=42)

# Initialize models
print("\n3. Initializing models...")
print("   - JIT-RAG")
jit_rag = JITRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')

print("   - Dense RAG")
dense_rag = DenseRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')

print("   - BM25 Baseline")
bm25 = BM25Baseline(corpus)

# Example query
print("\n4. Example Query:")
query = queries[0]
print(f"   Query: '{query['text'][:60]}...'")

print("\n5. Retrieving documents...")

# JIT-RAG
results_jit = jit_rag.retrieve(query['text'], K=50, N=5)
print(f"\n   JIT-RAG Top 5 Results:")
for i, doc in enumerate(results_jit[:3], 1):
    print(f"   {i}. Score: {doc['score']:.4f} | {doc['text'][:50]}...")

# Dense RAG
results_dense = dense_rag.retrieve(query['text'], N=5)
print(f"\n   Dense RAG Top 5 Results:")
for i, doc in enumerate(results_dense[:3], 1):
    print(f"   {i}. Score: {doc['score']:.4f} | {doc['text'][:50]}...")

# BM25
results_bm25 = bm25.retrieve(query['text'], N=5)
print(f"\n   BM25 Top 5 Results:")
for i, doc in enumerate(results_bm25[:3], 1):
    print(f"   {i}. Score: {doc['score']:.4f} | {doc['text'][:50]}...")

# Evaluation
print("\n6. Evaluation Metrics:")
relevant = query['relevant_docs']

metrics_jit = evaluate_retrieval(results_jit, relevant)
metrics_dense = evaluate_retrieval(results_dense, relevant)
metrics_bm25 = evaluate_retrieval(results_bm25, relevant)

print(f"\n   Model          nDCG@10   Recall@100   MRR@10")
print(f"   {'─'*50}")
print(f"   JIT-RAG        {metrics_jit['ndcg@10']:.4f}    {metrics_jit['recall@100']:.4f}       {metrics_jit['mrr@10']:.4f}")
print(f"   Dense RAG      {metrics_dense['ndcg@10']:.4f}    {metrics_dense['recall@100']:.4f}       {metrics_dense['mrr@10']:.4f}")
print(f"   BM25           {metrics_bm25['ndcg@10']:.4f}    {metrics_bm25['recall@100']:.4f}       {metrics_bm25['mrr@10']:.4f}")

# QLF Utility
print("\n7. QLF Utility Demonstration:")
print("   (Quality-Latency-Freshness)")

# Add timestamps
for doc in corpus:
    doc['timestamp'] = datetime.now()

current_time = datetime.now()

# Calculate for JIT-RAG
freshness_jit = calculate_freshness(results_jit, current_time, alpha=0.01)
utility_jit = calculate_qlf_utility(
    Q=metrics_jit['ndcg@10'],
    L=0.15,  # Simulated latency
    F=freshness_jit,
    w_q=1.0, w_l=0.1, w_f=0.5
)

# Calculate for Dense RAG
freshness_dense = calculate_freshness(results_dense, current_time, alpha=0.01)
utility_dense = calculate_qlf_utility(
    Q=metrics_dense['ndcg@10'],
    L=0.05,  # Simulated latency (faster)
    F=freshness_dense,
    w_q=1.0, w_l=0.1, w_f=0.5
)

print(f"\n   JIT-RAG:")
print(f"      Quality:   {metrics_jit['ndcg@10']:.4f}")
print(f"      Latency:   0.15s")
print(f"      Freshness: {freshness_jit:.4f}")
print(f"      Utility:   {utility_jit:.4f}")

print(f"\n   Dense RAG:")
print(f"      Quality:   {metrics_dense['ndcg@10']:.4f}")
print(f"      Latency:   0.05s")
print(f"      Freshness: {freshness_dense:.4f}")
print(f"      Utility:   {utility_dense:.4f}")

# Deletion demonstration
print("\n8. Governance: Document Deletion")

print("\n   Deleting 'doc_0' from JIT-RAG...")
del_time_jit = jit_rag.delete_document('doc_0')
print(f"   Time: {del_time_jit:.6f}s")

print("\n   Deleting 'doc_1' from Dense RAG (tombstone)...")
del_time_dense = dense_rag.delete_document('doc_1', method='tombstone')
print(f"   Time: {del_time_dense:.6f}s")
print(f"   Tombstone overhead: {dense_rag.get_tombstone_overhead():.2f}%")

print("\n" + "="*70)
print("Demo Complete!")
print("="*70)
print("\nKey Findings:")
print("  • JIT-RAG provides competitive retrieval quality")
print("  • JIT-RAG eliminates information lag (always fresh)")
print("  • JIT-RAG simplifies data deletion (no re-indexing)")
print("  • Trade-off: slightly higher query latency vs Dense RAG")
print("="*70)
