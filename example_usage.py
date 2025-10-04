"""
Example usage of JIT-RAG with custom data.
"""

from models import JITRAG
from utils import load_corpus_from_jsonl

# Load corpus from JSONL file
print("Loading corpus from example_corpus.jsonl...")
corpus = load_corpus_from_jsonl('example_corpus.jsonl')
print(f"Loaded {len(corpus)} documents\n")

# Initialize JIT-RAG
print("Initializing JIT-RAG...")
jit_rag = JITRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
print("JIT-RAG ready!\n")

# Example queries
queries = [
    "What is artificial intelligence?",
    "Tell me about renewable energy",
    "How does quantum computing work?",
    "What are the applications of CRISPR?"
]

print("="*70)
print("Running Example Queries")
print("="*70)

for i, query in enumerate(queries, 1):
    print(f"\nQuery {i}: '{query}'")
    print("-" * 70)
    
    # Retrieve documents
    results = jit_rag.retrieve(query, K=10, N=3)
    
    # Display results
    for rank, doc in enumerate(results, 1):
        print(f"\n  Rank {rank} (Score: {doc['score']:.4f})")
        print(f"  ID: {doc['id']}")
        print(f"  Text: {doc['text'][:100]}...")

print("\n" + "="*70)
print("Example Complete!")
print("="*70)
