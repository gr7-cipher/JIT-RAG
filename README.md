# JIT-RAG: Just-in-Time Embedding Architecture

A comprehensive Python implementation of the research paper **"JIT-RAG: A Just-in-Time Embedding Architecture for Fresh, Compliant, and Scalable Retrieval-Augmented Generation"**.

## Overview

JIT-RAG is a novel Retrieval-Augmented Generation (RAG) architecture that addresses critical challenges in dynamic environments by computing embeddings "just-in-time" rather than maintaining a static pre-computed vector index. This approach provides:

- **Maximum Data Freshness**: Always queries the live corpus with near-zero information lag
- **Simplified Compliance**: Easy data deletion without complex re-indexing or tombstoning
- **Competitive Quality**: Retrieval performance comparable to Dense RAG
- **Efficient Latency**: Fast query processing through a two-stage pipeline

## Architecture

JIT-RAG implements a three-stage pipeline:

1. **Stage 1: Candidate Generation** - Uses BM25 sparse retrieval to select top-K candidates from the live corpus
2. **Stage 2: JIT Semantic Reranking** - Computes dense embeddings on-the-fly for query and candidates
3. **Stage 3: Result Finalization** - Re-ranks candidates by cosine similarity and returns top-N results

This procedural hybrid approach eliminates the need for a persistent dense vector index while maintaining high retrieval quality.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `torch` - Deep learning framework
- `sentence-transformers` - Pre-trained embedding models
- `faiss-cpu` - Vector similarity search (for baseline comparisons)
- `rank-bm25` - BM25 sparse retrieval
- `numpy`, `matplotlib`, `tqdm` - Scientific computing and utilities

## Quick Start

### Basic Usage

```python
from models import JITRAG
from utils import create_sample_corpus

# Create a sample corpus
corpus = create_sample_corpus(n_docs=1000)

# Initialize JIT-RAG
jit_rag = JITRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')

# Retrieve documents
query = "artificial intelligence and machine learning"
results = jit_rag.retrieve(query, K=200, N=10)

# Display results
for i, doc in enumerate(results, 1):
    print(f"{i}. Score: {doc['score']:.4f}")
    print(f"   Text: {doc['text'][:100]}...")
```

### Running Demonstrations

The `main.py` script provides comprehensive demonstrations:

```bash
# Run all demonstrations
python main.py --demo all

# Run specific demonstrations
python main.py --demo basic          # Basic retrieval comparison
python main.py --demo qlf            # QLF utility calculation
python main.py --demo ablation       # Ablation study
python main.py --demo staleness      # Data staleness simulation
python main.py --demo governance     # Governance overhead analysis

# Customize corpus size and query count
python main.py --corpus-size 5000 --n-queries 100
```

## Data Format

### Corpus Format

Documents should be provided as a list of dictionaries with the following structure:

```python
corpus = [
    {
        'id': 'doc_0',
        'text': 'This is the document text content...',
        'timestamp': datetime(2024, 1, 1, 12, 0, 0)  # Optional, for temporal analysis
    },
    {
        'id': 'doc_1',
        'text': 'Another document with relevant information...',
        'timestamp': datetime(2024, 1, 2, 14, 30, 0)
    },
    # ... more documents
]
```

### JSONL Format

You can also load/save corpora in JSONL format:

```python
from utils import load_corpus_from_jsonl, save_corpus_to_jsonl

# Load corpus
corpus = load_corpus_from_jsonl('corpus.jsonl')

# Save corpus
save_corpus_to_jsonl(corpus, 'output_corpus.jsonl')
```

Example JSONL file:
```json
{"id": "doc_0", "text": "Document content here..."}
{"id": "doc_1", "text": "Another document..."}
```

## Models

### JITRAG

The main JIT-RAG implementation with on-the-fly embedding computation.

```python
from models import JITRAG

jit_rag = JITRAG(
    corpus=corpus,
    encoder_model_name='sentence-transformers/all-MiniLM-L6-v2'
)

results = jit_rag.retrieve(query, K=200, N=10)
```

**Parameters:**
- `K`: Number of candidates to retrieve in Stage 1 (default: 200)
- `N`: Number of final results to return (default: 10)

### DenseRAG

Conventional Dense RAG baseline with pre-computed FAISS index.

```python
from models import DenseRAG

dense_rag = DenseRAG(
    corpus=corpus,
    encoder_model_name='sentence-transformers/all-MiniLM-L6-v2',
    index_type='hnsw'  # or 'flat'
)

results = dense_rag.retrieve(query, N=10)
```

### BM25Baseline

Pure sparse retrieval using BM25.

```python
from models import BM25Baseline

bm25 = BM25Baseline(corpus=corpus)
results = bm25.retrieve(query, N=10)
```

### BM25CrossEncoder

High-quality baseline using BM25 + Cross-Encoder reranking.

```python
from models import BM25CrossEncoder

bm25_cross = BM25CrossEncoder(
    corpus=corpus,
    cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2'
)

results = bm25_cross.retrieve(query, K=1000, N=10)
```

## Evaluation Framework

### QLF Utility Function

The Quality-Latency-Freshness (QLF) utility function provides a holistic evaluation metric:

```python
from evaluation import calculate_qlf_utility, calculate_freshness
from datetime import datetime

# Calculate utility
utility = calculate_qlf_utility(
    Q=0.85,        # Quality (e.g., nDCG@10)
    L=0.15,        # Latency in seconds
    F=0.95,        # Freshness score
    w_q=1.0,       # Quality weight
    w_l=0.1,       # Latency weight (penalizes 100ms by 0.01)
    w_f=0.5        # Freshness weight
)

# Calculate freshness from retrieved documents
freshness = calculate_freshness(
    documents=retrieved_docs,
    current_time=datetime.now(),
    alpha=0.01  # Decay constant (per hour)
)
```

### Retrieval Metrics

```python
from utils import evaluate_retrieval

metrics = evaluate_retrieval(
    retrieved=results,
    relevant={'doc_5', 'doc_12', 'doc_23'}  # Set of relevant doc IDs
)

print(f"nDCG@10: {metrics['ndcg@10']:.4f}")
print(f"Recall@100: {metrics['recall@100']:.4f}")
print(f"MRR@10: {metrics['mrr@10']:.4f}")
```

## Experimental Analysis

### Ablation Study

Analyze the contribution of each component:

```python
from evaluation import ablation_study, plot_ablation_results

results = ablation_study(
    corpus=corpus,
    queries=queries,
    K=200
)

plot_ablation_results(results, output_path='ablation_study.png')
```

Tests three configurations:
1. **Full JIT-RAG**: BM25 + Dense Reranking
2. **BM25-Only**: No dense reranking
3. **Random + Dense Ranker**: Random candidates + dense reranking

### Data Staleness Simulation

Measure performance degradation due to information lag:

```python
from evaluation import data_staleness_simulation, plot_staleness_results

models = {
    'JIT-RAG': jit_rag,
    'Dense RAG': dense_rag
}

results = data_staleness_simulation(
    models=models,
    corpus=temporal_corpus,
    queries=queries,
    reindex_intervals=[1, 6, 12, 24, 48, 72]  # hours
)

plot_staleness_results(results, intervals=[1, 6, 12, 24, 48, 72])
```

### Governance Overhead Analysis

Quantify the cost of data deletion and compliance:

```python
from evaluation import governance_overhead_analysis, plot_governance_overhead

results = governance_overhead_analysis(
    corpus=corpus,
    queries=queries,
    deletion_percentages=[0, 1, 5, 10, 20, 30]
)

plot_governance_overhead(results)
```

Measures:
- **Deletion time**: Time to remove documents
- **Query latency overhead**: Impact of tombstoning on query performance

## Key Parameters

### Model Configuration

- **Encoder Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Candidate Size (K)**: 200 (optimal from ablation studies)
- **Final Results (N)**: 10
- **FAISS Index**: HNSW for Dense RAG

### QLF Weights

- **w_q** (Quality): 1.0
- **w_l** (Latency): 0.1 (penalizes each 100ms by 0.01 utility points)
- **w_f** (Freshness): 0.5

### Freshness Decay

- **α** (Alpha): 0.01 per hour

## File Structure

```
jit-rag/
├── models.py              # Core models: JITRAG, DenseRAG, BM25Baseline, BM25CrossEncoder
├── evaluation.py          # QLF utility, experimental analysis functions
├── utils.py               # Helper functions, metrics, data loading
├── main.py                # Demonstration script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Implementation Details

### Algorithm 1: JIT-RAG Query Processing

The core retrieval algorithm follows the paper's Algorithm 1:

1. **Sparse Retrieval**: BM25 retrieves top-K candidates from live corpus
2. **Query Encoding**: Encode query to dense vector v_q
3. **Candidate Encoding**: For each of K candidates, encode document to v_i
4. **Similarity Computation**: Calculate cosine similarity between v_q and each v_i
5. **Re-ranking**: Sort candidates by similarity score
6. **Return**: Return top-N documents

### Complexity Analysis

**Query Time Complexity:**
- JIT-RAG: O(K · L²_d) where K is candidate size and L_d is document length
- Dense RAG: O(L²_q + log|D|) where L_q is query length and |D| is corpus size

**Update/Deletion Complexity:**
- JIT-RAG: O(1) or O(log|D|) - simple deletion from source and BM25 index
- Dense RAG (Re-index): O(|D| · L²_d) - full corpus re-embedding
- Dense RAG (Tombstone): O(1) - but introduces query latency overhead

## Datasets

The implementation supports various datasets:

### Built-in Sample Data

```python
from utils import create_sample_corpus, generate_queries

corpus = create_sample_corpus(n_docs=1000, seed=42)
queries = generate_queries(corpus, n_queries=100, seed=42)
```

### Real Datasets

The paper uses:
- **MS MARCO Passage Ranking**: Large-scale passage retrieval
- **Natural Questions (NQ)**: Question-answering dataset
- **BEIR Benchmark**: 18 diverse retrieval datasets
- **Simulated Streaming News**: Jozef Stefan Institute corpus (2014-2022)

To use real datasets, install additional dependencies:
```bash
pip install datasets ir-datasets
```

## Performance Expectations

Based on the paper's findings:

### Retrieval Quality
- **JIT-RAG**: nDCG@10 ≈ 0.85-0.90 (highly competitive with Dense RAG)
- **Dense RAG**: nDCG@10 ≈ 0.87-0.92
- **BM25**: nDCG@10 ≈ 0.70-0.75

### Latency
- **JIT-RAG**: 150-300ms (depends on K and document length)
- **Dense RAG**: 10-50ms (fast ANN search)
- **BM25 + Cross-Encoder**: 500-1000ms (expensive cross-encoder)

### Freshness
- **JIT-RAG**: Information lag ≈ 0 (always queries live data)
- **Dense RAG**: Information lag = T_reindex / 2 (average)

### Governance
- **JIT-RAG**: O(1) deletion, zero query overhead
- **Dense RAG**: O(|D| · L²_d) re-index or constant tombstone overhead

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{jitrag2024,
  title={JIT-RAG: A Just-in-Time Embedding Architecture for Fresh, Compliant, and Scalable Retrieval-Augmented Generation},
  author={Almobydeen, Shahed and Irjoob, Ahmad and Bentahar, Jamal and Rjoub, Gaith and Kassaymeh, Sofian},
  year={2024}
}
```

## License

This implementation is provided for research and educational purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional embedding models
- Integration with real-world datasets
- Optimization of BM25 indexing
- Distributed computing support
- Additional evaluation metrics

## Contact

For questions or issues, please open an issue on the repository.

## Acknowledgments

This implementation is based on the research paper "JIT-RAG: A Just-in-Time Embedding Architecture for Fresh, Compliant, and Scalable Retrieval-Augmented Generation" by Almobydeen et al.

Special thanks to the authors for their innovative approach to addressing data freshness and compliance challenges in RAG systems.
