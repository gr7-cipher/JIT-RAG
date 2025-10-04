"""
Utility Functions for JIT-RAG Implementation

This module provides helper functions for data loading, metric calculation,
and corpus management.
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple
import json
from datetime import datetime, timedelta
import random


def load_corpus_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load corpus from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of documents with 'id' and 'text' fields
    """
    corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus.append(doc)
    return corpus


def save_corpus_to_jsonl(corpus: List[Dict[str, Any]], filepath: str):
    """
    Save corpus to JSONL file.
    
    Args:
        corpus: List of documents
        filepath: Path to output JSONL file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def create_sample_corpus(n_docs: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Create a sample corpus for testing.
    
    Args:
        n_docs: Number of documents to generate
        seed: Random seed
        
    Returns:
        List of sample documents
    """
    random.seed(seed)
    np.random.seed(seed)
    
    topics = [
        "artificial intelligence and machine learning",
        "climate change and environmental policy",
        "quantum computing and physics",
        "biotechnology and genetic engineering",
        "space exploration and astronomy",
        "renewable energy and sustainability",
        "cybersecurity and data privacy",
        "blockchain and cryptocurrency",
        "neuroscience and brain research",
        "robotics and automation"
    ]
    
    corpus = []
    for i in range(n_docs):
        topic = random.choice(topics)
        doc_text = f"This is a document about {topic}. " * random.randint(3, 10)
        doc_text += f"Document ID: {i}. "
        doc_text += f"Additional content related to {topic} and its applications."
        
        corpus.append({
            'id': f'doc_{i}',
            'text': doc_text,
            'timestamp': datetime.now() - timedelta(hours=random.randint(0, 720))
        })
    
    return corpus


def calculate_ndcg(retrieved: List[Dict[str, Any]], 
                   relevant: Set[str], 
                   k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@k).
    
    Args:
        retrieved: List of retrieved documents (ordered by relevance)
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        nDCG@k score
    """
    if len(relevant) == 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        if doc['id'] in relevant:
            # Gain = 1 for relevant, 0 for non-relevant
            # Discount = 1 / log2(i + 2)
            dcg += 1.0 / np.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_recall(retrieved: List[Dict[str, Any]], 
                     relevant: Set[str], 
                     k: int = 100) -> float:
    """
    Calculate Recall@k.
    
    Args:
        retrieved: List of retrieved documents
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Recall@k score
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_ids = {doc['id'] for doc in retrieved[:k]}
    intersection = retrieved_ids & relevant
    
    return len(intersection) / len(relevant)


def calculate_mrr(retrieved: List[Dict[str, Any]], 
                  relevant: Set[str], 
                  k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR@k).
    
    Args:
        retrieved: List of retrieved documents
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        MRR@k score (reciprocal rank of first relevant document)
    """
    for i, doc in enumerate(retrieved[:k]):
        if doc['id'] in relevant:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_precision(retrieved: List[Dict[str, Any]], 
                        relevant: Set[str], 
                        k: int = 10) -> float:
    """
    Calculate Precision@k.
    
    Args:
        retrieved: List of retrieved documents
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0
    
    retrieved_ids = {doc['id'] for doc in retrieved[:k]}
    intersection = retrieved_ids & relevant
    
    return len(intersection) / k


def evaluate_retrieval(retrieved: List[Dict[str, Any]], 
                       relevant: Set[str]) -> Dict[str, float]:
    """
    Evaluate retrieval results with multiple metrics.
    
    Args:
        retrieved: List of retrieved documents
        relevant: Set of relevant document IDs
        
    Returns:
        Dictionary with metric scores
    """
    return {
        'ndcg@10': calculate_ndcg(retrieved, relevant, k=10),
        'recall@100': calculate_recall(retrieved, relevant, k=100),
        'mrr@10': calculate_mrr(retrieved, relevant, k=10),
        'precision@10': calculate_precision(retrieved, relevant, k=10)
    }


def create_temporal_corpus(base_corpus: List[Dict[str, Any]], 
                           start_date: datetime, 
                           end_date: datetime) -> List[Dict[str, Any]]:
    """
    Add timestamps to corpus for temporal analysis.
    
    Args:
        base_corpus: Corpus without timestamps
        start_date: Start date for timestamp range
        end_date: End date for timestamp range
        
    Returns:
        Corpus with timestamps
    """
    temporal_corpus = []
    total_hours = int((end_date - start_date).total_seconds() / 3600)
    
    for doc in base_corpus:
        doc_copy = doc.copy()
        random_hours = random.randint(0, total_hours)
        doc_copy['timestamp'] = start_date + timedelta(hours=random_hours)
        temporal_corpus.append(doc_copy)
    
    return temporal_corpus


def filter_corpus_by_time(corpus: List[Dict[str, Any]], 
                          cutoff_time: datetime) -> List[Dict[str, Any]]:
    """
    Filter corpus to only include documents before cutoff time.
    
    Args:
        corpus: Corpus with timestamps
        cutoff_time: Cutoff datetime
        
    Returns:
        Filtered corpus
    """
    return [doc for doc in corpus if doc.get('timestamp', datetime.now()) <= cutoff_time]


def calculate_document_age(doc: Dict[str, Any], 
                           current_time: datetime) -> float:
    """
    Calculate document age in hours.
    
    Args:
        doc: Document with timestamp
        current_time: Current time
        
    Returns:
        Age in hours
    """
    timestamp = doc.get('timestamp', current_time)
    age_seconds = (current_time - timestamp).total_seconds()
    return age_seconds / 3600  # Convert to hours


def calculate_average_age(documents: List[Dict[str, Any]], 
                          current_time: datetime) -> float:
    """
    Calculate average age of documents in hours.
    
    Args:
        documents: List of documents with timestamps
        current_time: Current time
        
    Returns:
        Average age in hours
    """
    if len(documents) == 0:
        return 0.0
    
    total_age = sum(calculate_document_age(doc, current_time) for doc in documents)
    return total_age / len(documents)


def generate_queries(corpus: List[Dict[str, Any]], 
                     n_queries: int = 100, 
                     seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic queries from corpus.
    
    Args:
        corpus: Document corpus
        n_queries: Number of queries to generate
        seed: Random seed
        
    Returns:
        List of queries with relevant document IDs
    """
    random.seed(seed)
    queries = []
    
    for i in range(n_queries):
        # Select a random document as relevant
        relevant_doc = random.choice(corpus)
        
        # Extract keywords from document text (simple approach)
        words = relevant_doc['text'].split()
        query_words = random.sample(words, min(5, len(words)))
        query_text = ' '.join(query_words)
        
        queries.append({
            'id': f'query_{i}',
            'text': query_text,
            'relevant_docs': {relevant_doc['id']}
        })
    
    return queries


def load_ms_marco_sample(n_docs: int = 10000) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load a sample from MS MARCO dataset.
    
    Note: This is a placeholder. In practice, you would load from actual MS MARCO files.
    
    Args:
        n_docs: Number of documents to load
        
    Returns:
        Tuple of (corpus, queries)
    """
    # Placeholder implementation
    corpus = create_sample_corpus(n_docs)
    queries = generate_queries(corpus, n_queries=100)
    
    return corpus, queries


def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format evaluation results as a table.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        
    Returns:
        Formatted table string
    """
    # Get all metrics
    metrics = list(next(iter(results.values())).keys())
    
    # Create header
    header = "| Model | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    
    # Create rows
    rows = []
    for model_name, model_results in results.items():
        row = f"| {model_name} | "
        row += " | ".join([f"{model_results[m]:.4f}" for m in metrics])
        row += " |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)


def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure execution time of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
        
    Returns:
        Tuple of (function result, latency in seconds)
    """
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    latency = end_time - start_time
    
    return result, latency
