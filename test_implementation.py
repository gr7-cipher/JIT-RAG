"""
Simple test script to validate the JIT-RAG implementation.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from models import JITRAG, DenseRAG, BM25Baseline, BM25CrossEncoder
        from utils import create_sample_corpus, generate_queries, evaluate_retrieval
        from evaluation import calculate_qlf_utility, calculate_freshness
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_jitrag_basic():
    """Test basic JIT-RAG functionality."""
    print("\nTesting JIT-RAG basic retrieval...")
    try:
        from models import JITRAG
        from utils import create_sample_corpus
        
        # Create small corpus
        corpus = create_sample_corpus(n_docs=50, seed=42)
        
        # Initialize JIT-RAG
        jit_rag = JITRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # Test retrieval
        query = "artificial intelligence and machine learning"
        results = jit_rag.retrieve(query, K=20, N=5)
        
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        assert all('id' in doc and 'text' in doc and 'score' in doc for doc in results), "Missing fields in results"
        
        print(f"✓ JIT-RAG retrieved {len(results)} documents")
        print(f"  Top result score: {results[0]['score']:.4f}")
        return True
    except Exception as e:
        print(f"✗ JIT-RAG test failed: {e}")
        traceback.print_exc()
        return False


def test_baselines():
    """Test baseline models."""
    print("\nTesting baseline models...")
    try:
        from models import BM25Baseline, DenseRAG
        from utils import create_sample_corpus
        
        corpus = create_sample_corpus(n_docs=50, seed=42)
        query = "artificial intelligence"
        
        # Test BM25
        bm25 = BM25Baseline(corpus)
        results_bm25 = bm25.retrieve(query, N=5)
        assert len(results_bm25) == 5, "BM25 retrieval failed"
        print("✓ BM25 baseline works")
        
        # Test Dense RAG
        dense_rag = DenseRAG(corpus, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
        results_dense = dense_rag.retrieve(query, N=5)
        assert len(results_dense) == 5, "Dense RAG retrieval failed"
        print("✓ Dense RAG baseline works")
        
        return True
    except Exception as e:
        print(f"✗ Baseline test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation functions."""
    print("\nTesting evaluation functions...")
    try:
        from utils import calculate_ndcg, evaluate_retrieval
        from evaluation import calculate_qlf_utility, calculate_freshness
        from datetime import datetime
        
        # Test nDCG
        retrieved = [{'id': f'doc_{i}', 'text': 'test', 'score': 1.0-i*0.1} for i in range(10)]
        relevant = {'doc_0', 'doc_2', 'doc_5'}
        ndcg = calculate_ndcg(retrieved, relevant, k=10)
        assert 0 <= ndcg <= 1, f"Invalid nDCG: {ndcg}"
        print(f"✓ nDCG calculation works: {ndcg:.4f}")
        
        # Test QLF utility
        utility = calculate_qlf_utility(Q=0.85, L=0.15, F=0.95)
        print(f"✓ QLF utility calculation works: {utility:.4f}")
        
        # Test freshness
        docs_with_time = [
            {'id': 'doc_0', 'text': 'test', 'timestamp': datetime.now()}
        ]
        freshness = calculate_freshness(docs_with_time, datetime.now())
        assert 0 <= freshness <= 1, f"Invalid freshness: {freshness}"
        print(f"✓ Freshness calculation works: {freshness:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        traceback.print_exc()
        return False


def test_deletion():
    """Test document deletion functionality."""
    print("\nTesting document deletion...")
    try:
        from models import JITRAG, DenseRAG
        from utils import create_sample_corpus
        
        corpus = create_sample_corpus(n_docs=50, seed=42)
        
        # Test JIT-RAG deletion
        jit_rag = JITRAG(corpus.copy(), encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
        initial_size = len(jit_rag.corpus)
        del_time = jit_rag.delete_document('doc_0')
        assert len(jit_rag.corpus) == initial_size - 1, "JIT-RAG deletion failed"
        print(f"✓ JIT-RAG deletion works (time: {del_time:.6f}s)")
        
        # Test Dense RAG deletion (tombstone)
        dense_rag = DenseRAG(corpus.copy(), encoder_model_name='sentence-transformers/all-MiniLM-L6-v2')
        del_time = dense_rag.delete_document('doc_1', method='tombstone')
        assert len(dense_rag.tombstones) == 1, "Dense RAG tombstone deletion failed"
        print(f"✓ Dense RAG tombstone deletion works (time: {del_time:.6f}s)")
        
        return True
    except Exception as e:
        print(f"✗ Deletion test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("JIT-RAG Implementation Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("JIT-RAG Basic", test_jitrag_basic),
        ("Baselines", test_baselines),
        ("Evaluation", test_evaluation),
        ("Deletion", test_deletion)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
