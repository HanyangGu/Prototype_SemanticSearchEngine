#!/usr/bin/env python3
"""
Test Script - Verify all components of the search engine
Run this to check if everything is working correctly
"""

import os
import sys
import json
import numpy as np

def test_header(test_name):
    """Print test header"""
    print(f"\n{'='*80}")
    print(f"Testing: {test_name}")
    print('='*80)

def test_imports():
    """Test if all modules can be imported"""
    test_header("Module Imports")
    
    modules = [
        ('data_collector', 'NewsCollector'),
        ('embeddings_manager', 'EmbeddingsManager'),
        ('search_engine', 'SemanticSearchEngine'),
        ('summarizer', 'ResultsSummarizer'),
    ]
    
    all_passed = True
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - Error: {str(e)}")
            all_passed = False
    
    return all_passed

def test_dependencies():
    """Test if all required packages are installed"""
    test_header("Package Dependencies")
    
    packages = [
        'streamlit',
        'openai', 
        'numpy',
        'feedparser',
        'requests',
        'tqdm'
    ]
    
    all_passed = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  Install missing packages with: pip install -r requirements.txt")
    
    return all_passed

def test_data_collector():
    """Test data collection functionality"""
    test_header("Data Collector")
    
    try:
        from data_collector import NewsCollector
        
        collector = NewsCollector()
        
        # Test text cleaning
        dirty_text = "<p>Hello <b>World</b>!  Extra   spaces.</p>"
        clean_text = collector.clean_text(dirty_text)
        
        assert len(clean_text) > 0, "Clean text should not be empty"
        assert '<' not in clean_text, "HTML tags should be removed"
        print(f"✅ Text cleaning: '{dirty_text}' -> '{clean_text}'")
        
        # Test RSS feed configuration
        assert len(collector.rss_feeds) > 0, "Should have RSS feeds configured"
        print(f"✅ RSS feeds configured: {len(collector.rss_feeds)} feeds")
        
        print("✅ Data Collector tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Data Collector test failed: {str(e)}")
        return False

def test_file_structure():
    """Test if required files and directories exist"""
    test_header("File Structure")
    
    required_files = [
        'app.py',
        'data_collector.py',
        'embeddings_manager.py',
        'search_engine.py',
        'summarizer.py',
        'requirements.txt',
        'README.md',
        'config.py'
    ]
    
    all_exist = True
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - MISSING")
            all_exist = False
    
    # Check data directory
    if os.path.exists('data'):
        print(f"✅ data/ directory exists")
        
        # Check for data files
        data_files = ['articles.json', 'embeddings.npy', 'metadata.json']
        for df in data_files:
            path = f'data/{df}'
            if os.path.exists(path):
                print(f"  ✅ {df}")
            else:
                print(f"  ⚠️  {df} - not generated yet (run setup)")
    else:
        print(f"⚠️  data/ directory not created yet")
    
    return all_exist

def test_cosine_similarity():
    """Test cosine similarity calculation"""
    test_header("Cosine Similarity")
    
    try:
        from search_engine import SemanticSearchEngine
        
        engine = SemanticSearchEngine()
        
        # Test identical vectors
        vec1 = np.array([1, 2, 3], dtype=np.float32)
        vec2 = np.array([1, 2, 3], dtype=np.float32)
        sim = engine.cosine_similarity(vec1, vec2)
        
        assert abs(sim - 1.0) < 0.0001, "Identical vectors should have similarity of 1.0"
        print(f"✅ Identical vectors: similarity = {sim:.4f}")
        
        # Test orthogonal vectors
        vec3 = np.array([1, 0, 0], dtype=np.float32)
        vec4 = np.array([0, 1, 0], dtype=np.float32)
        sim2 = engine.cosine_similarity(vec3, vec4)
        
        assert abs(sim2) < 0.0001, "Orthogonal vectors should have similarity near 0"
        print(f"✅ Orthogonal vectors: similarity = {sim2:.4f}")
        
        # Test batch similarity
        query = np.array([1, 1, 1], dtype=np.float32)
        corpus = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [1, 0, 0]
        ], dtype=np.float32)
        
        sims = engine.cosine_similarity_batch(query, corpus)
        assert len(sims) == 3, "Should return 3 similarity scores"
        assert sims[0] > sims[2], "First vector should be more similar than third"
        print(f"✅ Batch similarity: {sims}")
        
        print("✅ Cosine similarity tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Cosine similarity test failed: {str(e)}")
        return False

def test_search_pipeline():
    """Test the complete search pipeline if data exists"""
    test_header("Search Pipeline (Integration Test)")
    
    if not os.path.exists('data/articles.json') or not os.path.exists('data/embeddings.npy'):
        print("⚠️  Data not found. Run setup first to test search pipeline.")
        print("   You can still use the search engine, just need to collect data and generate embeddings.")
        return True
    
    try:
        # Test loading data
        with open('data/articles.json', 'r') as f:
            articles = json.load(f)
        print(f"✅ Loaded {len(articles)} articles")
        
        embeddings = np.load('data/embeddings.npy')
        print(f"✅ Loaded embeddings with shape {embeddings.shape}")
        
        # Verify data consistency
        assert len(articles) == embeddings.shape[0], "Articles and embeddings must match"
        print(f"✅ Data consistency check passed")
        
        # Verify metadata
        if os.path.exists('data/metadata.json'):
            with open('data/metadata.json', 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata: {metadata['num_embeddings']} embeddings, dimension {metadata['embedding_dim']}")
        
        print("✅ Search pipeline data verified")
        return True
        
    except Exception as e:
        print(f"❌ Search pipeline test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪" * 40)
    print("   News Semantic Search Engine - Test Suite")
    print("🧪" * 40)
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_dependencies()))
    results.append(("Imports", test_imports()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Data Collector", test_data_collector()))
    results.append(("Cosine Similarity", test_cosine_similarity()))
    results.append(("Search Pipeline", test_search_pipeline()))
    
    # Summary
    test_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")
    print('='*80)
    
    if passed == total:
        print("\n🎉 All tests passed! Your search engine is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python quick_start.py' for setup")
        print("  2. Or run 'streamlit run app.py' to launch the interface")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Most likely you need to:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run setup to collect data and generate embeddings")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted. Goodbye! 👋")
        sys.exit(1)
