#!/usr/bin/env python3
"""
Demo Script - Demonstrates the News Semantic Search Engine capabilities
This script runs through the entire pipeline with sample data
"""

import json
import numpy as np
from typing import List, Dict

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def demo_data_collection():
    """Demonstrate data collection"""
    print_section("STEP 1: Data Collection & Preprocessing")
    
    from data_collector import NewsCollector
    
    print("Creating NewsCollector instance...")
    collector = NewsCollector()
    
    print(f"Number of RSS feeds configured: {len(collector.rss_feeds)}")
    print("\nSample feeds:")
    for i, feed in enumerate(collector.rss_feeds[:5], 1):
        print(f"  {i}. {feed}")
    
    print("\n📰 Starting article collection...")
    print("This will fetch articles from multiple news sources...")
    
    articles = collector.collect_news(20)  # Collect just 20 for demo
    
    print(f"\n✅ Collected {len(articles)} articles")
    
    if articles:
        print("\n📄 Sample Article:")
        sample = articles[0]
        print(f"  Title: {sample['title']}")
        print(f"  Source: {sample['source']}")
        print(f"  Published: {sample['published']}")
        print(f"  Content (first 200 chars): {sample['content'][:200]}...")
        print(f"  URL: {sample['url']}")
    
    return articles

def demo_embeddings(api_key: str):
    """Demonstrate embedding generation"""
    print_section("STEP 2: Embedding Generation")
    
    from embeddings_manager import EmbeddingsManager
    
    print(f"Initializing EmbeddingsManager...")
    print(f"Model: text-embedding-3-small")
    
    manager = EmbeddingsManager(api_key)
    
    # Load articles
    try:
        with open('data/articles.json', 'r') as f:
            articles = json.load(f)
        
        print(f"Loaded {len(articles)} articles from disk")
        
        # Show sample text that will be embedded
        sample_text = f"{articles[0]['title']}. {articles[0]['content'][:200]}"
        print(f"\n📝 Sample text for embedding:")
        print(f"  '{sample_text}...'")
        
        print("\n🧠 Generating embeddings...")
        success = manager.generate_and_save_embeddings()
        
        if success:
            print("✅ Embeddings generated successfully!")
            
            # Load and show embedding info
            embeddings = np.load('data/embeddings.npy')
            print(f"\n📊 Embedding Matrix Shape: {embeddings.shape}")
            print(f"  - Number of articles: {embeddings.shape[0]}")
            print(f"  - Embedding dimensions: {embeddings.shape[1]}")
            print(f"  - Total parameters: {embeddings.shape[0] * embeddings.shape[1]:,}")
            
            # Show sample embedding values
            print(f"\n🔢 Sample embedding (first 10 dimensions):")
            print(f"  {embeddings[0][:10]}")
            
            return True
        else:
            print("❌ Failed to generate embeddings")
            return False
            
    except FileNotFoundError:
        print("❌ No articles found. Please run data collection first.")
        return False

def demo_search(api_key: str):
    """Demonstrate semantic search"""
    print_section("STEP 3: Semantic Search with Cosine Similarity")
    
    from search_engine import SemanticSearchEngine
    
    print("Initializing SemanticSearchEngine...")
    engine = SemanticSearchEngine(api_key)
    
    if not engine.load_data():
        print("❌ Failed to load data")
        return
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"\n📊 Search Engine Statistics:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    print(f"  Number of sources: {stats['num_sources']}")
    print(f"  Sources: {', '.join(stats['sources'][:5])}...")
    
    # Perform searches
    queries = [
        "artificial intelligence and machine learning",
        "climate change and environmental policies",
        "economic growth and inflation"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 80)
        
        results = engine.search(query, k=3)
        
        if results:
            for idx, result in enumerate(results, 1):
                print(f"\n  Result {idx}:")
                print(f"    Title: {result['title'][:70]}...")
                print(f"    Source: {result['source']}")
                print(f"    Similarity Score: {result['similarity']:.4f}")
                print(f"    Content Preview: {result['content'][:100]}...")
        else:
            print("  No results found")

def demo_cosine_similarity():
    """Demonstrate cosine similarity calculation"""
    print_section("BONUS: Understanding Cosine Similarity")
    
    print("Cosine similarity measures the angle between two vectors.")
    print("It ranges from 0 (completely different) to 1 (identical).\n")
    
    # Create sample vectors
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([2, 4, 6])  # Similar direction
    vec3 = np.array([1, 0, -1])  # Different direction
    
    # Calculate similarities
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    sim_12 = cosine_sim(vec1, vec2)
    sim_13 = cosine_sim(vec1, vec3)
    
    print("Example vectors:")
    print(f"  Vector A: {vec1}")
    print(f"  Vector B: {vec2} (similar to A)")
    print(f"  Vector C: {vec3} (different from A)")
    
    print(f"\nSimilarity calculations:")
    print(f"  Similarity(A, B) = {sim_12:.4f} ← High (vectors point in same direction)")
    print(f"  Similarity(A, C) = {sim_13:.4f} ← Low (vectors point in different directions)")
    
    print("\nIn our search engine:")
    print("  - Query embedding is compared with all article embeddings")
    print("  - Higher similarity = more relevant article")
    print("  - Results are ranked by similarity score")

def demo_summarization(api_key: str):
    """Demonstrate result summarization"""
    print_section("STEP 4: AI-Powered Summarization")
    
    from summarizer import ResultsSummarizer
    from search_engine import SemanticSearchEngine
    
    print("Initializing components...")
    engine = SemanticSearchEngine(api_key)
    
    if not engine.load_data():
        print("❌ Failed to load data")
        return
    
    summarizer = ResultsSummarizer(api_key)
    
    query = "technology and innovation"
    print(f"\n🔍 Searching for: '{query}'")
    
    results = engine.search(query, k=5)
    
    if results:
        print(f"Found {len(results)} relevant articles")
        print("\n📋 Generating AI summary...")
        
        summary = summarizer.summarize_results(results, query)
        
        print("\n" + "-" * 80)
        print("SUMMARY:")
        print("-" * 80)
        print(summary)
        print("-" * 80)
    else:
        print("No results to summarize")

def main():
    """Run the complete demo"""
    print("\n" + "🔍" * 40)
    print("   News Semantic Search Engine - Interactive Demo")
    print("🔍" * 40)
    
    print("\nThis demo will walk you through each component of the search engine.")
    print("You'll see how data collection, embeddings, search, and summarization work.")
    
    # Check if we need API key
    import os
    if not os.path.exists('data/embeddings.npy'):
        print("\n⚠️  This demo requires OpenAI API access for embeddings and summarization.")
        api_key = input("Please enter your OpenAI API key (or press Enter to run partial demo): ").strip()
    else:
        api_key = input("\nEnter your OpenAI API key (for search and summarization): ").strip()
    
    try:
        # Step 1: Data Collection
        print("\n" + "▶" * 40)
        input("Press Enter to start Step 1: Data Collection...")
        demo_data_collection()
        
        if api_key:
            # Step 2: Embeddings
            print("\n" + "▶" * 40)
            input("Press Enter to continue to Step 2: Embedding Generation...")
            if demo_embeddings(api_key):
                
                # Step 3: Search
                print("\n" + "▶" * 40)
                input("Press Enter to continue to Step 3: Semantic Search...")
                demo_search(api_key)
                
                # Bonus: Cosine Similarity
                print("\n" + "▶" * 40)
                input("Press Enter for a bonus explanation of Cosine Similarity...")
                demo_cosine_similarity()
                
                # Step 4: Summarization
                print("\n" + "▶" * 40)
                input("Press Enter to continue to Step 4: Summarization...")
                demo_summarization(api_key)
        else:
            print("\n⚠️  Skipping embedding-dependent demos (no API key provided)")
            print("You can run the full demo later with: python demo.py")
        
        # Conclusion
        print_section("Demo Complete! 🎉")
        print("You've seen how the entire search engine pipeline works:")
        print("  1. ✅ Collect and preprocess news articles")
        print("  2. ✅ Generate semantic embeddings using OpenAI")
        print("  3. ✅ Perform similarity-based search")
        print("  4. ✅ Generate AI-powered summaries")
        print("\nNext steps:")
        print("  - Run 'streamlit run app.py' to use the web interface")
        print("  - Or run 'python quick_start.py' for guided setup")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
