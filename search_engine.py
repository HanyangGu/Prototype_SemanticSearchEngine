import numpy as np
import json
from typing import List, Dict, Tuple
from embeddings_manager import EmbeddingsManager

class SemanticSearchEngine:
    """Semantic search engine using cosine similarity"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize search engine
        
        Args:
            api_key: OpenAI API key (required for query embedding)
        """
        self.embeddings = None
        self.articles = None
        self.embeddings_manager = None
        
        if api_key:
            self.embeddings_manager = EmbeddingsManager(api_key)
    
    def load_data(self) -> bool:
        """
        Load embeddings and articles from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load embeddings
            self.embeddings = np.load('data/embeddings.npy')
            
            # Load articles
            with open('data/articles.json', 'r', encoding='utf-8') as f:
                self.articles = json.load(f)
            
            # Load metadata
            try:
                with open('data/metadata.json', 'r') as f:
                    metadata = json.load(f)
                    print(f"Loaded data - Model: {metadata['model']}, "
                          f"Embeddings: {metadata['num_embeddings']}, "
                          f"Dimension: {metadata['embedding_dim']}")
            except:
                pass
            
            print(f"Search engine ready with {len(self.articles)} articles")
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        
        # Calculate dot product
        similarity = np.dot(vec1_norm, vec2_norm)
        
        return float(similarity)
    
    def cosine_similarity_batch(self, query_embedding: np.ndarray, 
                               corpus_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and all corpus embeddings
        
        Args:
            query_embedding: Query vector
            corpus_embeddings: Matrix of corpus vectors
            
        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Normalize corpus embeddings
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        corpus_normalized = corpus_embeddings / corpus_norms
        
        # Calculate similarities (dot product)
        similarities = np.dot(corpus_normalized, query_norm)
        
        return similarities
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for top-K most relevant articles
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing article data and similarity scores
        """
        if self.embeddings is None or self.articles is None:
            print("Data not loaded. Call load_data() first.")
            return []
        
        if self.embeddings_manager is None:
            print("Embeddings manager not initialized. Provide API key.")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embeddings_manager.create_embedding(query)
            
            if query_embedding is None:
                print("Failed to create query embedding")
                return []
            
            query_embedding = np.array(query_embedding)
            
            # Calculate similarities
            similarities = self.cosine_similarity_batch(query_embedding, self.embeddings)
            
            # Get top-K indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Prepare results
            results = []
            for idx in top_k_indices:
                result = {
                    'title': self.articles[idx]['title'],
                    'content': self.articles[idx]['content'],
                    'url': self.articles[idx]['url'],
                    'source': self.articles[idx]['source'],
                    'published': self.articles[idx]['published'],
                    'similarity': float(similarities[idx]),
                    'rank': len(results) + 1
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
    
    def search_with_threshold(self, query: str, k: int = 5, 
                             threshold: float = 0.3) -> List[Dict]:
        """
        Search with a minimum similarity threshold
        
        Args:
            query: Search query
            k: Maximum number of results to return
            threshold: Minimum similarity score (0 to 1)
            
        Returns:
            List of dictionaries containing article data and similarity scores
        """
        results = self.search(query, k)
        
        # Filter by threshold
        filtered_results = [r for r in results if r['similarity'] >= threshold]
        
        return filtered_results
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the search engine
        
        Returns:
            Dictionary with statistics
        """
        if self.embeddings is None or self.articles is None:
            return {}
        
        return {
            'total_articles': len(self.articles),
            'embedding_dimension': self.embeddings.shape[1],
            'sources': list(set(article['source'] for article in self.articles)),
            'num_sources': len(set(article['source'] for article in self.articles))
        }

# Test the search engine if run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python search_engine.py <openai_api_key> [query]")
        sys.exit(1)
    
    api_key = sys.argv[1]
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "artificial intelligence"
    
    # Initialize and load
    engine = SemanticSearchEngine(api_key)
    
    if engine.load_data():
        print(f"\n🔍 Searching for: '{query}'")
        print("=" * 80)
        
        results = engine.search(query, k=5)
        
        if results:
            for result in results:
                print(f"\n{result['rank']}. {result['title']}")
                print(f"   Source: {result['source']} | Similarity: {result['similarity']:.3f}")
                print(f"   {result['content'][:150]}...")
        else:
            print("No results found")
        
        # Print statistics
        stats = engine.get_statistics()
        print("\n" + "=" * 80)
        print(f"📊 Statistics:")
        print(f"   Total articles: {stats['total_articles']}")
        print(f"   Sources: {stats['num_sources']}")
        print(f"   Embedding dimension: {stats['embedding_dimension']}")
