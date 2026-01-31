import openai
import numpy as np
import json
import os
from typing import List, Dict
import time
from tqdm import tqdm

class EmbeddingsManager:
    """Manages embedding generation and storage using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize embeddings manager
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text using OpenAI API
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Clean and truncate text if too long (max ~8000 tokens)
            text = text[:8000] if len(text) > 8000 else text
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            return response.data[0].embedding
        
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            return None
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Create embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        print(f"Creating embeddings for {len(texts)} texts...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            
            try:
                # Clean texts
                cleaned_batch = [text[:8000] if len(text) > 8000 else text for text in batch]
                
                response = self.client.embeddings.create(
                    input=cleaned_batch,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nError in batch {i//batch_size}: {str(e)}")
                # Add None placeholders for failed embeddings
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    def generate_and_save_embeddings(self) -> bool:
        """
        Generate embeddings for all articles and save to disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load articles
            with open('data/articles.json', 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            print(f"Loaded {len(articles)} articles")
            
            # Prepare texts for embedding (title + content)
            texts = [f"{article['title']}. {article['content']}" for article in articles]
            
            # Generate embeddings
            embeddings = self.create_embeddings_batch(texts)
            
            # Filter out None values
            valid_embeddings = []
            valid_articles = []
            
            for embedding, article in zip(embeddings, articles):
                if embedding is not None:
                    valid_embeddings.append(embedding)
                    valid_articles.append(article)
            
            print(f"Successfully created {len(valid_embeddings)} embeddings")
            
            if len(valid_embeddings) == 0:
                print("❌ No valid embeddings created. Check your API key and quota.")
                return False
            
            # Convert to numpy array
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            
            # Save embeddings
            os.makedirs('data', exist_ok=True)
            np.save('data/embeddings.npy', embeddings_array)
            
            # Save updated articles list
            with open('data/articles.json', 'w', encoding='utf-8') as f:
                json.dump(valid_articles, f, ensure_ascii=False, indent=2)
            
            # Save metadata
            metadata = {
                'model': self.model,
                'num_embeddings': len(valid_embeddings),
                'embedding_dim': len(valid_embeddings[0]),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('data/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved embeddings to data/embeddings.npy")
            print(f"Shape: {embeddings_array.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return False
    
    def load_embeddings(self) -> np.ndarray:
        """
        Load embeddings from disk
        
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = np.load('data/embeddings.npy')
            print(f"Loaded embeddings with shape: {embeddings.shape}")
            return embeddings
        except FileNotFoundError:
            print("No embeddings file found")
            return None
    
    def load_articles(self) -> List[Dict]:
        """
        Load articles from disk
        
        Returns:
            List of article dictionaries
        """
        try:
            with open('data/articles.json', 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Loaded {len(articles)} articles")
            return articles
        except FileNotFoundError:
            print("No articles file found")
            return None

# Test the embeddings manager if run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embeddings_manager.py <openai_api_key>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    manager = EmbeddingsManager(api_key)
    success = manager.generate_and_save_embeddings()
    
    if success:
        print("\n✅ Embeddings generated successfully!")
    else:
        print("\n❌ Failed to generate embeddings")
