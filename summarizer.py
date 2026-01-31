import openai
from typing import List, Dict

class ResultsSummarizer:
    """Summarizes search results using OpenAI GPT models"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize summarizer
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use for summarization
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
    
    def summarize_results(self, results: List[Dict], query: str) -> str:
        """
        Generate a summary of the top search results
        
        Args:
            results: List of search result dictionaries
            query: Original search query
            
        Returns:
            Summary text
        """
        if not results:
            return "No results to summarize."
        
        # Prepare context from results
        context = self._prepare_context(results)
        
        # Create prompt
        prompt = f"""You are a helpful assistant that summarizes news articles.

The user searched for: "{query}"

Here are the top {len(results)} most relevant articles:

{context}

Please provide a concise summary (3-4 paragraphs) that:
1. Synthesizes the key information from these articles related to the query
2. Highlights common themes or different perspectives
3. Provides actionable insights or important takeaways
4. Maintains objectivity and accuracy

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, informative summaries of news articles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Unable to generate summary at this time."
    
    def _prepare_context(self, results: List[Dict]) -> str:
        """
        Prepare context string from search results
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, result in enumerate(results, 1):
            # Truncate content to avoid token limits
            content_preview = result['content'][:500]
            
            article_text = f"""
Article {idx}:
Title: {result['title']}
Source: {result['source']}
Published: {result['published']}
Relevance Score: {result['similarity']:.3f}
Content: {content_preview}...
"""
            context_parts.append(article_text)
        
        return "\n".join(context_parts)
    
    def summarize_single_article(self, article: Dict) -> str:
        """
        Generate a summary for a single article
        
        Args:
            article: Article dictionary
            
        Returns:
            Summary text
        """
        prompt = f"""Summarize the following news article in 2-3 sentences:

Title: {article['title']}
Content: {article['content'][:1000]}

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Unable to generate summary."
    
    def generate_insights(self, results: List[Dict], query: str) -> Dict:
        """
        Generate structured insights from search results
        
        Args:
            results: List of search result dictionaries
            query: Original search query
            
        Returns:
            Dictionary with insights
        """
        if not results:
            return {
                'summary': 'No results to analyze.',
                'key_points': [],
                'sources': []
            }
        
        # Get main summary
        summary = self.summarize_results(results, query)
        
        # Extract key information
        sources = list(set(r['source'] for r in results))
        
        # Get top articles by relevance
        top_titles = [r['title'] for r in results[:3]]
        
        insights = {
            'summary': summary,
            'key_articles': top_titles,
            'sources_covered': sources,
            'num_results': len(results),
            'avg_relevance': sum(r['similarity'] for r in results) / len(results)
        }
        
        return insights

# Test the summarizer if run directly
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <openai_api_key>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Sample results for testing
    sample_results = [
        {
            'title': 'AI Advances in Healthcare',
            'content': 'Recent developments in artificial intelligence are transforming healthcare delivery. Machine learning models are now being used to detect diseases earlier and more accurately than traditional methods.',
            'source': 'TechNews',
            'published': '2024-01-15',
            'similarity': 0.85
        },
        {
            'title': 'New AI Regulations Proposed',
            'content': 'Government agencies are proposing new regulations for AI systems to ensure safety and ethical use. The regulations focus on transparency and accountability in AI decision-making.',
            'source': 'Policy Today',
            'published': '2024-01-14',
            'similarity': 0.78
        }
    ]
    
    summarizer = ResultsSummarizer(api_key)
    summary = summarizer.summarize_results(sample_results, "artificial intelligence")
    
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(summary)
