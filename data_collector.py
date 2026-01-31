import requests
import feedparser
import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict
import time

class NewsCollector:
    """Collects and preprocesses news articles from various sources"""
    
    def __init__(self):
        self.articles = []
        # Popular RSS feeds for news collection
        self.rss_feeds = [
            'http://rss.cnn.com/rss/cnn_topstories.rss',
            'http://feeds.bbci.co.uk/news/world/rss.xml',
            'https://feeds.reuters.com/reuters/topNews',
            'https://www.theguardian.com/world/rss',
            'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
            'https://www.npr.org/rss/rss.php?id=1001',
            'http://feeds.foxnews.com/foxnews/latest',
            'https://www.aljazeera.com/xml/rss/all.xml',
            'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
            'http://feeds.washingtonpost.com/rss/world',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.bloomberg.com/feed/podcast/businessweek-daily.xml',
            'https://techcrunch.com/feed/',
            'https://www.wired.com/feed/rss',
            'https://www.cnet.com/rss/news/',
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fetch_from_rss(self, feed_url: str, max_articles: int = 20) -> List[Dict]:
        """Fetch articles from an RSS feed"""
        articles = []
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:max_articles]:
                # Extract article data
                title = self.clean_text(entry.get('title', ''))
                
                # Get content from summary or description
                content = entry.get('summary', entry.get('description', ''))
                content = self.clean_text(content)
                
                # Skip if content is too short
                if len(content) < 100:
                    continue
                
                # Parse published date
                published = entry.get('published', entry.get('updated', ''))
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        published = str(published)
                
                article = {
                    'title': title,
                    'content': content,
                    'url': entry.get('link', ''),
                    'source': feed.feed.get('title', 'Unknown'),
                    'published': published,
                    'id': entry.get('id', entry.get('link', ''))
                }
                
                articles.append(article)
                
        except Exception as e:
            print(f"Error fetching from {feed_url}: {str(e)}")
        
        return articles
    
    def remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # Create a normalized title for comparison
            normalized_title = article['title'].lower().strip()
            
            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles
    
    def collect_news(self, target_count: int = 150) -> List[Dict]:
        """Collect news articles from multiple sources"""
        all_articles = []
        articles_per_feed = max(20, target_count // len(self.rss_feeds) + 10)
        
        print(f"Collecting articles from {len(self.rss_feeds)} RSS feeds...")
        
        for idx, feed_url in enumerate(self.rss_feeds, 1):
            print(f"Fetching from feed {idx}/{len(self.rss_feeds)}: {feed_url}")
            articles = self.fetch_from_rss(feed_url, articles_per_feed)
            all_articles.extend(articles)
            
            # Small delay to be respectful to servers
            time.sleep(0.5)
            
            # Check if we have enough articles
            if len(all_articles) >= target_count * 1.5:  # Get extra for deduplication
                break
        
        print(f"Collected {len(all_articles)} articles before deduplication")
        
        # Remove duplicates
        unique_articles = self.remove_duplicates(all_articles)
        print(f"After deduplication: {len(unique_articles)} unique articles")
        
        # Take only the required number
        final_articles = unique_articles[:target_count]
        
        # Save to file
        self.save_articles(final_articles)
        
        return final_articles
    
    def save_articles(self, articles: List[Dict]):
        """Save articles to JSON file"""
        os.makedirs('data', exist_ok=True)
        
        with open('data/articles.json', 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(articles)} articles to data/articles.json")
    
    def load_articles(self) -> List[Dict]:
        """Load articles from JSON file"""
        try:
            with open('data/articles.json', 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Loaded {len(articles)} articles from data/articles.json")
            return articles
        except FileNotFoundError:
            print("No saved articles found")
            return []

# Test the collector if run directly
if __name__ == "__main__":
    collector = NewsCollector()
    articles = collector.collect_news(150)
    print(f"\nSuccessfully collected {len(articles)} articles!")
    print(f"\nSample article:")
    if articles:
        print(f"Title: {articles[0]['title']}")
        print(f"Source: {articles[0]['source']}")
        print(f"Content preview: {articles[0]['content'][:200]}...")
