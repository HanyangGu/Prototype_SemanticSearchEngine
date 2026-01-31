# Configuration file for News Semantic Search Engine

# OpenAI Settings
EMBEDDING_MODEL = "text-embedding-3-small"  # Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
SUMMARY_MODEL = "gpt-4o-mini"  # Options: gpt-4o-mini, gpt-4o, gpt-3.5-turbo

# Search Settings
DEFAULT_TOP_K = 5  # Default number of search results
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to include in results

# Data Collection Settings
DEFAULT_ARTICLE_COUNT = 150  # Number of articles to collect
BATCH_SIZE = 100  # Batch size for embedding generation

# RSS Feed Sources
RSS_FEEDS = [
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
    'https://techcrunch.com/feed/',
    'https://www.wired.com/feed/rss',
    'https://www.cnet.com/rss/news/',
]

# UI Settings
PAGE_TITLE = "News Semantic Search Engine"
PAGE_ICON = "🔍"
LAYOUT = "wide"

# Data Storage
DATA_DIR = "data"
ARTICLES_FILE = "data/articles.json"
EMBEDDINGS_FILE = "data/embeddings.npy"
METADATA_FILE = "data/metadata.json"
