# Architecture

## Overview

The News Semantic Search Engine is composed of modular components that together provide a full pipeline from raw news feeds to AI‑powered search and summarization.  The system follows a **collect → embed → search → summarize** workflow and persists intermediate artefacts on disk so that costly operations only need to be performed once.  The diagram below describes the high‑level interactions between components.

1. **Data collection** – the `NewsCollector` fetches articles from a configurable list of RSS feeds, cleans HTML and other noise, deduplicates similar titles and writes the resulting articles to `data/articles.json`
2. **Embedding generation** – the `EmbeddingsManager` loads the articles, concatenates titles and content, and uses OpenAI’s embedding API (default model: `text‑embedding‑3‑small`) to create high‑dimensional vectors.  Embeddings are generated in batches to reduce rate‑limit overhead and are saved as a NumPy array (`data/embeddings.npy`) with accompanying metadata
3. **Semantic search** – the `SemanticSearchEngine` loads embeddings and articles from disk.  At query time it embeds the user’s query using the same model, computes cosine similarity between the query vector and each article vector, ranks the results and returns the top k articles; it can also filter by a similarity threshold and expose basic statistics
4. **Results summarization** – the `ResultsSummarizer` uses GPT models (default: `gpt‑4o‑mini`) to synthesize a coherent summary of the top search results, or to summarize a single article.  It can also produce structured insights such as key articles and average relevance.
5. **User interface** – `app.py` exposes the pipeline through a Streamlit web application.  The sidebar provides controls for data collection and embedding generation, while the main pane offers a search bar and displays the summary and results

## Components

### 1. Data Collection (`data_collector.py`)

- Maintains a list of RSS feed URLs covering major news outlets
- Uses **feedparser** and **requests** to fetch feeds and extract titles, summaries and publication dates
- Cleans text by removing HTML tags, URLs and excessive punctuation
- Removes duplicates by normalizing titles and filtering out short or repeated headlines.
- Saves the cleaned articles to `data/articles.json` for downstream processing.

### 2. Embedding Manager (`embeddings_manager.py`)

- Initializes an OpenAI client with the user’s API key and selected embedding model.
- Provides `create_embedding` for single texts and `create_embeddings_batch` to process large batches with delays to respect rate limits.
- The `generate_and_save_embeddings` method loads articles, concatenates each article’s title and content, generates embeddings, filters out failures, saves the embeddings matrix to `data/embeddings.npy`, writes a pruned `articles.json`, and records metadata including model name and embedding dimension.

### 3. Semantic Search Engine (`search_engine.py`)

- Loads embeddings, articles and optional metadata from disk.
- Computes cosine similarity between a query vector and the corpus using vector normalization and dot products.
- The `search` method embeds the query, computes similarity scores, sorts indices and prepares a list of result dictionaries containing title, content, URL, source and similarity score.
- `search_with_threshold` filters results by a minimum similarity threshold, and `get_statistics` reports the number of articles, embedding dimension and unique sources.

### 4. Summarizer (`summarizer.py`)

- Uses GPT models via the OpenAI chat API to summarize multiple articles into a concise multi‑paragraph summary.  The prompt instructs the model to synthesize key information, highlight themes and provide insights.
- The `_prepare_context` helper constructs a context string containing truncated content for each article.
- Also supports summarizing a single article and generating structured insights, including a list of top article titles and average similarity.

### 5. User Interface (`app.py`)

- Configures a Streamlit page with a title and wide layout.
- Uses session state to persist the search engine instance and API key between interactions.
- The sidebar is divided into sections for **Collect News Data**, **Generate Embeddings** and **Reload Data**.  Users specify the number of articles, input an API key and trigger embedding generation; status messages inform the user of progress.
- The main pane displays a search box and lets the user choose the number of results.  Once a query is entered, the engine retrieves results, displays a GPT‑generated summary and shows each article in expandable sections with source, date and content preview.
- A welcome message guides first‑time users through the three‑step setup when no data is available.

### 6. Utilities

- **Quick Start (`quick_start.py`)** – a command‑line helper that checks for missing packages, creates the `data` folder, calls the data collector, prompts the user for their API key to generate embeddings and then launches the Streamlit app.  It allows users to run individual steps or the full setup.
- **Tests (`test.py`)** – a suite of unit and integration tests that verify module imports, dependency installation, file structure, cosine similarity calculations and data consistency, and summarise results at the end.

## Data Flow

Below is the typical sequence of operations:

1. **Setup** – run `python quick_start.py` or `streamlit run app.py`.  Install dependencies and collect data as needed.
2. **Data collection** – articles are fetched and saved to JSON (`data/articles.json`).
3. **Embedding** – the embedding manager reads articles, sends batches to the OpenAI API and writes embeddings (`data/embeddings.npy`) and metadata (`data/metadata.json`).
4. **Search** – the search engine loads embeddings and articles.  When a user enters a query, it embeds the query, computes cosine similarity, sorts the scores and returns the most similar articles.
5. **Summarization** – the summarizer takes the top results and uses GPT to produce a concise multi‑paragraph summary and optional structured insights.
6. **Presentation** – results and summaries are displayed via Streamlit with interactive controls.

## Extending the system

- **Adding sources** – modify the `RSS_FEEDS` list in `config.py` to include additional RSS feed URLs.
- **Changing models** – set different values for `EMBEDDING_MODEL` or `SUMMARY_MODEL` in `config.py` to experiment with other OpenAI models.
- **Adjusting search** – tweak `DEFAULT_TOP_K` or `SIMILARITY_THRESHOLD` to control the number and quality of results.
- **Batching** – increase or decrease the `BATCH_SIZE` for embedding generation depending on your API quota and latency requirements.

