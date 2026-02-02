# Semantic Search Engine
**BANA 275 Course Project**
Group 8 Jiahua Jia, Hanyang Gu, Zheng Zhu, Xiangning Li, Mavis Xuan.
## Project Overview
A production-ready semantic search application that uses embeddings and LLM integration to search and analyze documents based on semantic meaning rather than just keyword matching.

## Features

### Core Functionality
- **Semantic Document Search**: Search through documents using natural language queries
- **Embedding-based Similarity**: Uses Sentence Transformers for semantic understanding
- **Top-K Results**: Returns most relevant documents with similarity scores
- **LLM Enhancement**: Summarizes search results using language models

### Domain
[Your chosen domain - update this section]
- Document collection: [Describe your dataset]
- Number of documents: [Min. 100 documents]

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Required Packages
- `sentence-transformers` - For generating semantic embeddings
- `numpy` - Numerical computations
- `scikit-learn` - Similarity calculations
- `torch` - Deep learning framework
- `openai` or `anthropic` - LLM integration (choose one)

## Usage

### Command Line Interface
```bash
python search.py --query "your search query" --top_k 5
```

### Python API
```python
from semantic_search import SemanticSearchEngine

# Initialize search engine
engine = SemanticSearchEngine()

# Index documents
engine.index_documents(your_documents)

# Search
results = engine.search("your query", top_k=5)

# Get LLM summary
summary = engine.summarize_results(results)
```

## Project Structure
```
Prototype_SemanticSearchEngine/
├── README.md                   # This file
├── PROCESS.md                  # Development process documentation
├── ARCHITECTURE.md             # Technical details
├── requirements.txt            # Dependencies
├── TEAM_CONTRIBUTIONS.md       # Team member contributions
├── search.py                   # Main search implementation
├── embeddings.py              # Embedding generation
├── llm_handler.py             # LLM integration
└── data/                      # Document corpus
```

## How It Works

1. **Document Processing**: Load and preprocess documents from your chosen domain
2. **Embedding Generation**: Convert documents to vector embeddings using Sentence Transformers
3. **Similarity Search**: Calculate cosine similarity between query and documents
4. **Ranking**: Return top-K most similar documents
5. **LLM Enhancement**: Summarize or analyze results using LLM

## Implementation Details

### Embedding Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Similarity metric: Cosine similarity

### LLM Integration
- **Option A**: Summarize top-K results

## Code Quality
- Modular, documented code
- Error handling for edge cases
- Type hints and docstrings
- requirements.txt included

## Demo Video
https://youtu.be/JaeaWrh4EeM

## Team Contributions
See `TEAM_CONTRIBUTIONS.md` for individual contributions.

## Deliverables
- ✅ GitHub repository with organized source code
- ✅ README.md (project overview and usage)
- ✅ ARCHITECTURE.md (technical implementation)
- ✅ requirements.txt (dependencies)
- ✅ 2-3 minute demo video
- ✅ Peer assessment forms

## Repository
https://github.com/HanyangGu/Prototype_SemanticSearchEngine

---

**Course**: BANA 275  
**Date**: February 2026
