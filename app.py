import streamlit as st
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from datetime import datetime

# Import our custom modules
from data_collector import NewsCollector
from embeddings_manager import EmbeddingsManager
from search_engine import SemanticSearchEngine
from summarizer import ResultsSummarizer

# Page configuration
st.set_page_config(
    page_title="News Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

def initialize_engine(api_key=None):
    """Initialize the search engine with pre-computed embeddings"""
    try:
        search_engine = SemanticSearchEngine(api_key)
        if search_engine.load_data():
            st.session_state.search_engine = search_engine
            st.session_state.embeddings_loaded = True
            return True
        return False
    except Exception as e:
        st.error(f"Error initializing search engine: {str(e)}")
        return False

def main():
    # Header
    st.title("🔍 News Semantic Search Engine")
    st.markdown("Search through news articles using AI-powered semantic understanding")
    
    # Sidebar for setup and configuration
    with st.sidebar:
        st.header("⚙️ Setup & Configuration")
        
        # Check if data exists
        data_exists = os.path.exists('data/articles.json') and os.path.exists('data/embeddings.npy')
        
        if data_exists:
            st.success("✅ Data ready")
            if st.button("🔄 Reload Data"):
                st.session_state.embeddings_loaded = False
                st.session_state.search_engine = None
                st.rerun()
        else:
            st.warning("⚠️ No data found")
        
        st.markdown("---")
        
        # Data collection section
        st.subheader("1️⃣ Collect News Data")
        num_articles = st.number_input("Number of articles", min_value=100, max_value=500, value=150)
        
        if st.button("📰 Collect & Process Articles"):
            with st.spinner("Collecting news articles..."):
                collector = NewsCollector()
                articles = collector.collect_news(num_articles)
                st.success(f"✅ Collected {len(articles)} articles")
        
        st.markdown("---")
        
        # Embeddings generation section
        st.subheader("2️⃣ Generate Embeddings")
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('api_key', ''))
        
        if st.button("🧠 Generate Embeddings"):
            if not api_key:
                st.error("Please provide OpenAI API key")
            elif not os.path.exists('data/articles.json'):
                st.error("Please collect articles first")
            else:
                with st.spinner("Generating embeddings... This may take a few minutes"):
                    st.session_state.api_key = api_key  # Save API key
                    embeddings_manager = EmbeddingsManager(api_key)
                    success = embeddings_manager.generate_and_save_embeddings()
                    if success:
                        st.success("✅ Embeddings generated successfully")
                        st.session_state.embeddings_loaded = False
                    else:
                        st.error("Failed to generate embeddings")
        
        st.markdown("---")
        st.info("💡 After setup, use the search box above to find relevant news articles")
    
    # Main search interface
    if not st.session_state.embeddings_loaded:
        if data_exists:
            with st.spinner("Loading search engine..."):
                initialize_engine(st.session_state.get('api_key'))
    
    if st.session_state.embeddings_loaded and st.session_state.search_engine:
        # Check if API key is available
        if not st.session_state.get('api_key'):
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable search")
            return
        
        # Search interface
        st.markdown("### 🔎 Search News Articles")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter your search query", placeholder="e.g., climate change policies, AI developments, economic trends...")
        with col2:
            top_k = st.selectbox("Results", [3, 5, 10], index=1)
        
        if query:
            with st.spinner("Searching..."):
                # Perform search
                results = st.session_state.search_engine.search(query, k=top_k)
                
                if results:
                    # Display summary
                    st.markdown("### 📋 Summary")
                    summarizer = ResultsSummarizer(st.session_state.search_engine.embeddings_manager.api_key)
                    summary = summarizer.summarize_results(results, query)
                    st.info(summary)
                    
                    st.markdown("---")
                    
                    # Display individual results
                    st.markdown(f"### 📰 Top {len(results)} Results")
                    
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"**{idx}. {result['title']}** (Similarity: {result['similarity']:.3f})", expanded=(idx == 1)):
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.markdown(f"**Source:** {result.get('source', 'N/A')}")
                                st.markdown(f"**Date:** {result.get('published', 'N/A')}")
                            
                            with col_b:
                                if result.get('url'):
                                    st.markdown(f"[🔗 Read Full Article]({result['url']})")
                            
                            st.markdown("**Content Preview:**")
                            # Show first 500 characters
                            preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                            st.markdown(preview)
                else:
                    st.warning("No results found. Try a different query.")
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to the News Semantic Search Engine! 👋
        
        This AI-powered search engine uses semantic understanding to find relevant news articles.
        
        **To get started:**
        1. Use the sidebar to collect news articles (minimum 100)
        2. Enter your OpenAI API key and generate embeddings
        3. Start searching with natural language queries!
        
        **Features:**
        - 🧠 Semantic search using OpenAI embeddings
        - 📊 Cosine similarity-based ranking
        - 📝 AI-generated summaries of search results
        - 🎯 Configurable number of results
        """)

if __name__ == "__main__":
    main()
