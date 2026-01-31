#!/usr/bin/env python3
"""
Quick Start Script for News Semantic Search Engine
This script automates the setup process
"""

import sys
import os
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def check_dependencies():
    """Check if all required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        'streamlit', 'openai', 'numpy', 'feedparser', 'requests', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed!")
    else:
        print("\n✅ All dependencies satisfied!")

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    os.makedirs('data', exist_ok=True)
    print("✅ Created data directory")

def collect_articles():
    """Collect news articles"""
    print_header("Collecting News Articles")
    
    from data_collector import NewsCollector
    
    print("Fetching articles from RSS feeds...")
    collector = NewsCollector()
    articles = collector.collect_news(150)
    
    print(f"\n✅ Successfully collected {len(articles)} articles!")
    return len(articles) > 0

def generate_embeddings():
    """Generate embeddings with user's API key"""
    print_header("Generating Embeddings")
    
    api_key = input("Please enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Skipping embedding generation.")
        return False
    
    from embeddings_manager import EmbeddingsManager
    
    print("Generating embeddings... This may take a few minutes.")
    manager = EmbeddingsManager(api_key)
    success = manager.generate_and_save_embeddings()
    
    if success:
        print("✅ Embeddings generated successfully!")
    else:
        print("❌ Failed to generate embeddings")
    
    return success

def launch_app():
    """Launch the Streamlit app"""
    print_header("Launching Application")
    
    print("Starting Streamlit app...")
    print("The app will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

def main():
    """Main setup function"""
    print("\n" + "🔍" * 40)
    print("   News Semantic Search Engine - Quick Start")
    print("🔍" * 40)
    
    print("\nThis script will help you set up the search engine.")
    print("You can run individual steps or complete the full setup.\n")
    
    print("Options:")
    print("1. Full setup (recommended for first time)")
    print("2. Just collect articles")
    print("3. Just generate embeddings")
    print("4. Launch app (skip setup)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Full setup
        check_dependencies()
        create_directories()
        
        if collect_articles():
            if generate_embeddings():
                print("\n✨ Setup complete! Launching application...")
                input("Press Enter to continue...")
                launch_app()
            else:
                print("\n⚠️  Setup incomplete. You can generate embeddings later from the UI.")
                print("You can still launch the app and complete setup there.")
                
                if input("Launch app anyway? (y/n): ").lower() == 'y':
                    launch_app()
        else:
            print("❌ Failed to collect articles. Please try again.")
    
    elif choice == '2':
        # Just collect articles
        check_dependencies()
        create_directories()
        collect_articles()
        print("\n✅ Article collection complete!")
    
    elif choice == '3':
        # Just generate embeddings
        check_dependencies()
        if os.path.exists('data/articles.json'):
            generate_embeddings()
        else:
            print("❌ No articles found. Please collect articles first (option 2).")
    
    elif choice == '4':
        # Just launch app
        check_dependencies()
        launch_app()
    
    elif choice == '5':
        print("Goodbye! 👋")
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check the error message and try again.")
