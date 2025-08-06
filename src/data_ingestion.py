import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings import get_embeddings
from src.loaders import load_pdf_file
from src.splitters import split_documents
from src.vector_store import build_vector_store, get_vector_store_stats
from src.utils import load_environment, ensure_data_folder

def ingest_data():
    """Main function to ingest data into Pinecone"""
    try:
        print("🚀 Starting data ingestion process...")
        
        # Load environment variables
        load_environment()
        
        # Ensure data folder exists
        ensure_data_folder()
        
        # Load PDF documents
        print("📚 Loading PDF documents...")
        docs = load_pdf_file("Data")
        
        # Split documents into chunks
        print("✂️ Splitting documents...")
        text_chunks = split_documents(docs, chunk_size=500, chunk_overlap=20)
        
        # Initialize embeddings
        print("🔤 Initializing embeddings...")
        embeddings = get_embeddings()
        
        # Build vector store
        print("🏗️ Building vector store...")
        vector_store = build_vector_store(text_chunks, "fintutor", embeddings)
        
        # Get and display stats
        print("📊 Getting vector store statistics...")
        stats = get_vector_store_stats("fintutor")
        print(f"Vector store created with {stats.get('total_vector_count', 0)} vectors")
        
        print("✅ Data ingestion completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data ingestion failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = ingest_data()
    if not success:
        sys.exit(1)
