from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
import time

def initialize_pinecone():
    """Initialize Pinecone client using v3 API"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    try:
        pc = Pinecone(api_key=api_key)
        print("Pinecone client initialized successfully")
        return pc
    except Exception as e:
        raise Exception(f"Failed to initialize Pinecone: {e}")

def ensure_index_exists(pc, index_name, dimension=384):
    """Create Pinecone index if it doesn't exist"""
    try:
        # Get list of existing indexes
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes]
        
        if index_name not in index_names:
            print(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                print("Waiting for index to be ready...")
                time.sleep(1)
            
            print(f"Index {index_name} created successfully")
        else:
            print(f"Index {index_name} already exists")
        
        return pc.Index(index_name)
        
    except Exception as e:
        raise Exception(f"Failed to ensure index exists: {e}")

def build_vector_store(docs, index_name, embeddings):
    """Build vector store from documents"""
    if not docs:
        raise ValueError("No documents provided for vector store creation")
    
    try:
        pc = initialize_pinecone()
        ensure_index_exists(pc, index_name)
        
        print(f"Building vector store with {len(docs)} documents...")
        vector_store = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name
        )
        
        print("Vector store built successfully")
        return vector_store
        
    except Exception as e:
        raise Exception(f"Failed to build vector store: {e}")

def load_vector_store(index_name, embeddings):
    """Load existing vector store"""
    try:
        pc = initialize_pinecone()
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes]
        
        if index_name not in index_names:
            raise Exception(f"Index '{index_name}' does not exist. Please run data ingestion first.")
        
        # Load existing vector store
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        
        print(f"Vector store loaded successfully from index: {index_name}")
        return vector_store
        
    except Exception as e:
        raise Exception(f"Failed to load vector store: {e}")

def get_vector_store_stats(index_name):
    """Get statistics about the vector store"""
    try:
        pc = initialize_pinecone()
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        raise Exception(f"Failed to get vector store stats: {e}")
