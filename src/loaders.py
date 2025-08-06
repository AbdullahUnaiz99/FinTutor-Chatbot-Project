import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_pdf_file(data_folder):
    """Load all PDF files in a folder using DirectoryLoader & PyPDFLoader."""
    # Normalize the folder path
    data_folder = os.path.normpath(data_folder)
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Directory not found: '{data_folder}'")
    
    # Check if there are any PDF files in the directory
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: '{data_folder}'")
    
    try:
        loader = DirectoryLoader(
            data_folder, 
            glob="*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()
        
        if not docs:
            raise ValueError("No documents loaded from the PDF files")
        
        print(f"Successfully loaded {len(docs)} documents from {len(pdf_files)} PDF files")
        return docs
        
    except Exception as e:
        raise Exception(f"Failed to load PDF files: {e}")

