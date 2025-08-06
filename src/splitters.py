from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs, chunk_size=500, chunk_overlap=20):
    """Split documents into smaller chunks for better retrieval"""
    if not docs:
        raise ValueError("No documents provided for splitting")
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text_chunks = splitter.split_documents(docs)
        
        if not text_chunks:
            raise ValueError("No text chunks were created from documents")
        
        print(f"Split {len(docs)} documents into {len(text_chunks)} chunks")
        return text_chunks
        
    except Exception as e:
        raise Exception(f"Failed to split documents: {e}")
