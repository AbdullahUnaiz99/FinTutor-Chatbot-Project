from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """Use HuggingFace embeddings"""
    try:
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
    except Exception as e:
        raise Exception(f"Failed to load embeddings: {e}")


