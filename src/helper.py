from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Extract the Data From the PDF File
def load_pdf_file(data):

    if not os.path.exists(data):
        raise FileNotFoundError(f"Directory not found: '{data}'")
        
    loader = DirectoryLoader(
                            data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    docs = loader.load()

    return docs



# Split the Data into Chunks

def text_split(extr_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extr_data)
    return text_chunks


# Download the Embedding model from Hugging Face
def download_hugging_face_embedding():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

    