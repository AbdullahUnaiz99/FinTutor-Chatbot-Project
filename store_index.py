from src.helper import load_pdf_file, text_split, download_hugging_face_embedding
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_api_key


docs = load_pdf_file(data='model/Data/')  
text_chunks = text_split(docs)
embeddings = download_hugging_face_embedding()


pc = pinecone(api_key=pinecone_api_key)

index_name = "fintutor"

pc.create_index(
    index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws",region="us-east-1")
)

 
docsearch=PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name= index_name,
    embedding=embeddings
    )



