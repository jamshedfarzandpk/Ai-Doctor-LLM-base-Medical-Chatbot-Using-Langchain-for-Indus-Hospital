from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import PineconeApiException,Pinecone,ServerlessSpec
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
# Ensure you have your OpenAI API key set as an environment variable (e.g., OPENAI_API_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#use when files are small

def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



## use this function when file size high no of pages are  > 4000 Batch Process PDF Files
def batch_process_pdfs(directory_path, batch_size=100, chunk_size=1000, chunk_overlap=100):
    """
    Process PDF files in batches of pages, extracting text and splitting it into chunks.
    This version loads and processes documents in batches to manage memory efficiently.

    Args:
        directory_path (str): Path to the directory containing PDF files.
        batch_size (int): Number of pages to process in each batch.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: List of Document objects containing text chunks from all processed files.
    """
    # Initialize the DirectoryLoader to get all documents (pages)
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    all_documents = loader.load()
    total_pages = len(all_documents)
    print(f"Total pages found: {total_pages}")

    all_text_chunks = []

    # Iterate through documents in batches of pages
    for i in range(0, total_pages, batch_size):

        batch_documents = all_documents[i:i + batch_size]

        print(f"Processing batch of pages: {i} to {min(i + batch_size, total_pages)}...")

        # NOTE: `filter_to_minimal_docs` is not a standard function.
        # If you have a custom implementation, you can call it here.
        # For now, we will assume this is not needed and remove the call.

        # Split the documents into chunks
        filter_data = filter_to_minimal_docs(batch_documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

        batch_chunks = text_splitter.split_documents(filter_data)
        #print(f"processed batch of pages: {filter_data} ")
        add_documents_to_pinecone(batch_chunks)
        all_text_chunks.extend(batch_chunks)

    return all_text_chunks
#Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    try:
        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
        return embeddings
    except Exception as e:
        # Handle other general exceptions (e.g., network issues)
        return print(f"❌ An unexpected error occurred: {e}")
def add_documents_to_pinecone(documents, index_name="ai-doctor-llm-medical-chatbot", dimension=384):
    """
    Add documents to a Pinecone index.

    Args:
        documents (List[Document]): List of Document objects to be added.
        index_name (str): Name of the Pinecone index.
        dimension (int): Dimension of the embeddings.
    """

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        # Ensure the index name is unique and descriptive
        # Check if the index exists, create it if not
        # Create or connect to the index
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
        # Connect to the index
        index = pc.Index(index_name)
        # Ensure the index is ready
        if not index.describe_index_stats().total_vector_count:
            print(f"Index '{index_name}' is empty. Ready to insert data.")
        # Compare this snippet from Data%20Science/Chatbot/database_process.py:
        # If the index already exists, you can skip creation
        else:
            print(f"Index '{index_name}' already exists. Proceeding with data insertion.")
        # Create a PineconeVectorStore instance with the documents and embeddings
        docsearch = PineconeVectorStore.from_documents(
            documents=documents,
            index_name=index_name,
            embedding=download_hugging_face_embeddings(),
        )

        print(f"✅ Successfully added documents to Pinecone index: '{index_name}'")
    except PineconeApiException as e:
        # Handle specific Pinecone API exceptions
        data = json.loads(e.body)
        print(f"❌ Pinecone API Error: {data["message"]}")

    except Exception as e:
        # Handle other general exceptions (e.g., network issues)
        print(f"❌ An unexpected error occurred: {e}")