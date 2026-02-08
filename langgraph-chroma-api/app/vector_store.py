import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
import io
import os
from typing import List, Optional

# Read Chroma connection info from environment so containers can connect
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "OCR_Documents")

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_chroma_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(
    collection_name: Optional[str] = None, embedding_function: Optional[any] = None
):
    """Get or create a collection. Uses DEFAULT_COLLECTION_NAME if not specified."""
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION_NAME
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )


def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file."""
    pdf_reader = PdfReader(io.BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str) -> List[str]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    return chunks


def embed_and_store(
    chunks: List[str],
    pdf_name: str,
    embeddings_model: OpenAIEmbeddings,
    collection_name: Optional[str] = None,
) -> dict:
    """Embed chunks and store in ChromaDB."""
    # We pass the metadata so Chroma knows how to handle future queries if initialized with it
    # But for manual embedding storage, we still provide the embeddings
    collection = get_collection(collection_name)

    # Generate embeddings
    embeddings = embeddings_model.embed_documents(chunks)

    # Store in ChromaDB with metadata
    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_name, "chunk": i} for i in range(len(chunks))]

    collection.add(
        ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
    )

    return {"pdf_name": pdf_name, "chunks_processed": len(chunks), "status": "success"}
