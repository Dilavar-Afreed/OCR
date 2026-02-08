from app.vector_store import get_collection
from langchain_openai import OpenAIEmbeddings


def vector_search(query: str) -> str:
    """
    Search ChromaDB for relevant documents using OpenAI embeddings.
    """
    # Initialize embeddings model
    embeddings_model = OpenAIEmbeddings()

    # Embed the query
    query_embedding = embeddings_model.embed_query(query)

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No relevant documents found."

    return "\n\n".join(documents)
