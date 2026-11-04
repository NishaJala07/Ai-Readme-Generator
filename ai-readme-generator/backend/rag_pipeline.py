
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

# --------------------------------------------------
# 1. Persistent ChromaDB configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

# Embeddings are generated automatically by ChromaDB.
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="repository_chunks",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)

# --------------------------------------------------
# 2. Split files into overlapping text chunks
# --------------------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# 3. Index a file in ChromaDB
# --------------------------------------------------

def index_file(
    repo_url: str,
    file_path: str,
    content: str,
) -> int:
    """Chunk a file and store its text, embeddings and metadata."""

    chunks = split_into_chunks(content)

    if not chunks:
        return 0

    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        identity = f"{repo_url}:{file_path}:{index}"
        chunk_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

        metadatas.append({
            "repo_url": repo_url,
            "file_path": file_path,
            "chunk_index": index,
        })

    # ChromaDB creates embeddings using embedding_function.
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


# --------------------------------------------------
# 4. Search relevant chunks using semantic search
# --------------------------------------------------

def search_chunks(
    repo_url: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve semantically relevant chunks for one repository."""

    if not query or not query.strip():
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    # Count chunks belonging to this repository.
    repo_count = collection.get(
    where={"repo_url": repo_url},
    include=[],
    )

    repo_count = len(repo_count["ids"])

    if repo_count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, repo_count),
        where={"repo_url": repo_url},
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved_chunks.append({
            "content": document,
            "file_path": metadata["file_path"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance,
        })

    return retrieved_chunks


# --------------------------------------------------
# 5. Delete all indexed chunks for one repository
# --------------------------------------------------

def delete_repository_index(repo_url: str) -> None:
    """Remove indexed chunks for a repository."""

    collection.delete(
        where={"repo_url": repo_url}
    )


# --------------------------------------------------
# 6. Get repository-specific index statistics
# --------------------------------------------------

def get_repository_stats(repo_url: str) -> dict:
    """Return the number of indexed chunks for a repository."""

    count = collection.count(
        where={"repo_url": repo_url}
    )

    return {
        "repository_url": repo_url,
        "indexed_chunks": count,
    }


if __name__ == "__main__":
    print("ChromaDB initialized successfully.")
    print("Database location:", DB_PATH)
    print("Collection:", collection.name)
    print("Total chunks:", collection.count())
