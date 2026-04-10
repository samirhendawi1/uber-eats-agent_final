"""Build and query the Chroma vector store for RAG."""

from __future__ import annotations
import json, os
import chromadb
from chromadb.utils import embedding_functions
from config import EMBEDDING_MODEL, CHROMA_DIR, KNOWLEDGE_FILE, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


class SimpleDocument:
    """Minimal document class to avoid langchain dependency."""
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _load_articles() -> list[dict]:
    with open(KNOWLEDGE_FILE, "r") as f:
        return json.load(f)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple character-based chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_vectorstore(force: bool = False):
    """Chunk documents, embed, and persist to Chroma."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Check if already built
    try:
        col = client.get_collection("uber_eats_kb", embedding_function=ef)
        if col.count() > 0 and not force:
            return col
    except Exception:
        pass

    # Build from scratch
    try:
        client.delete_collection("uber_eats_kb")
    except Exception:
        pass

    col = client.create_collection("uber_eats_kb", embedding_function=ef)
    articles = _load_articles()

    all_docs, all_ids, all_metas = [], [], []
    chunk_id = 0
    for a in articles:
        chunks = _chunk_text(a["content"])
        for chunk in chunks:
            all_docs.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metas.append({
                "id": a["id"],
                "title": a["title"],
                "category": a["category"],
                "source_url": a["source_url"],
            })
            chunk_id += 1

    col.add(documents=all_docs, ids=all_ids, metadatas=all_metas)
    print(f"[vectorstore] {len(articles)} articles → {chunk_id} chunks")
    return col


def retrieve(query: str, k: int = TOP_K) -> list[SimpleDocument]:
    """Retrieve top-k relevant documents."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("uber_eats_kb", embedding_function=ef)

    results = col.query(query_texts=[query], n_results=k)
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append(SimpleDocument(
            page_content=results["documents"][0][i],
            metadata=results["metadatas"][0][i],
        ))
    return docs
