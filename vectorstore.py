"""ChromaDB vector store — singleton pattern to avoid NotFoundError on Cloud."""

from __future__ import annotations
import json, os
import chromadb
from chromadb.utils import embedding_functions
from config import EMBEDDING_MODEL, CHROMA_DIR, KNOWLEDGE_FILE, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

# Singleton — one collection shared across build + retrieve
_collection = None


class SimpleDocument:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_vectorstore(force: bool = False):
    global _collection
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Use in-memory client on Cloud (no persist issues), persistent locally
    use_persistent = os.path.isdir(os.path.dirname(CHROMA_DIR))
    try:
        if use_persistent:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
        else:
            client = chromadb.Client()
    except Exception:
        client = chromadb.Client()

    # Check if already built
    try:
        col = client.get_collection("uber_eats_kb", embedding_function=ef)
        if col.count() > 0 and not force:
            _collection = col
            return col
    except Exception:
        pass

    # Build from scratch
    try:
        client.delete_collection("uber_eats_kb")
    except Exception:
        pass

    col = client.create_collection("uber_eats_kb", embedding_function=ef)

    with open(KNOWLEDGE_FILE, "r") as f:
        articles = json.load(f)

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
    print(f"[vectorstore] {len(articles)} articles -> {chunk_id} chunks")
    _collection = col
    return col


def retrieve(query: str, k: int = TOP_K) -> list[SimpleDocument]:
    global _collection
    if _collection is None:
        build_vectorstore()
    if _collection is None:
        return []

    try:
        results = _collection.query(query_texts=[query], n_results=k)
    except Exception:
        # Rebuild and retry once
        build_vectorstore(force=True)
        if _collection is None:
            return []
        results = _collection.query(query_texts=[query], n_results=k)

    docs = []
    if results and results.get("documents") and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            docs.append(SimpleDocument(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
            ))
    return docs
