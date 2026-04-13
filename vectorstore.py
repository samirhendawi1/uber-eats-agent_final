"""ChromaDB vector store with remote embedding endpoint."""

from __future__ import annotations
import json, os, requests
import chromadb
from config import (
    EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL,
    KNOWLEDGE_FILE, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, WRITABLE_DIR,
)

CHROMA_DIR = os.path.join(WRITABLE_DIR, "chroma_db")

_collection = None


class SimpleDocument:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class RemoteEmbeddingFunction:
    """ChromaDB-compatible embedding function that calls the remote API."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.url = f"{base_url}/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a list of texts via the remote API."""
        all_embeddings = []
        for i in range(0, len(input), 64):
            batch = input[i:i + 64]
            resp = requests.post(
                self.url,
                headers=self.headers,
                json={"model": self.model, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in sorted_data])
        return all_embeddings


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
    ef = RemoteEmbeddingFunction(EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if not force:
        try:
            col = client.get_collection("uber_eats_kb", embedding_function=ef)
            if col.count() > 0:
                _collection = col
                return col
        except Exception:
            pass

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

    results = _collection.query(query_texts=[query], n_results=k)

    docs = []
    if results and results.get("documents") and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            docs.append(SimpleDocument(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
            ))
    return docs