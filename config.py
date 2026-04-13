"""Central configuration for the Uber Eats Support Agent."""

import os

# ── LLM Endpoint ──────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://rsm-8430-finalproject.bjlkeng.io")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-30b-a3b-fp8")
LLM_API_KEY = os.getenv("LLM_API_KEY", "1007638335")

# ── Embedding ─────────────────────────────────────────────────────────
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://rsm-8430-a2.bjlkeng.io")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "1007638335")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── RAG ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 5

# ── Paths ─────────────────────────────────────────────────────────────
# Knowledge base (read-only, from repo)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge_base.json")

# Writable storage: use /tmp on Streamlit Cloud, data/ locally
_on_cloud = os.path.exists("/mount/src")
WRITABLE_DIR = "/tmp/uber_eats_agent" if _on_cloud else DATA_DIR
os.makedirs(WRITABLE_DIR, exist_ok=True)

MEMORY_DB = os.path.join(WRITABLE_DIR, "memory.db")
ORDERS_DB = os.path.join(WRITABLE_DIR, "orders.db")
TICKETS_DB = os.path.join(WRITABLE_DIR, "tickets.db")
VECTORSTORE_CACHE = os.path.join(WRITABLE_DIR, "vectorstore.pkl")