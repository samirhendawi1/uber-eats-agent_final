"""Central configuration for the Uber Eats Support Agent."""

import os

# ── LLM Endpoint ──────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://rsm-8430-finalproject.bjlkeng.io")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-30b-a3b-fp8")
LLM_API_KEY = os.getenv("LLM_API_KEY", "1007638335")

# ── Embedding ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── RAG ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 400          # tokens (roughly chars/4)
CHUNK_OVERLAP = 80
TOP_K = 5

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
MEMORY_DB = os.path.join(DATA_DIR, "memory.db")
