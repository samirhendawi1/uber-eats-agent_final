"""Conversation memory backed by SQLite for cross-session persistence."""

from __future__ import annotations
import json, sqlite3, uuid, time
from config import MEMORY_DB


def _connect():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS session_state (
            session_id TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (session_id, key)
        )"""
    )
    conn.commit()
    return conn


def create_session() -> str:
    sid = str(uuid.uuid4())[:8]
    conn = _connect()
    conn.execute("INSERT INTO sessions VALUES (?, ?)", (sid, time.time()))
    conn.commit()
    conn.close()
    return sid


def add_message(session_id: str, role: str, content: str):
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, time.time()),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def set_state(session_id: str, key: str, value):
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO session_state VALUES (?, ?, ?)",
        (session_id, key, json.dumps(value)),
    )
    conn.commit()
    conn.close()


def get_state(session_id: str, key: str, default=None):
    conn = _connect()
    row = conn.execute(
        "SELECT value FROM session_state WHERE session_id = ? AND key = ?",
        (session_id, key),
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return default


def list_sessions() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT session_id, created_at FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"session_id": s, "created_at": t} for s, t in rows]
