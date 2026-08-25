"""SQLite persistence for conversations, messages and documents."""

import sqlite3
import time
import uuid
from typing import List, Optional

from .config import DB_PATH, ensure_dirs


def _conn() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            embedding_provider TEXT DEFAULT '',
            embedding_model TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            created_at REAL NOT NULL
        );
        """)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(title: str) -> dict:
    cid = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute("INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
                  (cid, title, time.time()))
    return {"id": cid, "title": title}


def get_conversation(cid: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        return dict(row) if row else None


def list_conversations() -> List[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def set_embedding_spec(cid: str, provider: str, model: str) -> None:
    with _conn() as c:
        c.execute("UPDATE conversations SET embedding_provider = ?, embedding_model = ? "
                  "WHERE id = ?", (provider, model, cid))


def delete_conversation(cid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        c.execute("DELETE FROM documents WHERE conversation_id = ?", (cid,))
        c.execute("DELETE FROM conversations WHERE id = ?", (cid,))


# ---------------------------------------------------------------------------
# Messages / documents
# ---------------------------------------------------------------------------

def add_message(cid: str, role: str, content: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO messages (conversation_id, role, content, created_at) "
                  "VALUES (?, ?, ?, ?)", (cid, role, content, time.time()))


def get_messages(cid: str) -> List[dict]:
    with _conn() as c:
        rows = c.execute("SELECT role, content, created_at FROM messages "
                         "WHERE conversation_id = ? ORDER BY id", (cid,)).fetchall()
        return [dict(r) for r in rows]


def add_document(cid: str, filename: str, status: str = "queued") -> None:
    with _conn() as c:
        c.execute("INSERT INTO documents (conversation_id, filename, status, created_at) "
                  "VALUES (?, ?, ?, ?)", (cid, filename, status, time.time()))


def set_documents_status(cid: str, status: str, filenames: Optional[List[str]] = None) -> None:
    with _conn() as c:
        if filenames:
            for fn in filenames:
                c.execute("UPDATE documents SET status = ? WHERE conversation_id = ? "
                          "AND filename = ?", (status, cid, fn))
        else:
            c.execute("UPDATE documents SET status = ? WHERE conversation_id = ?",
                      (status, cid))


def list_documents(cid: str) -> List[dict]:
    with _conn() as c:
        rows = c.execute("SELECT filename, status FROM documents WHERE conversation_id = ? "
                         "ORDER BY id", (cid,)).fetchall()
        return [dict(r) for r in rows]
