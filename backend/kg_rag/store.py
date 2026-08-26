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
            system_prompt TEXT DEFAULT '',
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
        CREATE TABLE IF NOT EXISTS custom_providers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            api_key TEXT DEFAULT '',
            chat_model TEXT DEFAULT '',
            embedding_model TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        """)
        # lightweight migration for DBs created before system_prompt existed
        cols = [r["name"] for r in c.execute("PRAGMA table_info(conversations)")]
        if "system_prompt" not in cols:
            c.execute("ALTER TABLE conversations ADD COLUMN system_prompt TEXT DEFAULT ''")
        cols = [r["name"] for r in c.execute("PRAGMA table_info(conversations)")]
        if "user_id" not in cols:
            c.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT DEFAULT ''")


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(title: str, system_prompt: str = "", user_id: str = "") -> dict:
    cid = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute("INSERT INTO conversations (id, title, system_prompt, created_at, user_id) "
                  "VALUES (?, ?, ?, ?, ?)",
                  (cid, title, system_prompt or "", time.time(), user_id))
    return {"id": cid, "title": title}


def get_conversation(cid: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        return dict(row) if row else None


def update_conversation(cid: str, title: Optional[str] = None,
                        system_prompt: Optional[str] = None) -> None:
    with _conn() as c:
        if title is not None:
            c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, cid))
        if system_prompt is not None:
            c.execute("UPDATE conversations SET system_prompt = ? WHERE id = ?",
                      (system_prompt, cid))


def list_conversations(user_id: Optional[str] = None) -> List[dict]:
    with _conn() as c:
        if user_id:
            rows = c.execute(
                "SELECT id, title, created_at FROM conversations WHERE user_id = ? "
                "ORDER BY created_at DESC", (user_id,)).fetchall()
        else:
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


# ---------------------------------------------------------------------------
# Custom providers (user-defined OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------

def add_custom_provider(name: str, base_url: str, api_key: str = "",
                        chat_model: str = "", embedding_model: str = "") -> dict:
    pid = uuid.uuid4().hex[:10]
    with _conn() as c:
        c.execute("INSERT INTO custom_providers "
                  "(id, name, base_url, api_key, chat_model, embedding_model, created_at) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (pid, name, base_url, api_key, chat_model, embedding_model, time.time()))
    return {"id": pid, "name": name}


def list_custom_providers() -> List[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, name, base_url, api_key, chat_model, embedding_model "
                         "FROM custom_providers ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]


def delete_custom_provider(pid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM custom_providers WHERE id = ?", (pid,))
