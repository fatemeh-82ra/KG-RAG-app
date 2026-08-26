"""Multi-user authentication: signup / login / token verify.

- Passwords hashed with PBKDF2-HMAC-SHA256 + per-user random salt.
- Stateless tokens: base64url(payload).hmac  (JWT-like, no extra deps).
- Payload carries user_id + username + expiry (30 days).
"""

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import time
import uuid
from typing import Optional

from .config import DB_PATH, ensure_dirs

# ---------------------------------------------------------------------------
# Token secret — stable across restarts so tokens survive redeploys
# ---------------------------------------------------------------------------

def _load_secret() -> str:
    ensure_dirs()
    secret_file = DB_PATH.parent / "auth_secret.key"
    if secret_file.exists():
        return secret_file.read_text(encoding="utf-8").strip()
    secret = os.getenv("AUTH_SECRET") or base64.b64encode(os.urandom(32)).decode()
    secret_file.write_text(secret, encoding="utf-8")
    return secret

_SECRET = _load_secret()
TOKEN_TTL = 30 * 24 * 3600  # 30 days


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2, 200k iterations)
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return dk.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        candidate = _hash_password(password, bytes.fromhex(salt_hex))
        return hmac.compare_digest(candidate, hash_hex)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Users table
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        """)


# ---------------------------------------------------------------------------
# Signup / login
# ---------------------------------------------------------------------------

class AuthError(Exception):
    pass


def signup(username: str, password: str, display_name: str = "") -> dict:
    username = (username or "").strip()
    if len(username) < 3:
        raise AuthError("Username must be at least 3 characters")
    if len(password) < 4:
        raise AuthError("Password must be at least 4 characters")
    salt = os.urandom(16)
    uid = uuid.uuid4().hex
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (id, username, password_hash, display_name, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (uid, username, f"{salt.hex()}:{_hash_password(password, salt)}",
                 display_name or username, time.time()),
            )
    except sqlite3.IntegrityError:
        raise AuthError("This username is already taken")
    return {"user_id": uid, "username": username}


def login(username: str, password: str) -> dict:
    with _conn() as c:
        row = c.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ? COLLATE NOCASE",
            ((username or "").strip(),),
        ).fetchone()
    if not row or not _verify_password(password, row["password_hash"]):
        raise AuthError("Wrong username or password")
    return {"user_id": row["id"], "username": row["username"]}


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _from_b64url(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload_b64: str) -> str:
    return _b64url(hmac.new(_SECRET.encode(), payload_b64.encode(), hashlib.sha256).digest())


def issue_token(user_id: str, username: str) -> str:
    payload = _b64url(json.dumps({
        "uid": user_id, "u": username, "exp": time.time() + TOKEN_TTL,
    }).encode())
    return f"{payload}.{_sign(payload)}"


def verify_token(token: str) -> Optional[dict]:
    try:
        payload_b64, sig = token.split(".", 1)
        if not hmac.compare_digest(sig, _sign(payload_b64)):
            return None
        data = json.loads(_from_b64url(payload_b64))
        if data.get("exp", 0) < time.time():
            return None
        return {"user_id": data["uid"], "username": data["u"]}
    except Exception:
        return None
