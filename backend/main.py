"""FastAPI application — REST API over the hybrid KG-RAG pipeline.

Run:  uvicorn main:app --reload --port 8000   (from the backend folder)
Docs: http://localhost:8000/docs
"""

import hashlib
import hmac
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from kg_rag import store
from kg_rag.config import (
    CHAT_PROVIDER_ORDER,
    EMBEDDING_PROVIDER_ORDER,
    PROVIDERS,
    ensure_dirs,
)
from kg_rag.pipeline import KnowledgeGraphRAG
from kg_rag.vectorstore import pick_embedding_provider

# ---------------------------------------------------------------------------
# Multi-user auth: signup/login, HMAC-signed tokens, per-user conversations
# ---------------------------------------------------------------------------

from kg_rag import auth as auth_mod

_OPEN_PATHS = {"/api/auth/login", "/api/auth/signup", "/docs", "/redoc", "/openapi.json"}


def verify_auth(request: Request) -> None:
    path = request.url.path
    if not path.startswith("/api") or path in _OPEN_PATHS:
        request.state.user_id = None
        return
    token = request.headers.get("X-Auth-Token", "")
    user = auth_mod.verify_token(token)
    if not user:
        raise HTTPException(401, "Unauthorized — please log in.")
    request.state.user_id = user["user_id"]


class SignupIn(BaseModel):
    username: str
    password: str
    display_name: str = ""


class LoginIn(BaseModel):
    username: str
    password: str


app = FastAPI(title="KG-RAG API", version="0.1.0",
              dependencies=[Depends(verify_auth)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.post("/api/auth/signup")
def api_signup(body: SignupIn):
    try:
        user = auth_mod.signup(body.username, body.password, body.display_name)
    except auth_mod.AuthError as e:
        raise HTTPException(400, str(e))
    return {"token": auth_mod.issue_token(user["user_id"], user["username"]),
            "username": user["username"]}


@app.post("/api/auth/login")
def api_login(body: LoginIn):
    try:
        user = auth_mod.login(body.username, body.password)
    except auth_mod.AuthError as e:
        raise HTTPException(401, str(e))
    return {"token": auth_mod.issue_token(user["user_id"], user["username"]),
            "username": user["username"]}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

PIPELINES: dict = {}     # conversation_id -> KnowledgeGraphRAG
JOBS: dict = {}          # conversation_id -> {"status": ..., "detail": ...}
_LOCK = threading.Lock()


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()
    store.init_db()
    auth_mod.init_users_db()


def _get_pipeline(cid: str) -> KnowledgeGraphRAG:
    with _LOCK:
        if cid not in PIPELINES:
            conv = store.get_conversation(cid)
            if not conv:
                raise HTTPException(404, "Conversation not found")
            PIPELINES[cid] = KnowledgeGraphRAG(
                cid,
                embedding_provider=conv.get("embedding_provider") or None,
                embedding_model=conv.get("embedding_model") or None,
            )
        return PIPELINES[cid]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ConversationIn(BaseModel):
    title: str
    system_prompt: Optional[str] = ""
    memory_turns: Optional[int] = 5


class ConversationPatch(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    memory_turns: Optional[int] = None


class ProviderIn(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    chat_model: str = ""
    embedding_model: str = ""


class ChatIn(BaseModel):
    question: str
    chat_provider: Optional[str] = None
    chat_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Models (providers available to the frontend dropdown)
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models():
    chat = []
    for key in CHAT_PROVIDER_ORDER:
        p = PROVIDERS[key]
        has_key = (not p.api_key_env) or bool(_env(p.api_key_env))
        chat.append({"provider": key, "label": p.label,
                     "models": list(p.chat_models), "available": has_key})
    emb = []
    for key in EMBEDDING_PROVIDER_ORDER:
        p = PROVIDERS[key]
        has_key = (not p.api_key_env) or bool(_env(p.api_key_env))
        emb.append({"provider": key, "label": p.label,
                    "models": list(p.embedding_models), "available": has_key})
    # user-defined custom providers appear in the dropdown too
    for cp in store.list_custom_providers():
        if cp.get("chat_model"):
            chat.append({"provider": f"custom_{cp['id']}",
                         "label": f"{cp['name']} (custom)",
                         "models": [cp["chat_model"]], "available": True})
    return {"chat": chat, "embedding": emb}


def _env(name: str) -> str:
    import os
    return os.getenv(name, "")


# ---------------------------------------------------------------------------
# Custom providers (user-defined OpenAI-compatible endpoints, e.g. gapgpt)
# ---------------------------------------------------------------------------

@app.get("/api/providers")
def list_providers():
    return store.list_custom_providers()


@app.post("/api/providers")
def add_provider(body: ProviderIn):
    if not body.name.strip() or not body.base_url.strip():
        raise HTTPException(400, "Name and base_url are required")
    if not (body.chat_model or body.embedding_model):
        raise HTTPException(400, "Set at least one of chat_model / embedding_model")
    return store.add_custom_provider(
        body.name.strip(), body.base_url.strip().rstrip("/"),
        body.api_key.strip(), body.chat_model.strip(), body.embedding_model.strip())


@app.delete("/api/providers/{pid}")
def remove_provider(pid: str):
    store.delete_custom_provider(pid)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def _own_conv(request: Request, cid: str) -> dict:
    """Fetch conversation and enforce ownership (legacy rows with empty user_id are shared)."""
    conv = store.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    owner = conv.get("user_id") or ""
    if owner and owner != request.state.user_id:
        raise HTTPException(403, "This conversation belongs to another user")
    return conv


@app.post("/api/conversations")
def create_conversation(body: ConversationIn, request: Request):
    conv = store.create_conversation(body.title.strip() or "New conversation",
                                     body.system_prompt or "",
                                     user_id=request.state.user_id or "",
                                     memory_turns=body.memory_turns or 5)

    # Lock the embedding spec now so later ingests stay consistent
    try:
        spec, model = pick_embedding_provider()
        store.set_embedding_spec(conv["id"], spec.key, model)
    except RuntimeError:
        pass
    return conv


@app.patch("/api/conversations/{cid}")
def patch_conversation(cid: str, body: ConversationPatch, request: Request):
    _own_conv(request, cid)
    store.update_conversation(cid, body.title, body.system_prompt, body.memory_turns)
    return {"ok": True}


@app.get("/api/conversations")
def list_conversations(request: Request):
    return store.list_conversations(user_id=request.state.user_id)


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str, request: Request):
    _own_conv(request, cid)
    try:
        _get_pipeline(cid).delete_data()
    except Exception as exc:                       # noqa: BLE001
        print(f"[delete] cleanup warning: {exc}")
    with _LOCK:
        PIPELINES.pop(cid, None)
        JOBS.pop(cid, None)
    store.delete_conversation(cid)
    return {"ok": True}


@app.get("/api/conversations/{cid}/messages")
def get_messages(cid: str, request: Request):
    _own_conv(request, cid)
    return store.get_messages(cid)


@app.get("/api/conversations/{cid}/documents")
def get_documents(cid: str, request: Request):
    _own_conv(request, cid)
    return store.list_documents(cid)


# ---------------------------------------------------------------------------
# Document upload + background ingestion
# ---------------------------------------------------------------------------

@app.post("/api/conversations/{cid}/documents")
async def upload_documents(cid: str, files: List[UploadFile] = File(...),
                           graph_provider: str = Form(""),
                           graph_model: str = Form(""),
                           request: Request = None):
    conv = _own_conv(request, cid)
    job = JOBS.get(cid, {})
    if job.get("status") == "processing":
        raise HTTPException(409, "This conversation is already processing a document.")

    dest_dir = Path("data/uploads") / cid
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_paths, filenames = [], []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            raise HTTPException(400, f"Unsupported file type '{suffix}' for {f.filename}")
        safe_name = f"{uuid.uuid4().hex[:8]}_{Path(f.filename).name}"
        path = dest_dir / safe_name
        with open(path, "wb") as out:
            out.write(await f.read())
        saved_paths.append(str(path))
        filenames.append(f.filename)
        store.add_document(cid, f.filename)

    thread = threading.Thread(
        target=_ingest_worker,
        args=(cid, saved_paths, graph_provider or None, graph_model or None),
        daemon=True)
    thread.start()
    return {"ok": True, "queued": len(saved_paths)}


def _ingest_worker(cid: str, paths: List[str],
                   graph_provider: str | None = None,
                   graph_model: str | None = None) -> None:
    JOBS[cid] = {"status": "processing", "detail": "Starting ingestion"}
    try:
        pipe = _get_pipeline(cid)
        stats = pipe.ingest(paths,
                            progress_cb=lambda msg: JOBS[cid].update(detail=msg),
                            chat_provider=graph_provider,
                            chat_model=graph_model)
        store.set_documents_status(cid, "ready")
        JOBS[cid] = {"status": "ready",
                     "detail": f"{stats['chunks']} chunks, "
                               f"{stats['entities']} entities, "
                               f"{stats['relationships']} relationships"}
    except Exception as exc:                       # noqa: BLE001
        store.set_documents_status(cid, "error")
        JOBS[cid] = {"status": "error", "detail": str(exc)}


@app.get("/api/conversations/{cid}/status")
def ingest_status(cid: str, request: Request):
    _own_conv(request, cid)
    job = JOBS.get(cid)
    if job:
        return job
    docs = store.list_documents(cid)
    if any(d["status"] == "ready" for d in docs):
        return {"status": "ready", "detail": "Documents indexed"}
    return {"status": "no_docs", "detail": "No documents uploaded yet"}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/api/conversations/{cid}/chat")
def chat(cid: str, body: ChatIn, request: Request):
    _own_conv(request, cid)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")

    job = JOBS.get(cid, {})
    if job.get("status") == "processing":
        raise HTTPException(409, "Still indexing documents — please wait.")

    pipe = _get_pipeline(cid)
    result = pipe.query(question, body.chat_provider, body.chat_model)

    store.add_message(cid, "user", question)
    store.add_message(cid, "assistant", result["answer"])
    return result


@app.post("/api/conversations/{cid}/chat/stream")
def chat_stream(cid: str, body: ChatIn, request: Request):
    """Streaming chat: Server-Sent Events with meta -> token* -> done."""
    import json

    from fastapi.responses import StreamingResponse

    _own_conv(request, cid)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")

    job = JOBS.get(cid, {})
    if job.get("status") == "processing":
        raise HTTPException(409, "Still indexing documents — please wait.")

    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def event_gen():
        try:
            pipe = _get_pipeline(cid)
            meta, tokens = pipe.query_stream(
                question, body.chat_provider, body.chat_model)
            yield sse({"type": "meta", **meta})

            parts: list[str] = []
            for token in tokens:
                parts.append(token)
                yield sse({"type": "token", "text": token})

            answer = "".join(parts).strip()
            store.add_message(cid, "user", question)
            store.add_message(cid, "assistant", answer)
            yield sse({"type": "done"})
        except Exception as exc:                       # noqa: BLE001
            yield sse({"type": "error", "detail": str(exc)[:300]})

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
