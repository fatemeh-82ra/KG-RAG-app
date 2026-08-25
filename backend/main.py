"""FastAPI application — REST API over the hybrid KG-RAG pipeline.

Run:  uvicorn main:app --reload --port 8000   (from the backend folder)
Docs: http://localhost:8000/docs
"""

import shutil
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
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

app = FastAPI(title="KG-RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return {"chat": chat, "embedding": emb}


def _env(name: str) -> str:
    import os
    return os.getenv(name, "")


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@app.post("/api/conversations")
def create_conversation(body: ConversationIn):
    conv = store.create_conversation(body.title.strip() or "New conversation")

    # Lock the embedding spec now so later ingests stay consistent
    try:
        spec, model = pick_embedding_provider()
        store.set_embedding_spec(conv["id"], spec.key, model)
    except RuntimeError:
        pass
    return conv


@app.get("/api/conversations")
def list_conversations():
    return store.list_conversations()


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    if not store.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
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
def get_messages(cid: str):
    if not store.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
    return store.get_messages(cid)


@app.get("/api/conversations/{cid}/documents")
def get_documents(cid: str):
    if not store.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
    return store.list_documents(cid)


# ---------------------------------------------------------------------------
# Document upload + background ingestion
# ---------------------------------------------------------------------------

@app.post("/api/conversations/{cid}/documents")
async def upload_documents(cid: str, files: List[UploadFile] = File(...)):
    conv = store.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "Conversation not found")
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

    thread = threading.Thread(target=_ingest_worker, args=(cid, saved_paths), daemon=True)
    thread.start()
    return {"ok": True, "queued": len(saved_paths)}


def _ingest_worker(cid: str, paths: List[str]) -> None:
    JOBS[cid] = {"status": "processing", "detail": "Starting ingestion"}
    try:
        pipe = _get_pipeline(cid)
        stats = pipe.ingest(paths, progress_cb=lambda msg: JOBS[cid].update(detail=msg))
        store.set_documents_status(cid, "ready")
        JOBS[cid] = {"status": "ready",
                     "detail": f"{stats['chunks']} chunks, "
                               f"{stats['entities']} entities, "
                               f"{stats['relationships']} relationships"}
    except Exception as exc:                       # noqa: BLE001
        store.set_documents_status(cid, "error")
        JOBS[cid] = {"status": "error", "detail": str(exc)}


@app.get("/api/conversations/{cid}/status")
def ingest_status(cid: str):
    if not store.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
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
def chat(cid: str, body: ChatIn):
    if not store.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
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
