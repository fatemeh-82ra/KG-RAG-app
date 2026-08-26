"""Embedding clients + persistent ChromaDB vector store (hybrid fallback layer)."""

import re
import time
from pathlib import Path
from typing import List

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import (
    EMBEDDING_PROVIDER_ORDER,
    CONFIG,
    PROVIDERS,
    ProviderSpec,
    get_api_key,
)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class OpenAICompatEmbeddings(Embeddings):
    """/embeddings client for OpenAI-compatible endpoints
    (batching + exponential-backoff retries on transient 5xx)."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 use_input_type: bool = False) -> None:
        self.url = base_url.rstrip("/") + "/embeddings"
        self.headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        self.model = model
        self.use_input_type = use_input_type

    def _post(self, batch: List[str], input_type: str) -> List[List[float]]:
        """POST one batch; on 400 (input too long / bad item) split the batch
        or shrink the single text, then retry."""
        payload = {"input": batch, "model": self.model}
        if self.use_input_type:
            payload["input_type"] = input_type   # NVIDIA retrieval models require this

        for attempt in range(5):
            resp = requests.post(self.url, headers=self.headers,
                                 json=payload, timeout=120)
            if resp.status_code == 400:
                if len(batch) > 1:
                    mid = len(batch) // 2
                    return (self._post(batch[:mid], input_type)
                            + self._post(batch[mid:], input_type))
                shrunk = batch[0][:800].strip() or "(empty)"
                if payload["input"][0] == shrunk:      # already minimal -> give up
                    resp.raise_for_status()
                payload["input"] = [shrunk]
                continue
            if resp.status_code >= 500 and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()["data"]
            if all(isinstance(d, dict) and "index" in d for d in data):
                data = sorted(data, key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        raise RuntimeError(f"Embedding failed after retries: last status {resp.status_code}")

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), CONFIG.embedding_batch_size):
            batch = texts[i:i + CONFIG.embedding_batch_size]
            vectors.extend(self._post(batch, input_type))
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]


class LocalEmbeddings(Embeddings):
    """Runs a HuggingFace embedding model locally via sentence-transformers.
    Never expires, no API needed."""

    _cache: dict = {}

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        if model_name not in LocalEmbeddings._cache:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            LocalEmbeddings._cache[model_name] = SentenceTransformer(model_name, device=device)
        self.model = LocalEmbeddings._cache[model_name]

    @staticmethod
    def _clean(texts: List[str]) -> List[str]:
        return [_sanitize_for_embedding(t) for t in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            self._clean(texts), batch_size=16,
            normalize_embeddings=True, show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            self._clean([text]), normalize_embeddings=True)[0].tolist()


def _sanitize_for_embedding(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:CONFIG.embedding_max_chars] if text else "(empty)"


def pick_embedding_provider(preferred: str | None = None,
                            preferred_model: str | None = None) -> tuple[ProviderSpec, str]:
    """Resolve (provider, embedding_model). Falls back to local BGE-M3.
    User-defined custom providers (with an embedding model set) are tried too."""
    if preferred and preferred in PROVIDERS:
        p = PROVIDERS[preferred]
        models = p.embedding_models
        if models and (not p.api_key_env or get_api_key(p)):
            model = preferred_model if preferred_model in models else models[0]
            return p, model
    for key in EMBEDDING_PROVIDER_ORDER:
        p = PROVIDERS[key]
        if p.embedding_models and (not p.api_key_env or get_api_key(p)):
            return p, p.embedding_models[0]
    # custom providers from Settings (before giving up to local)
    from .store import list_custom_providers
    for cp in list_custom_providers():
        if cp.get("embedding_model") and cp.get("base_url"):
            spec = ProviderSpec(
                key=f"custom_{cp['id']}", label=cp["name"],
                base_url=cp["base_url"], api_key_env="",
                embedding_models=(cp["embedding_model"],),
                api_key_value=cp.get("api_key", ""),
            )
            return spec, cp["embedding_model"]
    raise RuntimeError("No embedding provider available.")


def get_embeddings(provider_key: str | None = None,
                   model: str | None = None) -> Embeddings:
    p, resolved_model = pick_embedding_provider(provider_key, model)
    if p.key == "local":
        return LocalEmbeddings(resolved_model)
    return OpenAICompatEmbeddings(
        base_url=p.base_url, api_key=get_api_key(p),
        model=resolved_model, use_input_type=p.embedding_uses_input_type,
    )


# ---------------------------------------------------------------------------
# Vector store (one persistent collection per conversation)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: Path) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder_spec: dict | None = None   # {"provider":..., "model":...}
        self.store: Chroma | None = None

    def _connect(self) -> Chroma:
        if self.store is None:
            spec = self.embedder_spec or {}
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=get_embeddings(spec.get("provider"), spec.get("model")),
                persist_directory=str(self.persist_directory),
            )
        return self.store

    def build(self, chunks: List[Document],
              embedding_provider: str | None, embedding_model: str | None) -> int:
        safe_chunks = [
            Document(page_content=_sanitize_for_embedding(c.page_content),
                     metadata=dict(c.metadata or {}))
            for c in chunks
        ]
        self.embedder_spec = {"provider": embedding_provider, "model": embedding_model}
        self.store = None   # force reconnect with the chosen embedder
        store = self._connect()
        if safe_chunks:
            store.add_documents(safe_chunks)
        return len(safe_chunks)

    def search(self, query: str, k: int | None = None) -> List[Document]:
        try:
            return self._connect().similarity_search(query, k=k or CONFIG.top_k_chunks)
        except Exception:
            # Collection missing/not created yet -> nothing indexed
            return []

    def count(self) -> int:
        try:
            return self._connect()._collection.count()
        except Exception:
            return 0


def format_chunks_as_text(chunks: List[Document], max_chars_per_chunk: int = 900) -> str:
    if not chunks:
        return ""
    lines = ["Relevant document excerpts:"]
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "document")
        text = chunk.page_content[:max_chars_per_chunk].replace("\n", " ").strip()
        lines.append(f"[Excerpt {i} | {source}]\n{text}")
    return "\n\n".join(lines)
