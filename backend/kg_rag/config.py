"""Central configuration for the Hybrid KG-RAG application."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_ROOT = DATA_DIR / "chroma"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    base_url: str            # OpenAI-compatible endpoint ("" for local HF)
    api_key_env: str         # "" => no key needed
    chat_models: tuple = ()
    embedding_models: tuple = ()
    embedding_uses_input_type: bool = False   # NVIDIA-style /embeddings
    api_key_value: str = ""  # direct key (custom providers stored in DB)


PROVIDERS = {
    "nvidia": ProviderSpec(
        key="nvidia", label="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        # llama-3.3-70b-instruct reached end-of-life (410) -> current free models:
        chat_models=("deepseek-ai/deepseek-v4-flash-0731",
                     "openai/gpt-oss-120b",
                     "nvidia/nemotron-3-super-120b-a12b"),
        embedding_models=("nvidia/nemotron-3-embed-1b",),
        embedding_uses_input_type=True,
    ),
    "google": ProviderSpec(
        key="google", label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GOOGLE_API_KEY",
        chat_models=("gemini-flash-latest", "gemini-3.5-flash"),
        embedding_models=("gemini-embedding-001",),
    ),
    "openrouter": ProviderSpec(
        key="openrouter", label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        chat_models=("z-ai/glm-5.3-flash",),
    ),
    "ollama": ProviderSpec(
        key="ollama", label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        api_key_env="",
        chat_models=("gemma3:4b", "llama3.1", "qwen2.5:7b"),
        embedding_models=("bge-m3",),
    ),
    "local": ProviderSpec(
        key="local", label="Local HuggingFace",
        base_url="", api_key_env="",
        embedding_models=("BAAI/bge-m3", "intfloat/multilingual-e5-base"),
    ),
}

# Preference order when auto-picking providers:
# Google first; NVIDIA takes over automatically if Google fails/quota exhausted;
# OpenRouter (free GLM) is the next safety net; Ollama/local last.
CHAT_PROVIDER_ORDER = ("google", "nvidia", "openrouter", "ollama")
EMBEDDING_PROVIDER_ORDER = ("google", "nvidia", "ollama", "local")


@dataclass
class Config:
    # ---- Neo4j ----
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    neo4j_username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))

    # ---- Text splitting ----
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Extraction / retrieval ----
    extraction_temperature: float = 0.0

    answer_temperature: float = 0.1
    max_k_hop_depth: int = 2
    max_nodes_per_entity: int = 25

    # ---- Vector fallback ----
    top_k_chunks: int = 6
    embedding_batch_size: int = 12
    embedding_max_chars: int = 3000
    max_tokens: int = 2048

    # ---- LLM failover speed ----
    llm_timeout_seconds: int = 60     # per-request timeout -> faster fallback on hangs/504s
    llm_max_retries: int = 1          # SDK-internal retries before failing over


CONFIG = Config()


def ensure_dirs() -> None:
    for d in (DATA_DIR, CHROMA_ROOT, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_api_key(p: ProviderSpec) -> str:
    if p.api_key_env:
        return os.getenv(p.api_key_env, "")
    return p.api_key_value or ""


def provider_available(p: ProviderSpec) -> bool:
    if p.key in ("ollama", "local"):
        return True
    return bool(get_api_key(p))
