# Hybrid KG-RAG Web App

Knowledge-Graph RAG chat application:
**Documents → Chunks → KG (Neo4j) ⟷ Vector index (ChromaDB) → Hybrid Retrieval → Grounded answer**

- Backend: **FastAPI + LangChain + Neo4j + ChromaDB** (Python)
- Frontend: **React (Vite)**
- Multi-conversation: each conversation has its own documents, graph slice and vector collection
- LLM selectable from the UI — any OpenAI-compatible provider (Gemini / Groq / NVIDIA NIM / Ollama)
- Persian & English documents and questions supported

## 1) Prerequisites

- Python 3.10+
- Node.js 18+
- Neo4j running locally (`neo4j://127.0.0.1:7687`) — [Neo4j Desktop](https://neo4j.com/download/) or Docker:
  ```bash
  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password neo4j:5
  ```

## 2) Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

copy .env.example .env           # then fill in your API key(s) + Neo4j password
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Providers (auto-picked by first available key in `.env`)
| Provider | Key | Chat | Embedding |
|---|---|---|---|
| Google Gemini | `GOOGLE_API_KEY` | gemini-2.0-flash | text-embedding-004 |
| Groq | `GROQ_API_KEY` | llama-3.3-70b-versatile | — |
| NVIDIA NIM | `NVIDIA_API_KEY` | llama-3.3-70b-instruct | nemotron-3-embed-1b |
| Ollama | no key needed | llama3.1 / qwen2.5 | bge-m3 |
| Local HuggingFace | no key needed | — | BAAI/bge-m3 |

For local embeddings install extras: `pip install sentence-transformers torch`

## 3) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Usage

1. **+ New conversation** — creates an isolated workspace
2. **Upload documents** — PDF / DOCX / TXT (Persian supported); ingestion builds the
   knowledge graph (Neo4j) and the vector index (ChromaDB). Watch the status badge.
3. Ask questions — answers are grounded in graph facts + relevant chunks,
   with source counts shown under each answer.
4. Pick a specific LLM from the dropdown, or leave it on auto.

## Notes

- Changing the embedding provider after ingesting documents is not supported per conversation
  (vectors must share one model). The spec is locked when the conversation is created.
- Conversation data lives in `backend/data/` (SQLite + Chroma) and Neo4j
  (nodes carry `conversation_id`, safe to share one database).
