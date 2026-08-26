"""End-to-end hybrid pipeline facade: Neo4j KG + ChromaDB vectors, per conversation."""

import shutil
from pathlib import Path
from typing import Callable, List, Optional

from .config import CHROMA_ROOT, CONFIG, ensure_dirs
from .loaders import clean_documents, load_document, split_documents
from .llm import get_llm
from .neo4j_mgr import get_neo4j_manager
from .retrieval import (
    format_subgraph_as_text,
    generate_answer,
    generate_answer_stream,
    retrieve_subgraph,
)
from .vectorstore import VectorStore, format_chunks_as_text


class KnowledgeGraphRAG:
    """One instance per conversation: isolated graph slice + vector collection."""

    def __init__(self,
                 conversation_id: str,
                 embedding_provider: str | None = None,
                 embedding_model: str | None = None) -> None:
        ensure_dirs()
        self.cid = conversation_id
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.manager = get_neo4j_manager()
        self.vector_store = VectorStore(
            collection_name=f"conv_{conversation_id}",
            persist_directory=CHROMA_ROOT / f"conv_{conversation_id}",
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_paths: List[str],
               progress_cb: Optional[Callable[[str], None]] = None) -> dict:
        def report(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)
            print(f"[ingest:{self.cid}] {msg}")

        report("Loading documents")
        docs = []
        for p in file_paths:
            docs.extend(load_document(p))
        docs = clean_documents(docs)
        report(f"Loaded {len(docs)} document sections")

        chunks = split_documents(docs)
        report(f"Created {len(chunks)} chunks")

        n = self.vector_store.build(chunks, self.embedding_provider, self.embedding_model)
        report(f"Indexed {n} chunks into ChromaDB")

        graph = build_graph_with_progress(chunks, report)
        self.manager.insert_graph(graph, self.cid)
        stats = self.manager.stats(self.cid)
        report("Ingestion complete")
        return {"chunks": len(chunks), **stats}

    # ------------------------------------------------------------------
    # Hybrid query
    # ------------------------------------------------------------------

    def _conversation_instructions(self) -> str:
        from . import store
        conv = store.get_conversation(self.cid) or {}
        return conv.get("system_prompt") or ""

    def query(self, question: str,
              chat_provider: str | None = None,
              chat_model: str | None = None) -> dict:
        llm = get_llm(temperature=CONFIG.answer_temperature,
                      provider=chat_provider, model=chat_model)

        subgraph = retrieve_subgraph(self.manager, question, self.cid,
                                     chat_provider=chat_provider,
                                     chat_model=chat_model)
        graph_context = format_subgraph_as_text(subgraph)

        chunk_context = ""
        n_chunks = 0
        if self.vector_store.count() > 0:
            chunks = self.vector_store.search(question)
            n_chunks = len(chunks)
            chunk_context = format_chunks_as_text(chunks)

        answer = generate_answer(question, graph_context, chunk_context, llm=llm,
                                 extra_instructions=self._conversation_instructions())
        return {
            "answer": answer,
            "graph_facts": len(subgraph.get("relationships", [])),
            "chunks_used": n_chunks,
        }

    def query_stream(self, question: str,
                     chat_provider: str | None = None,
                     chat_model: str | None = None) -> tuple[dict, object]:
        """Like query() but returns (meta, token_generator) for streaming answers."""
        llm = get_llm(temperature=CONFIG.answer_temperature,
                      provider=chat_provider, model=chat_model)

        subgraph = retrieve_subgraph(self.manager, question, self.cid,
                                     chat_provider=chat_provider,
                                     chat_model=chat_model)
        graph_context = format_subgraph_as_text(subgraph)

        chunk_context = ""
        n_chunks = 0
        if self.vector_store.count() > 0:
            chunks = self.vector_store.search(question)
            n_chunks = len(chunks)
            chunk_context = format_chunks_as_text(chunks)

        meta = {
            "graph_facts": len(subgraph.get("relationships", [])),
            "chunks_used": n_chunks,
        }
        tokens = generate_answer_stream(question, graph_context, chunk_context, llm=llm,
                                        extra_instructions=self._conversation_instructions())
        return meta, tokens

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_data(self) -> None:
        self.manager.clear_conversation(self.cid)
        chroma_dir = CHROMA_ROOT / f"conv_{self.cid}"
        shutil.rmtree(chroma_dir, ignore_errors=True)


def build_graph_with_progress(chunks, report):
    from .extractor import build_document_graph
    return build_document_graph(chunks, progress_cb=report)
