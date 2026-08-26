"""Graph-based retrieval (question -> entities -> seeds -> k-hop subgraph)
and grounded answer generation."""

import json
import re
from typing import Dict, List

from .config import CONFIG
from .llm import get_llm
from .neo4j_mgr import Neo4jManager


# ---------------------------------------------------------------------------
# Question entity extraction
# ---------------------------------------------------------------------------

QUESTION_ENTITY_PROMPT = """You extract query entities for knowledge graph retrieval. The text may be in English or Persian (Farsi).

Extract the key entities mentioned or implied by the QUESTION. These will be used to find starting nodes in a knowledge graph.

Rules:
1. Output ONLY a JSON array of strings. No explanations, no markdown fences.
2. Use short, general noun phrases (e.g. "Llama 3", "NVIDIA", "دیجی‌کالا", "هوش مصنوعی").
3. Keep Persian entities in Persian script.
4. Include at most 5 entities. If none are relevant, return [].

QUESTION: "{question}"

OUTPUT (JSON array only):"""


def extract_question_entities(question: str,
                              chat_provider: str | None = None,
                              chat_model: str | None = None) -> List[str]:
    llm = get_llm(temperature=0.0, provider=chat_provider, model=chat_model)
    try:
        raw = llm.invoke([("human", QUESTION_ENTITY_PROMPT.format(question=question))])
        text = raw.content.strip().replace("```json", "").replace("```", "").strip()
        start, end = text.find("["), text.rfind("]")
        data = json.loads(text[start:end + 1])
        return [str(x).strip() for x in data if str(x).strip()][:5]
    except Exception as exc:
        print(f"[graph_retriever] Question entity extraction failed: {exc}")
        return []


def resolve_seed_nodes(manager: Neo4jManager, names: List[str],
                       conversation_id: str) -> List[Dict]:
    """Match question entities to graph nodes: exact first, then substring."""
    seeds, matched_keys = [], set()
    for name in names:
        matches = manager.find_entities_by_name([name], conversation_id)
        if not matches:
            matches = manager.fuzzy_find_entities(name, conversation_id, limit=3)
        for m in matches:
            if m["id"].lower() not in matched_keys:
                matched_keys.add(m["id"].lower())
                seeds.append(m)
    return seeds


def retrieve_subgraph(manager: Neo4jManager, question: str, conversation_id: str,
                      chat_provider: str | None = None,
                      chat_model: str | None = None) -> Dict:
    question_entities = extract_question_entities(
        question, chat_provider, chat_model)
    print(f"[graph_retriever] Question entities: {question_entities}")

    seeds = resolve_seed_nodes(manager, question_entities, conversation_id)
    if not seeds:
        print("[graph_retriever] No matching seed nodes found.")
        return {"nodes": [], "relationships": []}

    seed_ids = [s["id"] for s in seeds]
    subgraph = manager.get_k_hop_subgraph(seed_ids, conversation_id,
                                          depth=CONFIG.max_k_hop_depth)
    print(f"[graph_retriever] Subgraph: {len(subgraph['nodes'])} nodes, "
          f"{len(subgraph['relationships'])} relationships.")
    return subgraph


def format_subgraph_as_text(subgraph: Dict) -> str:
    if not subgraph.get("nodes"):
        return ""
    lines = ["Knowledge graph facts related to the question:"]
    for i, rel in enumerate(subgraph.get("relationships", []), start=1):
        line = f"{i}. ({rel['source_id']}) -[{rel['relation']}]-> ({rel['target_id']})"
        if rel.get("description"):
            line += f" :: {rel['description']}"
        lines.append(line)

    connected = ({r["source_id"] for r in subgraph["relationships"]}
                 | {r["target_id"] for r in subgraph["relationships"]})
    for node in subgraph.get("nodes", []):
        if node["id"] not in connected:
            lines.append(f"- {node['id']} ({node.get('type', 'Unknown')})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grounded answer generation
# ---------------------------------------------------------------------------

REFUSAL_MESSAGE = "I don't have enough information to answer this."

HYBRID_ANSWER_PROMPT = """You are a precise question-answering assistant. You answer questions using ONLY the material provided below, which may contain:
  1. KNOWLEDGE GRAPH FACTS — triples like (Entity A) -[RELATION]-> (Entity B)
  2. DOCUMENT EXCERPTS — verbatim passages from the source documents

## RULES
1. Answer ONLY from the provided material. Do NOT use outside knowledge.
2. If neither the graph facts nor the excerpts contain enough information, reply exactly:
   "{refusal}"
3. Do not invent entities, relations, numbers, or facts. No speculation.
4. You may combine multiple connected facts (e.g. A->B plus B->C) or multiple excerpts to build the answer.
5. Prefer graph facts when both sources agree; use excerpts for details not present in the graph (task lists, schedules, contact details).
6. Answer in the same language as the user's question (Persian question -> Persian answer).
7. Keep answers concise, factual, and mention names/values exactly as they appear.

KNOWLEDGE GRAPH FACTS:
{graph_context}

DOCUMENT EXCERPTS:
{chunk_context}
"""


def _build_system_prompt(graph_context: str, chunk_context: str,
                         extra_instructions: str = "") -> str:
    prompt = HYBRID_ANSWER_PROMPT.format(
        refusal=REFUSAL_MESSAGE,
        graph_context=graph_context or "(no graph facts retrieved)",
        chunk_context=chunk_context or "(no document excerpts retrieved)",
    )
    if extra_instructions and extra_instructions.strip():
        prompt += ("\n\nADDITIONAL BOT INSTRUCTIONS (highest priority — follow these "
                   "as well, e.g. persona, tone, how to behave when information is "
                   "missing):\n" + extra_instructions.strip())
    return prompt


def generate_answer(question: str, graph_context: str = "", chunk_context: str = "",
                    llm=None, extra_instructions: str = "") -> str:
    graph_context = (graph_context or "").strip()
    chunk_context = (chunk_context or "").strip()
    if not graph_context and not chunk_context:
        return REFUSAL_MESSAGE

    llm = llm or get_llm(temperature=CONFIG.answer_temperature)
    system_prompt = _build_system_prompt(graph_context, chunk_context, extra_instructions)
    response = llm.invoke([
        ("system", system_prompt),
        ("human", f"Question: {question}\n\nAnswer:"),
    ])
    text = response.content.strip()
    if REFUSAL_MESSAGE.lower()[:30] in text.lower():
        return REFUSAL_MESSAGE
    return text


def generate_answer_stream(question: str, graph_context: str = "",
                           chunk_context: str = "", llm=None,
                           extra_instructions: str = ""):
    """Yield the answer in small chunks (token streaming) with provider failover."""
    graph_context = (graph_context or "").strip()
    chunk_context = (chunk_context or "").strip()
    if not graph_context and not chunk_context:
        yield REFUSAL_MESSAGE
        return

    llm = llm or get_llm(temperature=CONFIG.answer_temperature)
    system_prompt = _build_system_prompt(graph_context, chunk_context, extra_instructions)
    collected: list[str] = []
    for piece in llm.stream([
        ("system", system_prompt),
        ("human", f"Question: {question}\n\nAnswer:"),
    ]):
        if piece:
            collected.append(piece)
            yield piece
