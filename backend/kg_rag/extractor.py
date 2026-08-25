"""Knowledge graph extraction from text chunks using an LLM (JSON-only prompt)."""

import json
import re
from typing import Dict, List
import threading
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from .llm import get_llm


class Entity(BaseModel):
    id: str = Field(description="Canonical unique name of the entity")
    type: str = Field(default="Other", description="Entity type")
    description: str = Field(default="", description="One-sentence description")


class Relationship(BaseModel):
    source: str
    relation: str
    target: str
    description: str = ""


class ExtractedGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are an expert Information Extraction engine that converts unstructured text into a knowledge graph. You work with both English and Persian (Farsi) text.

## TASK
Read the given TEXT and extract:
1. ENTITIES — key named things or important concepts mentioned in the text.
2. RELATIONSHIPS — meaningful, directed connections between those entities.

## RULES (follow strictly)
1. Output ONLY a single valid JSON object. No explanations, no markdown code fences.
2. JSON schema:
{
  "entities": [
    {"id": "<canonical entity name>", "type": "<TYPE>", "description": "<one sentence>"}
  ],
  "relationships": [
    {"source": "<entity id>", "relation": "<RELATION_TYPE>", "target": "<entity id>", "description": "<optional short explanation>"}
  ]
}
3. Every "source" and "target" in relationships MUST exactly match the "id" of an entity listed in "entities".
4. Entity "type" is one of: Person, Organization, Location, Product, Technology, Model, Event, Date, Concept, Metric, Language, Role, ContactInfo, Other.
5. "relation" MUST be UPPER_SNAKE_CASE (e.g. WORKS_AT, LOCATED_IN, DEVELOPED_BY, PART_OF, USED_FOR).
6. Use canonical names for entity ids: merge spelling variants. Keep Persian names in Persian script; do not transliterate.
7. Only extract information explicitly present in the TEXT. Do NOT invent entities or infer facts not stated.
8. Prefer fewer, high-quality relations over many vague ones. Aim for up to ~15 entities per chunk.
9. CONTACT DETAILS AND CONCRETE VALUES must be captured EXACTLY and COMPLETELY as their own entities of type "ContactInfo":
   - Full email addresses — NEVER reduce an email to its domain.
   - Phone numbers exactly as written, URLs in full.
   - Link them to the owner with HAS_EMAIL, HAS_PHONE, HAS_URL, HAS_ADDRESS.
10. Extract concrete facts useful for answering questions: tasks, assignments, schedules, dates, deadlines, roles and job titles (as type "Role"), and quantities (as type "Metric").
11. If a person has a role/title mentioned, create an entity for the role and link it with HAS_ROLE, plus the organization with WORKS_AT.
12. If the text contains no meaningful entities, return {"entities": [], "relationships": []}.

## EXAMPLE 1 (English)
TEXT: "Llama 3.3 70B was trained by Meta and is served on NVIDIA NIM."
OUTPUT:
{"entities": [
  {"id": "Llama 3.3 70B", "type": "Model", "description": "A large language model developed by Meta."},
  {"id": "Meta", "type": "Organization", "description": "Technology company that trained Llama 3.3 70B."},
  {"id": "NVIDIA NIM", "type": "Technology", "description": "Optimized inference microservices by NVIDIA."}],
 "relationships": [
  {"source": "Llama 3.3 70B", "relation": "DEVELOPED_BY", "target": "Meta", "description": "Meta trained the model."},
  {"source": "Llama 3.3 70B", "relation": "HOSTED_BY", "target": "NVIDIA NIM", "description": "Served via NVIDIA NIM."}]}

## EXAMPLE 2 (Persian, includes contact details)
TEXT: "شرکت فناوری دیجی‌کالا در تهران تأسیس شد. سرپرست پشتیبانی، زهرا کیالی، از طریق ایمیل Zahra.Kiali@gmail.com و تلفن ۰۹۱۲۳۴۵۶۷۸۹ در دسترس است."
OUTPUT:
{"entities": [
  {"id": "دیجی‌کالا", "type": "Organization", "description": "یک شرکت فناوری ایرانی."},
  {"id": "تهران", "type": "Location", "description": "پایتخت ایران."},
  {"id": "زهرا کیالی", "type": "Person", "description": "سرپرست پشتیبانی دیجی‌کالا."},
  {"id": "سرپرست پشتیبانی", "type": "Role", "description": "سمت سازمانی زهرا کیالی."},
  {"id": "Zahra.Kiali@gmail.com", "type": "ContactInfo", "description": "آدرس ایمیل زهرا کیالی."},
  {"id": "۰۹۱۲۳۴۵۶۷۸۹", "type": "ContactInfo", "description": "شماره تماس زهرا کیالی."}],
 "relationships": [
  {"source": "دیجی‌کالا", "relation": "LOCATED_IN", "target": "تهران", "description": "محل تأسیس شرکت."},
  {"source": "زهرا کیالی", "relation": "HAS_ROLE", "target": "سرپرست پشتیبانی", "description": "سمت وی در شرکت."},
  {"source": "زهرا کیالی", "relation": "WORKS_AT", "target": "دیجی‌کالا", "description": "محل کار زهرا کیالی."},
  {"source": "زهرا کیالی", "relation": "HAS_EMAIL", "target": "Zahra.Kiali@gmail.com", "description": "ایمیل رسمی."},
  {"source": "زهرا کیالی", "relation": "HAS_PHONE", "target": "۰۹۱۲۳۴۵۶۷۸۹", "description": "شماره تماس."}]}

## NOW EXTRACT FROM THE FOLLOWING TEXT
"""


def _parse_llm_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(text[start:end + 1])


def _normalize_id(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def _normalize_relation(rel: str) -> str:
    rel = re.sub(r"[^0-9a-zA-Z\u0600-\u06FF]+", "_", rel.strip().upper())
    return rel.strip("_") or "RELATED_TO"


def _dedup_key(name: str) -> str:
    """Language-tolerant dedup key: case-folded, ي→ی / ك→ک, ZWNJ & whitespace removed."""
    s = name.lower()
    s = s.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    s = re.sub(r"[\s\u200c\u200f\u200e]+", "", s)
    return s


def extract_graph_from_chunk(text: str, llm=None) -> ExtractedGraph:
    llm = llm or get_llm(temperature=0.0)
    messages = [
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", f'TEXT:\n"""\n{text}\n"""\n\nOUTPUT (raw JSON only):'),
    ]
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            response = llm.invoke(messages)
            data = _parse_llm_json(response.content)
            graph = ExtractedGraph(
                entities=[
                    Entity(id=_normalize_id(e["id"]),
                           type=str(e.get("type", "Other")).strip() or "Other",
                           description=e.get("description", ""))
                    for e in data.get("entities", [])
                    if isinstance(e, dict) and e.get("id")
                ],
                relationships=[
                    Relationship(source=_normalize_id(r["source"]),
                                 target=_normalize_id(r["target"]),
                                 relation=_normalize_relation(r.get("relation", "")),
                                 description=r.get("description", ""))
                    for r in data.get("relationships", [])
                    if isinstance(r, dict) and r.get("source") and r.get("target")
                ],
            )
            valid_ids = {e.id for e in graph.entities}
            graph.relationships = [
                r for r in graph.relationships
                if r.source in valid_ids and r.target in valid_ids and r.source != r.target
            ]
            return graph
        except Exception as exc:
            last_error = exc
    print(f"[extractor] Failed to extract graph from chunk: {last_error}")
    return ExtractedGraph()


def build_document_graph(chunks: List[Document],
                         progress_cb=None) -> ExtractedGraph:
    """Extract per-chunk graphs IN PARALLEL and merge into one document-level graph."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    llm = get_llm(temperature=0.0)
    total = len(chunks)
    done_count = 0
    results = [None] * total
    counter_lock = threading.Lock()

    def work(i: int) -> None:
        nonlocal done_count
        results[i] = extract_graph_from_chunk(chunks[i].page_content, llm=llm)
        with counter_lock:
            done_count += 1
            if progress_cb:
                progress_cb(f"Extracting graph: {done_count}/{total} chunks")

    workers = min(8, total)
    if total > 0:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in range(total)]
            for f in as_completed(futures):
                f.result()

    merged = ExtractedGraph()
    seen_ids: Dict[str, str] = {}

    def canonical(eid: str) -> str:
        key = _dedup_key(eid)
        if key not in seen_ids:
            seen_ids[key] = eid
        return seen_ids[key]

    for sub in results:
        if sub is None:
            continue
        for e in sub.entities:
            cid = canonical(e.id)
            existing = next((x for x in merged.entities if x.id == cid), None)
            if existing is None:
                merged.entities.append(Entity(id=cid, type=e.type, description=e.description))
            elif not existing.description and e.description:
                existing.description = e.description

        for r in sub.relationships:
            src, tgt = canonical(r.source), canonical(r.target)
            if src and tgt and src != tgt:
                merged.relationships.append(Relationship(
                    source=src, target=tgt, relation=r.relation,
                    description=r.description))

    unique_rels, seen_triples = [], set()
    for r in merged.relationships:
        triple = (_dedup_key(r.source), r.relation, _dedup_key(r.target))
        if triple not in seen_triples:
            seen_triples.add(triple)
            unique_rels.append(r)
    merged.relationships = unique_rels

    print(f"[extractor] Merged graph: {len(merged.entities)} entities, "
          f"{len(merged.relationships)} relationships.")
    return merged
