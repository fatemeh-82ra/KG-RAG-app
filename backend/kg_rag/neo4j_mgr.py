"""Neo4j persistence layer — entities scoped per conversation via `key` uniqueness."""

import re
from typing import List

from neo4j import GraphDatabase

from .config import CONFIG
from .extractor import ExtractedGraph


def _safe_label(name: str) -> str:
    label = re.sub(r"[^0-9a-zA-Z]", "_", name).strip("_")
    if not label or label[0].isdigit():
        label = f"T_{label}"
    return label


class Neo4jManager:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            CONFIG.neo4j_uri,
            auth=(CONFIG.neo4j_username, CONFIG.neo4j_password),
        )
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
    # Drop legacy notebook-era constraint (unique on e.id) — the web app
    # scopes entities per conversation via `key` = "<cid>::<id>" instead.
        with self.driver.session() as session:
            session.run("DROP CONSTRAINT entity_id_unique IF EXISTS")
            session.run(
                "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.key IS UNIQUE"
            )

    def close(self) -> None:
        self.driver.close()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def insert_graph(self, graph: ExtractedGraph, conversation_id: str) -> None:
        with self.driver.session() as session:
            for entity in graph.entities:
                type_label = _safe_label(entity.type)
                session.execute_write(self._merge_entity_tx, conversation_id,
                                      entity.id, entity.type, entity.description,
                                      type_label)
            for rel in graph.relationships:
                rel_type = _safe_label(rel.relation)
                session.execute_write(self._merge_relationship_tx, conversation_id,
                                      rel.source, rel.target, rel_type, rel.description)

    @staticmethod
    def _merge_entity_tx(tx, cid: str, eid: str, etype: str,
                         description: str, type_label: str) -> None:
        tx.run(
            f"""
            MERGE (e:Entity {{key: $key}})
            SET e.id = $id,
                e.conversation_id = $cid,
                e.type = $type,
                e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END
            SET e:`{type_label}`
            """,
            key=f"{cid}::{eid}", cid=cid, id=eid,
            type=etype, description=description,
        )

    @staticmethod
    def _merge_relationship_tx(tx, cid: str, source: str, target: str,
                               rel_type: str, description: str) -> None:
        tx.run(
            f"""
            MATCH (a:Entity {{key: $skey}})
            MATCH (b:Entity {{key: $tkey}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.conversation_id = $cid,
                r.description = CASE WHEN $description <> '' THEN $description ELSE r.description END
            """,
            skey=f"{cid}::{source}", tkey=f"{cid}::{target}",
            cid=cid, description=description,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers (conversation-scoped)
    # ------------------------------------------------------------------

    def find_entities_by_name(self, names: List[str], conversation_id: str) -> list:
        lowered = [n.lower() for n in names]
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $names AS name
                MATCH (e:Entity {conversation_id: $cid})
                WHERE toLower(e.id) = name OR any(t IN labels(e) WHERE toLower(t) = name)
                RETURN e.id AS id, e.type AS type, e.description AS description
                LIMIT $limit
                """,
                names=lowered, cid=conversation_id,
                limit=CONFIG.max_nodes_per_entity * max(1, len(lowered)),
            )
            return [dict(r) for r in result]

    def fuzzy_find_entities(self, name: str, conversation_id: str, limit: int = 5) -> list:
        pattern = f"(?i).*{re.escape(name)}.*"
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {conversation_id: $cid})
                WHERE e.id =~ $pattern
                RETURN e.id AS id, e.type AS type, e.description AS description
                LIMIT $limit
                """,
                pattern=pattern, cid=conversation_id, limit=limit,
            )
            return [dict(r) for r in result]

    def get_k_hop_subgraph(self, seed_ids: List[str], conversation_id: str,
                           depth: int | None = None) -> dict:
        depth = depth or CONFIG.max_k_hop_depth
        cypher = (
            f"MATCH p = (seed:Entity)-[*1..{depth}]-(neighbor:Entity) "
            f"WHERE seed.id IN $seeds AND seed.conversation_id = $cid "
            f"AND neighbor.conversation_id = $cid "
            f"UNWIND relationships(p) AS rel "
            f"WITH DISTINCT startNode(rel) AS a, rel, endNode(rel) AS b "
            f"RETURN a.id AS source_id, a.type AS source_type, "
            f"       type(rel) AS relation, coalesce(rel.description, '') AS description, "
            f"       b.id AS target_id, b.type AS target_type"
        )
        with self.driver.session() as session:
            rows = [dict(r) for r in session.run(cypher, seeds=seed_ids, cid=conversation_id)]

        nodes: dict = {}
        edges: list = []
        seen_edges = set()
        for row in rows:
            nodes[row["source_id"]] = row["source_type"]
            nodes[row["target_id"]] = row["target_type"]
            edge_key = (row["source_id"], row["relation"], row["target_id"])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(row)

        for sid in seed_ids:
            if sid not in nodes:
                found = self.find_entities_by_name([sid], conversation_id)
                if found:
                    nodes[sid] = found[0]["type"]

        return {
            "nodes": [{"id": n, "type": t} for n, t in nodes.items()],
            "relationships": edges,
        }

    def stats(self, conversation_id: str) -> dict:
        with self.driver.session() as session:
            nodes = session.run(
                "MATCH (e:Entity {conversation_id: $cid}) RETURN count(e) AS c",
                cid=conversation_id).single()["c"]
            rels = session.run(
                "MATCH ()-[r]->() WHERE r.conversation_id = $cid RETURN count(r) AS c",
                cid=conversation_id).single()["c"]
        return {"entities": nodes, "relationships": rels}

    def clear_conversation(self, conversation_id: str) -> None:
        with self.driver.session() as session:
            session.run(
                "MATCH (n {conversation_id: $cid}) DETACH DELETE n",
                cid=conversation_id)


# Module-level singleton ------------------------------------------------------

_manager: Neo4jManager | None = None


def get_neo4j_manager() -> Neo4jManager:
    global _manager
    if _manager is None:
        _manager = Neo4jManager()
    return _manager
