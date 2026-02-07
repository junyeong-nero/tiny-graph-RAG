"""LLM-based entity resolution for cross-mention alias merges."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm import OpenAIClient
from .models import Entity, KnowledgeGraph


ENTITY_RESOLUTION_SYSTEM_PROMPT = """You resolve whether extracted PERSON entities refer to the same real-world character.

You will receive a list of PERSON entities from one story. Some entries are aliases, role names, or nicknames.
Examples: full name vs title, role label, pronoun-like mention, or metaphorical nickname.

Return strict JSON with this shape:
{
  "merge_groups": [
    {
      "canonical_entity_id": "id_to_keep",
      "duplicate_entity_ids": ["id_to_merge", "..."],
      "confidence": 0.0,
      "reason": "short reason"
    }
  ]
}

Rules:
- Merge only PERSON entities.
- Be conservative: merge only when evidence is strong from descriptions/relations/context.
- Do not invent IDs; use only provided IDs.
- Do not include canonical ID inside duplicate list.
- If unsure, do not merge.
"""


@dataclass
class LLMEntityResolver:
    """Resolve duplicate entities using a global LLM pass."""

    llm_client: OpenAIClient
    min_confidence: float = 0.75
    max_entities_per_pass: int = 80

    def resolve(self, graph: KnowledgeGraph) -> None:
        """Resolve duplicate PERSON entities in-place."""
        person_entities = [
            entity
            for entity in graph.entities.values()
            if entity.entity_type == "PERSON"
        ]

        if len(person_entities) < 2:
            return

        # Resolve in chunks to avoid oversized prompts.
        for start in range(0, len(person_entities), self.max_entities_per_pass):
            batch = person_entities[start:start + self.max_entities_per_pass]
            merge_groups = self._resolve_batch(graph, batch)
            self._apply_merge_groups(graph, merge_groups)

    def _resolve_batch(
        self,
        graph: KnowledgeGraph,
        entities: list[Entity],
    ) -> list[dict]:
        payload = []
        for entity in entities:
            payload.append(
                {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "description": entity.description,
                    "source_chunks": entity.source_chunks,
                    "neighbors": self._get_neighbor_signals(graph, entity.entity_id),
                }
            )

        user_prompt = (
            "Resolve duplicate PERSON entities from the following JSON array. "
            "Two names can still be the same person even with no lexical overlap if context/relations match.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            response = self.llm_client.chat_json(
                system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception:
            return []

        merge_groups = response.get("merge_groups", [])
        if not isinstance(merge_groups, list):
            return []
        return [group for group in merge_groups if isinstance(group, dict)]

    def _get_neighbor_signals(
        self,
        graph: KnowledgeGraph,
        entity_id: str,
    ) -> list[dict]:
        signals: list[dict] = []
        for rel in graph.get_relationships_for_entity(entity_id):
            other_id = (
                rel.target_entity_id
                if rel.source_entity_id == entity_id
                else rel.source_entity_id
            )
            other = graph.get_entity(other_id)
            if not other:
                continue
            signals.append(
                {
                    "relation_type": rel.relationship_type,
                    "other_name": other.name,
                    "other_type": other.entity_type,
                }
            )

        return signals[:12]

    def _apply_merge_groups(self, graph: KnowledgeGraph, merge_groups: list[dict]) -> None:
        for group in merge_groups:
            canonical_id = group.get("canonical_entity_id")
            duplicate_ids = group.get("duplicate_entity_ids", [])
            confidence = float(group.get("confidence", 0.0))

            if confidence < self.min_confidence:
                continue
            if not isinstance(canonical_id, str):
                continue
            if not isinstance(duplicate_ids, list):
                continue

            canonical = graph.get_entity(canonical_id)
            if not canonical or canonical.entity_type != "PERSON":
                continue

            for duplicate_id in duplicate_ids:
                if not isinstance(duplicate_id, str):
                    continue
                duplicate = graph.get_entity(duplicate_id)
                if not duplicate or duplicate.entity_type != "PERSON":
                    continue

                graph.merge_entities(canonical_id, duplicate_id)
