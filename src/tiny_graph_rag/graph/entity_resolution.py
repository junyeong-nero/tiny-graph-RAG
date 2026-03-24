"""LLM-based entity resolution for cross-mention alias merges."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm import OpenAIClient
from .models import Entity, KnowledgeGraph


ENTITY_RESOLUTION_SYSTEM_PROMPT = """You resolve whether extracted person-like entities refer to the same real-world character.

You will receive a list of person-like entities from one story. Each entity may have:
- name: the canonical name
- entity_type: PERSON or OTHER (OTHER can still be a person mention like patient/husband)
- aliases: known alternative names already identified during extraction
- description: context about the entity
- neighbors: relationship signals (type, other entity name/type, and description)

Some entries are aliases, role names, or nicknames with NO lexical overlap with the canonical name.
Examples: full name vs title, role label (남편/husband, 인력거꾼/driver), metaphorical nickname (송아지/calf used for a person).

Use neighbor overlap as strong evidence: if two entities share the same relationship partners with
the same relationship types, they are likely the same person.

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

You may also receive a "Candidate merge pairs with supporting evidence" section.
Each candidate pair contains computed signals such as:
- shared_neighbors: entities both candidates are connected to (with relation details)
- direct_relations: relation types directly between the two candidates
- same_role_bucket: whether both names fall into the same semantic role category (e.g. both are spouse terms)
- co_occurring_chunks: source chunk IDs where both entities appear

These signals are HINTS, not mandates. Use them as additional evidence alongside descriptions,
neighbor signals, and aliases to make your final merge decision. You may merge pairs not listed
as candidates, and you may decline to merge listed candidates if the evidence is insufficient.

Rules:
- Merge only entities that refer to humans/characters.
- Prefer the entity with the most specific proper name as canonical.
- Use descriptions, neighbor signals, and existing aliases as evidence.
- Two names can be the same person even with zero string similarity.
- Do not invent IDs; use only provided IDs.
- Do not include canonical ID inside duplicate list.
- If unsure, do not merge.
"""



@dataclass(frozen=True)
class RoleBucket:
    """A group of role terms that may refer to the same character."""

    name: str
    terms: frozenset[str]
    reassign_aliases: bool = False


@dataclass(frozen=True)
class EntityResolutionConfig:
    """Domain and language-specific settings for entity resolution."""

    person_like_keywords: frozenset[str]
    generic_role_terms: frozenset[str]
    role_buckets: tuple[RoleBucket, ...]
    non_merge_relation_types: frozenset[str]


def default_config() -> EntityResolutionConfig:
    """Default config for multilingual literature (Korean + English)."""
    spouse_terms = frozenset({
        "아내", "마누라", "그의 아내",
        "남편", "부인",
        "wife", "husband", "spouse",
    })
    patient_terms = frozenset({
        "병인", "병자", "환자", "앓는 이", "이 환자",
        "patient", "invalid",
    })
    parent_terms = frozenset({
        "어머니", "엄마", "아버지", "아빠", "부모",
        "mother", "father", "mom", "dad",
    })
    child_terms = frozenset({
        "아들", "딸", "자식",
        "son", "daughter", "child",
    })
    sibling_terms = frozenset({
        "형", "오빠", "누나", "언니", "동생",
        "brother", "sister", "sibling",
    })
    servant_worker_terms = frozenset({
        "하인", "종", "인력거꾼", "차부",
        "servant", "maid", "driver", "coachman",
    })
    return EntityResolutionConfig(
        person_like_keywords=frozenset({
            # Korean: family
            "아내", "마누라", "남편", "부인",
            # Korean: medical
            "환자", "병자", "병인",
            # Korean: insults / colloquial
            "오라질",
            # Korean: occupational / social
            "주정꾼", "주정뱅이", "인력거꾼", "차부",
            "하인", "종",
            # Korean: family (extended)
            "어머니", "엄마", "아버지", "아빠",
            "아들", "딸", "자식",
            "형", "오빠", "누나", "언니", "동생",
            # English: family
            "wife", "husband", "spouse",
            "mother", "father", "mom", "dad",
            "son", "daughter",
            "brother", "sister", "sibling",
            # English: medical
            "patient", "invalid",
            # English: occupational / social
            "driver", "drunkard", "servant", "maid", "coachman",
            "narrator", "doctor", "nurse", "teacher", "priest",
        }),
        generic_role_terms=frozenset({
            # Korean: roles
            "아내", "마누라", "남편", "부인",
            "환자", "병자", "병인",
            "주정꾼", "주정뱅이", "인력거꾼", "차부",
            "어머니", "엄마", "아버지", "아빠",
            "아들", "딸", "자식",
            "형", "오빠", "누나", "언니", "동생",
            "하인", "종",
            # Korean: pronouns / generic references
            "그", "그녀", "이 사람", "저 사람", "그 사람",
            # English: pronouns
            "he", "she", "they", "him", "her", "them",
            # English: role labels
            "wife", "husband", "spouse",
            "mother", "father", "mom", "dad",
            "son", "daughter",
            "brother", "sister",
            "patient", "doctor", "nurse",
            "narrator", "servant", "maid", "driver",
            "teacher", "priest",
        }),
        role_buckets=(
            RoleBucket("spouse", spouse_terms, reassign_aliases=True),
            RoleBucket("patient", patient_terms),
            RoleBucket("parent", parent_terms),
            RoleBucket("child", child_terms),
            RoleBucket("sibling", sibling_terms),
            RoleBucket("servant_worker", servant_worker_terms),
        ),
        non_merge_relation_types=frozenset({
            "MARRIED_TO",
            "PARENT_OF",
            "CHILD_OF",
            "SIBLING_OF",
            "FRIEND_OF",
            "KNOWS",
        }),
    )


@dataclass
class LLMEntityResolver:
    """Resolve duplicate entities using a global LLM pass."""

    llm_client: OpenAIClient
    min_confidence: float = 0.75
    max_entities_per_pass: int = 80
    config: EntityResolutionConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = default_config()
        pattern = "|".join(
            re.escape(t)
            for t in sorted(self.config.generic_role_terms, key=len, reverse=True)
        )
        self._generic_role_pattern = re.compile(rf"^({pattern})$")

    def resolve(self, graph: KnowledgeGraph) -> None:
        """Resolve duplicate person-like entities in-place."""
        self._merge_explicit_alias_relationships(graph)

        person_like_entities = [
            entity
            for entity in graph.entities.values()
            if self._is_person_like_entity(entity)
        ]

        if len(person_like_entities) < 2:
            return

        # Resolve in chunks to avoid oversized prompts.
        for start in range(0, len(person_like_entities), self.max_entities_per_pass):
            batch = person_like_entities[start:start + self.max_entities_per_pass]
            merge_groups = self._resolve_batch(graph, batch)
            self._apply_merge_groups(graph, merge_groups)

    def _resolve_batch(
        self,
        graph: KnowledgeGraph,
        entities: list[Entity],
    ) -> list[dict]:
        payload = []
        for entity in entities:
            entry: dict = {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "source_chunks": entity.source_chunks,
                "neighbors": self._get_neighbor_signals(graph, entity.entity_id),
            }
            if entity.aliases:
                entry["aliases"] = entity.aliases
            payload.append(entry)

        candidate_signals = self._collect_merge_signals(graph, entities)

        user_prompt = (
            "Resolve duplicate person-like entities from the following JSON array. "
            "Two names can still be the same person even with no lexical overlap if context/relations match.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        if candidate_signals:
            user_prompt += (
                "\n\nCandidate merge pairs with supporting evidence:\n"
                f"{json.dumps(candidate_signals, ensure_ascii=False)}"
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
            signal: dict = {
                "relation_type": rel.relationship_type,
                "other_name": other.name,
                "other_type": other.entity_type,
            }
            if rel.description:
                signal["relation_desc"] = rel.description
            signals.append(signal)

        return signals[:12]

    def _collect_merge_signals(
        self,
        graph: KnowledgeGraph,
        entities: list[Entity],
    ) -> list[dict]:
        """Collect evidence signals for all candidate entity pairs in a batch."""
        signals: list[dict] = []
        for i, left in enumerate(entities):
            for right in entities[i + 1:]:
                pair_signal = self._build_pair_signal(graph, left, right)
                if pair_signal is not None:
                    signals.append(pair_signal)
        return signals

    def _build_pair_signal(
        self,
        graph: KnowledgeGraph,
        left: Entity,
        right: Entity,
    ) -> dict | None:
        """Build evidence signals for a candidate entity pair.

        Returns None if no signals exist between the pair.
        """
        signal: dict = {
            "left_name": left.name,
            "left_id": left.entity_id,
            "right_name": right.name,
            "right_id": right.entity_id,
        }
        has_signal = False

        # Shared 1-hop neighbors with relation details per side
        left_neighbor_ids = graph.get_neighbors(left.entity_id, hops=1) - {right.entity_id}
        right_neighbor_ids = graph.get_neighbors(right.entity_id, hops=1) - {left.entity_id}
        shared_ids = left_neighbor_ids & right_neighbor_ids
        if shared_ids:
            shared_neighbors: list[dict] = []
            for neighbor_id in shared_ids:
                neighbor = graph.get_entity(neighbor_id)
                if not neighbor:
                    continue
                left_rels = [
                    rel.relationship_type
                    for rel in graph.get_relationships_for_entity(left.entity_id)
                    if rel.source_entity_id == neighbor_id or rel.target_entity_id == neighbor_id
                ]
                right_rels = [
                    rel.relationship_type
                    for rel in graph.get_relationships_for_entity(right.entity_id)
                    if rel.source_entity_id == neighbor_id or rel.target_entity_id == neighbor_id
                ]
                shared_neighbors.append({
                    "neighbor_name": neighbor.name,
                    "left_relations": left_rels,
                    "right_relations": right_rels,
                })
            if shared_neighbors:
                signal["shared_neighbors"] = shared_neighbors
                has_signal = True

        # Direct relations between the pair
        direct_types = self._get_direct_relation_types(graph, left.entity_id, right.entity_id)
        if direct_types:
            signal["direct_relations"] = sorted(direct_types)
            has_signal = True

        # Same role bucket
        left_bucket = self._role_bucket(left)
        right_bucket = self._role_bucket(right)
        if left_bucket and left_bucket == right_bucket:
            signal["same_role_bucket"] = left_bucket
            has_signal = True

        # Co-occurring source chunks
        if left.source_chunks and right.source_chunks:
            co_occurring = sorted(set(left.source_chunks) & set(right.source_chunks))
            if co_occurring:
                signal["co_occurring_chunks"] = co_occurring
                has_signal = True

        return signal if has_signal else None

    def _apply_merge_groups(self, graph: KnowledgeGraph, merge_groups: list[dict]) -> None:
        parent: dict[str, str] = {}
        canonical_votes: dict[str, int] = {}

        def find(node: str) -> str:
            root = parent.setdefault(node, node)
            if root != node:
                root = find(root)
                parent[node] = root
            return root

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

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
            if not canonical or not self._is_person_like_entity(canonical):
                continue

            canonical_votes[canonical_id] = canonical_votes.get(canonical_id, 0) + 1

            for duplicate_id in duplicate_ids:
                if not isinstance(duplicate_id, str):
                    continue

                duplicate = graph.get_entity(duplicate_id)
                if not duplicate or not self._is_person_like_entity(duplicate):
                    continue
                if self._is_non_mergeable_pair(graph, canonical_id, duplicate_id):
                    continue

                parent.setdefault(canonical_id, canonical_id)
                parent.setdefault(duplicate_id, duplicate_id)
                union(canonical_id, duplicate_id)

        clusters: dict[str, list[str]] = {}
        for entity_id in parent:
            clusters.setdefault(find(entity_id), []).append(entity_id)

        for cluster_entity_ids in clusters.values():
            if len(cluster_entity_ids) < 2:
                continue

            canonical_id = self._select_canonical_id(
                graph,
                cluster_entity_ids,
                canonical_votes,
            )
            if not canonical_id:
                continue

            for entity_id in cluster_entity_ids:
                if entity_id == canonical_id:
                    continue
                if not graph.get_entity(entity_id):
                    continue
                graph.merge_entities(canonical_id, entity_id)

    def _merge_explicit_alias_relationships(self, graph: KnowledgeGraph) -> None:
        alias_edges: list[tuple[str, str]] = []
        for rel in graph.relationships:
            rel_type = rel.relationship_type.upper()
            if rel_type not in {"ALIAS_OF", "SAME_AS"}:
                continue

            source = graph.get_entity(rel.source_entity_id)
            target = graph.get_entity(rel.target_entity_id)
            if not source or not target:
                continue
            if not self._is_person_like_entity(source):
                continue
            if not self._is_person_like_entity(target):
                continue

            alias_edges.append((source.entity_id, target.entity_id))

        for left_id, right_id in alias_edges:
            left = graph.get_entity(left_id)
            right = graph.get_entity(right_id)
            if not left or not right:
                continue

            canonical_id = self._select_canonical_id(graph, [left_id, right_id])
            if not canonical_id:
                continue
            duplicate_id = right_id if canonical_id == left_id else left_id
            graph.merge_entities(canonical_id, duplicate_id)

    def _is_person_like_entity(self, entity: Entity) -> bool:
        if entity.entity_type == "PERSON":
            return True

        text = f"{entity.name} {entity.description}".lower()
        return any(keyword in text for keyword in self.config.person_like_keywords)

    def _role_bucket(self, entity: Entity) -> str | None:
        text = f"{entity.name} {entity.description}".lower()
        for bucket in self.config.role_buckets:
            if any(term in text for term in bucket.terms):
                return bucket.name
        return None

    def _get_direct_relation_types(
        self,
        graph: KnowledgeGraph,
        left_id: str,
        right_id: str,
    ) -> set[str]:
        relation_types: set[str] = set()
        for rel in graph.relationships:
            is_direct = (
                rel.source_entity_id == left_id and rel.target_entity_id == right_id
            ) or (
                rel.source_entity_id == right_id and rel.target_entity_id == left_id
            )
            if is_direct:
                relation_types.add(rel.relationship_type.upper())
        return relation_types

    def _is_non_mergeable_pair(
        self,
        graph: KnowledgeGraph,
        left_id: str,
        right_id: str,
    ) -> bool:
        direct_types = self._get_direct_relation_types(graph, left_id, right_id)
        return bool(direct_types.intersection(self.config.non_merge_relation_types))

    def _select_canonical_id(
        self,
        graph: KnowledgeGraph,
        entity_ids: list[str],
        canonical_votes: dict[str, int] | None = None,
    ) -> str | None:
        candidates: list[tuple[tuple[int, int, int, int], str]] = []
        votes = canonical_votes or {}
        for entity_id in entity_ids:
            entity = graph.get_entity(entity_id)
            if not entity:
                continue

            vote_score = votes.get(entity_id, 0)
            person_bonus = 1 if entity.entity_type == "PERSON" else 0
            role_penalty = 0 if not self._generic_role_pattern.match(entity.name) else -1
            relation_count = len(graph.get_relationships_for_entity(entity_id))
            score = (vote_score, person_bonus, role_penalty, relation_count)
            candidates.append((score, entity_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]
