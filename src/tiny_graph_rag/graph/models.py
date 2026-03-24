"""Data models for the knowledge graph."""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""

    name: str
    entity_type: str  # PERSON, ORGANIZATION, PLACE, CONCEPT, EVENT, OTHER
    description: str = ""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: dict = field(default_factory=dict)
    source_chunks: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def merge_with(self, other: "Entity") -> "Entity":
        """Merge this entity with another entity.

        Args:
            other: Another entity to merge with

        Returns:
            A new merged entity
        """
        merged_description = self.description
        if other.description and other.description not in self.description:
            merged_description = f"{self.description} {other.description}".strip()

        merged_attributes = {**self.attributes, **other.attributes}
        merged_chunks = list(set(self.source_chunks + other.source_chunks))

        # Collect aliases: keep existing aliases and add other's name + aliases
        merged_aliases = list(self.aliases)
        if other.name != self.name and other.name not in merged_aliases:
            merged_aliases.append(other.name)
        for alias in other.aliases:
            if alias != self.name and alias not in merged_aliases:
                merged_aliases.append(alias)

        return Entity(
            name=self.name,
            entity_type=self.entity_type,
            description=merged_description,
            entity_id=self.entity_id,
            attributes=merged_attributes,
            source_chunks=merged_chunks,
            aliases=merged_aliases,
        )

    def to_dict(self) -> dict:
        """Convert entity to dictionary."""
        data = {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes,
            "source_chunks": self.source_chunks,
        }
        if self.aliases:
            data["aliases"] = self.aliases
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Create entity from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            entity_type=data["entity_type"],
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
            source_chunks=data.get("source_chunks", []),
            aliases=data.get("aliases", []),
        )


@dataclass
class Relationship:
    """Represents a relationship between two entities."""

    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # e.g., WORKS_FOR, LOCATED_IN, KNOWS
    description: str = ""
    relationship_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    weight: float = 1.0
    source_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert relationship to dictionary."""
        return {
            "relationship_id": self.relationship_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "weight": self.weight,
            "source_chunks": self.source_chunks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Create relationship from dictionary."""
        return cls(
            relationship_id=data["relationship_id"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            relationship_type=data["relationship_type"],
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            source_chunks=data.get("source_chunks", []),
        )


@dataclass
class KnowledgeGraph:
    """Represents a knowledge graph with entities and relationships."""

    entities: dict[str, Entity] = field(default_factory=dict)  # id -> Entity
    relationships: list[Relationship] = field(default_factory=list)
    entity_name_index: dict[str, str] = field(default_factory=dict)  # normalized_name -> id

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph.

        Args:
            entity: Entity to add

        Returns:
            The entity ID (may be existing if duplicate)
        """
        normalized_name = self._normalize_name(entity.name)

        # Check for existing entity with same name
        if normalized_name in self.entity_name_index:
            existing_id = self.entity_name_index[normalized_name]
            existing = self.entities[existing_id]
            self.entities[existing_id] = existing.merge_with(entity)
            return existing_id

        # Check if any of the entity's aliases match an existing entity
        for alias in entity.aliases:
            normalized_alias = self._normalize_name(alias)
            if normalized_alias in self.entity_name_index:
                existing_id = self.entity_name_index[normalized_alias]
                existing = self.entities[existing_id]
                self.entities[existing_id] = existing.merge_with(entity)
                self.entity_name_index[normalized_name] = existing_id
                return existing_id

        # Add new entity
        self.entities[entity.entity_id] = entity
        self.entity_name_index[normalized_name] = entity.entity_id
        for alias in entity.aliases:
            normalized_alias = self._normalize_name(alias)
            if normalized_alias not in self.entity_name_index:
                self.entity_name_index[normalized_alias] = entity.entity_id
        return entity.entity_id

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph.

        Args:
            relationship: Relationship to add
        """
        self.relationships.append(relationship)

    def merge_entities(self, canonical_id: str, duplicate_id: str) -> bool:
        """Merge duplicate entity into canonical entity.

        Args:
            canonical_id: Entity ID that should remain
            duplicate_id: Entity ID that should be merged and removed

        Returns:
            True if merge happened, False otherwise
        """
        if canonical_id == duplicate_id:
            return False
        if canonical_id not in self.entities or duplicate_id not in self.entities:
            return False

        canonical_entity = self.entities[canonical_id]
        duplicate_entity = self.entities[duplicate_id]
        self.entities[canonical_id] = canonical_entity.merge_with(duplicate_entity)
        del self.entities[duplicate_id]

        remapped_relationships: list[Relationship] = []
        for rel in self.relationships:
            source_id = canonical_id if rel.source_entity_id == duplicate_id else rel.source_entity_id
            target_id = canonical_id if rel.target_entity_id == duplicate_id else rel.target_entity_id

            # Drop self loops introduced by merge
            if source_id == target_id:
                continue

            remapped_relationships.append(
                Relationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel.relationship_type,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunks=rel.source_chunks,
                )
            )

        self.relationships = self._deduplicate_relationships(remapped_relationships)
        self._rebuild_entity_name_index()
        return True

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID.

        Args:
            entity_id: The entity ID

        Returns:
            Entity if found, None otherwise
        """
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Lookup entity by name.

        Args:
            name: Entity name to search for

        Returns:
            Entity if found, None otherwise
        """
        normalized_name = self._normalize_name(name)
        entity_id = self.entity_name_index.get(normalized_name)
        if entity_id:
            return self.entities.get(entity_id)
        return None

    def get_neighbors(self, entity_id: str, hops: int = 1) -> set[str]:
        """Get neighboring entity IDs within n hops.

        Args:
            entity_id: Starting entity ID
            hops: Number of hops to traverse

        Returns:
            Set of neighboring entity IDs
        """
        if hops <= 0:
            return set()

        # Build adjacency list
        adjacency: dict[str, set[str]] = defaultdict(set)
        for rel in self.relationships:
            adjacency[rel.source_entity_id].add(rel.target_entity_id)
            adjacency[rel.target_entity_id].add(rel.source_entity_id)

        # BFS traversal
        visited: set[str] = set()
        current_level: set[str] = {entity_id}

        for _ in range(hops):
            next_level: set[str] = set()
            for eid in current_level:
                for neighbor in adjacency[eid]:
                    if neighbor not in visited and neighbor != entity_id:
                        next_level.add(neighbor)
            visited.update(next_level)
            current_level = next_level

        return visited

    def get_relationships_for_entity(self, entity_id: str) -> list[Relationship]:
        """Get all relationships involving an entity.

        Args:
            entity_id: The entity ID

        Returns:
            List of relationships involving the entity
        """
        return [
            rel
            for rel in self.relationships
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching.

        Args:
            name: Entity name

        Returns:
            Normalized name
        """
        return name.lower().strip()

    def _rebuild_entity_name_index(self) -> None:
        """Rebuild normalized name -> entity ID index including aliases."""
        self.entity_name_index = {}
        for entity_id, entity in self.entities.items():
            self.entity_name_index[self._normalize_name(entity.name)] = entity_id
            for alias in entity.aliases:
                normalized_alias = self._normalize_name(alias)
                if normalized_alias not in self.entity_name_index:
                    self.entity_name_index[normalized_alias] = entity_id

    def _deduplicate_relationships(
        self,
        relationships: list[Relationship],
    ) -> list[Relationship]:
        """Deduplicate relationships after entity merges."""
        deduped: dict[tuple[str, str, str, str], Relationship] = {}
        for rel in relationships:
            key = (
                rel.source_entity_id,
                rel.target_entity_id,
                rel.relationship_type,
                rel.description.strip().lower(),
            )
            existing = deduped.get(key)
            if not existing:
                deduped[key] = Relationship(
                    source_entity_id=rel.source_entity_id,
                    target_entity_id=rel.target_entity_id,
                    relationship_type=rel.relationship_type,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunks=list(rel.source_chunks),
                )
                continue

            existing.weight += rel.weight
            existing.source_chunks = list(
                set(existing.source_chunks + rel.source_chunks)
            )

        return list(deduped.values())

    def to_dict(self) -> dict:
        """Convert graph to dictionary."""
        return {
            "entities": {
                eid: entity.to_dict() for eid, entity in self.entities.items()
            },
            "relationships": [rel.to_dict() for rel in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        graph = cls()

        for eid, entity_data in data.get("entities", {}).items():
            entity = Entity.from_dict(entity_data)
            graph.entities[eid] = entity

        graph._rebuild_entity_name_index()

        for rel_data in data.get("relationships", []):
            graph.relationships.append(Relationship.from_dict(rel_data))

        return graph
