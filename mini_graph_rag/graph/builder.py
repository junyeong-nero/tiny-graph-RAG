"""Graph construction from extraction results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import Entity, KnowledgeGraph, Relationship

if TYPE_CHECKING:
    from ..extraction.extractor import ExtractionResult


class GraphBuilder:
    """Build a knowledge graph from extraction results."""

    def __init__(self):
        """Initialize the graph builder."""
        self.graph = KnowledgeGraph()
        self._pending_relationships: list[tuple[Relationship, str, str]] = []

    def add_extraction_result(self, result: ExtractionResult) -> None:
        """Add extracted entities and relationships to the graph.

        Args:
            result: ExtractionResult from extractor
        """
        # Create temporary mapping from original entity IDs to graph entity IDs
        id_mapping: dict[str, str] = {}

        # Add entities
        for entity in result.entities:
            original_id = entity.entity_id
            new_id = self.graph.add_entity(entity)
            id_mapping[original_id] = new_id

        # Add relationships with mapped IDs
        for rel in result.relationships:
            source_id = id_mapping.get(rel.source_entity_id, rel.source_entity_id)
            target_id = id_mapping.get(rel.target_entity_id, rel.target_entity_id)

            # Only add if both entities exist in graph
            if source_id in self.graph.entities and target_id in self.graph.entities:
                new_rel = Relationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel.relationship_type,
                    description=rel.description,
                    source_chunks=rel.source_chunks,
                )
                self.graph.add_relationship(new_rel)

    def resolve_entities(self) -> None:
        """Resolve and merge duplicate entities.

        This is called after all extraction results are added.
        The KnowledgeGraph already handles basic name-based deduplication,
        but this method can be extended for more sophisticated resolution.
        """
        # Basic resolution is already handled by KnowledgeGraph.add_entity
        # Additional resolution logic can be added here if needed
        pass

    def build(self) -> KnowledgeGraph:
        """Finalize and return the knowledge graph.

        Returns:
            The constructed KnowledgeGraph
        """
        self.resolve_entities()
        return self.graph

    def reset(self) -> None:
        """Reset the builder for reuse."""
        self.graph = KnowledgeGraph()
        self._pending_relationships = []
