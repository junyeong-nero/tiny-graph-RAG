"""Parse LLM responses into structured data."""

from ..graph.models import Entity, Relationship


class ExtractionParser:
    """Parse extraction responses from LLM."""

    def parse(
        self, response: dict, chunk_id: str = ""
    ) -> tuple[list[Entity], list[Relationship]]:
        """Parse LLM JSON response into Entity and Relationship objects.

        Args:
            response: Parsed JSON response from LLM
            chunk_id: Source chunk ID for provenance

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []

        # Parse entities
        entity_map: dict[str, str] = {}  # name -> entity_id

        for entity_data in response.get("entities", []):
            entity = self._parse_entity(entity_data, chunk_id)
            if entity:
                entities.append(entity)
                entity_map[entity.name.lower()] = entity.entity_id

        # Parse relationships
        for rel_data in response.get("relationships", []):
            relationship = self._parse_relationship(rel_data, entity_map, chunk_id)
            if relationship:
                relationships.append(relationship)

        return entities, relationships

    def _parse_entity(self, data: dict, chunk_id: str) -> Entity | None:
        """Parse a single entity from data.

        Args:
            data: Entity data dictionary
            chunk_id: Source chunk ID

        Returns:
            Entity object or None if invalid
        """
        name = data.get("name", "").strip()
        if not name:
            return None

        entity_type = data.get("type", "OTHER").upper()
        valid_types = {"PERSON", "ORGANIZATION", "PLACE", "CONCEPT", "EVENT", "OTHER"}
        if entity_type not in valid_types:
            entity_type = "OTHER"

        return Entity(
            name=name,
            entity_type=entity_type,
            description=data.get("description", ""),
            source_chunks=[chunk_id] if chunk_id else [],
        )

    def _parse_relationship(
        self, data: dict, entity_map: dict[str, str], chunk_id: str
    ) -> Relationship | None:
        """Parse a single relationship from data.

        Args:
            data: Relationship data dictionary
            entity_map: Mapping of entity names to IDs
            chunk_id: Source chunk ID

        Returns:
            Relationship object or None if invalid
        """
        source_name = data.get("source", "").strip().lower()
        target_name = data.get("target", "").strip().lower()

        if not source_name or not target_name:
            return None

        # Look up entity IDs
        source_id = entity_map.get(source_name)
        target_id = entity_map.get(target_name)

        if not source_id or not target_id:
            return None

        rel_type = data.get("type", "RELATED_TO").upper().replace(" ", "_")

        return Relationship(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type=rel_type,
            description=data.get("description", ""),
            source_chunks=[chunk_id] if chunk_id else [],
        )
