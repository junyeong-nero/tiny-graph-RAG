"""Graph traversal algorithms."""

from collections import deque

from ..graph.models import Entity, KnowledgeGraph, Relationship


class GraphTraversal:
    """Graph traversal utilities."""

    def __init__(self, graph: KnowledgeGraph):
        """Initialize traversal with a graph.

        Args:
            graph: KnowledgeGraph to traverse
        """
        self.graph = graph

    def bfs(self, start_entity_id: str, max_depth: int = 2) -> set[str]:
        """Breadth-first search from starting entity.

        Args:
            start_entity_id: Starting entity ID
            max_depth: Maximum traversal depth

        Returns:
            Set of reachable entity IDs (excluding start)
        """
        if start_entity_id not in self.graph.entities:
            return set()

        visited: set[str] = {start_entity_id}
        queue: deque[tuple[str, int]] = deque([(start_entity_id, 0)])
        result: set[str] = set()

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            neighbors = self.graph.get_neighbors(current_id, hops=1)
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    result.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return result

    def get_subgraph(
        self, entity_ids: set[str]
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract subgraph containing given entities.

        Args:
            entity_ids: Set of entity IDs to include

        Returns:
            Tuple of (entities, relationships)
        """
        entities = [
            self.graph.entities[eid]
            for eid in entity_ids
            if eid in self.graph.entities
        ]

        relationships = [
            rel
            for rel in self.graph.relationships
            if rel.source_entity_id in entity_ids and rel.target_entity_id in entity_ids
        ]

        return entities, relationships

    def find_paths(
        self, source_id: str, target_id: str, max_length: int = 3
    ) -> list[list[str]]:
        """Find paths between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of entity IDs)
        """
        if source_id not in self.graph.entities or target_id not in self.graph.entities:
            return []

        paths: list[list[str]] = []
        queue: deque[list[str]] = deque([[source_id]])

        while queue:
            current_path = queue.popleft()

            if len(current_path) > max_length:
                continue

            current_id = current_path[-1]

            if current_id == target_id:
                paths.append(current_path)
                continue

            neighbors = self.graph.get_neighbors(current_id, hops=1)
            for neighbor_id in neighbors:
                if neighbor_id not in current_path:  # Avoid cycles
                    new_path = current_path + [neighbor_id]
                    queue.append(new_path)

        return paths
