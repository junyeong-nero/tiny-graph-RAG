"""PyVis-based interactive knowledge graph visualizer."""

import webbrowser
from collections import defaultdict
from pathlib import Path

from pyvis.network import Network

from ..graph.models import KnowledgeGraph


class PyVisVisualizer:
    """Interactive HTML visualizer for knowledge graphs using PyVis."""

    # Entity type color mapping
    ENTITY_COLORS = {
        "PERSON": "#3498db",  # Blue
        "ORGANIZATION": "#2ecc71",  # Green
        "PLACE": "#e67e22",  # Orange
        "CONCEPT": "#9b59b6",  # Purple
        "EVENT": "#e74c3c",  # Red
        "OTHER": "#95a5a6",  # Gray
    }

    def __init__(
        self,
        graph: KnowledgeGraph,
        filter_types: list[str] | None = None,
        min_weight: float = 0.0,
        max_nodes: int = 200,
    ):
        """Initialize the visualizer.

        Args:
            graph: Knowledge graph to visualize
            filter_types: List of entity types to include (e.g., ["PERSON", "PLACE"])
            min_weight: Minimum relationship weight to display
            max_nodes: Maximum number of nodes to display
        """
        self.graph = graph
        self.filter_types = set(filter_types) if filter_types else None
        self.min_weight = min_weight
        self.max_nodes = max_nodes
        self.network = None
        self._output_path = None

    def generate(self) -> None:
        """Generate the interactive network visualization."""
        # Create PyVis network
        self.network = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#000000",
            notebook=False,
        )

        # Configure physics for natural clustering
        self.network.barnes_hut(
            gravity=-50,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09,
        )

        # Filter and add entities
        filtered_entities = self._filter_entities()
        entity_degrees = self._calculate_degrees(filtered_entities)

        # Limit nodes if necessary
        if len(filtered_entities) > self.max_nodes:
            print(
                f"Warning: Graph has {len(filtered_entities)} entities. "
                f"Displaying top {self.max_nodes} by connectivity."
            )
            # Sort by degree (most connected first)
            sorted_entities = sorted(
                filtered_entities,
                key=lambda eid: entity_degrees[eid],
                reverse=True,
            )
            filtered_entities = sorted_entities[: self.max_nodes]

        # Add nodes
        for entity_id in filtered_entities:
            entity = self.graph.entities[entity_id]
            self._add_node(entity, entity_degrees[entity_id])

        # Add edges (only between visible nodes)
        filtered_entity_set = set(filtered_entities)
        for relationship in self.graph.relationships:
            if relationship.weight < self.min_weight:
                continue
            if (
                relationship.source_entity_id in filtered_entity_set
                and relationship.target_entity_id in filtered_entity_set
            ):
                self._add_edge(relationship)

        # Set options for better Korean text rendering
        self.network.set_options(
            """
            {
                "nodes": {
                    "font": {
                        "size": 14,
                        "face": "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
                    }
                },
                "edges": {
                    "font": {
                        "size": 12,
                        "face": "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif",
                        "align": "middle"
                    },
                    "smooth": {
                        "type": "continuous"
                    }
                },
                "physics": {
                    "enabled": true,
                    "stabilization": {
                        "iterations": 100
                    }
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 100
                }
            }
            """
        )

    def _filter_entities(self) -> list[str]:
        """Filter entities based on type filter.

        Returns:
            List of entity IDs that pass the filter
        """
        if not self.filter_types:
            return list(self.graph.entities.keys())

        filtered = []
        for entity_id, entity in self.graph.entities.items():
            if entity.entity_type in self.filter_types:
                filtered.append(entity_id)
        return filtered

    def _calculate_degrees(self, entity_ids: list[str]) -> dict[str, int]:
        """Calculate degree (number of connections) for each entity.

        Args:
            entity_ids: List of entity IDs to calculate degrees for

        Returns:
            Dictionary mapping entity_id to degree
        """
        degrees = defaultdict(int)
        entity_set = set(entity_ids)

        for relationship in self.graph.relationships:
            if relationship.weight < self.min_weight:
                continue
            if (
                relationship.source_entity_id in entity_set
                and relationship.target_entity_id in entity_set
            ):
                degrees[relationship.source_entity_id] += 1
                degrees[relationship.target_entity_id] += 1

        return degrees

    def _add_node(self, entity, degree: int) -> None:
        """Add a node to the network.

        Args:
            entity: Entity object to add
            degree: Number of connections (used for sizing)
        """
        # Determine color based on entity type
        color = self.ENTITY_COLORS.get(entity.entity_type, self.ENTITY_COLORS["OTHER"])

        # Scale node size based on degree (10-50 range)
        size = min(10 + degree * 2, 50)

        # Create hover tooltip
        title = f"<b>{entity.name}</b><br>"
        title += f"Type: {entity.entity_type}<br>"
        if entity.description:
            # Truncate long descriptions
            desc = entity.description[:200]
            if len(entity.description) > 200:
                desc += "..."
            title += f"Description: {desc}"

        # Add node
        self.network.add_node(
            entity.entity_id,
            label=entity.name,
            title=title,
            color=color,
            size=size,
            borderWidth=2,
            borderWidthSelected=4,
        )

    def _add_edge(self, relationship) -> None:
        """Add an edge to the network.

        Args:
            relationship: Relationship object to add
        """
        # Scale edge width based on weight (1-5 range)
        width = min(1 + relationship.weight * 2, 5)

        # Create hover tooltip
        title = f"{relationship.relationship_type}"
        if relationship.description:
            title += f": {relationship.description[:100]}"

        # Add edge with label
        self.network.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            label=relationship.relationship_type,
            title=title,
            width=width,
            color="#888888",
            arrows="to",
        )

    def save(self, path: str) -> None:
        """Save the visualization as an HTML file.

        Args:
            path: Output file path
        """
        if not self.network:
            raise ValueError("Generate visualization first by calling generate()")

        self._output_path = Path(path)
        self.network.save_graph(str(self._output_path))
        print(f"Visualization saved to {self._output_path}")

    def show(self) -> None:
        """Open the visualization in the default web browser."""
        if not self._output_path:
            raise ValueError("Save visualization first by calling save()")

        webbrowser.open(f"file://{self._output_path.absolute()}")
