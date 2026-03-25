"""PyVis-based interactive knowledge graph visualizer."""

import webbrowser
from pathlib import Path

from pyvis.network import Network

from ..graph.models import Entity, KnowledgeGraph, Relationship


class PyVisVisualizer:
    """Interactive HTML visualizer for knowledge graphs using PyVis."""

    ENTITY_COLORS = {
        "PERSON": "#3498db",
        "ORGANIZATION": "#2ecc71",
        "PLACE": "#e67e22",
        "CONCEPT": "#9b59b6",
        "EVENT": "#e74c3c",
        "OTHER": "#95a5a6",
    }

    def __init__(
        self,
        graph: KnowledgeGraph,
        filter_types: list[str] | None = None,
        min_weight: float = 0.0,
        max_nodes: int = 200,
    ):
        self.graph = graph
        self.filter_types = set(filter_types) if filter_types else None
        self.min_weight = min_weight
        self.max_nodes = max_nodes
        self.network = None
        self._output_path = None

    def generate(self) -> None:
        """Generate the interactive network visualization."""
        self.network = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#000000",
            notebook=False,
        )

        self.network.barnes_hut(
            gravity=-50,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09,
        )

        filtered_entity_ids = self._filter_entities()
        degrees, valid_edges = self._collect_degrees_and_edges(filtered_entity_ids)

        if len(filtered_entity_ids) > self.max_nodes:
            print(
                f"Warning: Graph has {len(filtered_entity_ids)} entities. "
                f"Displaying top {self.max_nodes} by connectivity."
            )
            top_ids = sorted(filtered_entity_ids, key=lambda eid: degrees[eid], reverse=True)
            filtered_entity_ids = set(top_ids[: self.max_nodes])
            valid_edges = [
                r for r in valid_edges
                if r.source_entity_id in filtered_entity_ids
                and r.target_entity_id in filtered_entity_ids
            ]

        for entity_id in filtered_entity_ids:
            self._add_node(self.graph.entities[entity_id], degrees[entity_id])

        for relationship in valid_edges:
            self._add_edge(relationship)

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

    def _filter_entities(self) -> set[str]:
        """Return entity IDs that pass the type filter."""
        if not self.filter_types:
            return set(self.graph.entities.keys())
        return {eid for eid, e in self.graph.entities.items() if e.entity_type in self.filter_types}

    def _collect_degrees_and_edges(
        self, entity_ids: set[str]
    ) -> tuple[dict[str, int], list[Relationship]]:
        """Single-pass over relationships: compute degrees and collect valid edges."""
        degrees: dict[str, int] = {eid: 0 for eid in entity_ids}
        valid_edges: list[Relationship] = []

        for rel in self.graph.relationships:
            if rel.weight < self.min_weight:
                continue
            if rel.source_entity_id in entity_ids and rel.target_entity_id in entity_ids:
                degrees[rel.source_entity_id] += 1
                degrees[rel.target_entity_id] += 1
                valid_edges.append(rel)

        return degrees, valid_edges

    def _add_node(self, entity: Entity, degree: int) -> None:
        color = self.ENTITY_COLORS.get(entity.entity_type, self.ENTITY_COLORS["OTHER"])
        size = min(10 + degree * 2, 50)

        desc = ""
        if entity.description:
            desc = (
                entity.description[:200] + "..."
                if len(entity.description) > 200
                else entity.description
            )

        title_parts = [f"<b>{entity.name}</b>", f"Type: {entity.entity_type}"]
        if desc:
            title_parts.append(f"Description: {desc}")
        title = "<br>".join(title_parts)

        self.network.add_node(
            entity.entity_id,
            label=entity.name,
            title=title,
            color=color,
            size=size,
            borderWidth=2,
            borderWidthSelected=4,
        )

    def _add_edge(self, relationship: Relationship) -> None:
        width = min(1 + relationship.weight * 2, 5)

        title = relationship.relationship_type
        if relationship.description:
            desc = (
                relationship.description[:100] + "..."
                if len(relationship.description) > 100
                else relationship.description
            )
            title += f": {desc}"

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
        """Save the visualization as an HTML file."""
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
