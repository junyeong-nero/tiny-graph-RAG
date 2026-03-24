"""Graph storage and loading utilities."""

import json
import pickle
from pathlib import Path

from .models import KnowledgeGraph


class GraphStorage:
    """Save and load knowledge graphs."""

    def save_json(self, graph: KnowledgeGraph, path: str | Path) -> None:
        """Save graph to JSON file.

        Args:
            graph: KnowledgeGraph to save
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)

    def load_json(self, path: str | Path) -> KnowledgeGraph:
        """Load graph from JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded KnowledgeGraph
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return KnowledgeGraph.from_dict(data)

    def save_pickle(self, graph: KnowledgeGraph, path: str | Path) -> None:
        """Save graph as pickle (faster for large graphs).

        Args:
            graph: KnowledgeGraph to save
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(graph, f)

    def load_pickle(self, path: str | Path) -> KnowledgeGraph:
        """Load graph from pickle.

        Args:
            path: File path to load from

        Returns:
            Loaded KnowledgeGraph
        """
        with open(path, "rb") as f:
            return pickle.load(f)
