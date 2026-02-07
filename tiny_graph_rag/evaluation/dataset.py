"""Evaluation dataset schema and loading utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalExample:
    """A single evaluation example.

    JSONL schema fields:
        id: Optional unique identifier for the example.
        query: The natural language query to evaluate.
        reference_entities: List of expected entity names the retriever should find.
        reference_relationships: Optional list of relationship objects
            containing "source", "target", and "type" keys.
        ground_truth: Optional list of ground truth answer strings.
        tags: Optional list of tags for filtering/grouping results.
    """

    query: str
    reference_entities: list[str]
    id: str = ""
    reference_relationships: list[dict[str, str]] = field(default_factory=list)
    ground_truth: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "query": self.query,
            "reference_entities": self.reference_entities,
        }
        if self.id:
            result["id"] = self.id
        if self.reference_relationships:
            result["reference_relationships"] = self.reference_relationships
        if self.ground_truth:
            result["ground_truth"] = self.ground_truth
        if self.tags:
            result["tags"] = self.tags
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalExample":
        """Create from dictionary.

        Args:
            data: Dictionary with at least ``query`` and ``reference_entities``
                keys.

        Returns:
            EvalExample instance.

        Raises:
            ValueError: If required fields are missing.
        """
        if "query" not in data:
            raise ValueError("EvalExample requires 'query' field")
        query = data["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("EvalExample requires 'query' as non-empty string")
        if "reference_entities" not in data:
            raise ValueError("EvalExample requires 'reference_entities' field")

        # Accept ground_truth as either a string or a list (take first element)
        raw_gt = data.get("ground_truth", "")
        if isinstance(raw_gt, list):
            ground_truth: str = raw_gt[0] if raw_gt else ""
        else:
            ground_truth = str(raw_gt) if raw_gt else ""

        reference_entities = data["reference_entities"]
        if not isinstance(reference_entities, list) or not all(
            isinstance(name, str) for name in reference_entities
        ):
            raise ValueError("EvalExample requires 'reference_entities' as list[str]")

        reference_relationships = data.get("reference_relationships", [])
        if reference_relationships and (
            not isinstance(reference_relationships, list)
            or not all(isinstance(item, dict) for item in reference_relationships)
        ):
            raise ValueError(
                "EvalExample requires 'reference_relationships' as list[dict]"
            )

        return cls(
            id=data.get("id", ""),
            query=query,
            reference_entities=reference_entities,
            reference_relationships=reference_relationships,
            ground_truth=ground_truth,
            tags=data.get("tags", []),
        )


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Load evaluation dataset from a JSONL file.

    Each line must be a valid JSON object with at least ``query`` and
    ``reference_entities`` fields.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of EvalExample instances.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a line contains invalid data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[EvalExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {e}"
                ) from e
            try:
                examples.append(EvalExample.from_dict(data))
            except ValueError as e:
                raise ValueError(
                    f"Invalid example on line {line_num}: {e}"
                ) from e

    return examples
