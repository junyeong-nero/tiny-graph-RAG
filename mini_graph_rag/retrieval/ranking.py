"""Subgraph ranking and scoring."""

from ..graph.models import Entity, KnowledgeGraph, Relationship


class SubgraphRanker:
    """Rank and score subgraphs for relevance."""

    def __init__(self, graph: KnowledgeGraph):
        """Initialize ranker with a graph.

        Args:
            graph: KnowledgeGraph to rank within
        """
        self.graph = graph

    def score_entity(self, entity: Entity, query: str) -> float:
        """Score entity relevance to query.

        Uses simple text matching for scoring.

        Args:
            entity: Entity to score
            query: Query string

        Returns:
            Relevance score (0.0 to 1.0)
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        score = 0.0

        # Exact name match
        if entity.name.lower() in query_lower:
            score += 1.0

        # Partial name match
        entity_name_lower = entity.name.lower()
        for term in query_terms:
            if term in entity_name_lower or entity_name_lower in term:
                score += 0.5

        # Description match
        desc_lower = entity.description.lower()
        for term in query_terms:
            if term in desc_lower:
                score += 0.2

        # Normalize score
        return min(score, 1.0)

    def score_subgraph(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        query: str,
    ) -> float:
        """Score overall subgraph relevance.

        Args:
            entities: List of entities in subgraph
            relationships: List of relationships in subgraph
            query: Query string

        Returns:
            Aggregate relevance score
        """
        if not entities:
            return 0.0

        entity_scores = [self.score_entity(e, query) for e in entities]
        avg_score = sum(entity_scores) / len(entity_scores)

        # Bonus for connectedness
        connectivity_bonus = min(len(relationships) * 0.1, 0.5)

        return avg_score + connectivity_bonus

    def rank_and_filter(
        self, candidates: list[Entity], query: str, top_k: int
    ) -> list[Entity]:
        """Rank candidate entities and return top-k.

        Args:
            candidates: List of candidate entities
            query: Query string
            top_k: Number of top entities to return

        Returns:
            Top-k entities sorted by relevance
        """
        scored = [(e, self.score_entity(e, query)) for e in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]
