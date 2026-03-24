"""Main retrieval orchestrator."""

from dataclasses import dataclass

from ..graph.models import Entity, KnowledgeGraph, Relationship
from ..llm.client import OpenAIClient
from .ranking import SubgraphRanker
from .traversal import GraphTraversal


QUERY_ENTITY_EXTRACTION_PROMPT = """Extract entity names mentioned in this query.
Return a JSON object with an "entities" array containing the entity names.
Only extract explicitly mentioned entities.

Example:
Query: "What is the relationship between John and Acme Corp?"
Response: {{"entities": ["John", "Acme Corp"]}}

Query: "{query}"
"""


@dataclass
class RetrievalResult:
    """Result of graph retrieval."""

    entities: list[Entity]
    relationships: list[Relationship]
    relevance_score: float
    context_text: str


class GraphRetriever:
    """Retrieve relevant subgraphs for queries."""

    def __init__(self, graph: KnowledgeGraph, llm_client: OpenAIClient):
        """Initialize retriever.

        Args:
            graph: KnowledgeGraph to retrieve from
            llm_client: OpenAI client for query processing
        """
        self.graph = graph
        self.llm_client = llm_client
        self.traversal = GraphTraversal(graph)
        self.ranker = SubgraphRanker(graph)

    def retrieve(self, query: str, top_k: int = 5, hops: int = 2) -> RetrievalResult:
        """Retrieve relevant subgraph for a query.

        Args:
            query: User query
            top_k: Maximum number of seed entities
            hops: Number of hops for graph traversal

        Returns:
            RetrievalResult with relevant context
        """
        # Extract entity mentions from query
        mentions = self._extract_query_entities(query)

        # Find matching entities in graph
        matched_entities = self._find_matching_entities(mentions)

        # If no matches, try fuzzy matching with all entities
        if not matched_entities:
            matched_entities = self._fuzzy_match_entities(query, top_k)

        # Expand to subgraph via traversal
        entity_ids = {e.entity_id for e in matched_entities}
        for entity in matched_entities:
            neighbors = self.traversal.bfs(entity.entity_id, max_depth=hops)
            entity_ids.update(neighbors)

        # Get subgraph
        entities, relationships = self.traversal.get_subgraph(entity_ids)

        # Rank and filter
        if len(entities) > top_k * 3:
            entities = self.ranker.rank_and_filter(entities, query, top_k * 3)
            entity_ids = {e.entity_id for e in entities}
            relationships = [
                r
                for r in relationships
                if r.source_entity_id in entity_ids and r.target_entity_id in entity_ids
            ]

        # Calculate relevance score
        relevance_score = self.ranker.score_subgraph(entities, relationships, query)

        # Format context
        context_text = self._format_context(entities, relationships)

        return RetrievalResult(
            entities=entities,
            relationships=relationships,
            relevance_score=relevance_score,
            context_text=context_text,
        )

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity mentions from query using LLM.

        Args:
            query: User query

        Returns:
            List of entity names mentioned
        """
        prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(query=query)

        try:
            response = self.llm_client.chat_json(
                system_prompt="You extract entity names from text.",
                user_prompt=prompt,
            )
            return response.get("entities", [])
        except Exception:
            return []

    def _find_matching_entities(self, mentions: list[str]) -> list[Entity]:
        """Match query mentions to graph entities.

        Args:
            mentions: Entity names from query

        Returns:
            List of matching entities
        """
        matched = []
        for mention in mentions:
            entity = self.graph.get_entity_by_name(mention)
            if entity:
                matched.append(entity)
        return matched

    def _fuzzy_match_entities(self, query: str, top_k: int) -> list[Entity]:
        """Fuzzy match query to all entities.

        Args:
            query: User query
            top_k: Number of entities to return

        Returns:
            List of best matching entities
        """
        all_entities = list(self.graph.entities.values())
        return self.ranker.rank_and_filter(all_entities, query, top_k)

    def _format_context(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> str:
        """Format retrieved subgraph as text context.

        Args:
            entities: List of entities
            relationships: List of relationships

        Returns:
            Formatted context string
        """
        if not entities:
            return "No relevant information found in the knowledge graph."

        # Create entity lookup for relationship formatting
        entity_map = {e.entity_id: e for e in entities}

        lines = ["Entities:"]
        for entity in entities:
            desc = f" - {entity.description}" if entity.description else ""
            lines.append(f"- {entity.name} ({entity.entity_type}){desc}")

        if relationships:
            lines.append("\nRelationships:")
            for rel in relationships:
                source = entity_map.get(rel.source_entity_id)
                target = entity_map.get(rel.target_entity_id)
                if source and target:
                    desc = f" ({rel.description})" if rel.description else ""
                    lines.append(
                        f"- {source.name} --[{rel.relationship_type}]--> {target.name}{desc}"
                    )

        return "\n".join(lines)
