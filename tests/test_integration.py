"""Integration tests for Tiny-Graph-RAG."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from unittest.mock import AsyncMock

from tiny_graph_rag import Config, GraphRAG
from tiny_graph_rag.graph import Entity, KnowledgeGraph, Relationship


class TestGraphStorageIntegration:
    """Tests for graph storage integration."""

    def test_save_and_load_json(self):
        """Test saving and loading graph as JSON."""
        from tiny_graph_rag.graph import GraphStorage

        storage = GraphStorage()

        # Create a test graph
        graph = KnowledgeGraph()
        e1 = Entity(name="Test Entity", entity_type="CONCEPT", description="A test")
        e2 = Entity(name="Another Entity", entity_type="PERSON")
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)
        graph.add_relationship(Relationship(
            source_entity_id=id1,
            target_entity_id=id2,
            relationship_type="RELATED_TO",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"

            # Save
            storage.save_json(graph, path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
                assert "entities" in data
                assert "relationships" in data

            # Load
            loaded = storage.load_json(path)

            assert len(loaded.entities) == 2
            assert len(loaded.relationships) == 1
            assert loaded.get_entity_by_name("Test Entity") is not None


class TestGraphRAGWithMocks:
    """Tests for GraphRAG with mocked LLM."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return Config(
            openai_api_key="test-key",
            model_name="gpt-4o-mini",
            chunk_size=500,
            chunk_overlap=50,
        )

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock = MagicMock()

        # Mock extraction response - returns different responses based on call
        def chat_json_side_effect(system_prompt, user_prompt, **kwargs):
            if "extract entity names" in system_prompt.lower():
                # Query entity extraction
                return {"entities": ["Alice"]}
            else:
                # Document entity extraction
                return {
                    "entities": [
                        {"name": "Alice", "type": "PERSON", "description": "A scientist"},
                        {"name": "Research Lab", "type": "ORGANIZATION", "description": "A laboratory"},
                    ],
                    "relationships": [
                        {
                            "source": "Alice",
                            "target": "Research Lab",
                            "type": "WORKS_AT",
                            "description": "Alice works at the lab",
                        }
                    ],
                }

        mock.chat_json.side_effect = chat_json_side_effect
        mock.async_chat_json = AsyncMock(side_effect=chat_json_side_effect)

        # Mock query response
        mock.chat.return_value = "Alice is a scientist who works at Research Lab."

        return mock

    def test_process_text_with_mock(self, mock_config, mock_llm_client):
        """Test processing text with mocked LLM."""
        with patch("tiny_graph_rag.OpenAIClient", return_value=mock_llm_client):
            rag = GraphRAG(config=mock_config)
            rag.llm_client = mock_llm_client

            text = "Alice is a brilliant scientist working at Research Lab."
            rag.process_text(text)

            assert rag.graph is not None
            assert len(rag.graph.entities) > 0
            assert rag.get_stats()["entities"] > 0

    def test_query_with_mock(self, mock_config, mock_llm_client):
        """Test querying with mocked LLM."""
        with patch("tiny_graph_rag.OpenAIClient", return_value=mock_llm_client):
            rag = GraphRAG(config=mock_config)
            rag.llm_client = mock_llm_client

            # Process some text first
            rag.process_text("Alice works at Research Lab.")

            # Query
            response = rag.query("Who is Alice?")

            assert response is not None
            assert len(response) > 0


class TestGraphBuilderIntegration:
    """Tests for graph builder integration."""

    def test_build_graph_from_multiple_chunks(self):
        """Test building graph from multiple extraction results."""
        from tiny_graph_rag.extraction import ExtractionResult
        from tiny_graph_rag.graph import Entity, GraphBuilder, Relationship

        builder = GraphBuilder()

        # First extraction result
        result1 = ExtractionResult(
            entities=[
                Entity(name="Alice", entity_type="PERSON"),
                Entity(name="Bob", entity_type="PERSON"),
            ],
            relationships=[
                Relationship(
                    source_entity_id="",  # Will be set by builder
                    target_entity_id="",
                    relationship_type="KNOWS",
                )
            ],
            source_chunk_id="chunk1",
        )

        # Fix relationship entity IDs
        result1.relationships[0].source_entity_id = result1.entities[0].entity_id
        result1.relationships[0].target_entity_id = result1.entities[1].entity_id

        builder.add_extraction_result(result1)

        # Second extraction result with overlapping entity
        result2 = ExtractionResult(
            entities=[
                Entity(name="Alice", entity_type="PERSON", description="A researcher"),
                Entity(name="Acme", entity_type="ORGANIZATION"),
            ],
            relationships=[],
            source_chunk_id="chunk2",
        )

        builder.add_extraction_result(result2)

        graph = builder.build()

        # Should have 3 unique entities (Alice merged, Bob, Acme)
        assert len(graph.entities) == 3

        # Alice should have merged description
        alice = graph.get_entity_by_name("Alice")
        assert alice is not None


class TestRetrievalIntegration:
    """Tests for retrieval integration."""

    def test_subgraph_retrieval(self):
        """Test retrieving subgraph."""
        from tiny_graph_rag.graph import Entity, KnowledgeGraph, Relationship
        from tiny_graph_rag.retrieval.traversal import GraphTraversal

        # Create a test graph
        graph = KnowledgeGraph()

        # Add entities forming a chain: A -> B -> C
        e_a = Entity(name="A", entity_type="CONCEPT")
        e_b = Entity(name="B", entity_type="CONCEPT")
        e_c = Entity(name="C", entity_type="CONCEPT")

        id_a = graph.add_entity(e_a)
        id_b = graph.add_entity(e_b)
        id_c = graph.add_entity(e_c)

        graph.add_relationship(Relationship(
            source_entity_id=id_a,
            target_entity_id=id_b,
            relationship_type="LINKS_TO",
        ))
        graph.add_relationship(Relationship(
            source_entity_id=id_b,
            target_entity_id=id_c,
            relationship_type="LINKS_TO",
        ))

        traversal = GraphTraversal(graph)

        # Get 2-hop neighbors from A
        neighbors = traversal.bfs(id_a, max_depth=2)
        assert id_b in neighbors
        assert id_c in neighbors

        # Get subgraph
        all_ids = {id_a} | neighbors
        entities, relationships = traversal.get_subgraph(all_ids)

        assert len(entities) == 3
        assert len(relationships) == 2
