"""Tiny-Graph-RAG: A naive implementation of Graph-based RAG."""

import asyncio
from pathlib import Path

from .chunking import TextChunker
from .config import Config
from .extraction import EntityRelationshipExtractor
from .graph import GraphBuilder, GraphStorage, KnowledgeGraph, LLMEntityResolver
from .llm import OpenAIClient
from .llm.prompts import RESPONSE_GENERATION_SYSTEM, build_response_prompt
from .retrieval import GraphRetriever


class GraphRAG:
    """Main entry point for the Tiny-Graph-RAG system."""

    def __init__(self, config: Config | None = None):
        """Initialize the GraphRAG system.

        Args:
            config: Optional configuration. If not provided, loads from environment.
        """
        self.config = config or Config.from_env()
        self.llm_client = OpenAIClient(
            api_key=self.config.openai_api_key,
            model=self.config.model_name,
        )
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.extractor = EntityRelationshipExtractor(self.llm_client)
        self.graph_builder = GraphBuilder(
            resolver=LLMEntityResolver(self.llm_client)
        )
        self.storage = GraphStorage()
        self.graph: KnowledgeGraph | None = None
        self.retriever: GraphRetriever | None = None

    def process_document(self, file_path: str | Path) -> None:
        """Process a document and add to knowledge graph.

        Args:
            file_path: Path to the document file
        """
        text = self._read_document(file_path)
        self.process_text(text, doc_id=str(file_path))

    async def async_process_text(self, text: str, doc_id: str = "document") -> None:
        """Process raw text and add to knowledge graph asynchronously.

        This method uses async/await to process chunks in parallel for better performance.

        Args:
            text: The text to process
            doc_id: Optional document identifier
        """
        # Chunk the text
        chunks = self.chunker.chunk(text, doc_id=doc_id)
        print(f"Created {len(chunks)} chunks from document")

        # Extract entities and relationships from all chunks in parallel
        print(f"Processing {len(chunks)} chunks in parallel...")
        results = await self.extractor.async_extract_batch(chunks)

        # Add results to graph builder
        for i, result in enumerate(results):
            self.graph_builder.add_extraction_result(result)
            print(f"  Chunk {i + 1}: Extracted {len(result.entities)} entities, {len(result.relationships)} relationships")

        # Build the graph
        self.graph = self.graph_builder.build()
        self.retriever = GraphRetriever(self.graph, self.llm_client)

        print(f"Knowledge graph built: {len(self.graph.entities)} entities, {len(self.graph.relationships)} relationships")

    def process_text(self, text: str, doc_id: str = "document") -> None:
        """Process raw text and add to knowledge graph.

        This method uses async processing internally for better performance.

        Args:
            text: The text to process
            doc_id: Optional document identifier
        """
        asyncio.run(self.async_process_text(text, doc_id))

    def query(self, question: str) -> str:
        """Query the knowledge graph and generate response.

        Args:
            question: The question to answer

        Returns:
            Generated response based on the knowledge graph
        """
        if not self.graph or not self.retriever:
            raise ValueError(
                "No document processed. Call process_document or process_text first."
            )

        # Retrieve relevant context
        retrieval_result = self.retriever.retrieve(question)

        # Generate response
        response = self._generate_response(question, retrieval_result.context_text)
        return response

    def save_graph(self, path: str | Path) -> None:
        """Save the knowledge graph to file.

        Args:
            path: File path to save to (JSON format)
        """
        if not self.graph:
            raise ValueError("No graph to save. Process a document first.")

        self.storage.save_json(self.graph, path)
        print(f"Graph saved to {path}")

    def load_graph(self, path: str | Path) -> None:
        """Load a previously saved knowledge graph.

        Args:
            path: File path to load from
        """
        self.graph = self.storage.load_json(path)
        self.retriever = GraphRetriever(self.graph, self.llm_client)
        print(f"Graph loaded: {len(self.graph.entities)} entities, {len(self.graph.relationships)} relationships")

    def get_stats(self) -> dict:
        """Get statistics about the current knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        if not self.graph:
            return {"entities": 0, "relationships": 0}

        entity_types = {}
        for entity in self.graph.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

        relationship_types = {}
        for rel in self.graph.relationships:
            relationship_types[rel.relationship_type] = (
                relationship_types.get(rel.relationship_type, 0) + 1
            )

        return {
            "entities": len(self.graph.entities),
            "relationships": len(self.graph.relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
        }

    def visualize(
        self,
        output_path: str = "graph_viz.html",
        filter_types: list[str] | None = None,
        min_weight: float = 0.0,
        max_nodes: int = 200,
        show: bool = True,
    ) -> None:
        """Visualize the knowledge graph as an interactive HTML file.

        Args:
            output_path: Path to save HTML visualization (default: graph_viz.html)
            filter_types: List of entity types to include (e.g., ["PERSON", "PLACE"])
            min_weight: Minimum relationship weight to display (default: 0.0)
            max_nodes: Maximum number of nodes to display (default: 200)
            show: Whether to open in browser automatically (default: True)
        """
        if not self.graph:
            raise ValueError("No graph to visualize. Load or process a document first.")

        from .visualization import PyVisVisualizer

        viz = PyVisVisualizer(
            graph=self.graph,
            filter_types=filter_types,
            min_weight=min_weight,
            max_nodes=max_nodes,
        )
        viz.generate()
        viz.save(output_path)

        if show:
            viz.show()

    def _read_document(self, file_path: str | Path) -> str:
        """Read document from file.

        Args:
            file_path: Path to the document

        Returns:
            Document text content
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM.

        Args:
            query: User's question
            context: Retrieved context from knowledge graph

        Returns:
            Generated response
        """
        user_prompt = build_response_prompt(query, context)

        response = self.llm_client.chat(
            system_prompt=RESPONSE_GENERATION_SYSTEM,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
        )

        return response


__all__ = ["GraphRAG", "Config"]
