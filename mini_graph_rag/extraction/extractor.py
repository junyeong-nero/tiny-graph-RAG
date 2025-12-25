"""Entity and relationship extraction from text."""

from dataclasses import dataclass

from ..chunking.chunker import Chunk
from ..graph.models import Entity, Relationship
from ..llm.client import OpenAIClient
from .parser import ExtractionParser
from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt


@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction from a chunk."""

    entities: list[Entity]
    relationships: list[Relationship]
    source_chunk_id: str


class EntityRelationshipExtractor:
    """Extract entities and relationships from text using LLM."""

    def __init__(self, llm_client: OpenAIClient):
        """Initialize the extractor.

        Args:
            llm_client: OpenAI client for LLM calls
        """
        self.llm_client = llm_client
        self.parser = ExtractionParser()

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and relationships from a single chunk.

        Args:
            chunk: Text chunk to extract from

        Returns:
            ExtractionResult with entities and relationships
        """
        user_prompt = build_extraction_prompt(chunk.text)

        try:
            response = self.llm_client.chat_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            entities, relationships = self.parser.parse(response, chunk.chunk_id)

            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                source_chunk_id=chunk.chunk_id,
            )

        except Exception as e:
            # Return empty result on error
            print(f"Extraction error for chunk {chunk.chunk_id}: {e}")
            return ExtractionResult(
                entities=[],
                relationships=[],
                source_chunk_id=chunk.chunk_id,
            )

    def extract_batch(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        """Extract from multiple chunks.

        Args:
            chunks: List of chunks to process

        Returns:
            List of ExtractionResults
        """
        results = []
        for chunk in chunks:
            result = self.extract(chunk)
            results.append(result)
        return results
