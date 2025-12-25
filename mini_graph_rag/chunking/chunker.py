"""Text chunking with overlap support."""

import uuid
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Represents a text chunk from a document."""

    text: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_index: int = 0
    end_index: int = 0
    doc_id: str = ""
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Split text into overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str = "") -> list[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk
            doc_id: Optional document ID

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # Don't go past the end of text
            if end > len(text):
                end = len(text)

            # Extract chunk text
            chunk_text = text[start:end]

            # Try to end at a sentence boundary if possible
            if end < len(text):
                chunk_text = self._adjust_to_boundary(chunk_text)
                end = start + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    doc_id=doc_id,
                )
            )

            # Move start position, accounting for overlap
            if end >= len(text):
                break

            start = end - self.overlap

            # Ensure we make progress
            if start <= chunks[-1].start_index:
                start = end

        return chunks

    def _adjust_to_boundary(self, text: str) -> str:
        """Adjust chunk to end at a sentence boundary.

        Args:
            text: The chunk text

        Returns:
            Adjusted text ending at a sentence boundary if possible
        """
        # Look for sentence endings in the last portion of the text
        search_start = max(0, len(text) - self.overlap)
        search_text = text[search_start:]

        # Try to find sentence boundaries (. ! ?)
        for boundary in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            last_idx = search_text.rfind(boundary)
            if last_idx != -1:
                # Include the boundary character
                return text[: search_start + last_idx + len(boundary)]

        # Fall back to paragraph boundary
        last_newline = search_text.rfind("\n\n")
        if last_newline != -1:
            return text[: search_start + last_newline + 2]

        # Fall back to any newline
        last_newline = search_text.rfind("\n")
        if last_newline != -1:
            return text[: search_start + last_newline + 1]

        # No good boundary found, return original
        return text
