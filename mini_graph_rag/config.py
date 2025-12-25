"""Configuration management for Mini-Graph-RAG."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the GraphRAG system."""

    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4096
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it with: export OPENAI_API_KEY='your-api-key'"
            )

        return cls(
            openai_api_key=api_key,
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "200")),
        )
