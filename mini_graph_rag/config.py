"""Configuration management for Mini-Graph-RAG."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    """Configuration for the GraphRAG system."""

    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4096
    temperature: float = 0.0

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to config.yaml file. If None, searches in current
                        directory and package directory.

        Returns:
            Config instance with merged settings.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it with: export OPENAI_API_KEY='your-api-key'"
            )

        # Default values
        config_data = {
            "openai": {
                "base_url": None,
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

        # Search for config.yaml
        if config_path:
            yaml_path = Path(config_path)
        else:
            # Search in current directory first, then package directory
            search_paths = [
                Path.cwd() / "config.yaml",
                Path(__file__).parent.parent / "config.yaml",
            ]
            yaml_path = None
            for path in search_paths:
                if path.exists():
                    yaml_path = path
                    break

        # Load YAML if found
        if yaml_path and yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                # Deep merge
                if "openai" in loaded:
                    config_data["openai"].update(loaded["openai"])
                if "chunking" in loaded:
                    config_data["chunking"].update(loaded["chunking"])

        # Environment variables override YAML
        openai_config = config_data["openai"]
        chunking_config = config_data["chunking"]

        return cls(
            openai_api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or openai_config.get("base_url"),
            model_name=os.environ.get("OPENAI_MODEL") or openai_config.get("model", "gpt-4o-mini"),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", openai_config.get("temperature", 0.0))),
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", openai_config.get("max_tokens", 4096))),
            chunk_size=int(os.environ.get("CHUNK_SIZE", chunking_config.get("chunk_size", 1000))),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", chunking_config.get("chunk_overlap", 200))),
        )

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables (legacy method).

        This method is kept for backward compatibility.
        Use from_yaml() for full configuration support.
        """
        return cls.from_yaml()
