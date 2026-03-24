"""Configuration management for Tiny-Graph-RAG."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


def _env_or_default(env_name: str, default_value: object) -> str:
    """Return environment value or fallback as string."""
    value = os.environ.get(env_name)
    if value is not None:
        return value
    return str(default_value)


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
    kg_dir: Optional[str] = None
    dataset_dir: Optional[str] = None
    results_dir: Optional[str] = None

    @staticmethod
    def _resolve_yaml_path(config_path: Optional[str] = None) -> Optional[Path]:
        """Resolve config.yaml path from explicit path or default search paths."""
        if config_path:
            path = Path(config_path)
            return path if path.exists() else None

        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                return path
        return None

    @classmethod
    def load_storage_config(
        cls,
        config_path: Optional[str] = None,
    ) -> dict[str, Optional[str]]:
        """Load storage directory defaults without requiring API credentials."""
        storage_config = {
            "kg_dir": None,
            "dataset_dir": None,
            "results_dir": None,
        }

        yaml_path = cls._resolve_yaml_path(config_path)
        if yaml_path and yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if "storage" in loaded:
                    storage_config.update(loaded["storage"])

        return {
            "kg_dir": os.environ.get("KG_DIR") or storage_config.get("kg_dir"),
            "dataset_dir": os.environ.get("DATASET_DIR") or storage_config.get("dataset_dir"),
            "results_dir": os.environ.get("RESULTS_DIR") or storage_config.get("results_dir"),
        }

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
            "storage": {
                "kg_dir": None,
                "dataset_dir": None,
                "results_dir": None,
            },
        }

        yaml_path = cls._resolve_yaml_path(config_path)

        # Load YAML if found
        if yaml_path and yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                # Deep merge
                if "openai" in loaded:
                    config_data["openai"].update(loaded["openai"])
                if "chunking" in loaded:
                    config_data["chunking"].update(loaded["chunking"])
                if "storage" in loaded:
                    config_data["storage"].update(loaded["storage"])

        # Environment variables override YAML
        openai_config = config_data["openai"]
        chunking_config = config_data["chunking"]
        storage_config = config_data["storage"]
        return cls(
            openai_api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or openai_config.get("base_url"),
            model_name=os.environ.get("OPENAI_MODEL") or openai_config.get("model", "gpt-4o-mini"),
            temperature=float(_env_or_default("OPENAI_TEMPERATURE", openai_config.get("temperature", 0.0))),
            max_tokens=int(_env_or_default("OPENAI_MAX_TOKENS", openai_config.get("max_tokens", 4096))),
            chunk_size=int(_env_or_default("CHUNK_SIZE", chunking_config.get("chunk_size", 1000))),
            chunk_overlap=int(_env_or_default("CHUNK_OVERLAP", chunking_config.get("chunk_overlap", 200))),
            kg_dir=os.environ.get("KG_DIR") or storage_config.get("kg_dir"),
            dataset_dir=os.environ.get("DATASET_DIR") or storage_config.get("dataset_dir"),
            results_dir=os.environ.get("RESULTS_DIR") or storage_config.get("results_dir"),
        )

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables (legacy method).

        This method is kept for backward compatibility.
        Use from_yaml() for full configuration support.
        """
        return cls.from_yaml()
