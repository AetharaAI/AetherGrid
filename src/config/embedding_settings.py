"""
Embedding Settings Configuration for AetherGrid
Manages configuration for local and API-based embedding providers.
"""

import os
from typing import Literal
from pydantic import BaseModel, Field


class EmbeddingSettings(BaseModel):
    """Configuration for embedding providers"""

    # Primary provider selection
    primary_provider: Literal["local", "cohere"] = Field(
        default="local",
        description="Primary embedding provider to use"
    )

    # Fallback configuration
    fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback to alternate provider on failure"
    )

    # Local embeddings configuration
    local_model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence-transformers model name"
    )

    # Cohere configuration
    cohere_model_name: str = Field(
        default="embed-english-v3.0",
        description="Cohere embedding model (embed-english-v3.0 or embed-multilingual-v3.0)"
    )
    cohere_api_key: str = Field(
        default="",
        description="Cohere API key (optional)"
    )

    # General configuration
    vector_dimensions: int = Field(
        default=768,
        description="Embedding vector dimensions"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        """Load settings from environment variables"""
        return cls(
            primary_provider=os.getenv("PRIMARY_EMBEDDING_PROVIDER", "local"),
            fallback_enabled=os.getenv("FALLBACK_ENABLED", "true").lower() == "true",
            local_model_name=os.getenv("LOCAL_MODEL_NAME", "all-mpnet-base-v2"),
            cohere_model_name=os.getenv("COHERE_MODEL_NAME", "embed-english-v3.0"),
            cohere_api_key=os.getenv("COHERE_API_KEY", ""),
            vector_dimensions=int(os.getenv("VECTOR_DIMENSIONS", "768")),
            batch_size=int(os.getenv("PROCESSING_BATCH_SIZE", "32")),
        )

    def get_fallback_provider(self) -> str:
        """Get the fallback provider name"""
        return "cohere" if self.primary_provider == "local" else "local"

    def is_cohere_available(self) -> bool:
        """Check if Cohere provider is available"""
        return bool(self.cohere_api_key)


# Model dimension mapping for reference
MODEL_DIMENSIONS = {
    # Sentence-transformers models
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-mpnet-base-v2": 768,
    "multi-qa-mpnet-base-dot-v1": 768,

    # Cohere models
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


def get_model_dimensions(model_name: str, provider: str = "local") -> int:
    """
    Get vector dimensions for a model

    Args:
        model_name: Name of the embedding model
        provider: Provider type ("local" or "cohere")

    Returns:
        Number of dimensions
    """
    if model_name in MODEL_DIMENSIONS:
        return MODEL_DIMENSIONS[model_name]

    # Default dimensions by provider
    if provider == "cohere":
        return 1024  # Cohere v3.0 default
    else:
        return 768  # mpnet default


# Global settings instance
_settings: EmbeddingSettings = None


def get_settings() -> EmbeddingSettings:
    """Get global embedding settings instance"""
    global _settings
    if _settings is None:
        _settings = EmbeddingSettings.from_env()
    return _settings


def reload_settings() -> EmbeddingSettings:
    """Reload settings from environment"""
    global _settings
    _settings = EmbeddingSettings.from_env()
    return _settings
