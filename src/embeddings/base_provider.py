"""
Base Embedding Provider Interface for AetherGrid
Defines the contract that all embedding providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers (local, Cohere, etc.) must implement this interface.
    """

    def __init__(self, model_name: str):
        """
        Initialize the embedding provider

        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the provider (load models, validate API keys, etc.)

        Should be called before first use. Can be called multiple times safely.
        """
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider

        Returns:
            Number of dimensions in embedding vectors
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of this provider

        Returns:
            Provider name (e.g., "local", "cohere")
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider has been initialized"""
        return self._initialized

    def ensure_initialized(self) -> None:
        """Ensure provider is initialized, initialize if not"""
        if not self._initialized:
            logger.info(f"Initializing {self.provider_name} embedding provider...")
            self.initialize()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, initialized={self._initialized})"
