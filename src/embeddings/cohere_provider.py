"""
Cohere Embedding Provider for AetherGrid
Uses Cohere API for high-quality embeddings when needed.
"""

from typing import List
import logging
import cohere

from .base_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class CohereEmbeddingProvider(EmbeddingProvider):
    """
    Cohere embedding provider using Cohere API.

    Features:
    - High-quality multilingual embeddings
    - 1024 dimensions (v3.0 models)
    - Batch processing support
    - Production-grade reliability
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }

    def __init__(self, model_name: str = "embed-english-v3.0", api_key: str = ""):
        """
        Initialize Cohere embedding provider

        Args:
            model_name: Cohere model name
            api_key: Cohere API key
        """
        super().__init__(model_name)
        self.api_key = api_key
        self.client = None

    def initialize(self) -> None:
        """Initialize the Cohere client"""
        if self._initialized:
            return

        if not self.api_key:
            raise ValueError(
                "Cohere API key is required. "
                "Set COHERE_API_KEY environment variable or pass api_key parameter."
            )

        try:
            logger.info(f"Initializing Cohere embedding provider: {self.model_name}")

            # Initialize Cohere client
            self.client = cohere.Client(self.api_key)

            # Get model dimensions
            self._dimensions = self.MODEL_DIMENSIONS.get(self.model_name, 1024)

            logger.info(
                f"✓ Cohere embedding provider initialized: {self.model_name} "
                f"({self._dimensions} dimensions)"
            )

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Cohere provider: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        self.ensure_initialized()

        try:
            # Generate embedding
            response = self.client.embed(
                texts=[text],
                model=self.model_name,
                input_type="search_document"  # For storage/indexing
            )

            return response.embeddings[0]

        except Exception as e:
            logger.error(f"Error generating Cohere embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self.ensure_initialized()

        if not texts:
            return []

        try:
            # Cohere supports batch processing natively
            # Process in chunks of 96 (Cohere's max batch size)
            max_batch_size = 96
            all_embeddings = []

            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]

                response = self.client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type="search_document"
                )

                all_embeddings.extend(response.embeddings)

                if len(texts) > max_batch_size:
                    logger.debug(f"Processed batch {i//max_batch_size + 1}/{(len(texts)-1)//max_batch_size + 1}")

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating batch Cohere embeddings: {e}")
            raise

    def get_dimensions(self) -> int:
        """Get embedding dimensions"""
        self.ensure_initialized()
        return self._dimensions

    @property
    def provider_name(self) -> str:
        """Provider name"""
        return "cohere"

    def get_model_info(self) -> dict:
        """
        Get detailed model information

        Returns:
            Dictionary with model details
        """
        self.ensure_initialized()

        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "dimensions": self._dimensions,
            "api_key_set": bool(self.api_key),
            "max_batch_size": 96,
        }
