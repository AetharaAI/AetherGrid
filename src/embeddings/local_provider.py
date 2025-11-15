"""
Local Embedding Provider for AetherGrid
Uses sentence-transformers for local, privacy-first embeddings.
"""

from typing import List
import logging
import torch
from sentence_transformers import SentenceTransformer

from .base_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Features:
    - Runs completely offline (after model download)
    - No API calls or costs
    - GPU acceleration if available
    - Privacy-first: your data never leaves your machine
    - Fast batch processing
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize local embedding provider

        Args:
            model_name: Sentence-transformers model name
                       Default: "all-mpnet-base-v2" (768 dimensions, best quality/speed)
        """
        super().__init__(model_name)
        self.model = None
        self.device = None

    def initialize(self) -> None:
        """Initialize the sentence-transformers model"""
        if self._initialized:
            return

        try:
            # Determine device (GPU if available, otherwise CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading local embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            # Load model (will download first time, then cache in ~/.cache/huggingface)
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Get model dimension
            self._dimensions = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"✓ Local embedding model loaded: {self.model_name} "
                f"({self._dimensions} dimensions, device: {self.device})"
            )

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize local embedding model: {e}")
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
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # L2 normalization for better similarity
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
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
            # Batch encoding for efficiency
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,  # Show progress for large batches
                batch_size=32,  # Process in batches of 32
                normalize_embeddings=True
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def get_dimensions(self) -> int:
        """Get embedding dimensions"""
        self.ensure_initialized()
        return self._dimensions

    @property
    def provider_name(self) -> str:
        """Provider name"""
        return "local"

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
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
