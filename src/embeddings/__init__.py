"""
Embedding Providers for AetherGrid
Local-first, privacy-focused embedding generation.
"""

from .base_provider import EmbeddingProvider
from .local_provider import LocalEmbeddingProvider
from .cohere_provider import CohereEmbeddingProvider
from ..config.embedding_settings import get_settings, EmbeddingSettings
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "CohereEmbeddingProvider",
    "get_embedding_provider",
    "get_embedding_provider_with_fallback"
]


def get_embedding_provider(
    provider_type: str = None,
    settings: EmbeddingSettings = None
) -> EmbeddingProvider:
    """
    Create an embedding provider instance

    Args:
        provider_type: Type of provider ("local" or "cohere").
                      If None, uses settings.primary_provider
        settings: Embedding settings. If None, loads from environment.

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider_type is invalid or required config is missing
    """
    if settings is None:
        settings = get_settings()

    if provider_type is None:
        provider_type = settings.primary_provider

    logger.info(f"Creating {provider_type} embedding provider...")

    if provider_type == "local":
        provider = LocalEmbeddingProvider(model_name=settings.local_model_name)

    elif provider_type == "cohere":
        if not settings.is_cohere_available():
            raise ValueError(
                "Cohere provider requires COHERE_API_KEY. "
                "Set it in your environment or .env file."
            )
        provider = CohereEmbeddingProvider(
            model_name=settings.cohere_model_name,
            api_key=settings.cohere_api_key
        )

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Must be 'local' or 'cohere'."
        )

    return provider


def get_embedding_provider_with_fallback(
    settings: EmbeddingSettings = None
) -> tuple:
    """
    Create primary and fallback embedding providers

    Args:
        settings: Embedding settings. If None, loads from environment.

    Returns:
        Tuple of (primary_provider, fallback_provider)
        fallback_provider will be None if fallback is disabled or unavailable

    Example:
        primary, fallback = get_embedding_provider_with_fallback()

        try:
            embeddings = primary.generate_embeddings_batch(texts)
        except Exception as e:
            if fallback:
                logger.warning(f"Primary failed, using fallback: {e}")
                embeddings = fallback.generate_embeddings_batch(texts)
            else:
                raise
    """
    if settings is None:
        settings = get_settings()

    # Create primary provider
    primary = get_embedding_provider(settings.primary_provider, settings)

    # Create fallback if enabled
    fallback = None
    if settings.fallback_enabled:
        fallback_type = settings.get_fallback_provider()

        try:
            fallback = get_embedding_provider(fallback_type, settings)
            logger.info(f"Fallback provider configured: {fallback_type}")
        except Exception as e:
            logger.warning(f"Could not create fallback provider ({fallback_type}): {e}")
            logger.info("Continuing without fallback")

    return primary, fallback
