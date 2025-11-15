"""
Base Adapter Interface for AetherGrid
Defines the contract that all model adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AdapterMetadata:
    """Metadata about a model adapter"""
    model_name: str
    provider: str
    vector_dimensions: int
    max_context_window: int
    supports_streaming: bool = False
    supports_function_calling: bool = False


class BaseAdapter(ABC):
    """
    Abstract base class for all model adapters.

    Each adapter handles model-specific operations for capturing conversations,
    querying intelligence, and interacting with the AetherGrid.
    """

    def __init__(self, conversation_monitor, weaviate_manager):
        """
        Initialize the adapter

        Args:
            conversation_monitor: ConversationMonitor instance
            weaviate_manager: WeaviateManager instance
        """
        self.monitor = conversation_monitor
        self.weaviate = weaviate_manager

    @property
    @abstractmethod
    def metadata(self) -> AdapterMetadata:
        """
        Return adapter metadata

        Returns:
            AdapterMetadata instance
        """
        pass

    @abstractmethod
    async def capture_conversation(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Capture a conversation from this model

        Args:
            messages: List of message dicts with 'role' and 'content'
            conversation_id: Optional conversation ID

        Returns:
            Dictionary with capture result
        """
        pass

    @abstractmethod
    async def query_intelligence(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query the intelligence grid

        Args:
            query: Query string
            filters: Optional filters
            max_results: Maximum results to return

        Returns:
            List of intelligence fragments
        """
        pass

    def build_context_from_intelligence(
        self,
        intelligence_fragments: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> str:
        """
        Build a context string from intelligence fragments

        Args:
            intelligence_fragments: List of fragments from query
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        if not intelligence_fragments:
            return ""

        context_parts = []
        total_length = 0

        for i, fragment in enumerate(intelligence_fragments, 1):
            content = fragment.get("content", "")
            source = fragment.get("sourceModel", "unknown")
            certainty = fragment.get("certainty", 0)

            # Format fragment
            fragment_text = (
                f"[Fragment {i} - {source} - confidence: {certainty:.2f}]\n"
                f"{content}\n"
            )

            # Check token limit (rough estimate: 4 chars per token)
            if total_length + len(fragment_text) > max_tokens * 4:
                break

            context_parts.append(fragment_text)
            total_length += len(fragment_text)

        return "\n---\n".join(context_parts)

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about this adapter

        Returns:
            Dictionary with adapter information
        """
        meta = self.metadata
        return {
            "model_name": meta.model_name,
            "provider": meta.provider,
            "vector_dimensions": meta.vector_dimensions,
            "max_context_window": meta.max_context_window,
            "supports_streaming": meta.supports_streaming,
            "supports_function_calling": meta.supports_function_calling,
            "adapter_type": self.__class__.__name__
        }
