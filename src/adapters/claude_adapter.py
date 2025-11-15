"""
Claude Adapter for AetherGrid
Handles Claude-specific conversation capture and intelligence queries.
"""

from typing import Dict, Any, List, Optional
import logging
from .base_adapter import BaseAdapter, AdapterMetadata

logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseAdapter):
    """
    Adapter for Claude AI models (Anthropic)

    Handles capturing Claude conversations and providing
    Claude-optimized intelligence queries.
    """

    @property
    def metadata(self) -> AdapterMetadata:
        """Claude adapter metadata"""
        return AdapterMetadata(
            model_name="claude-sonnet-4-5",
            provider="anthropic",
            vector_dimensions=1536,  # Using OpenAI embeddings
            max_context_window=200000,
            supports_streaming=True,
            supports_function_calling=True
        )

    async def capture_conversation(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Capture a Claude conversation

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Role can be 'user' or 'assistant'
            conversation_id: Optional conversation ID

        Returns:
            Dictionary with:
                - conversation_id: ID of the conversation
                - message_ids: List of captured message IDs
                - message_count: Number of messages captured
        """
        try:
            logger.info(
                f"Capturing Claude conversation with {len(messages)} messages"
            )

            result = await self.monitor.capture_conversation(
                messages=messages,
                conversation_id=conversation_id,
                model=self.metadata.model_name
            )

            logger.info(
                f"✓ Captured Claude conversation {result['conversation_id'][:8]}..."
            )

            return result

        except Exception as e:
            logger.error(f"Error capturing Claude conversation: {e}")
            raise

    async def query_intelligence(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: int = 10,
        min_certainty: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Query the intelligence grid for relevant fragments

        Args:
            query: Natural language query
            filters: Optional filters (e.g., {"sourceModel": "claude-sonnet-4-5"})
            max_results: Maximum number of results
            min_certainty: Minimum similarity threshold (0-1)

        Returns:
            List of intelligence fragments with metadata
        """
        try:
            logger.info(f"Querying intelligence grid: '{query[:50]}...'")

            # Query Weaviate
            results = self.weaviate.semantic_search(
                query=query,
                limit=max_results,
                filters=filters,
                min_certainty=min_certainty
            )

            logger.info(f"✓ Found {len(results)} intelligence fragments")

            return results

        except Exception as e:
            logger.error(f"Error querying intelligence: {e}")
            raise

    async def get_claude_context(
        self,
        query: str,
        max_results: int = 5,
        include_all_models: bool = False
    ) -> str:
        """
        Get a formatted context string optimized for Claude

        Args:
            query: What you're looking for
            max_results: Maximum number of fragments to include
            include_all_models: If False, only include Claude intelligence

        Returns:
            Formatted context string ready to inject into Claude prompt
        """
        try:
            # Set filters
            filters = None if include_all_models else {
                "sourceModel": self.metadata.model_name
            }

            # Query intelligence
            fragments = await self.query_intelligence(
                query=query,
                filters=filters,
                max_results=max_results
            )

            if not fragments:
                return "No relevant intelligence found in the grid."

            # Build Claude-optimized context
            context_parts = [
                "# Intelligence from AetherGrid\n",
                f"Query: {query}\n",
                f"Found {len(fragments)} relevant fragments:\n"
            ]

            for i, fragment in enumerate(fragments, 1):
                content = fragment.get("content", "")
                source = fragment.get("sourceModel", "unknown")
                topic = fragment.get("topic", "general")
                certainty = fragment.get("certainty", 0)

                context_parts.append(
                    f"\n## Fragment {i}\n"
                    f"**Source**: {source} | **Topic**: {topic} | "
                    f"**Relevance**: {certainty:.1%}\n\n"
                    f"{content}\n"
                )

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error building Claude context: {e}")
            return f"Error retrieving intelligence: {str(e)}"

    async def capture_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Capture a Claude conversation from a file

        Useful for batch importing historical conversations.

        Args:
            file_path: Path to conversation file (JSON format expected)

        Returns:
            Capture result dictionary
        """
        import json

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract messages
            messages = data.get("messages", [])

            if not messages:
                raise ValueError("No messages found in file")

            # Capture conversation
            return await self.capture_conversation(
                messages=messages,
                conversation_id=data.get("conversation_id")
            )

        except Exception as e:
            logger.error(f"Error capturing from file: {e}")
            raise
