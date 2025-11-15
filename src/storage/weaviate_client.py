"""
Weaviate Client for AetherGrid
Manages all vector database operations for storing and querying intelligence fragments.
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WeaviateManager:
    """Manages all Weaviate vector database operations for AetherGrid"""

    def __init__(self, url: str = None, openai_api_key: str = None):
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8101")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for Weaviate vectorization")

        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(self.url.split(":")[-1]) if ":" in self.url else 8080,
            headers={"X-OpenAI-Api-Key": self.openai_api_key}
        )

        logger.info(f"Connected to Weaviate at {self.url}")
        self._ensure_schema()

    def _ensure_schema(self):
        """Create the IntelligenceFragment collection if it doesn't exist"""
        try:
            # Check if collection exists
            if self.client.collections.exists("IntelligenceFragment"):
                logger.info("✓ IntelligenceFragment collection already exists")
                self.collection = self.client.collections.get("IntelligenceFragment")
                return

            # Create the collection with schema
            self.collection = self.client.collections.create(
                name="IntelligenceFragment",
                description="A fragment of AI intelligence from conversations",
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small",
                    dimensions=1536,
                    vectorize_collection_name=False
                ),
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="The actual text content",
                        vectorize_property_name=False,
                        skip_vectorization=False
                    ),
                    Property(
                        name="conversationId",
                        data_type=DataType.TEXT,
                        description="UUID of the parent conversation",
                        skip_vectorization=True
                    ),
                    Property(
                        name="messageId",
                        data_type=DataType.TEXT,
                        description="UUID of the specific message",
                        skip_vectorization=True
                    ),
                    Property(
                        name="timestamp",
                        data_type=DataType.DATE,
                        description="When this intelligence was captured",
                        skip_vectorization=True
                    ),
                    Property(
                        name="sourceModel",
                        data_type=DataType.TEXT,
                        description="Which AI model generated this",
                        skip_vectorization=True
                    ),
                    Property(
                        name="topic",
                        data_type=DataType.TEXT,
                        description="Main topic/category",
                        skip_vectorization=True
                    ),
                    Property(
                        name="taskType",
                        data_type=DataType.TEXT,
                        description="Type of task (coding, writing, analysis, etc)",
                        skip_vectorization=True
                    ),
                    Property(
                        name="complexity",
                        data_type=DataType.NUMBER,
                        description="Complexity score 0-1",
                        skip_vectorization=True
                    ),
                    Property(
                        name="tokensUsed",
                        data_type=DataType.INT,
                        description="Token count for this fragment",
                        skip_vectorization=True
                    )
                ]
            )

            logger.info("✓ Created IntelligenceFragment collection with schema")

        except Exception as e:
            logger.error(f"Error ensuring schema: {e}")
            # Try to get existing collection
            if self.client.collections.exists("IntelligenceFragment"):
                self.collection = self.client.collections.get("IntelligenceFragment")
            else:
                raise

    def store_fragment(self, fragment: Dict[str, Any]) -> str:
        """
        Store an intelligence fragment and return its UUID

        Args:
            fragment: Dictionary with keys: content, conversation_id, message_id,
                     timestamp, source_model, and optional: topic, task_type,
                     complexity, tokens

        Returns:
            UUID of the stored fragment
        """
        try:
            data_object = {
                "content": fragment["content"],
                "conversationId": fragment["conversation_id"],
                "messageId": fragment["message_id"],
                "timestamp": fragment["timestamp"],
                "sourceModel": fragment["source_model"],
                "topic": fragment.get("topic", "general"),
                "taskType": fragment.get("task_type", "chat"),
                "complexity": fragment.get("complexity", 0.5),
                "tokensUsed": fragment.get("tokens", 0)
            }

            uuid = self.collection.data.insert(properties=data_object)
            logger.debug(f"Stored fragment {uuid}")
            return str(uuid)

        except Exception as e:
            logger.error(f"Error storing fragment: {e}")
            raise

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
        min_certainty: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar intelligence fragments

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters (sourceModel, taskType, etc.)
            min_certainty: Minimum similarity threshold (0-1)

        Returns:
            List of matching fragments with metadata
        """
        try:
            # Build the query
            query_builder = self.collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(certainty=True, distance=True)
            )

            # Execute query
            response = query_builder

            # Format results
            results = []
            for obj in response.objects:
                result = {
                    "content": obj.properties.get("content"),
                    "conversationId": obj.properties.get("conversationId"),
                    "messageId": obj.properties.get("messageId"),
                    "timestamp": obj.properties.get("timestamp"),
                    "sourceModel": obj.properties.get("sourceModel"),
                    "topic": obj.properties.get("topic"),
                    "taskType": obj.properties.get("taskType"),
                    "certainty": obj.metadata.certainty if obj.metadata else None,
                    "distance": obj.metadata.distance if obj.metadata else None
                }

                # Filter by certainty if specified
                if result["certainty"] and result["certainty"] >= min_certainty:
                    results.append(result)

            logger.info(f"Semantic search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored intelligence fragments"""
        try:
            # Get total count
            aggregate = self.collection.aggregate.over_all(total_count=True)

            return {
                "total_fragments": aggregate.total_count,
                "collection_name": "IntelligenceFragment"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_fragments": 0, "error": str(e)}

    def close(self):
        """Close the Weaviate connection"""
        try:
            self.client.close()
            logger.info("Weaviate connection closed")
        except Exception as e:
            logger.error(f"Error closing Weaviate connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
