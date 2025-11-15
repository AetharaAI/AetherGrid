"""
Weaviate Client for AetherGrid
Manages all vector database operations for storing and querying intelligence fragments.

Updated for local-first embeddings - no OpenAI dependency.
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

    def __init__(self, url: str = None, vector_dimensions: int = None):
        """
        Initialize Weaviate client

        Args:
            url: Weaviate URL (default: from env or localhost:8101)
            vector_dimensions: Vector dimensions (default: from env or 768)
        """
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8101")
        self.vector_dimensions = vector_dimensions or int(os.getenv("VECTOR_DIMENSIONS", "768"))

        # Connect to Weaviate (no API keys needed - we provide vectors directly)
        self.client = weaviate.connect_to_local(
            host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(self.url.split(":")[-1]) if ":" in self.url else 8080
        )

        logger.info(f"Connected to Weaviate at {self.url}")
        logger.info(f"Using vector dimensions: {self.vector_dimensions}")
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
            # Use "none" vectorizer - we'll provide vectors directly
            self.collection = self.client.collections.create(
                name="IntelligenceFragment",
                description="A fragment of AI intelligence from conversations",
                vectorizer_config=Configure.Vectorizer.none(),  # No auto-vectorization
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="The actual text content"
                    ),
                    Property(
                        name="conversationId",
                        data_type=DataType.TEXT,
                        description="UUID of the parent conversation"
                    ),
                    Property(
                        name="messageId",
                        data_type=DataType.TEXT,
                        description="UUID of the specific message"
                    ),
                    Property(
                        name="timestamp",
                        data_type=DataType.DATE,
                        description="When this intelligence was captured"
                    ),
                    Property(
                        name="sourceModel",
                        data_type=DataType.TEXT,
                        description="Which AI model generated this"
                    ),
                    Property(
                        name="topic",
                        data_type=DataType.TEXT,
                        description="Main topic/category"
                    ),
                    Property(
                        name="taskType",
                        data_type=DataType.TEXT,
                        description="Type of task (coding, writing, analysis, etc)"
                    ),
                    Property(
                        name="complexity",
                        data_type=DataType.NUMBER,
                        description="Complexity score 0-1"
                    ),
                    Property(
                        name="tokensUsed",
                        data_type=DataType.INT,
                        description="Token count for this fragment"
                    )
                ]
            )

            logger.info(f"✓ Created IntelligenceFragment collection ({self.vector_dimensions}D vectors)")

        except Exception as e:
            logger.error(f"Error ensuring schema: {e}")
            # Try to get existing collection
            if self.client.collections.exists("IntelligenceFragment"):
                self.collection = self.client.collections.get("IntelligenceFragment")
            else:
                raise

    def store_fragment(self, fragment: Dict[str, Any], vector: List[float] = None) -> str:
        """
        Store an intelligence fragment with its vector

        Args:
            fragment: Dictionary with keys: content, conversation_id, message_id,
                     timestamp, source_model, and optional: topic, task_type,
                     complexity, tokens
            vector: Pre-computed embedding vector (required!)

        Returns:
            UUID of the stored fragment
        """
        if vector is None:
            raise ValueError("vector is required - must provide pre-computed embedding")

        if len(vector) != self.vector_dimensions:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dimensions}, "
                f"got {len(vector)}"
            )

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

            uuid = self.collection.data.insert(
                properties=data_object,
                vector=vector  # Provide the vector directly
            )

            logger.debug(f"Stored fragment {uuid}")
            return str(uuid)

        except Exception as e:
            logger.error(f"Error storing fragment: {e}")
            raise

    def semantic_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
        min_certainty: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar intelligence fragments using vector

        Args:
            query_vector: Query embedding vector (pre-computed!)
            limit: Maximum number of results
            filters: Optional filters (sourceModel, taskType, etc.)
            min_certainty: Minimum similarity threshold (0-1)

        Returns:
            List of matching fragments with metadata
        """
        if len(query_vector) != self.vector_dimensions:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.vector_dimensions}, "
                f"got {len(query_vector)}"
            )

        try:
            # Build the query using near_vector (not near_text)
            query_builder = self.collection.query.near_vector(
                near_vector=query_vector,
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
                "collection_name": "IntelligenceFragment",
                "vector_dimensions": self.vector_dimensions
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
