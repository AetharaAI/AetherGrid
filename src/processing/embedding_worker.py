"""
Embedding Worker for AetherGrid
Processes messages from queue and generates vector embeddings using local or Cohere providers.

Updated for local-first embeddings - no OpenAI dependency.
"""

import asyncio
from typing import List, Dict, Any
import logging
from datetime import datetime

from src.embeddings import get_embedding_provider_with_fallback
from src.config.embedding_settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """Processes messages from queue and generates embeddings"""

    def __init__(
        self,
        redis_manager,
        weaviate_manager,
        mongo_manager,
        postgres_manager,
        batch_size: int = 50
    ):
        """
        Initialize the embedding worker

        Args:
            redis_manager: RedisManager for queue operations
            weaviate_manager: WeaviateManager for vector storage
            mongo_manager: MongoManager for message updates
            postgres_manager: PostgresManager for metadata updates
            batch_size: Batch size for processing
        """
        self.redis = redis_manager
        self.weaviate = weaviate_manager
        self.mongo = mongo_manager
        self.postgres = postgres_manager
        self.batch_size = batch_size

        # Load embedding settings
        self.settings = get_settings()

        # Initialize embedding providers (primary and fallback)
        self.primary_provider, self.fallback_provider = get_embedding_provider_with_fallback()

        # Initialize the primary provider
        self.primary_provider.initialize()
        logger.info(
            f"Primary embedding provider: {self.primary_provider.provider_name} "
            f"(model: {self.primary_provider.model_name}, "
            f"dimensions: {self.primary_provider.get_dimensions()})"
        )

        # Initialize fallback if available
        if self.fallback_provider:
            try:
                self.fallback_provider.initialize()
                logger.info(
                    f"Fallback provider: {self.fallback_provider.provider_name} "
                    f"(model: {self.fallback_provider.model_name})"
                )
            except Exception as e:
                logger.warning(f"Fallback provider initialization failed: {e}")
                self.fallback_provider = None

        self.is_running = False
        self.stats = {
            "messages_processed": 0,
            "chunks_created": 0,
            "errors": 0,
            "primary_used": 0,
            "fallback_used": 0,
            "started_at": None
        }

    async def start(self):
        """Start processing queue"""
        self.is_running = True
        self.stats["started_at"] = datetime.utcnow().isoformat()
        logger.info("🟢 Embedding worker started")

        while self.is_running:
            try:
                # Get message from queue (blocking with timeout)
                message = await self.redis.pop_from_queue("processing:queue", timeout=5)

                if message:
                    await self._process_message(message)

            except Exception as e:
                logger.error(f"❌ Processing error: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(1)

    async def stop(self):
        """Stop the worker"""
        self.is_running = False
        logger.info("🔴 Embedding worker stopped")

    async def _process_message(self, message: Dict):
        """
        Process a single message: chunk, embed, store

        Args:
            message: Message document from queue
        """
        try:
            content = message["content"]
            message_id = message["message_id"]

            # Chunk if needed (>8000 characters - simplified from tokens)
            chunks = self._chunk_text(content, max_chars=8000)

            logger.info(
                f"📝 Processing {len(chunks)} chunk(s) for message {message_id[:8]}..."
            )

            # Process each chunk
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding for this chunk
                    vector = await self._generate_embedding(chunk)

                    if not vector:
                        logger.error(f"Failed to generate embedding for chunk {i}")
                        continue

                    # Create fragment
                    fragment = {
                        "content": chunk,
                        "conversation_id": message["conversation_id"],
                        "message_id": (
                            f"{message_id}-chunk{i}"
                            if len(chunks) > 1
                            else message_id
                        ),
                        "timestamp": message["timestamp"],
                        "source_model": message["model"],
                        "topic": self._extract_topic(chunk),
                        "task_type": self._classify_task(chunk),
                        "complexity": self._estimate_complexity(chunk),
                        "tokens": len(chunk.split())  # Simple word count
                    }

                    # Store in Weaviate with vector
                    uuid = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.weaviate.store_fragment,
                        fragment,
                        vector
                    )

                    self.stats["chunks_created"] += 1

                    logger.info(f"  ✓ Stored chunk {i+1}/{len(chunks)}: {uuid[:8]}...")

                except Exception as e:
                    logger.error(f"  ❌ Failed to process chunk {i}: {e}")
                    self.stats["errors"] += 1

            # Mark message as processed in MongoDB
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.mongo.mark_message_processed,
                message_id
            )

            # Update PostgreSQL metadata
            if self.postgres:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.postgres.mark_message_embedded,
                    message_id
                )

            self.stats["messages_processed"] += 1
            logger.info(f"✅ Completed processing message {message_id[:8]}...")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats["errors"] += 1

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using primary provider with fallback

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None on failure
        """
        # Try primary provider
        try:
            vector = await asyncio.get_event_loop().run_in_executor(
                None,
                self.primary_provider.generate_embedding,
                text
            )
            self.stats["primary_used"] += 1
            return vector

        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")

            # Try fallback if available
            if self.fallback_provider:
                try:
                    logger.info("Using fallback provider...")
                    vector = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.fallback_provider.generate_embedding,
                        text
                    )
                    self.stats["fallback_used"] += 1
                    return vector

                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {fallback_error}")

            return None

    def _chunk_text(self, text: str, max_chars: int = 8000) -> List[str]:
        """
        Split text into chunks at semantic boundaries

        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]

        # Simple sentence-based chunking
        chunks = []
        current_chunk = []
        current_length = 0

        # Split by sentences (simple approach)
        sentences = text.split('. ')

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > max_chars and current_chunk:
                # Save current chunk and start new one
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + ('.' if not current_chunk[-1].endswith('.') else ''))

        logger.debug(f"Chunked text into {len(chunks)} chunks")
        return chunks

    def _extract_topic(self, text: str) -> str:
        """
        Extract main topic from text (simplified keyword-based)

        Args:
            text: Text to analyze

        Returns:
            Topic string
        """
        text_lower = text.lower()

        # Simple keyword matching (can be improved with ML)
        if any(word in text_lower for word in ["code", "function", "class", "import", "def", "const"]):
            return "programming"
        elif any(word in text_lower for word in ["data", "analysis", "model", "predict", "train"]):
            return "data-science"
        elif any(word in text_lower for word in ["design", "ui", "ux", "interface", "component"]):
            return "design"
        elif any(word in text_lower for word in ["database", "query", "sql", "schema"]):
            return "database"
        elif any(word in text_lower for word in ["deploy", "docker", "kubernetes", "server"]):
            return "devops"
        else:
            return "general"

    def _classify_task(self, text: str) -> str:
        """
        Classify the type of task (simplified)

        Args:
            text: Text to analyze

        Returns:
            Task type string
        """
        text_lower = text.lower()

        if any(word in text_lower for word in ["write", "create", "generate", "build"]):
            return "creation"
        elif any(word in text_lower for word in ["fix", "debug", "error", "bug", "issue"]):
            return "debugging"
        elif any(word in text_lower for word in ["explain", "how", "what", "why", "describe"]):
            return "explanation"
        elif any(word in text_lower for word in ["review", "analyze", "evaluate", "assess"]):
            return "analysis"
        elif any(word in text_lower for word in ["optimize", "improve", "refactor", "enhance"]):
            return "optimization"
        else:
            return "chat"

    def _estimate_complexity(self, text: str) -> float:
        """
        Estimate complexity 0-1 based on text features

        Args:
            text: Text to analyze

        Returns:
            Complexity score (0-1)
        """
        # Simple heuristic: longer + more technical = more complex
        word_count = len(text.split())

        technical_terms = sum(
            1 for word in [
                "algorithm", "optimization", "architecture", "implementation",
                "asynchronous", "concurrent", "distributed", "scalability",
                "performance", "efficiency"
            ]
            if word in text.lower()
        )

        # Normalize complexity score
        length_score = min(1.0, word_count / 1000) * 0.7
        technical_score = min(1.0, technical_terms / 10) * 0.3

        complexity = length_score + technical_score

        return round(complexity, 2)

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        uptime_seconds = 0
        if self.stats["started_at"]:
            uptime_seconds = (
                datetime.utcnow() -
                datetime.fromisoformat(self.stats["started_at"])
            ).total_seconds()

        return {
            **self.stats,
            "uptime_seconds": uptime_seconds,
            "queue_length": 0,  # Will be set by async call
            "primary_provider": self.primary_provider.provider_name,
            "fallback_provider": self.fallback_provider.provider_name if self.fallback_provider else None
        }
