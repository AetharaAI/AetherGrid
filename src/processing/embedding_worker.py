"""
Embedding Worker for AetherGrid
Processes messages from queue and generates vector embeddings.
"""

import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
import tiktoken
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """Processes messages from queue and generates embeddings"""

    def __init__(
        self,
        redis_manager,
        weaviate_manager,
        mongo_manager,
        postgres_manager,
        openai_api_key: str,
        batch_size: int = 50
    ):
        """
        Initialize the embedding worker

        Args:
            redis_manager: RedisManager for queue operations
            weaviate_manager: WeaviateManager for vector storage
            mongo_manager: MongoManager for message updates
            postgres_manager: PostgresManager for metadata updates
            openai_api_key: OpenAI API key for embeddings
            batch_size: Batch size for processing
        """
        self.redis = redis_manager
        self.weaviate = weaviate_manager
        self.mongo = mongo_manager
        self.postgres = postgres_manager
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        self.batch_size = batch_size
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.is_running = False
        self.stats = {
            "messages_processed": 0,
            "chunks_created": 0,
            "errors": 0,
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

            # Chunk if needed (>8000 tokens)
            chunks = self._chunk_text(content)

            logger.info(
                f"📝 Processing {len(chunks)} chunk(s) for message {message_id[:8]}..."
            )

            # Process each chunk
            for i, chunk in enumerate(chunks):
                try:
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
                        "tokens": len(self.encoder.encode(chunk))
                    }

                    # Store in Weaviate (it will generate embeddings automatically)
                    uuid = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.weaviate.store_fragment,
                        fragment
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

    def _chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """
        Split text into chunks at semantic boundaries

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        tokens = self.encoder.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        # Simple sentence-based chunking
        chunks = []
        current_chunk = []
        current_length = 0

        # Split by sentences (simple approach)
        sentences = text.split('. ')

        for sentence in sentences:
            sentence_tokens = len(self.encoder.encode(sentence))

            if current_length + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk and start new one
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens

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
        token_count = len(self.encoder.encode(text))

        technical_terms = sum(
            1 for word in [
                "algorithm", "optimization", "architecture", "implementation",
                "asynchronous", "concurrent", "distributed", "scalability",
                "performance", "efficiency"
            ]
            if word in text.lower()
        )

        # Normalize complexity score
        length_score = min(1.0, token_count / 5000) * 0.7
        technical_score = min(1.0, technical_terms / 10) * 0.3

        complexity = length_score + technical_score

        return round(complexity, 2)

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            **self.stats,
            "uptime_seconds": (
                (datetime.utcnow() - datetime.fromisoformat(self.stats["started_at"])).total_seconds()
                if self.stats["started_at"] else 0
            ),
            "queue_length": asyncio.run(self.redis.get_queue_length("processing:queue"))
        }
