"""
Conversation Monitor for AetherGrid
Captures AI conversations and queues them for processing.
"""

import asyncio
from datetime import datetime
import uuid
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConversationMonitor:
    """Monitors and captures AI conversations in real-time"""

    def __init__(self, mongo_manager, redis_manager, postgres_manager=None):
        """
        Initialize the conversation monitor

        Args:
            mongo_manager: MongoManager instance for raw storage
            redis_manager: RedisManager instance for queuing
            postgres_manager: Optional PostgresManager for metadata
        """
        self.mongo = mongo_manager
        self.redis = redis_manager
        self.postgres = postgres_manager
        self.is_running = False
        self.stats = {
            "messages_captured": 0,
            "errors": 0,
            "started_at": None
        }

    async def start(self):
        """Start the monitoring service"""
        self.is_running = True
        self.stats["started_at"] = datetime.utcnow().isoformat()
        logger.info("🟢 Conversation monitor started")

        # In a production system, this would monitor actual conversation sources
        # (browser extension, API proxy, file watcher, etc.)
        # For now, it runs as a background service ready to accept captures
        while self.is_running:
            await asyncio.sleep(1)

    async def stop(self):
        """Stop the monitoring service"""
        self.is_running = False
        logger.info("🔴 Conversation monitor stopped")

    async def capture_message(
        self,
        content: str,
        role: str,
        conversation_id: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Capture a single message and queue for processing

        Args:
            content: The message content
            role: Message role (user/assistant)
            conversation_id: Optional conversation ID (generated if not provided)
            model: AI model name
            metadata: Optional additional metadata

        Returns:
            message_id of the captured message
        """
        try:
            # Generate IDs
            message_id = str(uuid.uuid4())
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Create message document
            message_doc = {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "role": role,
                "content": content,
                "model": model,
                "metadata": metadata or {},
                "processed": False
            }

            # Store raw in MongoDB immediately
            await self._store_raw_message(message_doc)

            # Queue for processing
            await self._queue_for_processing(message_doc)

            # Log to PostgreSQL if available
            if self.postgres:
                await self._log_metadata(message_doc)

            # Update stats
            self.stats["messages_captured"] += 1

            logger.info(
                f"✓ Captured message {message_id[:8]}... "
                f"({len(content)} chars, role={role})"
            )

            return message_id

        except Exception as e:
            logger.error(f"Error capturing message: {e}")
            self.stats["errors"] += 1
            raise

    async def capture_conversation(
        self,
        messages: list[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        model: str = "claude-sonnet-4-5"
    ) -> Dict[str, Any]:
        """
        Capture an entire conversation at once

        Args:
            messages: List of messages with 'role' and 'content' keys
            conversation_id: Optional conversation ID
            model: AI model name

        Returns:
            Dictionary with conversation_id and list of message_ids
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        message_ids = []

        for msg in messages:
            try:
                msg_id = await self.capture_message(
                    content=msg["content"],
                    role=msg["role"],
                    conversation_id=conversation_id,
                    model=model,
                    metadata=msg.get("metadata")
                )
                message_ids.append(msg_id)
            except Exception as e:
                logger.error(f"Error capturing message in conversation: {e}")
                continue

        logger.info(
            f"✓ Captured conversation {conversation_id[:8]}... "
            f"with {len(message_ids)} messages"
        )

        return {
            "conversation_id": conversation_id,
            "message_ids": message_ids,
            "message_count": len(message_ids)
        }

    async def _store_raw_message(self, message_doc: Dict):
        """Store raw message in MongoDB"""
        try:
            # Use sync MongoDB client (wrapped in executor for async)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.mongo.store_message,
                message_doc
            )
        except Exception as e:
            logger.error(f"Error storing raw message: {e}")
            raise

    async def _queue_for_processing(self, message_doc: Dict):
        """Add message to Redis processing queue"""
        try:
            await self.redis.push_to_queue("processing:queue", message_doc)
        except Exception as e:
            logger.error(f"Error queueing message: {e}")
            raise

    async def _log_metadata(self, message_doc: Dict):
        """Log message metadata to PostgreSQL"""
        try:
            # Use executor for sync PostgreSQL client
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.postgres.log_conversation,
                message_doc["conversation_id"],
                message_doc["model"]
            )

            await asyncio.get_event_loop().run_in_executor(
                None,
                self.postgres.log_message,
                message_doc["message_id"],
                message_doc["conversation_id"],
                message_doc["role"],
                len(message_doc["content"].split())  # Simple token estimate
            )
        except Exception as e:
            logger.error(f"Error logging metadata: {e}")
            # Don't raise - metadata logging is not critical

    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return {
            **self.stats,
            "uptime_seconds": (
                (datetime.utcnow() - datetime.fromisoformat(self.stats["started_at"])).total_seconds()
                if self.stats["started_at"] else 0
            )
        }
