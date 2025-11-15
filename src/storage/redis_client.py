"""
Redis Client for AetherGrid
Handles message queues, pub/sub, and caching.
"""

import redis.asyncio as aioredis
from redis.exceptions import RedisError
import os
import json
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis operations for queues, caching, and pub/sub"""

    def __init__(self, url: str = None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6395")
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        """Establish connection to Redis"""
        try:
            self.client = await aioredis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.client.ping()
            logger.info(f"✓ Connected to Redis at {self.url}")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def push_to_queue(self, queue_name: str, data: Dict[str, Any]) -> int:
        """
        Push data to a queue (list)

        Args:
            queue_name: Name of the queue
            data: Data to push (will be JSON serialized)

        Returns:
            New length of the queue
        """
        try:
            json_data = json.dumps(data)
            length = await self.client.lpush(queue_name, json_data)
            logger.debug(f"Pushed to queue {queue_name}, new length: {length}")
            return length
        except Exception as e:
            logger.error(f"Error pushing to queue {queue_name}: {e}")
            raise

    async def pop_from_queue(
        self,
        queue_name: str,
        timeout: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Pop data from a queue (blocking)

        Args:
            queue_name: Name of the queue
            timeout: Timeout in seconds (0 for no timeout)

        Returns:
            Deserialized data or None if timeout
        """
        try:
            result = await self.client.brpop(queue_name, timeout=timeout)

            if result:
                _, json_data = result
                return json.loads(json_data)
            return None

        except Exception as e:
            logger.error(f"Error popping from queue {queue_name}: {e}")
            return None

    async def get_queue_length(self, queue_name: str) -> int:
        """Get the current length of a queue"""
        try:
            return await self.client.llen(queue_name)
        except Exception as e:
            logger.error(f"Error getting queue length: {e}")
            return 0

    async def set_cache(
        self,
        key: str,
        value: Any,
        expire_seconds: Optional[int] = None
    ) -> bool:
        """
        Set a cache value

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if dict/list)
            expire_seconds: Optional expiration time in seconds

        Returns:
            Success boolean
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)

            await self.client.set(key, value)

            if expire_seconds:
                await self.client.expire(key, expire_seconds)

            logger.debug(f"Cached {key}")
            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def get_cache(self, key: str, as_json: bool = False) -> Optional[Any]:
        """
        Get a cache value

        Args:
            key: Cache key
            as_json: Whether to deserialize as JSON

        Returns:
            Cached value or None
        """
        try:
            value = await self.client.get(key)

            if value is None:
                return None

            if as_json:
                return json.loads(value)

            return value

        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None

    async def delete_cache(self, key: str) -> bool:
        """Delete a cache key"""
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False

    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """
        Publish a message to a pub/sub channel

        Args:
            channel: Channel name
            message: Message to publish (will be JSON serialized)

        Returns:
            Number of subscribers that received the message
        """
        try:
            json_message = json.dumps(message)
            subscribers = await self.client.publish(channel, json_message)
            logger.debug(f"Published to {channel}, {subscribers} subscribers")
            return subscribers
        except Exception as e:
            logger.error(f"Error publishing to {channel}: {e}")
            return 0

    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter

        Args:
            key: Counter key
            amount: Amount to increment by

        Returns:
            New value
        """
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing {key}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            info = await self.client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace": await self.client.dbsize()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            return await self.client.ping()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close Redis connection"""
        try:
            if self.client:
                await self.client.close()
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
