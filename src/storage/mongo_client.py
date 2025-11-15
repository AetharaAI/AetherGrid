"""
MongoDB Client for AetherGrid
Handles raw conversation storage and event logging.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MongoManager:
    """Manages MongoDB operations for raw conversation data and events"""

    def __init__(self, url: str = None):
        self.url = url or os.getenv("MONGODB_URL", "mongodb://localhost:27018/aethergrid")

        try:
            self.client = MongoClient(self.url)
            # Test connection
            self.client.admin.command('ping')

            # Get database
            self.db = self.client.aethergrid

            # Collections
            self.conversations = self.db.conversations
            self.messages = self.db.messages
            self.events = self.db.events

            logger.info(f"✓ Connected to MongoDB at {self.url}")
            self._ensure_indexes()

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def _ensure_indexes(self):
        """Create necessary indexes for performance"""
        try:
            # Conversations indexes
            self.conversations.create_index([("conversation_id", ASCENDING)], unique=True)
            self.conversations.create_index([("timestamp", DESCENDING)])
            self.conversations.create_index([("model", ASCENDING)])

            # Messages indexes
            self.messages.create_index([("message_id", ASCENDING)], unique=True)
            self.messages.create_index([("conversation_id", ASCENDING)])
            self.messages.create_index([("timestamp", DESCENDING)])
            self.messages.create_index([("processed", ASCENDING)])

            # Events indexes
            self.events.create_index([("event_type", ASCENDING)])
            self.events.create_index([("timestamp", DESCENDING)])

            logger.info("✓ MongoDB indexes ensured")

        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

    def store_message(self, message_doc: Dict[str, Any]) -> str:
        """
        Store a raw message document

        Args:
            message_doc: Message document with keys: message_id, conversation_id,
                        timestamp, role, content, model, metadata, processed

        Returns:
            message_id of the stored message
        """
        try:
            # Ensure required fields
            if "timestamp" not in message_doc:
                message_doc["timestamp"] = datetime.utcnow().isoformat()

            if "processed" not in message_doc:
                message_doc["processed"] = False

            # Insert message
            self.messages.insert_one(message_doc)
            logger.debug(f"Stored message {message_doc['message_id'][:8]}...")

            # Update conversation metadata
            self._update_conversation_metadata(
                message_doc["conversation_id"],
                message_doc.get("model", "unknown")
            )

            return message_doc["message_id"]

        except DuplicateKeyError:
            logger.warning(f"Message {message_doc['message_id']} already exists")
            return message_doc["message_id"]
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            raise

    def _update_conversation_metadata(self, conversation_id: str, model: str):
        """Update or create conversation metadata"""
        try:
            self.conversations.update_one(
                {"conversation_id": conversation_id},
                {
                    "$set": {
                        "model": model,
                        "last_updated": datetime.utcnow().isoformat()
                    },
                    "$inc": {"message_count": 1},
                    "$setOnInsert": {
                        "conversation_id": conversation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error updating conversation metadata: {e}")

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a message by ID"""
        try:
            return self.messages.find_one({"message_id": message_id})
        except Exception as e:
            logger.error(f"Error retrieving message: {e}")
            return None

    def get_unprocessed_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages that haven't been processed yet"""
        try:
            messages = self.messages.find(
                {"processed": False}
            ).limit(limit).sort("timestamp", ASCENDING)

            return list(messages)
        except Exception as e:
            logger.error(f"Error getting unprocessed messages: {e}")
            return []

    def mark_message_processed(self, message_id: str) -> bool:
        """Mark a message as processed"""
        try:
            result = self.messages.update_one(
                {"message_id": message_id},
                {
                    "$set": {
                        "processed": True,
                        "processed_at": datetime.utcnow().isoformat()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error marking message processed: {e}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation metadata"""
        try:
            return self.conversations.find_one({"conversation_id": conversation_id})
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return None

    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all messages in a conversation"""
        try:
            messages = self.messages.find(
                {"conversation_id": conversation_id}
            ).sort("timestamp", ASCENDING).limit(limit)

            return list(messages)
        except Exception as e:
            logger.error(f"Error getting conversation messages: {e}")
            return []

    def log_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Log an event for analytics

        Args:
            event_type: Type of event (e.g., 'capture', 'process', 'query')
            data: Event data

        Returns:
            Event ID
        """
        try:
            event_doc = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }

            result = self.events.insert_one(event_doc)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            return {
                "total_conversations": self.conversations.count_documents({}),
                "total_messages": self.messages.count_documents({}),
                "unprocessed_messages": self.messages.count_documents({"processed": False}),
                "total_events": self.events.count_documents({})
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def close(self):
        """Close MongoDB connection"""
        try:
            self.client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
