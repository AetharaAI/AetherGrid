"""
PostgreSQL Client for AetherGrid
Handles metadata storage, analytics, and relational data.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PostgresManager:
    """Manages PostgreSQL operations for metadata and analytics"""

    def __init__(self, url: str = None, pool_size: int = 5):
        self.url = url or os.getenv(
            "POSTGRES_URL",
            "postgresql://aether:development@localhost:5433/aethergrid"
        )

        try:
            # Create connection pool
            self.pool = SimpleConnectionPool(
                1, pool_size,
                self.url
            )

            # Test connection
            conn = self.pool.getconn()
            conn.close()
            self.pool.putconn(conn)

            logger.info(f"✓ Connected to PostgreSQL with pool size {pool_size}")

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _get_conn(self):
        """Get a connection from the pool"""
        return self.pool.getconn()

    def _put_conn(self, conn):
        """Return connection to pool"""
        self.pool.putconn(conn)

    def log_conversation(
        self,
        conversation_id: str,
        model_name: str
    ) -> bool:
        """
        Log a new conversation or update existing

        Args:
            conversation_id: Unique conversation identifier
            model_name: Name of the AI model

        Returns:
            Success boolean
        """
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (conversation_id, model_name)
                    VALUES (%s, %s)
                    ON CONFLICT (conversation_id)
                    DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP,
                        total_messages = conversations.total_messages + 1
                    """,
                    (conversation_id, model_name)
                )
            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._put_conn(conn)

    def log_message(
        self,
        message_id: str,
        conversation_id: str,
        role: str,
        token_count: int = 0
    ) -> bool:
        """
        Log message metadata

        Args:
            message_id: Unique message identifier
            conversation_id: Parent conversation ID
            role: Message role (user/assistant)
            token_count: Number of tokens

        Returns:
            Success boolean
        """
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO message_metadata
                        (message_id, conversation_id, role, token_count)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (message_id) DO NOTHING
                    """,
                    (message_id, conversation_id, role, token_count)
                )
            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error logging message: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._put_conn(conn)

    def mark_message_embedded(self, message_id: str) -> bool:
        """Mark a message as having embeddings generated"""
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE message_metadata
                    SET has_embeddings = TRUE,
                        processing_status = 'completed'
                    WHERE message_id = %s
                    """,
                    (message_id,)
                )
            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error marking message embedded: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._put_conn(conn)

    def log_query(
        self,
        query_text: str,
        filters: Optional[Dict] = None,
        results_count: int = 0,
        processing_time_ms: float = 0,
        queried_by: str = "system"
    ) -> bool:
        """
        Log a query for analytics

        Args:
            query_text: The query string
            filters: Optional filters applied
            results_count: Number of results returned
            processing_time_ms: Query processing time
            queried_by: Who/what made the query

        Returns:
            Success boolean
        """
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                import json
                cur.execute(
                    """
                    INSERT INTO query_logs
                        (query_text, filters, results_count, processing_time_ms, queried_by)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        query_text,
                        json.dumps(filters) if filters else None,
                        results_count,
                        processing_time_ms,
                        queried_by
                    )
                )
            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error logging query: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._put_conn(conn)

    def get_conversation_stats(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific conversation"""
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        c.conversation_id,
                        c.model_name,
                        c.total_messages,
                        c.total_tokens,
                        c.created_at,
                        c.updated_at,
                        COUNT(m.id) as metadata_entries,
                        SUM(CASE WHEN m.has_embeddings THEN 1 ELSE 0 END) as embedded_count
                    FROM conversations c
                    LEFT JOIN message_metadata m ON c.conversation_id = m.conversation_id
                    WHERE c.conversation_id = %s
                    GROUP BY c.id, c.conversation_id, c.model_name, c.total_messages,
                             c.total_tokens, c.created_at, c.updated_at
                    """,
                    (conversation_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None

        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return None
        finally:
            if conn:
                self._put_conn(conn)

    def get_global_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get conversation stats
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_conversations,
                        SUM(total_messages) as total_messages,
                        SUM(total_tokens) as total_tokens
                    FROM conversations
                    """
                )
                conv_stats = cur.fetchone()

                # Get message metadata stats
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_metadata_entries,
                        SUM(CASE WHEN has_embeddings THEN 1 ELSE 0 END) as embedded_messages,
                        SUM(token_count) as total_tokens_tracked
                    FROM message_metadata
                    """
                )
                msg_stats = cur.fetchone()

                # Get model distribution
                cur.execute(
                    """
                    SELECT model_name, COUNT(*) as count
                    FROM conversations
                    GROUP BY model_name
                    ORDER BY count DESC
                    """
                )
                model_dist = cur.fetchall()

                # Get query stats
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_queries,
                        AVG(processing_time_ms) as avg_processing_time,
                        AVG(results_count) as avg_results
                    FROM query_logs
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    """
                )
                query_stats = cur.fetchone()

                return {
                    "conversations": dict(conv_stats) if conv_stats else {},
                    "messages": dict(msg_stats) if msg_stats else {},
                    "models": [dict(row) for row in model_dist] if model_dist else [],
                    "queries_24h": dict(query_stats) if query_stats else {}
                }

        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {}
        finally:
            if conn:
                self._put_conn(conn)

    def get_models(self) -> List[Dict[str, Any]]:
        """Get all registered models"""
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM models
                    WHERE is_active = TRUE
                    ORDER BY model_name
                    """
                )
                results = cur.fetchall()
                return [dict(row) for row in results] if results else []

        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
        finally:
            if conn:
                self._put_conn(conn)

    def close(self):
        """Close all connections in the pool"""
        try:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL pool: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
