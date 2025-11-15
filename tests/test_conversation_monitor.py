"""
Tests for ConversationMonitor
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.capture.conversation_monitor import ConversationMonitor


@pytest.fixture
def mock_mongo():
    """Mock MongoDB client"""
    mongo = Mock()
    mongo.store_message = Mock(return_value="msg-123")
    return mongo


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    redis.push_to_queue = AsyncMock(return_value=1)
    return redis


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL client"""
    postgres = Mock()
    postgres.log_conversation = Mock(return_value=True)
    postgres.log_message = Mock(return_value=True)
    return postgres


@pytest.fixture
def monitor(mock_mongo, mock_redis, mock_postgres):
    """Create ConversationMonitor instance"""
    return ConversationMonitor(mock_mongo, mock_redis, mock_postgres)


@pytest.mark.asyncio
async def test_capture_message(monitor, mock_mongo, mock_redis):
    """Test capturing a single message"""
    message_id = await monitor.capture_message(
        content="Test message",
        role="user",
        model="claude-sonnet-4-5"
    )

    assert message_id is not None
    assert len(message_id) == 36  # UUID length
    assert monitor.stats["messages_captured"] == 1


@pytest.mark.asyncio
async def test_capture_conversation(monitor, mock_mongo, mock_redis):
    """Test capturing a full conversation"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    result = await monitor.capture_conversation(
        messages=messages,
        model="claude-sonnet-4-5"
    )

    assert "conversation_id" in result
    assert "message_ids" in result
    assert result["message_count"] == 2
    assert len(result["message_ids"]) == 2


@pytest.mark.asyncio
async def test_get_stats(monitor):
    """Test getting monitor statistics"""
    await monitor.capture_message(
        content="Test",
        role="user"
    )

    stats = monitor.get_stats()

    assert "messages_captured" in stats
    assert stats["messages_captured"] == 1
    assert "errors" in stats
