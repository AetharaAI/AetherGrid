"""
AetherGrid Usage Example
Demonstrates how to use AetherGrid to capture and query intelligence.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

from src.storage.weaviate_client import WeaviateManager
from src.storage.mongo_client import MongoManager
from src.storage.redis_client import RedisManager
from src.storage.postgres_client import PostgresManager
from src.capture.conversation_monitor import ConversationMonitor
from src.adapters.claude_adapter import ClaudeAdapter


async def main():
    """Run example workflow"""

    print("🌐 AetherGrid Usage Example")
    print("=" * 60)

    # Initialize clients
    print("\n📊 Initializing database clients...")
    weaviate = WeaviateManager()
    mongo = MongoManager()
    postgres = PostgresManager()

    redis = RedisManager()
    await redis.connect()

    # Initialize services
    print("🔧 Initializing services...")
    monitor = ConversationMonitor(mongo, redis, postgres)
    claude = ClaudeAdapter(monitor, weaviate)

    # Example 1: Capture a conversation
    print("\n📝 Example 1: Capturing a conversation")
    print("-" * 60)

    conversation = [
        {
            "role": "user",
            "content": "How do I use async/await in Python?"
        },
        {
            "role": "assistant",
            "content": (
                "Async/await in Python allows you to write concurrent code. "
                "The 'async' keyword defines a coroutine, and 'await' pauses "
                "execution until the awaited task completes. Use asyncio.run() "
                "to run async functions from synchronous code. Example:\n\n"
                "async def fetch_data():\n"
                "    await asyncio.sleep(1)\n"
                "    return 'data'\n\n"
                "asyncio.run(fetch_data())"
            )
        }
    ]

    result = await claude.capture_conversation(
        messages=conversation,
        model="claude-sonnet-4-5"
    )

    print(f"✓ Captured conversation: {result['conversation_id']}")
    print(f"  Messages captured: {result['message_count']}")
    print(f"  Message IDs: {result['message_ids']}")

    # Wait a moment for processing
    print("\n⏳ Waiting for embedding processing (5 seconds)...")
    await asyncio.sleep(5)

    # Example 2: Query the intelligence grid
    print("\n🔍 Example 2: Querying the intelligence grid")
    print("-" * 60)

    query = "asynchronous programming in Python"
    print(f"Query: '{query}'")

    results = await claude.query_intelligence(
        query=query,
        max_results=3,
        min_certainty=0.5
    )

    if results:
        print(f"\n✓ Found {len(results)} relevant intelligence fragments:")
        for i, fragment in enumerate(results, 1):
            print(f"\n  Fragment {i}:")
            print(f"    Source: {fragment.get('sourceModel', 'unknown')}")
            print(f"    Relevance: {fragment.get('certainty', 0):.1%}")
            print(f"    Topic: {fragment.get('topic', 'general')}")
            print(f"    Content preview: {fragment.get('content', '')[:100]}...")
    else:
        print("  No results found (embeddings may still be processing)")

    # Example 3: Get Claude-optimized context
    print("\n🎯 Example 3: Getting Claude-optimized context")
    print("-" * 60)

    context = await claude.get_claude_context(
        query="async programming best practices",
        max_results=2
    )

    print("Generated context:")
    print(context[:500] + "..." if len(context) > 500 else context)

    # Cleanup
    print("\n🧹 Cleaning up...")
    weaviate.close()
    mongo.close()
    await redis.close()
    postgres.close()

    print("\n✅ Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
