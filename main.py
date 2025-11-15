"""
AetherGrid - Collective AI Intelligence System
Main entry point for the application.
"""

import asyncio
import os
import sys
import signal
from dotenv import load_dotenv
import logging
import uvicorn
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aethergrid.log")
    ]
)

logger = logging.getLogger(__name__)

# Import components
from src.storage.weaviate_client import WeaviateManager
from src.storage.mongo_client import MongoManager
from src.storage.redis_client import RedisManager
from src.storage.postgres_client import PostgresManager
from src.capture.conversation_monitor import ConversationMonitor
from src.processing.embedding_worker import EmbeddingWorker
from src.adapters.claude_adapter import ClaudeAdapter
from src.api import server


class AetherGrid:
    """Main AetherGrid application orchestrator"""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.tasks = []

        # Clients
        self.weaviate = None
        self.mongo = None
        self.redis = None
        self.postgres = None

        # Services
        self.monitor = None
        self.worker = None
        self.claude_adapter = None

    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing AetherGrid...")

        try:
            # Initialize database clients
            logger.info("📊 Connecting to databases...")

            # Weaviate
            self.weaviate = WeaviateManager(
                url=os.getenv("WEAVIATE_URL"),
                vector_dimensions=int(os.getenv("VECTOR_DIMENSIONS", "768"))
            )

            # MongoDB
            self.mongo = MongoManager(
                url=os.getenv("MONGODB_URL")
            )

            # Redis
            self.redis = RedisManager(
                url=os.getenv("REDIS_URL")
            )
            await self.redis.connect()

            # PostgreSQL
            self.postgres = PostgresManager(
                url=os.getenv("POSTGRES_URL")
            )

            logger.info("✓ All databases connected")

            # Initialize services
            logger.info("🔧 Initializing services...")

            # Conversation Monitor
            self.monitor = ConversationMonitor(
                mongo_manager=self.mongo,
                redis_manager=self.redis,
                postgres_manager=self.postgres
            )

            # Embedding Worker
            self.worker = EmbeddingWorker(
                redis_manager=self.redis,
                weaviate_manager=self.weaviate,
                mongo_manager=self.mongo,
                postgres_manager=self.postgres,
                batch_size=int(os.getenv("PROCESSING_BATCH_SIZE", "50"))
            )

            # Claude Adapter
            self.claude_adapter = ClaudeAdapter(
                conversation_monitor=self.monitor,
                weaviate_manager=self.weaviate
            )

            # Set global references for API server
            server.weaviate_client = self.weaviate
            server.mongo_client = self.mongo
            server.redis_client = self.redis
            server.postgres_client = self.postgres
            server.conversation_monitor = self.monitor
            server.embedding_worker = self.worker
            server.claude_adapter = self.claude_adapter

            logger.info("✓ All services initialized")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    async def start_services(self):
        """Start background services"""
        logger.info("🟢 Starting background services...")

        # Create tasks for background services
        if os.getenv("CAPTURE_ENABLED", "true").lower() == "true":
            self.tasks.append(asyncio.create_task(self.monitor.start()))
            logger.info("  ✓ Conversation monitor started")

        self.tasks.append(asyncio.create_task(self.worker.start()))
        logger.info("  ✓ Embedding worker started")

        logger.info("✓ All background services running")

    async def shutdown(self):
        """Gracefully shutdown all services"""
        logger.info("👋 Shutting down AetherGrid...")

        # Stop services
        if self.monitor:
            await self.monitor.stop()

        if self.worker:
            await self.worker.stop()

        # Cancel tasks
        for task in self.tasks:
            task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Close database connections
        if self.weaviate:
            self.weaviate.close()

        if self.mongo:
            self.mongo.close()

        if self.redis:
            await self.redis.close()

        if self.postgres:
            self.postgres.close()

        logger.info("✓ Shutdown complete")

    async def run(self):
        """Run the main application"""
        try:
            # Initialize
            await self.initialize()

            # Start background services
            await self.start_services()

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise
        finally:
            await self.shutdown()


async def run_with_api():
    """Run AetherGrid with API server"""
    # Create AetherGrid instance
    grid = AetherGrid()

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        grid.shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize AetherGrid
    await grid.initialize()

    # Start background services
    await grid.start_services()

    # Get API configuration
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", 8100))

    # Create uvicorn server
    config = uvicorn.Config(
        server.app,
        host=api_host,
        port=api_port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )
    api_server = uvicorn.Server(config)

    logger.info(f"🌐 Starting API server on {api_host}:{api_port}")

    try:
        # Run API server
        await api_server.serve()
    except Exception as e:
        logger.error(f"API server error: {e}")
    finally:
        await grid.shutdown()


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("  AetherGrid - Collective AI Intelligence System")
    logger.info("  Version 1.0.0")
    logger.info("=" * 60)

    # No required API keys! AetherGrid works with local embeddings by default.
    # Optional: COHERE_API_KEY for Cohere embeddings
    # Optional: ANTHROPIC_API_KEY for Claude adapter features
    logger.info("🔓 No API keys required - running with local embeddings")

    # Run the application
    try:
        asyncio.run(run_with_api())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
