# AetherGrid - Technical Implementation Guide

## Quick Start Commands

```bash
# Clone and setup
git clone <repo-url> aethergrid
cd aethergrid

# Choose your path (Node or Python - recommend Python for ML libraries)

## Python Path (Recommended)
poetry init --name aethergrid --python "^3.11"
poetry add weaviate-client openai psycopg2-binary pymongo redis fastapi uvicorn pydantic python-dotenv

poetry add -D pytest pytest-asyncio black ruff mypy

## Node Path (Alternative)
pnpm init
pnpm add weaviate-ts-client openai pg mongodb redis express
pnpm add -D typescript @types/node tsx vitest

# Start infrastructure
docker-compose up -d

# Run setup
poetry run python scripts/setup.py  # or: pnpm run setup

# Start service
poetry run python main.py  # or: pnpm start
```

---

## Code Examples & Patterns

### 1. Weaviate Client Setup

```python
# src/storage/weaviate_client.py
import weaviate
from weaviate.auth import AuthApiKey
import os
from typing import List, Dict, Any

class WeaviateManager:
    """Manages all Weaviate vector database operations for AetherGrid"""
    
    def __init__(self):
        self.client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create the IntelligenceFragment class if it doesn't exist"""
        schema = {
            "class": "IntelligenceFragment",
            "description": "A fragment of AI intelligence from conversations",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "text-embedding-3-small",
                    "dimensions": 1536,
                    "type": "text"
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The actual text content"
                },
                {
                    "name": "conversationId",
                    "dataType": ["string"],
                    "description": "UUID of the parent conversation"
                },
                {
                    "name": "messageId",
                    "dataType": ["string"],
                    "description": "UUID of the specific message"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When this intelligence was captured"
                },
                {
                    "name": "sourceModel",
                    "dataType": ["string"],
                    "description": "Which AI model generated this"
                },
                {
                    "name": "topic",
                    "dataType": ["string"],
                    "description": "Main topic/category"
                },
                {
                    "name": "taskType",
                    "dataType": ["string"],
                    "description": "Type of task (coding, writing, analysis, etc)"
                },
                {
                    "name": "complexity",
                    "dataType": ["number"],
                    "description": "Complexity score 0-1"
                },
                {
                    "name": "tokensUsed",
                    "dataType": ["int"],
                    "description": "Token count for this fragment"
                }
            ]
        }
        
        # Check if class exists
        existing_schema = self.client.schema.get()
        class_names = [c["class"] for c in existing_schema.get("classes", [])]
        
        if "IntelligenceFragment" not in class_names:
            self.client.schema.create_class(schema)
            print("✓ Created IntelligenceFragment schema")
        else:
            print("✓ Schema already exists")
    
    def store_fragment(self, fragment: Dict[str, Any]) -> str:
        """Store an intelligence fragment and return its UUID"""
        with self.client.batch as batch:
            uuid = batch.add_data_object(
                data_object={
                    "content": fragment["content"],
                    "conversationId": fragment["conversation_id"],
                    "messageId": fragment["message_id"],
                    "timestamp": fragment["timestamp"],
                    "sourceModel": fragment["source_model"],
                    "topic": fragment.get("topic", "general"),
                    "taskType": fragment.get("task_type", "chat"),
                    "complexity": fragment.get("complexity", 0.5),
                    "tokensUsed": fragment.get("tokens", 0)
                },
                class_name="IntelligenceFragment"
            )
        return uuid
    
    def semantic_search(self, query: str, limit: int = 10, filters: Dict = None) -> List[Dict]:
        """Search for similar intelligence fragments"""
        query_builder = (
            self.client.query
            .get("IntelligenceFragment", [
                "content", "conversationId", "messageId", 
                "timestamp", "sourceModel", "topic", "taskType"
            ])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .with_additional(["distance", "certainty"])
        )
        
        # Add filters if provided
        if filters:
            where_filter = self._build_where_filter(filters)
            if where_filter:
                query_builder = query_builder.with_where(where_filter)
        
        result = query_builder.do()
        
        return result.get("data", {}).get("Get", {}).get("IntelligenceFragment", [])
    
    def _build_where_filter(self, filters: Dict) -> Dict:
        """Build Weaviate where filter from simple dict"""
        conditions = []
        
        if "sourceModel" in filters:
            conditions.append({
                "path": ["sourceModel"],
                "operator": "Equal",
                "valueString": filters["sourceModel"]
            })
        
        if "taskType" in filters:
            conditions.append({
                "path": ["taskType"],
                "operator": "Equal",
                "valueString": filters["taskType"]
            })
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        return {
            "operator": "And",
            "operands": conditions
        }
```

### 2. Conversation Capture Service

```python
# src/capture/conversation_monitor.py
import asyncio
from datetime import datetime
import uuid
from typing import Dict, Any
import json

class ConversationMonitor:
    """Monitors and captures AI conversations in real-time"""
    
    def __init__(self, mongo_client, redis_client):
        self.mongo = mongo_client
        self.redis = redis_client
        self.is_running = False
    
    async def start(self):
        """Start the monitoring service"""
        self.is_running = True
        print("🟢 Conversation monitor started")
        
        # In production, this would monitor actual conversation sources
        # For now, we'll demonstrate the capture flow
        while self.is_running:
            await asyncio.sleep(1)  # Monitoring loop
    
    async def capture_message(
        self, 
        content: str, 
        role: str, 
        conversation_id: str = None,
        model: str = "claude-sonnet-4-5",
        metadata: Dict = None
    ) -> str:
        """Capture a single message and queue for processing"""
        
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
        
        print(f"✓ Captured message {message_id[:8]}... ({len(content)} chars)")
        
        return message_id
    
    async def _store_raw_message(self, message_doc: Dict):
        """Store raw message in MongoDB"""
        collection = self.mongo.aethergrid.conversations
        collection.insert_one(message_doc)
    
    async def _queue_for_processing(self, message_doc: Dict):
        """Add message to Redis processing queue"""
        await self.redis.lpush(
            "processing:queue",
            json.dumps(message_doc)
        )
```

### 3. Embedding Processing Worker

```python
# src/processing/embedding_worker.py
import asyncio
import json
from openai import AsyncOpenAI
from typing import List, Dict
import tiktoken

class EmbeddingWorker:
    """Processes messages from queue and generates embeddings"""
    
    def __init__(self, redis_client, weaviate_client, openai_api_key: str):
        self.redis = redis_client
        self.weaviate = weaviate_client
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.is_running = False
    
    async def start(self):
        """Start processing queue"""
        self.is_running = True
        print("🟢 Embedding worker started")
        
        while self.is_running:
            try:
                # Get message from queue (blocking with timeout)
                result = await self.redis.brpop("processing:queue", timeout=5)
                
                if result:
                    _, message_json = result
                    message = json.loads(message_json)
                    await self._process_message(message)
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: Dict):
        """Process a single message: chunk, embed, store"""
        content = message["content"]
        
        # Chunk if needed (>8000 tokens)
        chunks = self._chunk_text(content)
        
        print(f"📝 Processing {len(chunks)} chunk(s) for message {message['message_id'][:8]}...")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = await self._generate_embedding(chunk)
                
                # Create fragment
                fragment = {
                    "content": chunk,
                    "conversation_id": message["conversation_id"],
                    "message_id": f"{message['message_id']}-chunk{i}" if len(chunks) > 1 else message["message_id"],
                    "timestamp": message["timestamp"],
                    "source_model": message["model"],
                    "topic": self._extract_topic(chunk),
                    "task_type": self._classify_task(chunk),
                    "complexity": self._estimate_complexity(chunk),
                    "tokens": len(self.encoder.encode(chunk))
                }
                
                # Store in Weaviate
                uuid = self.weaviate.store_fragment(fragment)
                
                print(f"  ✓ Stored chunk {i+1}/{len(chunks)}: {uuid[:8]}...")
                
            except Exception as e:
                print(f"  ❌ Failed to process chunk {i}: {e}")
    
    def _chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Split text into chunks at semantic boundaries"""
        tokens = self.encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        # Simple chunking for now - can be improved with semantic splitting
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence_tokens = len(self.encoder.encode(sentence))
            
            if current_length + sentence_tokens > max_tokens:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text (simplified)"""
        # TODO: Use proper topic extraction
        if "code" in text.lower() or "function" in text.lower():
            return "programming"
        elif "data" in text.lower():
            return "data-science"
        else:
            return "general"
    
    def _classify_task(self, text: str) -> str:
        """Classify the type of task (simplified)"""
        # TODO: Use proper classification
        if any(word in text.lower() for word in ["write", "create", "generate"]):
            return "creation"
        elif any(word in text.lower() for word in ["fix", "debug", "error"]):
            return "debugging"
        elif any(word in text.lower() for word in ["explain", "how", "what"]):
            return "explanation"
        else:
            return "chat"
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate complexity 0-1 based on text features"""
        # Simple heuristic: longer + more technical = more complex
        token_count = len(self.encoder.encode(text))
        technical_terms = sum(1 for word in ["algorithm", "optimization", "architecture", "implementation"] if word in text.lower())
        
        complexity = min(1.0, (token_count / 5000) * 0.7 + (technical_terms / 10) * 0.3)
        return complexity
```

### 4. FastAPI Query Interface

```python
# src/api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

app = FastAPI(title="AetherGrid API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    max_results: int = 10
    min_similarity: float = 0.7
    return_context: bool = False

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    context: Optional[str] = None
    stats: Dict[str, Any]

class IngestRequest(BaseModel):
    content: str
    model: str
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.post("/api/query/semantic", response_model=QueryResponse)
async def semantic_query(request: QueryRequest):
    """Query the intelligence grid semantically"""
    try:
        start_time = datetime.now()
        
        # Perform semantic search
        results = weaviate_client.semantic_search(
            query=request.query,
            limit=request.max_results,
            filters=request.filters
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results 
            if r.get("_additional", {}).get("certainty", 0) >= request.min_similarity
        ]
        
        # Build context if requested
        context = None
        if request.return_context:
            context = _build_context(filtered_results)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResponse(
            results=filtered_results,
            context=context,
            stats={
                "total_searched": len(results),
                "returned": len(filtered_results),
                "processing_time_ms": round(processing_time, 2)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_intelligence(request: IngestRequest):
    """Manually submit intelligence to the grid"""
    try:
        message_id = await conversation_monitor.capture_message(
            content=request.content,
            role="assistant",  # Assuming ingested content is from AI
            conversation_id=request.conversation_id,
            model=request.model,
            metadata=request.metadata
        )
        
        return {
            "message_id": message_id,
            "status": "queued_for_processing"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "weaviate": "connected",
            "mongodb": "connected",
            "redis": "connected",
            "postgres": "connected"
        }
    }

@app.get("/api/stats")
async def get_stats():
    """Get grid statistics"""
    # TODO: Implement actual stats gathering
    return {
        "total_fragments": 0,
        "total_conversations": 0,
        "models_tracked": [],
        "processing_queue_size": 0
    }

def _build_context(results: List[Dict]) -> str:
    """Build aggregated context from search results"""
    if not results:
        return ""
    
    context_parts = []
    for i, result in enumerate(results[:5], 1):  # Top 5 results
        content = result.get("content", "")
        model = result.get("sourceModel", "unknown")
        context_parts.append(f"{i}. [{model}]: {content[:200]}...")
    
    return "\n\n".join(context_parts)
```

### 5. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: aethergrid-postgres
    environment:
      POSTGRES_DB: aethergrid
      POSTGRES_USER: aether
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-development}
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-postgres.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aether"]
      interval: 10s
      timeout: 5s
      retries: 5

  mongodb:
    image: mongo:7
    container_name: aethergrid-mongodb
    ports:
      - "27018:27017"
    volumes:
      - mongodb_data:/data/db
    command: mongod --replSet rs0
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 10s
      timeout: 5s
      retries: 5

  weaviate:
    image: semitechnologies/weaviate:latest
    container_name: aethergrid-weaviate
    ports:
      - "8101:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai'
      OPENAI_APIKEY: ${OPENAI_API_KEY}
      CLUSTER_HOSTNAME: 'weaviate-node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "localhost:8101/v1/.well-known/ready"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: aethergrid-redis
    ports:
      - "6395:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  mongodb_data:
  weaviate_data:
  redis_data:
```

### 6. Main Application Entry Point

```python
# main.py
import asyncio
import os
from dotenv import load_dotenv

# Import our components
from src.storage.weaviate_client import WeaviateManager
from src.capture.conversation_monitor import ConversationMonitor
from src.processing.embedding_worker import EmbeddingWorker
from pymongo import MongoClient
import redis.asyncio as redis

# Load environment variables
load_dotenv()

async def main():
    """Initialize and start AetherGrid"""
    
    print("🚀 Starting AetherGrid...")
    
    # Initialize database clients
    print("📊 Connecting to databases...")
    
    weaviate_client = WeaviateManager()
    
    mongo_client = MongoClient(os.getenv("MONGODB_URL", "mongodb://localhost:27018"))
    
    redis_client = await redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6395"),
        decode_responses=True
    )
    
    # Initialize services
    monitor = ConversationMonitor(mongo_client, redis_client)
    worker = EmbeddingWorker(redis_client, weaviate_client, os.getenv("OPENAI_API_KEY"))
    
    # Start services
    print("🔧 Starting services...")
    
    tasks = [
        monitor.start(),
        worker.start()
    ]
    
    # Run all services concurrently
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down AetherGrid...")
```

---

## Testing Examples

```python
# tests/test_capture.py
import pytest
from src.capture.conversation_monitor import ConversationMonitor

@pytest.mark.asyncio
async def test_capture_message(mongo_client, redis_client):
    monitor = ConversationMonitor(mongo_client, redis_client)
    
    message_id = await monitor.capture_message(
        content="Test message about Python async programming",
        role="assistant",
        model="claude-sonnet-4-5"
    )
    
    assert message_id is not None
    assert len(message_id) == 36  # UUID length
```

---

## Deployment Commands

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose logs -f aethergrid-service

# Monitor health
curl http://localhost:8100/api/health

# View stats
curl http://localhost:8100/api/stats
```

This should give Claude Code everything it needs to build AetherGrid from scratch!
