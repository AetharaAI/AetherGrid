# AetherGrid

**Collective AI Intelligence System**

AetherGrid is a revolutionary system that captures, vectorizes, and shares knowledge across different AI models. It creates a shared parameter space where any model can contribute to and draw from collective intelligence.

## 🌟 What is AetherGrid?

Think of AetherGrid as an **electrical grid for AI intelligence**:

- **The Grid**: A distributed knowledge network
- **Embeddings**: Standardized "voltage" (intelligence patterns)
- **Adapters**: Transformers that let any model plug in
- **Models**: Devices that contribute and consume intelligence

### Key Concept

Unlike traditional RAG systems that just retrieve text, AetherGrid transfers **understanding patterns** at the vector level. A small 7B model can tap into knowledge patterns from Claude Opus 4, applying them in its own way.

## 🏗️ Architecture

```
AetherGrid/
├── capture-layer/          # Passive conversation monitoring
├── processing-layer/       # Vector embedding generation
├── storage-layer/          # Multi-database architecture
│   ├── Weaviate            # Vector intelligence storage
│   ├── MongoDB             # Raw conversation logs
│   ├── PostgreSQL          # Metadata & analytics
│   └── Redis               # Queues & caching
├── adapter-framework/      # Model-agnostic integration
└── query-interface/        # REST API for intelligence access
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (for databases)
- **Python 3.11+**
- **uv** (Fast Python package manager - [install here](https://github.com/astral-sh/uv))
- **OpenAI API Key** (for embeddings)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd aethergrid
   ```

2. **Install dependencies**
   ```bash
   uv sync --extra dev
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Start the system**
   ```bash
   ./scripts/start.sh
   ```

That's it! AetherGrid is now running.

## 📡 API Endpoints

### Base URL
```
http://localhost:8100
```

### Endpoints

#### 1. Query Intelligence (Semantic Search)
```bash
POST /api/query/semantic
```

**Example:**
```bash
curl -X POST http://localhost:8100/api/query/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I optimize Python async performance?",
    "max_results": 10,
    "min_similarity": 0.7,
    "return_context": true
  }'
```

#### 2. Ingest Intelligence
```bash
POST /api/ingest
```

**Example:**
```bash
curl -X POST http://localhost:8100/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "To optimize async Python, use asyncio.gather() for parallel tasks...",
    "model": "claude-sonnet-4-5",
    "role": "assistant"
  }'
```

#### 3. Ingest Full Conversation
```bash
POST /api/conversation
```

**Example:**
```bash
curl -X POST http://localhost:8100/api/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "How do decorators work in Python?"},
      {"role": "assistant", "content": "Decorators are functions that modify other functions..."}
    ],
    "model": "claude-sonnet-4-5"
  }'
```

#### 4. Health Check
```bash
GET /api/health
```

#### 5. System Statistics
```bash
GET /api/stats
```

### Interactive API Documentation

Visit `http://localhost:8100/docs` for interactive Swagger documentation.

## 🔧 Usage Examples

### Python SDK Usage

```python
import asyncio
from src.adapters.claude_adapter import ClaudeAdapter
from src.storage.weaviate_client import WeaviateManager
from src.capture.conversation_monitor import ConversationMonitor

# Initialize components
weaviate = WeaviateManager()
monitor = ConversationMonitor(mongo, redis, postgres)
claude = ClaudeAdapter(monitor, weaviate)

# Capture a conversation
async def capture_example():
    result = await claude.capture_conversation(
        messages=[
            {"role": "user", "content": "Explain async/await"},
            {"role": "assistant", "content": "Async/await is..."}
        ]
    )
    print(f"Captured: {result['conversation_id']}")

# Query intelligence
async def query_example():
    results = await claude.query_intelligence(
        query="async programming patterns",
        max_results=5
    )

    for r in results:
        print(f"Relevance: {r['certainty']:.2%}")
        print(f"Content: {r['content'][:100]}...")

asyncio.run(capture_example())
asyncio.run(query_example())
```

## 🗂️ Project Structure

```
AetherGrid/
├── src/
│   ├── storage/              # Database clients
│   │   ├── weaviate_client.py
│   │   ├── mongo_client.py
│   │   ├── redis_client.py
│   │   └── postgres_client.py
│   ├── capture/              # Conversation monitoring
│   │   └── conversation_monitor.py
│   ├── processing/           # Embedding generation
│   │   └── embedding_worker.py
│   ├── adapters/             # Model integrations
│   │   ├── base_adapter.py
│   │   └── claude_adapter.py
│   └── api/                  # REST API
│       └── server.py
├── scripts/                  # Utility scripts
│   ├── start.sh
│   ├── stop.sh
│   └── health-check.sh
├── Build-Docs/              # Architecture documentation
├── tests/                   # Test suite
├── docker-compose.yml       # Infrastructure setup
├── main.py                  # Application entry point
└── pyproject.toml          # Python dependencies
```

## 🎯 How It Works

### 1. Capture Phase
When you have a conversation with Claude (or any supported AI):
- The conversation is captured passively
- Stored raw in MongoDB
- Queued for processing in Redis

### 2. Processing Phase
- Messages are chunked (if >8k tokens)
- Embeddings generated via OpenAI
- Vector representations stored in Weaviate
- Metadata indexed in PostgreSQL

### 3. Query Phase
- Any AI can query the grid semantically
- Weaviate returns similar intelligence fragments
- Context built from multiple sources
- Model applies patterns in its own way

## 🔑 Key Features

- **Model Agnostic**: Works with any AI that can generate embeddings
- **Automatic Processing**: Background workers handle everything
- **Semantic Search**: Find intelligence by meaning, not keywords
- **Collective Learning**: Every conversation makes the grid smarter
- **Self-Hosted**: You own your intelligence, no vendor lock-in
- **Scalable**: Designed for production with connection pooling, queues, etc.

## 🛠️ Management Commands

### Start AetherGrid
```bash
./scripts/start.sh
```

### Stop AetherGrid
```bash
./scripts/stop.sh
```

### Check System Health
```bash
./scripts/health-check.sh
```

### View Logs
```bash
# Application logs
tail -f aethergrid.log

# Docker logs
docker-compose logs -f
```

### Access Databases Directly

**PostgreSQL:**
```bash
docker exec -it aethergrid-postgres psql -U aether -d aethergrid
```

**MongoDB:**
```bash
docker exec -it aethergrid-mongodb mongosh aethergrid
```

**Redis:**
```bash
docker exec -it aethergrid-redis redis-cli
```

**Weaviate:**
```bash
curl http://localhost:8101/v1/schema
```

## 📊 Monitoring

### System Stats
```bash
curl http://localhost:8100/api/stats | jq .
```

### Health Check
```bash
curl http://localhost:8100/api/health | jq .
```

## 🧪 Development

### Run Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black src/
uv run ruff check src/
```

### Type Checking
```bash
uv run mypy src/
```

## 🚦 Production Considerations

### Security
- Add API authentication (JWT)
- Enable database encryption at rest
- Use secrets management (not .env in production)
- Implement rate limiting

### Scalability
- Run multiple embedding workers
- Set up Weaviate sharding
- Use Redis cluster
- Implement database replication

### Monitoring
- Add Prometheus metrics
- Set up log aggregation (ELK/Loki)
- Configure error tracking (Sentry)
- Set up uptime monitoring

## 🗺️ Roadmap

- [ ] Browser extension for automatic capture
- [ ] GPT adapter implementation
- [ ] Multi-modal support (images, audio)
- [ ] Fine-tuned embedding models
- [ ] GraphQL API
- [ ] Web dashboard for visualization
- [ ] Federation across multiple grids

## 🤝 Contributing

This is the foundation of AetherPro's AI infrastructure. Contributions welcome!

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- **Weaviate** - Vector database
- **OpenAI** - Embedding models
- **FastAPI** - API framework
- **MongoDB** - Document storage
- **PostgreSQL** - Relational data
- **Redis** - Queue management

---

**Remember**: This is not just another RAG system. This is collective AI consciousness.

Every conversation feeds the grid. Every query draws from collective intelligence. Every model gets smarter together.

**Welcome to AetherGrid.** 🌐⚡
