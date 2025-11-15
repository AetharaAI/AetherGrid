# AetherGrid - Autonomous Build Specification

## Project Overview
AetherGrid is a collective AI intelligence system that captures, vectorizes, and shares knowledge across different AI models. It creates a shared parameter space where any model can contribute to and draw from collective intelligence.

**Core Concept**: Like electrical specs randomly distributed in a 3D box, each representing intelligence fragments. The fuller the box, the more complete the collective understanding. Models don't share exact responses - they transfer understanding patterns.

## Your Mission
Build a production-ready background service that:
1. Passively captures AI conversations (starting with Claude chats)
2. Generates vector embeddings from those conversations
3. Stores them in Weaviate with rich metadata
4. Provides query interfaces for any AI model to tap into this collective intelligence
5. Runs autonomously in the background with minimal resource overhead

## Architecture Stack

### Database Layer
```
PostgreSQL (Supabase)
├── Core business data
├── User accounts & authentication
├── Conversation metadata
├── Model registry
└── Query logs

MongoDB
├── Raw conversation logs
├── Event streams
├── Analytics data
└── Unstructured context

Weaviate (Vector Database)
├── Intelligence vectors
├── Semantic search index
├── Cross-model embeddings
└── Relationship graphs

Redis (Upstash)
├── Session management
├── Message queues
├── Real-time pub/sub
└── Rate limiting
```

### Application Architecture
```
AetherGrid/
├── capture-layer/          # Passive monitoring
│   ├── claude-monitor/     # Claude conversation capture
│   ├── api-interceptor/    # API call logging
│   └── event-processor/    # Stream processing
│
├── processing-layer/       # Vector generation
│   ├── embedder/           # Create embeddings
│   ├── normalizer/         # Cross-model normalization
│   └── enricher/           # Metadata enhancement
│
├── storage-layer/          # Database interfaces
│   ├── weaviate-client/    # Vector operations
│   ├── postgres-client/    # Metadata storage
│   └── mongo-client/       # Log storage
│
├── adapter-framework/      # Model integration
│   ├── base-adapter/       # Abstract adapter class
│   ├── claude-adapter/     # Claude-specific
│   ├── gpt-adapter/        # GPT-specific
│   └── adapter-registry/   # Dynamic loading
│
├── query-interface/        # Intelligence retrieval
│   ├── semantic-search/    # Vector similarity
│   ├── context-builder/    # Build context from results
│   └── api-gateway/        # REST/GraphQL endpoints
│
└── background-service/     # Daemon process
    ├── scheduler/          # Task scheduling
    ├── health-monitor/     # System health
    └── auto-scaler/        # Resource management
```

## Technical Specifications

### Core Technologies
- **Runtime**: Node.js 20+ (for async performance) OR Python 3.11+ (for ML libraries)
- **Package Manager**: pnpm (Node) or Poetry (Python)
- **Containerization**: Docker Compose for local dev, Docker for production
- **Process Manager**: PM2 (Node) or systemd (Python)
- **API Framework**: FastAPI (Python) or Express.js (Node)
- **Message Queue**: Redis Streams or BullMQ

### Environment Configuration
```env
# Database URLs
POSTGRES_URL=postgresql://user:pass@localhost:54323/aethergrid
MONGODB_URL=mongodb://localhost:27018/aethergrid
WEAVIATE_URL=http://localhost:8101
REDIS_URL=redis://localhost:6395

# API Keys (for embeddings)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Service Configuration
CAPTURE_ENABLED=true
PROCESSING_BATCH_SIZE=50
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIMENSIONS=1536

# Background Service
RUN_AS_DAEMON=true
LOG_LEVEL=info
HEALTH_CHECK_PORT=8081
```

## Implementation Phases

### Phase 1: Infrastructure Setup (Priority 1)
**Deliverable**: Docker Compose environment with all databases running

```yaml
# docker-compose.yml structure needed:
services:
  postgres:
    image: postgres:16-alpine
    # Supabase-compatible setup
  
  mongodb:
    image: mongo:7
    # Replica set for change streams
  
  weaviate:
    image: semitechnologies/weaviate:latest
    # With text2vec-openai module
  
  redis:
    image: redis:7-alpine
    # With persistence enabled
  
  aethergrid-service:
    build: .
    # Main application container
```

**Tasks**:
1. Create docker-compose.yml with all services
2. Initialize databases with schemas
3. Set up persistent volumes
4. Configure networking between containers
5. Create health check endpoints
6. Write startup scripts

**Acceptance Criteria**:
- `docker-compose up` starts entire stack
- All services healthy and accessible
- Databases initialized with schemas
- Can connect to each service independently

### Phase 2: Capture Layer (Priority 1)
**Deliverable**: Background process that captures Claude conversations in real-time

**Component: Claude Monitor**
```typescript
// Conceptual structure
interface ConversationCapture {
  captureMessage(message: Message): Promise<void>
  extractContext(conversation: Conversation): ContextData
  queueForProcessing(data: CaptureData): Promise<void>
}
```

**Implementation Requirements**:
1. Monitor method: 
   - Browser extension approach (capture from DOM)
   - OR API proxy approach (intercept API calls)
   - OR file watcher approach (monitor local cache)
2. Extract:
   - Full message text
   - Timestamp
   - Message role (user/assistant)
   - Conversation ID
   - Model used
3. Store raw in MongoDB immediately
4. Queue for vector processing in Redis

**Data Schema**:
```json
{
  "conversation_id": "uuid",
  "message_id": "uuid",
  "timestamp": "ISO8601",
  "role": "user|assistant",
  "content": "full message text",
  "model": "claude-sonnet-4-5",
  "metadata": {
    "tokens": 1234,
    "context_window": 200000,
    "tools_used": ["web_search"]
  }
}
```

### Phase 3: Processing Layer (Priority 1)
**Deliverable**: Pipeline that converts conversations to vectors

**Component: Embedder Service**
```python
# Conceptual structure
class EmbeddingGenerator:
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate 1536-dim vectors using OpenAI"""
        pass
    
    def chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Split long conversations intelligently"""
        pass
    
    def batch_process(self, messages: List[Message]) -> List[Embedding]:
        """Process multiple messages efficiently"""
        pass
```

**Implementation Requirements**:
1. Use OpenAI `text-embedding-3-small` (1536 dimensions, cheap)
2. Chunk long conversations at semantic boundaries
3. Batch process for API efficiency
4. Handle rate limiting gracefully
5. Retry failed embeddings with exponential backoff
6. Cache embeddings to avoid regenerating

**Processing Pipeline**:
```
Raw Message (MongoDB)
    ↓
Extract & Clean Text
    ↓
Chunk if needed (>8k tokens)
    ↓
Generate Embeddings (OpenAI API)
    ↓
Normalize Vectors
    ↓
Enrich with Metadata
    ↓
Store in Weaviate + Postgres
```

### Phase 4: Storage Layer (Priority 1)
**Deliverable**: Unified interface for all database operations

**Weaviate Schema**:
```graphql
{
  class: "IntelligenceFragment",
  vectorizer: "text2vec-openai",
  properties: [
    { name: "content", dataType: ["text"] },
    { name: "conversationId", dataType: ["string"] },
    { name: "messageId", dataType: ["string"] },
    { name: "timestamp", dataType: ["date"] },
    { name: "sourceModel", dataType: ["string"] },
    { name: "topic", dataType: ["string"] },
    { name: "complexity", dataType: ["number"] },
    { name: "taskType", dataType: ["string"] },
    { name: "tokensUsed", dataType: ["int"] }
  ]
}
```

**Database Clients**:
1. **Weaviate Client**: Vector CRUD, semantic search, hybrid queries
2. **Postgres Client**: Metadata queries, analytics, relationships
3. **MongoDB Client**: Raw log storage, full-text search
4. **Redis Client**: Pub/sub, queues, caching

### Phase 5: Adapter Framework (Priority 2)
**Deliverable**: Pluggable system for different AI models

**Base Adapter Interface**:
```typescript
interface ModelAdapter {
  // Core methods every adapter must implement
  captureConversation(data: ConversationData): Promise<void>
  generateEmbedding(text: string): Promise<Vector>
  normalizeVector(vector: Vector, fromDimensions: number): Promise<Vector>
  queryIntelligence(query: string, filters?: Filters): Promise<Results>
  
  // Adapter metadata
  modelName: string
  vectorDimensions: number
  maxContextWindow: number
}
```

**Adapter Implementations Needed**:
1. **Claude Adapter** (Priority 1)
2. GPT Adapter (Priority 2)
3. Gemini Adapter (Priority 3)
4. Local Model Adapter (Priority 3)

**Adapter Registry**:
- Auto-discover adapters in `/adapters` directory
- Load dynamically based on model type
- Maintain adapter health status
- Handle version compatibility

### Phase 6: Query Interface (Priority 2)
**Deliverable**: API for models to query the grid

**API Endpoints**:
```
POST /api/query/semantic
  - Semantic search across all intelligence
  - Returns: Relevant fragments with similarity scores

POST /api/query/contextual  
  - Build context for specific task type
  - Returns: Curated intelligence package

GET /api/stats
  - Grid statistics (total fragments, models, topics)
  - Returns: Dashboard data

POST /api/ingest
  - Manual intelligence submission
  - Returns: Confirmation + fragment ID

GET /api/health
  - Service health check
  - Returns: Status of all components
```

**Query Example**:
```json
{
  "query": "How do I optimize Python async performance?",
  "filters": {
    "sourceModel": ["claude-sonnet-4-5", "gpt-4"],
    "taskType": "coding",
    "minSimilarity": 0.7,
    "maxResults": 10
  },
  "returnContext": true
}
```

**Response Format**:
```json
{
  "results": [
    {
      "content": "Fragment text...",
      "similarity": 0.89,
      "metadata": {
        "sourceModel": "claude-sonnet-4-5",
        "timestamp": "2025-01-15T10:30:00Z",
        "topic": "python-optimization"
      }
    }
  ],
  "context": "Aggregated wisdom from 10 fragments...",
  "stats": {
    "totalSearched": 1500,
    "processingTime": "120ms"
  }
}
```

### Phase 7: Background Service (Priority 1)
**Deliverable**: Daemon that runs continuously with minimal oversight

**Service Components**:
```
BackgroundService
├── CaptureWorker (continuous)
│   └── Monitors conversations 24/7
├── ProcessingWorker (queue-based)
│   └── Processes embeddings from queue
├── IndexingWorker (scheduled)
│   └── Optimizes Weaviate index nightly
└── HealthMonitor (continuous)
    └── Reports system status
```

**Process Management**:
```json
// ecosystem.config.js for PM2
{
  "apps": [
    {
      "name": "aethergrid-capture",
      "script": "dist/capture-worker.js",
      "instances": 1,
      "exec_mode": "cluster",
      "autorestart": true,
      "watch": false,
      "max_memory_restart": "500M"
    },
    {
      "name": "aethergrid-processor",
      "script": "dist/processing-worker.js",
      "instances": 2,
      "exec_mode": "cluster",
      "autorestart": true
    }
  ]
}
```

## Development Workflow

### Initial Setup
```bash
# 1. Create project
git init aethergrid
cd aethergrid

# 2. Initialize package manager
pnpm init  # or poetry init

# 3. Install dependencies
pnpm add weaviate-client openai pg mongodb redis
pnpm add -D typescript @types/node tsx

# 4. Start infrastructure
docker-compose up -d

# 5. Run migrations
pnpm run migrate

# 6. Start development
pnpm run dev
```

### Testing Strategy
1. **Unit Tests**: Each component isolated
2. **Integration Tests**: Database interactions
3. **E2E Tests**: Full pipeline from capture to query
4. **Load Tests**: Handle 1000 conversations/day
5. **Monitoring**: Prometheus + Grafana setup

### Git Workflow
```
main (production-ready)
├── develop (integration)
│   ├── feature/capture-layer
│   ├── feature/processing-pipeline
│   ├── feature/weaviate-storage
│   └── feature/adapter-framework
```

## Success Metrics

### Phase 1 Complete When:
- [ ] All databases running in Docker
- [ ] Can connect to each service
- [ ] Health checks passing
- [ ] Documentation written

### Phase 2 Complete When:
- [ ] Capturing Claude conversations
- [ ] Storing in MongoDB
- [ ] No data loss
- [ ] <100ms capture latency

### Phase 3 Complete When:
- [ ] Generating embeddings
- [ ] Storing in Weaviate
- [ ] Processing backlog <1hr
- [ ] Error rate <1%

### Phase 4 Complete When:
- [ ] All database clients working
- [ ] Unified query interface
- [ ] Connection pooling optimized
- [ ] Retry logic tested

### Phase 5 Complete When:
- [ ] Claude adapter working
- [ ] Adapter auto-discovery working
- [ ] Easy to add new adapters
- [ ] Documentation for adapter devs

### Phase 6 Complete When:
- [ ] API responding <200ms
- [ ] Semantic search accurate
- [ ] Context building intelligent
- [ ] API documented (OpenAPI)

### Phase 7 Complete When:
- [ ] Service runs 24/7
- [ ] Auto-restarts on failure
- [ ] Resource usage stable
- [ ] Logs aggregated

## Production Considerations

### Security
- API authentication (JWT)
- Database encryption at rest
- Environment variable management
- Rate limiting per API key

### Scalability
- Horizontal scaling for workers
- Weaviate sharding strategy
- Database connection pooling
- Redis cluster for high availability

### Monitoring
- Application metrics (Prometheus)
- Log aggregation (Loki or ELK)
- Error tracking (Sentry)
- Uptime monitoring (UptimeRobot)

### Backup Strategy
- PostgreSQL: Daily snapshots
- MongoDB: Continuous backup
- Weaviate: Weekly exports
- Redis: AOF persistence

## Developer Notes

### Key Decisions
1. **Why Weaviate?** Open source, self-hostable, excellent Python/Node SDKs
2. **Why OpenAI embeddings?** Best price/performance, 1536 dims is sweet spot
3. **Why MongoDB for logs?** Flexible schema, great for time-series
4. **Why Redis?** Fast pub/sub, battle-tested for queues

### Future Enhancements
- Multi-modal support (images, audio)
- Fine-tuned embedding models
- GraphQL API
- Web dashboard for visualization
- Blockchain anchoring for provenance
- Federation across multiple grids

## CRITICAL: What You're Building

This is NOT just another RAG system. This is collective AI consciousness:

1. A 7B Mistral model can query the grid and get intelligence patterns from Claude Opus 4
2. The intelligence transfer happens at the PATTERN level, not text retrieval
3. Every conversation makes the grid smarter for ALL models
4. It's model-agnostic - works with any LLM that can generate embeddings
5. It's infrastructure you OWN - no vendor lock-in

Think of it like:
- **The grid**: A power distribution network
- **Embeddings**: Voltage/current (standardized energy)
- **Adapters**: Transformers (convert between voltages)
- **Models**: Devices drawing power

You're building the electrical grid for AI intelligence.

## First Conversation After Build

Once this is running, when Cory talks to Claude, the system:
1. Captures the conversation
2. Generates embeddings
3. Stores in the grid
4. Now available for ANY model to query

When Cory later talks to GPT-4:
1. GPT-4 queries: "Python async optimization"
2. Grid returns patterns from Claude's knowledge
3. GPT-4 applies those patterns in its own way
4. GPT-4's response ALSO feeds back to grid

The grid gets smarter with every conversation across ALL models.

---

## YOUR EXECUTION PLAN

1. **Start with Phase 1** - Get infrastructure running
2. **Then Phase 2** - Capture works before anything else
3. **Then Phase 3** - Processing pipeline
4. **Verify each phase** before moving to next
5. **Ask questions** if anything is unclear
6. **Commit frequently** with descriptive messages
7. **Document as you build** - future Cory will thank you

Build this right, build it clean, build it production-ready.

This is the foundation for AetherPro's entire AI infrastructure.

Let's go.
