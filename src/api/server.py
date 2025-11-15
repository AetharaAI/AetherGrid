"""
FastAPI Server for AetherGrid
Provides REST API endpoints for querying and ingesting intelligence.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Global references (will be set by main.py)
weaviate_client = None
mongo_client = None
redis_client = None
postgres_client = None
conversation_monitor = None
embedding_worker = None
claude_adapter = None


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity")
    return_context: bool = Field(False, description="Build aggregated context")


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    context: Optional[str] = None
    stats: Dict[str, Any]


class IngestRequest(BaseModel):
    content: str = Field(..., description="Content to ingest")
    model: str = Field("unknown", description="Source model name")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    role: str = Field("assistant", description="Message role")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class IngestResponse(BaseModel):
    message_id: str
    status: str
    queued_at: str


class ConversationRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of messages")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    model: str = Field("claude-sonnet-4-5", description="Model name")


class ConversationResponse(BaseModel):
    conversation_id: str
    message_ids: List[str]
    message_count: int
    status: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    weaviate: Dict[str, Any]
    mongodb: Dict[str, Any]
    redis: Dict[str, Any]
    postgres: Dict[str, Any]
    capture: Dict[str, Any]
    processing: Dict[str, Any]


# Create FastAPI app
app = FastAPI(
    title="AetherGrid API",
    description="Collective AI Intelligence Grid - Query and ingest intelligence fragments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "AetherGrid",
        "version": "1.0.0",
        "description": "Collective AI Intelligence Grid",
        "docs": "/docs"
    }


@app.post("/api/query/semantic", response_model=QueryResponse, tags=["Query"])
async def semantic_query(request: QueryRequest):
    """
    Perform semantic search across the intelligence grid

    Returns relevant intelligence fragments based on vector similarity.
    """
    try:
        start_time = datetime.now()

        if not weaviate_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Weaviate client not initialized"
            )

        # Perform semantic search
        results = weaviate_client.semantic_search(
            query=request.query,
            limit=request.max_results,
            filters=request.filters,
            min_certainty=request.min_similarity
        )

        # Build context if requested
        context = None
        if request.return_context and results:
            context = _build_context(results)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log query to PostgreSQL
        if postgres_client:
            postgres_client.log_query(
                query_text=request.query,
                filters=request.filters,
                results_count=len(results),
                processing_time_ms=processing_time,
                queried_by="api"
            )

        return QueryResponse(
            results=results,
            context=context,
            stats={
                "returned": len(results),
                "processing_time_ms": round(processing_time, 2),
                "query": request.query
            }
        )

    except Exception as e:
        logger.error(f"Error in semantic query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest_intelligence(request: IngestRequest):
    """
    Manually ingest intelligence into the grid

    Accepts a piece of content and queues it for embedding and storage.
    """
    try:
        if not conversation_monitor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation monitor not initialized"
            )

        # Capture the message
        message_id = await conversation_monitor.capture_message(
            content=request.content,
            role=request.role,
            conversation_id=request.conversation_id,
            model=request.model,
            metadata=request.metadata
        )

        return IngestResponse(
            message_id=message_id,
            status="queued_for_processing",
            queued_at=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error ingesting intelligence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/conversation", response_model=ConversationResponse, tags=["Ingest"])
async def ingest_conversation(request: ConversationRequest):
    """
    Ingest an entire conversation at once

    Useful for batch importing conversations.
    """
    try:
        if not conversation_monitor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation monitor not initialized"
            )

        # Capture the conversation
        result = await conversation_monitor.capture_conversation(
            messages=request.messages,
            conversation_id=request.conversation_id,
            model=request.model
        )

        return ConversationResponse(
            conversation_id=result["conversation_id"],
            message_ids=result["message_ids"],
            message_count=result["message_count"],
            status="queued_for_processing"
        )

    except Exception as e:
        logger.error(f"Error ingesting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint

    Returns the status of all system components.
    """
    services_status = {}

    # Check Weaviate
    try:
        if weaviate_client:
            weaviate_client.client.is_ready()
            services_status["weaviate"] = "healthy"
        else:
            services_status["weaviate"] = "not_initialized"
    except Exception as e:
        services_status["weaviate"] = f"unhealthy: {str(e)}"

    # Check MongoDB
    try:
        if mongo_client:
            mongo_client.client.admin.command('ping')
            services_status["mongodb"] = "healthy"
        else:
            services_status["mongodb"] = "not_initialized"
    except Exception as e:
        services_status["mongodb"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        if redis_client:
            health = await redis_client.health_check()
            services_status["redis"] = "healthy" if health else "unhealthy"
        else:
            services_status["redis"] = "not_initialized"
    except Exception as e:
        services_status["redis"] = f"unhealthy: {str(e)}"

    # Check PostgreSQL
    try:
        if postgres_client:
            services_status["postgres"] = "healthy"
        else:
            services_status["postgres"] = "not_initialized"
    except Exception as e:
        services_status["postgres"] = f"unhealthy: {str(e)}"

    # Overall status
    overall_status = "healthy" if all(
        s in ["healthy", "not_initialized"] for s in services_status.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Get system statistics

    Returns statistics about the intelligence grid and processing pipeline.
    """
    try:
        stats = {}

        # Weaviate stats
        if weaviate_client:
            stats["weaviate"] = weaviate_client.get_stats()
        else:
            stats["weaviate"] = {"error": "not_initialized"}

        # MongoDB stats
        if mongo_client:
            stats["mongodb"] = mongo_client.get_stats()
        else:
            stats["mongodb"] = {"error": "not_initialized"}

        # Redis stats
        if redis_client:
            stats["redis"] = await redis_client.get_stats()
        else:
            stats["redis"] = {"error": "not_initialized"}

        # PostgreSQL stats
        if postgres_client:
            stats["postgres"] = postgres_client.get_global_stats()
        else:
            stats["postgres"] = {"error": "not_initialized"}

        # Capture stats
        if conversation_monitor:
            stats["capture"] = conversation_monitor.get_stats()
        else:
            stats["capture"] = {"error": "not_initialized"}

        # Processing stats
        if embedding_worker:
            stats["processing"] = embedding_worker.get_stats()
        else:
            stats["processing"] = {"error": "not_initialized"}

        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _build_context(results: List[Dict]) -> str:
    """Build aggregated context from search results"""
    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results[:5], 1):  # Top 5 results
        content = result.get("content", "")
        model = result.get("sourceModel", "unknown")
        certainty = result.get("certainty", 0)

        context_parts.append(
            f"{i}. [{model} - {certainty:.1%} relevant]\n{content[:300]}..."
        )

    return "\n\n".join(context_parts)


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AetherGrid API server starting...")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 AetherGrid API server shutting down...")
