#!/bin/bash
# AetherGrid Startup Script

set -e

echo "🚀 Starting AetherGrid..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your API keys before continuing."
    echo "   Required: OPENAI_API_KEY"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start infrastructure
echo "📦 Starting database infrastructure..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."

# Check PostgreSQL
until docker exec aethergrid-postgres pg_isready -U aether > /dev/null 2>&1; do
    echo "  ⏳ Waiting for PostgreSQL..."
    sleep 2
done
echo "  ✓ PostgreSQL ready"

# Check MongoDB
until docker exec aethergrid-mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
    echo "  ⏳ Waiting for MongoDB..."
    sleep 2
done
echo "  ✓ MongoDB ready"

# Check Redis
until docker exec aethergrid-redis redis-cli ping > /dev/null 2>&1; do
    echo "  ⏳ Waiting for Redis..."
    sleep 2
done
echo "  ✓ Redis ready"

# Check Weaviate
until curl -s http://localhost:8101/v1/.well-known/ready > /dev/null 2>&1; do
    echo "  ⏳ Waiting for Weaviate..."
    sleep 2
done
echo "  ✓ Weaviate ready"

echo ""
echo "✅ All infrastructure services are ready!"
echo ""
echo "📋 Service URLs:"
echo "   PostgreSQL: localhost:5433"
echo "   MongoDB:    localhost:27018"
echo "   Weaviate:   http://localhost:8101"
echo "   Redis:      localhost:6395"
echo ""
echo "🚀 Starting AetherGrid application..."
echo ""

# Start AetherGrid
poetry run python main.py
