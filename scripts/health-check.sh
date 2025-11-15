#!/bin/bash
# AetherGrid Health Check Script

echo "🏥 AetherGrid Health Check"
echo "=========================="
echo ""

# Check Docker containers
echo "📦 Docker Containers:"
docker-compose ps

echo ""
echo "🔍 Service Health:"

# Check API
API_HEALTH=$(curl -s http://localhost:8100/api/health 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "  ✓ API Server: $(echo $API_HEALTH | jq -r '.status')"
else
    echo "  ❌ API Server: Not responding"
fi

# Check PostgreSQL
if docker exec aethergrid-postgres pg_isready -U aether > /dev/null 2>&1; then
    echo "  ✓ PostgreSQL: Healthy"
else
    echo "  ❌ PostgreSQL: Not ready"
fi

# Check MongoDB
if docker exec aethergrid-mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    echo "  ✓ MongoDB: Healthy"
else
    echo "  ❌ MongoDB: Not ready"
fi

# Check Redis
if docker exec aethergrid-redis redis-cli ping > /dev/null 2>&1; then
    echo "  ✓ Redis: Healthy"
else
    echo "  ❌ Redis: Not ready"
fi

# Check Weaviate
if curl -s http://localhost:8101/v1/.well-known/ready > /dev/null 2>&1; then
    echo "  ✓ Weaviate: Healthy"
else
    echo "  ❌ Weaviate: Not ready"
fi

echo ""
echo "📊 Stats:"
curl -s http://localhost:8100/api/stats 2>/dev/null | jq '.' || echo "  API not available"
