#!/bin/bash
# AetherGrid Stop Script

echo "🛑 Stopping AetherGrid..."

# Stop Docker containers
echo "📦 Stopping database infrastructure..."
docker-compose down

echo "✓ AetherGrid stopped"
