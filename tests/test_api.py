"""
Tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.server import app


client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AetherGrid"
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "timestamp" in data


# Note: These tests require database connections to be set up
# They are skipped by default in CI/CD environments

@pytest.mark.integration
def test_query_endpoint():
    """Test semantic query endpoint (integration test)"""
    response = client.post(
        "/api/query/semantic",
        json={
            "query": "test query",
            "max_results": 5,
            "min_similarity": 0.7
        }
    )

    assert response.status_code in [200, 503]  # 503 if services not running


@pytest.mark.integration
def test_ingest_endpoint():
    """Test ingest endpoint (integration test)"""
    response = client.post(
        "/api/ingest",
        json={
            "content": "This is test content",
            "model": "test-model",
            "role": "assistant"
        }
    )

    assert response.status_code in [200, 503]  # 503 if services not running
